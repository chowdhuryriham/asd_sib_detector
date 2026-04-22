"""
pipeline_f_plm.py  —  Perception-LM-8B (PLM)
ASD Self-Injurious Behaviour Detector

Lab    : Meta FAIR (USA)
Model  : facebook/Perception-LM-8B
Paper  : arXiv 2504.13180 (April 2025)
License: FAIR Non-Commercial Research License

venv : /raid/aio469/venvs/plm
run  : CUDA_VISIBLE_DEVICES=2,3,4,5 /raid/aio469/venvs/plm/bin/python \
           pipelines/pipeline_f_plm.py

OFFICIAL INFERENCE (https://huggingface.co/facebook/Perception-LM-8B):
  processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
  model     = AutoModelForImageTextToText.from_pretrained(MODEL_PATH).to("cuda")
  conversation = [{"role":"user","content":[
      {"type":"video","url":video_file},
      {"type":"text","text":"..."}]}]
  inputs = processor.apply_chat_template([conversation], num_frames=32,
      add_generation_prompt=True, tokenize=True, return_dict=True,
      return_tensors="pt", video_load_backend="decord")
  generate_ids = model.generate(**inputs, max_new_tokens=256)
  input_length = inputs["input_ids"].shape[1]
  output = processor.batch_decode(generate_ids[:, input_length:],
                                  skip_special_tokens=True)

  NOTE: in newer transformers, video_load_backend must go inside
  processor_kwargs={} rather than as a top-level kwarg.

  DEPS: pip install decord timm
"""

import os
import re
import cv2
import json
import shutil
import warnings
import subprocess
import numpy as np
import torch
import static_ffmpeg
static_ffmpeg.add_paths()
warnings.filterwarnings("ignore")

from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
BASE           = "/raid/aio469/asd_sib_detector"
MODEL_PATH     = f"{BASE}/models/Perception-LM-8B"
VIDEO_PATH     = f"{BASE}/data/raw/lab_videos/input_01.mp4"
OUTPUT_PATH    = f"{BASE}/outputs/plm/sample_annotated.mp4"
LOG_PATH       = f"{BASE}/logs/pipeline_f_plm_log.json"
INTERVALS_PATH = f"{BASE}/outputs/plm/behavior_intervals.json"
SUMMARY_PATH   = f"{BASE}/logs/pipeline_f_plm_summary.txt"
CHUNK_DIR      = f"{BASE}/outputs/plm/chunks"
DEBUG_DIR      = f"{BASE}/outputs/plm/chunk_debug_frames"
DEBUG_SAVE     = True
CHUNK_SEC      = 5
OVERLAP_SEC    = 1          # stride = 4s
NUM_FRAMES     = 32         # PLM native window
VALID_LABELS   = {"hand_biting", "head_banging", "hitting_others", "scratching", "self_directed_hit", "none"}

# ─────────────────────────────────────────────────────────
# 1.  Load model — official PLM loading pattern
#     AutoModelForImageTextToText (NOT AutoModelForCausalLM)
#     No trust_remote_code needed — native transformers class
# ─────────────────────────────────────────────────────────
n_gpus = torch.cuda.device_count()
print(f"Visible GPUs: {n_gpus}")
if n_gpus == 0:
    raise RuntimeError("No GPUs visible. Set CUDA_VISIBLE_DEVICES=2,3,4,5.")

print(f"Loading Perception-LM-8B from {MODEL_PATH} ...")
processor = AutoProcessor.from_pretrained(
    Path(MODEL_PATH),
    use_fast=True,          # as per official model card
)
model = AutoModelForImageTextToText.from_pretrained(
    Path(MODEL_PATH),
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()
print(f"Model loaded.\n")

# ─────────────────────────────────────────────────────────
# 2.  Video probe helpers
# ─────────────────────────────────────────────────────────
def get_video_info(path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr}")
    info = json.loads(r.stdout)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    n, d = vs["r_frame_rate"].split("/")
    fps = float(n) / float(d)
    w, h = int(vs["width"]), int(vs["height"])
    dur = float(info["format"]["duration"])
    return fps, w, h, dur, int(round(dur * fps))

def get_chunk_actual_duration(path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        try:
            return float(json.loads(r.stdout)["format"]["duration"])
        except (KeyError, ValueError):
            pass
    return 0.0

# ─────────────────────────────────────────────────────────
# 3.  Helpers
# ─────────────────────────────────────────────────────────
ALIASES = {
    "hand_biting_(own)": "hand_biting",  "hand_biting(own)": "hand_biting",
    "self_biting": "hand_biting",        "biting_hand": "hand_biting",
    "hand_bite": "hand_biting",          "finger_biting": "hand_biting",
    "wrist_biting": "hand_biting",
    "headbanging": "head_banging",       "head_bang": "head_banging",
    "head_hitting": "head_banging",      "hitting_head": "head_banging",
    "banging_head": "head_banging",
    "hitting_other": "hitting_others",   "hitting_another": "hitting_others",
    "slapping_others": "hitting_others", "punching_others": "hitting_others",
    "scratching_head": "scratching",     "scratching_own_head": "scratching",
    "head_scratching": "scratching",     "scratching_scalp": "scratching",
    "scratching_self": "scratching",
    "no_behavior": "none",               "neutral": "none",
    "self_hit": "self_directed_hit",       "hitting_self": "self_directed_hit",
    "self_directed_hitting": "self_directed_hit",
}

def normalize_label(raw):
    if not raw: return "none"
    lbl = (raw.strip().lower()
              .strip(" \t\r\n.,;:!?\"'`()[]{}").replace("-","_").replace(" ","_"))
    if lbl in VALID_LABELS: return lbl
    return ALIASES.get(lbl, "none")

def parse_chunk_response(text, chunk_duration):
    label = evidence = justification = ""
    start_sec = end_sec = 0.0
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"(?i)^label\s*:", line):
            label = re.sub(r"(?i)^label\s*:\s*", "", line).strip()
        elif re.match(r"(?i)^evidence\s*:", line):
            evidence = re.sub(r"(?i)^evidence\s*:\s*", "", line).strip()
        elif re.match(r"(?i)^justification\s*:", line):
            justification = re.sub(r"(?i)^justification\s*:\s*", "", line).strip()
        elif re.match(r"(?i)^start\s*:", line):
            m = re.search(r"[\d.]+", line)
            if m: start_sec = float(m.group())
        elif re.match(r"(?i)^end\s*:", line):
            m = re.search(r"[\d.]+", line)
            if m: end_sec = float(m.group())
    if not label:
        m = re.search(r"(?im)^\s*Label\s*:\s*(.+?)\s*$", text)
        if m: label = m.group(1).strip()
    if not evidence:
        m = re.search(
            r"(?i)evidence\s*:\s*(.+?)(?:\n|Label|Start|End|Justification|$)", text, re.S)
        if m: evidence = m.group(1).strip()
    if not justification:
        m = re.search(r"(?i)justification\s*:\s*(.+?)(?:\n|$)", text, re.S)
        if m: justification = m.group(1).strip()
    if start_sec == 0.0:
        m = re.search(r"(?i)start\s*:\s*.*?([\d.]+)", text)
        if m: start_sec = float(m.group(1))
    if end_sec == 0.0:
        m = re.search(r"(?i)end\s*:\s*.*?([\d.]+)", text)
        if m: end_sec = float(m.group(1))
    raw_label = label
    label = normalize_label(label)
    if raw_label and label != "none" and raw_label.lower().strip() != label:
        print(f"    [INFO] '{raw_label}' -> '{label}'")
    start_sec = max(0.0, min(start_sec, chunk_duration))
    end_sec   = max(0.0, min(end_sec,   chunk_duration))
    if label == "none":
        start_sec = end_sec = 0.0
    if label != "none" and end_sec <= start_sec:
        print("    [WARN] invalid time range -> dropping to none")
        label = "none"; start_sec = end_sec = 0.0
    return {"label": label, "evidence": evidence, "justification": justification,
            "start_sec": start_sec, "end_sec": end_sec,
            "raw_label": raw_label, "raw_text": text[:2000]}

def sec_to_ts(sec):
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def build_intervals(frame_labels, fps):
    if not frame_labels: return []
    intervals = []; sf = sorted(frame_labels.keys())
    cl = ce = cj = None; i0 = prev = None
    for f in sf:
        l, e, j = frame_labels[f]
        if l == cl and prev is not None and f == prev + 1:
            prev = f
        else:
            if cl is not None:
                intervals.append({"label": cl,
                    "start_frame": i0, "end_frame": prev,
                    "start_sec": round(i0/fps,2), "end_sec": round((prev+1)/fps,2),
                    "start_timestamp": sec_to_ts(i0/fps),
                    "end_timestamp": sec_to_ts((prev+1)/fps),
                    "duration_sec": round((prev-i0+1)/fps,2),
                    "evidence": ce, "justification": cj})
            cl, ce, cj, i0, prev = l, e, j, f, f
    if cl is not None:
        intervals.append({"label": cl,
            "start_frame": i0, "end_frame": prev,
            "start_sec": round(i0/fps,2), "end_sec": round((prev+1)/fps,2),
            "start_timestamp": sec_to_ts(i0/fps),
            "end_timestamp": sec_to_ts((prev+1)/fps),
            "duration_sec": round((prev-i0+1)/fps,2),
            "evidence": ce, "justification": cj})
    return intervals

def draw_bold_text(frame, text, x, y, scale, color, thickness=3):
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_DUPLEX,
                scale, (0,0,0), thickness+4, cv2.LINE_AA)
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────
# 4.  Video probe + chunk boundaries
# ─────────────────────────────────────────────────────────
fps, width, height, duration_sec, total_frames = get_video_info(VIDEO_PATH)
stride_sec   = CHUNK_SEC - OVERLAP_SEC
chunk_starts = []
t = 0.0
while t < duration_sec:
    chunk_starts.append(t); t += stride_sec

print(f"Video : {VIDEO_PATH}")
print(f"        {duration_sec:.1f}s ({duration_sec/60:.1f} min) | "
      f"{total_frames} frames @ {fps:.2f} fps | {width}x{height}")
print(f"Chunks: {len(chunk_starts)} x {CHUNK_SEC}s  "
      f"overlap={OVERLAP_SEC}s  stride={stride_sec}s\n")

# ─────────────────────────────────────────────────────────
# 5.  Cut chunks
# ─────────────────────────────────────────────────────────
os.makedirs(CHUNK_DIR, exist_ok=True)
for p in [OUTPUT_PATH, LOG_PATH, INTERVALS_PATH, SUMMARY_PATH]:
    os.makedirs(os.path.dirname(p), exist_ok=True)
if DEBUG_SAVE:
    os.makedirs(DEBUG_DIR, exist_ok=True)

chunk_info = []
for i, start in enumerate(chunk_starts):
    end        = min(start + CHUNK_SEC, duration_sec)
    chunk_dur  = end - start
    sf         = int(round(start * fps))
    ef         = int(round(end   * fps))
    chunk_path = os.path.join(CHUNK_DIR, f"chunk_{i:04d}.mp4")
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", VIDEO_PATH,
           "-t", f"{chunk_dur:.3f}", "-c", "copy",
           "-avoid_negative_ts", "make_zero", "-loglevel", "error", chunk_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  Chunk {i}: ffmpeg error: {r.stderr}"); continue
    if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) < 500:
        print(f"  Chunk {i}: too small / missing"); continue
    actual_dur          = get_chunk_actual_duration(chunk_path) or chunk_dur
    keyframe_offset     = max(0.0, actual_dur - chunk_dur)
    actual_global_start = start - keyframe_offset
    chunk_info.append({
        "chunk_idx": i, "chunk_path": chunk_path, "sf": sf, "ef": ef,
        "requested_start": start, "requested_end": end,
        "actual_duration": actual_dur,
        "actual_global_start": actual_global_start,
        "keyframe_offset": keyframe_offset,
    })
    if i % 100 == 0:
        print(f"  Chunk {i}/{len(chunk_starts)-1}: {start:.1f}-{end:.1f}s  "
              f"({os.path.getsize(chunk_path)//1024} KB)")

if not chunk_info:
    raise RuntimeError("No valid chunks were created.")
print(f"\n{len(chunk_info)} chunks ready.\n")

# ─────────────────────────────────────────────────────────
# 6.  Behaviour prompt
# ─────────────────────────────────────────────────────────
def make_prompt(chunk_dur):
    return (
        "Describe this video carefully in chronological order. "
        "There may be multiple people in the scene. "
        "Focus on the child patient's body movements and interactions.\n\n"
        "Then classify the child's primary behavior using exactly one of these labels:\n"
        "[hand_biting, head_banging, hitting_others, scratching, self_directed_hit, none]\n\n"
        "Brief definitions:\n"
        "- hand_biting: child bites own hand/fingers/wrist\n"
        "- head_banging: child hits own head against surface or object\n"
        "- hitting_others: child hits/slaps/pushes another person\n"
        "- scratching: child scratches own head/scalp/skin\n"
        "- self_directed_hit: child hits/slaps their own body (arm, leg, torso - not head)\n"
        "- none: none of the above\n\n"
        f"This clip is {chunk_dur:.1f} seconds long.\n"
        "If behavior is visible, estimate when it starts and ends (in seconds within this clip).\n\n"
        "Reply in this exact format:\n"
        "Evidence: <one sentence describing what you see>\n"
        "Label: <one label>\n"
        "Start: <seconds or 0>\n"
        "End: <seconds or 0>\n"
        "Justification: <one sentence explaining the label choice>\n\n"
        "If Label is none, set Start: 0 and End: 0."
    )

# ─────────────────────────────────────────────────────────
# 7.  Classify a chunk — official PLM inference pattern
#
#   From https://huggingface.co/facebook/Perception-LM-8B:
#     conversation = [{"role":"user","content":[
#         {"type":"video","url":video_file},
#         {"type":"text","text":"..."}]}]
#     inputs = processor.apply_chat_template([conversation], num_frames=32,
#         add_generation_prompt=True, tokenize=True, return_dict=True,
#         return_tensors="pt", video_load_backend="decord")
#     generate_ids = model.generate(**inputs, max_new_tokens=256)
#     input_length = inputs["input_ids"].shape[1]
#     output = processor.batch_decode(generate_ids[:, input_length:],
#                                     skip_special_tokens=True)
#
#   NOTE: newer transformers requires video_load_backend inside
#   processor_kwargs={} — passing it top-level is silently ignored.
# ─────────────────────────────────────────────────────────
def classify_chunk(chunk):
    chunk_path = chunk["chunk_path"]
    chunk_dur  = chunk["actual_duration"]
    chunk_idx  = chunk["chunk_idx"]

    # save debug frames via opencv
    if DEBUG_SAVE:
        cap = cv2.VideoCapture(chunk_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fi in [0, max(0, total // 2), max(0, total - 1)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(
                    os.path.join(DEBUG_DIR, f"chunk_{chunk_idx:04d}_f{fi:03d}.png"), frame)
        cap.release()

    # official PLM conversation format
    conversation = [{
        "role": "user",
        "content": [
            {"type": "video", "url": chunk_path},
            {"type": "text",  "text": make_prompt(chunk_dur)},
        ],
    }]

    # official PLM apply_chat_template
    # video_load_backend passed via processor_kwargs (required in newer transformers)
    inputs = processor.apply_chat_template(
        [conversation],
        processor_kwargs={"num_frames": NUM_FRAMES, "video_load_backend": "decord"},
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]
    print(f"    input_ids={list(inputs['input_ids'].shape)}")

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # official decode — slice off input tokens (from model card)
    response = processor.batch_decode(
        generate_ids[:, input_length:],
        skip_special_tokens=True,
    )[0].strip()

    print(f"    [DEBUG] response:\n{response}\n")

    result = parse_chunk_response(response, chunk_dur)
    print(f"\n=== CHUNK {chunk_idx} ===")
    print(f"Label: {result['label']}  (raw: '{result['raw_label']}')")
    print(f"Evidence: {result['evidence']}")
    print(f"Justification: {result['justification']}")
    print(f"Start: {result['start_sec']:.2f}  End: {result['end_sec']:.2f}")
    return result

# ─────────────────────────────────────────────────────────
# 8.  Main loop
# ─────────────────────────────────────────────────────────
frame_labels    = {}
all_logs        = []
behavior_chunks = []

for chunk in chunk_info:
    i                   = chunk["chunk_idx"]
    actual_duration     = chunk["actual_duration"]
    actual_global_start = chunk["actual_global_start"]
    keyframe_offset     = chunk["keyframe_offset"]
    requested_start     = chunk["requested_start"]
    requested_end       = chunk["requested_end"]

    print(f"\n{'='*60}")
    print(f"Chunk {i}/{len(chunk_info)-1}  "
          f"({requested_start:.1f}s-{requested_end:.1f}s  "
          f"actual_start={actual_global_start:.1f}s)")
    print(f"{'='*60}")

    result = classify_chunk(chunk)

    if result["label"] != "none":
        b_start = max(0.0, actual_global_start + result["start_sec"])
        b_end   = min(duration_sec, actual_global_start + result["end_sec"])
        b_sf    = max(0, int(round(b_start * fps)))
        b_ef    = min(total_frames, int(round(b_end * fps)))
        print(f"  Global: {b_start:.2f}s - {b_end:.2f}s  (frames {b_sf}-{b_ef})")
        if keyframe_offset > 0.1:
            print(f"  (Corrected by {keyframe_offset:.2f}s keyframe offset)")
        for f in range(b_sf, b_ef):
            frame_labels[f] = (result["label"], result["evidence"], result["justification"])
        behavior_chunks.append({
            "chunk_idx": i,
            "requested_start": requested_start, "requested_end": requested_end,
            "label": result["label"],
            "global_start": round(b_start, 3), "global_end": round(b_end, 3),
            "global_start_timestamp": sec_to_ts(b_start),
            "global_end_timestamp":   sec_to_ts(b_end),
            "evidence":      result["evidence"],
            "justification": result["justification"],
        })
    # positive always beats none — none never erases a marked frame

    all_logs.append({
        "chunk_idx": i,
        "requested_start":     requested_start,
        "requested_end":       requested_end,
        "actual_global_start": actual_global_start,
        "keyframe_offset":     keyframe_offset,
        "actual_duration":     actual_duration,
        **result,
    })

with open(LOG_PATH, "w") as fp:
    json.dump(all_logs, fp, indent=2)
print(f"\nChunk log -> {LOG_PATH}")

# ─────────────────────────────────────────────────────────
# 9.  Merge intervals + outputs
# ─────────────────────────────────────────────────────────
intervals    = build_intervals(frame_labels, fps)
label_counts = {}
total_bsec   = 0.0
for iv in intervals:
    label_counts[iv["label"]] = label_counts.get(iv["label"], 0) + 1
    total_bsec += iv["duration_sec"]

intervals_out = {
    "video": VIDEO_PATH, "model": "Perception-LM-8B",
    "lab":   "Meta FAIR (USA)", "license": "FAIR Non-Commercial Research",
    "paper": "arXiv 2504.13180",
    "duration_sec": round(duration_sec, 2),
    "duration_timestamp": sec_to_ts(duration_sec),
    "fps": fps, "total_frames": total_frames,
    "chunk_settings": {
        "chunk_sec": CHUNK_SEC, "overlap_sec": OVERLAP_SEC,
        "stride_sec": stride_sec,
        "total_chunks": len(chunk_info), "num_frames_per_chunk": NUM_FRAMES,
    },
    "summary": {
        "total_behavior_intervals": len(intervals),
        "total_behavior_seconds":   round(total_bsec, 2),
        "behavior_percentage":      round(100 * total_bsec / duration_sec, 2)
                                    if duration_sec > 0 else 0,
        "label_counts": label_counts,
    },
    "behavior_chunks": behavior_chunks,
    "intervals":       intervals,
}
with open(INTERVALS_PATH, "w") as fp:
    json.dump(intervals_out, fp, indent=2)
print(f"Intervals -> {INTERVALS_PATH}")

# ─────────────────────────────────────────────────────────
# 10.  Summary
# ─────────────────────────────────────────────────────────
L = ["="*60, "BEHAVIOUR DETECTION SUMMARY -- Perception-LM-8B", "="*60,
     f"Video   : {VIDEO_PATH}",
     f"Duration: {sec_to_ts(duration_sec)}  ({duration_sec:.1f}s)",
     f"FPS     : {fps:.2f}",
     f"Chunks  : {len(chunk_info)}  "
     f"({CHUNK_SEC}s each, {OVERLAP_SEC}s overlap, {stride_sec}s stride, "
     f"{NUM_FRAMES} frames/chunk)", ""]
if not intervals:
    L.append("  No self-injurious behaviours detected.")
else:
    L += [f"  {len(intervals)} interval(s) detected  ({total_bsec:.1f}s total, "
          f"{intervals_out['summary']['behavior_percentage']:.1f}% of video)",
          "", "  Label breakdown:"]
    for lbl, cnt in sorted(label_counts.items()):
        L.append(f"    {lbl:<20} {cnt} interval(s)")
    L += ["", "  Intervals (global time):"]
    for iv in intervals:
        L.append(f"    [{iv['start_timestamp']} -> {iv['end_timestamp']}]  "
                 f"{iv['label'].upper().replace('_',' ')}  ({iv['duration_sec']:.1f}s)")
        L.append(f"      Evidence:      {iv['evidence'][:80]}")
        L.append(f"      Justification: {iv['justification'][:80]}")
L += ["", "-"*60]
if behavior_chunks:
    L.append(f"CHUNK-LEVEL DETECTIONS ({len(behavior_chunks)}):")
    for bc in behavior_chunks:
        L.append(f"  Chunk {bc['chunk_idx']:04d}  "
                 f"[{bc['global_start_timestamp']} -> {bc['global_end_timestamp']}]  "
                 f"{bc['label'].upper().replace('_',' ')}")
        L.append(f"    Evidence:      {bc['evidence'][:80]}")
        L.append(f"    Justification: {bc['justification'][:80]}")
else:
    L.append("CHUNK-LEVEL DETECTIONS: None.")
summary_text = "\n".join(L)
print("\n" + summary_text)
with open(SUMMARY_PATH, "w") as fp:
    fp.write(summary_text + "\n")
print(f"\nSummary -> {SUMMARY_PATH}")

# ─────────────────────────────────────────────────────────
# 11.  Annotate video
# ─────────────────────────────────────────────────────────
print(f"\nAnnotating video -> {OUTPUT_PATH}")
LABEL_COLORS = {
    "hand_biting":    (0,   0, 255),
    "head_banging":   (0, 128, 255),
    "hitting_others": (0,   0, 200),
    "scratching":          (0, 165, 255),
    "self_directed_hit": (0, 255, 128),
}
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")
ffmpeg_cmd = [
    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
    "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
    "-r", str(fps), "-i", "-",
    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    "-loglevel", "error", OUTPUT_PATH,
]
proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
frame_idx = behavior_frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    info = frame_labels.get(frame_idx)
    if info and info[0] != "none":
        lbl, ev, jt = info
        color    = LABEL_COLORS.get(lbl, (255, 255, 255))
        lbl_disp = lbl.upper().replace("_", " ")
        ts       = sec_to_ts(frame_idx / fps)
        draw_bold_text(frame, ts, 10, 30, 0.7, (255, 255, 255), 2)
        ls, lt = 1.8, 4
        (tw, th), _ = cv2.getTextSize(lbl_disp, cv2.FONT_HERSHEY_DUPLEX, ls, lt)
        lx = (width - tw) // 2; ly = int(height * 0.75); px = py = 16
        cv2.rectangle(frame, (lx-px, ly-th-py), (lx+tw+px, ly+py), (0, 0, 0), -1)
        cv2.rectangle(frame, (lx-px, ly-th-py), (lx+tw+px, ly+py), color, 3)
        draw_bold_text(frame, lbl_disp, lx, ly, ls, color, lt)
        bt = height - 70
        ov = frame.copy()
        cv2.rectangle(ov, (0, bt), (width, height), (0, 0, 0), -1)
        frame = cv2.addWeighted(ov, 0.7, frame, 0.3, 0)
        if ev:
            draw_bold_text(frame, f"Evidence: {ev[:100]}", 10, bt+25,
                           0.55, (255, 255, 255), 1)
        if jt:
            draw_bold_text(frame, f"Justification: {jt[:100]}", 10, bt+55,
                           0.55, (200, 200, 200), 1)
        behavior_frame_count += 1
    try:
        proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        raise RuntimeError("ffmpeg pipe broke")
    frame_idx += 1
cap.release()
proc.stdin.close()
rc = proc.wait()
if rc != 0:
    raise RuntimeError(f"ffmpeg exited {rc}")
shutil.rmtree(CHUNK_DIR, ignore_errors=True)
print(f"\n{'='*60}\nDONE")
print(f"  Total frames     : {frame_idx}  ({frame_idx/fps:.1f}s)")
print(f"  Behaviour frames : {behavior_frame_count}")
print(f"  Annotated video  : {OUTPUT_PATH}")
print(f"  Chunk log        : {LOG_PATH}")
print(f"  Intervals JSON   : {INTERVALS_PATH}")
print(f"  Summary txt      : {SUMMARY_PATH}")
print(f"{'='*60}")