"""
pipeline_b_llava_video.py  —  LLaVA-Video-72B-Qwen2
ASD Self-Injurious Behaviour Detector

venv : /raid/aio469/venvs/llava_video
run  : CUDA_VISIBLE_DEVICES=0,2,3,4 /raid/aio469/venvs/llava_video/bin/python \
           pipelines/pipeline_b_llava_video.py

KEY DECODE FIX (root cause of empty output):
  llava calls super().generate(inputs_embeds=...) internally.
  When HuggingFace generates from inputs_embeds, it returns ONLY the
  newly generated tokens — not input+output.
  So output_ids[:, input_ids.shape[1]:] slices at ~800 into a tensor
  of ~200 new tokens → always empty.
  Fix: decode output_ids directly, exactly like the official model card:
    tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
"""

import os
import re
import cv2
import json
import shutil
import copy
import warnings
import numpy as np
import torch
import subprocess
import static_ffmpeg
static_ffmpeg.add_paths()
warnings.filterwarnings("ignore")

from decord import VideoReader, cpu

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.llava_arch import LlavaMetaForCausalLM

# ─────────────────────────────────────────────────────────
# MONKEY-PATCH — multi-GPU device alignment
# ─────────────────────────────────────────────────────────
_orig_encode_images = LlavaMetaForCausalLM.encode_images

def _encode_images_aligned(self, images):
    features = _orig_encode_images(self, images)
    embed_dev = self.model.embed_tokens.weight.device
    if isinstance(features, torch.Tensor):
        features = features.to(embed_dev)
    elif isinstance(features, (list, tuple)):
        features = type(features)(f.to(embed_dev) for f in features)
    return features

LlavaMetaForCausalLM.encode_images = _encode_images_aligned
print("[PATCH] encode_images → features moved to embed_tokens device")

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
BASE           = "/raid/aio469/asd_sib_detector"
MODEL_PATH     = f"{BASE}/models/LLaVA-Video-72B"
VIDEO_PATH     = f"{BASE}/data/raw/lab_videos/input_01.mp4"
OUTPUT_PATH    = f"{BASE}/outputs/llava_video/sample_annotated.mp4"
LOG_PATH       = f"{BASE}/logs/pipeline_b_llava_video_log.json"
INTERVALS_PATH = f"{BASE}/outputs/llava_video/behavior_intervals.json"
SUMMARY_PATH   = f"{BASE}/logs/pipeline_b_llava_video_summary.txt"
CHUNK_DIR      = f"{BASE}/outputs/llava_video/chunks"
DEBUG_DIR      = f"{BASE}/outputs/llava_video/chunk_debug_frames"
DEBUG_SAVE     = True
CHUNK_SEC      = 10
OVERLAP_SEC    = 3
NUM_FRAMES     = 32
CONV_TEMPLATE  = "qwen_1_5"
VALID_LABELS   = {"hand_biting", "head_banging", "hitting_others", "scratching", "none"}

# ─────────────────────────────────────────────────────────
# 1.  Load model — model_name MUST be "llava_qwen"
# ─────────────────────────────────────────────────────────
n_gpus = torch.cuda.device_count()
print(f"Visible GPUs: {n_gpus}")
if n_gpus == 0:
    raise RuntimeError("No GPUs visible. Set CUDA_VISIBLE_DEVICES=0,2,3,4 before running.")

max_mem = {i: "130GiB" for i in range(n_gpus)}
print(f"max_memory: {max_mem}")
print("Loading LLaVA-Video-72B-Qwen2 …")

tokenizer, model, image_processor, max_length = load_pretrained_model(
    model_path=MODEL_PATH,
    model_base=None,
    model_name="llava_qwen",
    torch_dtype="bfloat16",
    device_map="auto",
    attn_implementation="flash_attention_2",
    max_memory=max_mem,
)
model.eval()
embed_device = model.model.embed_tokens.weight.device
print(f"Model loaded.  embed_device={embed_device}\n")

# ─────────────────────────────────────────────────────────
# 2.  Video probe + chunk boundaries
# ─────────────────────────────────────────────────────────
def get_video_info(path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr}")
    info = json.loads(r.stdout)
    vs   = next(s for s in info["streams"] if s["codec_type"] == "video")
    n, d = vs["r_frame_rate"].split("/")
    fps  = float(n) / float(d)
    w, h = int(vs["width"]), int(vs["height"])
    dur  = float(info["format"]["duration"])
    return fps, w, h, dur, int(round(dur * fps))

def get_chunk_actual_duration(path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        try: return float(json.loads(r.stdout)["format"]["duration"])
        except (KeyError, ValueError): pass
    return 0.0

fps, width, height, duration_sec, total_frames = get_video_info(VIDEO_PATH)
stride_sec   = CHUNK_SEC - OVERLAP_SEC
chunk_starts = []
t = 0.0
while t < duration_sec:
    chunk_starts.append(t); t += stride_sec

print(f"Video : {VIDEO_PATH}")
print(f"        {duration_sec:.1f}s ({duration_sec/60:.1f} min) | "
      f"{total_frames} frames @ {fps:.2f} fps | {width}x{height}")
print(f"Chunks: {len(chunk_starts)} x {CHUNK_SEC}s  stride={stride_sec}s\n")

# ─────────────────────────────────────────────────────────
# 3.  Cut chunks with ffmpeg -c copy
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
        print(f"  Chunk {i}: ffmpeg failed: {r.stderr}"); continue
    if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) < 1000:
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
    if keyframe_offset > 0.1:
        print(f"  Chunk {i}: {start:.1f}-{end:.1f}s  "
              f"(actual_start={actual_global_start:.1f}s, offset={keyframe_offset:.1f}s)")
    else:
        print(f"  Chunk {i}: {start:.1f}-{end:.1f}s  "
              f"({os.path.getsize(chunk_path)//1024} KB)")

if not chunk_info:
    raise RuntimeError("No valid chunks were created.")
print(f"\n{len(chunk_info)} chunks ready.\n")

# ─────────────────────────────────────────────────────────
# 4.  Frame loader — decord (official method)
# ─────────────────────────────────────────────────────────
def load_video_decord(video_path, max_frames_num):
    """Returns (frames [N,H,W,3] uint8, frame_time_str, video_time_sec)"""
    vr         = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total      = len(vr)
    video_time = total / vr.get_avg_fps()
    idx        = np.linspace(0, total - 1, max_frames_num, dtype=int)
    frame_time = [i / vr.get_avg_fps() for i in idx]
    frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])
    frames     = vr.get_batch(idx).asnumpy()
    return frames, frame_time_str, video_time

# ─────────────────────────────────────────────────────────
# 5.  Helpers
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
            r"(?i)evidence\s*:\s*(.+?)(?:\n|Label|Start|End|Justification|$)",
            text, re.S)
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
# 6.  Classify a chunk — official LLaVA-Video-72B procedure
# ─────────────────────────────────────────────────────────
BEHAVIOR_PROMPT = (
    "Describe this video carefully in chronological order. "
    "There may be multiple people in the scene. "
    "Focus on the child patient's body movements and interactions.\n\n"
    "Then classify the child's primary behavior using exactly one of these labels:\n"
    "[hand_biting, head_banging, hitting_others, scratching, none]\n\n"
    "Brief definitions:\n"
    "- hand_biting: child bites own hand/fingers/wrist\n"
    "- head_banging: child hits own head against surface/ table or any object\n"
    "- hitting_others: child hits/slaps/pushes another person\n"
    "- scratching: child scratches own head/scalp/skin\n"
    "- none: none of the above\n\n"
    "If behavior is visible, estimate when it starts and ends (in seconds).\n\n"
    "Reply in this exact format:\n"
    "Evidence: <one sentence describing what you see>\n"
    "Label: <one label>\n"
    "Start: <seconds or 0>\n"
    "End: <seconds or 0>\n"
    "Justification: <one sentence explaining the label choice>\n\n"
    "If Label is none, set Start: 0 and End: 0."
)

def classify_chunk(chunk):
    chunk_path = chunk["chunk_path"]
    chunk_dur  = chunk["actual_duration"]
    chunk_idx  = chunk["chunk_idx"]

    # ── load frames via decord ─────────────────────────────────────────────
    frames_np, frame_time_str, video_time = load_video_decord(chunk_path, NUM_FRAMES)

    if DEBUG_SAVE:
        for fi in [0, NUM_FRAMES // 2, NUM_FRAMES - 1]:
            if fi < len(frames_np):
                dbg = cv2.cvtColor(frames_np[fi], cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(DEBUG_DIR, f"chunk_{chunk_idx:04d}_f{fi:02d}.png"), dbg)

    # ── preprocess ─────────────────────────────────────────────────────────
    video_tensor = (
        image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"]
        .to(device=embed_device, dtype=torch.bfloat16)
    )
    video = [video_tensor]

    # ── build prompt (official structure) ──────────────────────────────────
    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are "
        f"uniformly sampled from it. These frames are located at {frame_time_str}. "
        f"Please answer the following questions related to this video."
    )
    question = DEFAULT_IMAGE_TOKEN + "\n" + time_instruction + "\n" + BEHAVIOR_PROMPT

    conv = copy.deepcopy(conv_templates[CONV_TEMPLATE])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # ── tokenise ───────────────────────────────────────────────────────────
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0).to(embed_device)
    )

    n_img = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
    print(f"    img_tokens={n_img}  input_ids={list(input_ids.shape)}  "
          f"video_tensor={list(video_tensor.shape)}")
    if n_img == 0:
        print("    [WARN] No IMAGE_TOKEN_INDEX — visual features will NOT be injected!")

    # ── generate ───────────────────────────────────────────────────────────
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            use_cache=True,
        )

    # Diagnostics — print shapes so we know what was generated
    print(f"    input_ids.shape={list(input_ids.shape)}  "
          f"output_ids.shape={list(output_ids.shape)}  "
          f"new_tokens={output_ids.shape[1] - input_ids.shape[1]}")

    # ── decode — FULL output, exactly like the official model card ──────────
    # When llava calls super().generate(inputs_embeds=...), HuggingFace returns
    # ONLY the generated tokens (not input echoed back).
    # So output_ids[:, input_ids.shape[1]:] always slices into empty space.
    # Decode the full output_ids and let skip_special_tokens strip image tokens.
    full_text = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()

    print(f"    [DEBUG] full_text:\n{full_text}\n")

    # Extract just the model's answer (after the prompt text if echoed)
    response = full_text
    # If the prompt was echoed, strip it
    # (the prompt text ends with the last role separator)
    if BEHAVIOR_PROMPT.split("\n")[0] in full_text:
        # split on the start of our prompt to get only the answer part
        parts = full_text.split("If Label is none, set Start: 0 and End: 0.")
        if len(parts) > 1:
            response = parts[-1].strip()

    if not response:
        response = full_text

    print(f"    [DEBUG] response:\n{response}\n")

    result = parse_chunk_response(response, chunk_dur)
    print(f"\n=== CHUNK {chunk_idx} ===")
    print(f"Label: {result['label']}  (raw: '{result['raw_label']}')")
    print(f"Evidence: {result['evidence']}")
    print(f"Justification: {result['justification']}")
    print(f"Start: {result['start_sec']:.2f}  End: {result['end_sec']:.2f}")
    return result

# ─────────────────────────────────────────────────────────
# 7.  Main classification loop
# ─────────────────────────────────────────────────────────
frame_labels    = {}
all_logs        = []
behavior_chunks = []

for chunk in chunk_info:
    i                   = chunk["chunk_idx"]
    chunk_path          = chunk["chunk_path"]
    sf                  = chunk["sf"]
    ef                  = chunk["ef"]
    actual_duration     = chunk["actual_duration"]
    actual_global_start = chunk["actual_global_start"]
    keyframe_offset     = chunk["keyframe_offset"]
    requested_start     = chunk["requested_start"]
    requested_end       = chunk["requested_end"]

    print(f"\n{'='*60}")
    print(f"Chunk {i}/{len(chunk_info)-1}  "
          f"(requested {requested_start:.1f}s-{requested_end:.1f}s, "
          f"actual_start={actual_global_start:.1f}s)")
    print(f"{'='*60}")

    result = classify_chunk(chunk)

    if result["label"] != "none":
        b_start = max(0.0, actual_global_start + result["start_sec"])
        b_end   = min(duration_sec, actual_global_start + result["end_sec"])
        b_sf    = max(0, int(round(b_start * fps)))
        b_ef    = min(total_frames, int(round(b_end * fps)))
        print(f"  Global: {b_start:.1f}s-{b_end:.1f}s  (frames {b_sf}-{b_ef})")
        if keyframe_offset > 0.1:
            print(f"  (Corrected by {keyframe_offset:.1f}s keyframe offset)")
        for f in range(b_sf, b_ef):
            frame_labels[f] = (result["label"], result["evidence"], result["justification"])
        behavior_chunks.append({
            "chunk_idx": i,
            "requested_start": requested_start, "requested_end": requested_end,
            "label": result["label"],
            "global_start": round(b_start,2), "global_end": round(b_end,2),
            "global_start_timestamp": sec_to_ts(b_start),
            "global_end_timestamp":   sec_to_ts(b_end),
            "evidence":      result["evidence"],
            "justification": result["justification"],
        })
    # NOTE: when this chunk is "none" we do NOT delete overlap-region positives.
    # Positive always beats none — a later none must never erase an already
    # detected positive event.

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
# 8.  Merge into behaviour intervals
# ─────────────────────────────────────────────────────────
intervals    = build_intervals(frame_labels, fps)
label_counts = {}
total_bsec   = 0.0
for iv in intervals:
    label_counts[iv["label"]] = label_counts.get(iv["label"], 0) + 1
    total_bsec += iv["duration_sec"]

intervals_out = {
    "video":  VIDEO_PATH, "model": "LLaVA-Video-72B-Qwen2",
    "duration_sec": round(duration_sec,2),
    "duration_timestamp": sec_to_ts(duration_sec),
    "fps": fps, "total_frames": total_frames,
    "chunk_settings": {
        "chunk_sec": CHUNK_SEC, "overlap_sec": OVERLAP_SEC,
        "total_chunks": len(chunk_info), "num_frames_per_chunk": NUM_FRAMES,
    },
    "summary": {
        "total_behavior_intervals": len(intervals),
        "total_behavior_seconds":   round(total_bsec,2),
        "behavior_percentage":      round(100*total_bsec/duration_sec,2)
                                    if duration_sec>0 else 0,
        "label_counts": label_counts,
    },
    "behavior_chunks": behavior_chunks,
    "intervals":       intervals,
}
with open(INTERVALS_PATH, "w") as fp:
    json.dump(intervals_out, fp, indent=2)
print(f"Intervals -> {INTERVALS_PATH}")

# ─────────────────────────────────────────────────────────
# 9.  Summary
# ─────────────────────────────────────────────────────────
L = ["="*60, "BEHAVIOUR DETECTION SUMMARY -- LLaVA-Video-72B-Qwen2", "="*60,
     f"Video   : {VIDEO_PATH}",
     f"Duration: {sec_to_ts(duration_sec)}  ({duration_sec:.1f}s)",
     f"FPS     : {fps:.2f}", f"Chunks  : {len(chunk_info)}  ({NUM_FRAMES} frames each)", ""]
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
# 10.  Annotate output video
# ─────────────────────────────────────────────────────────
print(f"\nAnnotating video -> {OUTPUT_PATH}")
LABEL_COLORS = {
    "hand_biting":    (0,   0, 255),
    "head_banging":   (0, 128, 255),
    "hitting_others": (0,   0, 200),
    "scratching":     (0, 165, 255),
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
        color    = LABEL_COLORS.get(lbl, (255,255,255))
        lbl_disp = lbl.upper().replace("_"," ")
        ts       = sec_to_ts(frame_idx/fps)
        draw_bold_text(frame, ts, 10, 30, 0.7, (255,255,255), 2)
        ls, lt = 1.8, 4
        (tw, th), _ = cv2.getTextSize(lbl_disp, cv2.FONT_HERSHEY_DUPLEX, ls, lt)
        lx = (width-tw)//2; ly = int(height*0.75); px = py = 16
        cv2.rectangle(frame, (lx-px, ly-th-py), (lx+tw+px, ly+py), (0,0,0), -1)
        cv2.rectangle(frame, (lx-px, ly-th-py), (lx+tw+px, ly+py), color, 3)
        draw_bold_text(frame, lbl_disp, lx, ly, ls, color, lt)
        bt = height-70
        ov = frame.copy()
        cv2.rectangle(ov, (0,bt), (width,height), (0,0,0), -1)
        frame = cv2.addWeighted(ov, 0.7, frame, 0.3, 0)
        if ev:
            draw_bold_text(frame, f"Evidence: {ev[:100]}", 10, bt+25, 0.55, (255,255,255), 1)
        if jt:
            draw_bold_text(frame, f"Justification: {jt[:100]}", 10, bt+55, 0.55, (200,200,200), 1)
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
