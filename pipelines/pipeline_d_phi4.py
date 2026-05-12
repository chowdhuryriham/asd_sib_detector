"""
pipeline_d_phi4.py  —  Phi-4-multimodal-instruct (5.6B)
ASD Self-Injurious Behaviour Detector

Lab    : Microsoft Research (USA)
Model  : microsoft/Phi-4-multimodal-instruct
License: MIT
Paper  : arXiv 2503.01743 (March 2025)

venv : /raid/aio469/venvs/phi4
run  : CUDA_VISIBLE_DEVICES=0,2,3,4 /raid/aio469/venvs/phi4/bin/python \
           pipelines/pipeline_d_phi4.py

OFFICIAL INFERENCE NOTES (from HF model card):
  - No native video input; pass video frames as multiple images
  - Vision prompt: <|user|><|image_1|><|image_2|>...<|image_N|>text<|end|><|assistant|>
  - Model: AutoModelForCausalLM + AutoProcessor, trust_remote_code=True
  - Max 64 frames supported
  - Requires: transformers==4.48.2, torch==2.6.0, flash_attn==2.7.4.post1
  - Decode: output_ids[:, input_ids.shape[1]:] — standard HF slicing (correct here,
    unlike Pipeline B/E where llavaction returns only new tokens)
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
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
BASE           = "/raid/aio469/asd_sib_detector"
MODEL_PATH     = f"{BASE}/models/Phi-4-multimodal"
VIDEO_PATH     = f"{BASE}/data/raw/lab_videos/input_01.mp4"
OUTPUT_PATH    = f"{BASE}/outputs/phi4/sample_annotated.mp4"
LOG_PATH       = f"{BASE}/logs/pipeline_d_phi4_log.json"
INTERVALS_PATH = f"{BASE}/outputs/phi4/behavior_intervals.json"
SUMMARY_PATH   = f"{BASE}/logs/pipeline_d_phi4_summary.txt"
CHUNK_DIR      = f"{BASE}/outputs/phi4/chunks"
DEBUG_DIR      = f"{BASE}/outputs/phi4/chunk_debug_frames"
DEBUG_SAVE     = True
CHUNK_SEC      = 10
OVERLAP_SEC    = 3
NUM_FRAMES     = 32
VALID_LABELS   = {"hand_biting", "head_hit", "hitting_others", "scratching", "self_directed_hit",
                  "armflapping", "spinning", "headbanging", "handaction",
                  "unusual_behavior", "none"}

# RGB (lab) subject hint — preserved, currently inactive
# LAB_SUBJECT_HINT = (
#     "Watch this clinical observation video carefully. "
#     "Focus on the child — a young child wearing a hoodie that may appear "
#     "yellow, golden-yellow, or brownish-yellow depending on lighting. "
#     "Track only this child's behaviour throughout the clip."
# )

# Active: skeleton-mode subject hint (RTMPose Halpe-26 render)
LAB_SUBJECT_HINT = (
    "You are looking at a 2D skeleton/keypoint video — RTMPose Halpe-26 render on a black "
    "background. Coloured dots are joints, lines connecting them are bones. There is no "
    "skin, clothing, face, or scenery — only the stick figure(s).\n\n"
    "WHAT YOU MAY SEE ON THE RENDER:\n"
    "- Person IDs labelled near the head: \"P1\", \"P2\", \"P3\" — each in a unique colour.\n"
    "- Joint name labels printed next to each dot: \"Head\", \"Neck\", \"L.Eye\", \"R.Eye\", "
    "\"L.Ear\", \"R.Ear\", \"L.Shldr\", \"R.Shldr\", \"L.Elbow\", \"R.Elbow\", "
    "\"L.Wrist\", \"R.Wrist\", \"L.Hip\", \"R.Hip\", \"L.Knee\", \"R.Knee\", "
    "\"L.Ankle\", \"R.Ankle\", and toe/heel keypoints. Read labels directly.\n"
    "- A frame counter in a corner like \"F:45/149\".\n"
    "- Person IDs may swap if the tracker loses identity — judge by motion continuity.\n"
    "- The autistic child is the subject. If multiple skeletons are visible, the subject "
    "is whichever one performs the repetitive motion described in the rules below."
)

# ─────────────────────────────────────────────────────────
# CLI overrides — all CONFIG defaults above can be overridden
# ─────────────────────────────────────────────────────────
import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--video",       default=VIDEO_PATH)
_p.add_argument("--output",      default=OUTPUT_PATH)
_p.add_argument("--log",         default=LOG_PATH)
_p.add_argument("--intervals",   default=INTERVALS_PATH)
_p.add_argument("--summary",     default=SUMMARY_PATH)
_p.add_argument("--chunks_dir",  default=CHUNK_DIR)
_p.add_argument("--chunk_sec",   type=float, default=CHUNK_SEC)
_p.add_argument("--overlap_sec", type=float, default=OVERLAP_SEC)
_p.add_argument("--num_frames",  type=int,   default=NUM_FRAMES)
_p.add_argument("--debug_save",  action="store_true", default=DEBUG_SAVE)
_p.add_argument("--gt_label",    default="")
_p.add_argument("--subject_hint", default=LAB_SUBJECT_HINT,
                help="Setting/subject description injected into the prompt")
_p.add_argument("--prompt_mode",  choices=["rgb", "skeleton"], default="skeleton",
                help="rgb: appearance-based prompt for raw video; skeleton: geometric rules for keypoint renders")
_p.add_argument("--manifest",    default="",
                help="JSON file with list of jobs; bypasses single-video flags")
_args, _ = _p.parse_known_args()
VIDEO_PATH     = _args.video
OUTPUT_PATH    = _args.output
LOG_PATH       = _args.log
INTERVALS_PATH = _args.intervals
SUMMARY_PATH   = _args.summary
CHUNK_DIR      = _args.chunks_dir
CHUNK_SEC      = _args.chunk_sec
OVERLAP_SEC    = _args.overlap_sec
NUM_FRAMES     = _args.num_frames
DEBUG_SAVE     = _args.debug_save
GT_LABEL       = _args.gt_label
SUBJECT_HINT   = _args.subject_hint
PROMPT_MODE    = _args.prompt_mode
MANIFEST_PATH  = _args.manifest
del _ap, _p, _args

# ─────────────────────────────────────────────────────────
# 1.  Load model — official Phi-4-multimodal loading pattern
# ─────────────────────────────────────────────────────────
n_gpus = torch.cuda.device_count()
print(f"Visible GPUs: {n_gpus}")
if n_gpus == 0:
    raise RuntimeError("No GPUs visible. Set CUDA_VISIBLE_DEVICES=0,2,3,4.")

print(f"Loading Phi-4-multimodal from {MODEL_PATH} ...")
processor = AutoProcessor.from_pretrained(
    Path(MODEL_PATH),
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    Path(MODEL_PATH),
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation="flash_attention_2",  # H200 supports this; change to "eager" if issues
).eval()
generation_config = GenerationConfig.from_pretrained(Path(MODEL_PATH))
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
    "head_banging": "head_hit",          "headbanging": "head_hit",
    "head_bang": "head_hit",             "head_hitting": "head_hit",
    "hitting_head": "head_hit",          "banging_head": "head_hit",
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

def _parse_time(s):
    s = s.strip().rstrip("s").strip()
    if not s: return 0.0
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0

def parse_chunk_response(text, chunk_duration):
    label = evidence = justification = description = unusual = ""
    start_sec = end_sec = 0.0
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"(?i)^description\s*:", line):
            description = re.sub(r"(?i)^description\s*:\s*", "", line).strip()
        elif re.match(r"(?i)^unusual\s*:", line):
            unusual = re.sub(r"(?i)^unusual\s*:\s*", "", line).strip()
        elif re.match(r"(?i)^label\s*:", line):
            label = re.sub(r"(?i)^label\s*:\s*", "", line).strip()
        elif re.match(r"(?i)^evidence\s*:", line):
            evidence = re.sub(r"(?i)^evidence\s*:\s*", "", line).strip()
        elif re.match(r"(?i)^justification\s*:", line):
            justification = re.sub(r"(?i)^justification\s*:\s*", "", line).strip()
        elif re.match(r"(?i)^start\s*:", line):
            m = re.search(r"(\d[\d:\.]*)", line)
            if m: start_sec = _parse_time(m.group(1))
        elif re.match(r"(?i)^end\s*:", line):
            m = re.search(r"(\d[\d:\.]*)", line)
            if m: end_sec = _parse_time(m.group(1))
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
        m = re.search(r"(?i)start\s*:\s*(\d[\d:\.]*)", text)
        if m: start_sec = _parse_time(m.group(1))
    if end_sec == 0.0:
        m = re.search(r"(?i)end\s*:\s*(\d[\d:\.]*)", text)
        if m: end_sec = _parse_time(m.group(1))
    raw_label = label
    label = normalize_label(label)
    if raw_label and label != "none" and raw_label.lower().strip() != label:
        print(f"    [INFO] '{raw_label}' -> '{label}'")
    start_sec = max(0.0, min(start_sec, chunk_duration))
    end_sec   = max(0.0, min(end_sec,   chunk_duration))
    if label == "none":
        start_sec = end_sec = 0.0
    if label != "none" and end_sec <= start_sec:
        if start_sec == 0.0 and end_sec == 0.0:
            # model gave valid label but no timestamps — assume whole clip
            end_sec = chunk_duration
            print("    [INFO] start=end=0 with non-none label -> spanning whole clip")
        else:
            print("    [WARN] invalid time range -> dropping to none")
            label = "none"; start_sec = end_sec = 0.0
    return {"label": label, "evidence": evidence, "justification": justification,
            "description": description, "unusual": unusual,
            "start_sec": start_sec, "end_sec": end_sec,
            "raw_label": raw_label, "raw_text": text[:2000]}

def sec_to_ts(sec):
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def build_intervals(frame_labels, fps):
    if not frame_labels: return []
    GAP_FILL = 3
    sorted_f = sorted(frame_labels.keys())
    for idx in range(len(sorted_f) - 1):
        a, b = sorted_f[idx], sorted_f[idx + 1]
        if 1 < b - a <= GAP_FILL + 1 and frame_labels[a][0] == frame_labels[b][0]:
            for g in range(a + 1, b):
                frame_labels[g] = frame_labels[a]
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

# Sections 4 & 5 (chunking + cut chunks) moved into _run_one() at bottom of file.

# ─────────────────────────────────────────────────────────
# 6.  Behaviour prompt
# ─────────────────────────────────────────────────────────
# BEHAVIOR_PROMPT = (
#     "These are uniformly sampled frames from a video clip. "
#     "Describe what you observe in the video carefully in chronological order. "
#     "There may be multiple people in the scene. "
#     "Focus on the child patient's body movements and interactions.\n\n"
#     "Then classify the child's primary behavior using exactly one of these labels:\n"
#     "[hand_biting, head_hit, hitting_others, scratching, self_directed_hit, none]\n\n"
#     "Brief definitions:\n"
#     "- hand_biting: child bites own hand/fingers/wrist\n"
#     "- head_hit: child's head makes contact with any surface (table, wall, floor, object) — includes sudden bending head onto a surface\n"
#     "- hitting_others: child hits/slaps/pushes another person\n"
#     "- scratching: child scratches own head/scalp/skin\n- self_directed_hit: child hits/slaps their own body (arm, leg, torso - not head)\n"
#     "- none: none of the above\n\n"
#     "If behavior is visible, report the EARLIEST second the behavior first appears and when it ends.\n"
#     "Use plain seconds only (e.g. Start: 3.5) — do NOT use HH:MM:SS format.\n\n"
#     "Reply in this exact format:\n"
#     "Evidence: <one sentence describing what you see>\n"
#     "Label: <one label>\n"
#     "Start: <seconds or 0>\n"
#     "End: <seconds or 0>\n"
#     "Justification: <one sentence explaining the label choice>\n\n"
#     "If Label is none, set Start: 0 and End: 0."
# )

# (skeleton make_prompt removed — RGB prompt active below)

# RGB BEHAVIOR_DEFINITIONS — preserved, currently inactive
# BEHAVIOR_DEFINITIONS_RGB = { ...lab + ASBD textual definitions... }

# RGB behavior definitions — concrete visual signatures for raw video.
BEHAVIOR_DEFINITIONS_RGB = {
    "armflapping":
        "The subject repeatedly flaps or waves both arms or hands up and down or side to side "
        "in a rhythmic, stereotyped way.",
    "spinning":
        "The subject spins or rotates their whole body, or twirls in place, in a repetitive way.",
    "headbanging":
        "The subject repeatedly moves their head — nodding forcefully, shaking side-to-side, "
        "or striking their head against a surface — in a rhythmic, stereotyped way. "
        "Surface impact is NOT required.",
    "handaction":
        "Repetitive stereotyped hand movements that are not arm-flapping — e.g., "
        "finger-twisting, hand-rubbing, wrist-flapping, or rapid hand-fidgeting.",
    "unusual_behavior":
        "ANY clearly repetitive, stereotyped, ASD-related body movement — armflapping, "
        "spinning, or headbanging.",  # handaction removed: too broad, caused normal-clip false positives
}

# Skeleton behavior definitions — geometric rules for keypoint/pose renders.
# Softened: each rule accepts noisy/partial pose tracking and lower cycle counts.
BEHAVIOR_DEFINITIONS_SKELETON = {
    "armflapping":
        "The subject's L.Wrist and/or R.Wrist exhibit repeated, rhythmic up-down "
        "(or in-out) motion relative to their L.Shldr / R.Shldr. Motion repeats over "
        "time. Bilateral arm involvement is the typical pattern, but one arm may be "
        "intermittently lost or jittery in the pose tracker — accept the label if one "
        "wrist clearly oscillates and the other moves above its baseline at any point, "
        "or if both wrists move in any rhythmic pattern even briefly. Wrists are NOT "
        "in sustained contact with the Head.",
    "spinning":
        "The skeleton's body orientation rotates about a vertical axis. Indicators "
        "(any one suffices): (a) horizontal positions of L.Shldr / R.Shldr swap or "
        "rotate (front view ↔ back view); (b) L.Hip / R.Hip similarly rotate; (c) the "
        "body centroid stays roughly stationary while orientation changes; (d) limbs "
        "extend outward as the person rotates; (e) a clear ≥90° change in shoulder/hip "
        "orientation anywhere in the clip. Partial rotation qualifies — full rotation "
        "not required.",
    "headbanging":
        "The Head keypoint shows repetitive motion — at least one of: "
        "(a) forward-backward oscillation toward an implied surface; "
        "(b) side-to-side shaking (Head displaces left-right while Neck stays roughly fixed); "
        "(c) vertical nodding (Head moves up-down); "
        "(d) Head approaches Hip-level or below repeatedly (banging on table/floor). "
        "Two or more cycles is sufficient. Surface impact is NOT required. Tracker noise "
        "may obscure exact timing — accept the label when neighbouring frames show "
        "consistent direction-of-motion alternation, even if some frames lose the Head.",
    "handaction":
        "Repetitive stereotyped hand/wrist motion that is NOT both-arm flapping. Patterns: "
        "ONE wrist (L.Wrist OR R.Wrist) showing rhythmic oscillation while the other "
        "arm is relatively still; the two wrists rubbing or repeatedly contacting each "
        "other; or rapid rhythmic finger/wrist twisting near the wrist keypoint. "
        "Two or more cycles, or any clear repetitive stereotyped wrist pattern, qualifies.",
    "unusual_behavior":
        "ANY repetitive, stereotyped, ASD-related body movement — i.e. ANY of: "
        "armflapping (rhythmic up-down arm motion), spinning (whole-body rotation about "
        "a vertical axis), or headbanging (rhythmic head motion or banging). "
        "The label is 'unusual_behavior' if ANY of these patterns is present; "
        "the label is 'none' if the subject moves "
        "casually or stays still without rhythmic stereotypy.",
}

def make_prompt(chunk_dur, gt_label="", subject_hint=""):
    defs = BEHAVIOR_DEFINITIONS_RGB if PROMPT_MODE == "rgb" else BEHAVIOR_DEFINITIONS_SKELETON
    # Binary: gt_label vs none. For behavior clips → gt = armflapping/spinning/headbanging/handaction.
    # For normal samples → gt = unusual_behavior. Fallback to multi-class only if gt is missing.
    if gt_label and gt_label in defs:
        defn_block    = f"- {gt_label}: {defs[gt_label]}"
        label_choices = f"none or {gt_label}"
    else:
        defn_block    = "\n".join(f"- {k}: {v}" for k, v in defs.items())
        label_choices = "none, " + ", ".join(defs.keys())
    hint = subject_hint or LAB_SUBJECT_HINT
    if PROMPT_MODE == "rgb":
        return (
            f"{hint}\n\n"
            f"Watch this {chunk_dur:.1f}-second video clip carefully.\n\n"
            "STEP 1 — DESCRIBE: In 3-4 sentences, describe what the subject is doing. "
            "Mention which body parts move, whether the motion is repetitive/rhythmic or one-off, "
            "and any unusual stereotyped patterns you notice.\n\n"
            "STEP 2 — CLASSIFY: Based on your description, decide if the subject is performing "
            "the specific behaviour defined below.\n\n"
            "BEHAVIOR DEFINITION:\n"
            f"{defn_block}\n\n"
            "Only label the behavior if you are confident the motion is clearly repetitive and stereotyped. If uncertain, output none.\n\n"
            "Reply in this exact format:\n"
            "Description: <3-4 sentences describing the subject's movement>\n"
            f"Label: {label_choices}\n"
            "Start: <seconds when behaviour begins, 0 if none>\n"
            "End: <seconds when behaviour ends, 0 if none>\n"
            "Justification: <one sentence why this label fits>\n\n"
            f"Use plain seconds only (0 to {chunk_dur:.1f}). Do NOT use HH:MM:SS.\n"
            "If Label is none, set Start: 0 and End: 0."
        )
    else:
        return (
            f"{hint}\n\n"
            f"Watch this {chunk_dur:.1f}-second skeleton/keypoint video carefully.\n\n"
            "STEP 1 — DESCRIBE: In 3-4 sentences, describe what the skeleton subject is doing. "
            "Mention which joints move (e.g. L.Wrist, R.Shldr, Head), whether the motion is "
            "repetitive/rhythmic or one-off, and reference frame markers (F:N) when helpful.\n\n"
            "STEP 2 — CLASSIFY: Based on your description, decide if the skeleton is performing "
            "the specific behaviour defined below.\n\n"
            "GEOMETRIC RULE (judge by joint positions and motion):\n"
            f"{defn_block}\n\n"
            "NOISE NOTE: pose trackers occasionally produce 1-frame keypoint jumps; treat a "
            "single isolated event as evidence ONLY if neighbouring frames show a continuous "
            "trajectory.\n\n"
            "Only label the behavior if you are confident the motion is clearly repetitive and stereotyped. If uncertain, output none.\n\n"
            "Reply in this exact format:\n"
            "Description: <3-4 sentences describing the skeleton's movement, naming joints>\n"
            f"Label: {label_choices}\n"
            "Start: <seconds when behaviour begins, 0 if none>\n"
            "End: <seconds when behaviour ends, 0 if none>\n"
            "Justification: <one sentence why this label fits>\n\n"
            f"Use plain seconds only (0 to {chunk_dur:.1f}). Do NOT use HH:MM:SS.\n"
            "If Label is none, set Start: 0 and End: 0."
        )

# ─────────────────────────────────────────────────────────
# 7.  Classify a chunk — official Phi-4-multimodal pattern
#
#   Phi-4-multimodal has no native video input.
#   Official workaround: extract frames → pass as multiple images.
#   Vision prompt format (from HF model card):
#     <|user|><|image_1|><|image_2|>...<|image_N|>text<|end|><|assistant|>
#   Decode: output_ids[:, input_ids.shape[1]:] — standard HF slicing.
# ─────────────────────────────────────────────────────────
def load_frames_as_pil(video_path, num_frames):
    """Extract num_frames uniformly from video. Returns list of PIL Images at original resolution.
    32 frames at full resolution avoids the dynamic-HD multi-tile explosion that occurs with 64+."""
    vr  = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    idx = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
    frames_np = vr.get_batch(idx).asnumpy()   # [N, H, W, 3] uint8
    return [Image.fromarray(f) for f in frames_np]

def classify_chunk(chunk):
    chunk_path = chunk["chunk_path"]
    chunk_dur  = chunk["actual_duration"]
    chunk_idx  = chunk["chunk_idx"]

    # ── load frames as PIL images ────────────────────────────────────────────
    pil_frames = load_frames_as_pil(chunk_path, NUM_FRAMES)
    print(f"    frames={len(pil_frames)}  size={pil_frames[0].size}")

    if DEBUG_SAVE:
        for fi in [0, NUM_FRAMES // 2, NUM_FRAMES - 1]:
            dbg = cv2.cvtColor(np.array(pil_frames[fi]), cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(DEBUG_DIR, f"chunk_{chunk_idx:04d}_f{fi:02d}.png"), dbg)

    # ── build prompt — official Phi-4-multimodal vision format ───────────────
    # Each frame gets a placeholder: <|image_1|>, <|image_2|>, ..., <|image_N|>
    image_placeholders = "".join([f"<|image_{i+1}|>" for i in range(len(pil_frames))])
    prompt = (
        f"<|user|>{image_placeholders}"
        f"{make_prompt(chunk_dur, GT_LABEL, SUBJECT_HINT)}"
        f"<|end|><|assistant|>"
    )

    # ── tokenise + encode images ─────────────────────────────────────────────
    inputs = processor(
        text=prompt,
        images=pil_frames,
        return_tensors="pt",
    ).to("cuda:0")

    input_len = inputs["input_ids"].shape[1]
    print(f"    input_ids={list(inputs['input_ids'].shape)}")

    # ── generate ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "2048")),
            generation_config=generation_config,
        )

    # Standard HF slicing — Phi-4 generate() returns full input+output sequence
    output_ids = output_ids[:, input_len:]
    new_tokens = output_ids.shape[1]
    print(f"    new_tokens={new_tokens}")

    response = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
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
# Per-video processor — reads module globals (VIDEO_PATH, etc.)
# ─────────────────────────────────────────────────────────
def _run_one():
    # 4. Probe + chunking
    fps, width, height, duration_sec, total_frames = get_video_info(VIDEO_PATH)
    stride_sec   = CHUNK_SEC - OVERLAP_SEC
    chunk_starts = []
    if duration_sec <= CHUNK_SEC:
        chunk_starts = [0.0]
    else:
        t = 0.0
        while t < duration_sec:
            chunk_starts.append(t); t += stride_sec

    print(f"Video : {VIDEO_PATH}")
    print(f"        {duration_sec:.1f}s ({duration_sec/60:.1f} min) | "
          f"{total_frames} frames @ {fps:.2f} fps | {width}x{height}")
    print(f"Chunks: {len(chunk_starts)} x {CHUNK_SEC}s  stride={stride_sec}s\n")

    # 5. Cut chunks
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

    # 8. Resume
    all_logs     = []
    done_indices = set()
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH) as fp:
                all_logs = json.load(fp)
            done_indices = {entry["chunk_idx"] for entry in all_logs}
            print(f"[RESUME] Found existing log with {len(done_indices)} processed chunks.")
        except Exception as e:
            print(f"[RESUME] Could not load existing log ({e}), starting fresh.")
            all_logs = []; done_indices = set()

    frame_labels    = {}
    behavior_chunks = []

    for entry in all_logs:
        if entry.get("label", "none") != "none":
            matching = next((c for c in chunk_info if c["chunk_idx"] == entry["chunk_idx"]), None)
            if matching:
                b_start = max(0.0, matching["actual_global_start"] + entry["start_sec"])
                b_end   = min(duration_sec, matching["actual_global_start"] + entry["end_sec"])
                b_sf    = max(0, int(round(b_start * fps)))
                b_ef    = min(total_frames, int(round(b_end * fps)))
                for f in range(b_sf, b_ef):
                    frame_labels[f] = (entry["label"], entry["evidence"], entry["justification"])
                behavior_chunks.append({
                    "chunk_idx": entry["chunk_idx"],
                    "requested_start": entry["requested_start"],
                    "requested_end":   entry["requested_end"],
                    "label":           entry["label"],
                    "global_start":    round(b_start, 2),
                    "global_end":      round(b_end, 2),
                    "global_start_timestamp": sec_to_ts(b_start),
                    "global_end_timestamp":   sec_to_ts(b_end),
                    "evidence":      entry["evidence"],
                    "justification": entry["justification"],
                })

    # 9. Main loop
    for chunk in chunk_info:
        i                   = chunk["chunk_idx"]
        if i in done_indices:
            continue
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
        torch.cuda.empty_cache()

    print(f"\nChunk log -> {LOG_PATH}")

    # 10. Merge intervals
    intervals    = build_intervals(frame_labels, fps)
    label_counts = {}
    total_bsec   = 0.0
    for iv in intervals:
        label_counts[iv["label"]] = label_counts.get(iv["label"], 0) + 1
        total_bsec += iv["duration_sec"]

    intervals_out = {
        "video": VIDEO_PATH, "model": "Phi-4-multimodal-instruct",
        "lab":   "Microsoft Research (USA)", "license": "MIT",
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

    L = ["="*60, "BEHAVIOUR DETECTION SUMMARY -- Phi-4-multimodal-instruct", "="*60,
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

    # 11. Annotate video
    print(f"\nAnnotating video -> {OUTPUT_PATH}")
    LABEL_COLORS = {
        "hand_biting":    (0,   0, 255),
        "head_hit":       (0, 128, 255),
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
            color    = LABEL_COLORS.get(lbl, (255,255,255))
            lbl_disp = lbl.upper().replace("_"," ")
            ts       = sec_to_ts(frame_idx/fps)
            ls, lt = 1.8, 4
            (tw, th), _ = cv2.getTextSize(lbl_disp, cv2.FONT_HERSHEY_DUPLEX, ls, lt)
            lx = (width-tw)//2; ly = int(height*0.75); px = py = 16
            cv2.rectangle(frame, (lx-px, ly-th-py), (lx+tw+px, ly+py), (0,0,0), -1)
            cv2.rectangle(frame, (lx-px, ly-th-py), (lx+tw+px, ly+py), color, 3)
            draw_bold_text(frame, lbl_disp, lx, ly, ls, color, lt)
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (width, 88), (0, 0, 0), -1)
            frame = cv2.addWeighted(ov, 0.7, frame, 0.3, 0)
            draw_bold_text(frame, ts, 10, 25, 0.7, (255,255,255), 2)
            if ev:
                draw_bold_text(frame, f"Evidence: {ev[:100]}", 10, 52,
                               0.55, (255,255,255), 1)
            if jt:
                draw_bold_text(frame, f"Justification: {jt[:100]}", 10, 74,
                               0.55, (200,200,200), 1)
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


# ─────────────────────────────────────────────────────────
# Dispatcher: --manifest mode (multi-video) or single-video mode
# ─────────────────────────────────────────────────────────
if MANIFEST_PATH:
    with open(MANIFEST_PATH) as _fp:
        _JOBS = json.load(_fp)
    print(f"\n[MANIFEST] {len(_JOBS)} jobs from {MANIFEST_PATH}")
    for _i, _job in enumerate(_JOBS):
        print(f"\n{'#'*80}\n# JOB {_i+1}/{len(_JOBS)}: {_job['video']}\n{'#'*80}")
        VIDEO_PATH     = _job['video']
        OUTPUT_PATH    = _job['output']
        LOG_PATH       = _job['log']
        INTERVALS_PATH = _job['intervals']
        SUMMARY_PATH   = _job['summary']
        CHUNK_DIR      = _job['chunks_dir']
        DEBUG_DIR      = _job.get('debug_dir',
                                  os.path.join(os.path.dirname(SUMMARY_PATH),
                                               'chunk_debug_frames'))
        GT_LABEL       = _job.get('gt_label', '')
        SUBJECT_HINT   = _job.get('subject_hint', LAB_SUBJECT_HINT)
        PROMPT_MODE    = _job.get('prompt_mode', 'skeleton')
        if 'chunk_sec'   in _job: CHUNK_SEC   = _job['chunk_sec']
        if 'overlap_sec' in _job: OVERLAP_SEC = _job['overlap_sec']
        try:
            _run_one()
        except Exception as _e:
            import traceback
            print(f"[JOB FAILED] {VIDEO_PATH}: {_e}")
            traceback.print_exc()
else:
    _run_one()