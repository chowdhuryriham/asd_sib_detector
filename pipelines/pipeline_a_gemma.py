"""
pipeline_a_gemma.py  —  Gemma-4-31B
ASD Self-Injurious Behaviour Detector

venv : /raid/aio469/venvs/gemma
run  : CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 /raid/aio469/venvs/gemma/bin/python \
           pipelines/pipeline_a_gemma.py

FIXES vs previous version:
  - _parse_time() handles HH:MM:SS, MM:SS, plain seconds — fixes
    "could not convert string to float: '.'" crash when model
    outputs timestamps like "00:05" instead of "5"
  - make_prompt(chunk_dur) injects clip duration into prompt
  - attn_implementation="eager" for driver-state stability
"""

import os
import re
import cv2
import json
import shutil
import torch
import subprocess
import static_ffmpeg
static_ffmpeg.add_paths()
from pathlib import Path
from transformers import AutoProcessor, AutoModelForMultimodalLM

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
BASE           = "/raid/aio469/asd_sib_detector"
MODEL_PATH     = f"{BASE}/models/gemma-4-31b"
VIDEO_PATH     = f"{BASE}/data/raw/lab_videos/input_01.mp4"
OUTPUT_PATH    = f"{BASE}/outputs/gemma/sample_annotated.mp4"
LOG_PATH       = f"{BASE}/logs/pipeline_a_gemma_log.json"
INTERVALS_PATH = f"{BASE}/outputs/gemma/behavior_intervals.json"
SUMMARY_PATH   = f"{BASE}/logs/pipeline_a_gemma_summary.txt"
CHUNK_DIR      = f"{BASE}/outputs/gemma/chunks"
DEBUG_DIR      = f"{BASE}/outputs/gemma/chunk_debug_frames"
DEBUG_SAVE     = True
CHUNK_SEC      = 5
OVERLAP_SEC    = 1
VALID_LABELS   = {"hand_biting", "head_hit", "hitting_others", "scratching", "self_directed_hit",
                  "armflapping", "spinning", "headbanging", "handaction",
                  "unusual_behavior", "none"}

# RGB (lab) subject hint — preserved, currently inactive
# LAB_SUBJECT_HINT = (
#     "You are watching a short video clip from a clinical observation session. "
#     "Focus on the child — a young child wearing a hoodie. "
#     "Hoodie colour is yellow, golden-yellow, or brownish-yellow depending on the lighting. "
#     "When filmed from the front, the hoodie zip may be open, revealing a T-shirt underneath — "
#     "this is still the same child. Track only this child's behaviour throughout the clip."
# )

# Active: skeleton-mode subject hint (RTMPose Halpe-26 render)
LAB_SUBJECT_HINT = (
    "You are looking at a 2D skeleton/keypoint video — RTMPose Halpe-26 render on a black "
    "background. Coloured dots are joints, lines connecting them are bones. There is no "
    "skin, clothing, face, or scenery — only the stick figure(s).\n\n"
    "WHAT YOU MAY SEE ON THE RENDER:\n"
    "- Person IDs labelled near the head: \"P1\", \"P2\", \"P3\" — each in a unique colour "
    "(typically P1/yellow, P2/blue, P3/green).\n"
    "- Joint name labels printed next to each dot: \"Head\", \"Neck\", \"L.Eye\", \"R.Eye\", "
    "\"L.Ear\", \"R.Ear\", \"L.Shldr\", \"R.Shldr\", \"L.Elbow\", \"R.Elbow\", "
    "\"L.Wrist\", \"R.Wrist\", \"L.Hip\", \"R.Hip\", \"L.Knee\", \"R.Knee\", "
    "\"L.Ankle\", \"R.Ankle\", and toe/heel keypoints. Read labels directly — do not "
    "guess topology.\n"
    "- A frame counter in a corner like \"F:45/149\" — use it to anchor times.\n"
    "- Person IDs may swap if the tracker loses identity — judge by motion continuity, "
    "not ID alone.\n"
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
_p.add_argument("--num_frames",  type=int,   default=64)
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
# 1.  Load model
# ─────────────────────────────────────────────────────────
print("Loading Gemma-4-31B (BF16, eager attention) ...")
processor = AutoProcessor.from_pretrained(Path(MODEL_PATH))
model = AutoModelForMultimodalLM.from_pretrained(
    Path(MODEL_PATH),
    dtype="auto",
    device_map="auto",
    attn_implementation="eager",   # stable across driver states
    low_cpu_mem_usage=True,
)
model.eval()
print("Model loaded.\n")

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

def get_chunk_actual_duration(chunk_path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", chunk_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        try:
            return float(json.loads(r.stdout)["format"]["duration"])
        except (KeyError, ValueError):
            pass
    return 0.0

# ─────────────────────────────────────────────────────────
# 3.  Response / label helpers
# ─────────────────────────────────────────────────────────
def extract_final_text(parsed, raw_response):
    if isinstance(parsed, dict):
        for key in ("content", "response", "text"):
            if isinstance(parsed.get(key), str) and parsed[key].strip():
                return parsed[key].strip()
    if isinstance(parsed, list):
        return " ".join(str(item) for item in parsed)
    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()
    return raw_response.strip()

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
    """
    Convert model time output to float seconds.
    Handles:  "5"  "5.3"  "00:05"  "0:05.3"  "00:00:05"  "1:23.4"
    Returns 0.0 on any parse failure.
    """
    s = s.strip().rstrip("s").strip()   # strip trailing 's' e.g. "5s"
    if not s:
        return 0.0
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:                           # MM:SS
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:                         # HH:MM:SS
                return (float(parts[0]) * 3600
                        + float(parts[1]) * 60
                        + float(parts[2]))
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
            # match HH:MM:SS / MM:SS / plain number (require at least one digit)
            m = re.search(r"(\d[\d:\.]*)", line)
            if m: start_sec = _parse_time(m.group(1))
        elif re.match(r"(?i)^end\s*:", line):
            m = re.search(r"(\d[\d:\.]*)", line)
            if m: end_sec = _parse_time(m.group(1))

    # fallback regex sweeps
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
    # Fill same-label gaps of ≤3 frames (~100ms at 30fps) from chunk boundary rounding
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
# 6.  Behaviour prompt — chunk duration injected dynamically
# ─────────────────────────────────────────────────────────
# def make_prompt(chunk_dur):
#     return (
#         "Describe this video carefully in chronological order. "
#         "There may be multiple people in the scene. "
#         "Focus on the child patient's body movements and interactions.\n\n"
#         "Then classify the child's primary behavior using exactly one of these labels:\n"
#         "[hand_biting, head_hit, hitting_others, scratching, self_directed_hit, none]\n\n"
#         "Brief definitions:\n"
#         "- hand_biting: child bites own hand/fingers/wrist\n"
#         "- head_hit: child's head makes contact with any surface (table, wall, floor, object) — includes sudden bending head onto a surface\n"
#         "- hitting_others: child hits/slaps/pushes another person\n"
#         "- scratching: child scratches own head/scalp/skin\n"
#         "- self_directed_hit: child hits/slaps their own body (arm, leg, torso - not head)\n"
#         "- none: none of the above\n\n"
#         f"This clip is {chunk_dur:.1f} seconds long.\n"
#         "Use your reasoning to determine the most accurate label.\n"
#         "If behavior is visible, report the EARLIEST second the behavior first appears and when it ends "
#         "(in plain seconds, e.g. Start: 3.5  — do NOT use HH:MM:SS format).\n\n"
#         "Reply in this exact format:\n"
#         "Evidence: <one sentence describing what you see>\n"
#         "Label: <one label>\n"
#         "Start: <seconds or 0>\n"
#         "End: <seconds or 0>\n"
#         "Justification: <one sentence explaining the label choice>\n\n"
#         "If Label is none, set Start: 0 and End: 0."
#     )


# ─────────────────────────────────────────────────────────
# 6.  Behaviour prompt — skeleton-only video (COMMENTED — RGB prompt active below)
# ─────────────────────────────────────────────────────────
# def make_prompt(chunk_dur):
#     return (
#         "You are looking at a 2D skeleton/keypoint video — RTMPose or OpenPose render "
#         "on a dark background. Coloured dots are joints, lines connecting them are "
#         "bones. There is no skin, clothing, or face — only the stick figure.\n\n"
#         "WHAT YOU MAY SEE ON THE RENDER (use these if visible):\n"
#         "- Person IDs labelled near each head: \"P1\", \"P2\", \"P3\" — each in a unique "
#         "colour (e.g. P1/yellow, P2/blue, P3/green).\n"
#         "- Joint name labels printed next to dots: \"Head\", \"L.Wrist\", \"R.Wrist\", "
#         "\"L.Knee\", \"Hip.C\", \"L.Ankle\", etc. Read these directly — do not guess "
#         "topology.\n"
#         "- A frame counter in a corner like \"F:45/149\" — use it to anchor times "
#         "(frame_idx ÷ fps, fps ≈ 30).\n"
#         "- Person IDs may SWAP between frames if the tracker loses identity. Treat "
#         "the same-position skeleton as the same person even if the ID label "
#         "changes; judge by motion continuity, not by ID alone.\n\n"
#         f"This clip is {chunk_dur:.1f} seconds long.\n\n"
#         "TASK: Decide whether any of the 5 behaviours below occurs anywhere in the "
#         "clip, performed by ANY visible skeleton. Multiple behaviours by different "
#         "people may occur; report every occurrence.\n\n"
#         "GEOMETRIC RULES (no appearance — judge by joint positions and motion):\n\n"
#         "1. hand_biting\n"
#         "   One person's L.Wrist or R.Wrist stays within ~1 head-width of THAT SAME\n"
#         "   person's Head keypoint, with wrist height roughly at face level.\n"
#         "   Even one frame of that configuration is enough — the wrist may\n"
#         "   already be moving away.\n\n"
#         "2. head_hit\n"
#         "   A person's Head keypoint shows a sharp downward displacement, with\n"
#         "   either (a) Head dropping to roughly Hip-level or below, or (b) Head\n"
#         "   reaching the bottom ~10% of the frame (implied table/floor plane).\n"
#         "   Look for a velocity spike followed by deceleration. One such event is\n"
#         "   enough.\n\n"
#         "3. hitting_others\n"
#         "   ONE person's wrist (or elbow) lands at, or passes through, ANOTHER\n"
#         "   person's body keypoints — Head, Neck, Shoulder, Torso, Hip, or limb.\n"
#         "   Even ONE frame of wrist-on-body overlap counts; the arm may already\n"
#         "   be retracting in adjacent frames. Brief contact, gentle taps, and\n"
#         "   light pushes ALL count — do not require obvious aggression.\n"
#         "   The actor must be DIFFERENT from the target.\n\n"
#         "4. scratching\n"
#         "   A person's wrist stays near their OWN Head keypoint with small\n"
#         "   high-frequency oscillation (repeated short forearm strokes) for\n"
#         "   ≥0.5 seconds. Must be repetitive — a single touch does not count.\n\n"
#         "5. self_directed_hit\n"
#         "   A person's wrist trajectory shows fast extension toward THAT SAME\n"
#         "   person's Torso, Hip, Knee, or Thigh keypoint (NOT their Head — that\n"
#         "   is head_hit). High velocity at impact, possibly repeated.\n\n"
#         "6. none\n"
#         "   None of the above geometric patterns are observed.\n\n"
#         "NOISE NOTE: Pose trackers occasionally produce 1-frame keypoint jumps\n"
#         "that are not real motion. Treat a single isolated proximity event as\n"
#         "evidence ONLY if neighbouring frames show the limb visibly approaching\n"
#         "or retracting — not if a joint just teleports and snaps back to its\n"
#         "prior position.\n\n"
#         "Use plain seconds within this clip only — between 0 and "
#         f"{chunk_dur:.1f}. Do NOT use HH:MM:SS format.\n\n"
#         "OUTPUT FORMAT — fill each field after the colon, one per line, no brackets:\n"
#         "Evidence: one sentence describing which joints came close to which, and "
#         "the actor (e.g. \"P1/yellow's R.Wrist overlaps P2/blue's L.Shoulder\")\n"
#         "Label: one of hand_biting, head_hit, hitting_others, scratching, self_directed_hit, none\n"
#         "Start: plain seconds\n"
#         "End: plain seconds\n"
#         "Justification: one sentence on why this label fits the geometry\n\n"
#         "If multiple behaviours occur, repeat the 5-line block for each in\n"
#         "chronological order, separated by one blank line.\n\n"
#         "If none apply, output a single block with Label: none, Start: 0, End: 0."
#     )
# ─────────────────────────────────────────────────────────
# 6.  Behaviour prompt — RGB video, binary classification (gt_label vs none)
# ─────────────────────────────────────────────────────────
# RGB (lab + ASBD) BEHAVIOR_DEFINITIONS — preserved, currently inactive
# BEHAVIOR_DEFINITIONS_RGB = {
#     "hand_biting":
#         "The child bites or mouths their own hand, fingers, or wrist.",
#     "head_hit":
#         "The child's head makes contact with any surface (table, wall, floor, object) — "
#         "includes bending or lowering the head onto a surface, not only forceful banging.",
#     "hitting_others":
#         "The child hits, slaps, or pushes another person.",
#     "scratching":
#         "The child scratches their own head, scalp, or skin (repeated motion).",
#     "self_directed_hit":
#         "The child hits or slaps their own body — arm, leg, or torso (not the head).",
#     "armflapping":
#         "The subject repeatedly flaps or waves both arms or hands up and down or side to side "
#         "in a rhythmic, stereotyped way.",
#     "spinning":
#         "The subject spins or rotates their whole body, or twirls in place, in a repetitive way.",
#     "headbanging":
#         "The subject shows repetitive, rhythmic, stereotyped head movement. ... "
#         "rhythmic, repetitive head motion — surface impact is NOT required.",
#     "handaction":
#         "Repetitive stereotyped hand movements that are not arm-flapping, hand-biting, or hitting "
#         "— e.g., finger-twisting, hand-rubbing, posturing, or rapid hand-fidgeting.",
# }

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
        "the label is 'none' if the subject moves casually or stays still without rhythmic stereotypy.",
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

# 7.  Classify a chunk
# ─────────────────────────────────────────────────────────
def classify_chunk(chunk):
    chunk_path = chunk["chunk_path"]
    chunk_dur  = chunk["actual_duration"]
    chunk_idx  = chunk["chunk_idx"]

    if DEBUG_SAVE:
        for sec in [0, 1, 2]:
            if sec >= chunk_dur: continue
            cmd = ["ffmpeg", "-y", "-ss", str(sec), "-i", chunk_path,
                   "-frames:v", "1", "-f", "image2", "-update", "1",
                   os.path.join(DEBUG_DIR, f"chunk_{chunk_idx:04d}_t{sec}.png"),
                   "-loglevel", "error"]
            subprocess.run(cmd, capture_output=True, text=True)

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": chunk_path},
            {"type": "text",  "text": make_prompt(chunk_dur, GT_LABEL, SUBJECT_HINT)},
        ],
    }]

    # Use Gemma's processor default sampling. The orchestrator now ensures min trim 5s
    # and the chunker skips residues <2s, so chunks have ≥30 native frames at 30 fps —
    # comfortably above the processor's 32-frame default.
    print(f"  chunk_dur={chunk_dur:.2f}s  (processor default sampling)")

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "4096")),
            do_sample=False,
        )

    raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed       = processor.parse_response(raw_response)
    final_text   = extract_final_text(parsed, raw_response)
    print(f"    [DEBUG] final_text:\n{final_text}\n")

    result = parse_chunk_response(final_text, chunk_dur)
    print(f"\n=== CHUNK {chunk_idx} ===")
    print(f"Label: {result['label']}  (raw: '{result['raw_label']}')")
    print(f"Evidence: {result['evidence']}")
    print(f"Justification: {result['justification']}")
    print(f"Start: {result['start_sec']:.2f}  End: {result['end_sec']:.2f}")
    return result

# ─────────────────────────────────────────────────────────
# Per-video processor — reads module globals (VIDEO_PATH, OUTPUT_PATH,
# LOG_PATH, INTERVALS_PATH, SUMMARY_PATH, CHUNK_DIR, DEBUG_DIR, GT_LABEL,
# CHUNK_SEC, OVERLAP_SEC, DEBUG_SAVE) which the dispatcher updates per job.
# ─────────────────────────────────────────────────────────
def _run_one():
    # 4. Probe + chunking
    fps, width, height, duration_sec, total_frames = get_video_info(VIDEO_PATH)
    stride_sec   = CHUNK_SEC - OVERLAP_SEC
    min_chunk_sec = 2.0  # skip residue chunks shorter than this (avoid Gemma processor crashes)
    chunk_starts = []
    if duration_sec <= CHUNK_SEC:
        chunk_starts = [0.0]
    else:
        t = 0.0
        while t < duration_sec:
            remaining = duration_sec - t
            if remaining < min_chunk_sec and chunk_starts:
                break  # residue too short — drop it; preceding chunk(s) already cover the behavior
            chunk_starts.append(t); t += stride_sec

    print(f"Video : {VIDEO_PATH}")
    print(f"        {duration_sec:.1f}s ({duration_sec/60:.1f} min) | "
          f"{total_frames} frames @ {fps:.2f} fps | {width}x{height}")
    print(f"Chunks: {len(chunk_starts)} x {CHUNK_SEC}s  "
          f"overlap={OVERLAP_SEC}s  stride={stride_sec}s\n")

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
        if i % 100 == 0:
            print(f"  Chunk {i}/{len(chunk_starts)-1}: {start:.1f}-{end:.1f}s  "
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
            print(f"  Global: {b_start:.2f}s-{b_end:.2f}s  (frames {b_sf}-{b_ef})")
            if keyframe_offset > 0.1:
                print(f"  (Corrected by {keyframe_offset:.2f}s keyframe offset)")
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
        # Free VRAM after each chunk to avoid buildup across many videos
        torch.cuda.empty_cache()

    print(f"\nChunk log -> {LOG_PATH}")

    # 10. Merge intervals + outputs
    intervals    = build_intervals(frame_labels, fps)
    label_counts = {}
    total_bsec   = 0.0
    for iv in intervals:
        label_counts[iv["label"]] = label_counts.get(iv["label"], 0) + 1
        total_bsec += iv["duration_sec"]

    intervals_out = {
        "video": VIDEO_PATH, "model": "Gemma-4-31B",
        "duration_sec": round(duration_sec,2),
        "duration_timestamp": sec_to_ts(duration_sec),
        "fps": fps, "total_frames": total_frames,
        "chunk_settings": {
            "chunk_sec": CHUNK_SEC, "overlap_sec": OVERLAP_SEC,
            "stride_sec": stride_sec,
            "total_chunks": len(chunk_info),
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

    # 11. Summary
    L = ["="*60, "BEHAVIOUR DETECTION SUMMARY -- Gemma-4-31B", "="*60,
         f"Video   : {VIDEO_PATH}",
         f"Duration: {sec_to_ts(duration_sec)}  ({duration_sec:.1f}s)",
         f"FPS     : {fps:.2f}",
         f"Chunks  : {len(chunk_info)}  "
         f"({CHUNK_SEC}s each, {OVERLAP_SEC}s overlap, {stride_sec}s stride)", ""]
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

    # 12. Annotate video
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
            color    = LABEL_COLORS.get(lbl, (255, 255, 255))
            lbl_disp = lbl.upper().replace("_", " ")
            ts       = sec_to_ts(frame_idx / fps)
            ls, lt = 1.8, 4
            (tw, th), _ = cv2.getTextSize(lbl_disp, cv2.FONT_HERSHEY_DUPLEX, ls, lt)
            lx = (width - tw) // 2; ly = int(height * 0.75); px = py = 16
            cv2.rectangle(frame, (lx-px, ly-th-py), (lx+tw+px, ly+py), (0,0,0), -1)
            cv2.rectangle(frame, (lx-px, ly-th-py), (lx+tw+px, ly+py), color, 3)
            draw_bold_text(frame, lbl_disp, lx, ly, ls, color, lt)
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (width, 88), (0, 0, 0), -1)
            frame = cv2.addWeighted(ov, 0.7, frame, 0.3, 0)
            draw_bold_text(frame, ts, 10, 25, 0.7, (255, 255, 255), 2)
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