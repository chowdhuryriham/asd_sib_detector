"""
cutter_chunk.py  —  Ground-Truth Chunk Cutter
ASD Self-Injurious Behaviour Detector

Cuts 5-second clips from input_01.mp4 centered on each ground-truth
event, then saves a manifest.json that eval_gt.py reads.

run : python cutter_chunk.py

Output:
    data/gt_chunks/<label>_<idx>_<start>-<end>.mp4   (15 clips)
    data/gt_chunks/manifest.json
"""

import json
import os
import subprocess
import static_ffmpeg
static_ffmpeg.add_paths()

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
BASE        = "/raid/aio469/asd_sib_detector"
VIDEO_PATH  = f"{BASE}/data/raw/lab_videos/input_01.mp4"
OUT_DIR     = f"{BASE}/data/gt_chunks"
CHUNK_SEC   = 5.0

# ─────────────────────────────────────────────────────────
# GROUND TRUTH  (seconds in input_01.mp4)
# GT label  →  pipeline label mapping
# ─────────────────────────────────────────────────────────
GT_TO_PIPE = {
    "Hitting":                   "hitting_others",
    "Scratching the Head (SIB)": "scratching",
    "Hand Bite (SIB)":           "hand_biting",
    "Head Hit (SIB)":            "head_hit",
    "Self-Directed Hit (SIB)":   "self_directed_hit",
}

GROUND_TRUTH = {
    "Hitting": [
        (914,    915),
        (1245.9, 1246.3),
        (1264.6, 1266),
        (1269.1, 1270.1),
        (1278.2, 1278.9),
        (1375.35,1376.2),
        (2133,   2134.5),
        (2616.5, 2617.5),
    ],
    "Scratching the Head (SIB)": [
        (1283.9, 1285),
        (2041.6, 2045.3),
    ],
    "Hand Bite (SIB)": [
        (1198,   1200.7),
        (2103,   2107),
    ],
    "Head Hit (SIB)": [
        (1200.85, 1201.75),
    ],
    "Self-Directed Hit (SIB)": [
        (1415.9, 1416.6),
        (2648.9, 2649.9),
    ],
}

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def get_video_duration(path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr}")
    return float(json.loads(r.stdout)["format"]["duration"])

def get_actual_duration(path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        try:
            return float(json.loads(r.stdout)["format"]["duration"])
        except (KeyError, ValueError):
            pass
    return 0.0

def cut_chunk(video_path, chunk_start, out_path):
    """Re-encode a CHUNK_SEC-second clip for exact frame-accurate duration."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{chunk_start:.3f}",
        "-i", video_path,
        "-t", f"{CHUNK_SEC:.3f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        "-loglevel", "error",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {r.stderr}")
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 500:
        raise RuntimeError(f"Output too small or missing: {out_path}")

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

video_dur = get_video_duration(VIDEO_PATH)
print(f"Video : {VIDEO_PATH}")
print(f"Duration: {video_dur:.2f}s  ({video_dur/60:.1f} min)\n")

manifest = {
    "video": VIDEO_PATH,
    "video_duration_sec": round(video_dur, 3),
    "chunk_sec": CHUNK_SEC,
    "chunks": [],
}

total = 0
skipped = 0

for gt_label_orig, events in GROUND_TRUTH.items():
    pipe_label = GT_TO_PIPE[gt_label_orig]
    for idx, (ev_start, ev_end) in enumerate(events, start=1):
        # 5-second window centered on the event midpoint
        center      = (ev_start + ev_end) / 2.0
        chunk_start = max(0.0, center - CHUNK_SEC / 2.0)
        chunk_end   = chunk_start + CHUNK_SEC
        if chunk_end > video_dur:
            chunk_end   = video_dur
            chunk_start = max(0.0, chunk_end - CHUNK_SEC)

        # event offset within the chunk (for reference)
        event_offset_in_chunk = round(ev_start - chunk_start, 3)

        # filename: label_idx_evstart-evend.mp4  (dots replaced with p)
        def _fmt(v):
            s = f"{v:.1f}".rstrip("0").rstrip(".")
            return s if s else "0"

        chunk_name = f"{pipe_label}_{idx:02d}_{_fmt(ev_start)}-{_fmt(ev_end)}"
        chunk_path = os.path.join(OUT_DIR, f"{chunk_name}.mp4")

        if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 500:
            actual_dur = get_actual_duration(chunk_path)
            print(f"  [SKIP] {chunk_name}.mp4  (already exists, {actual_dur:.2f}s)")
            skipped += 1
        else:
            try:
                cut_chunk(VIDEO_PATH, chunk_start, chunk_path)
                actual_dur = get_actual_duration(chunk_path) or CHUNK_SEC
                size_kb = os.path.getsize(chunk_path) // 1024
                print(f"  [CUT]  {chunk_name}.mp4  "
                      f"window=[{chunk_start:.1f}, {chunk_end:.1f}]  "
                      f"actual={actual_dur:.2f}s  {size_kb}KB")
            except RuntimeError as e:
                print(f"  [FAIL] {chunk_name}: {e}")
                continue

        manifest["chunks"].append({
            "chunk_name":            chunk_name,
            "chunk_path":            chunk_path,
            "gt_label":              pipe_label,
            "gt_label_original":     gt_label_orig,
            "event_start":           ev_start,
            "event_end":             ev_end,
            "event_duration_sec":    round(ev_end - ev_start, 3),
            "chunk_start":           round(chunk_start, 3),
            "chunk_end":             round(chunk_end, 3),
            "actual_duration":       round(actual_dur, 3),
            "event_offset_in_chunk": event_offset_in_chunk,
        })
        total += 1

manifest_path = os.path.join(OUT_DIR, "manifest.json")
with open(manifest_path, "w") as fp:
    json.dump(manifest, fp, indent=2)

print(f"\n{'='*60}")
print(f"Done.  {total} chunks  ({skipped} skipped / already exist)")
print(f"\nChunks  : {OUT_DIR}/")
print(f"Manifest: {manifest_path}")
print(f"{'='*60}")
print("\nChunk summary:")
print(f"  {'LABEL':<25} {'#':>3}  {'EVENT':<22}  {'CHUNK WINDOW':<20}  ACTUAL   OFFSET")
print(f"  {'-'*25} {'-'*3}  {'-'*22}  {'-'*20}  {'-'*6}  {'-'*6}")
for i, c in enumerate(manifest["chunks"], start=1):
    print(f"  {c['gt_label']:<25} {i:>3}"
          f"  [{c['event_start']:.1f} – {c['event_end']:.1f}]"
          f"          [{c['chunk_start']:.1f} – {c['chunk_end']:.1f}]"
          f"  {c['actual_duration']:.2f}s"
          f"   @{c['event_offset_in_chunk']:.1f}s")
