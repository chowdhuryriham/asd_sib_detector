"""
chunk_cutter.py  —  Run ONCE before any pipeline.
Cuts the source video into overlapping chunks and writes chunk_manifest.json.
Both pipeline_a_gemma.py and pipeline_b_llava_video.py read from that manifest.

Usage:
    pip install static-ffmpeg ffmpeg-python
    python chunk_cutter.py
"""

import os
import json
import subprocess

import static_ffmpeg
static_ffmpeg.add_paths()   # makes ffmpeg + ffprobe available in PATH

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
BASE        = "/raid/aio469/asd_sib_detector"
VIDEO_PATH  = f"{BASE}/data/raw/lab_videos/input_01.mp4"
CHUNK_DIR   = f"{BASE}/shared_chunks"
MANIFEST    = f"{BASE}/shared_chunks/chunk_manifest.json"

CHUNK_SEC   = 10
OVERLAP_SEC = 3
# ─────────────────────────────────────────────────────────


def get_video_info(path):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr}")
    info = json.loads(r.stdout)
    vs = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if vs is None:
        raise RuntimeError("No video stream found")
    num, den = (float(x) for x in vs["r_frame_rate"].split("/"))
    if den == 0 or num / den <= 0:
        raise RuntimeError(f"Invalid FPS in {path}")
    fps      = num / den
    width    = int(vs["width"])
    height   = int(vs["height"])
    duration = float(info["format"]["duration"])
    total_frames = int(round(duration * fps))
    return fps, width, height, duration, total_frames


def get_chunk_actual_duration(chunk_path):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", chunk_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        try:
            return float(json.loads(r.stdout)["format"]["duration"])
        except (KeyError, ValueError):
            pass
    return 0.0


def get_chunk_actual_start(chunk_path):
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time",
        "-of", "csv=p=0",
        "-read_intervals", "%+#1",
        chunk_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        try:
            return float(r.stdout.strip().split("\n")[0])
        except ValueError:
            pass
    return 0.0


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
os.makedirs(CHUNK_DIR, exist_ok=True)

fps, width, height, duration_sec, total_frames = get_video_info(VIDEO_PATH)
stride_sec   = CHUNK_SEC - OVERLAP_SEC
chunk_starts = []
t = 0.0
while t < duration_sec:
    chunk_starts.append(t)
    t += stride_sec

print(f"Video : {VIDEO_PATH}")
print(f"        {duration_sec:.1f}s ({duration_sec/60:.1f} min) | "
      f"{total_frames} frames @ {fps:.2f} fps | {width}×{height}")
print(f"Chunks: {len(chunk_starts)} × {CHUNK_SEC}s  "
      f"(stride {stride_sec}s, overlap {OVERLAP_SEC}s)")
print(f"Output: {CHUNK_DIR}\n")

chunks = []
for i, start in enumerate(chunk_starts):
    end            = min(start + CHUNK_SEC, duration_sec)
    chunk_duration = end - start
    sf = int(round(start * fps))
    ef = int(round(end   * fps))
    chunk_path = os.path.join(CHUNK_DIR, f"chunk_{i:04d}.mp4")

    if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 1000:
        print(f"  Chunk {i:04d}: already exists, skipping cut")
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", VIDEO_PATH,
            "-t", f"{chunk_duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-loglevel", "error",
            chunk_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  Chunk {i:04d}: ffmpeg FAILED\n{r.stderr}")
            continue
        if not (os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 1000):
            print(f"  Chunk {i:04d}: output too small, skipping")
            continue

    actual_duration     = get_chunk_actual_duration(chunk_path) or chunk_duration
    keyframe_offset     = max(0.0, actual_duration - chunk_duration)
    actual_global_start = start - keyframe_offset

    entry = {
        "chunk_idx":           i,
        "chunk_path":          chunk_path,
        "requested_start":     round(start, 4),
        "requested_end":       round(end, 4),
        "sf":                  sf,
        "ef":                  ef,
        "actual_duration":     round(actual_duration, 4),
        "actual_global_start": round(actual_global_start, 4),
        "keyframe_offset":     round(keyframe_offset, 4),
    }
    chunks.append(entry)

    tag = (f"actual_start={actual_global_start:.2f}s  offset={keyframe_offset:.2f}s  "
           if keyframe_offset > 0.1 else "")
    print(f"  Chunk {i:04d}: {start:.1f}s – {end:.1f}s  "
          f"{tag}({os.path.getsize(chunk_path)//1024} KB)")

if not chunks:
    raise RuntimeError("No chunks were created.")

manifest = {
    "video_path":   VIDEO_PATH,
    "fps":          fps,
    "width":        width,
    "height":       height,
    "duration_sec": round(duration_sec, 4),
    "total_frames": total_frames,
    "chunk_sec":    CHUNK_SEC,
    "overlap_sec":  OVERLAP_SEC,
    "stride_sec":   stride_sec,
    "total_chunks": len(chunks),
    "chunks":       chunks,
}

with open(MANIFEST, "w") as fp:
    json.dump(manifest, fp, indent=2)

print(f"\n✓  {len(chunks)} chunks ready")
print(f"✓  Manifest → {MANIFEST}")
print("\nNow run either pipeline:")
print("  python pipelines/pipeline_a_gemma.py")
print("  python pipelines/pipeline_b_llava_video.py")
