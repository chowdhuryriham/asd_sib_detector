#!/usr/bin/env python3
"""
run_asbd_eval.py — run pipelines A, D, E, F on the ASBD dataset
                   (SSBD / ESBD / Wei_BD), per-source results table.

ASBD differs from gt_chunks:
  - Behavior labels: armflapping / spinning / headbanging->head_hit / handaction
  - Videos are full YouTube clips, NOT pre-trimmed; each row in metadata.csv
    has behavior_start_sec / behavior_end_sec — we pre-trim with 1s padding.
  - In-the-wild settings (not a clinical room) — pipelines receive a generic
    'autistic child' subject_hint instead of the lab's yellow-hoodie phrasing.

Usage:
    python3 run_asbd_eval.py [--cuda 4,5] [--asbd_dir data/asbd]
                             [--out_dir outputs/asbd_eval]
                             [--pipelines a,d,e,f] [--runs 1]
                             [--sources ssbd,esbd,wei_bd]
                             [--chunk_sec 10] [--pad_sec 1.0] [--limit N]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Reuse helpers from the lab eval orchestrator
from run_gt_eval import (
    PIPELINES,
    FFMPEG_BIN,
    run_pipeline_batch,
    read_intervals,
    print_table,
    ts,
)

PROJECT = Path(__file__).parent

ASBD_BEHAVIORS = ["armflapping", "headbanging", "spinning", "handaction", "none"]

# ASBD-specific category-to-label normaliser. Lowercase the category and
# apply alias map. Note: 'headbanging' stays distinct from the lab's 'head_hit'.
ASBD_LABEL_ALIASES = {
    "headbanging":   "headbanging",
    "head_banging":  "headbanging",
    "armflapping":   "armflapping",
    "arm_flapping":  "armflapping",
    "spinning":      "spinning",
    "handaction":    "handaction",
    "hand_action":   "handaction",
}

# Subject hint used when running on raw RGB video (in-the-wild ASBD clips).
ASBD_SUBJECT_HINT_RGB = (
    "You are watching a short video clip of an autistic child or person, recorded "
    "in an everyday setting (home, school, public space) — not a clinical room. "
    "Focus on the autistic child or person who is the subject of this clip — "
    "they will be the one showing repetitive or stereotyped body movement. "
    "Multiple people may be visible; the subject is the one performing the unusual "
    "motion. Track only this subject's behaviour throughout the clip."
)

# Keep old name as alias for backwards compatibility.
ASBD_SUBJECT_HINT = ASBD_SUBJECT_HINT_RGB

# Subject hint used when running on skeleton/keypoint renders of ASBD clips.
ASBD_SUBJECT_HINT_SKELETON = (
    "You are looking at a 2D skeleton/keypoint video — RTMPose Halpe-26 render on a black "
    "background. Coloured dots are joints, lines connecting them are bones. There is no "
    "skin, clothing, face, or scenery — only the stick figure(s).\n\n"
    "WHAT YOU MAY SEE ON THE RENDER:\n"
    "- Person IDs labelled near the head: \"P1\", \"P2\", \"P3\" — each in a unique colour.\n"
    "- Joint name labels printed next to each dot: \"Head\", \"Neck\", \"L.Shldr\", \"R.Shldr\", "
    "\"L.Elbow\", \"R.Elbow\", \"L.Wrist\", \"R.Wrist\", \"L.Hip\", \"R.Hip\", \"L.Knee\", "
    "\"R.Knee\", \"L.Ankle\", \"R.Ankle\", and toe/heel keypoints. Read labels directly.\n"
    "- A frame counter in a corner like \"F:45/149\".\n"
    "- Person IDs may swap if the tracker loses identity — judge by motion continuity.\n"
    "- The autistic child is the subject. If multiple skeletons are visible, the subject "
    "is whichever one performs the repetitive motion described in the rules below."
)


def normalise_label(category: str) -> str:
    return ASBD_LABEL_ALIASES.get(category.strip().lower(), category.strip().lower())


MIN_TRIM_SEC = 5.0

def trim_video(src: Path, dst: Path, start_sec: float, end_sec: float,
               pad_sec: float, video_duration: float) -> bool:
    """Trim src around the event window with at least MIN_TRIM_SEC of context.
    Window is centred on the event midpoint; clamped to [0, video_duration].
    Returns True if dst exists with non-trivial size after the call."""
    if dst.exists() and dst.stat().st_size > 1000:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Target window: max(MIN_TRIM_SEC, event + 2*pad). Centre on midpoint.
    event_dur = max(0.0, end_sec - start_sec)
    target = max(MIN_TRIM_SEC, event_dur + 2 * pad_sec)
    midpoint = (start_sec + end_sec) / 2.0
    a = max(0.0, midpoint - target / 2.0)
    if video_duration > 0:
        b = min(video_duration, a + target)
        # If we hit the right edge, try shifting left to preserve target size
        if b - a < target:
            a = max(0.0, b - target)
    else:
        b = a + target
    duration = max(0.5, b - a)

    cmd = [FFMPEG_BIN, "-y",
           "-ss", f"{a:.3f}", "-i", str(src),
           "-t",  f"{duration:.3f}",
           "-c", "copy", "-avoid_negative_ts", "make_zero",
           str(dst), "-loglevel", "error"]
    subprocess.run(cmd, capture_output=True)
    if not (dst.exists() and dst.stat().st_size > 1000):
        # copy failed (codec edge case) — fallback to re-encode
        cmd_re = [FFMPEG_BIN, "-y",
                  "-ss", f"{a:.3f}", "-i", str(src),
                  "-t",  f"{duration:.3f}",
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                  "-an", str(dst), "-loglevel", "error"]
        subprocess.run(cmd_re, capture_output=True)
    return dst.exists() and dst.stat().st_size > 1000


def _normalise_source(s: str) -> str:
    """Normalise source name: 'WEI BD' / 'WEI_BD' / 'wei bd' all → 'wei_bd'."""
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def load_asbd_rows(metadata_csv: Path, sources: set, include_timing_issues: bool):
    """Read metadata.csv, filter to usable rows. Returns list of dicts."""
    rows = []
    with metadata_csv.open() as fp:
        for r in csv.DictReader(fp):
            if _normalise_source(r.get("source", "")) not in sources:
                continue
            if r.get("download_status", "") not in ("downloaded", "exists"):
                continue
            path = r.get("downloaded_path", "").strip()
            if not path or not (PROJECT / path).exists():
                continue
            try:
                start = float(r["behavior_start_sec"])
                end   = float(r["behavior_end_sec"])
                vdur  = float(r.get("video_duration_sec") or 0.0)
            except (KeyError, ValueError, TypeError):
                continue
            if end <= start:
                continue
            if not include_timing_issues and r.get("timing_issue", "").lower() == "true":
                continue
            label = normalise_label(r.get("category", ""))
            if label not in ASBD_BEHAVIORS:
                continue
            rows.append({
                "source":      _normalise_source(r["source"]),
                "video":       r["video"].strip(),
                "src_path":    PROJECT / path,
                "start_sec":   start,
                "end_sec":     end,
                "video_dur":   vdur,
                "gt_label":    label,
                "category":    r.get("category", ""),
            })
    return rows


def build_asbd_job(row: dict, run_out: Path, trimmed_video: Path,
                   chunk_sec: int, prompt_mode: str = "rgb",
                   subject_hint: str = "") -> dict:
    """Build one manifest job for an ASBD pre-trimmed clip."""
    run_out.mkdir(parents=True, exist_ok=True)
    stem = trimmed_video.stem
    return {
        "video":        str(trimmed_video),
        "output":       str(run_out / f"{stem}_annotated.mp4"),
        "log":          str(run_out / "log.json"),
        "intervals":    str(run_out / "behavior_intervals.json"),
        "summary":      str(run_out / "summary.txt"),
        "chunks_dir":   str(run_out / "chunks"),
        "debug_dir":    str(run_out / "chunk_debug_frames"),
        "gt_label":     row["gt_label"],
        "chunk_sec":    chunk_sec,
        "overlap_sec":  0,
        "prompt_mode":  prompt_mode,
        "subject_hint": subject_hint,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda",      default="4,5")
    ap.add_argument("--asbd_dir",  default="data/asbd")
    ap.add_argument("--out_dir",   default="outputs/asbd_eval")
    ap.add_argument("--pipelines", default="a,d,e,f")
    ap.add_argument("--sources",   default="ssbd,esbd,wei_bd")
    ap.add_argument("--runs",      type=int,   default=1)
    ap.add_argument("--chunk_sec", type=int,   default=10)
    ap.add_argument("--pad_sec",   type=float, default=1.0)
    ap.add_argument("--limit",     type=int,   default=0,
                    help="If >0, take only N rows (after filtering)")
    ap.add_argument("--include_timing_issues", action="store_true",
                    help="Include the 9 rows flagged with timing_issue=true")
    ap.add_argument("--force",     action="store_true",
                    help="Re-run jobs even if behavior_intervals.json exists")
    ap.add_argument("--mode", choices=["rgb", "skeleton"], default="rgb",
                    help="Prompt mode: rgb (default) for raw video, skeleton for keypoint renders. "
                         "Automatically set to 'skeleton' when --skeleton_dir is provided.")
    ap.add_argument("--skeleton_dir", default="",
                    help="If set, swap the per-row source video to the matching skeleton .mp4 "
                         "under <skeleton_dir>/<source>/videos/<video.lower()>_skeleton.mp4. "
                         "Metadata stays the same (same start/end/label).")
    ap.add_argument("--normal_samples", type=int, default=0,
                    help="Add N 'normal' (no-behavior) samples taken from portions of ASBD "
                         "videos OUTSIDE the labeled behavior window. Each gets gt_label='' "
                         "(multi-class prompt) and is correct iff the model labels it 'none'.")
    ap.add_argument("--normal_seed", type=int, default=42,
                    help="Random seed for normal-sample selection")
    args = ap.parse_args()

    asbd_dir   = PROJECT / args.asbd_dir
    out_dir    = PROJECT / args.out_dir
    metadata   = asbd_dir / "metadata.csv"
    if not metadata.exists():
        print(f"metadata not found: {metadata}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    trim_root = out_dir / "_trimmed"

    sources = set(_normalise_source(s) for s in args.sources.split(",") if s.strip())
    pip_ids = [p.strip() for p in args.pipelines.split(",") if p.strip() in PIPELINES]

    rows = load_asbd_rows(metadata, sources, args.include_timing_issues)

    # Resolve prompt mode: skeleton_dir implies skeleton mode.
    prompt_mode = args.mode
    if args.skeleton_dir and prompt_mode == "rgb":
        prompt_mode = "skeleton"
        print("[mode] skeleton_dir provided — automatically using prompt_mode=skeleton")

    # Choose subject hint based on mode.
    subject_hint = ASBD_SUBJECT_HINT_SKELETON if prompt_mode == "skeleton" else ASBD_SUBJECT_HINT_RGB

    # Skeleton mode: swap each row's src_path to the matching skeleton .mp4.
    if args.skeleton_dir:
        skel_root = PROJECT / args.skeleton_dir
        kept, dropped = [], 0
        for r in rows:
            skel = skel_root / r["source"] / "videos" / f"{r['video'].lower()}_skeleton.mp4"
            if skel.exists():
                r["src_path"] = skel
                kept.append(r)
            else:
                dropped += 1
        print(f"[skeleton] mapped {len(kept)} rows; dropped {dropped} (no skeleton file).")
        rows = kept

    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        print(f"No usable ASBD rows found in {metadata} for sources={sources}", file=sys.stderr)
        sys.exit(1)

    # Normal/none samples: pick N rows, choose a 5s window OUTSIDE [start, end] from
    # the same source video. gt_label='' triggers the multi-class prompt; the model
    # is correct only when it labels the clip 'none'.
    if args.normal_samples > 0:
        import random
        rng = random.Random(args.normal_seed)
        normal_window = MIN_TRIM_SEC  # 5s normal clip
        eligible = []
        for r in rows:
            vd = r["video_dur"]
            after_avail  = vd - r["end_sec"]   if vd > 0 else 0
            before_avail = r["start_sec"]
            if max(after_avail, before_avail) >= normal_window + 1:
                eligible.append(r)
        rng.shuffle(eligible)
        normals = []
        for r in eligible[:args.normal_samples]:
            vd = r["video_dur"]
            # Prefer the longer side. Add 0.5s gap from the behavior boundary.
            after_avail  = vd - r["end_sec"]
            before_avail = r["start_sec"]
            if after_avail >= normal_window + 1:
                ns = r["end_sec"] + 0.5
                ne = ns + normal_window
            else:
                ne = r["start_sec"] - 0.5
                ns = ne - normal_window
            normals.append({
                "source":    r["source"],
                "video":     f"{r['video']}_normal",
                "src_path":  r["src_path"],
                "start_sec": ns,
                "end_sec":   ne,
                "video_dur": vd,
                # Binary prompt: "Label: none or unusual_behavior".
                # For a normal portion of the video, correct=none.
                "gt_label":  "unusual_behavior",
                "category":  "none",
                "is_normal": True,
                "expected":  "none",
            })
        print(f"[normal] added {len(normals)} 'none' samples (eligible pool = {len(eligible)})")
        rows = rows + normals

    # Per-class breakdown
    by_label = {b: 0 for b in ASBD_BEHAVIORS}
    for r in rows:
        # "none" rows are normal samples (gt_label is empty); count under the 'none' column
        col = "none" if r.get("is_normal") else r["gt_label"]
        by_label[col] = by_label.get(col, 0) + 1
    by_source = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1

    print(f"[{ts()}] ASBD eval start")
    print(f"  metadata    : {metadata}")
    print(f"  sources     : {sorted(sources)}  → {by_source}")
    print(f"  by-label    : {by_label}")
    print(f"  total rows  : {len(rows)}")
    print(f"  pipelines   : {pip_ids}")
    print(f"  prompt_mode : {prompt_mode}")
    print(f"  runs/clip   : {args.runs}")
    print(f"  CUDA        : {args.cuda}")
    print(f"  pad_sec     : {args.pad_sec}\n")

    # Phase 0: pre-trim every video into out_dir/_trimmed/{source}/{video}.mp4
    print(f"[{ts()}] Phase 0: pre-trimming videos with ±{args.pad_sec}s padding")
    trimmed_paths = {}
    skipped_trim = 0
    for r in rows:
        dst = trim_root / r["source"] / f"{r['video']}.mp4"
        ok = trim_video(r["src_path"], dst, r["start_sec"], r["end_sec"],
                        args.pad_sec, r["video_dur"])
        if ok:
            trimmed_paths[(r["source"], r["video"])] = dst
        else:
            skipped_trim += 1
            print(f"  [WARN] trim failed: {r['source']}/{r['video']}")
    print(f"  trimmed: {len(trimmed_paths)} / {len(rows)}  (skipped: {skipped_trim})")

    rows = [r for r in rows if (r["source"], r["video"]) in trimmed_paths]

    # Phase 1: per pipeline, build manifest of pending jobs and dispatch
    manifest_dir = out_dir / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for pid in pip_ids:
        cfg = PIPELINES[pid]
        print(f"\n{'='*70}\n[{ts()}] Pipeline {pid.upper()} ({cfg['name']})\n{'='*70}")
        pending = []
        for r in rows:
            for run_num in range(1, args.runs + 1):
                run_out = out_dir / pid / r["source"] / r["video"] / f"run_{run_num}"
                intervals = run_out / "behavior_intervals.json"
                if intervals.exists() and not args.force:
                    continue
                trimmed = trimmed_paths[(r["source"], r["video"])]
                pending.append(build_asbd_job(r, run_out, trimmed, args.chunk_sec,
                                               prompt_mode=prompt_mode,
                                               subject_hint=subject_hint))
        if pending:
            print(f"  [{ts()}] {len(pending)} jobs to run for {cfg['name']}")
            manifest_path = manifest_dir / f"{pid}_manifest.json"
            run_pipeline_batch(pid, cfg, pending, manifest_path, args.cuda)
        else:
            print(f"  [{ts()}] All {cfg['name']} jobs already have outputs. Skipping.")

    # Phase 2: collect results
    results = []
    for pid in pip_ids:
        cfg = PIPELINES[pid]
        for r in rows:
            for run_num in range(1, args.runs + 1):
                run_out = out_dir / pid / r["source"] / r["video"] / f"run_{run_num}"
                intervals = run_out / "behavior_intervals.json"
                if not intervals.exists():
                    print(f"  [WARN] missing output: {intervals}")
                    continue
                info = read_intervals(intervals)
                detected = info["label"]
                expected = r.get("expected") or r["gt_label"]
                correct  = (detected == expected)
                # Display column: 'none' for normal samples, gt_label otherwise
                gt_col   = "none" if r.get("is_normal") else r["gt_label"]
                mark     = "✓" if correct else "✗"
                print(f"  [{cfg['name']:12s}] {r['source']}/{r['video']}  run{run_num}  {mark} "
                      f"gt={gt_col}  detected={detected}")
                results.append({
                    "pipeline":    pid,
                    "name":        cfg["name"],
                    "source":      r["source"],
                    "video":       r["video"],
                    "gt":          gt_col,
                    "detected":    detected,
                    "correct":     correct,
                    "run":         run_num,
                    "description": info.get("description", ""),
                    "unusual":     info.get("unusual", ""),
                })

    raw_json = out_dir / "results.json"
    raw_json.write_text(json.dumps(results, indent=2))
    print(f"\n[{ts()}] Raw results → {raw_json}")

    # Per-source tables + overall
    for src in sorted(set(r["source"] for r in results)):
        src_results = [r for r in results if r["source"] == src]
        out_path = out_dir / f"eval_table_{src}.txt"
        print(f"\n\n{'#'*105}")
        print_table(src_results, args.runs, out_path,
                    behaviors=ASBD_BEHAVIORS,
                    title=f"ASBD EVALUATION — source={src.upper()}  —  pipelines A / D / E / F")

    print(f"\n\n{'#'*105}")
    print_table(results, args.runs, out_dir / "eval_table.txt",
                behaviors=ASBD_BEHAVIORS,
                title="ASBD EVALUATION TABLE  —  ALL SOURCES  —  pipelines A / D / E / F")
    print(f"\n[{ts()}] ASBD eval complete.")


if __name__ == "__main__":
    main()
