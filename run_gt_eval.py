#!/usr/bin/env python3
"""
run_gt_eval.py — run pipelines A, D, E, F on all GT chunks, N runs each,
                 and generate a results table.

Usage:
    python3 run_gt_eval.py [--cuda 1,2] [--gt_dir data/gt_chunks]
                           [--out_dir outputs/gt_eval] [--runs 5]
                           [--pipelines a,d,e,f]

Table format (per behavior, correct/total across all runs):

              | hand_biting | head_hit | hitting_others | scratching | self_directed_hit
  Gemma  run1 |    1/1      |   1/1    |      3/8       |    2/2     |       2/2
  Gemma  run2 |    1/1      |   1/1    |      3/8       |    2/2     |       2/2
  ...
  Gemma  ALL  |    5/5      |   5/5    |     15/40      |   10/10    |      10/10
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).parent

PIPELINES = {
    "a": {
        "name":   "Gemma",
        "venv":   "/raid/aio469/venvs/gemma/bin/python",
        "script": "pipelines/pipeline_a_gemma.py",
    },
    "d": {
        "name":   "Phi4",
        "venv":   "/raid/aio469/venvs/phi4/bin/python",
        "script": "pipelines/pipeline_d_phi4.py",
    },
    "e": {
        "name":   "LLaVAction",
        "venv":   "/raid/aio469/venvs/llavaction/bin/python",
        "script": "pipelines/pipeline_e_llavaction.py",
    },
    "f": {
        "name":   "PLM",
        "venv":   "/raid/aio469/venvs/plm/bin/python",
        "script": "pipelines/pipeline_f_plm.py",
    },
}

BEHAVIORS = ["hand_biting", "head_hit", "hitting_others", "scratching", "self_directed_hit"]

LABEL_ALIASES = {
    "head_banging": "head_hit", "headbanging": "head_hit",
    "head_bang":    "head_hit", "head_hitting": "head_hit",
    "hitting_head": "head_hit", "banging_head": "head_hit",
}

LAB_SUBJECT_HINT = (
    "You are watching a short video clip from a clinical observation session. "
    "Focus on the child — a young child wearing a hoodie. "
    "Hoodie colour is yellow, golden-yellow, or brownish-yellow depending on the lighting. "
    "When filmed from the front, the hoodie zip may be open, revealing a T-shirt underneath — "
    "this is still the same child. Track only this child's behaviour throughout the clip."
)


def ts():
    return datetime.now().strftime("%H:%M:%S")


def gt_label_from_stem(stem: str) -> str:
    """Extract GT label from filename like 'hitting_others_03_1264.6-1266' or '..._skeleton'."""
    for suffix in ("_skeleton", "_annotated"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    m = re.match(r"^([a-z_]+?)(?:_\d+)?(?:_[\d\.\-]+)?$", stem)
    raw = m.group(1) if m else stem
    return LABEL_ALIASES.get(raw, raw)


FFMPEG_BIN = "/raid/aio469/venvs/plm/lib/python3.10/site-packages/static_ffmpeg/bin/linux/ffmpeg"

def read_intervals(intervals_path: Path) -> dict:
    """Return {label, description, unusual} from behavior_intervals.json."""
    empty = {"label": "none", "description": "", "unusual": ""}
    if not intervals_path.exists():
        return empty
    try:
        data = json.loads(intervals_path.read_text())
        # label from intervals list
        ivs = data.get("intervals", data) if isinstance(data, dict) else data
        non_none = [iv["label"] for iv in ivs if iv.get("label", "none") != "none"]
        label = non_none[0] if non_none else "none"
        # description/unusual from chunk log (log.json sibling)
        log_path = intervals_path.parent / "log.json"
        description = unusual = ""
        if log_path.exists():
            logs = json.loads(log_path.read_text())
            for entry in logs:
                if entry.get("description"):
                    description = entry["description"]; break
            for entry in logs:
                if entry.get("unusual"):
                    unusual = entry["unusual"]; break
        return {"label": label, "description": description, "unusual": unusual}
    except Exception:
        return empty

def read_detected_label(intervals_path: Path) -> str:
    return read_intervals(intervals_path)["label"]


def build_job(video: Path, run_out: Path, gt: str, chunk_sec: int) -> dict:
    """Build one job dict for the manifest. Side-effect: trims hand_biting to 4s."""
    run_out.mkdir(parents=True, exist_ok=True)
    stem = video.stem

    actual_video = video
    if gt == "hand_biting":
        trimmed = run_out / f"{stem}_4s.mp4"
        subprocess.run([FFMPEG_BIN, "-y", "-i", str(video), "-t", "4",
                        "-c", "copy", str(trimmed), "-loglevel", "error"],
                       capture_output=True)
        if trimmed.exists() and trimmed.stat().st_size > 1000:
            actual_video = trimmed

    return {
        "video":        str(actual_video),
        "output":       str(run_out / f"{stem}_annotated.mp4"),
        "log":          str(run_out / "log.json"),
        "intervals":    str(run_out / "behavior_intervals.json"),
        "summary":      str(run_out / "summary.txt"),
        "chunks_dir":   str(run_out / "chunks"),
        "debug_dir":    str(run_out / "chunk_debug_frames"),
        "gt_label":     gt,
        "chunk_sec":    chunk_sec,
        "overlap_sec":  0,
        "subject_hint": LAB_SUBJECT_HINT,
    }


def run_pipeline_batch(pip_id: str, cfg: dict, jobs: list, manifest_path: Path,
                       cuda: str) -> None:
    """Run one pipeline on multiple videos via manifest mode.
    Model loads once, all jobs processed in same Python process."""
    manifest_path.write_text(json.dumps(jobs, indent=2))
    cmd = [
        cfg["venv"], str(PROJECT / cfg["script"]),
        "--manifest", str(manifest_path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda

    print(f"  [{ts()}] Spawning {cfg['name']} batch ({len(jobs)} jobs, model loads once)")
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT), capture_output=False)
    print(f"  [{ts()}] {cfg['name']} batch exit={result.returncode}")


def print_table(results: list, runs: int, out_path: Path,
                behaviors: list = None, title: str = "GT EVALUATION TABLE  —  pipelines A / D / E / F"):
    """
    Rows = (pipeline, run_number) then an ALL summary row per pipeline.
    Cols = per-behavior correct/total.
    """
    if behaviors is None:
        behaviors = BEHAVIORS
    lines = []
    lines.append("=" * 105)
    lines.append(title)
    lines.append(f"Runs per chunk: {runs}   |   GT chunks: {len(set(r['video'] for r in results))}")
    lines.append("=" * 105)

    beh_w  = 19
    name_w = 18
    header = f"{'Model / Run':<{name_w}}" + "".join(f"{b:<{beh_w}}" for b in behaviors) + f"{'TOTAL':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for pid in ["a", "d", "e", "f"]:
        pip_results = [r for r in results if r["pipeline"] == pid]
        if not pip_results:
            continue
        name = PIPELINES[pid]["name"]

        for run_num in range(1, runs + 1):
            run_results = [r for r in pip_results if r["run"] == run_num]
            per_beh = []
            for b in behaviors:
                brows = [r for r in run_results if r["gt"] == b]
                per_beh.append(f"{sum(r['correct'] for r in brows)}/{len(brows)}")
            tp_tot = sum(r["correct"] for r in run_results)
            line = (f"{name+' run'+str(run_num):<{name_w}}"
                    + "".join(f"{v:<{beh_w}}" for v in per_beh)
                    + f"{tp_tot}/{len(run_results):>7}")
            lines.append(line)

        # ALL-runs summary
        per_beh_all, tp_all, total_all = [], 0, 0
        for b in behaviors:
            brows = [r for r in pip_results if r["gt"] == b]
            tp = sum(r["correct"] for r in brows)
            per_beh_all.append(f"{tp}/{len(brows)}")
            tp_all += tp; total_all += len(brows)
        lines.append(
            f"{name+' ALL':<{name_w}}"
            + "".join(f"{v:<{beh_w}}" for v in per_beh_all)
            + f"{tp_all}/{total_all:>7}"
        )
        lines.append("")

    lines.append("=" * len(header))

    # Detailed log
    lines.append("")
    lines.append("DETAILED RESULTS:")
    lines.append(f"  {'Video':<45} {'GT':<22} {'Detected':<22} {'Pipeline':<13} {'Run':>4}  {'OK'}")
    lines.append(f"  {'-'*45} {'-'*22} {'-'*22} {'-'*13} {'-'*4}  {'-'*3}")
    for r in sorted(results, key=lambda x: (x["pipeline"], x["run"], x["gt"], x["video"])):
        ok = "✓" if r["correct"] else "✗"
        lines.append(
            f"  {r['video'][:44]:<45} {r['gt']:<22} {r['detected']:<22}"
            f" {r['name']:<13} {r['run']:>4}  {ok}"
        )

    # Model descriptions
    lines.append("")
    lines.append("=" * 105)
    lines.append("MODEL DESCRIPTIONS (what each model said was happening in the clip)")
    lines.append("=" * 105)
    for r in sorted(results, key=lambda x: (x["pipeline"], x["run"], x["gt"], x["video"])):
        if r.get("description") or r.get("unusual"):
            lines.append(
                f"  [{r['name']:<12} run{r['run']}]  {r['video'][:40]}"
            )
            if r.get("description"):
                lines.append(f"    Description : {r['description']}")
            if r.get("unusual"):
                lines.append(f"    Unusual     : {r['unusual']}")

    text = "\n".join(lines)
    print(text)
    out_path.write_text(text)
    print(f"\nTable saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda",      default="1,2,3,4",
                    help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--gt_dir",    default="data/gt_chunks")
    ap.add_argument("--out_dir",   default="outputs/gt_eval_v2")
    ap.add_argument("--runs",      type=int, default=5)
    ap.add_argument("--pipelines", default="a,d,e,f")
    ap.add_argument("--chunk_sec", type=int, default=10,
                    help="chunk_sec passed to each pipeline (use 10 so 5-sec GT clips = 1 chunk)")
    ap.add_argument("--force",     action="store_true",
                    help="Re-run jobs even if behavior_intervals.json exists")
    args = ap.parse_args()

    gt_dir  = PROJECT / args.gt_dir
    out_dir = PROJECT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(gt_dir.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {gt_dir}", file=sys.stderr)
        sys.exit(1)

    pip_ids = [p.strip() for p in args.pipelines.split(",") if p.strip() in PIPELINES]

    print(f"[{ts()}] GT eval start")
    print(f"  GT clips   : {len(videos)} in {gt_dir}")
    print(f"  Pipelines  : {pip_ids}")
    print(f"  Runs/clip  : {args.runs}")
    print(f"  CUDA       : {args.cuda}")
    print(f"  chunk_sec  : {args.chunk_sec}  overlap_sec=0\n")

    manifest_dir = out_dir / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: for each pipeline, build manifest of jobs that need to run
    # (skip jobs whose behavior_intervals.json already exists), then run pipeline once
    for pid in pip_ids:
        cfg = PIPELINES[pid]
        print(f"\n{'='*70}")
        print(f"[{ts()}] Pipeline {pid.upper()} ({cfg['name']})")
        print(f"{'='*70}")

        pending_jobs = []   # (video.stem, run_num, gt, run_out, job_dict)
        for video in videos:
            gt = gt_label_from_stem(video.stem)
            for run_num in range(1, args.runs + 1):
                run_out = out_dir / pid / video.stem / f"run_{run_num}"
                intervals = run_out / "behavior_intervals.json"
                if intervals.exists() and not args.force:
                    continue
                job = build_job(video, run_out, gt, args.chunk_sec)
                pending_jobs.append(job)

        if pending_jobs:
            print(f"  [{ts()}] {len(pending_jobs)} jobs to run for {cfg['name']}")
            manifest_path = manifest_dir / f"{pid}_manifest.json"
            run_pipeline_batch(pid, cfg, pending_jobs, manifest_path, args.cuda)
        else:
            print(f"  [{ts()}] All {cfg['name']} jobs already have outputs. Skipping.")

    # Phase 2: collect all results from disk (both fresh + previously skipped)
    results = []
    for pid in pip_ids:
        cfg = PIPELINES[pid]
        for video in videos:
            gt = gt_label_from_stem(video.stem)
            for run_num in range(1, args.runs + 1):
                run_out = out_dir / pid / video.stem / f"run_{run_num}"
                intervals = run_out / "behavior_intervals.json"
                if not intervals.exists():
                    print(f"  [WARN] missing output: {intervals}")
                    continue
                info = read_intervals(intervals)
                detected = info["label"]
                correct  = (detected == gt)
                mark     = "✓" if correct else "✗"
                print(f"  [{cfg['name']:12s}] {video.stem}  run{run_num}  {mark} gt={gt}  detected={detected}")
                results.append({
                    "pipeline":    pid,
                    "name":        cfg["name"],
                    "video":       video.stem,
                    "gt":          gt,
                    "detected":    detected,
                    "correct":     correct,
                    "run":         run_num,
                    "description": info.get("description", ""),
                    "unusual":     info.get("unusual", ""),
                })

    raw_json = out_dir / "results.json"
    raw_json.write_text(json.dumps(results, indent=2))
    print(f"\n[{ts()}] Raw results → {raw_json}")

    print(f"\n\n{'='*105}")
    print_table(results, args.runs, out_dir / "eval_table.txt")
    print(f"\n[{ts()}] GT eval complete.")


if __name__ == "__main__":
    main()
