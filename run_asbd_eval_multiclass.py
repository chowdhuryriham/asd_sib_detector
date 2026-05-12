#!/usr/bin/env python3
"""
run_asbd_eval_multiclass.py — Independent binary classifiers methodology.

For each clip (regardless of GT label), ask 4 separate binary yes/no questions:
    Q1: "Does this clip have armflapping?"
    Q2: "Does this clip have spinning?"
    Q3: "Does this clip have headbanging?"
    Q4: "Does this clip have handaction?"

Aggregation:
    - All 4 "no" → predicted = "none"
    - 1+ "yes"  → predicted set = all "yes" behaviors (multi-label)

Generates two tables:
    1. PER-BEHAVIOR BINARY ACCURACY — independent yes/no correctness for each
       of the 4 behaviors (precision/recall over all 162 clips).
    2. PER-CLIP MULTI-CLASS ACCURACY — clip-level correctness with `none` as
       a column. Also reports multi-label cases (where >1 behavior triggered).

Usage:
    python3 run_asbd_eval_multiclass.py [--cuda 1,3] [--mode rgb|skeleton]
        [--out_dir outputs/asbd_eval_multiclass_rgb] [--pipelines a,d,e,f]
        [--normal_samples 30] [--max_new_tokens 256]
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_gt_eval import PIPELINES, FFMPEG_BIN, read_intervals, ts

from run_asbd_eval import (
    ASBD_SUBJECT_HINT_RGB,
    ASBD_SUBJECT_HINT_SKELETON,
    MIN_TRIM_SEC,
    trim_video,
    load_asbd_rows,
    _normalise_source,
)

PROJECT = Path(__file__).parent

# The 4 behaviors we query (each as an independent binary question).
QUERY_BEHAVIORS = ["armflapping", "spinning", "headbanging", "handaction"]

# Column ordering for the multi-class table.
REPORT_LABELS = ["armflapping", "spinning", "headbanging", "handaction", "none"]


def build_multiclass_job(trimmed_video: Path, run_out: Path,
                         behavior_queried: str, gt_label: str,
                         chunk_sec: int, prompt_mode: str,
                         subject_hint: str) -> dict:
    """One manifest job: ask the model 'is this {behavior_queried}?'
    gt_label here is set to behavior_queried to trigger the binary prompt
    'none or {behavior_queried}'. The true GT is preserved separately in
    `true_gt` so the aggregation step knows what's correct."""
    run_out.mkdir(parents=True, exist_ok=True)
    stem = trimmed_video.stem
    return {
        "video":            str(trimmed_video),
        "output":           str(run_out / f"{stem}_annotated.mp4"),
        "log":              str(run_out / "log.json"),
        "intervals":        str(run_out / "behavior_intervals.json"),
        "summary":          str(run_out / "summary.txt"),
        "chunks_dir":       str(run_out / "chunks"),
        "debug_dir":        str(run_out / "chunk_debug_frames"),
        "gt_label":         behavior_queried,  # forces binary "none or {behavior_queried}"
        "chunk_sec":        chunk_sec,
        "overlap_sec":      0,
        "prompt_mode":      prompt_mode,
        "subject_hint":     subject_hint,
        # Multi-class metadata (not consumed by pipelines, used at aggregation)
        "_true_gt":         gt_label,
        "_behavior_queried": behavior_queried,
    }


def run_pipeline_batch_multiclass(pip_id: str, cfg: dict, jobs: list,
                                   manifest_path: Path, cuda: str,
                                   max_new_tokens: int) -> None:
    """Like run_pipeline_batch but sets MAX_NEW_TOKENS env var."""
    manifest_path.write_text(json.dumps(jobs, indent=2))
    cmd = [cfg["venv"], str(PROJECT / cfg["script"]),
           "--manifest", str(manifest_path)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda
    env["MAX_NEW_TOKENS"] = str(max_new_tokens)
    print(f"  [{ts()}] Spawning {cfg['name']} batch ({len(jobs)} jobs, "
          f"MAX_NEW_TOKENS={max_new_tokens})")
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT), capture_output=False)
    print(f"  [{ts()}] {cfg['name']} batch exit={result.returncode}")


def _print_binary_table(per_clip_records: list, pip_ids: list,
                         out_path: Path) -> None:
    """Per-behavior binary accuracy table.
    For each (pipeline, behavior) pair: how many of 162 clips got the binary
    yes/no question right.

    "Correct" for clip Q on behavior B:
      - If true_gt == B (or unusual_behavior implies a stereotypy): yes → correct
      - If true_gt != B and != normal: no (none) → correct
      - If true_gt == none (normal sample): no (none) → correct
    """
    lines = []
    title = "PER-BEHAVIOR BINARY ACCURACY  —  each clip asked 4 independent yes/no questions"
    lines.append("=" * 110)
    lines.append(title)
    lines.append("=" * 110)
    header = f"{'Pipeline':<14}" + "".join(f"  {b:>16}" for b in QUERY_BEHAVIORS) + f"  {'OVERALL':>16}"
    lines.append(header)
    lines.append("-" * 110)

    for pid in pip_ids:
        name = PIPELINES[pid]["name"]
        row = f"{name:<14}"
        all_c, all_t = 0, 0
        for b in QUERY_BEHAVIORS:
            c = t = 0
            for r in per_clip_records:
                if r["pipeline"] != pid:
                    continue
                ans = r["binary_answers"].get(b)
                if ans is None:
                    continue
                # "yes" means model said the behavior name; "no" means model said "none"
                model_yes = (ans == b)
                gt_yes = (r["true_gt"] == b)
                correct = (model_yes == gt_yes)
                t += 1
                if correct:
                    c += 1
            cell = f"{c}/{t} ({100*c/t:.0f}%)" if t else "—"
            row += f"  {cell:>16}"
            all_c += c; all_t += t
        ov = f"{all_c}/{all_t} ({100*all_c/all_t:.0f}%)" if all_t else "—"
        row += f"  {ov:>16}"
        lines.append(row)
    lines.append("=" * 110)

    text = "\n".join(lines)
    print(text)
    out_path.write_text(text + "\n")


def _print_multiclass_table(per_clip_records: list, pip_ids: list,
                             out_path: Path) -> None:
    """Per-clip multi-class table (clip-level correctness, includes `none`).

    For each (pipeline, true_gt) pair: how many clips with that GT were
    predicted correctly. Multi-label predictions (>1 yes) are reported in a
    separate MULTI column.
    """
    lines = []
    title = "PER-CLIP MULTI-CLASS ACCURACY  —  aggregated from 4 binary answers per clip"
    lines.append("=" * 130)
    lines.append(title)
    lines.append("Predicted = {behaviors where binary answer is 'yes'} | none = all 4 'no'")
    lines.append("Correct iff predicted set == {true_gt}, or both empty (for normal clips).")
    lines.append("=" * 130)

    header = f"{'Pipeline':<14}" + "".join(f"  {b:>14}" for b in REPORT_LABELS)
    header += f"  {'MULTI':>10}  {'OVERALL':>14}"
    lines.append(header)
    lines.append("-" * 130)

    for pid in pip_ids:
        name = PIPELINES[pid]["name"]
        # Per-GT bucket
        buckets = {b: {"c": 0, "t": 0} for b in REPORT_LABELS}
        multi_count = 0
        for r in per_clip_records:
            if r["pipeline"] != pid:
                continue
            true_gt = r["true_gt"]
            col = "none" if true_gt in ("none", "unusual_behavior") else true_gt
            if col not in buckets:
                continue
            buckets[col]["t"] += 1
            pred_set = r["predicted_set"]
            if true_gt in ("none", "unusual_behavior"):
                # Correct iff model said no to all 4 (empty predicted set)
                if len(pred_set) == 0:
                    buckets[col]["c"] += 1
            else:
                # Correct iff predicted set == {true_gt}
                if pred_set == {true_gt}:
                    buckets[col]["c"] += 1
            if len(pred_set) >= 2:
                multi_count += 1

        row = f"{name:<14}"
        all_c, all_t = 0, 0
        for b in REPORT_LABELS:
            c = buckets[b]["c"]
            t = buckets[b]["t"]
            all_c += c; all_t += t
            cell = f"{c}/{t} ({100*c/t:.0f}%)" if t else "—"
            row += f"  {cell:>14}"
        ov = f"{all_c}/{all_t} ({100*all_c/all_t:.0f}%)" if all_t else "—"
        row += f"  {multi_count:>10}  {ov:>14}"
        lines.append(row)

    lines.append("=" * 130)
    lines.append("MULTI column = number of clips where the model said 'yes' to 2+ behaviors "
                 "(reported separately, not counted as correct unless GT also had multi-label).")
    lines.append("=" * 130)

    text = "\n".join(lines)
    print(text)
    # Append to the same file (after binary table) for one combined view
    with out_path.open("a") as fp:
        fp.write("\n\n" + text + "\n")


def _print_multilabel_details(per_clip_records: list, pip_ids: list,
                                out_path: Path) -> None:
    """List each clip with multi-label prediction so we can see what's going on."""
    lines = ["\n\n" + "=" * 130,
             "MULTI-LABEL CASES (model said 'yes' to 2+ behaviors)",
             "=" * 130,
             f"{'Pipeline':<14}  {'Source':<8}  {'Video':<38}  {'GT':<18}  {'Predicted set'}"]
    lines.append("-" * 130)
    for pid in pip_ids:
        name = PIPELINES[pid]["name"]
        for r in per_clip_records:
            if r["pipeline"] != pid:
                continue
            if len(r["predicted_set"]) >= 2:
                preds = "+".join(sorted(r["predicted_set"]))
                gt = r["true_gt"]
                lines.append(f"{name:<14}  {r['source']:<8}  {r['video'][:38]:<38}  {gt:<18}  {preds}")
    text = "\n".join(lines)
    with out_path.open("a") as fp:
        fp.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda",      default="1,3")
    ap.add_argument("--asbd_dir",  default="data/asbd")
    ap.add_argument("--out_dir",   default="outputs/asbd_eval_multiclass_rgb")
    ap.add_argument("--pipelines", default="a,d,e,f")
    ap.add_argument("--sources",   default="ssbd,esbd,wei_bd")
    ap.add_argument("--runs",      type=int, default=1)
    ap.add_argument("--chunk_sec", type=int, default=10)
    ap.add_argument("--pad_sec",   type=float, default=1.0)
    ap.add_argument("--limit",     type=int, default=0)
    ap.add_argument("--include_timing_issues", action="store_true")
    ap.add_argument("--force",     action="store_true",
                    help="Re-run jobs even if behavior_intervals.json exists")
    ap.add_argument("--mode", choices=["rgb", "skeleton"], default="rgb")
    ap.add_argument("--skeleton_dir", default="")
    ap.add_argument("--normal_samples", type=int, default=30)
    ap.add_argument("--normal_seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=8192,
                    help="MAX_NEW_TOKENS env var for the pipeline subprocess. "
                         "Default 8192 — generous headroom for the binary CoT response.")
    args = ap.parse_args()

    asbd_dir = PROJECT / args.asbd_dir
    out_dir  = PROJECT / args.out_dir
    metadata = asbd_dir / "metadata.csv"
    if not metadata.exists():
        print(f"metadata not found: {metadata}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    trim_root = out_dir / "_trimmed"

    sources = set(_normalise_source(s) for s in args.sources.split(",") if s.strip())
    pip_ids = [p.strip() for p in args.pipelines.split(",") if p.strip() in PIPELINES]
    rows = load_asbd_rows(metadata, sources, args.include_timing_issues)

    prompt_mode = args.mode
    if args.skeleton_dir and prompt_mode == "rgb":
        prompt_mode = "skeleton"

    subject_hint = ASBD_SUBJECT_HINT_SKELETON if prompt_mode == "skeleton" else ASBD_SUBJECT_HINT_RGB

    if args.skeleton_dir:
        skel_root = PROJECT / args.skeleton_dir
        kept = []
        for r in rows:
            skel = skel_root / r["source"] / "videos" / f"{r['video'].lower()}_skeleton.mp4"
            if skel.exists():
                r["src_path"] = skel
                kept.append(r)
        rows = kept

    if args.limit > 0:
        rows = rows[:args.limit]

    # Add normal samples
    if args.normal_samples > 0:
        import random
        rng = random.Random(args.normal_seed)
        nw = MIN_TRIM_SEC
        eligible = [r for r in rows if max(
            (r["video_dur"] - r["end_sec"]) if r["video_dur"] > 0 else 0,
            r["start_sec"]) >= nw + 1]
        rng.shuffle(eligible)
        normals = []
        for r in eligible[:args.normal_samples]:
            vd = r["video_dur"]
            after  = vd - r["end_sec"]
            before = r["start_sec"]
            if after >= nw + 1:
                ns = r["end_sec"] + 0.5; ne = ns + nw
            else:
                ne = r["start_sec"] - 0.5; ns = ne - nw
            normals.append({
                "source":   r["source"],
                "video":    f"{r['video']}_normal",
                "src_path": r["src_path"],
                "start_sec": ns, "end_sec": ne,
                "video_dur": vd,
                "gt_label": "none",
                "category": "none",
                "is_normal": True,
            })
        rows = rows + normals
        print(f"[normal] added {len(normals)} 'none' samples")

    print(f"\n[{ts()}] Multiclass eval start")
    print(f"  prompt_mode    : {prompt_mode}")
    print(f"  pipelines      : {pip_ids}")
    print(f"  total clips    : {len(rows)}")
    print(f"  queries/clip   : {len(QUERY_BEHAVIORS)}")
    print(f"  total inference: {len(rows) * len(QUERY_BEHAVIORS) * len(pip_ids)} (per pipeline run)")
    print(f"  CUDA           : {args.cuda}")
    print(f"  max_new_tokens : {args.max_new_tokens}\n")

    # Phase 0: trim videos
    print(f"[{ts()}] Phase 0: pre-trimming")
    trimmed_paths = {}
    for r in rows:
        dst = trim_root / r["source"] / f"{r['video']}.mp4"
        ok = trim_video(r["src_path"], dst, r["start_sec"], r["end_sec"],
                        args.pad_sec, r["video_dur"])
        if ok:
            trimmed_paths[(r["source"], r["video"])] = dst
    rows = [r for r in rows if (r["source"], r["video"]) in trimmed_paths]
    print(f"  trimmed: {len(rows)}")

    # Phase 1: per pipeline, build manifest with 4 entries per clip
    manifest_dir = out_dir / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for pid in pip_ids:
        cfg = PIPELINES[pid]
        print(f"\n{'='*70}\n[{ts()}] Pipeline {pid.upper()} ({cfg['name']})\n{'='*70}")
        pending = []
        for r in rows:
            for behavior in QUERY_BEHAVIORS:
                for run_num in range(1, args.runs + 1):
                    run_out = (out_dir / pid / "per_behavior" / behavior
                               / r["source"] / r["video"] / f"run_{run_num}")
                    intervals = run_out / "behavior_intervals.json"
                    if intervals.exists() and not args.force:
                        continue
                    pending.append(build_multiclass_job(
                        trimmed_paths[(r["source"], r["video"])],
                        run_out, behavior, r["gt_label"],
                        args.chunk_sec, prompt_mode, subject_hint))
        if pending:
            print(f"  [{ts()}] {len(pending)} jobs (= {len(rows)} clips × {len(QUERY_BEHAVIORS)} behaviors)")
            mp = manifest_dir / f"{pid}_manifest.json"
            run_pipeline_batch_multiclass(pid, cfg, pending, mp, args.cuda,
                                           args.max_new_tokens)
        else:
            print(f"  [{ts()}] All {cfg['name']} jobs already have outputs. Skipping.")

    # Phase 2: aggregate per-clip
    print(f"\n[{ts()}] Phase 2: aggregating per-clip results")
    per_clip = []
    for pid in pip_ids:
        for r in rows:
            for run_num in range(1, args.runs + 1):
                answers = {}
                for behavior in QUERY_BEHAVIORS:
                    run_out = (out_dir / pid / "per_behavior" / behavior
                               / r["source"] / r["video"] / f"run_{run_num}")
                    intervals = run_out / "behavior_intervals.json"
                    if intervals.exists():
                        info = read_intervals(intervals)
                        answers[behavior] = info["label"]  # "none" or behavior name
                    else:
                        answers[behavior] = None
                predicted_set = {b for b in QUERY_BEHAVIORS
                                 if answers.get(b) == b}
                per_clip.append({
                    "pipeline":        pid,
                    "source":          r["source"],
                    "video":           r["video"],
                    "true_gt":         r["gt_label"],
                    "binary_answers":  answers,
                    "predicted_set":   predicted_set,
                    "run":             run_num,
                })

    # Save raw aggregation
    raw = {
        "per_clip": [{
            **{k: v for k, v in r.items() if k != "predicted_set"},
            "predicted_set": sorted(r["predicted_set"]),
        } for r in per_clip]
    }
    (out_dir / "results_multiclass.json").write_text(json.dumps(raw, indent=2))

    # Phase 3: print tables
    print(f"\n{'#'*110}")
    table_path = out_dir / "eval_table_multiclass.txt"
    # Wipe before writing
    table_path.write_text("")
    _print_binary_table(per_clip, pip_ids, table_path)
    _print_multiclass_table(per_clip, pip_ids, table_path)
    _print_multilabel_details(per_clip, pip_ids, table_path)

    print(f"\n[{ts()}] Multiclass eval complete.")
    print(f"  Table → {table_path}")
    print(f"  Raw   → {out_dir / 'results_multiclass.json'}")


if __name__ == "__main__":
    main()
