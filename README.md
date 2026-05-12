# ASD Stereotyped & Self-Injurious Behaviour Detector

Automated detection of self-injurious behaviours (SIB) and ASD stereotypies in
video using 4 video-language models. Supports two datasets and three evaluation
methodologies.

## Models

| Pipeline | Model | Vendor | Size | Path |
|---|---|---|---|---|
| A | [Gemma-4-31B](https://huggingface.co/google/gemma-4-31b-it) | Google DeepMind | 31B | `models/gemma-4-31b` |
| D | [Phi-4-multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | Microsoft | 5.6B | `models/Phi-4-multimodal` |
| E | [LLaVAction-7B](https://huggingface.co/MLAdaptiveIntelligence/LLaVAction-7B) | EPFL | 7B | `models/LLaVAction-7B` |
| F | [Perception-LM-8B](https://huggingface.co/facebook/Perception-LM-8B) | Meta FAIR | 8B | `models/Perception-LM-8B` |

## Datasets

### `data/gt_chunks/` — Clinical lab dataset (SIB)
Pre-trimmed clinical observation clips. **Labels:**
`hand_biting`, `head_hit`, `hitting_others`, `scratching`, `self_directed_hit`, `none`

### `data/asbd/` — ASBD (in-the-wild ASD stereotypies)
SSBD (75) + ESBD (117) + Wei_BD (9) = 201 metadata rows, 140 downloaded.
YouTube clips with `behavior_start_sec` / `behavior_end_sec` annotation —
trimmed per-row at eval time with 1s padding. **Labels:**
`armflapping`, `spinning`, `headbanging`, `handaction`, `none`

### `data/asbd_skeleton_output/` — Skeleton renders of ASBD
RTMPose Halpe-26 keypoint renders on a black background. Used when running
the skeleton-mode prompt (geometric joint-rule definitions instead of
appearance descriptions).

## How the codebase works

The codebase is organised in two layers — **orchestrators** and **pipelines**.
The split exists so that each model loads exactly once per eval (model weights
are ~30 GB for Gemma) and so that all 4 models can share the same workflow.

### Layer 1 — Orchestrators (top-level scripts)

`run_gt_eval.py`, `run_asbd_eval.py`, and `run_asbd_eval_multiclass.py`
are the entry points. They do **no inference themselves**. Each orchestrator
runs the same 5-phase pipeline:

```
Phase 0  Load dataset rows  →  filter to usable rows (download_status,
                                 timing_issue, valid start/end times)
Phase 1  Pre-trim videos    →  use ffmpeg to cut each row's behavior window
                                 with ±pad_sec padding. Cached in
                                 {out_dir}/_trimmed/{source}/{video}.mp4
Phase 2  Build manifest     →  one JSON job per (clip × pipeline). The job
                                 dict carries every parameter the pipeline
                                 subprocess needs (paths, gt_label, prompt
                                 mode, subject hint, chunk_sec).
Phase 3  Dispatch pipelines →  spawn one subprocess per pipeline with
                                 CUDA_VISIBLE_DEVICES set. The pipeline
                                 loads its model once and processes every
                                 job in the manifest in series.
Phase 4  Aggregate results  →  read each clip's behavior_intervals.json,
                                 compare to GT, write eval_table.txt and
                                 results.json.
```

### Layer 2 — Pipelines (`pipelines/pipeline_*.py`)

Each pipeline is a standalone script that can be run on a single video or
via manifest. When invoked with `--manifest path.json`, the pipeline:

1. Loads the model **once** (the expensive step)
2. Iterates each manifest entry → trims video into 10s chunks → builds the
   prompt (RGB or skeleton mode, single-behavior or multi-behavior) → runs
   `model.generate()` → parses the response → writes
   `behavior_intervals.json` + `log.json` to the per-clip output dir.

The pipelines share an identical `make_prompt(...)` shape but each loads its
model with model-specific code (HF `AutoModelForCausalLM` vs `LLaVA` vs
`AutoModelForImageTextToText`, etc.).

### How the prompts are constructed

`make_prompt(chunk_dur, gt_label, subject_hint)` (in every pipeline):

- Looks up `gt_label` in `BEHAVIOR_DEFINITIONS_RGB` or
  `BEHAVIOR_DEFINITIONS_SKELETON` (based on `PROMPT_MODE`).
- If `gt_label` is set and known → builds a **binary** prompt:
  `Label: none or {gt_label}` with just that one definition.
- If `gt_label` is empty → falls back to a **multi-class** prompt listing
  every behaviour.
- Adds a Chain-of-Thought (CoT) layer: STEP 1 describe → STEP 2 classify.
- Appends a confidence hedge and a "many clips contain no stereotyped
  behavior" reminder to fight false positives.

The multiclass orchestrator exploits this by overriding `gt_label` per query
(once per behaviour) so the same prompt machinery yields 4 independent
binary asks per clip.

### Output aggregation

`read_intervals(intervals_path)` (in `run_gt_eval.py`) is the single point
of truth for parsing a pipeline's output. It reads the `intervals` list
from `behavior_intervals.json` and returns the first non-`none` label found
(or `"none"` if all are `none`). Both single-question and multiclass evals
use this same parser.

## Evaluation orchestrators

### 1. `run_gt_eval.py` — Lab SIB eval
Runs the 4 pipelines on `data/gt_chunks/` and produces per-pipeline accuracy
tables for the 5 lab behaviours.

```bash
python3 run_gt_eval.py --pipelines a,d,e,f --cuda 4,5 \
    --out_dir outputs/gt_eval_v4 --chunk_sec 10 --runs 5
```

### 2. `run_asbd_eval.py` — Single-question ASBD eval
For each clip, the prompt is a **binary** question seeded by the GT label:
`"none or {gt_label}"`. Fast (one inference per clip), but the prompt leaks
the answer category to the model.

```bash
# RGB mode (raw video)
python3 run_asbd_eval.py --mode rgb --cuda 5,6 \
    --out_dir outputs/asbd_eval_rgb --normal_samples 30 \
    --pipelines a,d,e,f

# Skeleton mode (keypoint renders)
python3 run_asbd_eval.py --mode skeleton \
    --skeleton_dir data/asbd_skeleton_output/asbd_skeleton_output \
    --cuda 5,6 \
    --out_dir outputs/asbd_eval_skeleton --normal_samples 30 \
    --pipelines a,d,e,f
```

### 3. `run_asbd_eval_multiclass.py` — Independent binary classifiers
**Recommended methodology** for honest accuracy reporting. For every clip,
the orchestrator asks 4 **independent** binary questions — one per behaviour
(`armflapping`, `spinning`, `headbanging`, `handaction`). The prompt does not
leak the GT; each question is answered in isolation.

Aggregation per clip:
- All 4 answers = `"none"` → predicted **`none`**
- 1 answer = behaviour → predicted that behaviour
- 2+ answers = behaviours → reported as multi-label (`A+B`)

```bash
# RGB
python3 run_asbd_eval_multiclass.py --mode rgb --cuda 1,3 \
    --out_dir outputs/asbd_eval_multiclass_rgb --normal_samples 30 \
    --pipelines a,d,e,f

# Skeleton
python3 run_asbd_eval_multiclass.py --mode skeleton \
    --skeleton_dir data/asbd_skeleton_output/asbd_skeleton_output \
    --cuda 4,5 \
    --out_dir outputs/asbd_eval_multiclass_skeleton --normal_samples 30 \
    --pipelines a,d,e,f
```

The multiclass eval produces three tables in one file
(`{out_dir}/eval_table_multiclass.txt`):

1. **Per-behavior binary accuracy** — 4 independent yes/no scores per pipeline.
2. **Per-clip multi-class accuracy** — columns: `armflapping | spinning | headbanging | handaction | none | MULTI | OVERALL`. Correctness for a behavior clip requires the predicted set to equal `{true_gt}`; for a normal clip the predicted set must be empty.
3. **Multi-label details** — every clip where the model answered "yes" to 2+ behaviours, listing the predicted set.

Cost: 4× the inference of `run_asbd_eval.py`.

## Step-by-step: running an eval

This walkthrough assumes a clean checkout and that the model weights are
already downloaded to `models/`.

### Step 1. Sanity check the venv
```bash
# Confirm Python sees the right Torch + Transformers
venvs/gemma/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

### Step 2. Decide which eval you need

| If you want… | Use… |
|---|---|
| Lab SIB results on `data/gt_chunks/` | `run_gt_eval.py` |
| ASBD with the prompt told which behaviour to look for | `run_asbd_eval.py` |
| Honest ASBD multi-class (recommended) | `run_asbd_eval_multiclass.py` |

### Step 3. Pick GPU(s) and output dir

GPU assignments are passed through `--cuda`. Multiple GPUs are space-shared
within one eval (model split across them). Different evals on disjoint GPUs
run in parallel without interference.

```bash
# Example: claim GPUs 1 and 3
--cuda 1,3 --out_dir outputs/asbd_eval_multiclass_rgb
```

### Step 4. Launch the eval (in background) and watch the log
```bash
python3 run_asbd_eval_multiclass.py --mode rgb --cuda 1,3 \
    --out_dir outputs/asbd_eval_multiclass_rgb \
    --normal_samples 30 --pipelines a,d,e,f \
    > logs/asbd_eval_multiclass_rgb.log 2>&1 &
echo "PID: $!"

tail -f logs/asbd_eval_multiclass_rgb.log
```

What to expect during a run:

1. **Phase 0** prints the row count after filtering + ASBD subject hint resolution.
2. **Phase 1** trims videos. First run takes a few minutes; subsequent runs
   reuse cached trims (idempotent).
3. **Phase 2** prints `N jobs to run for {model}` and spawns the pipeline
   subprocess. The model loads (~30 s for Gemma) then logs each chunk's
   response.
4. **Phase 3** repeats Phase 2 for the next pipeline.
5. **Phase 4** prints the eval table to stdout AND writes
   `{out_dir}/eval_table.txt` (or `eval_table_multiclass.txt`).

### Step 5. Inspect results
```bash
# Aggregated table
cat outputs/asbd_eval_multiclass_rgb/eval_table_multiclass.txt

# Raw per-clip data (used for further analysis)
python3 -c "import json; d=json.load(open('outputs/asbd_eval_multiclass_rgb/results_multiclass.json')); print(d['per_clip'][:2])"

# Single clip's raw response
cat outputs/asbd_eval_multiclass_rgb/a/per_behavior/armflapping/ssbd/v_ArmFlapping_01/run_1/log.json
```

### Step 6. Re-run with overrides

To force re-inference even when outputs exist:
```bash
python3 run_asbd_eval_multiclass.py ... --force
```

To limit to a single source for a smoke test:
```bash
python3 run_asbd_eval_multiclass.py ... --sources ssbd --limit 5
```

To use a different max-token limit (e.g. for very short responses):
```bash
MAX_NEW_TOKENS=256 python3 run_asbd_eval.py ...
# OR for the multiclass eval:
python3 run_asbd_eval_multiclass.py --max_new_tokens 256 ...
```

### Step 7. (Optional) Run RGB and Skeleton in parallel

The two modes use independent prompts/inputs and don't conflict on disk.
Launch them on disjoint GPU sets:

```bash
# Terminal 1 — RGB on GPUs 1,3
python3 run_asbd_eval_multiclass.py --mode rgb --cuda 1,3 \
    --out_dir outputs/asbd_eval_multiclass_rgb --normal_samples 30 \
    > logs/asbd_eval_multiclass_rgb.log 2>&1 &

# Terminal 2 — Skeleton on GPUs 4,5
python3 run_asbd_eval_multiclass.py --mode skeleton \
    --skeleton_dir data/asbd_skeleton_output/asbd_skeleton_output \
    --cuda 4,5 \
    --out_dir outputs/asbd_eval_multiclass_skeleton --normal_samples 30 \
    > logs/asbd_eval_multiclass_skeleton.log 2>&1 &
```

## Prompt modes (`--mode rgb | skeleton`)

- **`rgb`** — appearance-based definitions ("subject repeatedly flaps both arms…")
- **`skeleton`** — geometric joint-rule definitions referencing keypoint names ("L.Wrist and R.Wrist repeatedly move up/down relative to L.Shldr / R.Shldr…")

Mode is auto-set to `skeleton` when `--skeleton_dir` is passed.

## Environment variables

| Var | Default | Where used | Purpose |
|---|---|---|---|
| `CUDA_VISIBLE_DEVICES` | — | all pipelines | Set per-pipeline GPU assignment (managed by the orchestrator's `--cuda` flag). |
| `MAX_NEW_TOKENS` | pipeline default (4096 for Gemma, 2048 for D/E/F) | all pipelines | Override `model.generate(...)` max tokens. The multiclass orchestrator sets this to **8192** via subprocess env. |

## Output structure

```
outputs/{eval_name}/
├── _trimmed/{source}/*.mp4           # pre-trimmed clips
├── _manifests/{pipe}_manifest.json   # per-pipeline job list
├── {pipe}/
│   └── {source}/{video}/run_{N}/     # single-question evals
│       ├── log.json                  # raw model responses
│       ├── behavior_intervals.json   # parsed labels
│       └── summary.txt
├── eval_table.txt                    # per-pipeline accuracy
├── eval_table_{source}.txt           # per-source breakdown
└── results.json                      # raw per-clip results
```

For the multiclass eval, per-clip outputs are under
`{pipe}/per_behavior/{behavior}/{source}/{video}/run_{N}/` (4 sets per clip).

## Setup

### h220six (x86_64, 6× H200 NVL)
```bash
python3 -m venv venvs/gemma
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers.git
pip install decord  # x86_64 only
```

### DGX Spark (ARM64, Blackwell GB10)
```bash
# Use --system-site-packages to inherit PyTorch from NGC container
python3 -m venv --system-site-packages venvs/gemma
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers.git
# Note: decord has no ARM64 wheel — OpenCV used instead
```

### Pipeline-specific dependencies
- **E (LLaVAction)**: `pip install git+https://github.com/AdaptiveMotorControlLab/LLaVAction.git`
- **D (Phi-4)**: requires `soundfile`

## Hardware

- **h220six**: 6× NVIDIA H200 NVL (143 GB each, GPUs 0–5)
- **DGX Spark**: NVIDIA GB10 Blackwell, 128 GB unified memory

## Parallel runs

Two evals on disjoint GPU sets are fully independent (separate Python
processes, separate model copies). Example: RGB on `--cuda 1,3` and skeleton
on `--cuda 4,5` can run simultaneously without contention.
