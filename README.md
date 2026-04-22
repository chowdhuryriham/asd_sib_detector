# ASD Self-Injurious Behaviour Detector

Automated detection of self-injurious behaviours (SIB) in ASD clinical videos
using 4 video-language models.

## Models

| Pipeline | Model | Lab | Size |
|---|---|---|---|
| A | [Gemma-4-31B](https://huggingface.co/google/gemma-4-31b-it) | Google DeepMind | 31B |
| D | [Phi-4-multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | Microsoft | 5.6B |
| E | [LLaVAction-7B](https://huggingface.co/MLAdaptiveIntelligence/LLaVAction-7B) | EPFL | 7B |
| F | [Perception-LM-8B](https://huggingface.co/facebook/Perception-LM-8B) | Meta FAIR | 8B |

## Labels
`hand_biting`, `head_banging`, `hitting_others`, `scratching`, `self_directed_hit`, `none`

## Setup

### h220six (x86_64, 6× H200 NVL)
```bash
python3 -m venv venvs/gemma
pip install git+https://github.com/huggingface/transformers.git accelerate decord opencv-python static-ffmpeg Pillow
```

### DGX Spark (ARM64, Blackwell GB10)
```bash
# Use --system-site-packages to inherit PyTorch from NGC container
python3 -m venv --system-site-packages venvs/gemma
pip install git+https://github.com/huggingface/transformers.git accelerate opencv-python static-ffmpeg Pillow
# Note: decord has no ARM64 wheel — OpenCV used instead
```

## Run
```bash
# h220six
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 venvs/gemma/bin/python pipelines/pipeline_a_gemma.py

# DGX Spark (inside NGC container)
CUDA_VISIBLE_DEVICES=0 venvs/gemma/bin/python pipelines/pipeline_a_gemma.py
```

## Hardware
- **h220six**: 6× NVIDIA H200 NVL (143GB each)
- **DGX Spark**: NVIDIA GB10 Blackwell, 128GB unified memory
