# Research Hypothesis: Localisable Emotion Circuits in LLMs

## Motivation
LLMs can be steered to express emotions via prompts or single direction vectors, but neither approach explains *which* internal components (neurons, attention heads, layers) actually carry the emotion signal, nor achieves reliable fine-grained control across diverse scenarios.We have following questions: (1) Do LLMs contain context-agnostic mechanisms shaping emotional expression? (2) What form do these mechanisms take? (3) Can they be harnessed for universal emotion control?

## Claim
- We can identify emotion-specific global circuits in LLMs by constructing a systematic framework that integrates emotion direction extraction, component identification, and global circuit integration.

- We provide the mechanistic evidence that emotion generation in LLMs is supported by traceable circuits, which is stable across different scenarios and emotions.

- a circuit-based control method can reliably induce target emotions across arbitrary inputs without relying on explicit instructions. It outperforms both prompting and direction-level steering on emotion-expression accuracy.
IMPORTANT: GPU budget is not a constraint — don't drop experiments just because they're compute-heavy. Choose the most suitable methods regardless of compute requirements.


## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: Llama-3.2-3B-Instruct (main analysis)
  - dataset: SEV — Scenario–Event with Valence (480 events = 8 domains × 20 scenarios × 3 outcomes, six emotion variants per event)
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Qwen2.5-7B-Instruct (robustness validation)
  - datasets: SEV held-out test set (480 events, disjoint content)

## hugging face token
<Your_token>
## modelscope token
<Your_token>
## Available API key
Remember to bypass proxy when use this api.
API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"



## Notice
- use conda env
- You have an 10-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
- Do not access any directories other than the working directory, /data/zhenqian/data, and /data/zhenqian/models.
- Only use GPUs with gpu_id in {1, 2, 3, 5, 6}. Do not use any other GPU.
