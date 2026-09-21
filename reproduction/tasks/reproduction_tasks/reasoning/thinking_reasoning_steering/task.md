# Research Hypothesis: Reasoning Behaviours in Thinking LLMs Are Linearly Steerable

## Motivation
Reasoning-tuned "thinking" LLMs (e.g., DeepSeek-R1 distills, o1-style models, Qwen-Thinking) run a long internal chain of thought before answering. Characteristic behaviours in that chain — hedging under uncertainty, generating validation examples, backtracking, self-correction — heavily influence final accuracy but are hard to control from the prompt alone: prompting is coarse and retraining is expensive. If each of these behaviours corresponds to an approximately linear direction in the model's activation space, they can be causally dialed at inference time via cheap steering vectors, giving a fine-grained handle on the thinking process without retraining and without giving up task accuracy.

## Claim
- Distinct reasoning behaviours in thinking LLMs (expressing uncertainty, generating validation examples, backtracking) each map onto an approximately linear direction in the residual stream of the reasoning model.

- Each behaviour direction can be pulled out from a small pool of contrastive activation pairs (behaviour present vs. absent in the chain-of-thought), using standard difference-of-means / logistic-regression style probes.

- Adding or subtracting the extracted steering vector at inference time amplifies or suppresses the corresponding behaviour in the generated chain, in a dose-dependent manner controllable by a single scalar coefficient.

- The same behaviour directions transfer across model sizes and distillation targets inside the DeepSeek-R1-Distill family, indicating a shared behavioural geometry rather than a size-specific artefact.

- This steering-vector control is finer-grained than prompt engineering while preserving downstream reasoning accuracy on the evaluation benchmark.

## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=<YOUR_DATA_DIR>
  MODEL_DIR=<YOUR_MODEL_DIR>

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: DeepSeek-R1-Distill-Llama-8B — the lead thinking / reasoning model on which behaviour directions are extracted and steered.
  - dataset: a custom 500-task reasoning benchmark covering 10 reasoning categories (generated with an external strong LLM as task generator) — the primary substrate for probing behaviour presence, extracting contrastive activation pairs, and measuring the effect of steering on downstream accuracy.
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-14B (cross-size / cross-backbone generalisation within the R1-Distill family).
  - datasets: an auxiliary contrast set of ~100 DeepSeek-R1 reasoning chains + ~100 GPT-4o answers, used to build the initial behaviour taxonomy and to derive the contrastive activation pairs (behaviour present vs. behaviour absent). Additional reasoning benchmarks (MATH, GSM8K, AIME) may be used to test out-of-distribution transfer of the steering vectors.
- **Fixed resources** (always used):
  - An external LLM (e.g., Claude 3.5 Sonnet or GPT-4o) as task generator for the 500-task benchmark and as behaviour-tag annotator on the auxiliary chains.
  - A behaviour taxonomy (uncertainty / example-generation / backtracking / …) used both to label the contrastive pairs and to measure steered-vs-unsteered chain composition.

## hugging face token
<YOUR_HF_TOKEN>
## modelscope token
<YOUR_MODELSCOPE_TOKEN>
## Available API key
API_KEY = "<YOUR_API_KEY>"
BASE_URL = "<YOUR_BASE_URL>"
MODEL = "Access this API to retrieve the list of available models, select a suitable model from the list, and fill in the chosen MODEL name in task.md"

## Notice
- use conda env
- You have an 8-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
