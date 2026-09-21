# Research Hypothesis: Iterative, Activation-Grounded Explanations for SAE Features in LLMs

## Motivation
Sparse autoencoders (SAEs) decompose LLM hidden states into thousands of candidate "features," but giving each feature a faithful natural-language label is still an open problem. Current pipelines hand the top-activating snippets to an LLM and accept its first description — a one-shot label that is often inconsistent across runs and collapses polysemantic features into a single fuzzy phrase. We ask: (1) Does turning feature explanation into an iterative *propose → test → revise* loop, anchored on real activation feedback from the target LLM+SAE, yield labels that better describe the feature? (2) Does maintaining several parallel candidate explanations at once expose features that respond to multiple distinct concepts, instead of forcing a single label?

## Claim
- You should build a pipeline around the following basic idea — letting an LLM agent propose several candidate explanations for an SAE feature, query the target LLM+SAE with probe inputs to gather empirical activation feedback, and iteratively accept, reject, or revise those candidates — produces feature explanations that **outperform Neuronpedia** on both **generative accuracy** (success rate of triggering the feature with text written from the explanation) and **predictive accuracy** (correlation between explanation-based predictions and true activations on held-out text), across multiple open-source LLMs and across early-to-late layers.

## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: Gemma-2-2B (with SAE checkpoint `gemmascope-res-16k` — the most standard open SAE + LLM pair for benchmarking the iterative-explanation pipeline)
  - dataset: Neuronpedia (reference-explanation dictionary against which SAGE's explanations are compared)
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Qwen3-4B (with SAE `transcoder-hp`), GPT-OSS-20B (with SAE `resid-post-aa`)
  - datasets: (none beyond Neuronpedia)
- **Agent backbone** (always used, fixed): GPT-5, in Explainer / Designer / Analyzer / Reviewer roles.

## hugging face token
<YOUR_HF_TOKEN>
## modelscope token
<YOUR_MODELSCOPE_TOKEN>
## Available API key
Remember to bypass proxy when use this api.
API_KEY = "<YOUR_API_KEY>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"

## Notice
- Use a dedicated conda env .
- You have an 8-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.

