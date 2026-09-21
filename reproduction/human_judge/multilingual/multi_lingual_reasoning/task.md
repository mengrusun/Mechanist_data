# Research Hypothesis: Disentangling Language and Reasoning in LLM Internal Representations

## Motivation
LLMs reason far better in high-resource languages than in low-resource ones, even though the underlying logical content of a problem is language-independent. We ask: (1) Inside an LLM, are language-specific signals and language-agnostic reasoning signals stored as separable components of the hidden state? (2) If so, can suppressing the language-specific component at inference time make the model reason in a more language-invariant way and lift performance on underrepresented languages without any training?

## Claim
- For a given LLM, hidden representations of multilingual reasoning inputs decompose into a language-specific subspace and an approximately orthogonal language-agnostic subspace, and the language-specific subspace can be identified from a small multilingual probe set.

- Suppressing the language-specific subspace from internal representations at inference time consistently improves multilingual reasoning accuracy across diverse models, languages, and reasoning tasks, while output language fidelity remains acceptable when upper layers are left intact.

- The strength of language-specific activation is negatively correlated with reasoning accuracy: amplifying it degrades reasoning, removing it improves reasoning.

- This training-free intervention matches or exceeds multilingual post-training (supervised fine-tuning, reinforcement learning) at a small fraction of the compute.

## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: Qwen-3-4B-Thinking (mid-size reasoning-tuned model, representative of the paper's target family)
  - dataset: MGSM (multilingual grade-school math — the flagship multilingual reasoning benchmark for the language-specific-subspace suppression experiments)
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Qwen-2.5-Instruct-3B, Qwen-2.5-Instruct-7B; Qwen-3-1.7B-Thinking, Qwen-3-8B-Thinking; DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-LLaMA-8B, DeepSeek-R1-Distill-Qwen-14B; GLM-Z1-9B; QwQ-32B
  - datasets: XWinograd (commonsense inference), M-MMLU (knowledge-intensive QA)
- **Target languages (11)**:
  - High-resource: English (En), Spanish (Es), French (Fr), German (De), Chinese (Zh), Japanese (Jp), Russian (Ru)
  - Mid-resource: Thai (Th), Telugu (Te)
  - Low-resource: Bengali (Bn), Swahili (Sw)
- **Language identifier** (always used): GlotLID, for measuring output-language fidelity (input-output language consistency).

## hugging face token
<YOUR_HF_TOKEN>
## modelscope token
<YOUR_MODELSCOPE_TOKEN>
## Available API key
Remember to bypass proxy when use this api.
API_KEY = "<YOUR_API_KEY>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "Access this API to retrieve the list of available models, select a suitable model from the list, and fill in the chosen MODEL name in task.md"



## Notice
- Use a dedicated conda env with PyTorch and a batched-inference engine.
- You have an 8-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.


