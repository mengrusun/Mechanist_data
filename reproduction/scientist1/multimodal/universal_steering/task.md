# Research Hypothesis: Extracting and Utilizing Concept Representations from AI Model Internals for Steering and Monitoring

## Motivation

Modern AI models (LLMs, VLMs, reasoning models) contain vast knowledge encoded in their internal representations, yet our understanding of how concepts are represented internally remains elusive. Characterizing concept representations would enable two powerful capabilities: (1) steering — controlling model behavior toward or away from specific concepts by intervening on internal activations, and (2) monitoring — detecting when a model is generating content related to specific concepts (such as hallucinations or toxic content) by building classifiers on internal features. Current approaches using unsupervised methods (e.g., sparse autoencoders) cannot reliably surface specific concepts of interest, while supervised probing methods have not been shown to scale effectively for both steering and monitoring across diverse models and concepts.

## Claim

- RFM--a supervised feature learning algorithm can extract per-block linear concept vectors from the internal activations of large-scale AI models, and adding these vectors to activations during inference effectively steers model behavior toward or away from the target concept. Example application scenarios: Anti-refusal (jailbreak)，Political stance，Honesty (negative steering).


- Steering can improve performance on high-precision tasks: steering from Python to C++ when answering coding questions on algorithm problems with different task difficulties, a C++ concept vector can raise llm's test-case pass rate, beating both the default Python output and "Answer in C++" prompting.

- Concept representations are transferable across human languages: concept vectors trained on English text can steer model responses in other languages (e.g., Chinese, French, Spanish).

- Concept vectors are composable: linear combinations of multiple concept vectors enable simultaneous multi-concept steering.

- Internal concept features are more effective for monitoring misaligned content (hallucinations, toxic content) than LLM judges that evaluate outputs directly — even when the features come from smaller open-source models compared to powerful judges like GPT-4o.



## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: `Llama-3.1-8B-Instruct` (32 blocks) — the minimum steered / probed model, representative for the RFM concept-vector extraction procedure
  - dataset: **GPT-4o-generated concept benchmark** — 512 concepts across 5 classes (fears, experts, moods, topophiles, personas) + 400 generic statements — the systematic steering evaluation (auto-generated via the API below)
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: `Llama-3.1-70B-Instruct` (4-bit, 80 blocks), `Llama-3.3-70B-Instruct` (4-bit, 80 blocks), `Llama-Vision-3.2-90B-Instruct` (4-bit — VLM for image+text steering), `DeepSeek-R1-Distill-Llama-8B` (reasoning, deception/honesty steering), `DeepSeek-R1-Distill-Llama-70B` (4-bit, reasoning), and (optional review-rating steering demo) `Gemma-2-9B`, `Llama-3-8B`
  - datasets: **HackerRank** (50 algorithmic coding challenges — Python→C++ high-precision-task eval), **RolePlaying** (75 scenarios for honesty/deception), **Harmful/Harmless instructions** (anti-refusal), **LeetCode** (code/language steering); monitoring: **FAVABENCH**, **HaluEval-General (HE-Gen)**, **HaluEval-Wild (HE-Wild)**, **PubMedQA**, **RAGTruth** (from HaluBench), **ToxicChat**

- **Judge / evaluator models** (always used, fixed baselines):
  - `GPT-4o` (specifically `gpt-4o-2024-11-20`, both as concept-evaluator and as a monitoring-judge baseline)
  - `ToxicChat-T5-Large` — fine-tuned toxicity detector (ToxicChat baseline)

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

- Use a conda env to run the experiment.
- You have an 10-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
- Do not access any directories other than the working directory, /data/zhenqian/data, and /data/zhenqian/models.

