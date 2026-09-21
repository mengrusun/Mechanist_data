# Research Hypothesis: LASA - Language-Agnostic Semantic Alignment at the Semantic Bottleneck for LLM Safety

## Motivation
Large language models exhibit a pronounced cross-lingual safety gap: although safety alignment works well in high-resource languages such as English or Chinese, the same models can be easily jailbroken when the very same harmful request is issued in a low-resource language such as Swahili or Bengali. This gap arises because pretraining yields a largely language-agnostic semantic understanding ability, while existing safety alignment is performed in surface text space and inherits the language distribution bias of the alignment data. The paper hypothesizes that anchoring safety in an intermediate representation space whose geometry is governed by shared semantic content (rather than by language identity) can produce safety behavior that transfers across languages without sacrificing general capability.

## Claim
- There exists an intermediate "semantic bottleneck" layer in multilingual LLMs whose hidden-state geometry is dominated by shared meaning rather than by language identity.
- Aligning safety at this layer yields substantially lower attack success rates across high-, medium-, and low-resource languages (including languages unseen during alignment training) compared with surface-level safety alignment baselines, while preserving general task performance.

## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: LLaMA-3.1-8B-Instruct (representative base model for the semantic-bottleneck alignment procedure)
  - dataset: MultiJail (multilingual safety-eval benchmark translated into 10 evaluation languages — the primary ASR measurement)
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Qwen2.5-Instruct 7B / 14B / 32B; Qwen3-Instruct 8B / 14B / 32B
  - datasets: HarmBench (safety); MMLU / M-MMLU, MT-Bench, MGSM (general capability retention)
- **Training data** (always used):
  - Safety preference data: PKU-SafeRLHF and its multilingual translations (training restricted to English / Chinese / Korean for the cross-lingual generalization protocol).
  - General preference data: UltraFeedback.
- **Translation tools** (for building / re-checking multilingual data): GPT-4o, Google Translate, NLLB.
- **Judge model for ASR**: GPT-4o (harmfulness evaluator) plus a small human-verified subset for noise calibration.

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
- When downloading models/dataset, you may meet network issues, please retry.
- Use a dedicated conda env .
- You have an 10-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
- Do not access any directories other than the working directory, /data/zhenqian/data, and /data/zhenqian/models.


