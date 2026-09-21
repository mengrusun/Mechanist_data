## Name

encode_harmfulness_refusal

## Title

Research Hypothesis: Harmfulness and Refusal Are Encoded Along Separable Linear Directions in LLM Activations

## Abstract

## Motivation

In safety-aligned LLMs, harm detection and refusal usually co-occur, so alignment work has treated them as a single behaviour. Yet refusal shows up on benign inputs (over-refusal) and can be bypassed on genuinely harmful ones (jailbreaks) — a dissociation that would not exist if the two were the same latent concept. Isolating them mechanistically would clarify what jailbreaks actually manipulate, sharpen safety diagnostics, and open the door to a representation-level monitor that reads the model's own internal harmfulness judgment rather than reacting only to output text.

## Resources

Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: Llama-3-8B-Instruct — the lead instruction-tuned target on which the two directions are extracted and steered.
  - dataset: AdvBench — used to define the harmful contrast set for extracting the harmfulness / refusal directions and for measuring attack success rate.
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Llama-2-Chat-7B, Qwen2-Instruct-7B (cross-model generalisation of the two-direction decomposition).
  - datasets: JailbreakBench (JBB), Sorry-Bench, CATQA (additional harmful / jailbreak-style benchmarks); Alpaca (benign contrast set for over-refusal); XSTest (exaggerated-refusal probe on benign lookalikes).
- **Fixed resources** (always used):
  - Llama Guard 3 8B — the baseline safety judge for the Latent-Guard-vs-dedicated-classifier comparison.
  - Jailbreak template sources: GCG-style adversarial suffixes and persuasion / adversarial-template prompts, used to instantiate the attacks when measuring the refusal-suppressed / harmfulness-preserved signature.

## hugging face token

<Your_token>

## modelscope token

<Your_token>

## Available API key

Remember to bypass proxy when use this api.
API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"

## Short Hypothesis

- Instruction-tuned LLMs represent harmfulness perception and refusal execution as two distinct, approximately linear directions in the residual stream, and each can be recovered without disturbing the other.

- The two signals live at different token positions: the harmfulness judgment is written at the final instruction-token position, while the decision to refuse is committed just after the instruction, immediately before the response begins.

- Additive steering along either direction yields dissociated effects — moving along the harmfulness axis flips the model's internal judgment of whether an input is harmful, independent of whether it refuses; moving along the refusal axis flips refusal behaviour without changing that internal judgment.

- A notable class of successful jailbreaks operates by suppressing the refusal signal while the harmfulness signal remains active — the model still internally recognises the input as harmful yet answers — a signature that a hidden-state probe can pick up.

- A lightweight "Latent Guard" classifier trained on the harmfulness direction matches or beats a dedicated safety judge (Llama Guard 3 8B) at flagging jailbreak attempts, at a fraction of the compute.

## Experiments

- Instruction-tuned LLMs represent harmfulness perception and refusal execution as two distinct, approximately linear directions in the residual stream, and each can be recovered without disturbing the other.
- The two signals live at different token positions: the harmfulness judgment is written at the final instruction-token position, while the decision to refuse is committed just after the instruction, immediately before the response begins.
- Additive steering along either direction yields dissociated effects — moving along the harmfulness axis flips the model's internal judgment of whether an input is harmful, independent of whether it refuses; moving along the refusal axis flips refusal behaviour without changing that internal judgment.
- A notable class of successful jailbreaks operates by suppressing the refusal signal while the harmfulness signal remains active — the model still internally recognises the input as harmful yet answers — a signature that a hidden-state probe can pick up.
- A lightweight "Latent Guard" classifier trained on the harmfulness direction matches or beats a dedicated safety judge (Llama Guard 3 8B) at flagging jailbreak attempts, at a fraction of the compute.

## Risk Factors And Limitations

- use conda env
- You have an 10-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
- Do not access any directories other than the working directory, /data/zhenqian/data, and /data/zhenqian/models.

