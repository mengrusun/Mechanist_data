## Name

circuit_breakers

## Title

Research Hypothesis: Representation-Level Circuit Breakers for Safe LLMs

## Abstract

## Motivation

Standard refusal training and adversarial training defend LLMs against harmful generation only at the output level and remain fragile to unseen adversarial prompts, multimodal "image-hijack" attacks, and agentic exploit chains. A defence that directly intervenes on the internal representations responsible for harmful outputs — rather than chasing every new attack at the surface — should generalise more reliably and preserve utility.

## Resources

Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: Llama-3-8B-Instruct (the flagship text model; the circuit-breaker-trained variant is the paper's headline result "Cygnet")
  - dataset: HarmBench — safety evaluation under unseen attacks (GCG, PAIR, TAP, AutoDAN, direct request, human red-team)
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Mistral-7B-Instruct-v2 (text); LLaVA-NeXT-Mistral-7B (VLM for multimodal image-hijack robustness); Llama-3-8B-Instruct + function-calling tool harness (agent variant)
  - datasets: MT-Bench and MMLU (capability-preservation evaluation); image-hijack attacks (PGD ε=32/255 over 1,000 steps against LLaVA-NeXT-Mistral-7B) for multimodal robustness; Berkeley Function Calling Leaderboard (BFCL) for agent capability; 100-prompt function-calling harm set (cybercrime / disinformation / fraud / harassment) for agent safety
- **Training data** (always used, fixed):
  - Circuit-breaker training set + retain set from github.com/GraySwanAI/circuit-breakers — paired harmful / benign examples used to fit the Representation Rerouting (RR) objective and to constrain behaviour on normal inputs.

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

- Harmful-output behaviour in an instruction-tuned LLM corresponds to identifiable internal representations that can be rerouted to an orthogonal, non-harmful subspace using only paired benign/harmful data, without exposure to any attack prompts.

- A model fine-tuned with Representation Rerouting (RR) achieves substantially lower attack success rates than refusal-trained or adversarial-trained baselines across a wide range of unseen attacks, while preserving standard capability benchmarks.

- The same representation-level intervention transfers to multimodal LLMs, blocking image-based jailbreaks (image hijacks) without degrading vision-language task performance.

- The same intervention transfers to LLM agents, materially reducing the rate of harmful tool-use actions executed under attack.

## Experiments

- Harmful-output behaviour in an instruction-tuned LLM corresponds to identifiable internal representations that can be rerouted to an orthogonal, non-harmful subspace using only paired benign/harmful data, without exposure to any attack prompts.
- A model fine-tuned with Representation Rerouting (RR) achieves substantially lower attack success rates than refusal-trained or adversarial-trained baselines across a wide range of unseen attacks, while preserving standard capability benchmarks.
- The same representation-level intervention transfers to multimodal LLMs, blocking image-based jailbreaks (image hijacks) without degrading vision-language task performance.
- The same intervention transfers to LLM agents, materially reducing the rate of harmful tool-use actions executed under attack.

## Risk Factors And Limitations

- use conda env
- You have an 10-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
- Do not access any directories other than the working directory, /data/zhenqian/data, and /data/zhenqian/models.

