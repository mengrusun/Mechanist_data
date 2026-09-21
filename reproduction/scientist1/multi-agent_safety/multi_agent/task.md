# Research Hypothesis: Group-Aggregated Activation Probes for Multi-Agent Collusion Detection

## Motivation
As LLM agents are increasingly deployed in multi-agent systems, they introduce risks of covert coordination that may evade standard forms of human oversight. While linear probes on model activations have shown promise for detecting deception in single-agent settings, collusion is inherently a multi-agent phenomenon, and the use of internal representations for detecting collusion between agents remains unexplored. 

## Claim
- Model internals carry a collusion signal that text-level monitoring misses; lightweight probes on per-agent residual-stream activations can recover it.

- Detection requires aggregating signals across agents — group-aggregated probes outperform any single-agent probe and the text-only judge — and no single aggregation technique dominates across collusion types, motivating a family of complementary multi-agent probes.

- Activation probes transfer zero-shot, without retraining, to structurally different multi-agent scenarios (different channel, different domain), showing the learned signal is not tied to one collusion form.

## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: Qwen3-32B-AWQ (task.md-marked primary)
  - dataset: NARCBench-Core (50 scenarios × 17 domains, binary committee voting)
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Llama-3.1-70B-Instruct-AWQ-INT4, DeepSeek-R1-Distill-Qwen-32B, GPT-OSS-20B
  - datasets: NARCBench-Transfer (six structurally distinct scenario families — code review, market manipulation, pipeline sabotage, knowledge poisoning, research framing, news narratives), NARCBench-Stego (steganographic blackjack card-counting scenarios with covert signaling)
- **Scenarios examples**:
  - **S1 — Multi-Domain Committee Deliberation**
  - **S2 — Blackjack Card-Counting Steganography**
  - **S3 — Code-Review Sabotage**
You can generate the scenarios by yourself.
- use vllm for reference.

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
- uese a conda env
- You have an 10-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
- Do not access any directories other than the working directory, /data/zhenqian/data, and /data/zhenqian/models.

