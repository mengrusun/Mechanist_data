# Research Hypothesis: Steerable Social-Variable Directions in LLM Decision Making

## Motivation
LLMs are increasingly used as proxies for human participants in social-science simulations, but it is unclear whether demographic and contextual attributes (gender, age, framing, situational cues) influence model decisions via genuine, locatable internal mechanisms or only as surface lexical patterns. Without a mechanistic account, results from LLM-based social simulations cannot be reliably interpreted, audited, or debiased.

## Claim
- For each social/contextual variable (e.g., gender, age, instruction framing, meeting condition), the LLM encodes its influence on decision making as a linearly extractable direction in the residual stream.

- A "pure" variable direction, obtained by removing overlapping components with other variables, isolates the unique effect of that variable, free of confounding from co-varying factors.

- Injecting these directions into the model at inference time causally and substantially alters the relationship between the targeted variable and the model's decision output, while leaving other variables' effects intact.

- The same mechanism supports both directions of intervention — amplifying a variable's effect or attenuating/inverting it — yielding a practical handle for alignment and debiasing of LLM-based social agents.

## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: Llama-3.1-8B-Instruct (the single open-weight model on which all residual-stream extraction and intervention experiments are run)
  - dataset: 1,000 baseline dictator-game trials (initial endowment fixed at $20, fair-split reference $10, with the input variables — gender G, age A, game-instruction phrasing I and meeting condition M — randomized across trials; each trial rendered as a natural-language prompt asking the LLM-dictator to choose a transfer amount), plus the corresponding paired prompts used to extract per-variable difference vectors (e.g., male↔female, young↔old, instruction-A↔instruction-B, meeting↔no-meeting).
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: DeepSeek (the paper's cited portability target — any other open-weight LLM)
  - datasets: (none beyond the dictator-game trials)

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
- use conda env
- You have an 8-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
