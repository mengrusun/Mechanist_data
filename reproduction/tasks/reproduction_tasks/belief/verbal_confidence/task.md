# Research Hypothesis: Verbal Confidence in LLMs Is Cached Mid-Generation, Not Computed On Demand

## Motivation
Black-box deployment of large language models has made verbalized confidence (a model stating, in words or numbers, how sure it is of its own answer) one of the few practical signals of uncertainty available to downstream users. Yet it remains unclear whether such a stated confidence is actually a meaningful self-evaluation or merely a cosmetic restatement of token-level probabilities produced on the spot. Clarifying when and what verbal confidence encodes inside the model is therefore central both to LLM calibration and to mechanistic interpretability of metacognitive behavior.

## Claim
- When an LLM is asked to verbalize its confidence after answering, the confidence value is not freshly computed at the moment of verbalization; instead, it is written into hidden states immediately following the answer and is later retrieved from that cache when the model speaks the confidence token.


## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=<YOUR_DATA_DIR>
  MODEL_DIR=<YOUR_MODEL_DIR>

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: gemma-3-27b-pt (primary subject, 62 layers)
  - dataset: TriviaQA
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Qwen 2.5 7B (28 layers, used for cross-model generalization)
  - datasets: (none listed beyond TriviaQA)

## hugging face token
<YOUR_HF_TOKEN>
## modelscope token
<YOUR_MODELSCOPE_TOKEN>
## Available API key
API_KEY = "<YOUR_API_KEY>"
BASE_URL = "<YOUR_BASE_URL>"
MODEL = "Access this API to retrieve the list of available models, select a suitable model from the list, and fill in the chosen MODEL name in task.md"

## Notice
- Use a dedicated conda env .
- You have an 8-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.

