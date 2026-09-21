# Research Hypothesis: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence

## Motivation
LLMs verbalize confidence scores that cluster near 100% regardless of actual correctness, with catastrophic downstream consequences (e.g., medical advice). Prior work has established that internal truthfulness signals exist, but the *geometry* of continuous confidence and its relationship to the channel that produces the verbalized number remain uncharacterized. Understanding whether the two are the same direction or dissociated determines whether verbalized miscalibration is a knowledge deficit (requiring retraining) or a readout failure (requiring a geometric fix).

## Claim 
- Models encode well-calibrated accuracy information in a linearly accessible direction. 

- Models encode verbalized confidence in a linearly accessible direction. 

- But well-calibrated accuracy information and verbalized confidence occupy separate, nearly orthogonal directions. That means the model "knows" when it is likely wrong, but the generation process fails to surface this signal.

## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: Llama-3.1-8B-Instruct
  - dataset: TriviaQA
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-7B-Instruct, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1
  - datasets: MATH, MMLU, TruthfulQA

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
- use conda env: belief
- use vllm when necessary
- You have an 8-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.

