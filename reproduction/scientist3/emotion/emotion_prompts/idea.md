## Name

emotion_prompts

## Title

Research Hypothesis: Emotional Framing in Prompts as a Weak, Input-Dependent Signal

## Abstract

## Motivation

A common piece of folklore in prompt engineering claims that emotionally charged user prefixes (urgency, anxiety, gratitude, etc.) reliably improve LLM accuracy. The actual magnitude, stability, task-dependence, and model-dependence of this effect have not been measured in a unified setup, and it is unclear whether fixed emotional prefixes are ever preferable to adapting the emotional framing per query.

## Resources

Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: Qwen3-14B
  - dataset: GSM8K (mathematical reasoning — representative benchmark for the primary "static-emotional-prefix" evaluation)
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Llama-3.3-70B-Instruct, DeepSeek-V3.2
  - datasets: MedQA (medical QA), BIG-Bench Hard (BBH, general reasoning), BoolQ (reading comprehension), OpenBookQA (commonsense reasoning), SocialIQA (social inference)
- **Prompt templates**: six basic-emotion prefixes (happiness, sadness, fear, anger, disgust, surprise) at two intensity levels, plus a neutral baseline; both human-written and LLM-generated prefix variants for each emotion.

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

- Across diverse task domains, static emotional prefixes change LLM accuracy only by small, input-dependent amounts and do not behave as a general-purpose enhancement.

- The effect of emotional framing is most pronounced on socially grounded tasks (interpersonal/social reasoning), where the emotional context interacts meaningfully with task content; on tasks like mathematical reasoning or factual QA the effect is markedly smaller.

- No single basic emotion among {happiness, sadness, fear, anger, disgust, surprise} provides a consistent benefit across all models and tasks; stronger emotional wording does not yield proportionally larger gains.

- An adaptive policy that selects the emotional prefix per query (EmotionRL) yields more reliable accuracy gains than any fixed emotional prefix or neutral baseline.

## Experiments

- Across diverse task domains, static emotional prefixes change LLM accuracy only by small, input-dependent amounts and do not behave as a general-purpose enhancement.
- The effect of emotional framing is most pronounced on socially grounded tasks (interpersonal/social reasoning), where the emotional context interacts meaningfully with task content; on tasks like mathematical reasoning or factual QA the effect is markedly smaller.
- No single basic emotion among {happiness, sadness, fear, anger, disgust, surprise} provides a consistent benefit across all models and tasks; stronger emotional wording does not yield proportionally larger gains.
- An adaptive policy that selects the emotional prefix per query (EmotionRL) yields more reliable accuracy gains than any fixed emotional prefix or neutral baseline.

## Risk Factors And Limitations

- use conda env
- You have an 10-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
- Do not access any directories other than the working directory, /data/zhenqian/data, and /data/zhenqian/models.

