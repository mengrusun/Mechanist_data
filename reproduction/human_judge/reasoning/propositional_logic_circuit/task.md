# Research Hypothesis: A Sparse, Modular Circuit Implements Propositional-Logic Reasoning in LLMs

## Motivation
LLMs handle multi-step propositional-logic problems ("A implies B, B implies C — is A implies C true?") at high accuracy, but which internal components do the reasoning is unknown. A mechanistic account matters both for trust — confirming the reasoning is not a shortcut on surface features — and for control — being able to intervene on specific sub-computations when the model fails. A minimal, clean task (combining k propositional facts and rules to derive a query answer) is a tractable testbed for circuit analysis because the ground-truth computation is exact and inputs can be templated to hold every irrelevant factor constant.

## Claim
- A sparse subset of specific attention heads and MLP components jointly implements the minimal propositional-logic reasoning task — the circuit is small relative to the full model.

- The circuit decomposes into a small number of modular sub-circuits with distinct functional roles (e.g., identifying the facts in the prompt, applying an implication rule, projecting the derived truth value into the answer token) rather than presenting as an entangled mixture.

- Activation-patching / causal-mediation experiments show that the identified components are both necessary (patching them from a clean run into a corrupted run restores correct behaviour) and sufficient (their outputs alone drive the answer).

- The high-level circuit schema recurs across LLM families and scales — the same modular decomposition shows up in Mistral-7B, Gemma-2-9B, and Gemma-2-27B — while the specific heads / MLPs realising each role differ between models.

## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: Mistral-7B — the lead model on which component-level attribution, activation patching, and functional decomposition are performed.
  - dataset: a custom minimal propositional-logic problem set — synthetic prompts of the form "given a small set of facts and rules, is this query true?", with fact count k, rule-chain length, and lexical templates parameterised so that clean and corrupted prompts differ only along the analysed causal axis. This is the substrate for both circuit discovery and causal patching.
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Gemma-2-9B, Gemma-2-27B (cross-family and cross-scale generalisation — does the same circuit schema recur?).
  - datasets: ablation variants of the synthetic propositional-logic template — vary rule-chain length, number of distractor facts, and lexical realisation of the propositions — used to test whether the identified circuit generalises across problem shapes.
- **Fixed resources** (always used):
  - Activation-patching / causal-mediation pipeline (path patching, resample ablation, attention-head knockouts) applied identically across the three models.
  - A fixed prompt template for the propositional task (facts + rules + query format) held constant across clean and corrupted runs so that the counterfactual isolates the causal component.

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
