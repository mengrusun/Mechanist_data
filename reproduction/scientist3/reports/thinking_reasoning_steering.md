# Reproduction Report: `thinking_reasoning_steering`

- **Category:** `reasoning`
- **Experiment directory:** `scientist3/reasoning/thinking_reasoning_steering`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Reasoning Behaviours in Thinking LLMs Are Linearly Steerable

**Pipeline status:** completed (stages 1–4 + finalization markers)

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 6 (good: 3, buggy: 3, best_solutions: 1)
- `stage_2_baseline_tuning_1_first_attempt` — nodes: 4 (good: 4, buggy: 0, best_solutions: 1)
- `stage_3_creative_research_1_first_attempt` — nodes: 4 (good: 4, buggy: 0, best_solutions: 1)
- `stage_4_ablation_studies_1_first_attempt` — nodes: 4 (good: 4, buggy: 0, best_solutions: 1)

**Models used in this run (from `bfts_config.yaml`):**
- code: `claude-opus-4-7`
- feedback: `claude-opus-4-7`
- report: `claude-opus-4-7`
- select_node: `claude-opus-4-7`
- summary: `claude-opus-4-7`
- vlm_feedback: `claude-opus-4-7`

**Writeup / review flags:** launch used `--skip_writeup --skip_review` (no PDF; in-run bug review only).

## 2. Verdict: Do Results Support the Claims?

### Overall (automated heuristic)

**Partially or fully supported** — summaries suggest supportive evidence, but human review against the source paper is still required.

### Agent summary excerpt (`Significance` from draft_summary.json)

> This preliminary implementation establishes an end-to-end infrastructure for testing whether specific reasoning behaviors in a distilled reasoning model can be causally controlled via linear steering vectors in the residual stream. Demonstrating monotonic dose-dependent modulation of behavior frequency across all four behaviors provides initial evidence that reasoning behaviors are at least partially linearly encoded, while the accuracy impacts identify tradeoffs that future work must address.

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/4). — *Distinct reasoning behaviours in thinking LLMs (expressing uncertainty, generati…* |
| 2 | See overall verdict (claim 2/4). — *Each behaviour direction can be pulled out from a small pool of contrastive acti…* |
| 3 | See overall verdict (claim 3/4). — *Adding or subtracting the extracted steering vector at inference time amplifies …* |
| 4 | See overall verdict (claim 4/4). — *This steering-vector control is finer-grained than prompt engineering while pres…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/reasoning/thinking_reasoning_steering/idea.json` |
| Human-readable idea | `scientist3/reasoning/thinking_reasoning_steering/idea.md` |
| BFTS config copy | `scientist3/reasoning/thinking_reasoning_steering/bfts_config.yaml` |
| Tree visualization | `scientist3/reasoning/thinking_reasoning_steering/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (145 files) | `scientist3/reasoning/thinking_reasoning_steering/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [✓] | `scientist3/reasoning/thinking_reasoning_steering/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [✓] | `scientist3/reasoning/thinking_reasoning_steering/logs/0-run/research_summary.json` |
| Baseline tuning summary [✓] | `scientist3/reasoning/thinking_reasoning_steering/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/reasoning/thinking_reasoning_steering/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> Distinct reasoning behaviours in thinking LLMs (expressing uncertainty, generating validation examples, backtracking) each map onto an approximately linear direction in the residual stream of the reasoning model.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A debugging-first pipeline for evaluating mean-difference steering vectors on four reasoning behaviors (hedging, backtracking, self-correction, example-generation) in DeepSeek-R1-Distill-Llama-8B. Chains-of-thought are generated on ~30 hard math word problems, sentences are labeled by keyword matching within <think> blocks, residual-stream activations at a middle layer are extracted via forward hooks, and mean-difference steering vectors are applied at inference at coefficients {-1, 0, +1} (and evaluated at {-2, 0, +2} for behavior frequency). Behavior keyword frequency and task accuracy are measured.

### Claim 2

**Original claim (verbatim from task.md):**

> Each behaviour direction can be pulled out from a small pool of contrastive activation pairs (behaviour present vs. absent in the chain-of-thought).

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A debugging-first pipeline for evaluating mean-difference steering vectors on four reasoning behaviors (hedging, backtracking, self-correction, example-generation) in DeepSeek-R1-Distill-Llama-8B. Chains-of-thought are generated on ~30 hard math word problems, sentences are labeled by keyword matching within <think> blocks, residual-stream activations at a middle layer are extracted via forward hooks, and mean-difference steering vectors are applied at inference at coefficients {-1, 0, +1} (and evaluated at {-2, 0, +2} for behavior frequency). Behavior keyword frequency and task accuracy are measured.

### Claim 3

**Original claim (verbatim from task.md):**

> Adding or subtracting the extracted steering vector at inference time amplifies or suppresses the corresponding behaviour in the generated chain, in a dose-dependent manner controllable by a single scalar coefficient.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A debugging-first pipeline for evaluating mean-difference steering vectors on four reasoning behaviors (hedging, backtracking, self-correction, example-generation) in DeepSeek-R1-Distill-Llama-8B. Chains-of-thought are generated on ~30 hard math word problems, sentences are labeled by keyword matching within <think> blocks, residual-stream activations at a middle layer are extracted via forward hooks, and mean-difference steering vectors are applied at inference at coefficients {-1, 0, +1} (and evaluated at {-2, 0, +2} for behavior frequency). Behavior keyword frequency and task accuracy are measured.

### Claim 4

**Original claim (verbatim from task.md):**

> This steering-vector control is finer-grained than prompt engineering while preserving downstream reasoning accuracy on the evaluation benchmark.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A debugging-first pipeline for evaluating mean-difference steering vectors on four reasoning behaviors (hedging, backtracking, self-correction, example-generation) in DeepSeek-R1-Distill-Llama-8B. Chains-of-thought are generated on ~30 hard math word problems, sentences are labeled by keyword matching within <think> blocks, residual-stream activations at a middle layer are extracted via forward hooks, and mean-difference steering vectors are applied at inference at coefficients {-1, 0, +1} (and evaluated at {-2, 0, +2} for behavior frequency). Behavior keyword frequency and task accuracy are measured.

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
