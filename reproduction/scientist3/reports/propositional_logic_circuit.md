# Reproduction Report: `propositional_logic_circuit`

- **Category:** `reasoning`
- **Experiment directory:** `scientist3/reasoning/propositional_logic_circuit`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: A Sparse, Modular Circuit Implements Propositional-Logic Reasoning in LLMs

**Pipeline status:** completed (stages 1–4 + finalization markers)

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 4 (good: 3, buggy: 1, best_solutions: 1)
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

> This seed experiment establishes an end-to-end pipeline for mechanistic interpretability on propositional logic reasoning. It reveals several important methodological concerns (weak corruption signal, distributed representations resistant to single-head/single-layer interventions) that must be addressed before circuit discovery can succeed. The layer-wise logit lens localization suggests answer formation happens in late layers (~25-32), giving a useful scope narrowing for future patching work.

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/3). — *A sparse subset of specific attention heads and MLP components jointly implement…* |
| 2 | See overall verdict (claim 2/3). — *The circuit decomposes into a small number of modular sub-circuits with distinct…* |
| 3 | See overall verdict (claim 3/3). — *Activation-patching / causal-mediation experiments show that the identified comp…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/reasoning/propositional_logic_circuit/idea.json` |
| Human-readable idea | `scientist3/reasoning/propositional_logic_circuit/idea.md` |
| BFTS config copy | `scientist3/reasoning/propositional_logic_circuit/bfts_config.yaml` |
| Tree visualization | `scientist3/reasoning/propositional_logic_circuit/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (109 files) | `scientist3/reasoning/propositional_logic_circuit/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [✓] | `scientist3/reasoning/propositional_logic_circuit/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [✓] | `scientist3/reasoning/propositional_logic_circuit/logs/0-run/research_summary.json` |
| Baseline tuning summary [✓] | `scientist3/reasoning/propositional_logic_circuit/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/reasoning/propositional_logic_circuit/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> A sparse subset of specific attention heads and MLP components jointly implements the minimal propositional-logic reasoning task — the circuit is small relative to the full model.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Baseline zero-shot evaluation of Mistral-7B on a synthetic propositional-logic reasoning task (forward chaining with facts and if-then rules) to establish infrastructure for downstream mechanistic circuit analysis. In addition to accuracy measurement, the pipeline includes preliminary counterfactual (corrupted prompt) evaluation, logit-lens layer analysis, single-head attention knockouts, and activation patching across residual/attention/MLP components.

### Claim 2

**Original claim (verbatim from task.md):**

> The circuit decomposes into a small number of modular sub-circuits with distinct functional roles (e.g., identifying the facts in the prompt, applying an implication rule, projecting the derived truth value into the answer token) rather than presenting as an entangled mixture.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Baseline zero-shot evaluation of Mistral-7B on a synthetic propositional-logic reasoning task (forward chaining with facts and if-then rules) to establish infrastructure for downstream mechanistic circuit analysis. In addition to accuracy measurement, the pipeline includes preliminary counterfactual (corrupted prompt) evaluation, logit-lens layer analysis, single-head attention knockouts, and activation patching across residual/attention/MLP components.

### Claim 3

**Original claim (verbatim from task.md):**

> Activation-patching / causal-mediation experiments show that the identified components are both necessary (patching them from a clean run into a corrupted run restores correct behaviour) and sufficient (their outputs alone drive the answer).

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Baseline zero-shot evaluation of Mistral-7B on a synthetic propositional-logic reasoning task (forward chaining with facts and if-then rules) to establish infrastructure for downstream mechanistic circuit analysis. In addition to accuracy measurement, the pipeline includes preliminary counterfactual (corrupted prompt) evaluation, logit-lens layer analysis, single-head attention knockouts, and activation patching across residual/attention/MLP components.

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
