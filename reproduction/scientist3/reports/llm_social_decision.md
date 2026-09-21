# Reproduction Report: `llm_social_decision`

- **Category:** `belief`
- **Experiment directory:** `scientist3/belief/llm_social_decision`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Steerable Social-Variable Directions in LLM Decision Making

**Pipeline status:** completed (stages 1–4 + finalization markers)

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 2 (good: 1, buggy: 1, best_solutions: 1)
- `stage_2_baseline_tuning_1_first_attempt` — nodes: 2 (good: 1, buggy: 1, best_solutions: 1)
- `stage_3_creative_research_1_first_attempt` — nodes: 2 (good: 2, buggy: 0, best_solutions: 1)
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

**Requires human review** — automated scan could not classify support level from available summaries.

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/4). — *For each social/contextual variable (e.g., gender, age, instruction framing, mee…* |
| 2 | See overall verdict (claim 2/4). — *A "pure" variable direction, obtained by removing overlapping components with ot…* |
| 3 | See overall verdict (claim 3/4). — *Injecting these directions into the model at inference time causally and substan…* |
| 4 | See overall verdict (claim 4/4). — *The same mechanism supports both directions of intervention — amplifying a varia…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/belief/llm_social_decision/idea.json` |
| Human-readable idea | `scientist3/belief/llm_social_decision/idea.md` |
| BFTS config copy | `scientist3/belief/llm_social_decision/bfts_config.yaml` |
| Tree visualization | `scientist3/belief/llm_social_decision/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (62 files) | `scientist3/belief/llm_social_decision/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [—] | `scientist3/belief/llm_social_decision/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [—] | `scientist3/belief/llm_social_decision/logs/0-run/research_summary.json` |
| Baseline tuning summary [—] | `scientist3/belief/llm_social_decision/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/belief/llm_social_decision/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> For each social/contextual variable (e.g., gender, age, instruction framing, meeting condition), the LLM encodes its influence on decision making as a linearly extractable direction in the residual stream.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

### Claim 2

**Original claim (verbatim from task.md):**

> A "pure" variable direction, obtained by removing overlapping components with other variables, isolates the unique effect of that variable, free of confounding from co-varying factors.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

### Claim 3

**Original claim (verbatim from task.md):**

> Injecting these directions into the model at inference time causally and substantially alters the relationship between the targeted variable and the model's decision output, while leaving other variables' effects intact.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

### Claim 4

**Original claim (verbatim from task.md):**

> The same mechanism supports both directions of intervention — amplifying a variable's effect or attenuating/inverting it — yielding a practical handle for alignment and debiasing of LLM-based social agents.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
