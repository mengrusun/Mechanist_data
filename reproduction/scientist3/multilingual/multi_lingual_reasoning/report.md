# Reproduction Report: `multi_lingual_reasoning`

- **Category:** `multilingual`
- **Experiment directory:** `scientist3/multilingual/multi_lingual_reasoning`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Disentangling Language and Reasoning in LLM Internal Representations

**Pipeline status:** incomplete (stages [1, 2, 3], no finalization; likely killed mid-run)

> ⚠ This run was terminated before natural completion (dmxapi hang / BFTS internal stall / manual kill). Artifacts up to the last completed BFTS node are preserved; final aggregate + token_tracker markers are missing.

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 6 (good: 3, buggy: 3, best_solutions: 1)
- `stage_2_baseline_tuning_1_first_attempt` — nodes: 4 (good: 4, buggy: 0, best_solutions: 1)
- `stage_3_creative_research_1_first_attempt` — nodes: 1 (good: 1, buggy: 0, best_solutions: 1)

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

**Not assessable via automation** — pipeline stopped before finalization; reports below list what was actually produced (best_solutions, exp_results) so a human can still eyeball claim support.

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/4). — *For a given LLM, hidden representations of multilingual reasoning inputs decompo…* |
| 2 | See overall verdict (claim 2/4). — *Suppressing the language-specific subspace from internal representations at infe…* |
| 3 | See overall verdict (claim 3/4). — *The strength of language-specific activation is negatively correlated with reaso…* |
| 4 | See overall verdict (claim 4/4). — *This training-free intervention matches or exceeds multilingual post-training (s…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/multilingual/multi_lingual_reasoning/idea.json` |
| Human-readable idea | `scientist3/multilingual/multi_lingual_reasoning/idea.md` |
| BFTS config copy | `scientist3/multilingual/multi_lingual_reasoning/bfts_config.yaml` |
| Tree visualization | `scientist3/multilingual/multi_lingual_reasoning/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (56 files) | `scientist3/multilingual/multi_lingual_reasoning/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [—] | `scientist3/multilingual/multi_lingual_reasoning/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [—] | `scientist3/multilingual/multi_lingual_reasoning/logs/0-run/research_summary.json` |
| Baseline tuning summary [—] | `scientist3/multilingual/multi_lingual_reasoning/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/multilingual/multi_lingual_reasoning/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: —, auto_plot_aggregator.py: — |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> For a given LLM, hidden representations of multilingual reasoning inputs decompose into a language-specific subspace and an approximately orthogonal language-agnostic subspace, and the language-specific subspace can be identified from a small multilingual probe set.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

### Claim 2

**Original claim (verbatim from task.md):**

> Suppressing the language-specific subspace from internal representations at inference time consistently improves multilingual reasoning accuracy across diverse models, languages, and reasoning tasks, while output language fidelity remains acceptable when upper layers are left intact.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

### Claim 3

**Original claim (verbatim from task.md):**

> The strength of language-specific activation is negatively correlated with reasoning accuracy: amplifying it degrades reasoning, removing it improves reasoning.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

### Claim 4

**Original claim (verbatim from task.md):**

> This training-free intervention matches or exceeds multilingual post-training (supervised fine-tuning, reinforcement learning) at a small fraction of the compute.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
