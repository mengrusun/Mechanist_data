# Reproduction Report: `emotion_prompts`

- **Category:** `emotion`
- **Experiment directory:** `scientist3/emotion/emotion_prompts`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Emotional Framing in Prompts as a Weak, Input-Dependent Signal

**Pipeline status:** incomplete (stages [1, 2, 3], no finalization; likely killed mid-run)

> ⚠ This run was terminated before natural completion (dmxapi hang / BFTS internal stall / manual kill). Artifacts up to the last completed BFTS node are preserved; final aggregate + token_tracker markers are missing.

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 4 (good: 3, buggy: 1, best_solutions: 1)
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
| 1 | See overall verdict (claim 1/4). — *Across diverse task domains, static emotional prefixes change LLM accuracy only …* |
| 2 | See overall verdict (claim 2/4). — *The effect of emotional framing is most pronounced on socially grounded tasks (i…* |
| 3 | See overall verdict (claim 3/4). — *No single basic emotion among {happiness, sadness, fear, anger, disgust, surpris…* |
| 4 | See overall verdict (claim 4/4). — *An adaptive policy that selects the emotional prefix per query (EmotionRL) yield…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/emotion/emotion_prompts/idea.json` |
| Human-readable idea | `scientist3/emotion/emotion_prompts/idea.md` |
| BFTS config copy | `scientist3/emotion/emotion_prompts/bfts_config.yaml` |
| Tree visualization | `scientist3/emotion/emotion_prompts/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (41 files) | `scientist3/emotion/emotion_prompts/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [—] | `scientist3/emotion/emotion_prompts/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [—] | `scientist3/emotion/emotion_prompts/logs/0-run/research_summary.json` |
| Baseline tuning summary [—] | `scientist3/emotion/emotion_prompts/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/emotion/emotion_prompts/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: —, auto_plot_aggregator.py: — |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> Across diverse task domains, static emotional prefixes change LLM accuracy only by small, input-dependent amounts and do not behave as a general-purpose enhancement.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

### Claim 2

**Original claim (verbatim from task.md):**

> The effect of emotional framing is most pronounced on socially grounded tasks (interpersonal/social reasoning), where the emotional context interacts meaningfully with task content; on tasks like mathematical reasoning or factual QA the effect is markedly smaller.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

### Claim 3

**Original claim (verbatim from task.md):**

> No single basic emotion among {happiness, sadness, fear, anger, disgust, surprise} provides a consistent benefit across all models and tasks; stronger emotional wording does not yield proportionally larger gains.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

### Claim 4

**Original claim (verbatim from task.md):**

> An adaptive policy that selects the emotional prefix per query (EmotionRL) yields more reliable accuracy gains than any fixed emotional prefix or neutral baseline.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> _No draft_summary.json — inspect stage journals and experiment_results._

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
