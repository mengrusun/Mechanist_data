# Reproduction Report: `multi_agent`

- **Category:** `multi-agent_safety`
- **Experiment directory:** `scientist3/multi-agent_safety/multi_agent`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Group-Aggregated Activation Probes for Multi-Agent Collusion Detection

**Pipeline status:** completed (stages 1–4 + finalization markers)

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 14 (good: 3, buggy: 11, best_solutions: 1)
- `stage_2_baseline_tuning_1_first_attempt` — nodes: 4 (good: 4, buggy: 0, best_solutions: 1)
- `stage_3_creative_research_1_first_attempt` — nodes: 4 (good: 3, buggy: 1, best_solutions: 1)
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

### Agent summary excerpt (`Significance` from draft_summary.json)

> This is a foundational sanity check for the paper's central claim that internal activations reveal covert coordination invisible to text-level features. The result — that a simple TF-IDF baseline matches probe performance at ceiling (1.0 AUROC) — indicates the current synthetic dataset lacks the lexical-cue-minimization needed to make the comparison scientifically meaningful. This drives an important design requirement for subsequent stages: either use adversarially controlled scenarios where text features are demonstrably insufficient, or elicit real behavioral diversity (currently the committee is stuck in ~100% unanimous-approve mode).

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/3). — *Model internals carry a collusion signal that text-level monitoring misses; ligh…* |
| 2 | See overall verdict (claim 2/3). — *Detection requires aggregating signals across agents — group-aggregated probes o…* |
| 3 | See overall verdict (claim 3/3). — *Activation probes transfer zero-shot, without retraining, to structurally differ…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/multi-agent_safety/multi_agent/idea.json` |
| Human-readable idea | `scientist3/multi-agent_safety/multi_agent/idea.md` |
| BFTS config copy | `scientist3/multi-agent_safety/multi_agent/bfts_config.yaml` |
| Tree visualization | `scientist3/multi-agent_safety/multi_agent/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (91 files) | `scientist3/multi-agent_safety/multi_agent/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [✓] | `scientist3/multi-agent_safety/multi_agent/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [✓] | `scientist3/multi-agent_safety/multi_agent/logs/0-run/research_summary.json` |
| Baseline tuning summary [✓] | `scientist3/multi-agent_safety/multi_agent/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/multi-agent_safety/multi_agent/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> Model internals carry a collusion signal that text-level monitoring misses; lightweight probes on per-agent residual-stream activations can recover it.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal end-to-end pipeline for LLM-committee collusion detection: loading Qwen3-32B-AWQ (bypassing transformers' AWQ quantizer via AutoAWQForCausalLM.from_quantized), generating a synthetic NARCBench-Core-like dataset of committee deliberations across 17 domains, extracting per-agent residual-stream activations from a middle layer, and training logistic-regression probes over multiple aggregations (per-agent, mean-pool, max-pool, concat) alongside TF-IDF and random-feature control baselines using 5-fold stratified cross-validation.

### Claim 2

**Original claim (verbatim from task.md):**

> Detection requires aggregating signals across agents — group-aggregated probes outperform any single-agent probe and the text-only judge — and no single aggregation technique dominates across collusion types, motivating a family of complementary multi-agent probes.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal end-to-end pipeline for LLM-committee collusion detection: loading Qwen3-32B-AWQ (bypassing transformers' AWQ quantizer via AutoAWQForCausalLM.from_quantized), generating a synthetic NARCBench-Core-like dataset of committee deliberations across 17 domains, extracting per-agent residual-stream activations from a middle layer, and training logistic-regression probes over multiple aggregations (per-agent, mean-pool, max-pool, concat) alongside TF-IDF and random-feature control baselines using 5-fold stratified cross-validation.

### Claim 3

**Original claim (verbatim from task.md):**

> Activation probes transfer zero-shot, without retraining, to structurally different multi-agent scenarios (different channel, different domain), showing the learned signal is not tied to one collusion form.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal end-to-end pipeline for LLM-committee collusion detection: loading Qwen3-32B-AWQ (bypassing transformers' AWQ quantizer via AutoAWQForCausalLM.from_quantized), generating a synthetic NARCBench-Core-like dataset of committee deliberations across 17 domains, extracting per-agent residual-stream activations from a middle layer, and training logistic-regression probes over multiple aggregations (per-agent, mean-pool, max-pool, concat) alongside TF-IDF and random-feature control baselines using 5-fold stratified cross-validation.

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
