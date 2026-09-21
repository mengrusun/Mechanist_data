# Reproduction Report: `sae_agentic_explainer`

- **Category:** `feature_description`
- **Experiment directory:** `scientist3/feature_description/sae_agentic_explainer`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Iterative, Activation-Grounded Explanations for SAE Features in LLMs

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

**Partially supported / inconclusive** — experiments ran but summaries indicate weak or mixed evidence relative to the original claims.

### Agent summary excerpt (`Significance` from draft_summary.json)

> This preliminary experiment establishes an end-to-end working pipeline for iteratively improving mechanistic interpretability explanations of SAE features and provides an initial signal that agentic refinement can outperform one-shot LLM labeling (Neuronpedia). Even if the observed advantage is small and not statistically significant at this scale, the infrastructure and evaluation methodology it provides are prerequisites for larger studies that could impact automated interpretability of large language models.

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | **Partially supported / inconclusive** — experiments ran but summaries indicate weak or mixed evidence relative to the original claims. — *You should build a pipeline around the following basic idea — letting an LLM age…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/feature_description/sae_agentic_explainer/idea.json` |
| Human-readable idea | `scientist3/feature_description/sae_agentic_explainer/idea.md` |
| BFTS config copy | `scientist3/feature_description/sae_agentic_explainer/bfts_config.yaml` |
| Tree visualization | `scientist3/feature_description/sae_agentic_explainer/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (95 files) | `scientist3/feature_description/sae_agentic_explainer/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [✓] | `scientist3/feature_description/sae_agentic_explainer/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [✓] | `scientist3/feature_description/sae_agentic_explainer/logs/0-run/research_summary.json` |
| Baseline tuning summary [✓] | `scientist3/feature_description/sae_agentic_explainer/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/feature_description/sae_agentic_explainer/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> You should build a pipeline around the following basic idea — letting an LLM agent propose several candidate explanations for an SAE feature, query the target LLM+SAE with probe inputs to gather empirical activation feedback, and iteratively accept, reject, or revise those candidates — produces feature explanations that **outperform Neuronpedia** on both **generative accuracy** (success rate of triggering the feature with text written from the explanation) and **predictive accuracy** (correlation between explanation-based predictions and true activations on held-out text), across multiple open-source LLMs and across early-to-late layers.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Initial implementation of a minimal SAGE (SAE-Guided Explanation) pipeline that iteratively refines SAE feature explanations via a propose→probe→revise loop using an LLM agent, benchmarked against Neuronpedia baseline explanations. The pipeline uses Gemma-2-2B with the GemmaScope 16k residual SAE (layer 12), evaluates a small set of features, and measures 'generative accuracy' by having a writer produce probe sentences from each candidate explanation, passing them through Gemma+SAE, and checking whether target feature activation exceeds a calibrated threshold.

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
