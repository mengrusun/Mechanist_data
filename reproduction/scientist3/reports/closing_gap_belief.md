# Reproduction Report: `closing_gap_belief`

- **Category:** `belief`
- **Experiment directory:** `scientist3/belief/closing_gap_belief`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence

**Pipeline status:** completed (stages 1–4 + finalization markers)

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 5 (good: 3, buggy: 2, best_solutions: 1)
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

**Partially or fully supported** — summaries suggest supportive evidence, but human review against the source paper is still required.

### Agent summary excerpt (`Significance` from draft_summary.json)

> Establishes empirical grounding for the paper's central hypothesis: LLMs are overconfident because their internal 'true correctness' representation lives in a different linear subspace than the direction driving their verbalized confidence, i.e., miscalibration is a readout failure rather than a knowledge deficit. Demonstrating (a) a large calibration gap and (b) near-orthogonal probe directions at mid/late layers motivates future intervention work (e.g., editing the readout direction to close the calibration gap).

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/3). — *Models encode well-calibrated accuracy information in a linearly accessible dire…* |
| 2 | See overall verdict (claim 2/3). — *Models encode verbalized confidence in a linearly accessible direction.* |
| 3 | See overall verdict (claim 3/3). — *But well-calibrated accuracy information and verbalized confidence occupy separa…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/belief/closing_gap_belief/idea.json` |
| Human-readable idea | `scientist3/belief/closing_gap_belief/idea.md` |
| BFTS config copy | `scientist3/belief/closing_gap_belief/bfts_config.yaml` |
| Tree visualization | `scientist3/belief/closing_gap_belief/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (107 files) | `scientist3/belief/closing_gap_belief/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [✓] | `scientist3/belief/closing_gap_belief/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [✓] | `scientist3/belief/closing_gap_belief/logs/0-run/research_summary.json` |
| Baseline tuning summary [✓] | `scientist3/belief/closing_gap_belief/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/belief/closing_gap_belief/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> Models encode well-calibrated accuracy information in a linearly accessible direction.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A preliminary end-to-end pipeline that probes whether verbalized confidence and true correctness are encoded along distinct linear directions in the hidden states of Llama-3.1-8B-Instruct on TriviaQA. The pipeline generates answers with verbalized confidence, caches hidden states at multiple layers (8, 16, 24), trains logistic-regression probes for correctness and ridge probes for verbalized confidence, and measures the orthogonality (1 - |cos|) between the two probe directions relative to random-direction controls.

### Claim 2

**Original claim (verbatim from task.md):**

> Models encode verbalized confidence in a linearly accessible direction.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A preliminary end-to-end pipeline that probes whether verbalized confidence and true correctness are encoded along distinct linear directions in the hidden states of Llama-3.1-8B-Instruct on TriviaQA. The pipeline generates answers with verbalized confidence, caches hidden states at multiple layers (8, 16, 24), trains logistic-regression probes for correctness and ridge probes for verbalized confidence, and measures the orthogonality (1 - |cos|) between the two probe directions relative to random-direction controls.

### Claim 3

**Original claim (verbatim from task.md):**

> But well-calibrated accuracy information and verbalized confidence occupy separate, nearly orthogonal directions. That means the model "knows" when it is likely wrong, but the generation process fails to surface this signal.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A preliminary end-to-end pipeline that probes whether verbalized confidence and true correctness are encoded along distinct linear directions in the hidden states of Llama-3.1-8B-Instruct on TriviaQA. The pipeline generates answers with verbalized confidence, caches hidden states at multiple layers (8, 16, 24), trains logistic-regression probes for correctness and ridge probes for verbalized confidence, and measures the orthogonality (1 - |cos|) between the two probe directions relative to random-direction controls.

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
