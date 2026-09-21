# Reproduction Report: `interplm`

- **Category:** `science`
- **Experiment directory:** `scientist3/science/interplm`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Sparse Autoencoders Recover Biologically Interpretable Features Inside Protein Language Models

**Pipeline status:** completed (stages 1–4 + finalization markers)

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 15 (good: 3, buggy: 12, best_solutions: 1)
- `stage_2_baseline_tuning_1_first_attempt` — nodes: 4 (good: 3, buggy: 1, best_solutions: 1)
- `stage_3_creative_research_1_first_attempt` — nodes: 4 (good: 3, buggy: 1, best_solutions: 1)
- `stage_4_ablation_studies_1_first_attempt` — nodes: 4 (good: 3, buggy: 1, best_solutions: 1)

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

> This is the seed baseline establishing whether SAE features learned on top of a protein language model provide better interpretability (via concept alignment) than raw neuron activations. Results here justify subsequent SAE-focused interpretability work by showing SAE features recover substantially more biological concepts than raw neurons, mirroring findings from InterPLM at a smaller scale (ESM-2-8M).

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/5). — *SAEs trained on the residual-stream activations of ESM-2 layers surface up to ~2…* |
| 2 | See overall verdict (claim 2/5). — *These SAE features align with up to ~143 distinct Swiss-Prot biological concepts…* |
| 3 | See overall verdict (claim 3/5). — *The gap between SAE features and raw neurons is direct evidence that PLMs encode…* |
| 4 | See overall verdict (claim 4/5). — *A subset of the SAE features corresponds to coherent biological concepts that ar…* |
| 5 | See overall verdict (claim 5/5). — *The extracted feature dictionary is practically useful: it supports filling in m…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/science/interplm/idea.json` |
| Human-readable idea | `scientist3/science/interplm/idea.md` |
| BFTS config copy | `scientist3/science/interplm/bfts_config.yaml` |
| Tree visualization | `scientist3/science/interplm/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (70 files) | `scientist3/science/interplm/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [✓] | `scientist3/science/interplm/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [✓] | `scientist3/science/interplm/logs/0-run/research_summary.json` |
| Baseline tuning summary [✓] | `scientist3/science/interplm/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/science/interplm/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> SAEs trained on the residual-stream activations of ESM-2 layers surface up to ~2,548 interpretable latent features per layer — orders of magnitude more concepts than can be pulled out of individual neurons.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Baseline comparison of InterPLM Sparse Autoencoder (SAE) features against raw ESM-2-8M neurons for detecting protein biological concepts (BINDING, ACT_SITE, MOTIF, DOMAIN, MOD_RES annotations) at the residue level using a Swiss-Prot subset (~200 proteins).

### Claim 2

**Original claim (verbatim from task.md):**

> These SAE features align with up to ~143 distinct Swiss-Prot biological concepts (binding sites, active sites, sequence motifs, structural / functional domains), whereas raw ESM-2 neurons align with only ~46 concepts on the same evaluation, of which only ~15 are cleanly recovered.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Baseline comparison of InterPLM Sparse Autoencoder (SAE) features against raw ESM-2-8M neurons for detecting protein biological concepts (BINDING, ACT_SITE, MOTIF, DOMAIN, MOD_RES annotations) at the residue level using a Swiss-Prot subset (~200 proteins).

### Claim 3

**Original claim (verbatim from task.md):**

> The gap between SAE features and raw neurons is direct evidence that PLMs encode biological concepts in superposition rather than in single units.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Baseline comparison of InterPLM Sparse Autoencoder (SAE) features against raw ESM-2-8M neurons for detecting protein biological concepts (BINDING, ACT_SITE, MOTIF, DOMAIN, MOD_RES annotations) at the residue level using a Swiss-Prot subset (~200 proteins).

### Claim 4

**Original claim (verbatim from task.md):**

> A subset of the SAE features corresponds to coherent biological concepts that are absent from existing annotation dictionaries; these can be surfaced by using an external LLM as an auto-interpreter over top-activating protein contexts.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Baseline comparison of InterPLM Sparse Autoencoder (SAE) features against raw ESM-2-8M neurons for detecting protein biological concepts (BINDING, ACT_SITE, MOTIF, DOMAIN, MOD_RES annotations) at the residue level using a Swiss-Prot subset (~200 proteins).

### Claim 5

**Original claim (verbatim from task.md):**

> The extracted feature dictionary is practically useful: it supports filling in missing Swiss-Prot annotations and steering ESM-2 sequence generation toward a target biological property.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Baseline comparison of InterPLM Sparse Autoencoder (SAE) features against raw ESM-2-8M neurons for detecting protein biological concepts (BINDING, ACT_SITE, MOTIF, DOMAIN, MOD_RES annotations) at the residue level using a Swiss-Prot subset (~200 proteins).

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
