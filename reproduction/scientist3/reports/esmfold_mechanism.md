# Reproduction Report: `esmfold_mechanism`

- **Category:** `science`
- **Experiment directory:** `scientist3/science/esmfold_mechanism`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: A Two-Stage Mechanism Inside ESMFold's Folding Trunk

**Pipeline status:** completed (stages 1–4 + finalization markers)

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 7 (good: 3, buggy: 4, best_solutions: 1)
- `stage_2_baseline_tuning_1_first_attempt` — nodes: 4 (good: 4, buggy: 0, best_solutions: 1)
- `stage_3_creative_research_1_first_attempt` — nodes: 4 (good: 2, buggy: 2, best_solutions: 1)
- `stage_4_ablation_studies_1_first_attempt` — nodes: 4 (good: 2, buggy: 2, best_solutions: 1)

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

> This baseline establishes the core measurement infrastructure (structure prediction, PDB caching, DSSP-based SS assignment, hairpin regex detection) required for the downstream mechanistic decomposition of how ESMFold's folding trunk (48 blocks) encodes β-hairpin decisions. Without a functioning end-to-end pipeline and a nontrivial baseline metric on wild-type inputs, subsequent per-block causal ablations and seq2pair/pair2seq pathway analyses cannot be meaningfully interpreted.

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/3). — ***Folding decisions for β-hairpin structures are localized in the early blocks o…* |
| 2 | See overall verdict (claim 2/3). — ***The early-block seq2pair pathway is the critical channel through which the β-h…* |
| 3 | See overall verdict (claim 3/3). — ***Charge is a linearly encoded chemical feature in early blocks of ESMFold and c…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/science/esmfold_mechanism/idea.json` |
| Human-readable idea | `scientist3/science/esmfold_mechanism/idea.md` |
| BFTS config copy | `scientist3/science/esmfold_mechanism/bfts_config.yaml` |
| Tree visualization | `scientist3/science/esmfold_mechanism/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (48 files) | `scientist3/science/esmfold_mechanism/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [✓] | `scientist3/science/esmfold_mechanism/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [✓] | `scientist3/science/esmfold_mechanism/logs/0-run/research_summary.json` |
| Baseline tuning summary [✓] | `scientist3/science/esmfold_mechanism/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/science/esmfold_mechanism/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> **Folding decisions for β-hairpin structures are localized in the early blocks of the folding trunk.** Whether the trunk will fold a target region into a β-hairpin is committed to within early blocks, and the sequence representation `s` is the active locus of that decision during this window.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Baseline ESMFold-based β-hairpin evaluation pipeline. The node predicts structures for a hardcoded panel of well-characterized β-hairpin-containing proteins plus a Trp-cage negative control, assigns secondary structure via pydssp/mkdssp/biotite cascade, and computes a hairpin DSSP accuracy metric using regex matching (E{3,}L{2,8}E{3,}) within annotated target regions. This is a first sub-stage to establish the baseline metric on wild-type sequences before performing localization/seq2pair/charge ablations.

### Claim 2

**Original claim (verbatim from task.md):**

> **The early-block seq2pair pathway is the critical channel through which the β-hairpin folding decision is transferred from `s` into `z`.** In the early blocks, the **seq2pair** operation carries the decision to fold a target region into a β-hairpin from the sequence representation `s` into the pairwise representation `z`.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Baseline ESMFold-based β-hairpin evaluation pipeline. The node predicts structures for a hardcoded panel of well-characterized β-hairpin-containing proteins plus a Trp-cage negative control, assigns secondary structure via pydssp/mkdssp/biotite cascade, and computes a hairpin DSSP accuracy metric using regex matching (E{3,}L{2,8}E{3,}) within annotated target regions. This is a first sub-stage to establish the baseline metric on wild-type sequences before performing localization/seq2pair/charge ablations.

### Claim 3

**Original claim (verbatim from task.md):**

> **Charge is a linearly encoded chemical feature in early blocks of ESMFold and causally influences β-hairpin formation.** This charge feature exerts a causal effect on β-hairpin formation, consistent with the physical principle that opposite-charge residues on facing β-strands favor pairing (same-charge configurations correspondingly increase cross-strand distance).The evaluation of β-hairpin formation must be based on DSSP secondary-structure assignment on the predicted structure.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Baseline ESMFold-based β-hairpin evaluation pipeline. The node predicts structures for a hardcoded panel of well-characterized β-hairpin-containing proteins plus a Trp-cage negative control, assigns secondary structure via pydssp/mkdssp/biotite cascade, and computes a hairpin DSSP accuracy metric using regex matching (E{3,}L{2,8}E{3,}) within annotated target regions. This is a first sub-stage to establish the baseline metric on wild-type sequences before performing localization/seq2pair/charge ablations.

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
