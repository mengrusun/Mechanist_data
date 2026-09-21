# Reproduction Report: `alignet_visual`

- **Category:** `multimodal`
- **Experiment directory:** `scientist3/multimodal/alignet_visual`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Hierarchical Human Alignment of Vision Foundation Models

**Pipeline status:** completed (stages 1–4 + finalization markers)

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 12 (good: 3, buggy: 9, best_solutions: 1)
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

> This preliminary run establishes a working end-to-end pipeline but reveals that the alignment procedure produces essentially no measurable improvement over baseline in this initial configuration. Identifying that the alignment loss magnitude (~0.01-0.03) is dwarfed by the total loss (~3.25-3.55) provides a concrete diagnosis for future stages: the loss weighting must be recalibrated for distillation to have any effect. It also confirms the teacher (SigLIP) offers a modest ceiling above the student (DINOv2), justifying distillation as a research direction.

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/4). — *A teacher model fitted to human triplet-similarity judgments on THINGS captures …* |
| 2 | See overall verdict (claim 2/4). — *Distilling this human-like similarity structure into pretrained vision foundatio…* |
| 3 | See overall verdict (claim 3/4). — *After alignment, the fine-tuned student models more accurately reproduce human b…* |
| 4 | See overall verdict (claim 4/4). — *Alignment is not at odds with utility: aligned student models match or exceed th…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/multimodal/alignet_visual/idea.json` |
| Human-readable idea | `scientist3/multimodal/alignet_visual/idea.md` |
| BFTS config copy | `scientist3/multimodal/alignet_visual/bfts_config.yaml` |
| Tree visualization | `scientist3/multimodal/alignet_visual/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (84 files) | `scientist3/multimodal/alignet_visual/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [✓] | `scientist3/multimodal/alignet_visual/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [✓] | `scientist3/multimodal/alignet_visual/logs/0-run/research_summary.json` |
| Baseline tuning summary [✓] | `scientist3/multimodal/alignet_visual/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/multimodal/alignet_visual/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> A teacher model fitted to human triplet-similarity judgments on THINGS captures human similarity structure well enough that, when applied to a large unlabelled image corpus (ImageNet), it can synthesise a large body of human-like similarity judgments spanning multiple abstraction levels.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Preliminary baseline implementation of AligNet-style representation distillation for THINGS triplet odd-one-out (OOO) alignment. A frozen DINOv2 ViT-B student is aligned to a frozen SigLIP-So400m teacher via a lightweight projection head trained to match pairwise cosine similarities over 1852 THINGS concepts, evaluated on OOO accuracy, one-shot downstream tasks, and ImageNet-A OOD.

### Claim 2

**Original claim (verbatim from task.md):**

> Distilling this human-like similarity structure into pretrained vision foundation models (DINOv2, supervised ViT-L, contrastive image–text models) via a dedicated alignment loss substantially improves their Spearman correlation with human similarity judgments at multiple abstraction levels.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Preliminary baseline implementation of AligNet-style representation distillation for THINGS triplet odd-one-out (OOO) alignment. A frozen DINOv2 ViT-B student is aligned to a frozen SigLIP-So400m teacher via a lightweight projection head trained to match pairwise cosine similarities over 1852 THINGS concepts, evaluated on OOO accuracy, one-shot downstream tasks, and ImageNet-A OOD.

### Claim 3

**Original claim (verbatim from task.md):**

> After alignment, the fine-tuned student models more accurately reproduce human behavioural patterns and uncertainty on similarity tasks than their unaligned counterparts.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Preliminary baseline implementation of AligNet-style representation distillation for THINGS triplet odd-one-out (OOO) alignment. A frozen DINOv2 ViT-B student is aligned to a frozen SigLIP-So400m teacher via a lightweight projection head trained to match pairwise cosine similarities over 1852 THINGS concepts, evaluated on OOO accuracy, one-shot downstream tasks, and ImageNet-A OOD.

### Claim 4

**Original claim (verbatim from task.md):**

> Alignment is not at odds with utility: aligned student models match or exceed the originals on diverse downstream tasks and improve out-of-distribution robustness.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Preliminary baseline implementation of AligNet-style representation distillation for THINGS triplet odd-one-out (OOO) alignment. A frozen DINOv2 ViT-B student is aligned to a frozen SigLIP-So400m teacher via a lightweight projection head trained to match pairwise cosine similarities over 1852 THINGS concepts, evaluated on OOO accuracy, one-shot downstream tasks, and ImageNet-A OOD.

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
