# Captured-Behavior Report

**Direction**: (empty — task.md at the project root is the sole authoritative direction source)
**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: task.md (faithful capture)
**Date**: 2026-07-14
**Pipeline**: research-lit → faithful behavior capture (task.md) → research-refine-pipeline
**Language**: English

## Executive Summary

Four behavioral claims are captured verbatim from `task.md` (the "Research Hypothesis: Hierarchical Human Alignment of Vision Foundation Models" doc) and turned into individually verifiable predicates without altering their meaning. Together they describe a THINGS-fit surrogate similarity teacher (SigLIP-So400m image-encoder, shaped by human odd-one-out triplets), that teacher's hierarchical pseudo-labels on unlabelled ImageNet, and the effect of distilling those pseudo-labels into a pretrained DINOv2 ViT-B student on (a) alignment with human similarity across coarse/mid/fine levels, (b) reproduction of human behavioural + uncertainty patterns, and (c) downstream utility + OOD robustness. All four claims are pending verification; the unified verification approach and per-claim milestones live in `refine-logs/FINAL_PROPOSAL.md` and `refine-logs/EXPERIMENT_PLAN.md`. Since `BEHAVIOR_SOURCE=given`, no ideation / novelty check / impact check / M0 phenomenon-validation gate is run — the behaviour is *assumed* to hold; the reproduction tests whether the four claims actually reproduce under the given resources.

## Literature Landscape

See `idea-stage/LANDSCAPE.md` for the full landscape. Highlights that shape verification design (not the claim itself):

- Post-hoc **linear-transform** alignment (Attarian et al. 2020; Muttenthaler et al. 2023) is the natural pre-distillation baseline for Claim 2 gains. Naive alignment can *hurt* local structure → Claim 4 downstream/OOD checks are load-bearing, not decorative.
- Muttenthaler et al. 2022 (arXiv:2211.01201) shows model scale/architecture barely affect human-similarity alignment; data + objective dominate. This is the "unaligned baseline" reference for Claim 2.
- KD-as-label-smoothing (Yuan et al. 2019, arXiv:1909.11723) warns that any similarity-KD gain is partially attributable to soft-label regularization — the verification suite therefore needs a **non-human-aligned soft-label control** (random-triplet teacher / unaligned SigLIP teacher / plain label smoothing) so Claim 2 gains are specifically attributable to the human-alignment mechanism.
- Triplet-metric KD (Oki et al. 2020, arXiv:2004.08116) and CLIP-teacher distillation (CLIP-TD, arXiv:2201.05729) supply general mechanical scaffolding for large-teacher → small-student similarity distillation.

**Not consulted:** the specific paper being reproduced is on the project's forbidden list (`.claude/forbidden-urls.txt`) and has not been read, cited, or paraphrased. `task.md`'s wording is the sole authority for what is claimed.

## Global Resources (from task.md — apply to every claim unless overridden)

- **Teacher (fixed across the whole project)**: SigLIP-So400m image-encoder, shaped by human triplet odd-one-out judgments on THINGS.
- **Student for the main experiment**: DINOv2 ViT-B (representative student vision foundation model; do NOT swap in the main experiment — swaps live in the verify stage).
- **Data (main experiment)**: THINGS (1,854 natural object concepts) + associated human triplet odd-one-out judgments — used both to fit the teacher and to evaluate alignment.
- **Data (verify stage, informational)**: ImageNet (ILSVRC-2012) as unlabelled source for teacher pseudo-labels; "Levels" novel hierarchical eval (coarse / fine / class-boundary); public human-similarity-judgment collection for RSA behaviour/uncertainty; 10 one-shot downstream classification datasets (Birds, UC Merced, Colon + 7 fine-grained specialty); BREEDS (entity13 / living17 / non-living26 / entity30) for subpopulation shift; ImageNet-A for OOD/natural-adversarial.
- **Verify-stage candidate student swaps (informational)**: Supervised ViT-S / ViT-B / ViT-L; DINOv1 ViT-B; SigLIP ViT-B; CapPa ViT-B.
- **Constraints**: 10 hr total GPU budget; GPU ids ∈ {0,1,2,3} only; directory access limited to project working dir, `/data/zhenqian/data`, `/data/zhenqian/models`; conda env; use symbolic links (do not copy); missing assets downloaded to those roots from HF / GitHub / ModelScope.
- **Note on `resource_fidelity`**: this run is `BEHAVIOR_SOURCE=given` + `MECHANISM=discovery` (not the reproduction combo `given + given`) → `resource_fidelity` **NOT stamped `strict`** at plan level. The named model + teacher + main-experiment dataset are captured as user-preferred defaults (still not gratuitously downscaled), but Power-Fidelity is cost-aware, per `/auto-claim` semantics.

## Claims to Verify

### Claim 1: THINGS-fit teacher yields hierarchical human-like similarity on unlabelled images

**Original (verbatim from task.md §Claim, bullet 1):**
> A teacher model fitted to human triplet-similarity judgments on THINGS captures human similarity structure well enough that, when applied to a large unlabelled image corpus (ImageNet), it can synthesise a large body of human-like similarity judgments spanning multiple abstraction levels.

**Extracted statement**: When SigLIP-So400m is fit to human odd-one-out triplet judgments on THINGS, its image-encoder-space similarities on THINGS agree with held-out human triplets, and the same teacher, applied over unlabelled ImageNet, produces per-triplet soft judgments that are systematically distinguishable across coarse (super-category), mid (basic-level), and fine (sub-category) abstraction levels.

**Hypothesis**: H1 — A THINGS-fit similarity teacher generalizes off-THINGS onto ImageNet in a way that preserves multi-level (coarse/mid/fine) human-like similarity structure, not just a single average agreement.

**Measurable predicate**:
- (i) Teacher's held-out THINGS triplet accuracy (predicting the human odd-one-out) is substantially above unaligned SigLIP-So400m and above chance.
- (ii) When applied to ImageNet-derived triplets whose within-triplet distance separations are constructed at coarse / mid / fine levels (e.g. same-super-category vs. cross-super-category, same-mid vs. cross-mid, same-fine vs. cross-fine), the teacher's synthesized triplet-choice distributions systematically distinguish these three levels (level-order preserved; separation strictly positive at each level, larger at finer levels or per the natural ordering).

**Expected direction**: teacher_triplet_accuracy(THINGS held-out) > unaligned_SigLIP_baseline; ImageNet-hierarchical-triplet separation is level-monotonic and strictly positive at each of the three levels.

**Resources (main-experiment binding)**: teacher = SigLIP-So400m (fixed); dataset = THINGS + human triplet odd-one-out judgments; ImageNet (ILSVRC-2012) as the unlabelled source for synthesized triplets. `used_n` for held-out THINGS eval: use the full held-out human-triplet split as task.md's evaluation resource specifies; for ImageNet-synthesized-triplet hierarchical eval: sample ≥ 3,000 triplets stratified across the three level buckets (≥ 1,000 per level) — sized to clear noise on a 10-hr budget while satisfying `/data-rule` sample-size floors.

**Status**: pending verification
**Notes**: Two logically separate sub-predicates are unified under one claim because task.md states them as a single "well enough that … it can synthesise …" clause. Split at experiment-plan level into two milestones (teacher-fit quality + hierarchical pseudo-label quality) but tracked under Claim 1.

---

### Claim 2: Distillation into pretrained backbones substantially improves multi-level Spearman with human similarity

**Original (verbatim from task.md §Claim, bullet 2):**
> Distilling this human-like similarity structure into pretrained vision foundation models (DINOv2, supervised ViT-L, contrastive image–text models) via a dedicated alignment loss substantially improves their Spearman correlation with human similarity judgments at multiple abstraction levels.

**Extracted statement**: Applying a dedicated alignment loss that distills the teacher's synthesized similarity onto pretrained vision backbones — with DINOv2 ViT-B as the main-experiment student — yields aligned students whose embedding-space similarities show substantially higher Spearman rank correlation with human similarity judgments than the corresponding unaligned pretrained backbone, and this improvement holds at each of coarse, mid, and fine abstraction levels (not only in aggregate).

**Hypothesis**: H2 — Distillation with a similarity-alignment loss transfers hierarchical human-similarity structure into a pretrained backbone, in a way that is (a) present at every abstraction level and (b) not attributable to generic soft-label regularization (i.e. it disappears or shrinks substantially under a human-alignment-free control teacher).

**Measurable predicate**:
- (i) Spearman(aligned_DINOv2_ViT-B embedding similarity, human triplet similarity) > Spearman(unaligned_DINOv2_ViT-B, humans) on THINGS held-out human triplets, with the gap "substantially" positive (task.md's wording — operationalized as the aligned score exceeding the unaligned score by an effect size that is both statistically significant at α=0.05 by paired bootstrap over triplets and practically meaningful — Δρ ≥ 0.05 on the aggregate is the pre-registered threshold, subject to power under the 10-hr budget).
- (ii) The gap is positive at each of coarse / mid / fine levels evaluated separately (level-wise Spearman gain > 0 at all three levels).
- (iii) The gain is *specific to human-alignment*: the aligned student outperforms a "unaligned-soft-label" control student trained with the same loss/schedule but a non-human-aligned teacher (either unaligned SigLIP-So400m image-encoder or a random-permutation teacher on the same triplets) by a positive margin on the same metric.

**Expected direction**: aligned > unaligned baseline at all levels; aligned > unaligned-soft-label control.

**Resources (main-experiment binding)**: student = DINOv2 ViT-B (fixed for the main experiment; supervised ViT-L / contrastive-image-text students belong to verify stage); teacher = SigLIP-So400m (fixed); train set = ImageNet-unlabelled + teacher-synthesized similarity signal; eval set = THINGS held-out human triplet judgments (multi-level breakdown per Claim 1's ImageNet-hierarchical construction, and on any "Levels" hierarchical eval available in the verify-stage resource list); used_n(train) = sized to fit inside the 10-hr budget (target: single DINOv2 ViT-B alignment finetune completes in ≤ 2.5–3 hr on 4× GPU 0/1/2/3, ImageNet subset ≥ 100k images); used_n(eval) = held-out THINGS triplets in full.

**Status**: pending verification
**Notes**: task.md's "DINOv2, supervised ViT-L, contrastive image–text models" phrasing enumerates the backbone families that were originally tested. Since task.md's `**Experiment stage**` block pins the main-experiment student to DINOv2 ViT-B, we test Claim 2 on DINOv2 ViT-B in the main run and record supervised ViT-L / contrastive-image-text students as verify-stage swap variants (the verify agent decides which of them to run). The "substantially" wording is preserved verbatim (see Original quote); we operationalize it as a pre-registered effect-size + significance threshold on the aggregate + per-level Spearman, not "any positive gain".

---

### Claim 3: Aligned students better reproduce human behavioural patterns AND per-triplet uncertainty

**Original (verbatim from task.md §Claim, bullet 3):**
> After alignment, the fine-tuned student models more accurately reproduce human behavioural patterns and uncertainty on similarity tasks than their unaligned counterparts.

**Extracted statement**: On similarity tasks (triplet odd-one-out on THINGS held-out and any RSA public human-similarity-judgment collection), the aligned DINOv2 ViT-B (a) matches the *human choice distribution* per triplet more closely than the unaligned baseline (i.e. picks the same odd-one-out as humans more often, and reproduces per-triplet ambiguity — high-uncertainty triplets where humans split — more faithfully), and (b) matches human *uncertainty* per triplet: the model's confidence / softmax over the three "odd-one-out" candidates tracks human disagreement (higher confidence where humans agree, lower where humans disagree).

**Hypothesis**: H3 — Alignment shifts the student beyond raw rank correlation: its *per-item* prediction distribution better matches the *per-item* human response distribution, both in choice and in graded confidence/uncertainty.

**Measurable predicate**:
- (i) Per-triplet human-vs-model choice agreement (top-1 pick match rate) of aligned > unaligned on THINGS held-out and on the RSA public-judgment collection.
- (ii) Per-triplet uncertainty match — the aligned model's per-triplet confidence (e.g. softmax margin over the three odd-one-out choices, or a calibrated similarity-derived probability) has higher rank-correlation with per-triplet human agreement rate (or lower cross-entropy / lower KL to the human distribution) than the unaligned model's.
- (iii) On an RSA metric (Spearman between aligned-student RDM and human RDM across concepts), aligned > unaligned.

**Expected direction**: choice agreement, uncertainty-calibration correlation, RSA(RDM) all move up for aligned relative to unaligned.

**Resources (main-experiment binding)**: student = DINOv2 ViT-B aligned vs. unaligned; datasets = THINGS held-out human triplet judgments + a public human-similarity-judgment collection listed in the verify-stage resource block (used here as the RSA supplementary set, per task.md's `**Verify stage — verify variants candidates**` description); used_n = full held-out THINGS + full RSA collection.

**Status**: pending verification
**Notes**: task.md explicitly conjoins "behavioural patterns AND uncertainty" — we do NOT drop the uncertainty half to make the test simpler (that would violate claim fidelity). Both must move in the predicted direction for Claim 3 to be judged supported; either one moving alone gets `conditional`.

---

### Claim 4: Alignment preserves or improves downstream utility and OOD robustness

**Original (verbatim from task.md §Claim, bullet 4):**
> Alignment is not at odds with utility: aligned student models match or exceed the originals on diverse downstream tasks and improve out-of-distribution robustness.

**Extracted statement**: The aligned DINOv2 ViT-B *does not lose* utility relative to the unaligned baseline on diverse downstream tasks — measured by few-shot / one-shot classification accuracy on a diverse suite (Birds, UC Merced, Colon + 7 fine-grained specialty datasets) — and *improves* OOD / distribution-shift robustness — measured on BREEDS (entity13 / living17 / non-living26 / entity30) subpopulation shift and on ImageNet-A natural-adversarial.

**Hypothesis**: H4 — The alignment loss preserves the utility of the pretrained backbone (no degradation across a diverse downstream suite) and yields a strictly positive improvement on OOD robustness benchmarks.

**Measurable predicate**:
- (i) Aligned mean one-shot / few-shot accuracy across the 10 downstream classification datasets is ≥ unaligned mean accuracy (non-inferiority test with a pre-registered margin; per-dataset drop no worse than δ = 1 accuracy point without a compensating gain elsewhere).
- (ii) Aligned OOD accuracy > unaligned OOD accuracy on BREEDS (each of entity13 / living17 / non-living26 / entity30) and on ImageNet-A (positive, statistically distinguishable from zero with the standard evaluation protocol for the respective benchmark).

**Expected direction**: downstream utility flat or up; OOD robustness strictly up.

**Resources (main-experiment binding)**: student = DINOv2 ViT-B aligned vs. unaligned; downstream = 10 one-shot classification datasets (Birds, UC Merced, Colon + 7 fine-grained specialty); OOD = BREEDS (all 4 splits) + ImageNet-A; used_n = full standard test set for each benchmark (or, if compute forces a subset under the 10-hr budget, use the same subset for aligned and unaligned so the comparison is paired — never subsample only one side).

**Status**: pending verification
**Notes**: The claim conjoins **two** separate conditions — utility non-degradation AND OOD improvement — both from a single sentence in task.md ("not at odds with utility … and improve OOD robustness"). They are split into two sub-predicates but tracked under Claim 4. Given the 10-hr budget, the experiment plan may downscale the downstream sweep from 10 → a stratified subset (e.g. 4–5 datasets covering natural + fine-grained + medical) — but this is a cost-aware compression, not a scope change; the missing datasets stay on the verify-stage list.

---

## Refined Proposal
- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering all four claims)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim(s) each verifies; encodes all HARD constraints from task.md)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md` (planning-level; the experiment stage updates rows in place)

## Next Steps

- [ ] `/auto-experiment` to implement + deploy the verification suite (Workflow 1.5) — routes mechanism family inline at Phase 1.5 within the `Tuning & Editing → Location → Decision Auditing` chain
- [ ] `/auto-verify` to stress-test admitted claims under model / dataset / method swaps from the verify-stage candidate lists (Workflow 1.75)
- [ ] `/auto-iteration-loop` to iterate the verification suite until reviewer-ready (Workflow 2)
- [ ] Or invoke `/auto` end-to-end for the autonomous claim → experiment → verify → review chain
