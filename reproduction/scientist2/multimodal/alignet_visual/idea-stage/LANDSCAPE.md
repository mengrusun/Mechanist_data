# Landscape: Distilling human-like similarity structure into pretrained vision foundation models

**Date**: 2026-07-14
**Scope**: General prior-art landscape for the *given* research hypothesis in `task.md` — fitting a surrogate teacher to human triplet judgments (THINGS), using it to synthesize hierarchical similarity pseudo-labels on unlabelled ImageNet, and distilling those into pretrained vision backbones (DINOv2 ViT-B, supervised ViT-L, contrastive image-text like CLIP/SigLIP). *Interpreted as*: a reproduction/verification study of the four claims stated in `task.md` — no ideation, no novelty search on the target itself.
**Based on**: 6 retrieved papers plus general background inherited from `task.md`. Complete raw retrieval in `RESEARCH_LIT.md`. **The specific paper being reproduced is on the project's forbidden list (`.claude/forbidden-urls.txt`) and has NOT been read, cited, paraphrased, or characterized here** — the landscape is written to support *how* to verify the given claims, not to derive them.

---

## 1. Structured Paper Table

| Paper | Venue / Year | Method | Key Result | Relevance to Us | Source |
|-------|--------------|--------|------------|-----------------|--------|
| Muttenthaler et al. — *Human alignment of neural network representations* (arXiv:2211.01201) | ICLR-track 2022 | Correlate 32+ vision models with 3 human-similarity-judgment datasets (triplet + arrangement); apply learned linear transforms across datasets | Model **scale/architecture** barely move alignment; **data + objective** dominate. Linear transforms trained on one dataset transfer to others. Some concepts (food, animals) already well-represented; others (royal, sports) are not. | Baseline for our "unaligned" reference: quantifies the pre-alignment gap on THINGS-like tasks. Motivates that scaling alone does not solve alignment → distillation from a human-fit teacher is needed. | arXiv API |
| Muttenthaler et al. — *Improving neural network representations using human similarity judgments* (arXiv:2306.04507) | NeurIPS 2023 | Linear **global-local transform** that aligns global structure with human judgments while preserving local structure | Naive alignment hurts local structure and downstream tasks; global-local transform improves few-shot classification + anomaly detection | Direct precursor to full-backbone distillation. Establishes the utility-preservation constraint that our Claim 4 (downstream utility not sacrificed) will test. | arXiv API |
| Attarian, Roads, Mozer — *Transforming NN visual reps to predict human similarity* (arXiv:2010.06512) | 2020 (NeurIPS wksp track) | Linear + asymmetric transformations on deep embeddings to predict human similarity | 72% → 89% on bird-image binary choice; asymmetric transform helps (classic Tversky-style asymmetry) | Alt. baseline for the *post-hoc linear* family that our *distillation* approach should beat at multiple abstraction levels. | arXiv API |
| Oki et al. — *Triplet Loss for Knowledge Distillation* (arXiv:2004.08116) | 2020 | Introduce triplet-metric-learning loss into KD; student mimics teacher pairwise/triplet similarity | Triplet-metric KD is competitive with logit KD | Prior art for "distill via similarity structure rather than logits" — one candidate mechanism family for our Claim 2 loss. | arXiv API |
| Yuan et al. — *Revisiting KD via Label Smoothing Regularization* (arXiv:1909.11723) | CVPR 2020 | Theoretical link: KD = learned label smoothing; even weak/poor teachers help; teacher-free KD works | KD success is partly regularization, not only "dark knowledge" | Warns us: an M0-style baseline of *random-teacher* / *label-smoothing* distillation is needed to prove the human-alignment gain is specific (not generic soft-label regularization). | arXiv API |
| CLIP-TD — *CLIP Targeted Distillation for VL tasks* (arXiv:2201.05729) | 2022 | Distill CLIP knowledge into a task model with dynamically weighted, adaptively selected tokens | Big gains in low-shot / domain-shift VL tasks | Prior art for distilling a large image-text teacher into a smaller vision model — analogous scaffolding for the SigLIP-So400m → DINOv2 ViT-B / ViT-L distillation in our reproduction. | arXiv API |

---

## 2. Core Landscape Narrative

**Human-alignment of vision representations has moved from measurement to intervention.** Early work simply *measured* the gap between deep-network embeddings and human similarity judgments — the correlation is real but partial, and it does not improve with scale or architecture the way accuracy does (Muttenthaler et al. 2022). This established a durable finding: to close the gap, the **training objective / data** must change, not just the backbone. Post-hoc linear transforms were the first intervention (Attarian et al. 2020; the global-local transform in Muttenthaler et al. 2023). They demonstrate two things our reproduction must reckon with: (i) alignment gains transfer across similarity datasets, so a teacher fit on THINGS is expected to generalize; (ii) *naive* alignment damages the local structure that downstream utility relies on — motivating loss designs that preserve it.

**Knowledge distillation is the natural next step, but it inherits a specificity trap.** Distilling *similarity structure* (rather than class logits) from a large teacher into a smaller student is well-explored in the metric-learning-based KD line (Oki et al. 2020) and in the CLIP-teacher line (CLIP-TD, 2022). What makes the *human*-aligned case sharper is Yuan et al.'s 2019 warning that KD's benefit partly comes from label-smoothing regularization independent of any semantic "dark knowledge". That means our reproduction of a human-similarity distillation loss needs at least one control where the teacher's soft targets are *human-alignment-free* (random-triplet teacher, unaligned SigLIP teacher, or plain label-smoothing) — otherwise we cannot attribute a Spearman-with-humans gain to the human-alignment mechanism specifically.

**Hierarchical / multi-level similarity is the missing axis.** All three human-alignment papers above operate on a single, mostly coarse similarity signal — one triplet task, one arrangement task, or one binary-similarity task. `task.md`'s Claim 1 explicitly asks for teacher-synthesized signals across **coarse (animal vs. object), mid (mammal vs. bird), fine (dog breeds)** levels drawn from an unlabelled image pool (ImageNet). This is exactly the direction the field points to but existing linear-transform / single-task-KD work does not fully occupy. Verifying that (a) the teacher's synthesized labels genuinely differ at the three levels and (b) students distilled from them actually match humans at each level (not just on average) is the load-bearing empirical work of Claim 2/3.

**Utility and OOD robustness are the pass/fail gates.** The 2023 global-local work already shows that alignment done wrong *hurts* downstream tasks. `task.md`'s Claim 4 elevates this to a hard requirement: alignment should match or beat the unaligned baseline on few-shot classification (Birds, UC Merced, Colon + 7 specialty sets) and on OOD/subpopulation shift (BREEDS, ImageNet-A). The verification design therefore has an asymmetric loss function: even large Spearman gains do not count if downstream drops. This maps cleanly onto ladder-of-evidence intervention design — cheap correlational alignment metric first, then downstream + OOD as the specificity/off-target control.

**What the landscape leaves open for the reproduction.** The pieces exist in the literature — human-similarity teacher fitting, triplet-based KD, CLIP-scale distillation, utility-preserving transforms — but the *composition* claimed in `task.md` (one THINGS-fit teacher → hierarchical pseudo-labels on ImageNet → whole-backbone distillation into DINOv2 ViT-B, with the four-way evaluation) is what needs to be reproduced faithfully. Our job is *verification*, not novelty: for each of the four claims, decide the strongest reasonable measurable predicate given a 10-hour GPU budget on 4× GPUs (ids 0–3), and design each experiment with the specificity controls the priors above imply.

---

## 3. Sub-direction-Specific Work

- **Measurement of DNN ↔ human similarity alignment**
  - *Human alignment of neural network representations* (Muttenthaler et al. 2022, arXiv:2211.01201) — no-training probe of 32+ models against three human datasets; establishes the pre-intervention baseline. Leaves open: how to *close* the gap without hurting utility.

- **Post-hoc linear alignment (no backbone finetune)**
  - *Transforming NN visual representations to predict human judgments* (Attarian, Roads, Mozer 2020, arXiv:2010.06512) — expressive linear + asymmetric transforms.
  - *Improving NN reps using human similarity judgments* (Muttenthaler et al. 2023, arXiv:2306.04507) — global-local transform preserves local structure while aligning global. Leaves open: alignment across *multiple* abstraction levels and effect on OOD robustness (not just few-shot).

- **Distillation of similarity structure**
  - *Triplet Loss for KD* (Oki et al. 2020, arXiv:2004.08116) — student mimics teacher triplet-similarity. Leaves open: teacher trained on *human* triplets rather than model triplets; hierarchical multi-level teacher signal.
  - *CLIP-TD* (Wang et al. 2022, arXiv:2201.05729) — task-aware distillation from CLIP. Leaves open: distilling a *human-aligned* re-embedding of a CLIP/SigLIP-scale teacher, and controlling for generic soft-label regularization.

- **KD-as-regularization caveat**
  - *Revisiting KD via Label Smoothing* (Yuan et al. 2019, arXiv:1909.11723) — establishes that random-teacher / self-teacher gains exist, so *any* similarity-distillation study needs a non-human-aligned soft-label control.

---

## 4. Structural Gaps (methodology issues the reproduction must handle, given the landscape above — NOT ideation of new work)

- **Gap G1 — Attribution ambiguity of gains.** The KD-as-label-smoothing prior means an aligned-vs-unaligned Spearman gain is *not* itself evidence of human-alignment transfer. — *Competitive set*: Yuan et al. 2019; Attarian et al. 2020 baseline. — *Implication for verification*: at minimum include a **random-triplet-teacher** (or shuffled-label teacher) control on the same distillation loss.

- **Gap G2 — Level-wise vs. average alignment.** Existing papers report a single Spearman value; `task.md` explicitly demands "multiple abstraction levels". — *Competitive set*: Muttenthaler et al. 2022 / 2023. — *Implication*: evaluation metric per level (coarse / mid / fine on THINGS + Levels-style eval), not a single aggregate number; report the *minimum* level as well as the mean.

- **Gap G3 — Utility ↔ alignment trade-off is one-sided in priors.** The global-local work shows utility can be preserved on few-shot classification, but OOD robustness (BREEDS subpopulation shift, ImageNet-A) is under-tested. — *Implication*: Claim 4 must run BREEDS + ImageNet-A as explicit specificity controls, not add-ons.

- **Gap G4 — Teacher-fit quality confound.** All human-alignment gains implicitly assume the surrogate teacher matches humans well enough. — *Implication*: verify the teacher's THINGS triplet accuracy (against held-out human triplets) **before** using it for pseudo-labeling; if it is weak, Claim 2 gains are unattributable regardless of loss choice.

- **Gap G5 — Behavioural + uncertainty match is a distinct evaluation from Spearman.** Claim 3 says the aligned student better reproduces *human choice pattern + per-triplet uncertainty*, not just correlation. — *Implication*: predict per-triplet choice probability against the human distribution (RSA / KL / calibration on triplet answer distributions), not just aggregate Spearman.

## 5. Banlist — Failed Ideas (do not regenerate)

*(no prior banlist — this is round 1 and `research-wiki/` was absent)*
