# Final Proposal — Unified Testing Approach for the Hierarchical Human-Alignment Reproduction

**Date**: 2026-07-14
**Behavior-source**: given (from `task.md`)
**Mechanism**: discovery
**Language**: English
**Problem anchor (frozen — do NOT drift)**: Reproduce and verify the four claims in `task.md` about a THINGS-fit similarity teacher (SigLIP-So400m image-encoder) that synthesizes hierarchical human-like similarity signal on unlabelled ImageNet and is distilled into DINOv2 ViT-B, evaluated for (i) teacher-quality + hierarchical pseudo-label quality, (ii) improved multi-level Spearman with human triplets, (iii) improved behavioural + uncertainty match, (iv) preserved/improved downstream utility and OOD robustness — all under a 10 hr GPU budget across GPU ids {0,1,2,3}.

---

## Machine metadata (Phase 4.5 stamps)

```yaml
resource_fidelity: cost-aware        # NOT stamped strict — this is given + discovery, not the reproduction combo (given + given)
mechanism_strategy:
  directions: [Tuning & Editing, Location, Decision Auditing]   # in execution order
  rejected:
    - Causal Intervention — claims are about a changed model (post-distillation), not "component X drives B in the base model"; ablating parts of the aligned student is not asked and would not verify any of the four given claims.
    - Formation Tracing — no origin claim across finetune trajectory; influence-function / checkpoint-sweep work does not fit the 10-hr GPU budget.
    - Unit Interpretation — no claim asks what an internal unit means; full SAE training on ViT-B activations does not fit the budget and is not on the critical path.
  note: Tuning & Editing carries Claim 2 (alignment loss finetune); Location supports Claim 3 (final-layer RSA / choice-uncertainty behavioural comparison of aligned vs. unaligned); Decision Auditing gates Claim 4 (downstream + OOD as specificity / off-target checks that no shortcut was introduced).
# chosen_mechanism intentionally omitted — MECHANISM=discovery routes at /auto-experiment Phase 1.5 within the strategy chain above; no specific /mechanism-skills family committed here.
```

---

## 1. Thesis of the Verification Approach

The four claims form a single evidence chain: **teacher-quality → alignment-transfer → behavioural fidelity → utility guardrail**. A supported result requires all four links to hold (or, where a link is only partially supported, an explicit `conditional` verdict on that claim). The verification suite is designed to *fail loudly* rather than yield a mush of "positive-ish" numbers:

- The **aggregation of gains** across levels/tasks does not substitute for *level-wise* / *task-wise* gains — task.md phrases the claims at multiple abstraction levels and across a diverse downstream + OOD suite, so we report per-level and per-dataset numbers plus the aggregate.
- Every alignment gain is paired with a **specificity control** — a matched-cost non-human-aligned soft-label finetune of DINOv2 ViT-B, plus an unmodified pretrained baseline — so that Yuan et al.'s KD-as-label-smoothing prior cannot silently claim the credit.
- The **behavioural + uncertainty match** (Claim 3) is decomposed into per-triplet choice-agreement AND per-triplet confidence-calibration AND RSA(RDM) — all three must move in the predicted direction for a "supported" verdict; two out of three → `conditional`.
- **Utility non-inferiority AND OOD improvement** (Claim 4) are two conjoined sub-predicates; either failing alone → `conditional`.

## 2. Dominant Contribution of the Verification Package

A single reproducible pipeline that, for each of the four given claims, produces (a) the pre-registered measurable predicate, (b) at least one specificity / non-inferiority control, and (c) a paired-comparison significance test — bundled so downstream `/auto-verify` can stress-test the top-K admitted claims by swapping teacher / student / dataset without re-planning.

## 3. Complexity Intentionally Rejected

- No mechanism-family commitment at claim time (would pre-empt `/mechanism-skills` routing at experiment stage).
- No causal-intervention milestones (Claim shape does not ask "component X causes B in base model").
- No formation-tracing milestones (no origin claim; budget-prohibitive).
- No SAE / unit-labeling milestones (not on the critical path for any of the four claims).
- No student-swap in the *main* experiment (task.md pins DINOv2 ViT-B — swaps live in `/auto-verify` per the verify-stage candidate list).
- No teacher-swap ever (SigLIP-So400m is fixed across the whole project per task.md).
- No unlabelled-ImageNet scale-up beyond what the 10 hr budget can afford (cost-aware; the finetune data subset is documented in EXPERIMENT_PLAN.md, and cannot be inflated silently).

## 4. Global Resources (binding for the main experiment; from task.md)

- **Teacher (fixed, whole project)**: SigLIP-So400m image-encoder (HF `google/siglip-so400m-patch14-384` or ModelScope mirror).
- **Student (main experiment)**: DINOv2 ViT-B (HF `facebook/dinov2-base`).
- **Datasets (main experiment)**: THINGS (1,854 natural object concepts) + human odd-one-out triplet judgments (fit teacher + hold-out eval); ImageNet (ILSVRC-2012) unlabelled for teacher pseudo-labelling + alignment finetune data.
- **Datasets (verify-stage informational, used-as-needed)**: "Levels" (coarse/fine/class-boundary), a public human-similarity-judgment collection (RSA); 10 one-shot classification (Birds, UC Merced, Colon + 7 fine-grained specialty); BREEDS (entity13 / living17 / non-living26 / entity30); ImageNet-A.
- **DATA_DIR** = `/data/zhenqian/data`; **MODEL_DIR** = `/data/zhenqian/models`; symbolic links from working dir; missing → download to those roots via HF / GitHub / ModelScope. No other directory may be accessed.
- **Compute**: 10 hr total GPU budget across GPU ids {0,1,2,3}. Do not pause citing budget until actual GPU usage reaches 10 hr.
- **Runtime**: conda env for all runs.
- **Credentials (bypass proxy when calling the LLM API)**: HF token `<Your_token>`; ModelScope token `<Your_token>`; LLM API `API_KEY=<Your_api>`, `BASE_URL=https://www.dmxapi.cn/v1`, `MODEL=gpt-5.4`.

## 5. Claim-by-Claim Testing Strategy

Compact narrative here; the per-milestone experiment specs are in `EXPERIMENT_PLAN.md`.

### Claim 1 — teacher quality + hierarchical pseudo-label quality
- **1a (teacher fit)**: Fit an alignment head on SigLIP-So400m image-encoder from THINGS train-split triplets; evaluate held-out THINGS triplet accuracy vs. (i) unaligned SigLIP-So400m and (ii) chance (1/3). Report bootstrap CI on triplet accuracy.
- **1b (hierarchical pseudo-labels)**: Construct ≥ 3,000 ImageNet-derived triplets stratified into three level buckets (coarse: cross-super-category anchor vs. cross-super odd-one-out; mid: within-super but cross-basic-level; fine: within-basic but cross-sub-category). Score each with the fit teacher; report per-level agreement/separation of the teacher's triplet-choice distributions. Predicate: separation strictly positive at each level; monotonically level-organized.

### Claim 2 — alignment loss substantially improves multi-level Spearman
- **2a (main alignment run)**: Distill the teacher's triplet-similarity signal into DINOv2 ViT-B via a KL/triplet alignment loss on an ImageNet-unlabelled subset (size = whatever fits inside ≤ 3 hr on 4× GPUs 0/1/2/3 with the given student, target ≥ 100k images). Save checkpoint.
- **2b (multi-level eval)**: Evaluate Spearman of aligned-vs-human on THINGS held-out triplets globally AND per level (coarse/mid/fine buckets on the same held-out set). Report Δρ vs. unaligned baseline (M4).
- **2c (specificity control)**: Repeat 2a with a *matched-cost* non-human-aligned teacher — Option A: unaligned SigLIP-So400m image-encoder (before THINGS fitting), Option B: random-permutation teacher on the same THINGS triplets (see M5). Predicate: aligned > control by a positive margin at α=0.05.

### Claim 3 — behavioural + uncertainty match
- **3-choice**: per-triplet top-1 pick match rate on THINGS held-out and on the public RSA-collection triplets, aligned vs. unaligned.
- **3-uncertainty**: per-triplet confidence (softmax margin over the three odd-one-out candidates from similarity-derived probabilities) — Spearman between model confidence and human agreement rate; KL divergence from model to human choice distribution.
- **3-RSA**: Spearman between aligned-student RDM and human RDM on THINGS concepts, vs. same for unaligned.
- **Verdict rule**: all three sub-predicates in the predicted direction → `established`; two of three → `conditional`; ≤ one → `not-established`.

### Claim 4 — utility non-inferiority + OOD improvement
- **4a (utility)**: One-shot classification sweep on the 10-dataset panel (cost-aware compression allowed — see EXPERIMENT_PLAN.md — but always paired aligned vs. unaligned on the same subset). Predicate: mean-across-datasets accuracy non-inferior (per-dataset drop ≤ 1 point unless compensated).
- **4b (OOD)**: BREEDS entity13 / living17 / non-living26 / entity30 + ImageNet-A, aligned vs. unaligned. Predicate: strictly positive OOD gain, statistically distinguishable.
- **Verdict rule**: both sub-predicates hold → `established`; either fails → `conditional`; both fail → `not-established`.

## 6. Verification of the Verification (self-consistency gates)

Before we trust any of the four verdicts, the plan enforces three internal checks at experiment stage:

1. **Teacher-quality gate (Claim 1a) is a soft prerequisite for Claim 2 attribution.** If the teacher's held-out THINGS triplet accuracy is barely above chance / not clearly above unaligned SigLIP, then a *positive* Claim 2 gain still counts (KD-as-label-smoothing may explain it) but is downgraded to `conditional` regardless of the Δρ. This is enforced by the result-to-claim reviewer in the iteration loop, not by a hard M0 gate (BEHAVIOR_SOURCE=given → no M0).
2. **Specificity gate (Claim 2c) attributes the alignment gain.** If aligned − non-aligned-control Δρ ≤ 0 or not distinguishable from zero, Claim 2 is downgraded from `established` to `conditional`.
3. **Paired evaluation gate.** All aligned-vs-unaligned comparisons in every claim must be *paired* on the same evaluation items (same triplets, same downstream test sets, same OOD samples) — no unpaired subsampling.

## 7. Risks and Mitigations

- **Budget overrun.** With DINOv2 ViT-B + SigLIP-So400m teacher pass on 100–200k ImageNet samples, a single alignment finetune can easily consume 2–3 hr on 4× GPUs. Two finetunes (main + specificity control) + all evals must fit in 10 hr. Mitigations: (i) cache teacher features (single teacher forward-pass over ImageNet subset, saved to `/data/zhenqian/data/things_alignet_cache/`) so the teacher runs only once, not per-student-epoch; (ii) LoRA / partial-block finetune of DINOv2 ViT-B if full-backbone does not fit; (iii) stratified 4-of-10 downstream subset for M7 with the remaining 6 datasets logged for `/auto-verify`.
- **Teacher fit under-power.** If THINGS train-split human triplets are limited, the surrogate teacher may not clearly beat unaligned SigLIP. Mitigation: use all available THINGS triplets for training (already `used_n=all`), and report bootstrap CI so under-power is visible rather than hidden.
- **Level construction confound.** The ImageNet-hierarchical triplet construction depends on the ImageNet WordNet hierarchy — a coarse↔mid↔fine grouping that may not match human basic-level intuitions perfectly. Mitigation: document the grouping rule verbatim in the plan's M2 milestone; report per-level results transparently; a `conditional` verdict on Claim 1b is acceptable if the hierarchy is imperfect.
- **OOD evaluation cost.** ImageNet-A + all 4 BREEDS splits × aligned + unaligned = 10 eval jobs. Mitigation: use pre-computed features (a single forward pass per model on the OOD sets) and cheap linear-probe / cosine-similarity classification, not full retraining.

## 8. Handoffs

- **`/auto-experiment`** consumes `EXPERIMENT_PLAN.md` and routes the `Tuning & Editing` chain at Phase 1.5 to a concrete family — expected: alignment-loss / targeted-finetune family (likely full-backbone or LoRA variant on DINOv2 ViT-B; `/mechanism-skills` picks the submethod). Fields tagged `method_sensitive:` in the milestones may be re-bound at routing without counting as a plan rewrite.
- **`/auto-verify`** stress-tests the top admitted claim(s) using student swaps (Supervised ViT-S/B/L, DINOv1 ViT-B, SigLIP ViT-B, CapPa ViT-B) and dataset swaps ("Levels", RSA public collection); teacher swaps are not permitted (fixed by task.md).
- **`/auto-iteration-loop`** iterates on FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS verdicts.

## 9. Publication Fit (not a goal here — reproduction only)

This is a **reproduction / verification** run per task.md, not a novel-contribution run. `IDEA_REPORT.md` records the four given claims verbatim; this proposal codifies how each is tested with pre-registered predicates and controls. No novelty check was run (BEHAVIOR_SOURCE=given), no external review was invoked (BEHAVIOR_SOURCE=given).
