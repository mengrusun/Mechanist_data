# Claim Ledger — Hierarchical Human-Alignment Reproduction (given behavior + discovery mechanism)

**Direction**: Distill hierarchical human-similarity structure (from THINGS triplets, via a SigLIP-So400m teacher) into pretrained vision backbones (DINOv2 ViT-B main) and test whether alignment improves multi-level human-similarity correlation, reproduces human behaviour/uncertainty, and preserves/improves downstream utility + OOD robustness.
**Date**: 2026-07-14 → 2026-07-15
**Pipeline**: completed | **Iteration**: 8/10 "ready" (5/6 iterations; claim-reentries 2/2 exhausted; termination=positive_verdict)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1a — teacher beats unaligned SigLIP + chance | established (+13pp, non-overlap CI95) | integrity_only PASS (cap) | — | ⚪ integrity_only (audit PASS; swap-test deferred) |
| C1b — teacher hierarchy monotonic | conditional (non-monotonic; fine best) | integrity_only WARN (cap) | narrative flagged (non-blocking) | ⚪ integrity_only WARN (numeric-citation mismatch fine-level) |
| C2a — aligned Δρ ≥ 0.05 (α=0.05) | established (Δρ=+0.366, 7× threshold) | **PASS** (robustness=1.00) | — | ✓ holds |
| C2b — per-level gain (coarse/mid/fine) | conditional (coarse+mid ✓; fine 0 pairs) | integrity_only WARN (cap) | narrative flagged (non-blocking) | ⚪ integrity_only WARN — coarse+mid solid |
| C2c — gain is human-specific vs. control | established (77% specific) | integrity_only PASS (cap) | — | ⚪ integrity_only (audit PASS; swap-test deferred) |
| C3 — behaviour + uncertainty + RSA (3/3) | established (3/3 predicates ✓) | integrity_only PASS (cap) | narrative flagged (non-blocking) | ⚪ integrity_only (audit PASS; swap-test deferred) |
| C4a — 1-shot non-inferiority | not-established (Δ=−0.284) | integrity_only WARN (cap) | ✗ falsified — rewritten as C4a_v2 | ✗ falsified as originally stated — see C4a_v2 |
| C4b — OOD strict per-split gain | conditional (2/5 win — both true OOD) | integrity_only WARN (cap) | ✗ falsified — rewritten as C4b_v2 | ✗ falsified as originally stated — see C4b_v2 |
| C4a_v2 — documented trade-off (LoRA best) | LoRA mean Δ=−0.239 (attenuated) | integrity_only (inherit; cap) | ✓ narrowed & honest | ✓ narrowed & honest — documented LoRA-best-regime trade-off |
| C4b_v2 — selective BREEDS-only OOD gain | LoRA BREEDS super13 Δ=+0.248, super26 Δ=+0.080 | integrity_only (inherit; cap) | ✓ narrowed & honest | ✓ narrowed & honest — selective-BREEDS-only OOD gain |

---
## C1a — THINGS-fit teacher beats unaligned SigLIP + chance
- **Statement**: THINGS-fit SigLIP-So400m teacher head has held-out THINGS triplet accuracy strictly above the unaligned SigLIP-So400m baseline AND above the 1/3 chance floor (bootstrap-CI-95 does not cross either comparator).
- **Origin**: task.md Claim 1 (teacher captures human similarity structure well enough)
- **Data**: THINGS held-out — provenance=existing; available=full THINGS held-out, used=full THINGS held-out (M1 eval)
- **Models**: SigLIP-So400m (frozen backbone; small alignment head learned on THINGS train)
- **Method**: Triplet-KL fit on frozen SigLIP-So400m features; held-out triplet accuracy vs unaligned SigLIP + chance — **Screen** step of the Screen → Decode → Verify → Recover composition (Parameter-Space Task Vectors family).
- **Main experiment**: **established** — teacher_triplet_acc=0.590 [0.582, 0.598] vs unaligned SigLIP=0.460 [0.452, 0.467] vs chance=0.333; Δ(teacher−unaligned)=+0.130 with non-overlapping CI95.
- **Verify**: audit PASS; swap-test deferred (max_verify_claims cap).
- **Iteration**: —
- **Final**: ⚪ integrity_only (audit PASS; swap-test deferred — max_verify_claims cap)
- **Caveats**: —
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M1, refine-logs/EXPERIMENT_RESULTS.md#M1, runs/M1_teacher_fit/, verify/C1a_teacher_beats_baselines/

---
## C1b — Teacher hierarchy monotonic on ImageNet-synth triplets
- **Statement**: The THINGS-fit teacher's triplet-choice signal on ImageNet-synthesised triplets is monotonically level-organised across coarse / mid / fine (each level above chance; ordering pre-declared).
- **Origin**: task.md Claim 1 (spans multiple abstraction levels)
- **Data**: ImageNet val + WordNet-hierarchy triplets (leaf-parent construction for fine) — provenance=constructed; used=3,000 triplets (1,000/level, seed 42)
- **Models**: SigLIP-So400m + M1 teacher head
- **Method**: Forward-only per-triplet choice-agreement rate per level (M2).
- **Main experiment**: **conditional** — coarse=0.750, mid=0.747, fine=0.817 (all beat chance 0.333, ordering fine > coarse ≈ mid — non-monotonic; likely SigLIP training-signal bias).
- **Verify**: audit WARN (numeric citation mismatch fine-level); swap-test deferred (max_verify_claims cap).
- **Iteration**: narrative-only flagged as manuscript-writing scope (non-blocking).
- **Final**: ⚪ integrity_only WARN (numeric-citation mismatch fine-level; swap-test deferred)
- **Caveats**: The plan's strict monotonicity predicate may not match SigLIP's natural ordering; reconsider the ordering claim rather than the per-level above-chance claim. Phase 2 integrity WARN: numeric citation mismatch in fine-level.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M2, refine-logs/EXPERIMENT_RESULTS.md#M2, runs/M2_hierarchical_pseudo/, verify/C1b_teacher_hierarchy_monotonic/

---
## C2a — Aligned DINOv2 ViT-B multi-level Spearman gain
- **Statement**: Alignment finetune of DINOv2 ViT-B against the THINGS-fit teacher increases aggregate Spearman correlation with human THINGS-triplet similarity by Δρ ≥ 0.05 over the unaligned DINOv2 ViT-B baseline, with paired-bootstrap significance at α = 0.05.
- **Origin**: task.md Claim 2 (substantial multi-level improvement)
- **Data**: eval=THINGS held-out (1.71M pairs); finetune=ImageNet-val 50k pool (40k sampled, seed 42) — provenance=existing (eval) + constructed (finetune); used=40k for finetune + full THINGS held-out for eval; subset: plan asked for 150k ImageNet-train subset; substituted with 40k of ImageNet-val (val was on disk) — cost-aware. Matched-cost between M3 and M5 preserved.
- **Models**: DINOv2 ViT-B (student, main), SigLIP-So400m + M1 head (teacher, frozen)
- **Method**: M3 full-backbone KD finetune (triplet_kl loss, α=1.0, lr=5e-5, batch=512, 1 epoch, seed 42); then held-out THINGS Spearman vs M4 (paired-bootstrap α=0.05). **Verify** step of the Screen → Decode → Verify → Recover composition (Parameter-Space Task Vectors family).
- **Main experiment**: **established** — Δρ_aggregate = +0.366 (M3=0.555 vs M4=0.189), 7× the +0.05 threshold, non-overlapping CI95; triplet accuracy Δ = +0.125.
- **Verify**: robustness=1.00 — method n/a / dataset n/a / model **pass** (DINOv2 ViT-B → ViT-S swap also produces Δρ=+0.324); integrity=PASS; verdict=PASS.
- **Iteration**: —
- **Final**: ✓ holds (PASS at verify: model-swap DINOv2 ViT-B → ViT-S also produces large Spearman gain Δρ=+0.324 vs main +0.366; robustness=1.00)
- **Caveats**: Absolute levels may shift under the plan's original 150k ImageNet-train pool; ranking direction and effect magnitude should hold.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M3, refine-logs/EXPERIMENT_RESULTS.md#M3, runs/M3_aligned_dinov2/, runs/M4_unaligned_dinov2/, verify/C2a_aligned_spearman_gain/, runs/verify/C2a_model_swap_dinov2_vits/

---
## C2b — Per-level gain (coarse / mid / fine)
- **Statement**: The aggregate Spearman gain in C2a holds separately at each of the coarse / mid / fine abstraction levels (per-level Δρ > 0 for all three levels).
- **Origin**: task.md Claim 2 (multiple abstraction levels)
- **Data**: THINGS held-out, stratified — used=coarse n=723 / 183 pairs, mid n=5,705 / 1,194 pairs, fine n=33 / 0 pairs; subset: fine-level stratification yielded only 33 triplets and 0 Spearman-usable pairs (construction artifact, not under-power).
- **Models**: DINOv2 ViT-B aligned (M3) vs. unaligned (M4)
- **Method**: Per-level Spearman aligned − unaligned; predicted-direction gain at every level.
- **Main experiment**: **conditional** — coarse Δρ = +0.526, mid Δρ = +0.427; fine 0 pairs available — inconclusive.
- **Verify**: audit WARN (fine-level 0 pairs); swap-test deferred (max_verify_claims cap).
- **Iteration**: narrative-only flagged as manuscript-writing scope (non-blocking).
- **Final**: ⚪ integrity_only WARN (fine-level 0 pairs; swap-test deferred) — coarse+mid solid
- **Caveats**: Fine-level Spearman required 0 pairs to be constructible under the current stratification; increase fine-level construction size to resolve. Phase 2 integrity WARN.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M3, refine-logs/EXPERIMENT_RESULTS.md#M3, verify/C2b_perlevel_spearman_gain/

---
## C2c — Gain is specific to human alignment vs. control
- **Statement**: The multi-level Spearman gain from C2a is specifically attributable to human-similarity alignment (the aligned student strictly beats a matched non-human-aligned soft-label control at α = 0.05), not to generic KD label smoothing.
- **Origin**: task.md Claim 2 (specificity — dedicated alignment loss with human structure)
- **Data**: Same ImageNet-val 40k finetune subset as M3; teacher = unaligned SigLIP-So400m soft labels — provenance=constructed; used=40k identical to M3 for matched cost; subset: matched-cost specificity control.
- **Models**: DINOv2 ViT-B aligned (M3) vs. DINOv2 ViT-B control (M5)
- **Method**: M5 matched-schedule finetune with unaligned SigLIP-So400m soft labels; paired-bootstrap Spearman comparison against M3 on THINGS held-out (α=0.05).
- **Main experiment**: **established** — M3−M4 aggregate gain = +0.366; M5−M4 matched-cost control gain = +0.085; aligned gain is 4.3× control gain; human-alignment-specific component = +0.281 = **77% of total gain**.
- **Verify**: audit PASS; swap-test deferred (max_verify_claims cap).
- **Iteration**: —
- **Final**: ⚪ integrity_only (audit PASS; swap-test deferred — max_verify_claims cap)
- **Caveats**: —
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M5, refine-logs/EXPERIMENT_RESULTS.md#M5, runs/M5_control_dinov2/, verify/C2c_gain_alignment_specific/

---
## C3 — Behaviour + uncertainty + RSA (3/3 predicates)
- **Statement**: The aligned DINOv2 ViT-B more accurately reproduces human behaviour AND per-triplet uncertainty than the unaligned baseline, conjunctively across three predicates: (i) higher per-triplet choice-agreement with humans, (ii) lower model-to-human uncertainty-KL AND higher rank correlation between model confidence and human agreement, (iii) higher RSA between model RDM and human RDM.
- **Origin**: task.md Claim 3 (behavioural patterns and uncertainty)
- **Data**: THINGS testset2 vs testset2_repeat (two independent workers per triplet, row-aligned ≥99.9%) — provenance=existing; used=36,187 usable triplets, human two-worker agreement = 86.3%.
- **Models**: DINOv2 ViT-B aligned (M3) vs. unaligned (M4)
- **Method**: M6 forward-only — choice-agreement on human-consensus subset, uncertainty-Spearman, model→human KL, model-RDM vs human-RDM Spearman; 3/3 predicates → established.
- **Main experiment**: **established** — choice_acc_on_agreed 0.586 vs 0.434 (+15.2pp); uncertainty_spearman 0.054 vs 0.017 (weak but correct direction); RSA_spearman 0.555 vs 0.189 (+0.366); KL 0.228 vs 0.271 (supporting).
- **Verify**: audit PASS; swap-test deferred (max_verify_claims cap).
- **Iteration**: narrative-only flagged (uncertainty framing needs careful wording — non-blocking).
- **Final**: ⚪ integrity_only (audit PASS; swap-test deferred — max_verify_claims cap)
- **Caveats**: Uncertainty Spearman is weak in absolute terms — the binary two-worker agreement label is a sparse signal; direction is correct but strength is low.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M6, refine-logs/EXPERIMENT_RESULTS.md#M6, runs/M6_behavioural/, verify/C3_behavioural_uncertainty_match/

---
## C4a — 1-shot classification non-inferiority (superseded by C4a_v2)
- **Statement**: The aligned DINOv2 ViT-B is non-inferior to the unaligned baseline on downstream one-shot classification: mean top-1 across the 4-dataset stratified subset (Birds / UC Merced / Colon-pathology / Aircraft) is ≥ the unaligned mean, with no per-dataset drop > 1 point uncompensated elsewhere.
- **Origin**: task.md Claim 4 (alignment is not at odds with utility)
- **Data**: SUBSTITUTED — plan datasets not on disk → DTD, Fashion-MNIST, imagenet_val_top100, imagenet_val_top20 — provenance=existing; used=4 substitute datasets; subset: dataset substitution.
- **Models**: DINOv2 ViT-B aligned (M3) vs. unaligned (M4)
- **Method**: M7 per-dataset 1-shot cosine/linear-probe top-1 with paired-bootstrap; non-inferiority + per-dataset drop check.
- **Main experiment**: **not-established** — aligned strictly worse on 4/4 substitute datasets; mean Δ=−0.284; p_positive=0.00 on every dataset.
- **Verify**: audit WARN (dataset substitution); swap-test deferred (max_verify_claims cap).
- **Iteration** (superseded_by_C4a_v2):
  - iter 1 ②: α=1.0 → α=0.1 rerun (M3+M7) — mean Δ = −0.294; α↓ hypothesis FALSIFIED
  - iter 2 ②: LoRA scope (r=16, α=1.0) rerun (M3+M7) — mean Δ = −0.239 (16% relative improvement); best regime confirmed; code fixes to align_student.py + eval_downstream.py + eval_student_similarity.py (silent LoRA-key discard bug)
  - iter 3 ③ lightweight: claim rewrite to C4a_v2 (documented-trade-off)
  - falsified: original "preserved or improved fine-grained ImageNet 1-shot utility" framing
  - narrowed_to: see C4a_v2
- **Final**: ✗ falsified as originally stated — rewritten as C4a_v2 (see C4a_v2 for the honest LoRA-best-regime narrowing)
- **Caveats**: Verdict is on the on-disk substitute panel; adaptation scope (not loss weight) is the utility bottleneck — reviewer confirmed at iter 3; Phase 2 integrity WARN.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M7, refine-logs/EXPERIMENT_RESULTS.md#M7, runs/M7_downstream/, verify/C4a_downstream_noninferior/, runs/iteration_round_1/M3_alpha0p1/, runs/iteration_round_1/M7_alpha0p1/, runs/iteration_round_2/M3_lora_a1p0/, runs/iteration_round_2/M7_lora_a1p0/, review-stage/AUTO_ITERATION_FINAL_REPORT.md#4b.1

---
## C4b — OOD strict per-split gain (superseded by C4b_v2)
- **Statement**: The aligned DINOv2 ViT-B strictly improves OOD robustness over the unaligned baseline on every held-out split: BREEDS-{entity13, living17, non-living26, entity30} and ImageNet-A.
- **Origin**: task.md Claim 4 (improve out-of-distribution robustness)
- **Data**: SUBSTITUTED — plan panel not on disk → BREEDS super13, BREEDS super26, imagenet_val_20_easy, imagenet_val_20_hard, fashion_mnist_ood — provenance=existing; used=5 substitute splits (BREEDS super13/26 = true subpopulation-shift OOD; other 3 in-dist / non-standard); subset: substitute panel.
- **Models**: DINOv2 ViT-B aligned (M3) vs. unaligned (M4)
- **Method**: M8 feature-extraction + protocol-standard top-1 per split.
- **Main experiment**: **conditional** — BREEDS super13 Δ=+0.201 (p_pos=1.00), BREEDS super26 Δ=+0.115 (p_pos=1.00); imagenet_val_20_easy Δ=−0.272, imagenet_val_20_hard Δ=−0.394, fashion_mnist_ood Δ=−0.202. 2/5 splits strict Δ > 0, both true OOD.
- **Verify**: audit WARN (dataset substitution + OOD scope); swap-test deferred (max_verify_claims cap).
- **Iteration** (superseded_by_C4b_v2):
  - iter 1 ②: α=1.0 → α=0.1 M8 rerun — same 2/5 splits; BREEDS gains slightly larger; verdict unchanged
  - iter 2 ②: LoRA scope M8 rerun — BREEDS super13 Δ=+0.248 (up from +0.201); mean Δ improved from −0.110 to −0.063; same 2/5 splits
  - iter 4 ③ lightweight: claim rewrite to C4b_v2 (selective-BREEDS-only)
  - falsified: original "strict improvement on every held-out split" universal-quantifier framing
  - narrowed_to: see C4b_v2
- **Final**: ✗ falsified as originally stated — rewritten as C4b_v2 (see C4b_v2 for the honest selective-BREEDS narrowing)
- **Caveats**: Substitute panel replaces plan's BREEDS entity13/living17/non-living26/entity30 + ImageNet-A; universal-quantifier wording flagged as systematic overclaim risk; Phase 2 integrity WARN.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M8, refine-logs/EXPERIMENT_RESULTS.md#M8, runs/M8_ood/, verify/C4b_ood_strict_improve/, runs/iteration_round_1/M8_alpha0p1/, runs/iteration_round_2/M8_lora_a1p0/, review-stage/AUTO_ITERATION_FINAL_REPORT.md#4b.2

---
## C4a_v2 — Documented one-shot fine-grained trade-off (LoRA best regime)
- **Statement**: Under parameter-efficient LoRA-scope alignment adaptation (r=16, α=1.0), the aligned DINOv2 ViT-B exhibits a documented one-shot fine-grained classification trade-off vs the unaligned baseline: on the 4-dataset substitute panel mean top-1 drops below unaligned (LoRA best-regime mean Δ = −0.239), substantially attenuated relative to full-backbone α=1.0 KD (Δ = −0.284). The original "preserved or improved fine-grained ImageNet 1-shot utility" framing is retracted.
- **Origin**: iter 3 ③ lightweight rewrite of C4a (adaptation-scope-is-the-bottleneck pattern confirmed at iter 3)
- **Data**: DTD / Fashion-MNIST / imagenet_val_top100 / imagenet_val_top20 (substitute for plan's Birds/UC-Merced/Colon/Aircraft) — provenance=existing (substitute panel); used=4 datasets across regimes {full α=1.0, full α=0.1, LoRA α=1.0}; subset: substitute panel + best-regime disclosure.
- **Models**: DINOv2 ViT-B aligned (LoRA r=16, α=1.0, seed 42) vs. unaligned
- **Method**: M7 1-shot cosine/linear-probe top-1 with paired-bootstrap, evaluated across three adaptation regimes; narrowing rewrite anchors the claim to the honestly-supported best regime (LoRA).
- **Main experiment**: **documented-trade-off** — LoRA best-regime mean Δ = −0.239 (fashion_mnist recovered from −0.202 to −0.095); full α=1.0 Δ = −0.284; full α=0.1 Δ = −0.294.
- **Verify**: audit inherited from C4a; swap-test deferred (max_verify_claims cap).
- **Iteration**: accepted at iter 3 (reviewer: rewrite is materially more honest, specific, reviewer-acceptable).
- **Final**: ✓ narrowed & honest — documented LoRA-best-regime trade-off supersedes C4a
- **Caveats**: Adaptation scope, not loss weight, drives the trade-off; substitute panel still applies.
- **Artifacts**: review-stage/AUTO_ITERATION_FINAL_REPORT.md#4b.1, runs/iteration_round_2/M3_lora_a1p0/, runs/iteration_round_2/M7_lora_a1p0/

---
## C4b_v2 — Selective BREEDS-only OOD gain (LoRA best regime)
- **Statement**: Under parameter-efficient LoRA-scope alignment (r=16, α=1.0), the aligned DINOv2 ViT-B shows selective OOD-robustness gains concentrated on subpopulation-shift splits with BREEDS-style taxonomy: BREEDS super13 mean Δ = +0.248 (LoRA best; vs full α=1.0 KD +0.201) and BREEDS super26 Δ = +0.080 (vs full +0.115). The original claim of strict per-split improvement on every held-out split of BREEDS entity13/living17/non-living26/entity30 + ImageNet-A is retracted; the evidence supports "alignment reshapes representation toward taxonomy-compatible subpopulation-shift structure, not universal OOD robustness".
- **Origin**: iter 4 ③ lightweight rewrite of C4b (alignment-gains-concentrate-on-taxonomy-compatible-structure pattern from iter 3, confirmed at iter 5)
- **Data**: BREEDS super13, BREEDS super26, imagenet_val_20_easy, imagenet_val_20_hard, fashion_mnist_ood (substitute for plan's BREEDS entity13/living17/non-living26/entity30 + ImageNet-A) — provenance=existing (substitute panel); used=5 splits across regimes; subset: substitute panel + selective-gain framing.
- **Models**: DINOv2 ViT-B aligned (LoRA r=16, α=1.0, seed 42) vs. unaligned
- **Method**: M8 protocol-standard top-1 per split, evaluated across three adaptation regimes; narrowing rewrite ties the supported gain to BREEDS-style subpopulation-shift.
- **Main experiment**: **selective-BREEDS-only** — LoRA best-regime: BREEDS super13 Δ=+0.248, super26 Δ=+0.080; imagenet_val_20_easy Δ=−0.166, imagenet_val_20_hard Δ=−0.380, fashion_mnist_ood Δ=−0.095. Mean Δ = −0.063 (improved from full's −0.110).
- **Verify**: audit inherited from C4b; swap-test deferred (max_verify_claims cap).
- **Iteration**: accepted at iter 5 (reviewer: removes false universal/OOD-improvement framing; anchors to actually-supported regime; explicitly acknowledges dataset substitution).
- **Final**: ✓ narrowed & honest — selective-BREEDS-only OOD gain supersedes C4b
- **Caveats**: Substitute panel still applies; universal-quantifier wording pattern (both C4a and C4b) flagged as systematic overclaim risk.
- **Artifacts**: review-stage/AUTO_ITERATION_FINAL_REPORT.md#4b.2, runs/iteration_round_2/M8_lora_a1p0/

---
## Journey Summary
- **Claim**: 1 given behavior (task.md) → single-idea capture — Hierarchical Human-Alignment Reproduction; 4 task.md claims decomposed into 9 C-claims across 8 milestones + 1 enabler.
- **Mechanism strategy**: Tuning & Editing → Location → Decision Auditing
- **Mechanism routing**: family=Representation and Parameter Analysis / Parameter-Space Task Vectors
- **Experiment**: 10 runs, ~2.21 GPU-hours across GPUs {0,1,2,3}, headline positive on the alignment core (C1a/C2a/C2c/C3 established) with a general-ability trade-off (C4a not-established; C4b/C1b/C2b conditional).
- **Verify**: 8 claim(s): 1 PASS / 0 FAIL / 0 INCONCLUSIVE / 0 ZEV / 7 INTEGRITY_ONLY (cap=7, swap_off=0); integrity[Phase2 WARN=4-PASS+4-WARN admit-all / Phase9 PASS]; Stage-2 pick=C2a with model swap DINOv2 ViT-B → ViT-S (Δρ=+0.324 vs main +0.366, robustness=1.00).
- **Iteration**: 5/6 iterations, claim-reentries=2/2 (exhausted), score 8/10 verdict ready, termination=positive_verdict; 6 /run-experiment calls, 1.12 GPU-hr; two ② main-experiment fixes (α↓ falsified; LoRA best regime confirmed) + two ③ lightweight claim rewrites (C4a→C4a_v2 documented-trade-off, C4b→C4b_v2 selective-BREEDS-only).
- **Figures**: 0 across 10 claims; 10 judgment-skipped (per-claim key stats already fully surfaced in ledger prose — bar/grouped-bar renders would not add information beyond the numbers already present); 0 render-skipped, 0 errored.

## Open Items
- Verify swap-test deferred (max_verify_claims cap = 1) — 7 INTEGRITY_ONLY claims (C1a, C1b, C2b, C2c, C3, C4a, C4b) still need cross-model/dataset/method robustness evidence; upgrade with `/auto-verify <id> — resume: true` per claim (Stage 1 audit reused). Iteration produced C4a_v2 and C4b_v2 that inherit this same INTEGRITY_ONLY status.
- M3/M5 finetune substituted ImageNet-train pool with ImageNet-val 50k (40k sampled) — plan asked for 150k of ImageNet-train (not on disk); matched-cost preserved so C2c comparability holds, but absolute Spearman levels may differ from plan-scale.
- M7 downstream substituted plan's Birds/UC-Merced/Colon/Aircraft with on-disk DTD/Fashion-MNIST/imagenet_val_top100/top20 (plan datasets not on disk) — the C4a → C4a_v2 rewrite documents the trade-off honestly, but re-running on the plan's exact panel would tighten the manuscript claim.
- M8 OOD substituted plan's BREEDS entity13/living17/non_living26/entity30 + ImageNet-A with on-disk BREEDS super13/super26 + imagenet_val_20_easy/hard + fashion_mnist_ood — the C4b → C4b_v2 rewrite ties the selective gain to BREEDS-style subpopulation shift; the plan's exact panel would validate the framing.
- C1b non-monotonic (fine > coarse) — likely SigLIP training-signal bias toward within-basic-level distinctions; iteration flagged as narrative-only manuscript-writing scope (not blocking).
- C2b fine-level Spearman required 0 pairs to be constructible under the current stratification — a construction artifact rather than a real inconclusive; iteration flagged as narrative-only.
- C3 uncertainty-Spearman is weak in absolute terms (0.054 vs 0.017) — direction is correct but strength is low; iteration flagged as narrative-only (needs careful wording in the manuscript).
- ~6.5 GPU-hr of the 10-hr budget remaining unused at pipeline termination — additional swap-tests or figure-panel expansion are cheap to add.
