# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: false
chosen_family: none
chosen_idea_title: Cross-modal subliminal transmission of safety-competence loss in multimodal Gemma-3-4B-it (single unified claim; the mechanism sub-claim attaches iff the M0 phenomenon gate passes)
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Steering Vectors — the diff-in-means direction extraction in M1 IS Contrastive Activation Addition (CAA) / a Steering Vector construction; the M2 dose-response sweep IS the same submethod's causal-write mode. One canonical family covers both M1 (Location, correlational — extract `d̂_ℓ`) and M2 (Causal Intervention, causal — add α·d̂_ℓ), which is exactly the plan's Location → Causal Intervention ladder. Prior subliminal-transfer mechanism literature (Blank/Rajamanoharan/Conmy/Nanda et al. 2026; Morgulis-Hewitt 2026) also converges on steering-vector-shaped direction.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
2. Probing / Residual Stream States — the plan's M1 submethod (b) linear probe. Provides an alternative correlational Location signal (probe AUROC on treated-vs-CtrlB residual-stream classification). Fits as a companion / cross-check to the primary Steering Vectors direction. Not the primary because the probe alone is not causal — it would still need Steering Vectors to close the causal loop at M2.
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md
3. Causal Attribution / Patching — activation patching from treated-student onto Ctrl-A / Ctrl-B student at the M1-located site. Provides an alternative M2 verification (patch instead of steer, to test sufficiency), and by design is the standard-bearer for causal-necessity claims. Fits as a robustness cross-check on M2's steering result; not the primary because activation patching's per-item cost is much higher than steering (one full forward with counterfactual per patched site × item) and the plan's M1 already extracts a direction the steering path can consume directly.
   - path: skills/mechanism-skills/causal-attribution/patching/SKILL.md

## Composition plan
- **Screen (M1, ~1 GPU-hour)**: capture residual-stream activations at every language-tower layer (0..33 of gemma-3-4b-it) for a held-out batch of 500 QA_I items on BOTH treated-s42 and ctrlb-s42 students. Three parallel Location submethods:
  - (a) **primary — diff-in-means** (Steering Vectors family): `d̂_ℓ = (mean_treated − mean_ctrlb) / ‖·‖`; rank layers by `‖Δ_ℓ‖` (z-scored).
  - (b) companion — 5-fold linear probe (Probing family): sweep AUROC per layer.
  - (c) companion — direct-logit-attribution on items where treated errs but ctrlb is correct (Gradient Detection family, post-processing).
- **Verify (M2, ~60–90 GPU-hours)**: dose-response steering sweep at the M1-top site (with layer ± 1 as sensitivity control). α ∈ {−2, −1, −0.5, 0, 0.5, 1, 2} × per-layer std. Two arms:
  - Sufficiency: base student (Ctrl-A) + α·d̂_ℓ → measure QA_I. Match-random-direction control for specificity.
  - Necessity: treated-s42 student − α·d̂_ℓ → measure QA_I. Match-random-direction control.
  - Off-target specificity: MMLU-lite (≥500) + helpfulness-lite (≥200) at α★ to guard against general-ability collapse.
- **Recover / interpret (M3, optional, 0.5–12 GPU-hours)**: SAE projection of `d̂_ℓ` onto Gemma-3 language-tower features (if a public SAE exists; otherwise skip or train a small TopK-SAE within budget).

Cost notes:
- M1: 0.5–1 GPU-hour (activation capture is a forward pass; 500 items × 4 GPUs is trivial).
- M2: 88 evals × 0.5–1 h each = 60–90 GPU-hours (dominated by QA_I greedy generation + judge).
- Aggregate M1+M2 fits comfortably under task.md's ample compute budget.

## Plan reconciliation
<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->
- n_pairs: plan=500 → matches (Steering Vectors / CAA in the mechanism-skills catalog reports typical N∈[200,2000] for stable direction extraction; 500 with 4 GPUs is well within that range for this model / data).
- sites: plan=all_language_tower (layers 0..33) → matches (Steering Vectors expects a per-layer sweep to find the top-1 candidate).
- metric: plan=`‖Δ‖` (screen) + `Acc(QA_I)` at each α (verify) → matches (canonical for Steering Vectors: norm-of-difference at screen, target-metric under intervention at verify).
- gpu_hours: plan~60–90 → revised ~60–90 (unchanged; Steering Vectors is compute-efficient at eval time since the intervention is a residual-stream hook, not a forward-pass swap).
reconciliation_status: ok

## Rationale
Steering Vectors is recommended because:

1. **Fit to plan**: the plan's M1 primary submethod (diff-in-means) IS the CAA construction; the M2 dose-response (α · direction added to residual stream) IS the CAA intervention. One canonical family covers the entire mechanism ladder without requiring a second family to bridge Location → Causal Intervention.

2. **Fit to phenomenon**: Blank/Nanda et al. 2026 (arXiv 2606.00995 — "Subliminal Learning Is Steering Vector Distillation") argues the subliminal object *is* a steering direction; Morgulis & Hewitt 2026 predicts layer-localized transmission fitting a per-layer steering-vector sweep. Priors from same-topic prior work fall directly into this family.

3. **Fit to model**: gemma-3-4b-it's language tower is a standard 34-layer decoder — Steering Vectors' assumption of a linear encoding at a single (or narrow window of) layer(s) is well-supported by the LVLM cross-modal-safety literature (Xu et al. 2024 localizes safety to specific language-tower layers of a Gemma-family model).

4. **Fit to budget**: the compute profile is one forward pass per α-eval (steering is a hook), matching the plan's ~60–90 GPU-hour estimate exactly. No expensive backward passes as with attribution patching.

Companion probing and DLA (submethods 2, 3) are kept as CORROBORATING evidence at M1 — the mechanism claim is stronger when three different Location signals converge on the same layer/site. They live in the composition plan as parallel screens, not as separate mechanism families.

Cross-round exclusions: none — this is round 1 (no `families_already_settled` in EXPERIMENT_PLAN.md).

**Note**: this file is written pre-M0-verdict as a scaffold. `committed` will be flipped to `true` after M0 passes (Phase 1.25 → Phase 1.5). If M0 verdict is `not-established` or terminal `inconclusive`, the mechanism claim is dropped and this routing is not exercised — the file stays at `committed: false` as a scientific record of the plan that was not run.
