# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Parameter-Space Task Vectors
chosen_idea_title: Subliminal Learning in Diffusion Image Models (Qwen-Image)
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/parameter-space-task-vectors/SKILL.md
  - skills/mechanism-skills/probing/SKILL.md
  - skills/mechanism-skills/causal-attribution/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Parameter-Space Task Vectors — cheapest, most direct: we have 8 teacher-arm × 8 Ctrl-B LoRA checkpoints (paired by seed). Task vector ΔW = θ_ft − θ_pre is exactly what the method computes; per-block/per-module Frobenius delta + top-PCA of ΔW give the shortlist and the "banana direction" in one shot, with zero forward passes on the mechanism side. The direction then becomes the M2 steering handle (Family §3 rule: read-out = projection, write-in = addition).
   - path: skills/mechanism-skills/representation-and-parameter-analysis/parameter-space-task-vectors/SKILL.md

2. Probing / Residual Stream States — the plan's second declared signal ("linear probe on `is_teacher_arm ∈ {0,1}` per (layer × site × timestep)"). Complements #1 by providing an activation-space signal that resolves *timestep*, which #1 (weight-space) cannot. Higher cost (forward passes on ≥ 80 preference prompts × ~48 DiT blocks × 3 timesteps × 16 student LoRAs). Runs as a secondary check in M1 if compute permits.
   - path: skills/mechanism-skills/probing/SKILL.md

3. Causal Attribution / Attribution Patching — activation patching from Ctrl-B student → teacher-arm student at shortlist sites is a gold-standard C3 verification. Included as a fallback / cross-check for M2 alongside the primary Steering-Vectors approach.
   - path: skills/mechanism-skills/causal-attribution/SKILL.md

## Composition plan

**M1 (Location) — screen + decode:**
1. Load 8 teacher-arm and 8 Ctrl-B student LoRA adapters (from M0.6 outputs) + the teacher anchor LoRA.
2. For each `(DiT block × target-module-family)`: compute per-seed `ΔW = B @ A` (LoRA product), then per-block Frobenius norm `||ΔW||_F` and top-1 PCA / SVD of the *mean* teacher-arm `ΔW` vs. the mean Ctrl-B `ΔW`.
3. Rank blocks by `overlap_gap_k` (Grassmann principal-angle overlap of top-k singular vectors, teacher-arm vs. Ctrl-B) at `k ∈ {1, 2, 4, 8}`. Emit a shortlist ≤ 20 % of `blocks × module-family × timestep-buckets`.
4. Extract the top-1 left singular vector of the mean-teacher-arm ΔW at `(b*, attn.to_out.0)` — this is the "banana direction" `v̂*`. Also emit a top-2 direction and a matched-random-control direction (adjacent off-shortlist block).
5. **Optional (compute-permitting):** run the Probing candidate #2 on the top-3 shortlist blocks × 3 timesteps to add an activation-space cross-check. Skipped if the budget cannot fit; recorded as `probing_status: skipped-for-budget`.

**M2 (Causal Intervention) — verify + recover:**
1. Apply forward-hooks on `transformer_blocks[b*]` of the teacher-arm student for each intervention type ∈ `{baseline, ablate, amplify_x2, amplify_x3, amplify_x4, random_ablate}`. Sigma calibration: express coefficient in `σ_proj` units per steering-coefficient-tuning tip (per-site `σ = std(h · v̂)` estimated on 4 warm-up prompts).
2. Sweep interventions × 160 preference prompts; judge each generation with gpt-5.4; compute `ΔP(banana)` and off-target `fluency` (fruit_count / total).
3. Specificity checks:
   - Matched-random-control direction @ same block → same intervention → require `|ΔP_random| < 0.02`.
   - Off-target fluency preserved (< 5% relative drop).
4. Dose-response: `α ∈ {0, ±0.25, ±0.5, ±1, ±2, ±3, ±4}` at the top-1 site (per steering-coefficient-tuning tip's coarse sweep). Report the smallest α with target effect.

**M3-stretch (Unit Interpretation):** if C2+C3 both `confirmed` AND ≤ 0.1 GPU-h remains → light labeling via top-activating input images at `(b*, target_module)`.

**Cost notes (post-reconciliation):**
- M1 Parameter-Space: ~0.2 GPU-h (weight arithmetic only, no forward passes on the mechanism side; SVD of ~48 × 12 module deltas is trivial).
- M1 optional Probing: ~0.4 GPU-h (activation extraction on 80 prompts × 3 timesteps × 16 students).
- M2 Steering + specificity + dose: ~0.5 GPU-h (6 interventions × 160 prompts / 4 GPUs → ~15 min wall clock + ~5 min judge; dose-response +~0.15 h).
- **Total M1+M2 revised: ~0.7–1.1 GPU-h** (vs. plan's ~1.4 h at claim time — the committed submethod is cheaper).

## Plan reconciliation

<!-- Written by Step 7 once the family is committed. One row per method_sensitive field
     declared on M1 and M2 in EXPERIMENT_PLAN.md. -->

**M1 (mechanism-location):**
- n_pairs: plan=80 → re-bound `n_pairs=8` (paired student *checkpoints*, not paired image pairs) — the Parameter-Space submethod pairs seed-matched teacher-arm LoRA vs. Ctrl-B LoRA weights; we have exactly 8 seed-matched pairs from M0.6. Image-pair `n=80` would apply only to a probing/patching secondary check, and is re-bound `n_pairs_probing=80` for that optional sub-check.
- sites: plan=`residual/attention/MLP × all DiT layers × ≥3 timesteps` → **matches** for the site-type axis (Parameter-Space delivers per-block, per-site-type deltas across all layers). The **timestep** axis is naturally satisfied by inference-time application in M2; weight-space is timestep-agnostic, so M1's timestep resolution is a *design choice* deferred to M2 (steering hook applied at chosen timesteps).
- metric: plan=`probe AUC + ΔW Frobenius delta` → re-bound `Grassmann-overlap gap (k∈{1,2,4,8}) + Frobenius delta + top-1 PCA variance` — the Parameter-Space submethod's canonical metric is subspace overlap, not probe AUC. Probe AUC (if truly required by C2's discriminating test) is the optional cross-check in candidate #2; documented but not the primary signal.
- gpu_hours: plan~0.8 → revised ~0.2 (primary) + up to ~0.4 (optional probing) = **~0.2–0.6 h total** — weight-space arithmetic + SVD is dominated by memory I/O, not compute.

**M2 (mechanism-intervention):**
- n_pairs: plan=80 → re-bound `n=160 (unpaired preference set)` — steering interventions do not require paired inputs; the specificity check uses a matched-random direction on the same 160 prompts. If activation-patching is later added as cross-check (fallback candidate #3), that would need `n_pairs=80` seed-matched pairs.
- sites: plan=`shortlist sites × timesteps` → **matches** — M2 reads the M1 shortlist directly (`banana_direction.pt`, block b*, target module).
- metric: plan=`ΔP(banana) + specificity (sibling site, off-target metric) + dose-response` → **matches** exactly.
- gpu_hours: plan~0.6 → revised ~0.5 — 6 interventions × 160 prompts across 4 GPUs → ~15 min wall + ~5 min judge; dose-response adds another 0.15 h.

reconciliation_status: ok

## Rationale

The plan's mechanism-strategy metadata explicitly names three competing LLM-side accounts as the discriminative target (LoRA-artifact rank inverted-U, single steering vector, divergence-latent + single-early-layer). **Parameter-Space Task Vectors** is the single most efficient method that directly addresses each:

- **LoRA-artifact test**: rank of the ΔW subspace across seeds — if the top-1 PCA already explains > 50 % of variance, the effect is low-rank (consistent with rank-inverted-U at r=32).
- **Single steering vector test**: top-1 singular vector variance-explained + whether the same direction transfers across seeds — direct evidence.
- **Divergence-latent test**: which DiT block dominates the Frobenius delta — early-block dominance supports the account; distributed dominance refutes it. Timestep resolution requires the optional Probing cross-check (candidate #2), which is *why* it is included in the composition plan.

The plan explicitly flagged `method_sensitive` fields anticipating this re-bind ("routing can re-bind them without a plan rewrite"), consistent with what we did above. The claim stage's estimated `n_pairs=80` was a generic activation-based-method assumption; the committed submethod uses paired *checkpoints* (n=8), which is a natural strengthening (each pair is a full model instance, not a single prompt-image datum).

Family §3's built-in composition guarantee ("one direction serves both as read-out and write-in") makes M2 a natural extension: the M1-extracted `banana_direction.pt` becomes the M2 steering vector — no method mismatch across milestones. This is the cheapest path that verifies C2 *and* C3 within the same conceptual handle.
