# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Feature Dictionary Learning / SAE
chosen_idea_title: Reproduction of Five SAE-on-ESM-2 Interpretability Claims
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/feature-dictionary-learning/SAE/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/SKILL.md
  - skills/mechanism-skills/probing/SKILL.md

## Candidates

1. **[recommended]** Feature Dictionary Learning / SAE — All five claims (C1 feature count, C2 concept alignment, C3 superposition ladder, C4 novel-concept auto-interp, C5a probing utility, C5b steering) are stated over an SAE dictionary trained on ESM-2 residual-stream activations. Pretrained SAEs (`InterPLM-esm2-650m/layer_{1,9,18,24,30,33}`) are fixed inputs — evaluation-only, no training. Every downstream analysis (F1 per-residue alignment, LLM auto-interp on top-activating windows, feature-clamp steering) uses the SAE dictionary as the primary unit of measure.
   - path: skills/mechanism-skills/feature-dictionary-learning/SAE/SKILL.md

2. Representation and Parameter Analysis / Steering Vectors — Powers M6's supporting arms: the mean-activation-addition baseline (`Δ_c = mean(h_L | c=1) − mean(h_L | c=0)`) is a canonical steering-vector implementation, exactly the "pre-SAE" baseline the plan calls out. The SAE-clamp intervention in M6 itself is also a directional additive intervention, so this family co-composes with #1 as the write-in operator over the dictionary atom `d_f`.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/SKILL.md

3. Probing — Powers M5 (annotation-filling): per-concept binary linear probes trained on SAE code `Z_L*` vs. raw residual stream `H_L*`, PR-AUC on held-out Swiss-Prot test split. This is a classic frozen-encoder probing setup; the probe-utility comparison is the SAE-vs-neuron test for downstream decodability.
   - path: skills/mechanism-skills/probing/SKILL.md

## Composition plan

Screen → decode → verify → recover, composed across milestones:

- **Screen (M1)** — Cheap per-feature statistics (density, dead / ultra-low-density fraction, thin auto-interp gate at `τ_auto_gate = 0.3`) over the six pretrained SAEs. Reports the "interpretable feature count" headline vs. raw-neuron baseline. **Cost:** ~1.5 GPU-h (one forward pass over Swiss-Prot test split × 6 layers, plus GPT calls for the gate).
- **Decode (M2, M4)** — M2 aligns every unit (SAE feature *and* raw neuron) to Swiss-Prot per-residue concept masks via F1 (`q_top = 0.99, τ_F1 = 0.5, τ_clean = 0.7`); sensitivity sweep at 6 grid points. M4 extends the label vocabulary beyond Swiss-Prot by prompting `gpt-5.4` on top-activating protein contexts, gated by held-out predictivity + synonym check + random-feature control. **Cost:** M2 ~2.0 GPU-h; M4 ~1.5 GPU-h (GPU for UniRef activations, LLM wall-clock separate).
- **Verify (M3, M6)** — M3 is the specificity ladder that licenses the superposition interpretation (SAE > PCA ≈ random-rotation ≈ neurons > shuffled-SAE). M6 is the causal-intervention verify: SAE-feature-clamp on labeled features vs. no-steer, mean-add (steering-vector baseline), random-clamp, dose ladder `α ∈ {0.5, 1, 2, 4}·σ_f`, plausibility band ≤ 1.5× PPL. **Cost:** M3 ~1.0 GPU-h (reuses cached activations); M6 ~3.0 GPU-h (generation-heavy).
- **Recover (M5)** — Downstream utility validation via linear probes: SAE-code probe vs. raw-neuron probe on 50 top-prevalent concepts, three seeds, paired-Wilcoxon. **Cost:** ~1.0 GPU-h.

Downstream analysis steps (PCA fit in M3 Arm B, shuffled-code control in M3 Arm C, permutation nulls in M4 control) are post-processing on cached SAE codes / residual-stream activations — not stand-alone mechanism families, so they belong here in the composition plan, not as a top-level candidate.

**Total revised GPU-hours:** ~10.0h (matches plan estimate; no re-binding needed — SAE dictionary is a pretrained fixed resource, and the composition plan lines up with the plan's milestone graph and per-milestone estimates exactly).

## Plan reconciliation

<!-- Written by Step 7. One row per method_sensitive field declared on the intervention milestone(s). -->

- **M1 — gpu_hours**: plan~1.5h → matches — SAE eval-only (encoder forward pass) at expansion=8 (10,240 features × 6 layers) on Swiss-Prot test (10k sequences, avg ~350 residues) is dominated by the ESM-2-650M forward pass. A800 80GB fits ESM-2-650M with 4k-token batches; 6 layers × ~30 min = ~3h wall-clock across 4 GPUs = ~0.75 GPU-h + LLM calls. Estimate is conservative and stands.
- **M2 — gpu_hours**: plan~2.0h → matches — F1 computation is a matrix operation on cached codes; the compute is dominated by writing per-feature × per-concept F1 tensor (10,240 features × ~200 concepts × 6 layers ≈ 12M entries) — negligible on GPU when done as batched sparse ops. GPU is only for the neuron-arm forward pass on the residual stream (already cached). Stands.
- **M3 — gpu_hours**: plan~1.0h → matches — PCA fit + random rotation + shuffle are numpy operations on cached activations; F1 code is reused from M2. Stands.
- **M4 — n_pairs, gpu_hours**: plan `n_features=500, top_k=20, held_out_k=20, low_k=20`, ~1.5h → matches — this is the auto-interp gate scope declared by the plan (LLM rate-limited); no re-bind needed. GPU for UniRef activation extraction fits in the estimate.
- **M5 — gpu_hours, metric**: plan~1.0h, paired-Wilcoxon on PR-AUC → matches — linear-probe training on cached codes is fast; paired-Wilcoxon on K=50 concepts × 3 seeds is standard for this design. Stands.
- **M6 — n_pairs, sites, metric, gpu_hours**: plan `M_features=4, doses=[0.5,1,2,4]·σ_f, arms=[no_steer, sae_clamp, mean_add, random_clamp], site=L*, plausibility_band_factor=1.5`, ~3.0h → matches — generation-heavy but M_features × doses × arms × seeds × 25-seq batches ≈ 4×4×4×3×25 = 4800 generations of ≤128 residues fits in 3h across 4 GPUs. Site is `L*` (data-driven from M2, not hardcoded). Stands.

reconciliation_status: ok

## Rationale

The claim stage already pinned three mechanism directions (Unit Interpretation, Decision Auditing, Causal Intervention) and stamped `SAE dictionary decomposition is fixed` in the FINAL_PROPOSAL handoff. Routing has effectively one degree of freedom: which mechanism *family* holds the dictionary. **SAE** (family 5, submethod SAE) is the only catalog family that (a) is a residual-stream dictionary decomposition, (b) is downstream-composable with steering + probing, and (c) accepts pretrained checkpoints without retraining. Transcoder / Crosscoder / ICA-Lens are all inapplicable — the pretrained checkpoints under `$MODEL_DIR/I*-esm2-650m/` are single-site autoencoders, not transcoders/crosscoders, and ICA is a training-free baseline (would replace, not complement, the pretrained SAEs).

The routing is aligned with the direction tags in `mechanism_strategy.directions` (Unit Interpretation = SAE feature decoding = M1/M2/M4; Decision Auditing = coverage / novel-concept surfacing / superposition ladder = M2/M3/M4; Causal Intervention = feature-clamp steering = M6). `families_already_settled` is empty (round 1). No re-routing needed.

Auto-selection under `AUTO_PROCEED=true` picks the `[recommended]` candidate #1 (SAE) automatically per Phase 1.5 Step 5.
