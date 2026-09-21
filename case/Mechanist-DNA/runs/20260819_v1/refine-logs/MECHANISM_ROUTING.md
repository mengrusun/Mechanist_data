# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Feature Dictionary Learning / sparse-autoencoder
chosen_idea_title: α-Helix-Directed DNA Generation by SAE Feature Amplification in Evo2-7B (Layer-26 Mixed SAE)
effective_domain: mechanistic-interpretability (genomics)
routing_mode: Mode B (MECHANISM=given — user-named mechanism committed directly, no /mechanism-skills re-routing per orchestrator instruction)
candidate_paths:
  - skills/mechanism-skills/Feature Dictionary Learning/sparse-autoencoder/SKILL.md

## Candidates

1. **[recommended / committed]** Feature Dictionary Learning / sparse-autoencoder — user-given mechanism. The released Evo-2 Layer-26 Mixed SAE (tied-weight TopK, d_sae=32768, k=64, hooked at the `blocks.26` residual) is used as-released to (a) *identify* α-helix-selective latents on labeled coding DNA and (b) *amplify* exactly those latents' decoder directions during autoregressive decoding. This is the canonical catalog mapping of "SAE feature amplification".
   - path: skills/mechanism-skills/Feature Dictionary Learning/sparse-autoencoder/SKILL.md

## Composition plan
screen → decode → verify → recover:
- **screen (C1/M1)**: forward E. coli CDS through Evo2-7B, capture the layer-26 residual (`blocks.26` output), SAE-encode to 32768 latents, rank latents by α-helix-vs-rest AUROC (gene/cluster unit) with a gene/codon-position-preserving shuffle null (BH-FDR) and matched β-sheet/coil control features; freeze feature set S on validation.
- **decode / intervene (C2,C3 / M2,M3)**: residual-add `x' = x + α·Σ_{i∈S} s_i·d̂_i` at `blocks.26` at every decoding step (d̂_i = unit decoder direction W[:,i]; s_i = median positive train activation), sweep α, ESMFold→DSSP %-helix readout.
- **verify / specificity (M-CTRL)**: random-feature, β-sheet-feature, and null-direction control arms at α*; helix gain of S must exceed all controls.
- **post-processing (not a mechanism family)**: AUROC/FDR statistics, cluster-blocked bootstrap CIs, dose-response curve fitting — downstream analysis on the SAE-derived signals, not a separate mechanism.

## Plan reconciliation
<!-- resource_fidelity: strict pins all method_sensitive fields exact; nothing to re-bind. -->
- feature-set size K: plan grid K∈{1,5,10,20} cap 32, chosen on validation → matches (SAE-native; no re-bind).
- read/intervene site: plan = layer-26 residual (`blocks.26`) → matches (pinned by the released SAE, exact).
- metric: plan = per-codon latent AUROC (selectivity) + ESMFold→DSSP %-helix (generation) → matches (SAE encode + folding readout are exactly what the family supports).
- gpu_hours: plan ~6.5–8 h total → unchanged (SAE encode/steer is negligible marginal cost over the Evo2 forward/generate passes already budgeted).
reconciliation_status: n/a   <!-- strict fidelity: method_sensitive fields pinned exact, no re-bind possible -->

## Rationale
MECHANISM=given: the user named the mechanism (SAE feature identification + amplification, released Evo-2 Layer-26 Mixed SAE) in task.md; the claim stage stamped it as `chosen_mechanism` in FINAL_PROPOSAL.md / EXPERIMENT_PLAN.md. Per the orchestrator's build-mode instruction this is committed directly (Phase 1.5 Mode B) — no `/mechanism-skills` candidate routing and no family mini-prompt. "SAE feature amplification" maps unambiguously to the catalog's Feature Dictionary Learning / sparse-autoencoder family. Under `resource_fidelity: strict` the SAE is used exactly as released (no retraining, no new SAE, no smaller model).
