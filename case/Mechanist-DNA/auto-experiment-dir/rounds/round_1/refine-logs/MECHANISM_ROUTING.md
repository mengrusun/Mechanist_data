# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Feature Dictionary Learning / SAE (feature amplification steering)
chosen_idea_title: SAE feature steering — amplify Layer-26 α-helix-selective features during autoregressive DNA generation (dose-response)
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/feature-dictionary-learning/SAE/SKILL.md
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md

## Candidates

1. **[recommended / committed]** Feature Dictionary Learning / SAE (feature amplification steering) — GIVEN by task.md (`chosen_mechanism`, MECHANISM=given, Mode B direct commit). The pre-trained Layer-26 BatchTopK SAE decomposes Evo2-7B's residual stream into ~32,768 features; M0 localizes the α-helix-selective subset S via per-codon AUROC/F1 concept-alignment (InterPLM method), and the causal test amplifies S's latents (`act + α·s_f`) during autoregressive nucleotide decoding.
   - path: skills/mechanism-skills/feature-dictionary-learning/SAE/SKILL.md
2. Causal Attribution / patching (the intervention arm) — the amplification is a clamp/scale causal intervention on identified SAE latents; the specificity controls (matched-null feature, β-sheet off-target) are the double-dissociation form of activation patching.
   - path: skills/mechanism-skills/causal-attribution/patching/SKILL.md

## Composition plan
screen → decode → verify → recover:
- **screen (M0):** cache Layer-26 SAE feature activations per codon on natural CDS with real DSSP labels; score every feature's α-helix discrimination (AUROC/F1), apply selectivity margin over β-sheet/coil, confound + BH-FDR control → freeze feature set S. (SAE feature-dictionary localization.)
- **decode+calibrate (M1):** build DNA→translate→ESMFold→DSSP readout + the Layer-26 SAE steering hook; calibrate baseline helix distribution, valid-ORF/pLDDT gating, measurement noise, samples-per-dose.
- **verify (M2):** amplify S across a dose sweep α∈{-2..32}; Spearman dose-response of encoded-protein α-helix fraction; find α*. (Causal intervention.)
- **recover/specificity (M3):** matched-null features + β-sheet off-target feature at shared doses → double dissociation; quality guardrail (valid-ORF, perplexity, pLDDT) so the effect is not collapse-driven. (Post-processing: dissociation stats — not a mechanism family.)
Cost notes: M0 activation caching is the GPU-heavy screen (~3h/organism, shared across the 12 grid runs); M2/M3 GPU cost is dominated by ESMFold structure prediction, not Evo2 generation.

## Plan reconciliation
<!-- One row per method_sensitive field on the intervention milestones (M1–M3). resource_fidelity: not-strict, so re-binds allowed. -->
- n_pairs: plan=n_per_dose 300 (M2/M3) → matches — SAE feature amplification needs no paired-difference estimation (unlike attribution patching); 300 generations/dose is set by M1's power calc (detect Δhelix≥0.1 at power 0.8), retained.
- sites: plan=Layer-26 SAE feature space → matches — pinned by HC1 (only Layer-26 mixed SAE exists); the empirically-resolved residual read/write site is `blocks.26` input-normalized space (M(-1) site-resolution), the SAE's native activation space. No layer sweep (forbidden/unavailable).
- metric: plan=DSSP α-helix fraction on ESMFold-predicted structure of translated protein (+ β-sheet fraction for M3) → matches — this is the readout the SAE-steering claim requires; unchanged.
- gpu_hours: plan~M1 6h + M2 24×4h + M3 36×4h → revised ~ lower per-run for Evo2 generation (7B fits one A800, generation is fast) but ESMFold folding dominates; net per-run ≈ plan estimate. No material change; plan figure stands.
reconciliation_status: ok

## Rationale
#1 is committed directly because task.md names the mechanism (SAE feature steering / amplification) and the claim stage stamped it as `chosen_mechanism` — this is the first-class MECHANISM=given path, no /mechanism-skills routing or mini-prompt. The family maps cleanly to the canonical catalog entry **Feature Dictionary Learning / SAE**; the intervention/specificity arm is the **Causal Attribution / patching** double-dissociation. HC1 pins the exact SAE, so there is no alternative feature-dictionary submethod to route to.
