# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Feature Dictionary Learning / SAE
chosen_idea_title: Hardening the α-helix feature-steering knob in Evo2-7B (Round 2)
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/feature-dictionary-learning/SAE/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/SKILL.md

## Candidates

1. **[recommended]** Feature Dictionary Learning / SAE — GIVEN family (Mode B, `MECHANISM=given`). The load-bearing internal object is the pre-trained Layer-26 BatchTopK SAE (expansion-8, k=64, ~32,768 atoms) at `blocks.26.post_norm`. C1 is a monosemantic-feature-**set** claim (which SAE atoms selectively mark α-helix codons) — squarely Feature Dictionary Learning. C2/C3 use the family's native intervention mode ("SAE-based steering" — scale a feature's weight before reconstruction and add the decoder delta back to the residual stream), which the FDL family explicitly owns as a downstream tool. Pre-trained SAE reused per FDL's practical rule (never trained from scratch).
   - path: skills/mechanism-skills/feature-dictionary-learning/SAE/SKILL.md
2. Representation and Parameter Analysis / Steering features — the amplification of an *existing internal feature* is described verbatim under this family's "Steering features" submethod. It is the intervention-side view of the same operation; kept as the sibling reference for the C2/C3 steering step. Not chosen as the primary slot because the feature basis itself (an over-complete learned SAE dictionary, and the C1 set-existence claim) is FDL-owned.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/SKILL.md

## Composition plan
Screen → decode → verify → recover, grounded on the frozen SAE dictionary:
- **Screen/decode (C1, M0):** re-confirm set S of α-helix-selective SAE features on cached Layer-26 activations with real DSSP codon labels — per-feature + set-level AUROC, confound/FDR/null controls. Re-identify a stronger β set (`beta_features_v2`). Pure FDL/SAE feature analysis on caches (no new GPU generation). ~cheap.
- **Verify (C2/C3, M1–M3):** SAE-based steering — amplify S's latents during Evo2-7B autoregressive DNA decoding via the decoder delta `Δ = Σ_{f∈S}(c·σ_proj)·unit_dir(f)` at `blocks.26.post_norm`; read out DNA→protein→{ESMFold, OmegaFold}→DSSP α-helix fraction (pLDDT-weighted primary). Dose-response over σ_proj-unit grid (M2, interior c*); specificity vs ≥30-direction norm-matched random null + matched-control + β off-target arm (M3).
- **Downstream analysis (post-processing, not a family):** split-sample c* selection, cluster-robust bootstrap CIs over prompt-seed clusters, BH-FDR on secondary tests, pLDDT-covariate regression. These are statistics on collected effect vectors, not a mechanism family.
- Cost notes: M0 cheap (cached); M1 ~2–4h (σ_proj calib + dual-predictor baseline); M2 ~60 runs; M3 ~72 runs + ≥30-dir null sub-run. All `max_parallel:4` on GPUs 3,4,5,6.

## Plan reconciliation
<!-- One row per method_sensitive field on the intervention milestones (M1–M3). resource_fidelity: not-strict → fields present & re-bindable per plan line 210. -->
- n_pairs (steering analog = n_per_dose / samples-per-arm): plan=300/dose (M2/M3), ≥80/dir null → **re-bound: keep n_per_dose=300; null sub-run n_per_dose≥120** — round-1 M1 power calc gave n≈101 for Δhelix=0.1 @ power 0.8; 300 gives headroom for the narrower CIs round-2 demands (final N re-confirmed from M1's round-2 power calc). Serves the milestone intent (adequate power), not a scope change.
- sites: plan=`blocks.26.post_norm` (single SAE-native site) → **matches** — pinned by HC1; SAE dictionary exists only at this site (validated M(-1) by reproducing paper features f/28741, f/22326). No block-count sweep (see EXPERIMENT_TIPS steering-block-selection).
- metric: plan = pLDDT-weighted α-helix fraction (primary; hard-gated + threshold-sweep as sensitivity) → **matches** — the SAE-steering readout supports it directly (per-residue DSSP SS × per-residue pLDDT weight).
- gpu_hours: plan ~M1 8h + M2 60×4h + M3 72×4h+null → **revised down ~1.5–2.5h per M2/M3 run** — round-1 measured ~1h/generation-fold run (M1=3930s for 400 prompts+fold); dual predictor (ESMFold+OmegaFold) roughly doubles the fold cost, n_per_dose=300<400 offsets. Net suite still ~250–350 GPU-h (dual-predictor triples-vs-round-1 as the plan intends); displayed at the Phase-4 gate.
reconciliation_status: ok

## Rationale
Mechanism is GIVEN in task.md / EXPERIMENT_PLAN.md (`chosen_mechanism`: SAE feature steering / feature amplification). Committed directly (Mode B) — no re-routing, no mini-prompt. Feature Dictionary Learning / SAE is the canonical catalog home: the SAE is the substrate for both the C1 set-existence claim and the C2/C3 steering intervention (FDL lists "SAE-based steering" among its native tools). The Representation-and-Parameter-Analysis "Steering features" submethod is the intervention-side alias of the same operation and is cited as the sibling. All `method_sensitive` fields reconcile to `matches`/benign re-bind → status ok.
