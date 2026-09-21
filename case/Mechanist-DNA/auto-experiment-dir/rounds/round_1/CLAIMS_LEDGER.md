# Claim Ledger — Feature Steering an α-Helix Knob in Evo2-7B

**Direction**: Validate that Evo2-7B Layer-26 SAE features selectively responding to α-helix can causally control protein secondary structure via feature-amplification during autoregressive DNA generation.
**Date**: 2026-07-18
**Pipeline**: completed | **Iteration**: 6.8/10 "almost" (1/6, terminated on positive verdict)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 α-helix-selective SAE feature set exists (M0-gated) | conditional (set-level, AUROC 0.86–0.89) | ⚪ INTEGRITY_ONLY (WARN, cap) | relabeled established→conditional; eukaryote replicated | ⚪ conditional set-level, cross-organism, real |
| C2 amplification → dose-response α-helix rise | supported (ρ=0.76, p=1.7e-5, +0.36) | ✅ PASS (robustness 1.00, SS-swap) | PASS held ("strongest part") | ✓ supported + verify PASS |
| C3 specificity / causally manipulable knob | supported (clean-dose α=8, random-null) | ⚪ INTEGRITY_ONLY (was INCONCLUSIVE) | ② harness fixed; helix-axis specificity | ⚪ repaired — specific vs 33-dir null (p=0.029) |

---
## C1 — α-helix-selective Layer-26 SAE feature set exists (M0-gated)
- **Statement**: In Evo2-7B's Layer-26 SAE, a non-empty set of features selectively marks α-helix codons, beyond confounds and multiple-testing chance.
- **Origin**: task.md — given behavior, upstream half (existence/selectivity), validated by the M0 phenomenon-validation gate.
- **Data**: Natural CDS (RefSeq/Ensembl/ENA) with experimental PDB structures; per-residue DSSP labels mapped to codons via CDS↔protein alignment — provenance=adapted; used=648 E. coli / 181,359 codons + 350,268 H. sapiens codons (both established). No reverse-translation. Cross-organism replication complete.
- **Models**: Evo2-7B (StripedHyena2) · Layer-26 BatchTopK SAE (exp 8, k=64, ~32,768 features)
- **Method**: Per-codon F1 + AUROC of ~32,768 SAE features vs DSSP helix label; BH-FDR-controlled feature set S=19; confound controls.
- **Main experiment**: conditional (set-level) — set AUROC 0.89 (E. coli) / 0.858 (H. sapiens); confound-only 0.50-0.53; shuffle-null ~0.50; 321-380 FDR-sig; best single feature 0.64 (<τ=0.75); f/28741 rank-0. Disclosure: pre-registered seed_jaccard=0.0 is degenerate (empty strict set); relaxed-set cross-seed Jaccard ~0.7-0.8.
- **Verify**: INTEGRITY_ONLY (Stage 2 deferred by MAX_VERIFY_CLAIMS cap); Phase-2 integrity=WARN (label overclaim + undisclosed seed stat)
- **Iteration**: ⓪ honesty-relabel accepted (reviewer "strongly warranted") — established→conditional on set-level/single-feature grounds (NOT organism-limited); seed_jaccard disclosed; eukaryote leg corrected to COMPLETE & POSITIVE (6/6 human configs established, 350k codons, set AUROC 0.858)
- **Final**: ⚪ integrity_only → conditional (set-level); phenomenon real and GENERAL (replicated E. coli + H. sapiens, confound/FDR/null-controlled). Swap-test deferred (cap).
- **Caveats**: relabeled established→conditional (set-level, not single-feature; NOT organism-limited); degenerate seed_jaccard disclosed (real relaxed-set Jaccard ~0.7-0.8); cross-organism replication complete; no single feature clears τ=0.75 (peak 0.64)
- **Artifacts**: refine-logs/EXPERIMENT_RESULTS.md#M0, results/m0_feature_set.json, results/m0_eukaryote_*.json
- **Figures**:
  - ![A distributed set of ~19 Layer-26 SAE features detects α-helix codons, replicating across E. coli and H. sapiens (set AUROC 0.89 / 0.86) far above confound and shuffle-null baselines (~0.5).](figures/C1/c1_setauroc_crossorg.png) — vector: `figures/C1/c1_setauroc_crossorg.pdf`

## C2 — Feature amplification raises α-helix content (dose-response)
- **Statement**: Amplifying the C1 α-helix feature set during autoregressive DNA generation increases the encoded-protein α-helix fraction, monotonically with amplification strength up to an optimum.
- **Origin**: task.md — given behavior, core causal claim (dose-response).
- **Data**: Evo2-7B steered generation → ORF translation → ESMFold → DSSP — provenance=constructed; used=110-150 generated/dose (165-257 folded-gated) × 8 doses × 3 seeds = 24 runs (> M1 power floor 101).
- **Models**: Evo2-7B + Layer-26 SAE steering hook (blocks.26.post_norm) · ESMFold · DSSP
- **Method**: Dose-response sweep α∈{-2,0,1,2,4,8,16,32}; Spearman trend of DSSP helix fraction vs α; optimal α*; quality gate.
- **Main experiment**: supported — Spearman ρ=0.759, p=1.7e-5; helix 0.475(α=0)→0.834(α=32), effect +0.36; α*=32 (max), clean optimum α=8 (+0.10 at baseline quality); valid-ORF 0.87-0.89(≤α8)/0.65(α16)/0.78(α32); pLDDT 57-80
- **Verify**: robustness=**1.00** (1/1 eligible) — method **pass** / dataset,model excluded (HC1); integrity Phase2=WARN, Phase9=WARN; verdict=**PASS**. SS-swap mkdssp→pydssp reproduces dose-response (ρ=0.886, effect +0.327 vs +0.332). Robust to SS-assignment axis; structure-predictor axis untested.
- **Iteration**: PASS held (reviewer: "strongest part of the paper"); ⓪ paper caveats carried, no back-edge
- **Final**: ✓ supported + verify PASS — robust to SS-assignment axis. Scope: structure-predictor axis untested.
- **Caveats**: PROXY readout (ESMFold-predicted structure + DSSP on generated sequences, not experimental GT); structure-predictor swap (ESMFold→OmegaFold) untested (network-infeasible); α*=32 at grid edge (still rising); valid-ORF non-monotone (dips 0.65 @α16, recovers 0.78 @α32); clean operating point α=8
- **Artifacts**: refine-logs/EXPERIMENT_RESULTS.md#M2, results/m2_dose_response_curve.json, verify/C2_dose_response_steering/
- **Figures**:
  - ![Dose-response: amplifying the α-helix feature set monotonically raises encoded-protein α-helix fraction from 0.48 (baseline) to 0.83 (α=32), Spearman ρ=0.76, p=1.7e-5; clean-quality optimum at α=8.](figures/C2/c2_dose_response.png) — vector: `figures/C2/c2_dose_response.pdf`
  - ![Capability control: valid-ORF rate holds at or above baseline (0.872) through α=8 and dips only at α≥16, confirming α=8 as the clean-quality optimum.](figures/C2/c2_quality_overlay.png) — vector: `figures/C2/c2_quality_overlay.pdf`

## C3 — Specificity / causally manipulable knob
- **Statement**: The amplification-induced α-helix rise is specific to the α-helix feature set — matched-control and (originally) β-sheet off-target features do not reproduce it, at a quality-preserved dose — establishing the feature set as a causally manipulable knob.
- **Origin**: task.md — given behavior, specificity/knob claim.
- **Data**: Steered generation across feature kinds (α-helix S vs matched-control vs β-sheet off-target vs 33-direction random null) → ESMFold → DSSP (helix + sheet) — provenance=constructed; used=36 runs (main grid) + 33 random-direction control (iteration ②).
- **Models**: Evo2-7B + Layer-26 SAE steering hook · ESMFold · DSSP
- **Method**: Clean-dose (α=8) specificity: S vs 33-direction norm-matched random null (PRIMARY) + matched-control; DSSP helix/sheet with capability (valid-ORF) matched; β-sheet-amplification arm reported honestly.
- **Main experiment**: supported (capability-preserved, random-null-controlled) — FIXED in iteration ②: decisive dose re-locked α=16→α=8 (valid-ORF 0.888 vs 0.667). At α=8: S helix +0.096; **S beats all 33 random directions (0/33 ≥ S, empirical p=0.029, z=3.02)**, capability matched (S valid-ORF 0.888 ≈ random 0.887); matched-control Mann-Whitney p=3.5e-4; β-amplification arm NEGATIVE (Δsheet +0.005). [Superseded prior degraded α=16 stat: S helix +0.285, p≤3e-23.]
- **Verify**: INTEGRITY_ONLY (was INCONCLUSIVE at Phase-2 mechanism-audit FAIL); integrity=WARN post-fix; swap-robustness deferred (cap)
- **Iteration**: ② main-experiment mechanism-harness fix — mechanism-audit re-run FAIL→WARN; C3 INCONCLUSIVE→INTEGRITY_ONLY. Falsified the symmetric "double dissociation" framing (only S→helix moves; β→sheet does not). Narrowed to helix-axis specificity at a capability-preserved dose.
- **Final**: 🟡 INCONCLUSIVE → ⚪ INTEGRITY_ONLY (repaired). Specificity SUPPORTED on the helix axis at a clean dose (S beats all 33 random directions, p=0.029, z=3.02); β-amplification arm honestly negative. Swap-robustness deferred (cap).
- **Caveats**: β-sheet amplification arm is a NEGATIVE result (Δsheet +0.005) — helix-axis specificity + β-does-not-raise-helix, NOT a symmetric double dissociation; effect modest in absolute terms (+0.096) though cleanly separated from random null (z=3.0); mechanism-audit residual WARN (α not in σ_proj units; α=8 at capability-preserved edge); swap-robustness not evaluated (cap)
- **Artifacts**: refine-logs/EXPERIMENT_RESULTS.md#M3-v2, results/m3_specificity_summary_v2.json, code/m3_random_control.py, verify/C3_specificity_double_dissociation/main_experiment_audit/MECHANISM_AUDIT.md
- **Figures**:
  - ![At the clean dose α=8, only the α-helix feature set raises helix (0.49→0.59) while lowering sheet; matched-control and β-sheet off-target change neither — the β-sheet amplification arm is an honest negative (Δsheet +0.005).](figures/C3/c3_arm_helix_sheet.png) — vector: `figures/C3/c3_arm_helix_sheet.pdf`
  - ![Specificity vs a genuine 33-direction random-null: the α-helix set's helix gain (+0.096) exceeds all 33 norm-matched random directions (0/33 ≥ S; p=0.029, z=3.0), with capability matched (valid-ORF 0.89 ≈ 0.89).](figures/C3/c3_random_null.png) — vector: `figures/C3/c3_random_null.pdf`

---
## Journey Summary
- **Claim**: Faithful capture of task.md behavior → top idea "Feature Steering an α-Helix Knob in Evo2-7B"; 3 claims (C1 M0-gated existence, C2 dose-response, C3 specificity/knob).
- **Mechanism strategy**: n/a (mechanism given)
- **Mechanism routing**: family = SAE feature steering / feature amplification (Layer-26, site blocks.26.post_norm), committed directly (Mode B)
- **Experiment**: M(-1) setup + M0 grid + M1 calibration + 60 steering runs (M2 24, M3 36); gpu_ids⊆{2,3,4,5}; headline POSITIVE — C1 established, C2 supported (+0.36, p=1.7e-5), C3 supported. One scheduler stall recovered.
- **Verify**: 3 claims: 1 PASS (C2, robustness 1.00 SS-swap) / 1 INCONCLUSIVE (C3, mechanism-audit FAIL — degraded dose) / 1 INTEGRITY_ONLY (C1, cap+label WARN); Phase2=FAIL(C3)/Phase9=WARN(C2). C2 robust to SS-assignment; predictor axis untested.
- **Iteration**: 1/6 iterations, 0 claim-reentries. C3 fixed via ② harness repair (α=16→α=8 + genuine 33-direction random-null control; audit FAIL→WARN) → INCONCLUSIVE→INTEGRITY_ONLY. C1 ⓪ relabel established→conditional + seed disclosed + eukaryote corrected to complete. C2 PASS held. Reviewer 5.5→6.8 "almost"; terminated on positive verdict. A real mkdssp-PATH bug caught+fixed mid-run.
- **Figures**: 5 across 3 claims (C1:1 cross-organism set-AUROC bar, C2:2 dose-response + quality overlay, C3:2 arm helix/sheet bar + 33-direction random-null box) — all rendered ok (PDF+PNG from real result JSONs). See `figures/INDEX.md`.

## Open Items
- **C1**: cross-organism replication COMPLETE — H. sapiens leg established (6/6 configs, 350,268 codons, set AUROC 0.858) alongside E. coli; generality confirmed. Remaining: C1 swap-test deferred by MAX_VERIFY_CLAIMS cap — upgrade via `/auto-verify C1 — resume: true`.
- **C3**: REPAIRED in iteration ② (INCONCLUSIVE→INTEGRITY_ONLY; audit FAIL→WARN). Decisive dose re-locked to α=8 + genuine 33-direction random-null control (S beats all 33, p=0.029, z=3.02). Remaining: (a) β-sheet amplification arm is a documented NEGATIVE (Δsheet +0.005) — helix-axis specificity, not symmetric double dissociation; (b) swap-robustness deferred by cap — `/auto-verify C3 — resume: true`; (c) mechanism-audit residual WARN (α not in σ_proj units; α=8 at capability-preserved edge).
- **C2**: structure-predictor robustness axis (ESMFold→OmegaFold) untested (network-infeasible this pass) — SS-assignment axis PASSed; flagged future work.
- **Cross-run**: ~11.7 iteration GPU-hours (6.1 productive + 5.4 wasted on the mkdssp-PATH-bug run + ~0.2 pilots); GPU 2 held by another user the entire run — all compute on {3,4,5} ⊆ {2,3,4,5}.
