# Claim Ledger — Reproduction of Five SAE-on-ESM-2 Interpretability Claims

**Direction**: Sparse autoencoders recover biologically interpretable features inside protein language models (task.md)
**Date**: 2026-07-15 → 2026-07-15
**Pipeline**: completed | **Iteration**: 6/10 "ready" (2/6)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| c1 feature-count | partial [prov. under-power] — ratio 19.3× MET, absolute 58% of target | ⚪ integrity_only (cap) | iter 2 type-② L9 rerun @ 3000×300; under-power falsified, ratio 12.9× still MET, 95% CI [1034,1825] excludes 2548 | ⚠ well-characterized partial — direction robust; absolute credibly not met |
| c2 concept-alignment | partial [prov. under-power] — SAE=15 vs neurons=0 at primary | ✓ PASS robustness=1.0 (ESM-2-8M swap SAE=14 vs 15) | PASS held; paper-mandatory caveat on absolute count | ✓ verify PASS — SAE≫neurons scale-invariant |
| c3 superposition ladder | partial [prov. under-power] — ladder holds; Δ=15 (target ≥20) | ⚪ integrity_only (cap) | no fix — optional characterization skipped per STOP rule | ⚠ partial (⚪ swap-deferred) — ladder direction met; Δ near-miss at primary |
| c4 novel-concept LLM | not-supported [prov. criterion-null + under-power] — 0/100 novel | ⚪ integrity_only (cap) | no fix — criterion-redesign reviewer-deprioritized | ✗ not-supported — assay failure, NOT evidence against phenomenon |
| c5a annotation-filling | not-supported [prov. under-power] — p=0.19, SAE≈neurons | ⚪ integrity_only (cap) | iter 1 type-② dead-code + probe fix; credible negative SAE 0.539 vs neurons 0.548, p=0.516 | ✗ not-supported — credible negative under fair test |
| c5b steering | not-supported [prov. under-power] — no_steer > every steered arm | ⚪ integrity_only (cap) | no fix — dose-ladder widening reviewer-deprioritized | ✗ not-supported — evidence limited by weak intervention sweep |

---
## c1 — Feature count: SAE ≥ 10× raw-neuron interpretable count per layer
- **Statement**: SAE trained on ESM-2-650M residual-stream activations surfaces up to ~2,548 interpretable latent features per layer, ≥10× the raw-neuron interpretable count at the same layer, on at least one of ESM-2-650M layers {1,9,18,24,30,33}.
- **Origin**: task.md — given claim 1
- **Data**: Swiss-Prot test residues — provenance=existing; available=Swiss-Prot test 10k seqs, used=1500 seqs (387,195 residues) [iter 2: L9-only at 3000 seqs × 300 features]; UniRef supplement skipped (pre-tokenized); subset: used 15% of the planned SP-test pool; UniRef fell back to SP windows only
- **Models**: ESM-2-650M
- **Method**: M1 — per-layer interpretable-feature-count harness: forward pass over Swiss-Prot test, SAE latent stats (density, dead-frac), auto-interp gate (K=20 top windows, τ_auto_gate=0.3) via gpt-5.4; ratio SAE-interp-count / neuron-interp-count per layer. Family=Feature Dictionary Learning / SAE — screen → decode → verify → recover
- **Main experiment**: partial [provisional — suspected under-power] — best layer L9: SAE_interp = 1480 (target ~2548, 58%), ratio SAE/neuron = 19.3× (target ≥10×, MET); LLM-gate pass estimated on 100-feature sample, ±15% margin.
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=WARN; verdict=INTEGRITY_ONLY (stage2 skipped by max_verify_claims cap)
- **Iteration**: well-characterized partial after L9 rerun (iter 2 type-② plan_script_rerun @ 3000 seqs × 300 gated features): under-power hypothesis falsified — SAE gate pass rate stable at 14.3% (±4% 95% CI); SAE_interp=1430, 95% CI [1034, 1825] excludes target 2548; ratio SAE/neurons=12.9× (still MET target ≥10×). Reviewer bumped score 5→6 explicitly for quality-of-characterization improvement.
- **Final**: ⚠ well-characterized partial — direction (ratio ≥10×) robust across expanded sample; absolute count target credibly not met (⚪ swap-test deferred by max_verify_claims cap)
- **Caveats**: [suspected under-power: used_n 1500/10000 seqs (15%), LLM-gate sample 100/full-dictionary] — falsified by iter 2 L9 rerun; well-characterized partial; phase-2 audit WARN: auto-interp proxy GT + scope under-power (see verify/c1_sae_interpretable_feature_count/main_experiment_audit/); note: post-iter-2 L9 uses expanded sample; other 5 layers remain at baseline 1500/100 — a future swap test would need to decide which
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M1, refine-logs/EXPERIMENT_RESULTS.md#M1, runs/m1/, verify/c1_sae_interpretable_feature_count/, runs/iteration_round_2/m1_c1_gate_expand_L9/
- **Figures**:
  - ![Per-layer estimated interpretable SAE features vs raw-neuron features (ESM-2-650M). Ratio SAE/neuron ≥10× is MET at every non-L1 layer (8.0–19.3×); absolute target ~2548 (dashed line) is not hit at any layer — best is L9 at 1480 (main experiment) / 1430 (iter-2 rerun @ 3000 seqs × 300 features, 95% CI [1034, 1825]).](figures/c1/c1_per_layer_bar.png) — vector: figures/c1/c1_per_layer_bar.pdf

## c2 — Concept-alignment gap: SAE ~143 vs neurons ~46/15 clean
- **Statement**: SAE features align with up to ~143 distinct Swiss-Prot biological concepts under identical per-residue F1 protocol (τ_F1=0.5, q_top=0.99), whereas raw ESM-2 neurons cover only ~46 concepts of which only ~15 are cleanly recovered (τ_clean=0.7).
- **Origin**: task.md — given claim 2
- **Data**: Swiss-Prot per-residue annotations (full uniprot_sprot.dat re-downloaded — 466,006 entries) — provenance=existing; available=Swiss-Prot test 10k seqs, used=1500 seqs (322,287 annotated residues eligible); 400 fine-grained sub-typed concepts (≥25 positive residues); subset: concept universe finer-grained than reference paper; ~15% of planned test residue pool
- **Models**: ESM-2-650M
- **Method**: M2 — per-(unit,concept) residue-level F1 alignment under identical protocol across SAE and neuron arms; union over layers; sensitivity sweep at (q_top × τ_F1) grid. Family=Feature Dictionary Learning / SAE
- **Main experiment**: partial [provisional — suspected under-power] — primary (q=0.99, τ=0.5): SAE=15, neurons=0 (target SAE~143 / neurons~46/15 clean not hit; ratio infinite at primary); at looser τ=0.3: SAE=65, neurons=0 (SAE~143 approached).
- **Verify**: robustness=1.0 — method n/a / dataset n/a / model pass; integrity=WARN; verdict=PASS (ESM-2-650M→8M scale swap: SAE=14 vs 15, neurons=0 in both, ratio=∞ preserved)
- **Iteration**: PASS held — no iteration fix; reviewer confirmed the paper's strongest positive across all 3 iterations. Comparative gap replicated robustly (SAE≫neurons under 650M→8M swap); absolute count NOT hit at primary but approached at τ=0.3. Paper-mandatory caveat: state comparative headline reproduces but absolute count does not.
- **Final**: ✓ verify PASS (robustness 1.0) — SAE≫neurons direction is scale-invariant (ESM-2-650M SAE=15, ESM-2-8M SAE=14, neurons=0 in both); absolute-count target still not met at either scale (paper-mandatory caveat)
- **Caveats**: [suspected under-power: used_n 1500/10000 seqs (15%); concept universe finer-grained than likely reference aggregation]; phase-2 audit WARN: scope under-power vs. target ~143 (see verify/c2_sae_concept_alignment_gap/main_experiment_audit/)
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M2, refine-logs/EXPERIMENT_RESULTS.md#M2, runs/m2_concept_alignment/, verify/c2_sae_concept_alignment_gap/
- **Figures**:

  #### Sensitivity sweep of Swiss-Prot concept coverage (SAE vs raw neurons) across (q_top, τ_F1) grid, for the main ESM-2-650M experiment and the ESM-2-8M model-swap verify variant. SAE≫neurons gap is preserved across the 80× parameter-count reduction: SAE covered=15 (650M) vs 14 (8M) at primary (q=0.99, τ=0.5); neurons=0 in both.

  | $q_{\text{top}}$ | $\tau_{F1}$ | SAE (650M) | Neu (650M) | Ratio (650M) | SAE (8M) | Neu (8M) | Ratio (8M) |
  |---|---|---|---|---|---|---|---|
  | 0.95 | 0.3 | 64 | 2 | 32.0× | 56 | 2 | 28.0× |
  | 0.95 | 0.5 | 16 | 1 | 16.0× | 13 | 2 | 6.5× |
  | 0.95 | 0.7 | 3 | 0 | ∞ | 2 | 0 | ∞ |
  | 0.99 | 0.3 | 65 | 0 | ∞ | 58 | 3 | 19.3× |
  | 0.99 * | 0.5 | 15 | 0 | ∞ | 14 | 0 | ∞ |
  | 0.99 | 0.7 | 3 | 0 | ∞ | 2 | 0 | ∞ |

  `*` primary setting used for the main verdict.

  Source `.tex`: `figures/c2/c2_sensitivity_table.tex`

## c3 — Superposition (specificity-control ladder)
- **Statement**: The SAE-vs-neuron concept-alignment gap is direct evidence PLMs encode biological concepts in superposition rather than in single units — surviving three matched specificity controls (random-orthogonal, PCA, shuffled-SAE) at the same alignment protocol.
- **Origin**: task.md — given claim 3
- **Data**: same activations as M2; no new data draw — provenance=existing; available=same as M2, used=same as M2 (1500 SP-test seqs)
- **Models**: ESM-2-650M
- **Method**: M3 — three baseline arms under M2 protocol: (A) random-orthogonal rotation of d=1280 residual basis, (B) PCA basis of same dim, (C) shuffled-SAE (per-feature residue-shuffle); 3 seeds per random arm; ordered-coverage ladder + SAE-vs-PCA margin. Family=Feature Dictionary Learning / SAE + representation analysis controls
- **Main experiment**: partial [provisional — suspected under-power] — primary (q=0.99, τ=0.5): SAE=15, PCA=0, random-rotation=0.0±0.0, neurons=0, shuffled-SAE=0.0±0.0; strict Δ_SAE-PCA≥20 near-miss at 15; at τ=0.3 Δ=65 (well above 20).
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=WARN; verdict=INTEGRITY_ONLY (stage2 skipped by max_verify_claims cap)
- **Iteration**: no fix — optional characterization run skipped per reviewer + STOP-rule termination. By analogy to c1's under-power falsification, c3's Δ_SAE-PCA=15 is likely a real limit rather than a compute artifact (conjectured, not tested). Ladder direction remains robust (SAE > every control at primary; Δ=65 at τ=0.3).
- **Final**: ⚠ partial (⚪ swap-test deferred by max_verify_claims cap) — ladder direction met; strict Δ margin near-miss at primary (met at τ=0.3); paper caveat needed
- **Caveats**: [suspected under-power: used_n 1500/10000 seqs (15%)] — inference-by-analogy from c1: likely a real limit not a compute artifact; phase-2 audit WARN: strict margin not met; direction supported (see verify/c3_sae_superposition_specificity/main_experiment_audit/)
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M3, refine-logs/EXPERIMENT_RESULTS.md#M3, runs/m3_superposition_ladder/, verify/c3_sae_superposition_specificity/
- **Figures**:
  - ![Six-arm specificity ladder (Swiss-Prot concept coverage under identical F1 protocol) at the primary setting (q_top=0.99, τ_F1=0.5) and at the looser setting (τ_F1=0.3). SAE strictly beats every control (PCA, random-orthogonal-rotation, raw neurons, shuffled-SAE); strict target Δ_SAE-PCA ≥ 20 is a near-miss at primary (Δ=15) but clearly met at τ=0.3 (Δ=65).](figures/c3/c3_ladder_bar.png) — vector: figures/c3/c3_ladder_bar.pdf

## c4 — Novel-concept discovery via LLM auto-interpretation
- **Statement**: ≥10% of Swiss-Prot-unaligned SAE features receive coherent, non-synonym LLM auto-interp labels (gpt-5.4 via dmxapi.cn) at above-random-feature-control specificity, evidencing biological concepts absent from the annotation dictionary.
- **Origin**: task.md — given claim 4
- **Data**: Swiss-Prot windows only (UniRef unavailable — pre-tokenized parquet with no raw sequence column) — provenance=existing; available=planned 500 features × 60 windows, used=100 features × 45 windows/feature; 50 control features (matched fire_frac); subset: 20% of planned features; UniRef windows unavailable; SP-only fallback
- **Models**: ESM-2-650M, gpt-5.4 (auto-interp LLM)
- **Method**: M4 — three-prompt auto-interp: (A) label elicit from top-20 windows, (B) predictivity score on 20 held-out + 20 low-activation, (C) synonym check against Swiss-Prot vocabulary; s_auto ≥ τ_auto=0.3 AND synonym-check-pass ⇒ novel-concept-coherent. Family=Feature Dictionary Learning / SAE + LLM auto-interpretation
- **Main experiment**: not-supported [provisional — criterion-design null + suspected under-power] — 0/100 features passed (real arm) and 0/50 (control arm) — the strict synonym-check-null gate has zero acceptance rate on either arm, so it cannot discriminate. LLM did emit coherent-sounding labels (e.g. 'C2H2 zinc finger alpha-helix start', 'Hydrophobic signal peptide core', 'Gly-rich small-residue transmembrane helix motif') but every one was flagged as a paraphrase.
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=WARN; verdict=INTEGRITY_ONLY (stage2 skipped by max_verify_claims cap)
- **Iteration**: no fix — reviewer explicitly ranked as lowest strategic priority (criterion redesign has 'review-optics risk' after seeing zero/zero; not evidence against the phenomenon). Framed as assay-failure, not reproduction-failure.
- **Final**: ✗ not-supported — assay failure (criterion non-discriminative), NOT evidence against novel-concept phenomenon (⚪ swap-test deferred by max_verify_claims cap; criterion redesign required before swap-test would make sense)
- **Caveats**: [suspected under-power: n_features 100/500 (20%); UniRef windows unavailable — SP fallback shrinks LLM sample pool]; criterion-design: LLM synonym-check catches every vocabulary-adjacent label — the metric cannot discriminate; stricter novelty prompt or embedding-distance test needed; phase-2 audit WARN: synonym-check GT proxy + scope + criterion non-discriminative (see verify/c4_novel_concept_discovery/main_experiment_audit/); reviewer-deprioritized: criterion-redesign has review-optics risk after seeing zero/zero and is low-priority under budget
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M4, refine-logs/EXPERIMENT_RESULTS.md#M4, runs/m4_novel_concept_llm/, runs/cache/llm_calls/, verify/c4_novel_concept_discovery/
- **Figures**: *(judgment-skipped — 0/100 null on both real and control arms; the assay-failure story is fully carried by prose.)*

## c5a — Downstream utility: SAE-probe > neuron-probe on annotation-filling
- **Statement**: SAE-feature linear probes outperform matched-capacity raw-neuron linear probes on held-out Swiss-Prot annotation filling across the top-K=50 most-prevalent concepts (paired-Wilcoxon p<0.05).
- **Origin**: task.md — given claim 5 (sub-claim 5a: annotation-filling utility)
- **Data**: Swiss-Prot train (probe fit) + test (evaluation) parquet; best layer L*=9 from M2 — provenance=existing; available=Swiss-Prot train/test splits, used=top-50 concepts × 3 seeds × 2 arms = 300 probes; 30 concepts effectively scored (20 had no test positives in the 25k test subsample → NaN → excluded); subset: test subsample 25k residues (of 387k) [iter 1: 50k]; per-concept train cap 5k pos + 10k neg; 20/50 concepts dropped due to zero test positives
- **Models**: ESM-2-650M
- **Method**: M5 — per-residue binary linear probes on SAE code Z_{L*} vs raw residual H_{L*}, 50 concepts × 3 seeds; per-concept PR-AUC + paired-Wilcoxon(SAE, neuron). Baseline probe: SGDClassifier(max_iter=30) [dead-code path]; iter-1 fix: SGDClassifier(loss='log_loss', max_iter=1500, tol=1e-4, class_weight='balanced'). Family=Probing
- **Main experiment**: not-supported [provisional — suspected under-power] — mean PR-AUC(SAE)=0.5907, mean PR-AUC(neurons)=0.5906; paired-Wilcoxon one-sided p=0.191; per-concept diff mean 0.0002, std 0.081; only 30/50 concepts scored.
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=WARN; verdict=INTEGRITY_ONLY (stage2 skipped by max_verify_claims cap)
- **Iteration**: credible negative (iter 1 type-② plan_script_rerun): dead code + SGD substitution corrected — new probe is well-converged, class-balanced SGDClassifier(max_iter=1500, tol=1e-4, class_weight='balanced'); test subsample 25k→50k. Post-fix mean PR-AUC(SAE)=0.5391 < mean PR-AUC(neurons)=0.5476, p=0.516 — SAE probe does NOT beat raw-neuron probe under fair test. Iter-3 reviewer note: scope-limited negative, 20/50 dropped concepts create selection-on-evaluable caveat.
- **Final**: ✗ not-supported — credible negative under corrected fair test (SAE 0.539 ≈ neurons 0.548, p=0.516) (⚪ swap-test deferred by max_verify_claims cap)
- **Caveats**: [suspected under-power: 30/50 concepts scored (20 dropped); SGD max_iter=30 likely under-fits both arms; test subsample 25k of 387k residues] — probe under-fit RESOLVED via iter-1 fix; 30/50 scope-limitation persists; probe-choice compromise: SGDClassifier substituted for LogisticRegression(lbfgs) due to wall-clock hang on 322k×10240 dense matrix — RESOLVED via iter-1 well-converged SGD substitute; phase-2 audit WARN: dead code in per_concept_pr_auc + scope + SGD substitution — dead-code fixed in iter 1
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M5, refine-logs/EXPERIMENT_RESULTS.md#M5, runs/m5_annotation_probes/, verify/c5a_annotation_filling_probes/, runs/iteration_round_1/m5_c5a_logreg_fix/
- **Figures**:

  #### Per-arm PR-AUC on annotation-filling before (baseline: SGDClassifier max_iter=30, no class_weight, dead-code path) and after (iter-1 fix: well-converged log-loss SGD, class_weight='balanced', tol-early-stop) the probe-code fix. The fair-test null holds: mean PR-AUC(SAE)=0.539 ≤ mean PR-AUC(neurons)=0.548, paired-Wilcoxon p=0.516. Credible negative — the SAE code does not carry more annotation-decodable info than the raw residual at layer 9.

  | Stage | n_concepts | mean PR-AUC (SAE) | mean PR-AUC (neurons) | Paired-Wilcoxon W | one-sided p (SAE > Neu) |
  |---|---|---|---|---|---|
  | Before fix (baseline) | 30 | 0.5907 | 0.5906 | 276.0 | 0.191 |
  | After fix (iter-1) | 30 | 0.5391 | 0.5476 | 231.0 | 0.516 |

  **Interpretation.** The fair-test null holds both before and after the probe-code fix (well-converged log-loss SGD, `class_weight='balanced'`, tol-early-stop). Post-fix, mean PR-AUC(SAE) = 0.5391 is actually below mean PR-AUC(neurons) = 0.5476, with paired-Wilcoxon p = 0.516. Credible negative: the SAE code at layer 9 does not carry more annotation-decodable information than the raw residual.

  Source `.tex`: `figures/c5a/c5a_before_after_table.tex`

## c5b — Downstream utility: SAE-feature-clamp steering
- **Statement**: Clamping a labeled SAE feature during ESM-2 masked-token generation steers the generation toward the target biological property with monotone dose-response, above no-steer / random-clamp / mean-activation-addition baselines and within a preserved-pseudo-perplexity plausibility band, on at least one concrete target property.
- **Origin**: task.md — given claim 5 (sub-claim 5b: causal steering utility)
- **Data**: Swiss-Prot test short-sequence seeds (≤128 residues, 20% masked); 4 external checkers (signal_peptide, transmembrane, zinc_binding, n_glycosylation) — only 3 features surfaced from M4 — provenance=existing + constructed (generations); available=planned 4 features × 4 doses × 4 arms × 3 seeds × 25 seqs = 4800 generations, used=3 features × 4 doses × 4 arms × 3 seeds × 10 seqs = 1440 generations (30% of planned); subset: only 3 of planned 4 target-property features surfaced by M4; n_seqs_per_batch 10/25
- **Models**: ESM-2-650M
- **Method**: M6 — SAE-feature-clamp steering with dose ladder α∈{0.5,1,2,4}σ_f at layer L*=9; vs no-steer + mean-activation-addition + random-clamp; measure yield under external checker + pseudo-perplexity plausibility band (≤1.5× no-steer mean). Family=Feature Dictionary Learning / SAE + Steering Vectors composition
- **Main experiment**: not-supported [provisional — suspected under-power] — per feature: TM 3998 (no_steer 0.333 > sae_clamp 0.200 all α > mean_add 0.133–0.200 ≈ random_clamp 0.200); Zn 4209 (no_steer 0.133 > all others 0.100); SP 1240 (no_steer 0.167 > all others 0.133). Plausibility band-pass ≥ 0.67 on all steered arms — no off-distribution collapse.
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=WARN; verdict=INTEGRITY_ONLY (stage2 skipped by max_verify_claims cap)
- **Iteration**: no fix — reviewer explicitly ranked as 'moderate but uncertain' (dose ladder <1 OOM likely insufficient in principle; even widened ladder has low upside per reviewer). Framed as 'protocol too weak', not 'steering demonstrably fails'.
- **Final**: ✗ not-supported — evidence limited by weak intervention sweep (dose ladder <1 OOM, 3/4 features, 30% of planned generations); cannot confidently reject nor confirm (⚪ swap-test deferred by max_verify_claims cap)
- **Caveats**: [suspected under-power: 3/4 features (75%); 10/25 seqs per batch (40%); 30% of planned generations]; steering-vs-detector gap: M4 labels are correlational features, not verified causal, so clamp may lack causal sufficiency; rule-based checkers may be too strict on short refilled sequences; phase-2 audit WARN: scope (experiment); mechanism-audit WARN: sweep <3 OOM / <5 grid points (see verify/c5b_feature_clamp_steering/main_experiment_audit/); reviewer-deprioritized: widened ladder considered low-upside under budget; not a confirmed falsification
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M6, refine-logs/EXPERIMENT_RESULTS.md#M6, runs/m6_feature_clamp_steering/, verify/c5b_feature_clamp_steering/
- **Figures**:

  #### Yield per (target-property feature × steering arm × dose α∈{0.5,1,2,4}·σ_f) across 3 features (TM helix 3998, Zn finger 4209, SP core 1240). No-steer baseline yields dominate every steered arm on every feature; plausibility band-pass ≥ 0.67 on all steered arms shows no off-distribution collapse. Interpretation: steering is ineffective (not destructive) under this narrow ≤1-OOM dose ladder + rule-based checkers.

  | Feature | Property | no_steer (baseline) | sae_clamp (α∈{0.5,1,2,4}·σ_f) | mean_add | random_clamp |
  |---|---|---|---|---|---|
  | 3998 | TM helix | **0.333** | 0.200 (all α) | 0.133–0.200 | 0.200 (all α) |
  | 4209 | Zn finger | **0.133** | 0.100 (all α) | 0.100 (all α) | 0.100 (all α) |
  | 1240 | SP core | **0.167** | 0.133 (all α) | 0.133 (all α) | 0.133 (all α) |

  Notes: yield = fraction of the 30 generated sequences (10 seqs × 3 seeds) that satisfy the target property's rule-based checker (Kyte–Doolittle window for TM, ProSite regex for Zn, SignalP-like N-terminal core rule for SP). Baseline yields dominate every steered arm on every feature. Plausibility band-pass ≥ 0.67 on all steered arms, so drops are not off-distribution collapse — the steering is ineffective, not destructive, under this narrow ≤1-OOM dose ladder.

  Source `.tex`: `figures/c5b/c5b_yield_matrix_table.tex`

---
## Journey Summary
- **Claim**: 5 given claims from task.md faithfully captured; behavior-source=given, mechanism-strategy=discovery. Top idea: unified reproduction harness across all five.
- **Mechanism strategy**: Unit Interpretation → Decision Auditing → Causal Intervention (rejected: Location [pre-committed], Tuning & Editing [user constraint], Formation Tracing [user constraint])
- **Mechanism routing**: family=Feature Dictionary Learning / SAE, submethod=SAE
- **Experiment**: 11 runs, ~2.14 GPU-hours (of 10-h budget), headline mixed — direction supported for c1/c2/c3, criterion-design failure c4, null c5a, negative c5b (all downscaled from plan)
- **Verify**: 6 claim(s): 1 PASS (c2, robustness=1.0 on ESM-2-650M→8M model swap) / 0 FAIL / 0 INCONCLUSIVE / 0 ZEV / 5 INTEGRITY_ONLY (cap=5, swap_off=0); Phase 2 WARN/Phase 9 PASS. Baseline integrity WARN on all 6 (no FAIL) — mostly scope/under-power caveats; c5a dead-code + c5b sweep-under-power surfaced by /experiment-audit + /mechanism-audit.
- **Iteration**: 2/6 iterations (iter 3 was ⓪-narrative, no budget), claim-reentries=0/2, score 6/10 verdict ready, termination=positive_verdict (3-D STOP satisfied). 3.01 GPU-h in iteration; aggregate 5.14/10 GPU-h. Type-② fixes: c5a (dead-code+SGD→LogReg substitute, credible negative p=0.516); c1 (L9 rerun 3000 seqs × 300 features, under-power falsified — ratio 12.9× MET, absolute 1430 [1034,1825] excludes target 2548).
- **Figures**: 5 across 5 claims (c1 bar, c2 sensitivity table, c3 grouped_bar, c5a before/after table, c5b yield matrix table); 1 judgment-skipped (c4 — 0/100 null on both arms, prose-only); 0 render-skipped, 0 errored.

## Open Items
- 5 claims INTEGRITY_ONLY (stage2_skip_reason=max_verify_claims_cap) — Stage-2 swap-tests deferred by MAX_VERIFY_CLAIMS=1 cap. Upgrade individually via `/auto-verify <id> — resume: true` (Phase 2 audit reused): c1, c3, c4, c5a, c5b. Note: c1 iter-2 L9 expanded sample vs 5 other layers still at baseline — future swap-test must decide which to use; c5a iter-1 uses fixed LogReg-substitute probe; c4 needs criterion-redesign before swap-test makes scientific sense.
- UniRef gap: /data/zhenqian/data/UniRef/data/*.parquet contains only tokenized input_ids (no raw sequences); M1/M4 fell back to Swiss-Prot windows only. Fix: obtain raw UniRef fasta or reconstruct sequences from tokens for a fuller LLM sample pool.
- Truncated .dat.gz on disk at $DATA_DIR/Swiss-Prot: workspace file was 50 KB (truncated); experiment agent downloaded the full 700 MB from UniProt and cached to runs/cache/swissprot/all_annotations.pkl (466,006 entries). Optional: replace on-disk truncated file with the full version.
- C4 criterion-design null (reviewer-deprioritized): LLM synonym-check flags every emitted label as vocabulary-adjacent, so 0/100 novel on both real AND control arm. Redesign as 3-way adjudication (paraphrase / coherent-but-not-in-SP / incoherent) or embedding-distance test — but reviewer noted 'review-optics risk' after seeing zero/zero result.
- C5b protocol weakness (reviewer-deprioritized): dose ladder α∈{0.5,1,2,4}·σ_f spans <1 OOM (target ≥3 OOM). Widen to {0,0.5,1,2,4,8,16,32} + all 4 features + batch 25 (~2-3 GPU-h) would test claim fairly but reviewer said 'low upside'.
- Concept-granularity mismatch conjecture (unresolved): using 400 fine-grained concepts (each Pfam family separately, each PTM sub-class separately) may depress absolute-count targets vs reference likely-coarser aggregation. Affects c1/c2/c3 absolute counts. Explicit paper caveat needed.
- ~4.78 GPU-h unused of 10-h budget (deliberately preserved per reviewer STOP-AND-WRITE recommendation). Available for optional c3 characterization run (~1.5 GPU-h) or one INTEGRITY_ONLY swap-test upgrade if needed.
