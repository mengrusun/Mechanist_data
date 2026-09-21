---
title: "Experiment Plan — Hardening the α-Helix Feature-Steering Knob in Evo2-7B (given-validation × given, Round 2)"
behavior_source: given-validation
mechanism: given
chosen_mechanism: "SAE feature steering / feature amplification (amplify Layer-26 α-helix-selective SAE features at blocks.26.post_norm during autoregressive DNA generation; σ_proj-unit dose-response coefficient sweep)"
mechanism_strategy: n/a
resource_fidelity: not-strict   # NOT the given+given reproduction combo → no strict harness; model & SAE still pinned by HC1, data at FULL scale (rigor round, no downscaling) under the 4-GPU budget
max_parallel: 4
gpus: [3, 4, 5, 6]
max_verify_claims: 3
date: 2026-07-19
claim_source: task.md
claims: [C1, C2, C3]
round: 2
builds_on: rounds/round_1/
---

# Experiment Plan — Round 2 (Rigor Hardening)

**Reading order:** M(-1) setup-precondition (incl. **second structure predictor**) → **M0 phenomenon-validation gate (C1)** (reuse frozen S) → M1 harness calibration (**σ_proj** + **dual-predictor** + **pLDDT-weighted** readout) → **M2 σ_proj dose-response (C2)** (dual predictor, pLDDT-in-stats, interior plateau) → **M3 specificity / knob (C3)** (locked interior dose, ≥30-direction null, tight CIs, β-arm resolution). Mechanism milestones (M1–M3) run only after M0 returns `established`/`conditional`.

## HARD CONSTRAINTS (plan rules — non-negotiable, apply to every milestone)

- **HC1 Model fidelity:** main experiment uses **only** Evo2-7B (`/data1/share_model/evo2/evo2_7b/evo2_7b.pt`) + the pre-trained Layer-26 BatchTopK SAE (`/data1/share_model/evo2/evo2_sae_layer26_mixed/sae-layer26-mixed-expansion_8-k_64.pt`, expansion 8, k=64, ~32,768 features, site `blocks.26.post_norm`). No substitute genomic LM / SAE. The **second structure predictor** is an added readout axis, NOT a substitute LM/SAE.
- **HC2 Compute:** ≤4 GPUs simultaneously, pinned **CUDA_VISIBLE_DEVICES=3,4,5,6**; `max_parallel: 4`. Generous budget — this is a rigor round: run at FULL scale, never downscale/subset/skip to save cost.
- **HC3 Data construction:** feature discovery MUST use **natural CDS + real DSSP labels from experimental structures** after proper **CDS↔protein alignment**. **Reverse-translation of proteins into DNA is forbidden.** (M0 reuses the frozen round-1 S + `data/` caches, built under this rule.)
- **HC4 No degradation:** full intended scale. Tooling gaps: **install-and-continue** for pip/conda deps + downloadable weights (incl. the second predictor); **STOP + approval-needed** only for sudo/apt/root, credentials, licenses, or disk-quota. Never silently drop the pLDDT stats, the second predictor, or the swap-robustness.

## Round-2 improvement → milestone map (all six encoded as requirements)

| # | Required improvement | Encoded in |
|---|---|---|
| 1 | pLDDT into statistics (weighted fraction; mean-pLDDT-per-dose + conditioning; threshold sweep) | M1 (define), **M2** (dose-response), **M3** (specificity) |
| 2 | Second, independent structure predictor (OmegaFold/ColabFold-AF2) | **M(-1)** (pre-stage), M1 (wire), **M2**, **M3** |
| 3 | Swap-robustness on all three claims (`MAX_VERIFY_CLAIMS=3`) | claim authoring (C1/C2/C3 swap axes) + top-metadata `max_verify_claims: 3` |
| 4 | σ_proj-unit dosing + mapped interior plateau | **M1** (σ_proj calibration), **M2** (refined grid brackets interior α\*) |
| 5 | Tighter statistics on C3 (narrower CIs; ≥30-direction norm-matched null PRIMARY; effect sizes+CIs) | **M3** |
| 6 | β-sheet arm resolution (stronger β set → double dissociation, else documented helix-axis specificity) | **M0** (β re-identification), **M3** (decision) |

## Statistics & pre-registration protocol (round-2 — from external review; binding on M2/M3)

These rules are pre-registered before any M2/M3 run and prevent the round from over-claiming through analytic flexibility. They are consumed by M2/M3 verbatim.

- **P1 Primary estimand (resolves "survives conditioning" ambiguity).** The **single primary structural endpoint is the pLDDT-weighted α-helix fraction** (confidence folded in as a weight, not a filter). Hard-gating, the pLDDT-threshold sweep {50,60,70,80}, and pLDDT-covariate regression are **sensitivity analyses only** — reported, never the headline number. One primary test per claim (C2: Spearman dose-trend on the primary endpoint; C3: S-vs-random-null empirical p on the primary endpoint).
- **P2 Split-sample dose selection (kills M2→M3 post-selection leakage).** The interior optimum **c\* is chosen on a dedicated selection seed-set** (e.g. seed 42) using the primary predictor only; **C2's headline trend and C3's decisive specificity statistic are computed at the fixed c\* on held-out seed-sets** (200, 201) never used for c\* selection. c\* is frozen before the held-out readout.
- **P3 Predictor independence.** Generation, ORF filtering, and c\* selection are frozen **without** the second predictor. Both predictors are then run **once** on the held-out sequences; second-predictor agreement is the robustness result. (Caveat recorded: ESMFold/AF2-family share training priors — agreement is corroboration, not full independence; OmegaFold if used is the more independent axis.)
- **P4 Stronger random-null matching (beyond norm).** Each random direction matches S on **(i) feature-count |S|, (ii) injection-site norm ||S.base_dir||, and (iii) natural-activation weighting**; additionally **report per-direction generation impact** (Evo2 perplexity / logit-drift and valid-ORF rate) so S is not special merely by a milder or harsher global perturbation. Directions with degenerate generation are reported, not silently dropped.
- **P5 Multiplicity & effective N.** One pre-registered primary test per claim; **all secondary analyses (per-dose Mann–Whitney, threshold sweep, matched-control, β-arm) are BH-FDR adjusted**. Report **effective N**, not raw ORF count: AR generations sharing a prompt/seed family are correlated → aggregate/cluster by prompt-seed and use cluster-robust CIs (bootstrap **over prompt-seed clusters**, not over individual ORFs). Power (M1) is computed on this corrected plan.
- **P6 Composition / length / ORF-collider confounds.** Helix fraction is driven by protein length, hydrophobicity, AA composition, low-complexity and TM-segment enrichment; and **valid-ORF selection is a potential collider** (steering may shift which sequences pass the ORF filter). Mitigation: (a) report the dose/direction effect on a **length- and composition-matched subsample** in addition to the raw effect; (b) report how the valid-ORF population itself shifts with dose/direction; (c) the **capability-preserved (interior) dose is chosen precisely so the ORF population is not degraded** — report helix at **matched valid-ORF** across arms so the effect is not an ORF-selection artifact.
- **P7 Interior optimum = selectivity, not toxicity.** Treat generation quality as a potential **mediator/confound**, not just a descriptive guardrail: c\* is defined as the helix maximum **within the capability-preserved region** (valid-ORF ≈ baseline), and the S-vs-null contrast at c\* is compared **at matched capability** — so a mid-dose peak reflects feature selectivity, not a degradation tradeoff.
- **P8 Prompt/context balancing.** The **same prompt/context set** is used across all doses, feature-kinds, random directions, predictors, and seeds, so site-26 intervention effects are not confounded by context differences across conditions.
- **P9 β-arm logic (explicit).** Failing to find a β-selective set that raises sheet is **NOT** evidence for helix specificity. C3's helix specificity rests **entirely on S-vs-random-null (+ matched-control)** on the helix axis; the β-arm only *upgrades* the result to a symmetric double dissociation **if** the stronger β set positively raises sheet. Documented-negative β is reported as such and does not weaken or strengthen the S-vs-null conclusion.

---

## M(-1): Setup precondition (environment + tooling, incl. second predictor)

**Kind:** setup-gate (not a phenomenon gate).
**Claims:** none (enabler).
**Goal:** make the `scientist` env able to (a) load Evo2-7B + run/steer its SAE, (b) reuse natural-CDS + experimental-structure data + DSSP, (c) run **ESMFold**, and (d) run a **second, independent structure predictor** (OmegaFold or ColabFold/AF2).

**Actions & halt policy (per HC4):**
- **Install-and-continue (default, NOT a gate):** verify the round-1 `scientist` env still loads Evo2-7B + the Layer-26 SAE + ESMFold + DSSP; **pre-stage the second predictor and its weights**. For **OmegaFold**: resolve the round-1 torch-pin download problem (install into an isolated venv/conda subenv pinned to OmegaFold's torch, or fetch the release weights directly and bypass the pinned installer). If OmegaFold's weights still download impractically slowly, **pick a predictor whose weights download cleanly** (ColabFold/AF2 params via the standard param download, or the ESM-family alternative) and record the choice — the requirement is *a second independent predictor*, not specifically OmegaFold.
- **STOP + approval-needed (only these):** a dependency whose only install route is `sudo`/`apt-get`/root; a licensed/credentialed download (e.g., full AlphaFold DBs) with no free path; a disk-quota action the pipeline cannot perform. STOP and surface — do **not** degrade (do not drop the second predictor, the pLDDT stats, or structure verification).

**Verification (must all pass before M0):**
- `python -c "import evo2, torch; ...load Evo2-7B on cuda:3; forward a 512-nt seq; extract Layer-26 residual at blocks.26.post_norm"` succeeds.
- Load the Layer-26 SAE checkpoint; confirm encode→(top-k=64)→decode round-trips; dictionary ≈ 32,768.
- DSSP runs on a test PDB (mkdssp + pydssp both return per-residue SS); **ESMFold** predicts a test sequence returning coords+pLDDT; the **second predictor** predicts the same test sequence returning coords+its own confidence (pLDDT or equivalent) — and DSSP runs on its output.
- Frozen round-1 S loads from `rounds/round_1/results/m0_feature_set.json` (19 helix + 5 β + 19 matched-control features + s_f scales).

**Expected output:** `results/setup_report.json` (versions, weight paths, per-tool smoke-test PASS/FAIL, **the chosen second predictor + why**, confirmation the frozen S loaded).
**Priority:** MUST-RUN (blocker). **Estimated GPU-hours:** ~1–2h (mostly downloads; second-predictor weights dominate).

---

## M0: Phenomenon-validation gate — the α-helix-selective SAE feature set exists (C1)

**kind: phenomenon-validation**   <!-- STABLE MACHINE MARKER — experiment stage keys on this, not the title -->
**Claims:** C1.
**Depends on:** M(-1).

**Hypothesis (H1):** In Evo2-7B's Layer-26 SAE there is a non-empty set of features whose activation selectively marks α-helix codons, beyond confounds and multiple-testing chance (set-level; no single detector required).

**Reuse policy (round-2):** the behavior is already established cross-organism, so **reuse the frozen round-1 feature set S** (`rounds/round_1/results/m0_feature_set.json`) and the cached activations/labels under `data/` as the starting point. M0 still **re-emits the four-state verdict** on the frozen S using the round-1 selectivity computation (recompute set-level test AUROC + confound/FDR/null checks from the caches — no re-mining, no reverse-translation). The gate is a re-confirmation, not a fresh discovery, but it is a real gate: if the frozen S fails to re-confirm, the four-state branch below governs.

**Data rule (HC3 — binding):**
- **Source:** natural **CDS** from public genomic databases whose protein products have **experimental** PDB structures. No reverse-translation.
- **Labels:** **DSSP** on the experimental PDB structures → per-residue SS; map to codons via **CDS↔protein alignment** (translate CDS, align to PDB SEQRES/observed, require exact/near-exact match, propagate residue SS to its codon; drop misaligned/gapped/low-coverage regions). Helix = DSSP {H,G,I} (primary), H-only variant.
- **Provenance/splits (`/data-rule`):** deduplicate by sequence identity (≤30% between splits via mmseqs2); train/threshold-selection / validation / held-out test splits by protein cluster, not residue; balance prokaryote+eukaryote.
- **Scale (full, per HC2 budget):** the round-1 caches (prokaryote + eukaryote, ≳500k labeled codons) meet the floor; re-use at full scale. Floor: ≥300 proteins / ≥50k codons and ≥50 helix-positive proteins per split. If a re-run yields fewer, STOP and surface.

**Method:** using cached Layer-26 SAE feature activations per codon, recompute for each of the ~32,768 features its α-helix-vs-rest discrimination (AUROC threshold-free + F1) on the validation split, final on held-out test. Re-confirm **set S** = features clearing τ_auc/τ_f1 with a selectivity margin over non-helix, β-sheet, coil (frozen before test). Report the **set-level combined AUROC** (round-1: 0.89–0.90) as the primary C1 statistic — C1 is a **set-level** claim (round-1 finding: no single feature clears AUROC 0.75; f/28741 top single at 0.64).

**β-sheet re-identification (improvement 6 — β-arm resolution, step 1):** on the same caches, attempt to identify a **stronger β-sheet-selective feature set** (β-vs-rest set AUROC, same selectivity/confound/FDR bar as S) than the round-1 5-feature β set. Record the best β set + its set-level AUROC into `m0_feature_set.json` as `beta_features_v2`. Whether it clears the bar decides the M3 β-arm framing (symmetric double dissociation vs documented helix-axis specificity) — the M0 verdict itself does not gate on the β set.

**Confound controls (report):** length, codon/GC frequency, position-in-CDS, coding-vs-noncoding, and **label identity** — via matched negative sampling and partial-correlation/logistic controls; show S survives. **FDR:** Benjamini–Hochberg across 32,768 features.

**Trivial-explanation check:** shuffle-label null → AUROC ≈ 0.5; feature-permutation null; per-codon vs per-nucleotide aggregation consistent.

**Robustness (M0 paraphrase/seed/decoding, domain-adapted):** holds across ≥2 organism groups (prokaryote + eukaryote); stable set across ≥3 cluster-split seeds (report Jaccard); stable across helix def (HGI vs H-only) and threshold choice.

**Four-state verdict (governs downstream):**
- `established` → set-level S clears the bar, confounds controlled, FDR-significant, robust → run M1–M3 on frozen S.
- `conditional` → holds only in a sub-condition → tag C1/C2/C3 `conditional`, restrict M2/M3 to that condition (runtime scoping; plan not rewritten).
- `not-established` → S fails re-confirmation after controls → STOP pipeline, write negative-result report (skip M1–M3 + verify).
- `inconclusive` → M0 test itself broken/underpowered → fix data/script, re-run M0.

**Pass criteria (explicit):** set-level test AUROC ≥ **0.80** (round-1 achieved 0.89–0.90), margin over β-sheet/coil ≥ 0.1 AUROC, BH-FDR q<0.05, robustness overlaps as above, shuffle-null AUROC ≈ 0.5. *(Expected: re-confirms `established` — round-1 result.)*

**Grid:**
  organism_group: [eukaryote, prokaryote]
  helix_def: [HGI, H_only]
  split_seed: [42, 200, 201]
**Cmd template:** `CUDA_VISIBLE_DEVICES=3,4,5,6 python m0_feature_selectivity.py --evo2 /data1/share_model/evo2/evo2_7b/evo2_7b.pt --sae /data1/share_model/evo2/evo2_sae_layer26_mixed/sae-layer26-mixed-expansion_8-k_64.pt --reuse_frozen rounds/round_1/results/m0_feature_set.json --organism ${organism_group} --helix_def ${helix_def} --split_seed ${split_seed} --tau_auc 0.75 --tau_f1 0.3 --fdr bh --identify_beta_v2 --out results/m0_${organism_group}_${helix_def}_s${split_seed}.json`
**Expected output (template):** `results/m0_${organism_group}_${helix_def}_s${split_seed}.json` + consolidated `results/m0_feature_set.json` (frozen S + `beta_features_v2` + per-feature stats + four-state verdict).
**Priority:** MUST-RUN (hard gate). **Estimated GPU-hours per run:** ~1h (activations cached from round 1); 12 runs, `max_parallel: 4`.

---

## M1: Generation + verification harness (σ_proj calibration + dual predictor + pLDDT-weighted readout; enables C2/C3)

**Claims:** enabler for C2/C3.
**Depends on:** [M0].

**Goal:** build/calibrate the DNA→protein→structure→DSSP readout, the steering hook in **σ_proj units**, the **second-predictor** path, and the **pLDDT-weighted** statistic, so M2/M3 measure signal, not pipeline noise.

**Components:**
- **σ_proj calibration (improvement 4):** at `blocks.26.post_norm`, compute the **projection std σ_proj** of S's activation onto its steering direction over a natural-CDS reference set (and record the per-feature s_f already in the frozen file). Define the steering coefficient in **σ_proj units**: steered activation = base + `(c · σ_proj) · unit_dir(S)` (equivalently clamp each f∈S to `act + α·s_f` with α mapped to c·σ_proj). Persist the raw-α ↔ σ_proj mapping so round-1 α values (0,1,2,4,8,16,32) are locatable on the new c-axis.
- **Steering hook:** amplify S's latents during **autoregressive** nucleotide decoding; decode SAE→residual; continue the forward pass. Fixed generation config (prompt policy, temperature, length) held constant across doses.
- **Translate + validity filter:** ORF-aware translation; keep valid ORFs (start, in-frame, no premature stop, length ≥ L_min). Record valid-ORF rate (capability metric).
- **Dual structure predictor (improvement 2):** **ESMFold** (default) **and** the M(-1) second predictor on the *same* translated proteins; both emit coords + a per-residue confidence (pLDDT or equivalent).
- **DSSP readout:** mkdssp (primary) + pydssp (swap); α-helix fraction = helix residues / resolved residues (HGI primary, H-only secondary).
- **pLDDT-in-statistics definitions (improvement 1):** define, once here, (a) the **hard-gated** α-helix fraction (drop residues/structures below a pLDDT gate) and (b) the **pLDDT-weighted** α-helix fraction (each residue's/structure's helix contribution weighted by its confidence); define the **mean-pLDDT-per-dose** report, the **pLDDT-conditioning** analysis (stratify into pLDDT bins + pLDDT-as-covariate regression of helix on dose), and the **pLDDT-threshold sensitivity sweep** grid (e.g. {50,60,70,80}). These definitions are consumed unchanged by M2/M3.
- **Power/CI calibration (improvement 5):** from the α=0 baseline variance under each predictor, set **N per dose** to reach a target CI half-width on the α-helix fraction (target detect Δhelix ≥ ~0.1 at power 0.8; size up so the modest C3 effect gets narrow CIs — aim samples-per-dose ≳ 300 valid-ORF structures, more at the decisive dose).

**Verification:** on α=0 (c=0), both predictors recover a sane baseline helix fraction with stable variance; a naturally helix-rich prompt set reads higher than a helix-poor set under both predictors; pLDDT-weighted and hard-gated fractions agree at baseline.
**method_sensitive: [n_pairs, sites, metric, gpu_hours]**
**Cmd:** `CUDA_VISIBLE_DEVICES=3,4,5,6 python m1_harness_calibrate.py --evo2 ... --sae ... --feature_set results/m0_feature_set.json --compute_sigma_proj --predictors esmfold,${predictor2} --n_samples 500 --plddt_grid 50,60,70,80 --out results/m1_calibration.json`
**Expected output:** `results/m1_calibration.json` (σ_proj + raw-α↔c mapping, baseline helix dist per predictor, noise, N/dose for target CI, valid-ORF & pLDDT-gated + pLDDT-weighted baselines).
**Priority:** MUST-RUN. **Estimated GPU-hours:** ~8h (dual structure prediction dominates).

---

## M2: σ_proj dose-response steering sweep (C2 — causal control, interior optimum, dual predictor, pLDDT-in-stats)

**Claims:** C2.
**Depends on:** [M0, M1].

**Hypothesis (H2):** encoded-protein α-helix fraction increases monotonically with the σ_proj-unit coefficient c up to an **interior** optimum c\*, robustly across two predictors and under pLDDT-weighting, not confounded by pLDDT changing with dose.

**Design:** generate N (from M1, ≳300 valid-ORF) sequences per dose across a **σ_proj-unit grid refined to bracket the interior optimum** (improvement 4): points below, at, and above c\* with generation quality preserved on both sides of the peak, plus a negative c for sign check. Translate → predict with **both predictors** → DSSP → compute per-dose mean α-helix fraction with **CIs**, under **both** the hard-gated and **pLDDT-weighted** statistics.
- **Expected sign:** up. **Dose-response:** monotone non-decreasing over low-mid c; identify **interior c\*** = helix maximum **within the capability-preserved region** (P7), with the grid demonstrating the peak is **interior** (helix rises then plateaus/falls while capability holds) — not the largest c tested. **c\* is selected on the selection seed-set (seed 42) via the primary predictor only (P2); the headline trend is then read on held-out seeds (200, 201).**
- **Primary endpoint (P1):** the **pLDDT-weighted α-helix fraction**. **pLDDT-in-statistics (improvement 1):** report **mean pLDDT per dose** (per predictor); as **sensitivity analyses** show the rise survives hard-gating, the **pLDDT-threshold sweep {50,60,70,80}**, and **pLDDT-covariate regression** (dose coefficient stays positive & significant) — ruling out "helix rise is just pLDDT drifting with dose."
- **Second predictor (improvement 2, P3):** generation/ORF/c\* frozen without the second predictor; both predictors run once on held-out sequences; the positive trend and interior c\* must reproduce under the second predictor (report both curves; agreement is the C2 robustness result).
- **Confounds (P6):** report the dose effect on a **length- and composition-matched subsample** and at **matched valid-ORF**, plus how the valid-ORF population shifts with dose, so the trend is not an ORF-selection/composition artifact.
- **Statistics (P5):** primary = Spearman trend of the primary endpoint vs c on held-out seeds (H0: no trend); secondary per-dose Mann–Whitney vs c=0 are **BH-FDR adjusted**; effect sizes with CIs bootstrapped **over prompt-seed clusters** (effective N, not raw ORF count); N/dose from M1 power calc.

**Pass criteria (C2):** significant positive Spearman trend (p<0.05) on the **primary endpoint** on **held-out seeds** up to an **interior** c\*, with mean helix at c\* significantly above c=0 (Δ ≥ ~0.1, cluster-bootstrap CIs excluding 0), **reproduced under the second predictor**, and **stable across the pLDDT-threshold sensitivity sweep** and the composition-matched subsample; quality metrics within M1 tolerance up to c\*. If the second predictor disagrees on direction/interior-peak, C2 is downgraded to predictor-conditional and surfaced.
**method_sensitive: [n_pairs, sites, metric, gpu_hours]**
**Grid:**
  c_sigma: [-1, 0, 0.5, 1, 1.5, 2, 3, 4, 6, 8]   # σ_proj units; refined to bracket interior c* on both sides (raw-α↔c map in M1); exact points re-bindable at commit
  predictor: [esmfold, ${predictor2}]
  seed: [42, 200, 201]
**Cmd template:** `CUDA_VISIBLE_DEVICES=3,4,5,6 python m2_dose_response.py --evo2 ... --sae ... --feature_set results/m0_feature_set.json --c_sigma ${c_sigma} --predictor ${predictor} --seed ${seed} --n_per_dose 300 --plddt_stats weighted,gated --plddt_grid 50,60,70,80 --out results/m2_c${c_sigma}_${predictor}_s${seed}.json`
**Expected output (template):** `results/m2_c${c_sigma}_${predictor}_s${seed}.json` + `results/m2_dose_response_curve.json` (per-predictor curves, interior c\*, pLDDT-per-dose, pLDDT-conditioned trend, threshold-sweep table).
**Priority:** MUST-RUN. **Estimated GPU-hours per run:** ~4h; 10 c × 2 predictors × 3 seeds = 60 runs, `max_parallel: 4`.

---

## M3: Specificity / "causally manipulable knob" (C3 — locked interior dose, ≥30-direction null, tight CIs, dual predictor, β-arm resolution)

**Claims:** C3.
**Depends on:** [M0, M1, M2].

**Hypothesis (H3):** the helix rise at the capability-preserved interior dose is attributable to S's feature identity, not generic perturbation; S's effect exceeds a norm-matched random-null with a tight effect-size CI.

**Decisive dose (locked, improvement 4; P2/P7):** the specificity statistics are computed at the **M2 interior plateau dose c\*** (σ_proj units) — **c\* selected on the selection seed-set, C3 decisive stats computed on held-out seeds (P2, no post-selection leakage)** — where S's generation quality is preserved (valid-ORF ≈ baseline; P7), **NOT** at a grid-edge dose. The α-span trend (S rises then plateaus while controls stay flat) characterizes specificity as a trend; the decisive stat is locked to the interior operating point at matched capability.

**Controls:**
- **Random-direction null (PRIMARY specificity control, improvement 5; P4):** **≥30** (target ≥50 for tighter CIs) **independent** random directions, each = |S| SAE features drawn uniformly excluding S ∪ β ∪ matched-control, weighted by their own natural s_f and **rescaled to match ||S.base_dir||** at the injection site (norm-matched, σ_proj-consistent). **Per-direction generation impact (Evo2 perplexity/logit-drift + valid-ORF) is reported (P4)** so S is not special merely by a milder/harsher global perturbation. Steered at the locked c\* and read out on the **primary endpoint (pLDDT-weighted, P1)**. Primary statistic: **S's helix rise vs this null** — empirical one-sided p=(1+#{Δ_rand≥Δ_S})/(N+1), a z-score, **effect size with cluster-bootstrap CI (P5)**, at **matched valid-ORF (P6)**. More directions + more samples/dir than round-1's 33×80 for narrower CIs.
- **Matched-control feature(s):** the frozen 19 matched-control features (activation-matched to S), steered at c\* → expect no α-helix rise (secondary sanity control).
- **Off-target β-sheet (improvement 6 — β-arm resolution, decision; P9):** steer the **stronger β set** (`beta_features_v2` from M0 if it cleared the bar; else the round-1 β set) at c\* and along the c-span → **if it raises β-sheet (not helix)**, report the **symmetric double dissociation** (S↑helix not sheet; β↑sheet not helix). **Else** (β set does not raise sheet, as in round 1) report it as a **documented NEGATIVE / failed manipulation** and frame C3 explicitly as **helix-axis specificity**. **Per P9, the β outcome does NOT affect the S-vs-null conclusion** — helix specificity rests entirely on S-vs-random-null (+ matched-control); the β-arm only *upgrades* to a symmetric double dissociation when the β set positively raises sheet. M3 records which held.
- **Quality guardrail:** valid-ORF rate, Evo2 perplexity, pLDDT distribution stay within M1 tolerance across arms (no gaming via degenerate sequences).

**Round-2 rigor on the readout (improvements 1, 2, 5):**
- **Dual predictor:** compute S-vs-null specificity under **both** ESMFold and the second predictor; the specificity result must hold under both (report both).
- **pLDDT-in-statistics:** report the specificity contrast under **hard-gated and pLDDT-weighted** helix fraction, with **mean pLDDT per arm** (rule out S winning merely by higher confidence), and confirm the S-vs-null outlier status is **stable across the pLDDT-threshold sweep**.
- **Tighter CIs:** effect sizes (S − null-mean, S − matched) reported with bootstrap CIs; increased samples/seeds per arm vs round 1.

**Pass criteria (C3):** at the locked interior c\* **on held-out seeds (P2)**, on the **primary pLDDT-weighted endpoint (P1)** and at **matched valid-ORF (P6)**, S's helix increase is a significant outlier vs the ≥30-direction norm-matched null (empirical p<0.05, effect size + cluster-bootstrap CI excluding 0) AND ≫ matched-control, with valid-ORF/perplexity not degraded, **reproduced under the second predictor** and **stable across the pLDDT-threshold sensitivity sweep** → **C3: S is a causally manipulable, specific knob** (specificity established on the helix axis; β-arm reported as double dissociation if the stronger β set raised sheet, else documented negative → helix-axis specificity, per P9).
**method_sensitive: [n_pairs, sites, metric, gpu_hours]**
**Grid:**
  feature_kind: [alpha_helix_S, matched_control, beta_sheet_offtarget]
  c_sigma: [0, c_star_low, c_star, c_star_high]   # interior-plateau bracket from M2; bound at commit
  predictor: [esmfold, ${predictor2}]
  seed: [42, 200, 201]
**Cmd template:** `CUDA_VISIBLE_DEVICES=3,4,5,6 python m3_specificity.py --evo2 ... --sae ... --feature_set results/m0_feature_set.json --feature_kind ${feature_kind} --c_sigma ${c_sigma} --predictor ${predictor} --seed ${seed} --n_per_dose 300 --readout helix_and_sheet --plddt_stats weighted,gated --plddt_grid 50,60,70,80 --out results/m3_${feature_kind}_c${c_sigma}_${predictor}_s${seed}.json`
**Expected output (template):** `results/m3_${feature_kind}_c${c_sigma}_${predictor}_s${seed}.json` + `results/m3_specificity_summary.json`.
**Priority:** MUST-RUN. **Estimated GPU-hours per run:** ~4h; 3 kinds × 4 c × 2 predictors × 3 seeds = 72 runs, `max_parallel: 4`.

**Random-direction null sub-run (PRIMARY C3 statistic):** ≥30 (target ≥50) independent norm-matched directions at the locked c\* under **both predictors**, dispatched detached across GPUs {3,4,5,6}, `n_per_dose` ≥ round-1's 80 (size up for narrower CIs). Consolidated with the M3 arm data into `results/m3_specificity_summary.json` (locked dose c\*, S-vs-null primary stat with effect size + CI, per-predictor, pLDDT-stratified).
**Cmd:** `CUDA_VISIBLE_DEVICES=3 python m3_random_control.py --dir_start 0 --dir_end 16 --c_sigma c_star --predictor esmfold --n_per_dose 120 --out results/m3_random_control_esm_d0-16.json` (× ranges over GPUs 3/4/5/6, × both predictors).

---

## Run order & budget
1. M(-1) setup (incl. second-predictor pre-stage) → 2. **M0 gate** (12 runs, cached-fast) → branch on verdict → 3. M1 calibration (σ_proj + dual predictor + pLDDT defs + power) → 4. M2 σ_proj sweep (60 runs) → 5. M3 specificity (72 runs + random-null sub-run), all `max_parallel: 4` on GPUs 3,4,5,6.
- Queue routing: M0/M2/M3 declare `grid:` (+ `depends_on:`) → `/auto-experiment` Phase 4.B (`/experiment-queue`). M1 is a single calibration run (Phase 4.A).
- `method_sensitive` fields on M1–M3 may be re-bound by the experiment stage at commit time (n_pairs/sites/metric/gpu_hours; e.g. exact c-grid points, N/dose from M1's power calc, second-predictor id) without counting as a plan rewrite; M0 carries none (its data scale is fixed by HC3).
- Dual-predictor + larger null + larger N/dose roughly triple M2/M3 GPU-hours vs round 1 — intended (rigor round, generous budget, no downscaling).

## Claim → milestone traceability
- **C1** → M0 (gate; reuses frozen S; + β_v2 re-identification). **C2** → M1 + M2 (σ_proj sweep, dual predictor, pLDDT-in-stats, interior c\*). **C3** → M3 (locked interior dose, ≥30-direction null PRIMARY, tight CIs, dual predictor, pLDDT-stratified, β-arm resolution). All mechanism milestones `depends_on: [M0]`.
- **Swap-testability (`MAX_VERIFY_CLAIMS=3`):** C1 — DSSP algo / organism / metric; C2 — predictor / DSSP algo / trend stat / seed; C3 — predictor / null construction / DSSP algo / locked-dose. All three authored swap-testable.
