---
title: "Experiment Plan — α-Helix Feature Steering Knob in Evo2-7B (given-validation × given)"
behavior_source: given-validation
mechanism: given
chosen_mechanism: "SAE feature steering / feature amplification (amplify Layer-26 α-helix-selective SAE features during autoregressive DNA generation; dose-response coefficient sweep)"
mechanism_strategy: n/a
resource_fidelity: not-strict   # NOT the given+given reproduction combo → no strict harness; model & SAE still pinned by HC1, data planned at full scale under the generous 4-GPU budget
max_parallel: 4
gpus: [2, 3, 4, 5]
date: 2026-07-18
claim_source: task.md
claims: [C1, C2, C3]
---

# Experiment Plan

**Reading order:** M(-1) setup-precondition → **M0 phenomenon-validation gate (C1)** → M1 generation+verification harness → **M2 dose-response steering (C2)** → **M3 specificity / knob (C3)**. Mechanism milestones (M1–M3) run only after M0 returns `established`/`conditional`.

## HARD CONSTRAINTS (plan rules — non-negotiable, apply to every milestone)

- **HC1 Model fidelity:** main experiment uses **only** Evo2-7B (`/data1/share_model/evo2/evo2_7b/evo2_7b.pt`) + the pre-trained Layer-26 BatchTopK SAE (`/data1/share_model/evo2/evo2_sae_layer26_mixed/sae-layer26-mixed-expansion_8-k_64.pt`, expansion 8, k=64, ~32,768 features). No substitute genomic LM / SAE.
- **HC2 Compute:** ≤4 GPUs simultaneously, pinned **CUDA_VISIBLE_DEVICES=2,3,4,5**; `max_parallel: 4`. Generous budget — use it fully, never downscale to save cost.
- **HC3 Data construction:** feature discovery MUST use **natural CDS + real DSSP labels from experimental structures** after proper **CDS↔protein alignment**. **Reverse-translation of proteins into DNA is forbidden.**
- **HC4 No degradation:** full intended scale. Tooling gaps: **install-and-continue** for pip/conda deps + downloadable weights; **STOP + approval-needed** only for sudo/apt/root, credentials, licenses, or quota. Never silently subset/skip.

---

## M(-1): Setup precondition (environment + tooling)

**Kind:** setup-gate (not a phenomenon gate).
**Claims:** none (enabler).
**Goal:** make the `scientist` env able to (a) load Evo2-7B + run its SAE and steer it, (b) fetch natural CDS + experimental structures, (c) run DSSP, (d) predict structures.

**Actions & halt policy (per HC4):**
- **Install-and-continue (default, NOT a gate):** `pip`/`conda` installs — `evo2`/`vortex` (Arc `arcinstitute/evo2`), `biopython`, `fair-esm`/`esm` (+ ESMFold weights download), `pydssp` **or** conda-forge `dssp`/`mkdssp`, structure/CDS fetch utils. Download model/predictor weights. The experiment stage performs these and proceeds.
- **STOP + approval-needed (only these):** a dependency whose *only* install route is `sudo`/`apt-get`/root; a licensed/credentialed download (e.g., full AlphaFold DBs) with no free path; a disk-quota action the pipeline cannot perform. In that case STOP and surface — do **not** degrade (e.g., do not silently drop structure verification or swap DSSP for a heuristic).

**Verification (must all pass before M0):**
- `python -c "import evo2, torch; ...load Evo2-7B on cuda:2; forward a 512-nt seq; extract Layer-26 residual"` succeeds.
- Load the Layer-26 SAE checkpoint; confirm encode→(top-k=64)→decode round-trips; dictionary size ≈ 32,768.
- DSSP runs on a test PDB and returns per-residue SS; ESMFold predicts a test sequence and returns coords+pLDDT.

**Expected output:** `results/setup_report.json` (versions, weight paths, per-tool smoke-test PASS/FAIL, chosen structure predictor).
**Priority:** MUST-RUN (blocker). **Estimated GPU-hours:** ~1h (mostly downloads).

---

## M0: Phenomenon-validation gate — the α-helix-selective SAE feature set exists (C1)

**kind: phenomenon-validation**   <!-- STABLE MACHINE MARKER — experiment stage keys on this, not the title -->
**Claims:** C1.
**Depends on:** M(-1).

**Hypothesis (H1):** In Evo2-7B's Layer-26 SAE there is a non-empty set of features whose activation selectively marks α-helix codons, beyond confounds and multiple-testing chance.

**Data rule (HC3 — binding):**
- **Source:** natural **CDS** from public genomic databases (e.g., NCBI RefSeq / Ensembl / ENA) whose protein products have **experimental** 3D structures in the **PDB**. No reverse-translation.
- **Labels:** run **DSSP** on the experimental PDB structures → per-residue secondary structure; map to codons via **CDS↔protein alignment** (translate CDS, align to the PDB SEQRES/observed sequence, require exact/near-exact match, propagate residue SS to its codon; drop misaligned/gapped/low-coverage regions). Helix label = DSSP {H,G,I} (primary), with an **H-only** robustness variant.
- **Provenance/splits (see `/data-rule`):** deduplicate by sequence identity (e.g., ≤30% identity between splits via mmseqs2) to prevent leakage; **train/threshold-selection / validation / held-out test** splits by protein cluster, not by residue. Balance organisms (prokaryote+eukaryote, matching the mixed SAE).
- **Scale (full, per HC2 budget):** target ≥ ~3,000 structurally-resolved proteins (≳500k labeled codons) after QC; **minimum floor** ≥ 300 proteins / ≥ 50k codons and ≥ 50 helix-positive proteins per split (honors the ≥~50 statistical-reality floor with wide margin). If DB/QC yields fewer, STOP and surface (do not proceed underpowered).

**Method:** cache Layer-26 SAE feature activations for every codon position of every CDS (Evo2-7B forward, extract Layer-26 residual, SAE-encode). For each of the ~32,768 features, score α-helix-vs-rest discrimination at codon resolution:
- Primary metric: **F1** (thresholded activation as binary predictor of helix) and **AUROC** (threshold-free), per InterPLM-style concept alignment, computed on the validation split; final numbers on the **held-out test** split.
- Define the α-helix feature **set S** = features with test AUROC ≥ τ_auc and F1 ≥ τ_f1 with a **selectivity margin** over (i) non-helix, (ii) β-sheet, (iii) coil — chosen on validation, frozen before test.

**Confound controls (must be reported):** length, codon/GC frequency, position-in-CDS, coding-vs-noncoding, and **label identity** (helix vs the *specific* residue) — via matched negative sampling and partial-correlation / logistic controls; show S's discrimination survives them. **FDR control:** Benjamini–Hochberg (or Model-X knockoffs) across the 32,768 features so |S| is not a multiple-testing artifact.

**Trivial-explanation check:** confirm the signal is not a tokenizer/window/aggregation artifact (shuffle-label null → AUROC ≈ 0.5; feature-permutation null; ensure per-codon vs per-nucleotide aggregation is consistent).

**Robustness (M0 "paraphrase/seed/decoding" adapted to this domain):**
- **paraphrase →** holds across ≥2 independent CDS datasets / organism groups (e.g., human + a prokaryote set) and across homolog-disjoint splits;
- **seed →** stable feature set across ≥3 cluster-split seeds (Jaccard overlap of S reported);
- **decoding →** stable across helix definition (HGI vs H-only) and activation-threshold choice.

**Four-state verdict (governs downstream):**
- `established` → |S| ≥ 1 with test AUROC ≥ τ_auc, confounds controlled, FDR-significant, robust → run M1–M3 on S.
- `conditional` → selectivity holds only in a sub-condition (e.g., one organism group / helix def) → tag C1/C2/C3 `conditional`, restrict M2/M3 to that condition (runtime scoping; plan not rewritten).
- `not-established` → no feature clears the bar after controls → STOP pipeline, write negative-result report (skip M1–M3 + verify).
- `inconclusive` → M0 test itself underpowered/broken (alignment coverage too low, data floor missed) → fix data/script and re-run M0; never run mechanism on an untested phenomenon.

**Pass criteria (explicit):** test AUROC ≥ **0.75** and F1 ≥ **0.3** for ≥1 feature (thresholds τ frozen on validation; report the full ranked list + |S|), margin over β-sheet/coil ≥ 0.1 AUROC, BH-FDR q < 0.05, robustness overlaps as above, null AUROC ≈ 0.5.

**Grid:**
  organism_group: [eukaryote, prokaryote]
  helix_def: [HGI, H_only]
  split_seed: [42, 200, 201]
**Cmd template:** `CUDA_VISIBLE_DEVICES=2,3,4,5 python m0_feature_selectivity.py --evo2 /data1/share_model/evo2/evo2_7b/evo2_7b.pt --sae /data1/share_model/evo2/evo2_sae_layer26_mixed/sae-layer26-mixed-expansion_8-k_64.pt --organism ${organism_group} --helix_def ${helix_def} --split_seed ${split_seed} --tau_auc 0.75 --tau_f1 0.3 --fdr bh --out results/m0_${organism_group}_${helix_def}_s${split_seed}.json`
**Expected output (template):** `results/m0_${organism_group}_${helix_def}_s${split_seed}.json` + consolidated `results/m0_feature_set.json` (the frozen S + per-feature stats + verdict).
**Priority:** MUST-RUN (hard gate). **Estimated GPU-hours per run:** ~3h (activation caching dominates); 12 runs, `max_parallel: 4`.

---

## M1: Generation + verification harness (calibration; enables C2/C3)

**Claims:** enabler for C2/C3.
**Depends on:** [M0].

**Goal:** build and **calibrate** the DNA→protein→structure→DSSP readout and the steering hook, so M2/M3 measure signal, not pipeline noise.

**Components:**
- **Steering hook:** at the Layer-26 SAE site, amplify S's latents during **autoregressive** nucleotide decoding — clamp/scale each f∈S to `act + α·s_f` (s_f a per-feature scale, e.g. its max/mean active value), decode SAE→residual, continue the forward pass. Fixed generation config (context prompt policy, temperature, length) held constant across doses.
- **Translate + validity filter:** 6-frame/ORF-aware translation; keep sequences with a valid ORF (start, in-frame, no premature stop, length ≥ L_min). Record valid-ORF rate.
- **Structure predict:** ESMFold (default) on translated proteins; **pLDDT gating** (drop low-confidence; report gated fraction).
- **DSSP readout:** α-helix fraction = helix residues / resolved residues (HGI primary, H-only secondary).
- **Baselines/calibration:** unsteered generation (α=0) distribution of α-helix fraction; natural helix-rich vs helix-poor CDS reference distributions; establish measurement noise + samples-per-dose needed for power (target detect Δhelix ≥ ~0.1 at power 0.8).

**Verification:** on α=0, the pipeline recovers a sane baseline helix fraction with stable variance; positive control (a naturally helix-rich prompt set) reads higher than a helix-poor set.
**method_sensitive: [n_pairs, sites, metric, gpu_hours]**
**Cmd:** `CUDA_VISIBLE_DEVICES=2,3,4,5 python m1_harness_calibrate.py --evo2 ... --sae ... --feature_set results/m0_feature_set.json --n_samples 500 --predictor esmfold --plddt_min 60 --out results/m1_calibration.json`
**Expected output:** `results/m1_calibration.json` (baseline helix dist, noise, samples/dose, valid-ORF & pLDDT-gated rates).
**Priority:** MUST-RUN. **Estimated GPU-hours:** ~6h (structure prediction dominates).

---

## M2: Dose-response steering sweep (C2 — causal control)

**Claims:** C2.
**Depends on:** [M0, M1].

**Hypothesis (H2):** encoded-protein α-helix fraction increases monotonically with amplification coefficient α up to an optimum.

**Design:** generate N sequences per dose across a swept α (including α=0 baseline and negative α as a sign check), translate→predict→DSSP, compute mean α-helix fraction per dose with CIs.
- **Expected sign:** up. **Expected magnitude/dose-response:** monotone non-decreasing helix fraction over the low-mid α range; identify **optimal α*** (max helix fraction subject to generation-quality tolerance) and the degradation onset.
- **Specificity control (in-milestone):** carry α=0 and (from M3) matched-control feature at ≥1 shared dose for reference.
- **Statistics:** Spearman trend of helix-fraction vs α (H0: no trend); per-dose Mann–Whitney vs α=0; N per dose from M1 power calc (≥ ~50, likely a few hundred after gating).

**Pass criteria (C2):** significant positive Spearman trend (p<0.05) over the swept range up to α*, with mean helix fraction at α* significantly above α=0 (effect Δ ≥ ~0.1, non-overlapping CIs), and quality metrics within M1 tolerance up to α*.
**method_sensitive: [n_pairs, sites, metric, gpu_hours]**
**Grid:**
  alpha: [-2, 0, 1, 2, 4, 8, 16, 32]
  seed: [42, 200, 201]
**Cmd template:** `CUDA_VISIBLE_DEVICES=2,3,4,5 python m2_dose_response.py --evo2 ... --sae ... --feature_set results/m0_feature_set.json --alpha ${alpha} --seed ${seed} --n_per_dose 300 --predictor esmfold --plddt_min 60 --out results/m2_a${alpha}_s${seed}.json`
**Expected output (template):** `results/m2_a${alpha}_s${seed}.json` + `results/m2_dose_response_curve.json`.
**Priority:** MUST-RUN. **Estimated GPU-hours per run:** ~4h; 24 runs, `max_parallel: 4`.

---

## M3: Specificity / "causally manipulable knob" (C3)

**Claims:** C3.
**Depends on:** [M0, M2].

**Hypothesis (H3):** the helix rise is attributable to the α-helix feature identity, not generic perturbation.

**Controls (double dissociation + quality):**
- **Matched-control feature(s):** random/unrelated SAE features (matched for activation frequency/magnitude to S), steered at the same doses → expect **no** α-helix rise. *(Secondary sanity control only — see the random-direction null below.)*
- **Random-direction null (PRIMARY specificity control — added in iteration-loop type-② repair; `code/m3_random_control.py`):** ≥30 (run: 33) **independent** random directions, each = |S|=19 SAE features drawn uniformly from the dictionary **excluding** S ∪ β ∪ matched-control, weighted by their own natural activation magnitude s_f (same convention as S) and then **rescaled to match ||S.base_dir||** at the injection site (activation-magnitude matched). Steered at the **locked mid-plateau dose** and read out identically. The primary specificity statistic is **S's helix rise vs this null distribution**: empirical one-sided p = (1 + #{Δ_rand ≥ Δ_S}) / (N+1), plus a z-score, plus a valid-ORF comparison (rule out a capability-selection artifact).
- **Off-target β-sheet feature:** an M0-identified β-sheet-selective feature steered at the same doses → expect **β-sheet** rise, **not** α-helix rise. *(In this run the β-arm did NOT amplify sheet — reported honestly as a **NEGATIVE result / failed manipulation**, not as half a double dissociation. C3 rests on S-specificity (S vs random-null / matched) on the helix axis + β-does-not-raise-helix.)*
- **Quality guardrail:** coding fraction, valid-ORF rate, Evo2 perplexity, pLDDT distribution stay within M1 tolerance across the effective dose range (no "gaming via degenerate sequences").

**Decisive dose (locked, iteration-loop type-② repair):** the double-dissociation / specificity statistics are computed at the **mid-plateau dose α=8**, where the S arm's own generation quality is **preserved** (valid-ORF ≈ 0.89, baseline level), **NOT** at the grid-max α=16 where valid-ORF crashes to 0.667 (>2× the ~10% capability-degradation tolerance — the mechanism-audit FAIL trigger). The α-span trend (S rises monotonically 0.49→0.51→0.59→0.78 across α∈{0,4,8,16} while matched/β stay flat) still characterizes specificity as a trend, not a single edge-of-grid snapshot; the decisive stat is just locked to the capability-preserved operating point.

**Pass criteria (C3):** at the locked capability-preserved dose (α=8), helix increase for S is a significant outlier vs the ≥30-direction random-magnitude-matched null (empirical p<0.05) AND ≫ matched-control, while matched-control and β raise neither helix meaningfully and S's valid-ORF is not degraded → **C3: the α-helix feature set is a causally manipulable, specific knob** (specificity established on the helix axis; the β-sheet-amplification arm is a documented negative result).
**method_sensitive: [n_pairs, sites, metric, gpu_hours]**
**Grid:**
  feature_kind: [alpha_helix_S, matched_control, beta_sheet_offtarget]
  alpha: [0, 4, 8, 16]
  seed: [42, 200, 201]
**Cmd template:** `CUDA_VISIBLE_DEVICES=2,3,4,5 python m3_specificity.py --evo2 ... --sae ... --feature_set results/m0_feature_set.json --feature_kind ${feature_kind} --alpha ${alpha} --seed ${seed} --n_per_dose 300 --predictor esmfold --plddt_min 60 --readout helix_and_sheet --out results/m3_${feature_kind}_a${alpha}_s${seed}.json`
**Expected output (template):** `results/m3_${feature_kind}_a${alpha}_s${seed}.json` + `results/m3_specificity_summary.json`.
**Priority:** MUST-RUN. **Estimated GPU-hours per run:** ~4h; 36 runs, `max_parallel: 4`.

**Random-direction null (iteration-loop type-② repair — `code/m3_random_control.py`, `code/m3_consolidate_v2.py`):** 33 independent random norm-matched directions at the locked dose α=8, dispatched fully detached across GPUs {3,4,5}, `n_per_dose=80` each. Consolidated (with the already-collected α=8 S/matched/β arm data) into `results/m3_specificity_summary_v2.json`, which locks the decisive dose at α=8 and reports S-vs-random-null as the primary specificity statistic.
**Cmd:** `CUDA_VISIBLE_DEVICES=3 python code/m3_random_control.py --dir_start 0 --dir_end 10 --alpha 8 --n_per_dose 80 --out results/m3_random_control_d0-10.json` (× 3 ranges over GPUs 3/4/5).

---

## Run order & budget
1. M(-1) setup → 2. **M0 gate** (12 runs) → branch on verdict → 3. M1 calibration → 4. M2 sweep (24 runs) ∥ 5. M3 controls (36 runs), all `max_parallel: 4` on GPUs 2,3,4,5.
- Queue routing: M0/M2/M3 declare `grid:` (+ `depends_on:`) → `/auto-experiment` Phase 4.B (`/experiment-queue`). M1 is a single calibration run (Phase 4.A).
- `method_sensitive` fields on M1–M3 may be re-bound by the experiment stage at commit time (n_pairs/sites/metric/gpu_hours) without counting as a plan rewrite; M0 carries none (its data scale is fixed by HC3).

## Claim → milestone traceability
- **C1** → M0 (gate). **C2** → M1 + M2. **C3** → M3. All mechanism milestones `depends_on: [M0]`.
