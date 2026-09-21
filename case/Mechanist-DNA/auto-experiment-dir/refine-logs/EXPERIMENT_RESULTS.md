# Initial Experiment Results — Round 2 (Rigor Hardening)

**Date:** 2026-07-19
**Plan:** refine-logs/EXPERIMENT_PLAN.md
**phenomenon_status: established**
**GPUs witnessed:** {3,4,5,6} (pinned, HC2)
**resource_fidelity:** not-strict (full scale — rigor round, no downscaling)

Mechanism (GIVEN, Mode B): **Feature Dictionary Learning / SAE** — amplify Layer-26 α-helix-selective
BatchTopK-SAE features at `blocks.26.post_norm` during Evo2-7B autoregressive DNA generation;
σ_proj-unit dose-response. Committed in `refine-logs/MECHANISM_ROUTING.md` (committed:true,
reconciliation_status: ok).

---

## Data Actually Used

| Claim/Block | Provenance | Source | Available N | Used N | Subset note |
|---|---|---|---|---|---|
| C1 / M0 | existing (frozen round-1) | prokaryote CDS+DSSP 181,359 codons; eukaryote 350,268 codons | 531,627 codons | 531,627 (full) | — (reused round-1 caches at full scale, HC3-compliant natural CDS + experimental DSSP) |
| C2 / M1,M2 | constructed (Evo2 steered generation) | natural-CDS prompts (prokaryote) | — | ≳300 valid-ORF/dose (target) | deploying |
| C3 / M3 | constructed (Evo2 steered generation) | natural-CDS prompts (prokaryote) | — | ≳300/arm-dose; ≥48 null dirs × ≥120 | deploying |

---

## M(-1): Setup precondition — PASSED

- **Second structure predictor chosen: OmegaFold (release2 / model 2).**
  - **Why:** single-sequence predictor with its own OmegaPLM language model — NOT MSA-based, NOT the
    ESM2 trunk ESMFold uses → the most *independent* second axis obtainable without licensed
    AlphaFold DBs or MSA search (plan P3 notes ESMFold/AF2-family share priors; OmegaFold is the more
    independent axis). Installed from the GitHub repo on `PYTHONPATH` because its pip `setup.py`
    rejects Python 3.11 (bypassed, HC4 install-and-continue, no sudo/creds); release2 weights
    (~3.18 GB) fetched directly from helixon S3.
  - **Validated end-to-end on GPU:** villin HP36 → mean pLDDT 89.9, 67% helix (correct); per-residue
    pLDDT (CA b-factor) → DSSP join verified; both predictors emit 0-100 pLDDT into a common
    pLDDT-weighted readout.
- **ESMFold:** round-1 left `facebook/esmfold_v1` weights un-cached; HF's xet backend corrupted the
  first re-download (deleted the `.incomplete` without producing the blob). Re-fetched with xet
  disabled (standard resumable HTTPS), ~5.2 GB `pytorch_model.bin`.
- Evo2-7B loads (3.5s), Layer-26 residual at `blocks.26.post_norm` extracted, SAE encode→top-k64→
  decode round-trips (L0≈64, dict 32,768, beats dimension-shuffled input decisively), mkdssp 4.6.1 on
  PATH, frozen round-1 S loads (19 helix + s_f, 5 β, 19 matched-control).
- **No STOP/approval condition hit.**
- Output: `results/setup_report.json`.

## M0: Phenomenon-validation gate (C1) — established

Frozen round-1 feature set S **re-confirmed** on the `data/` caches (reuse, no re-mining, no
reverse-translation). 12 configs (organism × helix_def × split_seed).

- **Primary statistic — set-level combined AUROC on the FROZEN S (held-out test):**
  - prokaryote (steer organism), mean over configs: **0.901** (round-1: ~0.892) — reproduced.
  - per-config (prok|HGI|s200): 0.896; confound-only (GC3 + position) AUROC 0.506 → **S adds +0.39
    over confounds**; set shuffle-label null 0.53 (gap 0.37); **19/19 frozen features BH-significant**
    in-split; per-feature confounds survive 100%; single-feature shuffle-null ≈ 0.4997.
- **Cross-organism transfer (new vs round-1):** eukaryote set-AUROC **0.866** ≥ 0.80 →
  `holds_multi_organism = True` (round-1 had it False). The α-helix SAE code generalizes prok→euk.
- **β re-identification (`beta_features_v2`):** the same selectivity/FDR bar re-derived the round-1 β
  set (features 22326,21653,13992,17067,31467; set-AUROC 0.826) — **no *stronger* β set exists**
  (`beta_v2_clears_bar = False`). → M3 β-arm will be a **documented helix-axis-specificity negative**
  (plan P9), not a symmetric double dissociation. This does NOT affect the S-vs-null C3 conclusion.
- **Verdict: `established`** (set-AUROC ≥ 0.80, confounds controlled, FDR-significant, null clean,
  robust across seeds/helix-defs/organisms). → M1–M3 proceed.
- Output: `results/m0_feature_set.json` (frozen S + beta_v2 + per-config re-confirmation + verdict);
  per-config `results/m0_${organism}_${helix_def}_s${seed}.json`.

## M1: Harness calibration — DONE

- **σ_proj = 0.4104**, ||base_dir|| = 1.1019 at `blocks.26.post_norm` (natural-CDS reference).
  Raw-α↔c map: **c(raw α=8) = 21.48**, c(α=16)=42.96, c(α=32)=85.92 (round-1 raw doses located on the
  σ_proj c-axis).
- **c=0 baseline, dual predictor** (pLDDT-weighted HGI helix fraction, primary): ESMFold 0.461,
  OmegaFold 0.479 (predictors agree); valid-ORF 0.863, folded 345/400; mean pLDDT 63.7 / 69.5.
- **Power:** N=107 per dose for Δhelix=0.1 @ power 0.8 → n_per_dose=300 (used) is well-powered.
- **Positive control separates under BOTH predictors** (natural helix-rich > helix-poor).
- Output: `results/m1_calibration.json`.

## M2: σ_proj dose-response (C2) — DONE (positive, dual-predictor, interior c*)

30 runs (10 σ_proj doses = round-1 raw-α grid mapped to c, × 3 seeds; both predictors folded per run).
Split-sample: c* selected on seed 42, headline read on held-out seeds 200,201.

- **Interior optimum c\* = 21.48** (= round-1's clean α=8). Selected as the pLDDT-weighted-helix max
  **within the contiguous capability-preserved region** c∈{0, 2.7, 5.4, 10.7, 21.48} (valid-ORF ≈
  baseline ~0.88–0.89); the region ends at c=32.22 where valid-ORF collapses to 0.807. **c\* valid-ORF
  = 0.893 ≈ baseline** — capability preserved. interior_c*=True (helix rises to c* then capability
  degrades at higher doses; high-dose helix resurgences at c=43–86 are in the degraded regime and
  correctly excluded per P7).
  - *Correction logged:* the first consolidation pass took a global helix argmax (c=85.92, valid-ORF
    0.807 — degraded) via too-lenient a capability tolerance; the c*-selection was fixed to the
    contiguous-preserved-region rule and re-run, and any M3 dispatched at the wrong dose was killed
    (0 completed cells) and re-dispatched at c*=21.48. C2's dose-response data was unaffected.
- **Primary trend (held-out seeds, pLDDT-weighted endpoint):**
  - ESMFold: Spearman ρ=**0.867**, p=**0.0012**; Δ(c*) vs c=0 = **+0.129** [95% cluster-bootstrap CI
    0.100, 0.159], excludes 0.
  - OmegaFold: Spearman ρ=**0.879**, p=**0.0008**; Δ(c*) = **+0.135** [0.109, 0.161].
  - **c2_positive_trend = True, dual_predictor_agree = True.** Both predictors reproduce a significant
    positive dose-response up to an interior c* with Δ ≥ 0.1 at matched capability.
- Output: `results/m2_dose_response_curve.json` (+ 30 `results/m2_alpha_helix_S_c*_s*.json`).

## M3: Specificity (C3) — RUNNING at corrected c*=21.48

36 arm runs [S / matched-control / β] × 4-dose bracket {0, 10.74, 21.48, 32.22} × 3 seeds + 8
≥48-direction norm-matched random-null chunks, both predictors folded per run. Dispatched 4-wide on
GPUs 3,4,5,6. Primary = S-vs-null empirical p + effect size + cluster-bootstrap CI on the pLDDT-
weighted endpoint at the locked interior c*. _(Results on completion.)_

---

## Summary (so far)

- M(-1) setup: PASSED (OmegaFold = independent 2nd predictor; ESMFold re-cached).
- **M0 phenomenon gate: `established`** — frozen S re-confirmed, set-AUROC 0.901, now holds across
  prokaryote **and** eukaryote; β_v2 no stronger than round-1.
- M1–M3: implemented, cross-model reviewed, sanity-checked, deploying detached on GPUs 3,4,5,6.

## Next Step
→ M1→M2→M3 complete on the detached orchestrator, then `/auto-verify`.
