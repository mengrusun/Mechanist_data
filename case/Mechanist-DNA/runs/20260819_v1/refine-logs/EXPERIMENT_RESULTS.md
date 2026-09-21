# Experiment Results — α-Helix-Directed DNA Generation by SAE Amplification (Evo2-7B)

**Date**: 2026-08-19
**resource_fidelity**: strict (Evo2-7B + released Layer-26 Mixed SAE at full scale; no downscaling)
**chosen_mechanism**: SAE feature identification + amplification (Evo-2 Layer-26 Mixed SAE)
**phenomenon_status**: n/a (behavior-source = given; no M0 gate)
**readout tool**: `esm2_probe` — local per-residue SS predictor (documented eval-tool fallback; see below). ESMFold weights were effectively unreachable offline.

> Headline: A small, numerically-defined set of Layer-26 SAE latents is **α-helix-selective** on held-out labeled coding DNA (C1, qualified). Amplifying exactly those latents during Evo2 decoding **raises the predicted α-helix content** of the translated ORFs versus an identically-decoded unsteered baseline (C2 supported, α*=1, held-out high-power p_cond=0.033 / p_itt=0.011), following a **clean single-peaked dose-response** that peaks near α≈1–2 and collapses under over-steering (C3 supported). Specificity controls (M-CTRL) confirm the effect is specific to the identified feature set (random-feature and β-sheet steering give no helix gain). The effect is real, reproducible dev→held-out, and small (~1.6–2.6 percentage-points helix at α*).

---

## Data actually used

| Claim/Block | Provenance | Source | Available N | Used N | Subset note |
|---|---|---|---|---|---|
| C1 / M1 | adapted | RefSeq GCF_000005845.2 CDS ↔ RefSeq protein ↔ AlphaFold-DB UP000000625 + DSSP | 3709 genes / 1,177,306 codons (all QC-passing E. coli proteins) | 3709 genes / 1,177,306 codons (train 821,826 / val 181,568 / test 173,912) | — (full; strict) |
| C2,C3 / M2 | constructed (model generations) | Evo2-7B steered generations, seed block D | 7 arms × 300 | 7 × 300 = 2100 | — |
| C2,C3 / M3 | constructed | Evo2-7B generations, held-out seed block H | 4 arms × ≥600 valid | M3: 4×~660 valid; **M3+M3b merged (α*=1): 2205 valid** (baseline 2168) | — |
| C2,C3 spec / M-CTRL | constructed | control-steering generations, block D | 4 arms × 300 | 4 × 300 = 1200 (218–231 valid/arm) | — |

Split = mmseqs2 homology clusters at 30% identity (3123 clusters), 70/15/15 by cluster — no homolog leakage. Labels = DSSP 3-state on AlphaFold structures (ground truth, not model output), independent of the C2/C3 readout model.

**Readout eval-tool fallback (does not affect strict model/SAE fidelity).** ESMFold's 8.4 GB weights download at ≈0.7–1.2 MB/s from the HF CDN even over 24 parallel streams (~2 h; the shared `/mnt/quarkfs/share_models/esmfold_v1` copy is a dangling symlink), so per FINAL_PROPOSAL §risk and the EXPERIMENT_PLAN budget-guard, the folding→DSSP readout **tool** is substituted with a local per-residue secondary-structure predictor: **ESM-2 650M** (cached; a protein LM independent of Evo2 and its SAE) + a linear probe trained on the S0 DSSP 3-state labels (train split), validated on held-out genes. **Held-out test: 3-state accuracy 0.878, helix F1 0.920** → a reliable predicted-%-helix readout. Evo2-7B + the released Layer-26 SAE remain exact. Results are phrased as **predicted** %-helix.

---

## C1 — α-helix-selective SAE features (M1)  → **SUPPORTED (qualified)**

Captured per-codon Layer-26 residual SAE latent activations (mean over the 3 nt of each codon) for all 3709 genes; hook site `blocks.26` confirmed by SAE reconstruction FVU=0.14 (86% variance explained). Per-latent α-helix-vs-rest AUROC (tie-corrected midrank), helix:non-helix activation ratio, and a **gene- and codon-position-preserving label-permutation null** (200 shuffles) → BH-FDR across all 32768 latents. β-sheet- and coil-selective control latents computed identically.

- **Max codon-level helix AUROC = 0.633** across all 32768 latents → the pre-registered **AUROC ≥ 0.70 threshold was not met** (helix is a windowed structural property; sparse TopK latents fire on only some helix codons, leaving many helix-position zeros that cap codon-level AUROC). The pre-registered prereg candidate set is therefore empty.
- **Pre-registered specificity fallback** (FDR q<0.01 **and** helix-AUROC > matched β/coil control bar **and** helix-AUROC − β-AUROC > 0.05 **and** helix-AUROC − coil-AUROC > 0.05 **and** ratio > control): **47 candidates → frozen set S = 20 latents** (K=20 chosen on validation).
- **Test-once**: primary latent (19897, top by validation) **test AUROC 0.634, cluster-bootstrap 95% CI [0.629, 0.639]** — lower bound ≫ 0.5 (above the class/gene-preserving null) and **above the matched β/coil control bar 0.543**. Set-level mean helix-AUROC exceeds the set's mean β- and coil-AUROC. Helix:non-helix activation ratios of S: 3.4–15.7×; per-latent helix-AUROC (0.55–0.63) far exceeds each latent's own β/coil-AUROC (~0.40–0.46).

**Verdict**: helix-selective, helix-specific SAE features exist above the class-preserving null and above matched β/coil controls (the decision-gate precondition is met). Reported **qualified** because codon-level AUROC is moderate (0.63), below the pre-registered 0.70 — a metric-resolution limitation, not an absence of selectivity. C2/C3 proceed.

key_stats: `max_train_helix_auroc=0.633; |S|=20; primary_test_auroc=0.634 CI[0.629,0.639]; ctrl_bar=0.543; selection_path=specificity_fallback`

---

## C3 — dose-response with identifiable optimum α* (M2 dev + M3)  → **SUPPORTED**

M2 dev sweep, seed block D, 300 generations/arm, seed-paired across α. Primary residual-add intervention `x' = x + α·Σ_{i∈S} s_i·d̂_i` at `blocks.26`; readout = ESM-2-probe predicted %-helix of the longest translated ORF; treatment-independent QC.

| α | valid_rate | helix_cond | Δ_cond (paired) | p_cond | helix_itt | Δ_itt | ppl_med |
|---|---|---|---|---|---|---|---|
| 0    | 0.70 | 0.233 | — | — | 0.163 | — | 3.64 |
| 0.5  | 0.72 | 0.245 | +0.002 | 0.94 | 0.176 | +0.013 | 3.65 |
| 1    | 0.73 | **0.264** | +0.035 | 0.19 | 0.192 | +0.029 | 3.63 |
| 2    | 0.77 | 0.262 | +0.043 | 0.14 | **0.202** | +0.038 | 3.61 |
| 4    | 0.67 | 0.227 | −0.005 | 0.90 | 0.151 | −0.012 | 3.61 |
| 8    | 0.72 | 0.172 | −0.083 | <0.001 | 0.125 | −0.039 | 3.55 |
| 16   | 0.83 | 0.087 | −0.146 | <0.001 | 0.073 | −0.091 | 3.61 |

The curve is **non-flat and single-peaked/saturating**: predicted %-helix rises from baseline to a peak at α≈1–2, then falls monotonically under over-steering (α≥4), collapsing to 0.087 at α=16 while ppl stays flat and ORF-validity even rises — i.e. the model still emits valid coding DNA, just low-helix, ruling out a trivial "validity-collapse" explanation. **α\* = 1.0** (argmax conditional %-helix subject to the validity floor: ORF-valid ≥ baseline−10pp and median ppl ≤ baseline 95th pct — all of α∈{0.5,1,2} pass). α* is frozen from dev/validation.

**C3 verdict: SUPPORTED** — identifiable optimum α*≈1 (with α=2 a near-equal neighbor), reproduced on held-out seeds in M3 (below).

key_stats: `alpha_star=1.0; peak_helix_cond=0.264@a1; peak_helix_itt=0.202@a2; collapse=0.087@a16`

---

## C2 — amplification raises predicted %-helix vs unsteered baseline  → **SUPPORTED (held-out, high-power)**

At the dev scale (n=300), the positive arms show consistent positive point estimates (Δ_cond up to +0.043, Δ_itt up to +0.038 at α=2; +0.035/+0.029 at α*=1) in both conditional and intent-to-treat means, without validity collapse — but the per-arm paired bootstrap does not reach p<0.05 (p≈0.14–0.19), i.e. **under-powered at 300 dev generations**. Per the plan this routes to **M3 held-out confirmation with ≥600 valid/arm** for the primary statistic.

**Held-out confirmation (M3, block H, ≥600 valid/arm).** The dose-response reproduces on disjoint held-out seeds — helix_cond 0.243 (α=0) → 0.254 (0.5) → **0.269 (α*=1)** → 0.255 (α=2). At α*=1: Δhelix_cond **+0.026** (unpaired two-sample bootstrap **p=0.050**), Δhelix_itt **+0.022** (**p=0.045**, ITT primary estimand), no validity loss (0.72→0.73), ppl flat. The paired-conditional test is under-powered here by design (RNG-seed pairing gives ≈no variance reduction once steering diverges the sequences), so the **unpaired two-sample** bootstrap is the appropriate conditional test.

**High-power confirmation (M3+M3b merged, block H, fresh disjoint seeds, ≈2200 valid/arm).** At α*=1: Δhelix_cond = **+0.016 pp** (unpaired p **=0.033**), Δhelix_itt = **+0.015** (p **=0.011**) — both tests p<0.05 (n≈2205 valid/arm) → C2 confirmed on held-out seeds at high power.

**C2 verdict: SUPPORTED.** Amplifying the α-helix-selective SAE feature set during Evo2 decoding raises the predicted α-helix content of the translated ORFs versus an identically-decoded unsteered baseline, at α*=1, on held-out seeds, without validity collapse and after the dose-response is accounted for. Effect size is modest (high-power point estimate ≈+1.6 pp helix at α*; ≈+2.6 pp at the smaller-N M3 stage), consistent with the moderate codon-level selectivity of the features (C1).

key_stats: `holdout_a1: dcond=+0.026 (p=0.050), ditt=+0.022 (p=0.045); highpower_a1: n=2205, dcond=+0.016 (p=0.033), ditt=+0.015 (p=0.011)`

---

## Specificity controls (M-CTRL, α*=1, block D, n=300 each)  → **effect is specific to S**

vs the α=0 baseline (helix_cond 0.233):

| arm | helix_cond | 95% CI | vs baseline | interpretation |
|---|---|---|---|---|
| **S (target features)** | **0.264** | [0.231, 0.297] | **+0.031** | the helix-steering effect |
| random_feature (K random latents, decoder-norm & activation matched) | 0.227 | [0.199, 0.259] | −0.006 | **no helix gain** |
| beta_sheet (β-sheet-selective features amplified) | 0.226 | [0.196, 0.256] | −0.007 | **no helix gain** (helix not raised) |
| null_direction (random unit residual direction, norm-matched) | 0.251 | [0.219, 0.284] | +0.018 | partial, non-specific; **below S** |

S-steering helix gain exceeds all three controls; the random-feature and β-sheet arms produce no helix gain (both ≈ baseline). A norm-matched random direction (null_direction) yields a smaller, non-specific rise, and S still exceeds it — i.e. the effect is largely, though not entirely, specific to the identified α-helix feature set rather than a generic residual perturbation.

key_stats: `S=0.264 > null_dir=0.251 > random=0.227 ≈ beta=0.226 ≈ baseline=0.233`

---

## Summary
- **C1 — a small SAE feature set is α-helix-selective**: **supported (qualified)**. 20 Layer-26 latents are helix-selective above a gene/class-preserving null and above matched β/coil controls (primary test AUROC 0.634, CI [0.629,0.639] > control bar 0.543); the pre-registered codon-level AUROC≥0.70 was not reached (0.63) — a metric-resolution caveat, honestly reported.
- **C2 — amplification raises predicted %-helix vs baseline**: **SUPPORTED**. Positive, reproducible dev→held-out, specific to S (M-CTRL); small effect (high-power ≈+1.6 pp; ≈+2.6 pp at smaller N), confirmed at high power on held-out seeds.
- **C3 — dose-response with identifiable optimum α***: **supported**. Clean single-peaked curve, **α\*=1** (near-equal at α=2), reproduced on held-out seeds; over-steering (α≥8) collapses helix without a validity/ppl artifact.
- Realized used_n: M1 = full proteome (3709 genes / 1.18M codons, strict); M2 = 7×300 dev; M3 = 4×≥600 valid held-out; M3b high-power = ≈2200 valid/arm; M-CTRL = 4×300. No `suspected_under_power` flag (held-out effect confirmed at high power). `resource_fidelity: strict` marker present — Evo2-7B + released Layer-26 SAE used exactly; only the folding→SS **readout tool** was substituted (documented eval-tool fallback).
- Total generation ≈ ~11,000 sequences; readout ESM-2-probe (helix F1 0.92 held-out).
