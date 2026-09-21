# Experiment Results — Steering Evo2-7B toward high α-helical content

**Date**: 2026-08-24 (finalized; supersedes the 2026-08-23 halt draft — weights were later staged
locally from `/mnt/quarkfs`, `HF_HUB_OFFLINE=1`, and the full M1→M4 suite ran)
**Plan**: refine-logs/EXPERIMENT_PLAN.md
**Phenomenon status**: n/a (behavior-source: given; `m0_gate: none` — no phenomenon-validation gate)
**Committed routing**: Representation/Parameter Analysis / Steering Vectors (CAA) primary, SAE-clamp
secondary arm, Probing localization screen (refine-logs/MECHANISM_ROUTING.md, `committed: true`,
`reconciliation_status: ok`).

## Pipeline status: `complete`
Full suite deployed: sanity ✓, M1 ✓, M2 ✓, M3 ✓ (6 coef × 3 seed × site28/site30/site26), M4 ✓
(baseline / caa_win / matched_control / sham, 750 gens each). All statistics computed from on-disk
raw files by `experiments/finalize_stats.py`; nothing fabricated.

## Frozen measurement contract (from M1 `harness_config.json`)
Evo2-7B gen (900 tok, T=1, top-k=4), deterministic 6-frame longest-ATG→stop ORF (30–300 aa),
**ESMFold→DSSP** α-helix (codes H/G/I), pLDDT-weighted. **Primary endpoint = mean α-helix fraction
over ALL generations on the TEST split, invalid→0** (survivorship-safe, couples C1 to C2). Validity =
pre-registered composite (in-frame start/stop, no premature stop, length 30–300, Evo2 NLL ≤ 1.5).
DEV/TEST disjoint split enforced (intervention built on DEV, tested on TEST — anti-circularity).

---

## Claim verdicts

### C1 — intervention raises α-helix fraction (up): **SUPPORTED**

**Primary test** (pre-registered): permutation dose-response trend test (statistic = Spearman ρ, one-
sided positive) on the coefficient in the primary all-generations helix endpoint, TEST split, on the
pre-chosen site family **site 28 CAA** (6 coefficients × 3 seeds = 18 points).

- **site 28 CAA: ρ = 0.922, permutation p = 5.0e-5, BH-FDR q = 1.5e-4** → significant monotone positive
  dose-response. **C1 rests on this and passes.**
- Winning setting (locked by the frontier rule below) = **site 28 CAA, coef 1.0**:
  **helix_all = 0.482 vs baseline 0.414 (+0.068)**.
- **Effect size, winning vs baseline** (per-generation, n=750/arm): **Cliff's δ = 0.135, Hedges g =
  0.237**, bootstrap 95% CI on the helix difference **[+0.040, +0.098]** (excludes 0).
- **Dose ladder (site 28, mean over seeds)**: 0.0→0.414 · 0.5→0.475 · 1.0→0.482 · 2.0→0.524 ·
  4.0→0.575 · 8.0→0.563. Helix rises monotonically through coef 4 (**+0.161 gain at coef 4**), with a
  slight roll-off at coef 8 as validity collapses.
- **Not a length or composition artifact** (plan-flagged confounds, checked): CAA shortens proteins
  (100→64 aa) but within a matched length band [40,90 aa] CAA still shows **+0.089 helix (0.495 vs
  0.406)**, and within-condition corr(length, helix) is negligible (~0.06–0.10); helix-favoring aa
  fraction is **unchanged/slightly lower** (0.313 vs 0.326) — the gain is not driven by A/E/L/M/Q/K
  enrichment. This supports genuine structural steering rather than a codon/length side effect.

**Secondary site families (BH-FDR corrected across the 3 tested):**
- site 30 CAA: non-monotone, **ρ = 0.044, p = 0.43, q = 0.65** (peaks at coef 2 then collapses; not a
  trend) — does not support C1 on its own; C1 does not rest on it.
- site 26 **SAE-clamp: honest negative** — helix *decreases* with clamp strength (**ρ = −0.95, p = 1.0**),
  driven by high SAE reconstruction error (sanity `recon_rel_error ≈ 1.0`; secondary-arm caveat noted
  at sanity). The learned SAE feature is not a usable causal lever here; reported as a negative result,
  not a failure of C1.

**Specificity (M4, per-generation helix, BH-FDR across the 3 control-arm tests):**
- caa_win vs baseline: **+0.068, p = 5.9e-6, q = 1.8e-5** (gain is real).
- matched_control (orthogonal same-norm direction) vs baseline: **−0.024, p = 0.088, q = 0.13** — **no
  gain** (if anything slightly lower).
- sham (same-site same-norm permuted perturbation) vs baseline: **−0.005, p = 0.72** — **no gain**.
- CAA − matched_control = **+0.093** (Cliff's δ 0.182, p = 9.3e-10); CAA − sham = **+0.073** (Cliff's δ
  0.145, p = 1.1e-6). → the helix gain is **specific to the steering direction**, not to the norm or the
  act of perturbing. (`results/m4/specificity.json`)

### C2 — intervention preserves validity (non-inferior): **SUPPORTED**

Pre-registered non-inferiority test on validity rate over the **full generated set** at the winning
setting, margin **Δ = −0.05** (`results/m3/winning.json`).

- validity: **caa_win 0.968 vs baseline 0.991, Δ = −0.023**; one-sided 95% bootstrap lower bound
  **−0.035 > −0.05** → **non-inferior** (two-sided 95% CI [−0.037, −0.008]). The small validity dip is
  within the pre-registered margin.
- **Off-target vs baseline** (winning; reported per harness, "not disqualifying"; BH-FDR across the 5):
  | metric | baseline | caa_win | Δ | disposition |
  |---|---|---|---|---|
  | β-sheet frac | 0.050 | 0.045 | −0.005 | not degraded (slightly lower) |
  | GC content | 0.464 | 0.325 | −0.139 | **CAA-specific shift** (δ −0.81) — documented, controls stay ~0.46 |
  | mean Evo2 NLL | 1.217 | 1.278 | +0.061 | slightly less "natural" (δ 0.21) but well under validity NLL ≤ 1.5 |
  | protein length | 100.1 | 63.5 | −36.6 | **CAA-specific shortening** (δ −0.45) — but C1 survives length-matching (above) |
  | aa helix-favoring | 0.326 | 0.313 | −0.013 | unchanged — gain is not composition-driven |
  GC and length shift meaningfully under CAA; per the pre-registered contract these are **reported, not
  disqualifying**, and neither explains the helix gain. Naturalness and validity remain within bounds.

### Joint C1 ∧ C2: **SUPPORTED**
At the winning conservative setting (site 28 CAA, coef 1.0) the α-helix effect is **real, direction-
specific, dose-responsive, and validity-preserving** (non-inferior within the 5% margin). The full
helix-vs-validity **frontier** (`results/m4/validity_frontier.json`) shows the expected tradeoff: higher
coefficients buy more helix (up to +0.161 at coef 4) at the cost of validity (0.991→0.945 at coef 2,
→0.895 at coef 4, →0.819 at coef 8). The winning setting is the **max helix gain subject to validity
non-inferiority** (coef 0.5 and 1.0 pass NI; coef ≥ 2 fail). This is a clean dose–validity Pareto story:
steer conservatively to preserve validity, or push harder for larger structural change at a validity
cost.

## Power / caveats
- **No `suspected_under_power` tags.** n = 750 gens/condition (M4) and 3 seeds × 250 gens/dose (M3); the
  C1 trend test reaches q = 1.5e-4 and the specificity contrasts p ≤ 1e-6 — adequately powered vs the
  plan's ≥3-seed / TEST-split floor. The winning-vs-baseline effect is *small* per-generation (Cliff's δ
  0.135 / Hedges g 0.237) but the CI excludes 0 and the trend is decisive.
- **SAE-clamp arm** is an honest negative (recon error ≈ 1.0); the CAA lever, not the SAE feature,
  carries the result. Not disqualifying for C1 (which pre-registered on the CAA site family).
- **Off-target GC / length shifts** are CAA-specific and documented; they do not invalidate C1 (survives
  length-matching, no composition bias) or C2 (validity non-inferior), but are flagged for the verify
  stage as the most interesting robustness follow-up.

## Artifacts
- refine-logs/MECHANISM_ROUTING.md (`committed: true`)
- refine-logs/EXPERIMENT_TRACKER.md
- refine-logs/EXPERIMENT_TIPS.md
- results/m4/specificity.json — control-arm comparison
- results/m4/validity_frontier.json — helix-vs-validity frontier + winning rule
- results/m3/{s28_caa,s30_caa,s26_sae_clamp}.json, results/m3/winning.json
- results/m4/raw_{baseline,caa_win,matched_control,sham}.json (750 gens each)
- experiments/finalize_stats.py — reproduces every number above

## Realized cost
**≈ 27.3 GPU-hours** (HARD cap 40 respected; ≤ 8 concurrent cards; GPUs 1/3/4/6/7 used, 0/2/5 left to
other users). Breakdown: sanity ~0.1 + M1 ~4.0 + M2 ~3.5 + M3 16.0 (reconstructed cost.json) + M4 3.7
(reconstructed cost.json). See EXPERIMENT_TRACKER.md.

## Ready for /auto-verify: YES
Joint C1 ∧ C2 supported. Suggested verify focus: robustness of the direction-specific helix gain under
(a) an alternate SS assay / pLDDT filtering policy, (b) length-regressed helix endpoint (given the CAA
length shift), and (c) held-out primer/seed swaps.
