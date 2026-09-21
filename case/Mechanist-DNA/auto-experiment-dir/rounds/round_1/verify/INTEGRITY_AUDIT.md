# Integrity Audit

**Overall**: FAIL    ← max severity across Phase 2 (FAIL, from C3) + Phase 9 (WARN, from C2's variant)
**Main-experiment integrity (Phase 2)**: FAIL    ← max severity across main-experiment combined verdicts (C3 = FAIL)
**Variant integrity (Phase 9)**: WARN    ← C2's single picked variant (method-swap-ssdef-pydssp): exp WARN / mech WARN → combined WARN

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision                                          | Detail |
|-------|------------|-------------|----------|---------------------------------------------------------|--------|
| C1    | WARN       | N/A         | WARN     | continue-with-warn                                       | verify/C1_helix_feature_set/main_experiment_audit/ |
| C2    | WARN       | WARN        | WARN     | continue-with-warn                                       | verify/C2_dose_response_steering/main_experiment_audit/ |
| C3    | WARN       | FAIL        | FAIL     | INCONCLUSIVE (main-experiment mechanism rigor broken)    | verify/C3_specificity_double_dissociation/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim (methodology honesty: GT
>   provenance, normalization, result-file existence, dead code, scope, eval type).
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim (steering coefficient
>   sweep rigor). `N/A` for C1 (M0 uses no additive intervention — pure feature-discrimination gate).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a`, `n/a` treated as `pass`.
> - **Gate decision** — C1/C2 admitted to Stages 2–3 (with a WARN caveat carried through); C3 is
>   short-circuited to INCONCLUSIVE and skips Stages 3–10 (see `inconclusive_reason` in its
>   `ROBUSTNESS.md`).

### Summary of findings (see per-claim EXPERIMENT_AUDIT.md / MECHANISM_AUDIT.md for full detail)

- **C1** (α-helix-selective feature set exists, M0): methodology is honest (real DSSP GT from
  experimental PDB structures, no reverse-translation, no self-referential normalization, numbers
  match artifacts). WARN is driven by **scope overclaim**: the plan's pre-registered seed-robustness
  statistic (Jaccard overlap of S) is a *degenerate 0.0* in the artifact (computed on the always-empty
  strict single-feature set) and this is not disclosed in `EXPERIMENT_RESULTS.md`, which instead
  reports a different, favorable statistic (AUROC stability); the "established" verdict routes
  entirely through a set-level statistic not spelled out as an alternative pass path in the plan's
  literal criteria; and the multi-organism robustness axis is missing (eukaryote leg pending) yet
  "established" is issued on the prokaryote leg alone. The reviewer computed the actual steering
  set's cross-seed Jaccard by hand (~0.7–0.8) and found it genuinely stable — the underlying
  phenomenon looks real, but the verdict label ("established" vs the plan's own "conditional") is not
  fully earned as reported.
- **C2** (dose-response steering, M1+M2): no fabrication or self-normalization found; artifact
  numbers match exactly. WARN (both audits) driven by (a) the target metric is a structure-*predictor*
  proxy (ESMFold+DSSP on generated sequences, inherent to a generation experiment, but not fully
  caveated as predictor-confidence vs validation); (b) at the two highest doses, β-sheet content
  collapses to near-zero while pLDDT (the predictor's own confidence) *rises above baseline* with a
  non-monotonic valid-ORF dip-then-recovery — a pattern consistent with degenerate/low-complexity
  sequence collapse that a structure predictor could over-confidently fold as helical; **no raw
  generated sequence text was ever saved** (the code explicitly discards it), so this alternative
  explanation is currently uninspectable; (c) the reported "optimal α*=32" sits at the edge of the
  tested grid with helix still rising (not a located plateau); (d) the steering sweep does not span
  ≥3 orders of magnitude and is not expressed in σ_proj units; no random-direction control within
  M2's own scope.
- **C3** (specificity / double dissociation, M3): experiment-audit WARN (same proxy-readout pattern,
  plus the plan's own β-sheet-specific pass criterion is not actually met — sheet never meaningfully
  rises for the β-sheet off-target feature at any tested dose, so only half of the "double
  dissociation" was demonstrated, though S-vs-matched-control specificity IS strong, p=3.2e-23).
  **Mechanism-audit FAIL**: the decisive dose (α=16, the maximum of M3's own 4-point grid) is a dose
  at which the S arm's own generation quality has already substantially degraded (valid-ORF
  0.888→0.667, ~25% relative drop, more than double the catalogue's ~10% tolerance) — the headline
  double-dissociation statistics rest on a capability-degraded operating point, no plateau was
  established, and no genuine random-direction control (≥30 directions) was run (only one matched
  fixed direction). **This FAIL short-circuits C3 to INCONCLUSIVE** — Stages 3–10 do not run for C3
  this pass; the main-experiment mechanism-tuning process, not the claim itself, needs fixing.

## Variant integrity (Phase 9)

**Variants audited**: 1 (0 pass, 1 warn, 0 fail) — C2's picked variant only (C1 deferred by
MAX_VERIFY_CLAIMS cap, un-audited this pass; C3 rejected at Phase 2, never reached Phase 9).

### Per-claim audit verdicts

| Claim | Exp. audit (variants) | Mech. audit (variants) | Combined |
|-------|------------------------|-------------------------|----------|
| C2    | WARN                   | WARN                    | WARN     |

### Findings (per variant)
- C2 / variant `method-swap-ssdef-pydssp`: **[INTEGRITY: WARN — experiment + mechanism]**.
  Experiment-audit WARN: no fabrication/self-normalization, all 10 (α,seed) cells present and
  internally consistent, but (a) scope should be read as a targeted SS-assignment-algorithm
  robustness check, not a full replication of the main experiment's scale (5 doses×2 seeds vs
  8 doses×3 seeds), and (b) a sample only counts toward either tool's aggregate if BOTH mkdssp and
  pydssp succeed on it (intersection, not each tool's independent success set) — defensible for a
  head-to-head comparison but noted as a minor selectivity caveat. Mechanism-audit WARN: reuses the
  main experiment's own steering apparatus unchanged (same base limitations already flagged in C2's
  main-experiment mechanism audit — narrow span, no σ_proj units, no random-direction control); this
  variant does not lock/report a single "optimal" α at a capability-degraded dose (unlike the main
  experiment's α*=32 or C3's α=16 FAIL pattern), so the sharper capability-crash FAIL trigger does
  not apply here — hence WARN, not FAIL. **WARN, not FAIL → the variant counts in both numerator and
  denominator of C2's robustness (N_eligible=1, #pass=1).**
