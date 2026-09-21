# Verification Report

> **⚙️ ITERATION UPDATE (2026-07-18, post-verify) — C3 repaired.** The iteration loop's type-② fix
> repaired C3's M3 mechanism harness (decisive dose re-locked from the capability-crashed α=16 to the
> capability-preserved mid-plateau α=8; a genuine 33-direction norm-matched random-direction control
> added). C3's Phase-2 mechanism-audit was re-run and moved **FAIL → WARN**, so **C3 is no longer
> INCONCLUSIVE — it is now ⚪ INTEGRITY_ONLY** (main-experiment integrity pass-with-warn; swap-test still
> deferred by the MAX_VERIFY_CLAIMS cap). Fixed evidence: `results/m3_specificity_summary_v2.json`
> (S helix +0.096 beats all 33 random directions, empirical p=0.029, z=3.02; capability preserved).
> See `C3_specificity_double_dissociation/ROBUSTNESS.md` + `main_experiment_audit/MECHANISM_AUDIT.md`
> (prior FAIL archived as `MECHANISM_AUDIT.pre_iteration_fix.md`). The original verify-stage snapshot
> below is retained unedited for the audit trail.

**Date**: 2026-07-18
**Swap variants**: true (full 3-stage pipeline)
**Dimensions tested**: method (DIMENSIONS=method, per task override — model swap excluded since HC1
pins Evo2-7B as the claims' subject)
**Threshold**: robustness ≥ 0.5, min eligible variants = 1
**Main-experiment integrity (Phase 2)**: see per-claim table below; full detail in
`verify/INTEGRITY_AUDIT.md`.

## Summary

| Claim | Statement (short) | Main-experiment verdict | Main-experiment integrity (Phase 2, combined) | Variant integrity (Phase 9, combined) | Eligible variants (post-audit) | Robustness | State | Notes |
|-------|--------------------|--------------------------|------------------------------------------------|-----------------------------------------|--------------------------------|------------|-------|-------|
| C1 | α-helix-selective Layer-26 SAE feature set exists (M0-gated) | supported (established, set-level) | WARN (exp WARN / mech N/A) | skipped (MAX_VERIFY_CLAIMS cap) | — | — | ⚪ INTEGRITY_ONLY (skip=max_verify_claims_cap) | Admitted by Phase 2 but not the top-K picked (C2 judged more central this pass). Main-experiment WARN driven by scope overclaim (undisclosed degenerate seed_jaccard=0.0, post-hoc set-level pass-criterion substitution, missing multi-organism robustness leg) — see `C1_helix_feature_set/main_experiment_audit/EXPERIMENT_AUDIT.md`. Upgrade via `/auto-verify C1 — resume: true`. |
| C2 | Amplifying S causally raises encoded-protein α-helix fraction (dose-response) | supported (Spearman ρ=0.759, p=1.7e-5, +0.36) | WARN (exp WARN / mech WARN) | WARN (exp WARN / mech WARN) | 1/1 | **1.00** | ✅ **PASS** | Method-swap picked (core causal claim). Swap executed: mkdssp → pydssp SS-assignment algorithm on identical ESMFold-predicted structures (originally-planned ESMFold→OmegaFold structure-predictor swap abandoned as network-infeasible — see variant's `PIVOT_NOTE.md`). Pydssp reproduces the dose-response almost exactly (ρ=0.886 p=6.4e-4, effect +0.327 vs mkdssp's own +0.332 on the same structures). **Scoped PASS: robust to the SS-assignment-algorithm axis only — the structure-predictor axis (the specific "ESMFold pLDDT rises while sheet collapses at high α" concern) remains untested this pass.** |
| C3 | Amplification effect is specific to S (double dissociation) / causally manipulable knob | supported (double dissociation, p≤3e-23) | **FAIL** (exp WARN / mech **FAIL**) | — (never ran) | — | — | 🟡 **INCONCLUSIVE** | Main-experiment **mechanism rigor broken**: decisive dose (α=16, the max of M3's own 4-point grid) is a dose at which the S arm's own generation quality has already substantially degraded (valid-ORF 0.888→0.667, ~25% relative drop, >2× the ~10% capability-degradation tolerance); no plateau located; no genuine random-direction control. Variants never ran. Iteration must fix the M3 mechanism harness (locate a genuine mid-plateau shared dose, add a real random-direction control), not the claim. |

> Column glossary: **Main-experiment integrity (Phase 2)** = `max_severity(/experiment-audit,
> /mechanism-audit)` on `refine-logs/`, scoped per claim; `n/a` treated as `pass`. **Variant integrity
> (Phase 9)** = same pair on the variant's own directory. **Robustness** = `#pass / N_eligible` over
> the variant's `consistent_with_main_experiment` outcomes (integrity-FAIL variants excluded from
> both; none were FAIL here).

## Integrity Audit

**Overall**: FAIL (max severity across Phase 2 [FAIL, from C3] + Phase 9 [WARN, from C2's variant]) —
see `verify/INTEGRITY_AUDIT.md` for full detail.

**Phase 2 headline — reported prominently, not relabeled**: C3's mechanism-audit returned **FAIL**.
The decisive double-dissociation statistics in `results/m3_specificity_summary.json` are computed at
α=16 (the maximum of M3's own 4-point grid), a dose at which the treatment arm's (`alpha_helix_S`)
own generation quality has already substantially degraded (valid-ORF rate 0.888→0.667, a ~25%
relative drop — read against the catalogue's ~10% degradation tolerance, more than double that
tolerance band, in the "capability crashed" range the catalogue's FAIL criterion is meant to catch).
No plateau was established anywhere in M3's narrow grid, and no genuine random-direction control
(≥30 independent directions) was run — only one fixed "matched-control" direction. This is a rigor
failure in the steering-tuning process feeding C3's headline numbers, not a fabrication finding; C1
and C2 both returned WARN (not PASS/FAIL) on scope-overclaim grounds documented in their own audits.

## Stage-2 Selection

Phase 3 step 0 picked 1 of 2 admitted claims (C1, C2 — C3 rejected at Phase 2 → INCONCLUSIVE) for
Stage 2 (cap = MAX_VERIFY_CLAIMS = 1).

**Picked**: C2 — the core causal dose-response claim, most exposed to a measurement-pipeline
confound (its headline effect is read out entirely through structure prediction on generated
sequences), and the natural target for the method-axis swap the task brief called for.

**Stage-2-deferred** (marked INTEGRITY_ONLY with `stage2_skip_reason: max_verify_claims_cap`):
- C1: α-helix-selective SAE feature set exists — main-experiment verdict: supported (established) —
  swap-test later via `/auto-verify C1 — resume: true`.

## The method swap actually run (and the one that wasn't)

**Attempted first (per the task brief's top suggestion): ESMFold → OmegaFold** structure-predictor
swap. Code was designed, reviewer-critiqued (Phase 4, trust score 3/5 → strengthened per feedback:
wider grid, same-sequence cross-check, sequence-diagnostics), implemented, and code-reviewed (Phase
6, CRITICAL/MAJOR findings fixed: idx-parsing brittleness, missing-file tracking, required conf-gate
calibration instead of a guessed default, exception-safe cleanup). **Abandoned at install time**:
OmegaFold's `setup.py` hardcodes `torch==1.12.0+cu113` from `download.pytorch.org`, which crawled at
a few MB/min over this environment's outbound proxy (~830MB of a ~1.9GB wheel after 15+ minutes); a
patched `setup.py` pulling modern `torch` from PyPI instead still needed ~2.5-3.5GB of separate
`nvidia-cu12` wheels and was equally slow. This is **not** a sudo/credential/quota HC4-STOP condition
— just an impractically slow network path for this verify pass's time budget. Full detail, including
the accidental (caught and reverted) `numpy` ABI upgrade in the shared `scientist` env from an
earlier `colabfold` attempt, is in `C2_dose_response_steering/variants/method-swap-omegafold/PIVOT_NOTE.md`
(kept on disk for transparency, not deleted).

**Executed instead: mkdssp → pydssp SS-assignment-algorithm swap**, the second method-axis candidate
the task brief explicitly named as acceptable ("swap the secondary-structure definition/tool").
Requires zero new package installs (`pydssp` already present per `results/setup_report.json`).
Design: identical Evo2-7B steering hook, frozen feature set S, prompts, translation/ORF filter, and
ESMFold structure prediction as the main experiment; for each predicted structure, BOTH mkdssp (main
experiment's tool, 8-state DSSP) and pydssp (independent from-scratch hydrogen-bond-map
reimplementation, 3-state {-,H,E}) compute secondary structure on the identical coordinates. Grid:
α∈{0,4,8,16,32}×seed∈{42,200}, n=150 generated/cell, 10 cells, dispatched fully detached
(setsid+nohup+disown, own process group, `timeout 2400s` per cell) across GPUs 3/4/5 (GPU 2 held by
another user the entire session). A sanity pilot (n=5) caught and fixed a real bug first (missing
`HF_HOME` env var causing ESMFold to silently re-download its checkpoint instead of reusing the
project's cache) before the full grid was launched.

**Result**: pydssp closely tracks mkdssp at every dose (within 0.01–0.02 absolute helix fraction,
within 0.002 absolute sheet fraction; Spearman ρ=0.886 p=6.4e-4 for pydssp vs ρ=0.935 p=7.0e-5 for
mkdssp on the identical structures within this variant, both closely matching the main experiment's
own ρ=0.759 p=1.7e-5). The dose-response conclusion is robust to this method swap.

**Byproduct**: because this variant necessarily saves the raw generated protein sequences (needed to
compute both SS-assignments), it incidentally closes — for the 10 cells it covers — the
sequence-inspectability gap flagged in C2's own `EXPERIMENT_AUDIT.md` (the main experiment discards
generated sequences after computing metrics).

## Details
- C1: `verify/C1_helix_feature_set/ROBUSTNESS.md`
- C2: `verify/C2_dose_response_steering/ROBUSTNESS.md`
- C3: `verify/C3_specificity_double_dissociation/ROBUSTNESS.md`

## Next Step

→ **C2 PASSes** (robust to SS-assignment choice) → this robustness story, with its explicit scope
caveat (untested structure-predictor axis), goes into the next-round draft.
→ **C1 is INTEGRITY_ONLY** → no back-edge action this pass; record in Open Items with the upgrade
suggestion `/auto-verify C1 — resume: true` when GPU/time budget allows.
→ **C3 is INCONCLUSIVE** → hand to iteration: `/auto-iteration-loop "α-helix knob — verify-inconclusive: C3"`.
Instruction to iteration: **fix the failing M3 mechanism harness (locate a genuine mid-plateau shared
dose where the treatment arm's own capability is preserved, add a real ≥30-direction random-direction
control, widen the α span); do not change the C3 claim wording.** The evaluation code itself
(experiment-audit was only WARN) can stay largely as-is; the mechanism sweep needs the fix.
→ Recommended future work (not this pass): retry an ESMFold↔alternate-structure-predictor swap for
C2 with a pre-staged model-weight cache or a longer time allowance, to close the one robustness axis
this pass left untested.
