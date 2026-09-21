## C3: main-experiment integrity REPAIRED (iteration-② fix) → ⚪ INTEGRITY_ONLY (was 🟡 INCONCLUSIVE)

- swap_variants_run: false
- Main-experiment verdict on C3: supported (per main experiment; robustness-under-swaps not evaluated this pass — MAX_VERIFY_CLAIMS cap spent on C2)
- Main-experiment integrity (Phase 2): **WARN** — combined = max_severity(exp=WARN, mech=**WARN**). *Was FAIL (mech FAIL) → repaired by the iteration-loop type-② mechanism-harness fix.*
- prior_state: INCONCLUSIVE (mechanism-audit FAIL: α=16 capability-degraded decisive dose, no random-direction control)
- stage2_skip_reason: max_verify_claims_cap (C2 was the single Stage-2 pick; C3 admitted post-repair but not swap-tested this pass)
- robustness: null (swap variants not run)
- n_eligible: 0
- n_pass: 0

Interpretation: the iteration loop fixed the M3 mechanism harness that caused the original FAIL. The
decisive specificity statistics are now computed at the **capability-preserved mid-plateau dose α=8**
(S valid-ORF 0.888 ≈ baseline; vs 0.667 at the old α=16), and a **genuine 33-direction random-direction
control** (norm-matched to ‖S.base_dir‖, sampled excluding S∪β∪matched, identical fold→gate→DSSP
pipeline) was added. Result (`results/m3_specificity_summary_v2.json`): S's helix rise +0.096 exceeds
all 33 random directions (0/33 ≥ S; empirical one-sided p=0.029; z=3.02), with S's valid-ORF (0.888)
≈ the random-null mean (0.887) so the effect is not a capability-selection artifact; secondary
matched-control Mann-Whitney p=3.5e-4. The β-sheet-amplification arm is reported honestly as a
**NEGATIVE result** (Δsheet ≈ +0.005) — C3 rests on helix-axis specificity (S vs random-null / matched)
at a clean dose plus β-does-not-raise-helix, not on a symmetric double dissociation.

Re-audit: `main_experiment_audit/MECHANISM_AUDIT.md` (verdict WARN; prior FAIL archived as
`MECHANISM_AUDIT.pre_iteration_fix.md`). Because combined = WARN (not FAIL), C3 is no longer
INCONCLUSIVE; it is now INTEGRITY_ONLY (main-experiment integrity pass-with-warn; swap-test deferred by
the MAX_VERIFY_CLAIMS cap). Upgrade to a swap-robustness verdict via `/auto-verify C3 — resume: true`
when GPU/time budget allows.
