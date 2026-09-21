# Mechanism Audit — C1 (main experiment, steering-coefficient sweep)

**overall_verdict: WARN**  ·  reviewer: gpt-5.6-luna-2026-07-09

| Check | Verdict | Detail |
|---|---|---|
| A. Coefficient sweep | WARN | unit_stated ✓ (raw additive multiplier; α=16 ≈ 1.2× block-26 mean act-norm); capability_metric_logged ✓ (ORF-valid, loglik, %E, GC per α); random_control ✓ (norm-matched, swept, flat); 32× geometric sweep + α=0 ✓; **plateau_locked ✗** (%H still rising 43.1→48.1 at the max tested α=16, no plateau/mid-plateau lock); wide_sweep_if_shortfall = met_criteria (pass criterion reached at α=16, so wider sweep not required). Off-target GC/loglik confound incompletely disentangled. |
| B–F | not_implemented | reserved |

**Net:** sweep + controls are adequate and the criterion was met, so this is not a mechanism FAIL — but the effect is unsaturated at the boundary α and the compositional confound is not fully separated → WARN.
