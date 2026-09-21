# Experiment Audit — C2 (main experiment, E1)

**overall_verdict: WARN**  ·  reviewer: gpt-5.6-luna-2026-07-09

| Check | Verdict | Note |
|---|---|---|
| A. GT provenance | PASS | Reference = experimental PDB X-ray + DSSP; real structure-based GT, not model-derived. |
| B. Score normalization | PASS | Direct Pearson r(fast %H, DSSP %H); no self-referential normalization. |
| C. Result-file existence | WARN→(pass on disk) | r=0.9874, Q3=0.858, frameRec=1.0, n=200 present in results/E1_eval_harness_validation.json. |
| D. Dead code | WARN | Harness source not re-executed in audit. |
| E. Scope | WARN | Correlation n=200 vs frame-recovery n=100 — keep scopes distinct. |
| F. Eval type | **real_gt** | Experimental-DSSP reference. |

**Net WARN** driven by verification-limitation + a mild scope-distinctness note; methodology is sound. C2 **admitted** to Stage 2 with `warn_source=experiment`.
