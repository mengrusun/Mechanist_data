# Variant Experiment Audit — C1 (model-swap-evo2-7b-262k)

**Scope:** `verify/C1_helix_gain_doseresponse/variants/`. **Overall verdict: PASS.**

| Check | Verdict | Finding |
|---|---|---|
| A. GT provenance | PASS | Reuses the frozen ESMFold→DSSP assay (model-independent); helix GT not derived from the swapped model. |
| B. Score normalization | PASS | Identical `helix_frac` + pLDDT policy; same `score_generations` code path; no model-max normalization. |
| C. Result-file existence | PASS | 8 dose cells with `per_gen_helix` arrays in `result.json`; `analysis.json` recomputes ρ=0.878, Δ=0.094, Cliff's δ=0.201 from raw. |
| D. Dead code | PASS | Full CAA→steer→generate→score→analyze path invoked; no orphaned metric. |
| E. Scope | PASS | Single intended model swap; reduced grid documented as cost-driven and applied symmetrically (incl. coef-0 in-model baseline); reported as one transfer probe, no overclaim. |
| F. Evaluation type | PASS | `synthetic_proxy` (ESMFold + DSSP), identical to main experiment. |

Anti-circularity preserved (CAA on DEV, eval on TEST-split primers, same seed offset); survivorship-safe endpoint inherited.
