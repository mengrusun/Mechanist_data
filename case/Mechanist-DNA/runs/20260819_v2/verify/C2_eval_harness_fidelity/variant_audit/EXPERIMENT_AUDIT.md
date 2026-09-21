# Variant Experiment Audit — C2 (verify variants)

**overall_verdict: WARN**  ·  reviewer: gpt-5.6-luna-2026-07-09

| Check | Verdict | Note |
|---|---|---|
| A. GT provenance | PASS | Experimental-PDB DSSP (p['ss8'], p['pctH']); not model-derived. |
| B. Score normalization | PASS | %H normalized by sequence length; no self-referential max/mean. |
| C. Result-file existence | PASS | Cited V1 (r=0.8505) & V2 (r=0.9792) values verified present in on-disk result.json; computed on live main() paths. |
| D. Dead code | PASS | All scoring/correlation/frame-recovery code executed. |
| E. Scope | WARN | Finite tested sets (200 / 310), one run per variant, frame recovery ≤100 proteins — keep wording to tested conditions. |
| F. Eval type | real_gt | Central correlation vs experimental DSSP; frame recovery is a method-independent synthetic-ORF proxy. |

**Net WARN** (scope only). Both variants trusted → counted in robustness with an `[INTEGRITY: WARN — experiment]` tag. (Note: an initial audit pass FAILed Check C solely because the result files were not shown to the reviewer; re-audited with the on-disk artifacts, Check C is PASS.)
