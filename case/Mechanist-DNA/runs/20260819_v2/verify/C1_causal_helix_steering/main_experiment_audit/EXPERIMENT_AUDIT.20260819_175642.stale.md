# Experiment Audit — C1 (main experiment, M1–M4)

**overall_verdict: FAIL**  ·  reviewer: gpt-5.6-luna-2026-07-09

| Check | Verdict | Note |
|---|---|---|
| A. GT provenance | PASS | SS-probe labels from real experimental PDB/DSSP; but generated-seq %H is a probe PREDICTION, not direct structural GT. |
| B. Score normalization | PASS | No self-referential normalization; random control norm-matched. |
| C. Result-file existence | PASS | Cited M1–M4 numbers present in results/*.json (verified on disk). |
| D. Dead code | WARN | Eval call-graph not re-executed in audit; result files are indirect evidence. |
| E. Scope | **FAIL** | Dose curve non-monotonic (α=4 dip 33.06); "monotonic" rests on arbitrary Spearman>0.5. Gains coincide with GC collapse 0.40→0.13 + loglik shift, so "preserved coding" + clean α-specific causal reading not established. |
| F. Eval type | **synthetic_proxy** | Generated-seq %H = ESM2-650M+probe, validated on NATURAL proteins (r=0.987) but applied to steered, GC-collapsed, OOD sequences with NO structural (ESMFold/DSSP) validation. %H rise may be a probe/compositional artifact. |

**Why FAIL:** the PRIMARY endpoint (%H of steered generations) is a synthetic proxy used far outside its validated domain, confounded by a severe GC-composition collapse, and the dose curve is non-monotonic. Computing swap-robustness around this anchor would be meaningless until the main experiment validates %H structurally on steered sequences and disentangles the GC confound.

**Iteration fix (not verify's job):** rerun /auto-experiment with (a) a structural %H validation of a steered-sequence subset (fold via an available predictor → DSSP, or an independent SS method) and (b) a GC-matched / composition-controlled analysis separating helix-propensity from AT-rich-codon compositional drift.
