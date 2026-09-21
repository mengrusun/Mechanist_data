# Claim Ledger Figures

_Generated 2026-08-19T18:22:00.571024+08:00 - auto-ledger mode, style=publication, formats=pdf,png_

## C1 - Strong-form causal-steering claim (later falsified)

### `c1_dose_response` (line) - OK

_C1 dose-response: mean predicted %H vs steering coefficient alpha on the block-26 direction; rise is non-monotonic (alpha=4 dip) and later shown composition-confounded._

![c1_dose_response](figures/C1/c1_dose_response.png)

- vector: `figures/C1/c1_dose_response.pdf`
- source: `results/M2_dose_response.json`

### `c1_specificity` (line) - OK

_C1 specificity: the real direction raises %H with dose while a norm-matched random direction stays flat/declines._

![c1_specificity](figures/C1/c1_specificity.png)

- vector: `figures/C1/c1_specificity.pdf`
- source: `results/M3_specificity.json`

## C1_v2 - Honest narrowed claim (composition confound is the crux)

### `c1v2_predictor_triangulation` (grouped_bar) - OK

_C1_v2: the +14-16pt predicted-%H uplift reproduces across 3 independent SS predictors - not a single-probe artifact._

![c1v2_predictor_triangulation](figures/C1_v2/c1v2_predictor_triangulation.png)

- vector: `figures/C1_v2/c1v2_predictor_triangulation.pdf`
- source: `runs/iteration_round_1/M5_structural_gc_control.json`

### `c1v2_composition_confound` (multi_panel) - OK

_C1_v2 confound: steering collapses GC (0.44->0.13) and shifts amino-acid composition (Lys up, low-complexity up) in lockstep with the %H gain - the effect is not shown separable from composition._

![c1v2_composition_confound](figures/C1_v2/c1v2_composition_confound.png)

- vector: `figures/C1_v2/c1v2_composition_confound.pdf`
- source: `runs/iteration_round_1/M5_structural_gc_control.json`

## C2 - Eval-harness fidelity across method/dataset swaps

### `c2_swap_robustness` (bar) - OK

_C2 eval-harness fidelity holds across method and dataset swaps (all r >> 0.7 threshold)._

![c2_swap_robustness](figures/C2/c2_swap_robustness.png)

- vector: `figures/C2/c2_swap_robustness.pdf`
- source: `verify/C2_eval_harness_fidelity/ROBUSTNESS.md + results/E1_eval_harness_validation.json`
