# DIFF — C2 model-swap-h-only-helix-frac vs Main Experiment

## What changed
**Single change**: primary structural readout endpoint swapped from **helix_hgi_w** (pLDDT-weighted HGI helix fraction, H+G+I) to **helix_h_w** (pLDDT-weighted H-only helix fraction, strict alpha only).

## What is frozen
- Evo2-7B generation: same (reads from existing M2 result files, no re-run)
- Generated sequences: same (same 30 M2 run files, held-out seeds 200+201)
- Structure prediction: same (ESMFold + OmegaFold, both predictors)
- DSSP assignment: same mkdssp 4.6.1 (helix_h_w_mean stored per run alongside helix_hgi_w_mean)
- Dose grid: same 10 c_sigma points
- c*: same 21.4801 σ_proj units
- pLDDT weighting: same (helix_h_w = pLDDT-weighted fraction, H-only residues)

## Why this is a valid model swap
helix_hgi_w includes all three DSSP helix types (H=alpha, G=3-10, I=pi); helix_h_w includes only strict alpha. If the dose-response trend is due to alpha-helix features, it should persist under the strict H-only definition.

## Config change summary
```
main experiment: endpoint = helix_hgi_w   → variant: endpoint = helix_h_w
```
