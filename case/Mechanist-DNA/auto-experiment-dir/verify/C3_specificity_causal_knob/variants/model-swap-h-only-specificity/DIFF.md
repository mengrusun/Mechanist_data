# DIFF — C3 model-swap-h-only-specificity vs Main Experiment

## What changed
**Single change**: specificity readout endpoint swapped from **helix_hgi_w** (pLDDT-weighted HGI, H+G+I) to **helix_h_w** (pLDDT-weighted H-only, strict alpha only).

## What is frozen
- S arm generation: same generated sequences (reads from M3 arm files)
- 48 null directions: same (reads from M3 random null files)
- Matched control: same
- Structure prediction: same (ESMFold + OmegaFold)
- c*: same 21.4801
- Held-out seeds: same 200, 201
- Norm-matching: same (nulls constructed with |S|=19 features, same injection-site norm)

## Config change summary
```
main experiment: endpoint = helix_hgi_w   → variant: endpoint = helix_h_w
```

## Note on CI computation
Per-sample H-only values are not stored in M3 result files (only aggregate helix_h_w_mean per arm/direction). The variant uses aggregate-level empirical p-value and z-score (S delta vs null distribution) rather than cluster-bootstrap CI. This is a limitation of reading pre-computed aggregates; the test is still valid for establishing the empirical p-value.
