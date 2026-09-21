| Model | Target | K | Head set | target_drop | other_drop | wk_drop | Pile PPL ratio | All 4 thresholds | Fisher outside 2σ | Scope |
|---|---|---:|---|---:|---:|---:|---:|:---:|:---:|:---:|
| pythia-410m | PB | 20 | — (out-of-scope; K=20 spanning L8–L23) | 0.386 ✓ | -0.308 ✓ | —  | — ✗ | ✗ | ✗ | out-of-scope |
| pythia-1b | PB | 3 | (L12,H1), (L13,H3), (L9,H1) | 0.403 ✓ | -0.115 ✓ | 0.000 ✓ | 1.023× ✓ | ✓ | ✓ | in-scope |
| pythia-1b | AB | 1 | (L4,H1) | 0.513 ✓ | +0.070 ✓ | 0.004 ✓ | 1.012× ✓ | ✓ | ✓ | in-scope |
| pythia-2.8b | PB | 13 | (L14,H16), (L15,H3), (L27,H21), (L13,H1), (L6,H6), (L15,H14), … (K=13) | 0.452 ✓ | -0.093 ✓ | 0.018 ✓ | 1.023× ✓ | ✓ | ✓ | in-scope |
| pythia-2.8b | AB | 1 | (L5,H22) | 0.500 ✓ | +0.000 ✓ | 0.004 ✓ | 1.003× ✓ | ✓ | ✓ | in-scope |
