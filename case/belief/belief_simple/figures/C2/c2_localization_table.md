| Model | Target | |H*| | Heads (layer, head) | Δ target | Δ off-belief | Δ WK | PPL ratio | mean_rh + 2σ | C2a | C2b | C2c | C2d | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pythia-410m | personal | 2 | (13,1), (8,8) | 0.372 | -0.314 | +0.000 | 1.010× | 0.081 | PASS | PASS | PASS | PASS | **LOCALIZED** |
| pythia-1b | personal | 2 | (12,1), (9,1) | 0.471 | -0.123 | +0.004 | 1.019× | 0.151 | PASS | PASS | PASS | PASS | **LOCALIZED** |
| pythia-1b | attributed | 1 | (4,1) | 0.463 | +0.090 | +0.004 | 1.011× | 0.257 | PASS | PASS | PASS | PASS | **LOCALIZED** |
| pythia-2.8b | personal | 4 | (14,16), (15,3), (13,1), (12,4) | 0.307 | -0.150 | +0.004 | 1.007× | 0.035 | PASS | PASS | PASS | PASS | **LOCALIZED** |
| pythia-2.8b | attributed | 1 | (5,22) | 0.355 | +0.001 | +0.004 | 1.003× | 0.033 | PASS | PASS | PASS | PASS | **LOCALIZED** |
