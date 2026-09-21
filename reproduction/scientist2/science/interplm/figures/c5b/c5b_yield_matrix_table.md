| Feature | Property | no_steer (baseline) | sae_clamp (α∈{0.5,1,2,4}·σ_f) | mean_add | random_clamp |
|---|---|---|---|---|---|
| 3998 | TM helix | **0.333** | 0.200 (all α) | 0.133–0.200 | 0.200 (all α) |
| 4209 | Zn finger | **0.133** | 0.100 (all α) | 0.100 (all α) | 0.100 (all α) |
| 1240 | SP core | **0.167** | 0.133 (all α) | 0.133 (all α) | 0.133 (all α) |

Notes: yield = fraction of the 30 generated sequences (10 seqs × 3 seeds) that satisfy the target property's rule-based checker (Kyte–Doolittle window for TM, ProSite regex for Zn, SignalP-like N-terminal core rule for SP). Baseline yields dominate every steered arm on every feature. Plausibility band-pass ≥ 0.67 on all steered arms, so drops are not off-distribution collapse — the steering is ineffective, not destructive, under this narrow ≤1-OOM dose ladder.
