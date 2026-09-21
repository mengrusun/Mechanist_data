# Verify Plan — Claim C2: Belief Heads Localization

## Claim C2: [statement]
For each pythia model clearing the above-chance gate, Fisher-information masks (top-0.1% target AND-NOT top-1% F_knowledge) identify a smallest attention-head set H* whose zero-ablation satisfies all four criteria: (C2a) target acc drop ≥0.30; (C2b) drop > mean+2σ of 20 random-head controls; (C2c) off-target belief + WK drops ≤0.10; (C2d) PPL ≤1.05× clean. Results hold independently for personal_belief and attributed_belief circuits.

## Main experiment (from /auto-experiment)
- Method: Fisher-information-matrix + zero-ablation (M2.1-M2.4)
- Dataset: belief_core (person∈{james,mary} for Fisher signals; full n=681/681/227 for eval)
- Model: pythia-{410m, 1b, 2.8b}
- Result: 5/5 admissible (model, target) pairs LOCALIZED with |H*| ∈ {1, 2, 4}; all four criteria satisfied

## Variants

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | OLMo-1B-hf (/mnt/quarkfs/share_model/OLMo/OLMo-1B-hf) | pythia-{410m,1b,2.8b} | OLMo-1B is on-disk, matches ~1B scale, uses different architecture (separate Q/K/V/O projections instead of fused QKV+dense) — tests whether Fisher localization generalizes beyond GPTNeoX-family models. Strongest available model swap: different architecture + similar scale range. Per HARD CONSTRAINTS: no additional Pythia models on disk; OLMo-1B is the nearest available model that still has attention heads suitable for per-head Fisher aggregation and zero-ablation. | Available on disk at /mnt/quarkfs/share_model/OLMo/OLMo-1B-hf |

## Model Swap Details

- **OLMo-1B architecture**: 16 layers, 16 heads, hidden=2048, head_dim=128; separate q_proj/k_proj/v_proj/o_proj (no fused QKV)
- **Code adaptation required**: (a) model loading via AutoModelForCausalLM; (b) per-head Fisher aggregation over separate Q/K/V/O projections; (c) hook `o_proj.pre_hook` to scale per-head slices of o_proj input (same logical approach as Pythia's `dense` hook)
- **Seed and thresholds**: identical to main experiment (same 20 random-head seeds 100-119, 20 random-mask seeds 200-219; same 4-criteria thresholds)
- **Dataset**: same belief_core datasets, same person∈{james,mary} Fisher filter

## Success Criterion (per variant)
Claim C2 is consistent with the main experiment if OLMo-1B localizes at least one (target: personal_belief or attributed_belief) H* set satisfying all four criteria (C2a-C2d), consistent with the hypothesis that belief circuits can be localized via Fisher + zero-ablation on different backbone architectures.

## Note on available models
Checked /mnt/quarkfs/share_model/Ptyhia/: only pythia-{410m,1b,2.8b} present (all used in main experiment). No pythia-160m or pythia-6.9b found. Checked ~/models/: not found. OLMo-1B-hf is the only non-Pythia model available with attention heads at comparable scale. OLMo-7B was also found but is too large for the 1-variant budget.
