# C3 Model-Swap Variant: Qwen2.5-7B-Instruct

**Source**: runs/A4_verify_qwen/ (M4 milestone from main experiment)
**Dimension**: model
**Model path**: /data/zhenqian/models/Qwen2.5-7B-Instruct

## Artifacts (in runs/A4_verify_qwen/)

- qwen_C_e.json — Qwen's per-emotion component set C_e
- qwen_kstar.json — Qwen's selected (k_h*, k_n*)
- qwen_directions.npz — Qwen's per-layer emotion directions
- qwen_shortlist.json — Stage A shortlist for Qwen
- qwen_arm_A_val.json — Arm A val sweep results (9 configs per emotion)
- qwen_arm_B_val.json — Arm B val sweep results (9 configs per emotion)
- qwen_arm_C_val.json — Arm C val sweep results (9 configs per emotion)
- qwen_selected_configs.json — per-emotion best config per arm
- qwen_eval_generations.json — 2160 greedy continuations (120 stems x 6 emotions x 3 arms)
- qwen_judge_results.json — gpt-5.4 judge verdicts (2160 items)
- qwen_metrics.json — per-emotion accuracy, macro, pairwise CIs, verdict

## Result Summary

| | Arm A (Circuit) | Arm B (Prompting) | Arm C (Steering) |
|--|---|---|---|
| joy | 0.000 | 0.983 | 0.092 |
| sadness | 0.092 | 0.967 | 0.100 |
| anger | 0.008 | 0.975 | 0.175 |
| fear | 0.200 | 1.000 | 0.108 |
| surprise | 0.150 | 0.958 | 0.267 |
| disgust | 0.008 | 0.933 | 0.008 |
| **Macro** | **0.076** | **0.969** | **0.125** |

A > B: 0/6; A > C: 0/6. Verdict: not-supported. Agrees with Llama main experiment.
