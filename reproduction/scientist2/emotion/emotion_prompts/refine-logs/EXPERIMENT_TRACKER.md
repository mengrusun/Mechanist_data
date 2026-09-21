# EXPERIMENT TRACKER (plan-level)

Plan-level rows written by `/auto-claim` Phase 4.5. `/auto-experiment` Phase 5 updates `Status` in place (`pending → running → done/failed`) and fills the last three columns; Phase 5.6 appends ablation rows. `/auto-iteration-loop` does not touch this file.

| Milestone | Run ID | Cmd (template-instantiated) | Est GPU-h | Status | Result | Notes |
|---|---|---|---|---|---|---|
| M1 | M1_corpus | python scripts/build_prefixes.py | 0.00 | done | 26 conditions + 8 noise-floor variants | dmxapi gpt-5.4; used pre-baked LLM variants as fallback if API failed on individual cells |
| M2 | M2_neutral | run_prefix_eval.py --task gsm8k --condition_id neutral --n_items 500 (vLLM) | 0.10 | done | acc=0.820 | baseline |
| M2 | M2_happiness_1_human | ... condition_id happiness_1_human ... | 0.11 | done | acc=0.708 (Δ=-0.112) | biggest -Δ; crosses noise floor |
| M2 | M2_happiness_1_llm | ... condition_id happiness_1_llm ... | 0.10 | done | acc=0.856 (Δ=+0.036) | |
| M2 | M2_happiness_2_human | ... condition_id happiness_2_human ... | 0.10 | done | acc=0.884 (Δ=+0.064) | crosses noise floor |
| M2 | M2_happiness_2_llm | ... condition_id happiness_2_llm ... | 0.10 | done | acc=0.868 (Δ=+0.048) | |
| M2 | M2_sadness_1_human | ... condition_id sadness_1_human ... | 0.11 | done | acc=0.766 (Δ=-0.054) | crosses noise floor |
| M2 | M2_sadness_1_llm | ... condition_id sadness_1_llm ... | 0.11 | done | acc=0.820 (Δ=+0.000) | |
| M2 | M2_sadness_2_human | ... condition_id sadness_2_human ... | 0.11 | done | acc=0.810 (Δ=-0.010) | |
| M2 | M2_sadness_2_llm | ... condition_id sadness_2_llm ... | 0.11 | done | acc=0.830 (Δ=+0.010) | |
| M2 | M2_fear_1_human | ... condition_id fear_1_human ... | 0.11 | done | acc=0.846 (Δ=+0.026) | |
| M2 | M2_fear_1_llm | ... condition_id fear_1_llm ... | 0.10 | done | acc=0.802 (Δ=-0.018) | |
| M2 | M2_fear_2_human | ... condition_id fear_2_human ... | 0.11 | done | acc=0.808 (Δ=-0.012) | |
| M2 | M2_fear_2_llm | ... condition_id fear_2_llm ... | 0.11 | done | acc=0.884 (Δ=+0.064) | crosses noise floor |
| M2 | M2_anger_1_human | ... condition_id anger_1_human ... | 0.11 | done | acc=0.838 (Δ=+0.018) | |
| M2 | M2_anger_1_llm | ... condition_id anger_1_llm ... | 0.11 | done | acc=0.822 (Δ=+0.002) | |
| M2 | M2_anger_2_human | ... condition_id anger_2_human ... | 0.11 | done | acc=0.888 (Δ=+0.068) | crosses noise floor |
| M2 | M2_anger_2_llm | ... condition_id anger_2_llm ... | 0.11 | done | acc=0.782 (Δ=-0.038) | |
| M2 | M2_disgust_1_human | ... condition_id disgust_1_human ... | 0.11 | done | acc=0.826 (Δ=+0.006) | |
| M2 | M2_disgust_1_llm | ... condition_id disgust_1_llm ... | 0.11 | done | acc=0.782 (Δ=-0.038) | |
| M2 | M2_disgust_2_human | ... condition_id disgust_2_human ... | 0.11 | done | acc=0.826 (Δ=+0.006) | |
| M2 | M2_disgust_2_llm | ... condition_id disgust_2_llm ... | 0.11 | done | acc=0.894 (Δ=+0.074) | crosses noise floor |
| M2 | M2_surprise_1_human | ... condition_id surprise_1_human ... | 0.11 | done | acc=0.820 (Δ=+0.000) | |
| M2 | M2_surprise_1_llm | ... condition_id surprise_1_llm ... | 0.11 | done | acc=0.832 (Δ=+0.012) | |
| M2 | M2_surprise_2_human | ... condition_id surprise_2_human ... | 0.11 | done | acc=0.826 (Δ=+0.006) | |
| M2 | M2_surprise_2_llm | ... condition_id surprise_2_llm ... | 0.11 | done | acc=0.756 (Δ=-0.064) | crosses noise floor |
| M2 | M2_filler_matched | ... condition_id filler_matched_length ... | 0.11 | done | acc=0.816 (Δ=-0.004) | specificity control, essentially null |
| M2b | M2b_ws | run_prefix_eval.py --condition_id neutral_ws --skip_activations | 0.04 | done | acc=0.834 (Δ=+0.014) | |
| M2b | M2b_casing | ... condition_id neutral_casing ... | 0.05 | done | acc=0.852 (Δ=+0.032) | |
| M2b | M2b_listmark | ... condition_id neutral_listmark ... | 0.05 | done | acc=0.770 (Δ=-0.050) | |
| M2b | M2b_para1 | ... condition_id neutral_para1 ... | 0.06 | done | acc=0.842 (Δ=+0.022) | |
| M2b | M2b_para2 | ... condition_id neutral_para2 ... | 0.05 | done | acc=0.840 (Δ=+0.020) | |
| M2b | M2b_para3 | ... condition_id neutral_para3 ... | 0.06 | done | acc=0.812 (Δ=-0.008) | |
| M2b | M2b_para4 | ... condition_id neutral_para4 ... | 0.05 | done | acc=0.878 (Δ=+0.058) | |
| M2b | M2b_para5 | ... condition_id neutral_para5 ... | 0.05 | done | acc=0.868 (Δ=+0.048) | 90th-pct \|Δ\| = 0.0524 (C1 noise-floor threshold) |
| M3 | M3_socialiqa_x26 | grid: task=socialiqa × 26 conditions × 500 items × MCQ_LL | 1.24 | done | spread=0.030 (max-min over 24 emotional conditions) | |
| M3 | M3_medqa_x26 | grid: task=medqa × 26 conditions × 500 items × MCQ_LL | 1.28 | done | spread=0.028 | GSM8K spread (0.186) >> Social (0.030) — C2 clearly failed |
| M4 | M4_c3_analysis | python scripts/analyze_c3.py --gsm8k runs/M2 --socialiqa runs/M3 --medqa runs/M3 --out reports/M4_c3_analysis.json | 0.00 | done | C2 fail; C3a pass; C3b pass (3/6) | argmax_by_task = {gsm8k: fear, socialiqa: surprise, medqa: fear} |
| M5 | M5_probe_location | python scripts/probe_location.py --activations_dir runs/M2 --layers 0,4,8,12,16,20,24,28,32,36 --n_seeds 3 --out reports/M5_location.json | 0.50 | done | probe acc=100% at layers 4–36; top-2=[4,8] | 6-way emotion identity linearly separable; null baseline ~28% |
| M6 | M6_steer_baseline_L4_a0 | mechanism_causal_batch.py --intervention steer --alpha 0 --site 4 (50 items) | 0.29 | done | acc=0.94 (α=0 sanity) | matches M2 baseline on first 50 items |
| M6 | M6_steer_L4_am1 | ... --alpha -1.0 --site 4 | 0.32 | done | acc=0.92 (Δ=-0.02) | at noise floor |
| M6 | M6_steer_L4_ap0p5 | ... --alpha +0.5 --site 4 | 0.32 | done | acc=0.92 (Δ=-0.02) | flat dose-response |
| M6 | M6_steer_L4_ap1 | ... --alpha +1.0 --site 4 | 0.27 | done | acc=0.94 (Δ=+0.00) | flat dose-response |
| M6 | M6_steer_L8_am1 | ... --alpha -1.0 --site 8 | 0.32 | done | acc=0.92 (Δ=-0.02) | |
| M6 | M6_steer_L8_ap0p5 | ... --alpha +0.5 --site 8 | 0.34 | done | acc=0.92 (Δ=-0.02) | |
| M6 | M6_steer_L8_ap1 | ... --alpha +1.0 --site 8 | 0.29 | done | acc=0.90 (Δ=-0.04) | slight monotone decrease |
| M6 | M6_patch_happiness2h_L4 | ... --intervention patch --site 4 (source=happiness_2_human) | 0.25 | done | acc=0.94 (Δ=+0.00) | patching same-with-same is a no-op-projection |
| M6 | M6_patch_happiness2h_L8 | ... --intervention patch --site 8 | 0.29 | done | acc=0.90 (Δ=-0.04) | |
| M6 | M6_patch_filler_L4 | filler-control patching at L4 | 0.30 | descoped | not run | budget cap hit at 9.08 GPU-h; deferred to /auto-verify |
| M6 | M6_patch_filler_L8 | filler-control patching at L8 | 0.30 | descoped | not run | budget cap hit |
| M6 | M6_steer_medqa_L8_ap1 | off-target MedQA steering | 0.30 | descoped | not run | budget cap hit |
| M6 | M6_steer_L20_null_ap1 | off-layer null control | 0.30 | descoped | not run | budget cap hit |
| M7 | M7a_reward_table | build 2000×13 GSM8K reward table | 2.17 | descoped | not run | 10 GPU-h budget consumed by M2+M3+M6; C4 deferred to /auto-verify |
| M7 | M7b_sft_seed{42,200,201} | policy SFT 3 seeds | 0.60 | descoped | not run | (depends on M7a) |
| M7 | M7c_rl_seed{42,200,201} | policy REINFORCE 3 seeds | 0.60 | descoped | not run | |
| M7 | M7d_eval_seed{42,200,201} | held-out eval 3 seeds | 0.13 | descoped | not run | |

## GPU-hours accounting

| Milestone | Est GPU-h | Actual GPU-h | Δ |
|---|---|---|---|
| M1 | 0.00 | 0.00 | 0.00 |
| M2 | 1.08 | 2.83 | +1.75 (larger due to model reload overhead per subprocess; each run ≈ 6 min including model load) |
| M2b | 0.33 | 0.54 | +0.21 |
| M3 socialiqa | 0.54 | 1.24 | +0.70 |
| M3 medqa | 0.54 | 1.28 | +0.74 |
| M4 | 0.00 | 0.00 | 0.00 |
| M5 | 0.50 | 0.50 | 0.00 |
| M6 (9 of 13 runs completed) | 0.60 | 2.69 | +2.09 (transformers hooks slower than vLLM; per-run wall ≈ 17 min at 50 items) |
| M7 | 2.90 | 0.00 | descoped |
| **Total** | **6.53** | **9.08** | **+2.55** (91% of 10-GPU-h budget) |
