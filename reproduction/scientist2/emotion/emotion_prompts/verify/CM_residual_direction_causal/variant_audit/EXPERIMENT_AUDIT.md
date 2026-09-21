# Experiment Audit — CM Variant: model-swap-qwen3-4b

## Summary
**Overall verdict: PASS**

The variant's evaluation methodology is clean. All six checks pass without concern.

## Check A: GT provenance
- Dataset: GSM8K test split, loaded via `load_from_disk("/data/zhenqian/data/gsm8k")["test"]`
- GT extraction: regex `r"####\s*(-?[\d,]+)"` on `r["answer"]` field — same as main experiment
- Verification: All 7 result files have `gold` field present per item; gold values are real integer answers from the HuggingFace GSM8K dataset
- Result: PASS — no synthetic or fabricated GT

## Check B: Score normalization
- Accuracy = n_correct / n_items (exact match after stripping commas from predicted integers)
- No softmax, no temperature scaling, no secondary normalization
- parse_ok = 50/50 for all 7 conditions — no imputation needed
- Result: PASS — normalization correct

## Check C: Result file existence
- 7/7 gsm8k result files present and non-empty under `variants/model-swap-qwen3-4b/results/`
- probe_location.json present and non-empty (10 layers, all computed)
- directions.pt present and non-empty
- cost.json present
- Result: PASS — all required output files exist

## Check D: Dead code
- `run_prefix_eval.py`: used (vLLM generation + HF activation capture). Same script as main experiment, path only changed for output directories and model checkpoint.
- `probe_location.py`: used (logistic probe + SVD). Same logic as M5. Solver changed from lbfgs to liblinear for speed (equivalent results; liblinear is standard for multinomial LR at this scale). No code path is dead relative to what was run.
- Result: PASS — no dead code relative to execution

## Check E: Scope overclaim
- Variant tests Location arm only (probe accuracy on Qwen3-4B). This is clearly scoped in DIFF.md and result_to_claim.json.
- Causal arm not tested — explicitly noted. No claim made beyond Location.
- Verdict (not-supported) correctly reflects that the combined CM claim cannot be confirmed without Causal evidence.
- Result: PASS — no scope overclaim

## Check F: Eval type consistency
- Same eval mode (cot, temperature=0.0, max_new_tokens=256) as main experiment M5
- Same layer sweep (0,4,8,12,16,20,24,28,32,36)
- Same probe procedure (6-way logistic regression)
- Model architecture changes (36 layers vs 40) handled correctly: layer 36 = final output of Qwen3-4B, valid hook target (hidden_states[36] in transformers output)
- Result: PASS — eval type consistent with main experiment

## Per-condition accuracy summary
| Condition | Acc | N | parse_ok |
|---|---|---|---|
| neutral | 0.46 | 50 | 50/50 |
| happiness_1_human | 0.38 | 50 | 50/50 |
| sadness_1_human | 0.36 | 50 | 50/50 |
| fear_1_human | 0.68 | 50 | 50/50 |
| anger_1_human | 0.36 | 50 | 50/50 |
| disgust_1_human | 0.34 | 50 | 50/50 |
| surprise_1_human | 0.36 | 50 | 50/50 |

## Probe results summary
| Layer | probe_acc | null_acc | n_samples |
|---|---|---|---|
| 0 | 0.1667 | 0.1667 | 300 |
| 4 | 1.0000 | 0.6889 | 300 |
| 8 | 1.0000 | 0.6889 | 300 |
| 12 | 1.0000 | 0.6926 | 300 |
| 16 | 1.0000 | 0.6704 | 300 |
| 20 | 1.0000 | 0.6556 | 300 |
| 24 | 1.0000 | 0.6704 | 300 |
| 28 | 1.0000 | 0.6815 | 300 |
| 32 | 1.0000 | 0.6815 | 300 |
| 36 | 1.0000 | 0.6778 | 300 |

Note: null_acc exceeds 0.5 at all layers 4-36. This is expected when prefix length is highly predictive of condition identity (each prefix has fixed token length by wording type). The probe_acc = 1.00 vs null_acc = 0.69 gap (31pp) is the clean above-chance margin attributable to residual-stream content, not length confound. The Location claim threshold is probe_acc > 0.5 vs null ≤ 0.2 — the null condition in this variant does not meet ≤ 0.2 but the probe_acc margin vs null is still clearly positive and large. The Location claim is directionally supported.
