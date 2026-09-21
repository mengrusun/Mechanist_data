"""
Emit a compact summary table for the final report.
"""
import json
import sys

runs = [
    ("Llama-3.1-8B-Instruct / TriviaQA",
     "results/probe_llama31_8b_instruct_triviaqa.json",
     "results/orth_llama31_answer_end.json",
     "results/orth_llama31_conf_prefix.json"),
    ("Qwen2.5-7B-Instruct / TriviaQA",
     "results/probe_qwen25_7b_instruct_triviaqa.json",
     "results/orth_qwen25_answer_end.json",
     "results/orth_qwen25_conf_prefix.json"),
]

print(f"{'model/dataset':<40} {'acc':>6} {'verbAUC':>8} {'A_bestAUC':>10} {'C_bestAUC':>10} {'cos@ans':>8} {'cos@cf':>8} {'|cos|/(1/sqrtD)':>16}")
for name, probe_json, orth_ans_json, orth_cf_json in runs:
    with open(probe_json) as f:
        p = json.load(f)
    with open(orth_ans_json) as f:
        oa = json.load(f)
    with open(orth_cf_json) as f:
        oc = json.load(f)
    v = p["verbalized_calibration"]
    aucs_ans = [d["probeA"]["auc"] for d in p["per_position"]["answer_end"]]
    chis_ans = [d["probeC_hi"]["auc"] for d in p["per_position"]["answer_end"]]
    print(f"{name:<40} "
          f"{v['accuracy']:>6.3f} "
          f"{v['auc_vs_correct']:>8.3f} "
          f"{max(aucs_ans):>10.3f} "
          f"{max(chis_ans):>10.3f} "
          f"{oa['across_AC_mean_abs_cos']:>8.3f} "
          f"{oc['across_AC_mean_abs_cos']:>8.3f} "
          f"{oa['across_AC_mean_abs_cos']/oa['expected_random_abs_cos_1_over_sqrtD']:>16.2f}")
