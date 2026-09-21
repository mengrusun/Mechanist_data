import numpy as np, os, json
z = np.load("/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/jailbreak_llama3/prefill.npz", allow_pickle=True)
outs = z["outputs"]; ref = z["refusal"]; base = z["base_prompts"]
print(f"prefill: n={len(outs)} refusal={ref.mean():.3f}")
jb_idx = np.where(~ref)[0]
print(f"jailbroken n={len(jb_idx)}")
for i in jb_idx[:8]:
    print(f"---\n GOAL: {base[i]}")
    print(f" OUT:  {str(outs[i])[:220]}")
