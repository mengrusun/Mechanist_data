import numpy as np, os
JB = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/jailbreak_llama3"
for atk in ["plain", "gcg", "dan", "persuasion"]:
    z = np.load(os.path.join(JB, f"{atk}.npz"), allow_pickle=True)
    outs = z["outputs"]
    ref = z["refusal"]
    print(f"=== {atk}  refusal_rate={ref.mean():.2f} ===")
    for i in [0, 1, 5, 12, 25]:
        if i >= len(outs): break
        tag = "REFUSED" if ref[i] else "COMPLIED"
        print(f"  [{tag}]: {str(outs[i])[:180]!r}")
    print()
