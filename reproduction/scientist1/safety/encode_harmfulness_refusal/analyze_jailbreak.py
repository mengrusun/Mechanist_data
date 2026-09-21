"""Analyze jailbreak-attack activations for the harmfulness-preserved / refusal-suppressed
signature (Claim 4).

Uses the extracted activations produced by jailbreak_attacks.py plus the directions from
directions.py.  For each (attack, jailbreak-success) group we compute the mean projection
onto v_inst and v_post at the chosen layers.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

JB_DIR  = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/jailbreak_llama3"
RES_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/results"
ACT_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/acts_llama3"


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def main():
    d = np.load(os.path.join(RES_DIR, "directions_llama3.npz"))
    # We use layer 11 for both directions (chosen earlier).
    LAYER = 11
    v_h = unit(d["v_inst"][LAYER])   # harmfulness direction, measured at t_inst
    v_r = unit(d["v_post"][LAYER])   # refusal direction, measured at t_post
    # Also record the mean baseline projections from AdvBench harmful vs Alpaca benign
    for base_name in ["advbench_train", "alpaca_train"]:
        z = np.load(os.path.join(ACT_DIR, f"{base_name}.npz"), allow_pickle=True)
        hi = z["hs_inst"][:, LAYER].astype(np.float32)
        hp = z["hs_post"][:, LAYER].astype(np.float32)
        print(f"[{base_name}]  proj_h@inst={ (hi @ v_h).mean():+.3f}"
              f"   proj_r@post={ (hp @ v_r).mean():+.3f}")

    print()
    results = {"layer": LAYER, "attacks": {}}
    for attack in ["plain", "gcg", "dan", "persuasion", "jbb_role", "prefill"]:
        fp = os.path.join(JB_DIR, f"{attack}.npz")
        if not os.path.exists(fp):
            print(f"[missing] {fp}"); continue
        z = np.load(fp, allow_pickle=True)
        hi = z["hs_inst"][:, LAYER].astype(np.float32)   # (N, D)
        hp = z["hs_post"][:, LAYER].astype(np.float32)
        refusal = z["refusal"].astype(bool)
        outs = z["outputs"]

        proj_h = hi @ v_h  # (N,) projection onto harmfulness direction at t_inst
        proj_r = hp @ v_r  # (N,) projection onto refusal direction at t_post

        # split by refusal / not
        jb = ~refusal  # 'jailbroken' if not a refusal
        rec = {
            "n": int(len(refusal)),
            "refusal_rate": float(refusal.mean()),
            "asr_regex": float(jb.mean()),
            "proj_harm_all_mean":     float(proj_h.mean()),
            "proj_refuse_all_mean":   float(proj_r.mean()),
            "proj_harm_refused_mean": float(proj_h[refusal].mean()) if refusal.any() else None,
            "proj_refuse_refused_mean": float(proj_r[refusal].mean()) if refusal.any() else None,
            "proj_harm_jb_mean":       float(proj_h[jb].mean()) if jb.any() else None,
            "proj_refuse_jb_mean":     float(proj_r[jb].mean()) if jb.any() else None,
            "example_output_refused": None if not refusal.any() else str(outs[refusal.tolist().index(True)])[:200],
            "example_output_jb":      None if not jb.any() else str(outs[jb.tolist().index(True)])[:200],
        }
        results["attacks"][attack] = rec
        print(f"=== {attack} ===  n={rec['n']}  refusal={rec['refusal_rate']:.2f}  asr={rec['asr_regex']:.2f}")
        print(f"    proj_harm  (refused / jb): {rec['proj_harm_refused_mean']} / {rec['proj_harm_jb_mean']}")
        print(f"    proj_refuse(refused / jb): {rec['proj_refuse_refused_mean']} / {rec['proj_refuse_jb_mean']}")

    with open(os.path.join(RES_DIR, "jailbreak_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {os.path.join(RES_DIR, 'jailbreak_analysis.json')}")


if __name__ == "__main__":
    main()
