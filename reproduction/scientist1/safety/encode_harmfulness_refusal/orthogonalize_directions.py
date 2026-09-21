"""Compute orthogonalized harmfulness/refusal directions.

Given DoM directions v_h @ layer L (position t_inst) and v_p @ layer L (position t_post),
the two are correlated (both reflect harmful vs benign inputs).  We orthogonalize:
   v_h_orth = v_h
   v_p_orth = v_p - (v_p · v_h_hat) * v_h_hat     with v_h_hat = v_h / ||v_h||
Then v_p_orth is the component of v_p that is orthogonal to v_h.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RES_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/results"


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def main():
    d = np.load(os.path.join(RES_DIR, "directions_llama3.npz"))
    v_inst = d["v_inst"].astype(np.float32)   # (L+1, D)
    v_post = d["v_post"].astype(np.float32)
    L, D = v_inst.shape[0] - 1, v_inst.shape[1]

    v_h_orth = v_inst.copy()
    v_p_orth = np.empty_like(v_post)
    for l in range(v_inst.shape[0]):
        vh = unit(v_inst[l])
        # remove component of v_post along v_h
        v_p_orth[l] = v_post[l] - (v_post[l] @ vh) * vh

    # cosine similarities
    def cos(a, b):
        return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
    cos_orth = cos(v_h_orth, v_p_orth)
    cos_before = cos(v_inst, v_post)
    print("Layer  cos(v_h, v_p_before)  cos(v_h, v_p_orth)  ||v_p_orth||/||v_p||")
    for l in range(0, v_inst.shape[0], 2):
        rn = np.linalg.norm(v_p_orth[l]) / (np.linalg.norm(v_post[l]) + 1e-8)
        print(f"  L{l:2d}    {cos_before[l]:+.4f}                {cos_orth[l]:+.4f}              {rn:.4f}")

    np.savez(os.path.join(RES_DIR, "directions_llama3_orth.npz"),
             v_h=v_h_orth, v_p_orth=v_p_orth)
    print("\nSaved orthogonalized directions.")


if __name__ == "__main__":
    main()
