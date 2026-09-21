"""Steering with orthogonalized directions.

v_h_pure     = v_inst  (unit norm)
v_r_pure     = v_p_orth (unit norm), which is v_post minus its projection onto v_h.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    LLAMA3_PATH, load_model_and_tokenizer, format_llama3_chat, is_refusal,
    load_advbench, load_alpaca_benign, T_INST, T_POST,
)
from steering import Steerer, forward_and_extract, generate_all, unit

RES_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--read_layer", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()

    d = np.load(os.path.join(RES_DIR, "directions_llama3_orth.npz"))
    v_h = unit(d["v_h"][args.layer])
    v_r = unit(d["v_p_orth"][args.layer])

    # For reading projections we use the orthogonal directions at read_layer as well.
    v_h_r = unit(d["v_h"][args.read_layer])
    v_r_r = unit(d["v_p_orth"][args.read_layer])

    print("Loading Llama-3-8B-Instruct ...")
    model, tok = load_model_and_tokenizer(LLAMA3_PATH)
    device = next(model.parameters()).device

    benign  = load_alpaca_benign(n=args.n, seed=5)
    harmful = load_advbench(n=args.n, seed=5)

    conditions = [
        ("benign_baseline",       benign,  None, 0.0),
        ("harmful_baseline",      harmful, None, 0.0),
        ("benign_plus_vh",        benign,  v_h, +args.alpha),
        ("harmful_minus_vh",      harmful, v_h, -args.alpha),
        ("benign_plus_vr_orth",   benign,  v_r, +args.alpha),
        ("harmful_minus_vr_orth", harmful, v_r, -args.alpha),
    ]

    results = {"alpha": args.alpha, "layer": args.layer, "read_layer": args.read_layer, "n": args.n, "conditions": {}}
    layers_to_read = [args.read_layer]
    positions = (T_INST, T_POST)

    for name, prompts, v, alpha in conditions:
        print(f"\n=== {name} (alpha={alpha}) ===")
        t0 = time.time()
        if v is None:
            buf = forward_and_extract(model, tok, prompts, layers_to_read, positions,
                                       batch_size=args.batch_size, device=device)
            gens = generate_all(model, tok, prompts, max_new_tokens=args.max_new_tokens,
                                 batch_size=args.batch_size, device=device)
        else:
            with Steerer(model, args.layer, v, alpha):
                buf = forward_and_extract(model, tok, prompts, layers_to_read, positions,
                                           batch_size=args.batch_size, device=device)
                gens = generate_all(model, tok, prompts, max_new_tokens=args.max_new_tokens,
                                     batch_size=args.batch_size, device=device)
        p_h_i = buf[(args.read_layer, T_INST)] @ v_h_r
        p_r_p = buf[(args.read_layer, T_POST)] @ v_r_r
        refuse_rate = float(np.mean([is_refusal(o) for o in gens]))
        results["conditions"][name] = {
            "proj_h_at_inst_mean":  float(p_h_i.mean()),
            "proj_h_at_inst_std":   float(p_h_i.std()),
            "proj_r_orth_at_post_mean": float(p_r_p.mean()),
            "proj_r_orth_at_post_std":  float(p_r_p.std()),
            "refusal_rate_regex":   refuse_rate,
            "sample_outputs":       gens[:3],
        }
        print(f"  proj_h_orth @ inst = {p_h_i.mean():+.3f}    proj_r_orth @ post = {p_r_p.mean():+.3f}    refusal={refuse_rate:.3f}   ({time.time()-t0:.1f}s)")

    out_fp = os.path.join(RES_DIR, "steering_orth.json")
    with open(out_fp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_fp}")


if __name__ == "__main__":
    main()
