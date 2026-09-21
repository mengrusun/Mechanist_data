"""Measure the correlation between language-specific activation strength and reasoning accuracy.

For each test prompt used in the baseline eval, we run a forward pass with output_hidden_states
and compute, at a set of layers, the projection norm ||P^T (h - c)|| of the *last token* onto
the language subspace. Then we correlate this with baseline correctness (binary) both
overall and within each language.

Usage:
  python code/lang_strength_corr.py --baseline results/main_baseline.json
"""
import os, sys, json, argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import pointbiserialr, pearsonr

sys.path.insert(0, os.path.dirname(__file__))
from common import MODEL_PATH, LANGS, load_mgsm, build_prompt, apply_chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--layers", type=str, default="4,8,12,16,20,24,28",
                    help="hidden_states indices (0=embed)")
    ap.add_argument("--k_lang", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="results/lang_strength_corr.json")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    print("layers:", layers, "K:", args.k_lang)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    subs = np.load("cache/subspace.npz")
    bases = subs["bases"]      # (L+1, D, K_all)
    centres = subs["centres"]  # (L+1, D)

    with open(args.baseline) as f:
        base = json.load(f)
    per_lang_rows = base["results"]

    data = load_mgsm("test")

    strength_per_lang = {}       # {lg: {layer: array of shape (N,)}}
    correct_per_lang  = {}       # {lg: array of shape (N,)}

    for lg in LANGS:
        rows = per_lang_rows.get(lg, {}).get("rows", [])
        if not rows:
            continue
        idxs = [r["idx"] for r in rows]
        prompts = [apply_chat(tok, build_prompt(data[lg].iloc[i]["question"], lg),
                              enable_thinking=False) for i in idxs]

        strengths = {L: [] for L in layers}
        with torch.inference_mode():
            for b0 in range(0, len(prompts), args.batch):
                b_prompts = prompts[b0:b0 + args.batch]
                enc = tok(b_prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=1024).to("cuda:0")
                out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
                attn = enc["attention_mask"]
                # last valid token in each row
                # with left padding, last valid token is the LAST index
                last_idx = attn.shape[1] - 1
                for L in layers:
                    h = out.hidden_states[L][:, last_idx].float().cpu().numpy()  # (B, D)
                    c = centres[L].astype(np.float32)
                    P = bases[L, :, :args.k_lang].astype(np.float32)              # (D, K)
                    proj = (h - c[None]) @ P                                      # (B, K)
                    norm = np.sqrt((proj ** 2).sum(axis=1))                       # (B,)
                    total = np.sqrt(((h - c[None]) ** 2).sum(axis=1))
                    frac  = norm / np.clip(total, 1e-6, None)
                    for j in range(h.shape[0]):
                        strengths[L].append({"norm": float(norm[j]),
                                             "frac": float(frac[j])})
                del out, enc
                torch.cuda.empty_cache()
        strength_per_lang[lg] = strengths
        correct_per_lang[lg]  = [int(r["correct"]) for r in rows]
        print(f"[{lg}] N={len(rows)}, mean correct = {np.mean(correct_per_lang[lg]):.2f}")

    # Correlations
    corr = {}
    for lg in strength_per_lang:
        corr[lg] = {}
        y = np.array(correct_per_lang[lg], dtype=float)
        if y.std() == 0:
            corr[lg]["_note"] = "no variance in correctness"
            continue
        for L, arr in strength_per_lang[lg].items():
            xn = np.array([a["norm"] for a in arr])
            xf = np.array([a["frac"] for a in arr])
            r_n, p_n = pearsonr(xn, y)
            r_f, p_f = pearsonr(xf, y)
            corr[lg][str(L)] = {"n": len(y), "r_norm": float(r_n), "p_norm": float(p_n),
                                "r_frac": float(r_f), "p_frac": float(p_f)}

    # Pooled across langs (residualize by mean per-lang correctness to remove lang confound)
    pooled = {L: {"x_norm": [], "x_frac": [], "y_res": []} for L in layers}
    for lg in strength_per_lang:
        y = np.array(correct_per_lang[lg], dtype=float)
        y_mean = y.mean()
        for L, arr in strength_per_lang[lg].items():
            xn = np.array([a["norm"] for a in arr])
            xf = np.array([a["frac"] for a in arr])
            xn_res = xn - xn.mean()
            xf_res = xf - xf.mean()
            y_res  = y - y_mean
            pooled[L]["x_norm"].append(xn_res)
            pooled[L]["x_frac"].append(xf_res)
            pooled[L]["y_res"].append(y_res)
    pooled_corr = {}
    for L, d in pooled.items():
        xn = np.concatenate(d["x_norm"]); xf = np.concatenate(d["x_frac"]); y = np.concatenate(d["y_res"])
        if y.std() == 0 or xn.std() == 0:
            pooled_corr[str(L)] = {"note": "no variance"}; continue
        r_n, p_n = pearsonr(xn, y); r_f, p_f = pearsonr(xf, y)
        pooled_corr[str(L)] = {"n": len(y), "r_norm": float(r_n), "p_norm": float(p_n),
                               "r_frac": float(r_f), "p_frac": float(p_f)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_lang": corr, "pooled_within_lang": pooled_corr,
                   "layers": layers, "k_lang": args.k_lang}, f, indent=2)
    print("wrote", args.out)
    print("\n== pooled within-lang correlations (residualized) ==")
    for L, d in pooled_corr.items():
        print(f"L={L}: r_norm={d.get('r_norm', 'NA')} p={d.get('p_norm', 'NA')}  "
              f"r_frac={d.get('r_frac', 'NA')} p={d.get('p_frac', 'NA')}")


if __name__ == "__main__":
    main()
