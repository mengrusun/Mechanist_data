"""Necessity & sufficiency of the top-K components (claim 3).

NECESSITY (knockout): ablate the top-K attention heads + top-M MLPs in
the CLEAN run and observe how much the clean logit_diff drops.
    Ablation = mean-ablation: replace the head's z (resp. MLP output)
    with its MEAN over a distinct batch of corrupted prompts (resample
    ablation).

SUFFICIENCY (patch-in): starting from the CORRUPTED run, patch ONLY
these top-K components from the clean run and see how much of the
clean logit_diff is restored.

For scale we also compare against a control set of K RANDOM heads to
show the top-K set outperforms a random sparse set of the same size.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from patching_utils import (
    load_hooked_mistral, load_hooked_gemma2_9b, load_hooked_gemma2_2b,
    load_dataset, token_id_for, logit_diff,
)


def pad_stack(seqs, pad_id, device):
    m = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, m - len(s):] = s
    return padded.to(device)


def compute_ld(model, tok, ds, key, clean_ans, corr_ans, device, max_batch=32, extra_hooks=None):
    """Compute mean logit_diff on prompts in ds[i][key].
    extra_hooks: list of (name, hook_fn) applied throughout."""
    total = 0.0
    n = 0
    for i in range(0, len(ds), max_batch):
        batch = ds[i:i + max_batch]
        ids = [torch.tensor(tok.encode(r[key])) for r in batch]
        cin = pad_stack(ids, tok.pad_token_id, device)
        with torch.no_grad():
            if extra_hooks:
                logits = model.run_with_hooks(cin, fwd_hooks=extra_hooks)[:, -1, :]
            else:
                logits = model(cin)[:, -1, :]
        ld = logit_diff(logits, clean_ans[i:i + max_batch], corr_ans[i:i + max_batch])
        total += ld.sum().item()
        n += ld.numel()
    return total / n


def sufficiency(model, tok, ds, top_heads, top_mlps, clean_ans, corr_ans, device, max_batch=16):
    """Patch ONLY the given components from clean into corrupt run."""
    lds = []
    for i in range(0, len(ds), max_batch):
        batch = ds[i:i + max_batch]
        cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
        crids = [torch.tensor(tok.encode(r["corrupt"])) for r in batch]
        cin = pad_stack(cids, tok.pad_token_id, device)
        crin = pad_stack(crids, tok.pad_token_id, device)

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(
                cin, names_filter=lambda n: n.endswith("hook_z") or n.endswith("hook_mlp_out"),
            )

        # Group by layer for efficiency
        head_by_layer = {}
        for _, L, H in top_heads:
            head_by_layer.setdefault(L, []).append(H)

        hooks = []
        for L, heads in head_by_layer.items():
            clean_z = clean_cache[f"blocks.{L}.attn.hook_z"]
            heads_t = torch.tensor(heads, device=device)
            def hook_fn(z, hook, cz=clean_z, hh=heads_t):
                z[:, :, hh, :] = cz[:, :, hh, :]
                return z
            hooks.append((f"blocks.{L}.attn.hook_z", hook_fn))

        for _, L in top_mlps:
            clean_mlp = clean_cache[f"blocks.{L}.hook_mlp_out"]
            def mhook(mo, hook, cm=clean_mlp):
                return cm
            hooks.append((f"blocks.{L}.hook_mlp_out", mhook))

        with torch.no_grad():
            patched = model.run_with_hooks(crin, fwd_hooks=hooks)[:, -1, :]

        ld = logit_diff(patched, clean_ans[i:i + max_batch], corr_ans[i:i + max_batch])
        lds.extend(ld.cpu().tolist())
        del clean_cache
        torch.cuda.empty_cache()
    return float(np.mean(lds))


def necessity(model, tok, ds, top_heads, top_mlps, clean_ans, corr_ans, device, max_batch=16, resample_batch=None):
    """Ablate the given components in the CLEAN run and measure ld."""
    lds = []
    for i in range(0, len(ds), max_batch):
        batch = ds[i:i + max_batch]
        cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
        cin = pad_stack(cids, tok.pad_token_id, device)

        # Use corrupt-run activations for resample ablation
        crids = [torch.tensor(tok.encode(r["corrupt"])) for r in batch]
        crin = pad_stack(crids, tok.pad_token_id, device)
        with torch.no_grad():
            _, corr_cache = model.run_with_cache(
                crin, names_filter=lambda n: n.endswith("hook_z") or n.endswith("hook_mlp_out"),
            )

        head_by_layer = {}
        for _, L, H in top_heads:
            head_by_layer.setdefault(L, []).append(H)

        hooks = []
        for L, heads in head_by_layer.items():
            corr_z = corr_cache[f"blocks.{L}.attn.hook_z"]
            heads_t = torch.tensor(heads, device=device)
            def hook_fn(z, hook, cz=corr_z, hh=heads_t):
                z[:, :, hh, :] = cz[:, :, hh, :]
                return z
            hooks.append((f"blocks.{L}.attn.hook_z", hook_fn))

        for _, L in top_mlps:
            corr_mlp = corr_cache[f"blocks.{L}.hook_mlp_out"]
            def mhook(mo, hook, cm=corr_mlp):
                return cm
            hooks.append((f"blocks.{L}.hook_mlp_out", mhook))

        with torch.no_grad():
            ablated = model.run_with_hooks(cin, fwd_hooks=hooks)[:, -1, :]
        ld = logit_diff(ablated, clean_ans[i:i + max_batch], corr_ans[i:i + max_batch])
        lds.extend(ld.cpu().tolist())
        del corr_cache
        torch.cuda.empty_cache()
    return float(np.mean(lds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--data", default="data/logic_ds.jsonl")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--head-npz", default="results/head_patching_mistral.npz")
    ap.add_argument("--out", default="results/nec_suff_mistral.json")
    ap.add_argument("--top-k-values", type=str, default="4,8,16,24,32,48")
    ap.add_argument("--top-m-mlps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    d = np.load(args.head_npz)
    head_eff = d["head_effects"]
    mlp_eff = d["mlp_effects"]
    n_layers, n_heads = head_eff.shape

    top_h_idx = np.argsort(-np.abs(head_eff).ravel())
    all_top_heads = []
    for idx in top_h_idx:
        L, H = np.unravel_index(idx, head_eff.shape)
        all_top_heads.append(("head", int(L), int(H)))
    top_m_idx = np.argsort(-np.abs(mlp_eff))[: args.top_m_mlps]
    top_mlps = [("mlp", int(L)) for L in top_m_idx]

    if args.model == "mistral":
        model, tok = load_hooked_mistral(
            "/data/zhenqian/models/Mistral-7B-v0.1", device=args.device
        )
    elif args.model == "gemma-2-9b":
        model, tok = load_hooked_gemma2_9b(
            "/data/zhenqian/models/gemma-2-9b", device=args.device
        )
    elif args.model == "gemma-2-2b":
        model, tok = load_hooked_gemma2_2b(
            "/data/zhenqian/models/gemma-2-2b", device=args.device
        )
    else:
        raise ValueError(args.model)

    ds = load_dataset(args.data)[: args.n]
    clean_ans = torch.tensor([token_id_for(tok, r["clean_answer"]) for r in ds]).to(args.device)
    corr_ans = torch.tensor([token_id_for(tok, r["corrupt_answer"]) for r in ds]).to(args.device)

    print("baseline...")
    t0 = time.time()
    clean_ld = compute_ld(model, tok, ds, "clean", clean_ans, corr_ans, args.device, args.batch)
    corr_ld = compute_ld(model, tok, ds, "corrupt", clean_ans, corr_ans, args.device, args.batch)
    print(f"  clean={clean_ld:.3f} corrupt={corr_ld:.3f} took {time.time()-t0:.1f}s")

    K_values = [int(x) for x in args.top_k_values.split(",")]
    rng = np.random.RandomState(args.seed)
    results = {
        "clean_ld": clean_ld, "corrupt_ld": corr_ld,
        "K": [], "top_suff": [], "top_nec": [],
        "rand_suff_mean": [], "rand_nec_mean": [],
        "rand_suff_std": [], "rand_nec_std": [],
    }

    for K in K_values:
        top_k = all_top_heads[:K]
        print(f"K={K}:")
        t0 = time.time()
        s = sufficiency(model, tok, ds, top_k, top_mlps, clean_ans, corr_ans, args.device, args.batch)
        n = necessity(model, tok, ds, top_k, top_mlps, clean_ans, corr_ans, args.device, args.batch)
        print(f"  top: suff={s:.3f} nec={n:.3f} took {time.time()-t0:.1f}s")

        # random control (3 seeds)
        rand_s = []
        rand_n = []
        for seed in range(3):
            rs = np.random.RandomState(seed + 1)
            idxs = rs.choice(n_layers * n_heads, size=K, replace=False)
            rand_heads = []
            for idx in idxs:
                L, H = np.unravel_index(idx, head_eff.shape)
                rand_heads.append(("head", int(L), int(H)))
            rs_ = sufficiency(model, tok, ds, rand_heads, top_mlps, clean_ans, corr_ans, args.device, args.batch)
            rn_ = necessity(model, tok, ds, rand_heads, top_mlps, clean_ans, corr_ans, args.device, args.batch)
            rand_s.append(rs_)
            rand_n.append(rn_)
        print(f"  random: suff={np.mean(rand_s):.3f}±{np.std(rand_s):.3f} nec={np.mean(rand_n):.3f}±{np.std(rand_n):.3f}")

        results["K"].append(K)
        results["top_suff"].append(s)
        results["top_nec"].append(n)
        results["rand_suff_mean"].append(float(np.mean(rand_s)))
        results["rand_suff_std"].append(float(np.std(rand_s)))
        results["rand_nec_mean"].append(float(np.mean(rand_n)))
        results["rand_nec_std"].append(float(np.std(rand_n)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
