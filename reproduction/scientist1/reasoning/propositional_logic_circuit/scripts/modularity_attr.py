"""Modularity via per-corruption-axis attribution patching (claim 2).

For each corruption axis (rule_flip, fact_flip, query_flip) we compute
the per-head + per-MLP attribution to logit_diff. If the circuit is
MODULAR we expect different components to matter for each axis; if
ENTANGLED we expect the same components everywhere.
"""
import argparse
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


def attribution_for_axis(model, tok, ds, corrupt_key, answer_key, device, max_batch=32, verbose=True):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_ans = torch.tensor([token_id_for(tok, r["clean_answer"]) for r in ds]).to(device)
    corr_ans = torch.tensor([token_id_for(tok, r[answer_key]) for r in ds]).to(device)

    # baseline
    clean_ld_total = corr_ld_total = 0.0
    for i in range(0, len(ds), max_batch):
        batch = ds[i:i + max_batch]
        cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
        crids = [torch.tensor(tok.encode(r[corrupt_key])) for r in batch]
        cin = pad_stack(cids, tok.pad_token_id, device)
        crin = pad_stack(crids, tok.pad_token_id, device)
        with torch.no_grad():
            clean_ld_total += logit_diff(
                model(cin)[:, -1, :], clean_ans[i:i+max_batch], corr_ans[i:i+max_batch]
            ).sum().item()
            corr_ld_total += logit_diff(
                model(crin)[:, -1, :], clean_ans[i:i+max_batch], corr_ans[i:i+max_batch]
            ).sum().item()
    clean_ld = clean_ld_total / len(ds)
    corr_ld = corr_ld_total / len(ds)
    denom = clean_ld - corr_ld
    if verbose:
        print(f"[{corrupt_key}] clean_ld={clean_ld:.3f} corr_ld={corr_ld:.3f} denom={denom:.3f}")

    head_attr = np.zeros((n_layers, n_heads), dtype=np.float32)
    mlp_attr = np.zeros(n_layers, dtype=np.float32)

    def name_filter(n):
        return n.endswith("hook_z") or n.endswith("hook_mlp_out")

    for bi in range(0, len(ds), max_batch):
        batch = ds[bi:bi + max_batch]
        cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
        crids = [torch.tensor(tok.encode(r[corrupt_key])) for r in batch]
        cin = pad_stack(cids, tok.pad_token_id, device)
        crin = pad_stack(crids, tok.pad_token_id, device)
        # Align sequence lengths for clean and corrupt (rare mismatch for
        # query_flip when prop name differs in # of tokens).
        if cin.shape[1] != crin.shape[1]:
            S = min(cin.shape[1], crin.shape[1])
            cin = cin[:, -S:]
            crin = crin[:, -S:]

        ca = clean_ans[bi:bi + max_batch]
        cra = corr_ans[bi:bi + max_batch]

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(cin, names_filter=name_filter)

        model.reset_hooks()
        corr_acts = {}
        corr_grads = {}

        def fwd_hook_factory(nm):
            def fh(v, hook, nm=nm):
                v.requires_grad_(True)
                v.retain_grad()
                corr_acts[nm] = v
                def bh(g, nm=nm):
                    corr_grads[nm] = g.detach()
                v.register_hook(bh)
                return v
            return fh

        for h in list(model.hook_dict.keys()):
            if name_filter(h):
                model.add_hook(h, fwd_hook_factory(h))

        corr_logits = model(crin)[:, -1, :]
        metric = (
            corr_logits.gather(1, ca[:, None]).squeeze(1)
            - corr_logits.gather(1, cra[:, None]).squeeze(1)
        ).sum()
        metric.backward()
        model.reset_hooks()

        for L in range(n_layers):
            zname = f"blocks.{L}.attn.hook_z"
            clean_z = clean_cache[zname]
            corr_z = corr_acts[zname].detach()
            grad_z = corr_grads[zname]
            per_head = ((clean_z - corr_z) * grad_z).float().sum(dim=(0, 1, 3))
            head_attr[L] += per_head.cpu().numpy()

            mname = f"blocks.{L}.hook_mlp_out"
            clean_m = clean_cache[mname]
            corr_m = corr_acts[mname].detach()
            grad_m = corr_grads[mname]
            mlp_attr[L] += (clean_m - corr_m).mul(grad_m).float().sum().cpu().item()

        del clean_cache, corr_acts, corr_grads
        torch.cuda.empty_cache()
        if verbose:
            print(f"  [{corrupt_key}] batch {bi // max_batch + 1}/{(len(ds) + max_batch - 1) // max_batch} done")

    head_attr /= (len(ds) * denom)
    mlp_attr /= (len(ds) * denom)
    return head_attr, mlp_attr, {"clean_ld": clean_ld, "corr_ld": corr_ld, "denom": denom}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--data", default="data/logic_ds.jsonl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="results/modularity_attr_mistral.npz")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

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
    for p in model.parameters():
        p.requires_grad_(False)

    ds = load_dataset(args.data)[: args.n]

    results = {}
    for corrupt_key, answer_key in [
        ("corrupt", "corrupt_answer"),
        ("fact_flip", "fact_flip_answer"),
        ("query_flip", "query_flip_answer"),
    ]:
        t0 = time.time()
        h, m, stats = attribution_for_axis(
            model, tok, ds, corrupt_key, answer_key, args.device, args.batch
        )
        print(f"[{corrupt_key}] took {time.time()-t0:.1f}s")
        results[f"{corrupt_key}_heads"] = h
        results[f"{corrupt_key}_mlps"] = m
        results[f"{corrupt_key}_clean_ld"] = stats["clean_ld"]
        results[f"{corrupt_key}_corr_ld"] = stats["corr_ld"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **results)
    print("saved", args.out)

    for key in ("corrupt", "fact_flip", "query_flip"):
        h = results[f"{key}_heads"]
        top = np.argsort(-np.abs(h).ravel())[:10]
        print(f"[{key}] top heads:", [(int(a), int(b)) for a, b in [np.unravel_index(t, h.shape) for t in top]])

    # Correlation table
    def corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    print("\nHead-effect correlations (higher = more entangled):")
    print(f"  rule_flip vs fact_flip : {corr(results['corrupt_heads'], results['fact_flip_heads']):+.3f}")
    print(f"  rule_flip vs query_flip: {corr(results['corrupt_heads'], results['query_flip_heads']):+.3f}")
    print(f"  fact_flip vs query_flip: {corr(results['fact_flip_heads'], results['query_flip_heads']):+.3f}")


if __name__ == "__main__":
    main()
