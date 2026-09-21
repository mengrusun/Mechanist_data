"""Modularity decomposition via different corruption axes (claim 2).

For each of three corruption variants of the SAME clean prompt --
    - rule_flip  (default 'corrupt' key: flip rule consequent polarity)
    - fact_flip  (flip antecedent fact polarity)
    - query_flip (point query at distractor proposition)
-- we run per-head activation patching (clean -> corrupted variant).

If the circuit is MODULAR, different heads should recover different
corruption types (each head implements a distinct sub-computation).
If it is ENTANGLED, the same heads should recover all corruption types.
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
    load_hooked_mistral, load_hooked_gemma2_9b,
    load_dataset, token_id_for, logit_diff,
)


def pad_stack(seqs, pad_id, device):
    m = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, m - len(s):] = s
    return padded.to(device)


def patch_by_corruption(model, tok, ds, corrupt_key: str, answer_key: str,
                        device, max_batch=16, verbose=True):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_ans = torch.tensor([token_id_for(tok, r["clean_answer"]) for r in ds]).to(device)
    corr_ans = torch.tensor([token_id_for(tok, r[answer_key]) for r in ds]).to(device)

    # baseline
    clean_ld_total = corr_ld_total = 0.0
    with torch.no_grad():
        for i in range(0, len(ds), max_batch):
            batch = ds[i:i + max_batch]
            cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
            crids = [torch.tensor(tok.encode(r[corrupt_key])) for r in batch]
            cin = pad_stack(cids, tok.pad_token_id, device)
            crin = pad_stack(crids, tok.pad_token_id, device)
            clean_logits = model(cin)[:, -1, :]
            corr_logits = model(crin)[:, -1, :]
            clean_ld_total += logit_diff(clean_logits, clean_ans[i:i+max_batch], corr_ans[i:i+max_batch]).sum().item()
            corr_ld_total += logit_diff(corr_logits, clean_ans[i:i+max_batch], corr_ans[i:i+max_batch]).sum().item()
    clean_ld = clean_ld_total / len(ds)
    corr_ld = corr_ld_total / len(ds)
    denom = clean_ld - corr_ld
    if verbose:
        print(f"[{corrupt_key}] clean={clean_ld:.3f} corrupt={corr_ld:.3f} denom={denom:.3f}")

    head_eff = np.zeros((n_layers, n_heads), dtype=np.float32)
    mlp_eff = np.zeros(n_layers, dtype=np.float32)

    for bi in range(0, len(ds), max_batch):
        batch = ds[bi:bi + max_batch]
        cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
        crids = [torch.tensor(tok.encode(r[corrupt_key])) for r in batch]
        cin = pad_stack(cids, tok.pad_token_id, device)
        crin = pad_stack(crids, tok.pad_token_id, device)
        # Note: for some corruption axes (query_flip) clean and corrupt may
        # have different lengths (query proposition might be a different-length
        # word). We pad them separately; patches take place in per-batch space.
        # For head patching this only matters if the SHAPES differ; since we
        # pad both to their own max, the shapes may differ between cache and
        # patched forward. Check and skip if mismatch.
        if cin.shape != crin.shape:
            # Only take the last min(S_c, S_cr) tokens to align on the right.
            S = min(cin.shape[1], crin.shape[1])
            cin = cin[:, -S:]
            crin = crin[:, -S:]

        ca = clean_ans[bi:bi + max_batch]
        cra = corr_ans[bi:bi + max_batch]

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(
                cin, names_filter=lambda n: n.endswith("hook_z") or n.endswith("hook_mlp_out"),
            )

        for layer in range(n_layers):
            clean_z = clean_cache[f"blocks.{layer}.attn.hook_z"]
            for head in range(n_heads):
                def hook_fn(z, hook, cz=clean_z, h=head):
                    z[:, :, h, :] = cz[:, :, h, :]
                    return z
                with torch.no_grad():
                    patched = model.run_with_hooks(
                        crin, fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)]
                    )[:, -1, :]
                ld_p = logit_diff(patched, ca, cra).mean().item()
                head_eff[layer, head] += (ld_p - corr_ld) * len(batch)

        for layer in range(n_layers):
            clean_mlp = clean_cache[f"blocks.{layer}.hook_mlp_out"]
            def mhook(mo, hook, cm=clean_mlp):
                return cm
            with torch.no_grad():
                patched = model.run_with_hooks(
                    crin, fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", mhook)]
                )[:, -1, :]
            ld_p = logit_diff(patched, ca, cra).mean().item()
            mlp_eff[layer] += (ld_p - corr_ld) * len(batch)

        del clean_cache
        torch.cuda.empty_cache()
        if verbose:
            print(f"  [{corrupt_key}] batch {bi // max_batch + 1}/{(len(ds) + max_batch - 1) // max_batch} done")

    head_eff /= (len(ds) * denom)
    mlp_eff /= (len(ds) * denom)
    return head_eff, mlp_eff, {"clean_ld": clean_ld, "corr_ld": corr_ld, "denom": denom}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--data", default="data/logic_ds.jsonl")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/modularity_mistral.npz")
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
    else:
        raise ValueError(args.model)

    ds = load_dataset(args.data)[: args.n]

    results = {}
    for corrupt_key, answer_key in [
        ("corrupt", "corrupt_answer"),         # rule polarity flip
        ("fact_flip", "fact_flip_answer"),     # denied antecedent
        ("query_flip", "query_flip_answer"),   # different query prop
    ]:
        t0 = time.time()
        h, m, stats = patch_by_corruption(model, tok, ds, corrupt_key, answer_key, args.device, args.batch)
        print(f"[{corrupt_key}] took {time.time()-t0:.1f}s")
        results[f"{corrupt_key}_heads"] = h
        results[f"{corrupt_key}_mlps"] = m
        results[f"{corrupt_key}_clean_ld"] = stats["clean_ld"]
        results[f"{corrupt_key}_corr_ld"] = stats["corr_ld"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **results)
    print("saved", args.out)

    # Quick summary
    for key in ("corrupt", "fact_flip", "query_flip"):
        h = results[f"{key}_heads"]
        top = np.argsort(-np.abs(h).ravel())[:8]
        print(f"[{key}] top heads:", [np.unravel_index(t, h.shape) for t in top])


if __name__ == "__main__":
    main()
