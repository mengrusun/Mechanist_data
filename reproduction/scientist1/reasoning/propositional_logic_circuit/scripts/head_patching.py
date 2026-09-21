"""Activation patching over attention heads and MLPs.

For each attention head (layer, head), we patch head-output `z` from the
CLEAN run into the CORRUPTED run (across ALL sequence positions) and
measure the resulting change in the final-token logit_diff.

The metric:
    normalized_effect =
        (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)

  0   -> no effect (patched behaves like corrupted)
  1   -> full recovery (patched behaves like clean)

We also patch each MLP layer's output the same way.
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
    load_hooked_mistral,
    load_hooked_gemma2_9b,
    load_dataset,
    token_id_for,
    logit_diff,
)


def pad_stack(seqs, pad_id, device):
    m = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        # left-pad so the last token is at position -1 for all rows
        padded[i, m - len(s):] = s
    return padded.to(device)


def compute_baseline_ld(model, tok, ds, device, max_batch=32):
    """Compute clean_ld, corrupt_ld averaged over dataset (also return per-example)."""
    clean_lds, corrupt_lds = [], []
    clean_ans = torch.tensor([token_id_for(tok, r["clean_answer"]) for r in ds]).to(device)
    corr_ans = torch.tensor([token_id_for(tok, r["corrupt_answer"]) for r in ds]).to(device)

    for i in range(0, len(ds), max_batch):
        batch = ds[i:i + max_batch]
        cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
        crids = [torch.tensor(tok.encode(r["corrupt"])) for r in batch]
        cin = pad_stack(cids, tok.pad_token_id, device)
        crin = pad_stack(crids, tok.pad_token_id, device)
        with torch.no_grad():
            clean_logits = model(cin)[:, -1, :]
            corr_logits = model(crin)[:, -1, :]
        clean_lds.append(
            logit_diff(clean_logits, clean_ans[i:i + max_batch], corr_ans[i:i + max_batch])
        )
        corrupt_lds.append(
            logit_diff(corr_logits, clean_ans[i:i + max_batch], corr_ans[i:i + max_batch])
        )
    return torch.cat(clean_lds), torch.cat(corrupt_lds), clean_ans, corr_ans


def run_head_patching(
    model, tok, ds, device, patch_positions="all", max_batch=32, verbose=True
):
    """For every (layer, head), patch head-output z from clean into corrupt.
    Returns array of shape (n_layers, n_heads) of normalized effect."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_lds, corrupt_lds, clean_ans, corr_ans = compute_baseline_ld(
        model, tok, ds, device, max_batch=max_batch
    )
    denom = (clean_lds.mean() - corrupt_lds.mean()).item()
    if verbose:
        print(f"baseline clean_ld={clean_lds.mean().item():.3f}, "
              f"corrupt_ld={corrupt_lds.mean().item():.3f}, denom={denom:.3f}")

    # We'll run patching in mini-batches over the dataset to control memory.
    head_effects = np.zeros((n_layers, n_heads), dtype=np.float32)
    mlp_effects = np.zeros(n_layers, dtype=np.float32)

    for bi in range(0, len(ds), max_batch):
        batch = ds[bi:bi + max_batch]
        cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
        crids = [torch.tensor(tok.encode(r["corrupt"])) for r in batch]
        cin = pad_stack(cids, tok.pad_token_id, device)
        crin = pad_stack(crids, tok.pad_token_id, device)
        ca = clean_ans[bi:bi + max_batch]
        cra = corr_ans[bi:bi + max_batch]

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(
                cin,
                names_filter=lambda n: n.endswith("hook_z") or n.endswith("hook_mlp_out"),
            )

        # Attention-head patching
        for layer in range(n_layers):
            clean_z = clean_cache[f"blocks.{layer}.attn.hook_z"]  # (B, S, H, D)
            for head in range(n_heads):
                def hook_fn(z, hook, layer=layer, head=head):
                    # z shape: (B, S, H, D). Replace only the given head.
                    z[:, :, head, :] = clean_z[:, :, head, :]
                    return z
                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        crin,
                        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)],
                    )[:, -1, :]
                ld_p = logit_diff(patched_logits, ca, cra).mean().item()
                head_effects[layer, head] += (ld_p - corrupt_lds[bi:bi + max_batch].mean().item()) * len(batch)

        # MLP patching
        for layer in range(n_layers):
            clean_mlp = clean_cache[f"blocks.{layer}.hook_mlp_out"]  # (B, S, D)
            def mlp_hook(mlp_out, hook, clean=clean_mlp):
                return clean
            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    crin,
                    fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", mlp_hook)],
                )[:, -1, :]
            ld_p = logit_diff(patched_logits, ca, cra).mean().item()
            mlp_effects[layer] += (ld_p - corrupt_lds[bi:bi + max_batch].mean().item()) * len(batch)

        del clean_cache
        torch.cuda.empty_cache()
        if verbose:
            print(f"  processed batch {bi // max_batch + 1}/{(len(ds) + max_batch - 1) // max_batch}")

    # normalise
    head_effects /= (len(ds) * denom)
    mlp_effects /= (len(ds) * denom)
    return head_effects, mlp_effects, {
        "clean_ld_mean": float(clean_lds.mean().item()),
        "corrupt_ld_mean": float(corrupt_lds.mean().item()),
        "denom": denom,
        "n_examples": len(ds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--data", default="data/logic_ds.jsonl")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="results/head_patching_mistral.npz")
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
    print(f"loaded {len(ds)} examples")

    t0 = time.time()
    head_eff, mlp_eff, stats = run_head_patching(
        model, tok, ds, args.device, max_batch=args.batch
    )
    dt = time.time() - t0
    print(f"patching took {dt:.1f}s")
    print("top-10 heads by |effect|:")
    flat = np.argsort(-np.abs(head_eff).ravel())[:10]
    for idx in flat:
        L, H = np.unravel_index(idx, head_eff.shape)
        print(f"  L{L} H{H}: {head_eff[L, H]:+.3f}")
    print("top-10 MLPs by |effect|:")
    for L in np.argsort(-np.abs(mlp_eff))[:10]:
        print(f"  L{L}: {mlp_eff[L]:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, head_effects=head_eff, mlp_effects=mlp_eff, **stats)
    print("saved", args.out)


if __name__ == "__main__":
    main()
