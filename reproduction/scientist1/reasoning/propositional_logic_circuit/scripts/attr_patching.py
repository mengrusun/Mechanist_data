"""Attribution (gradient-based) patching for fast circuit discovery.

For each attention head and each MLP output, attribution patching
approximates the effect of activation patching via first-order Taylor:

    delta_metric ≈ (clean_act - corrupt_act) · grad(metric wrt act)

Cost: 1 forward on CLEAN (cache activations), 1 forward+backward on
CORRUPT (cache gradients). Instead of 1024 forwards per batch, we do
~3. This lets us process much more data very fast.

Reference: Nanda et al., "Attribution Patching" blog post.
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


def get_activation_grads(model, prompt_ids, target_ids, other_ids, name_filter, model_dtype):
    """Run forward + backward on prompt_ids, return (activations, gradients)
    for hook names matching name_filter, evaluated on
    metric = mean(logit(target) - logit(other)) at the last position."""
    cache = {}
    grad_cache = {}
    hook_handles = []

    def make_fwd_hook(name):
        def hook(module_out, hook):
            module_out.retain_grad()
            cache[name] = module_out
            def bwd(g, name=name):
                grad_cache[name] = g.detach()
            module_out.register_hook(bwd)
            return module_out
        return hook

    # register hooks
    hooks = []
    for hook_name in list(model.hook_dict.keys()):
        if name_filter(hook_name):
            hooks.append((hook_name, make_fwd_hook(hook_name)))

    # forward with hooks
    model.reset_hooks()
    for name, hook_fn in hooks:
        model.add_hook(name, hook_fn)

    logits = model(prompt_ids)[:, -1, :]
    metric = (
        logits.gather(1, target_ids[:, None]).squeeze(1)
        - logits.gather(1, other_ids[:, None]).squeeze(1)
    ).mean()
    metric.backward()
    model.reset_hooks()

    # detach values
    acts = {n: v.detach() for n, v in cache.items()}
    grads = grad_cache
    return acts, grads, metric.detach().item()


def run_attribution(model, tok, ds, device, max_batch=16, verbose=True):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    model_dtype = model.cfg.dtype

    clean_ans = torch.tensor([token_id_for(tok, r["clean_answer"]) for r in ds]).to(device)
    corr_ans = torch.tensor([token_id_for(tok, r["corrupt_answer"]) for r in ds]).to(device)

    # baseline metric: metric on clean and corrupt runs
    clean_ld_total = corr_ld_total = 0.0
    for i in range(0, len(ds), max_batch):
        batch = ds[i:i + max_batch]
        cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
        crids = [torch.tensor(tok.encode(r["corrupt"])) for r in batch]
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
        print(f"clean_ld={clean_ld:.3f} corr_ld={corr_ld:.3f} denom={denom:.3f}")

    # attribution
    head_attr = np.zeros((n_layers, n_heads), dtype=np.float32)
    mlp_attr = np.zeros(n_layers, dtype=np.float32)

    def name_filter(n):
        return n.endswith("hook_z") or n.endswith("hook_mlp_out")

    for bi in range(0, len(ds), max_batch):
        batch = ds[bi:bi + max_batch]
        cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
        crids = [torch.tensor(tok.encode(r["corrupt"])) for r in batch]
        cin = pad_stack(cids, tok.pad_token_id, device)
        crin = pad_stack(crids, tok.pad_token_id, device)
        ca = clean_ans[bi:bi + max_batch]
        cra = corr_ans[bi:bi + max_batch]

        # Get clean activations (no grad needed)
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(cin, names_filter=name_filter)

        # Get corrupt activations + gradients wrt metric
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

        hook_names = [h for h in model.hook_dict if name_filter(h)]
        for h in hook_names:
            model.add_hook(h, fwd_hook_factory(h))

        # We want metric = logit_diff(patched) = logit(clean_ans) - logit(corr_ans)
        # If we PATCH corrupt with clean acts, metric should INCREASE.
        # So we measure d(metric)/d(activation_corrupt) and then
        # attribution = (clean_act - corrupt_act) . grad
        corr_logits = model(crin)[:, -1, :]
        metric = (
            corr_logits.gather(1, ca[:, None]).squeeze(1)
            - corr_logits.gather(1, cra[:, None]).squeeze(1)
        ).sum()   # sum across batch so per-example grads are preserved
        metric.backward()
        model.reset_hooks()

        # compute attributions
        # heads: shape (B, S, H, D) for z; sum over B, S, D -> per-head scalar
        for L in range(n_layers):
            zname = f"blocks.{L}.attn.hook_z"
            clean_z = clean_cache[zname]              # (B, S, H, D)
            corr_z = corr_acts[zname].detach()
            grad_z = corr_grads[zname]                # (B, S, H, D)
            delta = clean_z - corr_z                  # (B, S, H, D)
            per_head = (delta * grad_z).float().sum(dim=(0, 1, 3))  # (H,)
            head_attr[L] += per_head.cpu().numpy()

            mname = f"blocks.{L}.hook_mlp_out"
            clean_m = clean_cache[mname]
            corr_m = corr_acts[mname].detach()
            grad_m = corr_grads[mname]
            mlp_attr[L] += (clean_m - corr_m).mul(grad_m).float().sum().cpu().item()

        del clean_cache, corr_acts, corr_grads
        torch.cuda.empty_cache()
        if verbose:
            print(f"  batch {bi // max_batch + 1}/{(len(ds) + max_batch - 1) // max_batch} done")

    # normalize
    # Attribution above is a first-order estimate of delta_metric_sum. To get
    # normalized recovery per example: divide by (n_examples * denom).
    head_attr /= (len(ds) * denom)
    mlp_attr /= (len(ds) * denom)
    return head_attr, mlp_attr, {"clean_ld": clean_ld, "corr_ld": corr_ld, "denom": denom}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--data", default="data/logic_ds.jsonl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/attr_patching_mistral.npz")
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

    # need grads on the model
    for p in model.parameters():
        p.requires_grad_(False)

    ds = load_dataset(args.data)[: args.n]
    print(f"loaded {len(ds)} examples")

    t0 = time.time()
    head_eff, mlp_eff, stats = run_attribution(model, tok, ds, args.device, max_batch=args.batch)
    dt = time.time() - t0
    print(f"attribution took {dt:.1f}s")

    print("top-15 heads by |attr|:")
    flat = np.argsort(-np.abs(head_eff).ravel())[:15]
    for idx in flat:
        L, H = np.unravel_index(idx, head_eff.shape)
        print(f"  L{L} H{H}: {head_eff[L, H]:+.3f}")
    print("top-10 MLPs by |attr|:")
    for L in np.argsort(-np.abs(mlp_eff))[:10]:
        print(f"  L{L}: {mlp_eff[L]:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, head_effects=head_eff, mlp_effects=mlp_eff, **stats)
    print("saved", args.out)


if __name__ == "__main__":
    main()
