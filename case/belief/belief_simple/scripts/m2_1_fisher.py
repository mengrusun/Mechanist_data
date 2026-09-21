"""M2.1: Empirical Fisher signal for a (model, signal) cell.

Fisher signal:
    F_i = (1/N) Σ_x ( ∂/∂θ_i  Σ_t log p_θ(y⁺_t | x, y⁺_<t) )²

Signals:
  - F_attributed: on `follow_belief.jsonl`, filter person ∈ {james, mary} → n=454.
  - F_personal:   on `believe_truth.jsonl`, filter person ∈ {james, mary} → n=454.
  - F_knowledge:  on `reality.jsonl`, no filter → n=227.

Gradients accumulated in fp32. Model forward in fp32 as well (empirical Fisher requires
faithful gradients; fp16 gradients can underflow).

Also computes and saves per-attention-head aggregated Fisher:
  head_score(l, h) = Σ_{θ_i ∈ {W_Q^h, W_K^h, W_V^h, W_O^h}}  F_i
where head parameter membership follows the GPTNeoX fused-QKV convention laid out in
belief_utils.gpt_neox_head_param_slices / head_param_indices_flat.

When `--jackknife` is passed, we ALSO compute two "half-Fisher" tensors on 2 stratified halves
(seed 0), and save them next to the main output. M2.2 consumes them for the ρ report.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from belief_utils import (
    load_model_and_tokenizer,
    load_task,
    model_arch_info,
    head_param_names,
    save_json,
    set_seed,
)


SIGNAL_SPEC = {
    "F_attributed": {"task": "attributed_belief", "person_filter": ["james", "mary"]},
    "F_personal":   {"task": "personal_belief",   "person_filter": ["james", "mary"]},
    "F_knowledge":  {"task": "world_knowledge",   "person_filter": None},
}


def compute_fisher_for_examples(net, tok, examples, device: str, param_names: list):
    """Compute F_i per parameter in `param_names` as fp32 tensors."""
    # zero-init accumulators
    fisher = {name: torch.zeros_like(dict(net.named_parameters())[name].data, dtype=torch.float32,
                                     device=device) for name in param_names}
    n = 0
    for ex in examples:
        prompt_ids = tok.encode(ex.prompt, add_special_tokens=False)
        cont_ids = tok.encode(ex.gold, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long, device=device)
        # forward
        net.zero_grad(set_to_none=True)
        logits = net(input_ids=input_ids).logits[0]
        p0 = len(prompt_ids) - 1
        cont_logits = logits[p0 : p0 + len(cont_ids)].float()
        lp = F.log_softmax(cont_logits, dim=-1)
        tgt = torch.tensor(cont_ids, dtype=torch.long, device=device)
        lp_gold = lp.gather(1, tgt[:, None]).squeeze(1)
        loss = lp_gold.sum()  # Σ_t log p(y⁺_t | ...)
        loss.backward()
        # accumulate gradient^2
        with torch.no_grad():
            for name in param_names:
                g = dict(net.named_parameters())[name].grad
                if g is not None:
                    fisher[name] += g.detach().float().pow(2)
        n += 1
    # normalize
    with torch.no_grad():
        for name in param_names:
            fisher[name] /= max(1, n)
    return fisher, n


def head_aggregated_scores(net, fisher: dict) -> dict:
    """Reduce parameter-level Fisher to per-head scores by summing over the head's slots."""
    info = model_arch_info(net)
    n_layers, n_heads, d, hidden = info["n_layers"], info["n_heads"], info["head_dim"], info["hidden_size"]
    scores = {}
    for l in range(n_layers):
        names = head_param_names(l)
        F_qkv = fisher[names["qkv_weight"]]  # [3*hidden, hidden]
        F_qkv_b = fisher.get(names["qkv_bias"])  # [3*hidden] or None
        F_dense = fisher[names["dense_weight"]]  # [hidden, hidden]
        for h in range(n_heads):
            q_start = h * 3 * d
            v_end = h * 3 * d + 3 * d
            # rows q_start:v_end cover W_Q^h, W_K^h, W_V^h all together
            s_qkv_w = F_qkv[q_start:v_end, :].sum().item()
            s_qkv_b = 0.0 if F_qkv_b is None else F_qkv_b[q_start:v_end].sum().item()
            # dense: cols h*d : (h+1)*d
            s_dense = F_dense[:, h * d : (h + 1) * d].sum().item()
            scores[f"{l},{h}"] = {
                "layer": l,
                "head": h,
                "score": s_qkv_w + s_qkv_b + s_dense,
                "F_qkv_weight": s_qkv_w,
                "F_qkv_bias": s_qkv_b,
                "F_dense_weight": s_dense,
            }
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["pythia-410m", "pythia-1b", "pythia-2.8b"])
    ap.add_argument("--signal", required=True, choices=list(SIGNAL_SPEC.keys()))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--grad-dtype", default="fp32", choices=["fp32"])
    ap.add_argument("--batch-size", type=int, default=1, help="unused; enforced batch=1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True, help="per-parameter Fisher tensor state dict (.pt)")
    ap.add_argument("--head-scores-output", required=True, help="per-head aggregated Fisher json")
    ap.add_argument("--jackknife", action="store_true",
                    help="ALSO compute and save two half-Fisher tensors for M2.2 stability check.")
    ap.add_argument("--jackknife-half-a", default=None)
    ap.add_argument("--jackknife-half-b", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"[m2.1] {args.model} × {args.signal}")

    # load model in fp32 for faithful gradients
    net, tok = load_model_and_tokenizer(args.model_root, args.model, dtype="fp32", device=args.device)
    for p in net.parameters():
        p.requires_grad_(True)

    spec = SIGNAL_SPEC[args.signal]
    examples = load_task(args.data_root, spec["task"], person_filter=spec["person_filter"])
    print(f"[m2.1] {args.signal}: {len(examples)} examples ({spec['task']}, filter={spec['person_filter']})")

    # enumerate parameter names to accumulate (all attention params in every layer)
    info = model_arch_info(net)
    param_names = []
    for l in range(info["n_layers"]):
        names = head_param_names(l)
        param_names.extend([names["qkv_weight"], names["qkv_bias"], names["dense_weight"]])

    t0 = time.time()
    fisher_full, n_full = compute_fisher_for_examples(net, tok, examples, device=args.device, param_names=param_names)
    print(f"[m2.1] full Fisher over {n_full} examples in {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in fisher_full.items()}, args.output)
    print(f"[m2.1] wrote {args.output}")

    head_scores = head_aggregated_scores(net, fisher_full)
    head_scores_meta = {
        "model": args.model, "signal": args.signal, "n_examples": n_full,
        "head_scores": head_scores,
    }
    save_json(args.head_scores_output, head_scores_meta)
    print(f"[m2.1] wrote {args.head_scores_output}")

    # optional jackknife halves
    if args.jackknife:
        assert args.jackknife_half_a and args.jackknife_half_b, "provide --jackknife-half-{a,b}"
        # deterministic split: split_seed=0, stratified by person if applicable
        import random as _r
        rng = _r.Random(0)
        idx = list(range(len(examples)))
        rng.shuffle(idx)
        half = len(idx) // 2
        idx_a, idx_b = sorted(idx[:half]), sorted(idx[half:])
        for split_name, split_idx, out in (
            ("a", idx_a, args.jackknife_half_a),
            ("b", idx_b, args.jackknife_half_b),
        ):
            t0 = time.time()
            sub = [examples[i] for i in split_idx]
            fh, nh = compute_fisher_for_examples(net, tok, sub, device=args.device, param_names=param_names)
            print(f"[m2.1] jackknife half-{split_name}: n={nh} in {time.time()-t0:.1f}s")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in fh.items()}, out)
            print(f"[m2.1] wrote {out}")


if __name__ == "__main__":
    main()
