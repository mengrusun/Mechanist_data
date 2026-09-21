"""M4.e — Pile PPL preservation under amplifier.

Apply the amplifier during Pile forward passes and measure PPL vs clean.
Tolerance: ≤ 1.05 × clean.

The amplifier reads the classifier at layer L_star for every window; typical
Pile text is far from the belief frames, so the classifier will usually predict
world_knowledge → identity fallback → PPL should be very close to clean.
"""
import argparse, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from belief_lib import PILE_DIR, load_model, save_json, set_seed, _load_pile_windows
from run_m4c_amplifier import AmplifierContext, load_headsets


@torch.no_grad()
def pile_ppl_with_amp(model, tok, windows_tensor, device, amp_ctx=None):
    total_nll = 0.0
    total_toks = 0
    W = windows_tensor.shape[0]
    for i in range(W):
        batch = windows_tensor[i : i + 1].to(device)  # [1, T]
        Lp = batch.shape[1]  # entire input is "prompt" for PPL — last-position read
        if amp_ctx is not None:
            amp_ctx.pred_frame = None
            amp_ctx.current_prompt_len = Lp
        out = model(batch)
        logits = out.logits[:, :-1, :].float()
        targets = batch[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             targets.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_toks += targets.numel()
        if (i + 1) % 100 == 0:
            print(f"  pile {i+1}/{W}  running_ppl={math.exp(total_nll/total_toks):.3f}", flush=True)
    return math.exp(total_nll / total_toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--classifier_ckpt", required=True)
    ap.add_argument("--headset_personal", default=None)
    ap.add_argument("--headset_attributed", default=None)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pile_docs", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--pile_windows", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda"
    print(f"[M4e] model={args.model} alpha={args.alpha}", flush=True)

    ckpt = torch.load(args.classifier_ckpt, map_location="cpu", weights_only=False)
    L_star = int(ckpt["best_layer_L_star"])
    head_sets = load_headsets(args.headset_personal, args.headset_attributed)

    model, tok, meta = load_model(args.model, device=device, dtype=torch.float16)

    docs = [int(x) for x in args.pile_docs.split(",")]
    windows = _load_pile_windows(docs, args.pile_windows, ctx_len=2048)
    print(f"[M4e] pile: {windows.shape[0]} windows × {windows.shape[1]} tokens", flush=True)

    print("[M4e] clean PPL …", flush=True)
    clean_ppl = pile_ppl_with_amp(model, tok, windows, device, amp_ctx=None)
    print(f"[M4e] clean PPL = {clean_ppl:.4f}", flush=True)

    print("[M4e] amplifier PPL …", flush=True)
    amp = AmplifierContext(model, L_star, ckpt["coef"], ckpt["intercept"],
                          head_sets, args.alpha)
    with amp:
        amp_ppl = pile_ppl_with_amp(model, tok, windows, device, amp_ctx=amp)
    print(f"[M4e] amp PPL = {amp_ppl:.4f} ratio={amp_ppl/clean_ppl:.4f}", flush=True)

    out = {
        "model": args.model, "alpha": args.alpha, "seed": args.seed,
        "L_star": L_star,
        "head_sets": {k: [list(x) for x in v] for k, v in head_sets.items()},
        "clean_pile_ppl": clean_ppl,
        "amplified_pile_ppl": amp_ppl,
        "ppl_ratio": amp_ppl / clean_ppl,
        "passes_1p05": amp_ppl / clean_ppl <= 1.05,
    }
    save_json(out, args.out)


if __name__ == "__main__":
    main()
