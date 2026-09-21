"""Generate MGSM answers under 3 conditions per language:

  baseline    : no intervention
  suppress    : h <- h - ((h - c_L) @ P_L) @ P_L.T   on layers in INTERVENE_LAYERS
  amplify     : h <- h + α * ((h - c_L) @ P_L) @ P_L.T

At generation time we hook a subset of transformer blocks. The intervention is applied
on every forward pass (prefill + each decode step), so the KV-cache is built from the
modified hidden states throughout generation.

Usage:
  CUDA_VISIBLE_DEVICES=1 python code/generate_mgsm.py \
      --condition baseline --n_per_lang 40 --enable_thinking 0
"""
import os, sys, time, json, argparse, gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_PATH, LANGS, load_mgsm, build_prompt, apply_chat,
    parse_answer, is_correct, dump_json,
)

# ---- defaults -------------------------------------------------------------
DEFAULT_LAYER_START = 4
DEFAULT_LAYER_END   = 24         # exclusive
DEFAULT_K           = 10
DEFAULT_N = 40
DEFAULT_MAX_NEW = 512
# ---------------------------------------------------------------------------


class SubspaceIntervener:
    """Forward-hook that modifies the output hidden state of a transformer block.

    op: 'suppress' -> h - alpha * proj , 'amplify' -> h + alpha * proj , 'none' -> passthrough
    (alpha=1.0 is full removal for 'suppress'.)
    """
    def __init__(self, P: torch.Tensor, c: torch.Tensor, op: str, alpha: float = 1.0):
        self.P = P              # (D, k)
        self.c = c              # (D,)
        self.op = op
        self.alpha = float(alpha)

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        Pf = self.P.to(dtype=h.dtype, device=h.device)
        cf = self.c.to(dtype=h.dtype, device=h.device)
        diff = h - cf
        proj = torch.matmul(torch.matmul(diff, Pf), Pf.transpose(0, 1))
        if self.op == "suppress":
            h_new = h - self.alpha * proj
        elif self.op == "amplify":
            h_new = h + self.alpha * proj
        else:
            h_new = h

        if rest is not None:
            return (h_new,) + rest
        return h_new


def install_hooks(model, bases_np, centres_np, block_ids, op, K, alpha=1.0):
    """Install hooks on the given transformer block indices.

    Use only the top-K components of the language subspace (K <= bases shape).
    """
    hooks = []
    if op == "none":
        return hooks
    for bi in block_ids:
        P_full = bases_np[bi + 1]                      # (D, K_all)
        P = torch.from_numpy(P_full[:, :K].astype(np.float32))
        c = torch.from_numpy(centres_np[bi + 1].astype(np.float32))
        intv = SubspaceIntervener(P, c, op, alpha)
        h = model.model.layers[bi].register_forward_hook(intv)
        hooks.append(h)
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["baseline", "suppress", "amplify"], required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k_lang", type=int, default=DEFAULT_K)
    ap.add_argument("--layer_start", type=int, default=DEFAULT_LAYER_START)
    ap.add_argument("--layer_end",   type=int, default=DEFAULT_LAYER_END)
    ap.add_argument("--n_per_lang", type=int, default=DEFAULT_N)
    ap.add_argument("--max_new", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument("--enable_thinking", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--langs", type=str, default=None,
                    help="comma-separated language codes; default all 11")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"condition={args.condition} alpha={args.alpha} n_per_lang={args.n_per_lang} "
          f"max_new={args.max_new} thinking={args.enable_thinking}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    subs = np.load("cache/subspace.npz")
    bases = subs["bases"]      # (L+1, D, K)
    centres = subs["centres"]  # (L+1, D)

    print("loading MGSM test")
    data = load_mgsm("test")

    # Fix same probe indices as extract_probes.py; use test items AFTER that range
    # to avoid data leakage in the probe set. probes were indices 0..63, so we use 64..64+n.
    start = 64
    idxs = list(range(start, start + args.n_per_lang))

    op = {"baseline": "none", "suppress": "suppress", "amplify": "amplify"}[args.condition]
    layer_blocks = list(range(args.layer_start, args.layer_end))
    hooks = install_hooks(model, bases, centres, layer_blocks, op,
                          K=args.k_lang, alpha=args.alpha)
    print(f"installed {len(hooks)} hooks on blocks {layer_blocks} K={args.k_lang}")

    gen_kwargs = dict(
        max_new_tokens=args.max_new,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tok.pad_token_id,
    )

    results = {}
    total = 0; correct = 0
    t0 = time.time()
    langs_to_run = args.langs.split(",") if args.langs else LANGS
    for lg in langs_to_run:
        df = data[lg]
        rows = []
        prompts = []
        golds = []
        for i in idxs:
            q = df.iloc[i]["question"]
            gold = df.iloc[i]["answer_number"]
            prompt = apply_chat(tok, build_prompt(q, lg),
                                enable_thinking=bool(args.enable_thinking))
            prompts.append(prompt)
            golds.append(gold)
        n_correct = 0
        for b0 in range(0, len(prompts), args.batch):
            b_prompts = prompts[b0:b0 + args.batch]
            enc = tok(b_prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1536).to("cuda:0")
            with torch.inference_mode():
                out = model.generate(**enc, **gen_kwargs)
            # split off input tokens per sample (they may vary because of pad on left/right)
            input_lens = enc["attention_mask"].sum(dim=1).tolist()
            for j, o in enumerate(out):
                # For right-padded input, we need to skip the padded tokens; easier: decode
                # the full output and strip the decoded prompt.
                full = tok.decode(o, skip_special_tokens=True)
                # decoded prompt for stripping
                dec_prompt = tok.decode(enc["input_ids"][j], skip_special_tokens=True)
                gen = full[len(dec_prompt):] if full.startswith(dec_prompt) else full
                pred = parse_answer(gen)
                ok = is_correct(pred, golds[b0 + j])
                n_correct += int(ok)
                rows.append({
                    "idx": idxs[b0 + j],
                    "gold": None if golds[b0 + j] is None else float(golds[b0 + j]),
                    "pred": pred,
                    "correct": bool(ok),
                    "gen": gen[:2000],
                })
        acc = n_correct / len(prompts)
        results[lg] = {"n": len(prompts), "correct": n_correct, "acc": acc, "rows": rows}
        correct += n_correct
        total += len(prompts)
        print(f"[{lg}] acc {acc:.3f} ({n_correct}/{len(prompts)})  elapsed {time.time()-t0:.0f}s")

    remove_hooks(hooks)

    overall_acc = correct / total if total else 0.0
    summary = {
        "condition": args.condition,
        "alpha": args.alpha,
        "n_per_lang": args.n_per_lang,
        "max_new": args.max_new,
        "enable_thinking": bool(args.enable_thinking),
        "layers": layer_blocks,
        "K": args.k_lang,
        "overall_acc": overall_acc,
        "per_lang_acc": {lg: r["acc"] for lg, r in results.items()},
    }
    print("SUMMARY:", json.dumps(summary, indent=2))

    out_path = args.out or (
        f"results/mgsm_{args.condition}_k{args.k_lang}_a{args.alpha}"
        f"_L{args.layer_start}-{args.layer_end}_t{args.enable_thinking}.json"
    )
    dump_json({"summary": summary, "results": results}, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
