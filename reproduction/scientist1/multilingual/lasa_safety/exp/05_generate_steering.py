"""
Generate LLaMA-3.1-8B-Instruct completions on multilingual MultiJail prompts
under (a) no steering (baseline) and (b) residual-stream addition of the
per-layer refusal direction at a chosen layer.

Steering is applied to the OUTPUT of decoder layer `--layer` (0-indexed among
transformer blocks). The vector added is `alpha * v_unit_layer` where the
scale is measured in units of unit-norm refusal direction.
"""

import os, argparse, json, math, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def build_chat_ids(tokenizer, user_text, device):
    msgs = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(device)


class ResidualSteer:
    """Register a forward-hook on `model.model.layers[layer]` that adds
    `alpha * v` (a torch tensor on device) to the layer output residual for
    every position in the sequence."""
    def __init__(self, model, layer, v, alpha):
        self.model = model
        self.layer = layer
        self.v = v
        self.alpha = alpha
        self.handle = None
        self.dtype = None

    def __enter__(self):
        mod = self.model.model.layers[self.layer]
        self.dtype = next(mod.parameters()).dtype
        v_local = self.v.to(self.dtype)
        alpha = self.alpha
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                h = h + alpha * v_local.view(1, 1, -1)
                return (h,) + output[1:]
            else:
                return output + alpha * v_local.view(1, 1, -1)
        self.handle = mod.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()


def generate_batch(model, tok, prompts, device, max_new_tokens=128, steer=None):
    outs = []
    for p in prompts:
        if not isinstance(p, str) or not p.strip():
            outs.append(""); continue
        ids = build_chat_ids(tok, p, device)
        cm = steer if steer is not None else _NullCtx()
        with torch.inference_mode(), cm:
            gen = model.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.0,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        outs.append(text)
    return outs


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--multijail", default="/data/zhenqian/data/multijail/MultiJail.csv")
    ap.add_argument("--refusal_dir", default="cache/refusal_dir.npz")
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--langs", nargs="+",
                    default=["en","zh","it","vi","ar","ko","th","bn","sw","jv"])
    ap.add_argument("--layer", type=int, default=-1, help="-1 => no steering (baseline)")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()

    df = pd.read_csv(args.multijail).head(args.n_prompts).reset_index(drop=True)

    steer_cm = None
    if args.layer >= 0 and args.alpha != 0.0:
        d = np.load(args.refusal_dir)
        v_unit = d["v_unit"]                # [L+1, H]
        # Layer index in v_unit is offset by 1 (0=embedding, 1=layer0 output).
        # We steer the OUTPUT of decoder layer `l` so we use v_unit[l+1].
        v = torch.from_numpy(v_unit[args.layer + 1].astype(np.float32)).to(device)
        print(f"[steer] layer={args.layer}, alpha={args.alpha}, ||v||=1.0 (unit)")
        steer_cm = None  # created per-call

    all_rows = []
    t0 = time.time()
    for lang in args.langs:
        prompts = df[lang].tolist()
        # steer per generation
        cm = None
        if args.layer >= 0 and args.alpha != 0.0:
            v = torch.from_numpy(np.load(args.refusal_dir)["v_unit"][args.layer + 1].astype(np.float32)).to(device)
            cm = ResidualSteer(model, args.layer, v, args.alpha)
        outs = generate_batch(model, tok, prompts, device,
                              max_new_tokens=args.max_new_tokens, steer=cm)
        for i, (p, o) in enumerate(zip(prompts, outs)):
            all_rows.append({"prompt_id": i, "lang": lang, "prompt": p, "response": o,
                             "layer": args.layer, "alpha": args.alpha})
        print(f"[{lang}] done  elapsed={time.time() - t0:.1f}s")

    out_df = pd.DataFrame(all_rows)
    out_df.to_json(args.out, orient="records", force_ascii=False, lines=True)
    print(f"[save] {args.out}  total_time={time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
