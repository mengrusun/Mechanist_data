"""Steered generation.

Hook every transformer block's forward pass so that when producing tokens we
add alpha * v_l to the residual stream output. Supports multi-concept
composition (sum of alpha_k * v_k).
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unknown model layout")


class Steerer:
    def __init__(self, model, vectors_list, alphas, blocks_range=None,
                 scale_by_norm=True, norm_scale_stats=None):
        assert len(vectors_list) == len(alphas)
        self.model = model
        self.blocks = _get_blocks(model)
        self.n_blocks = len(self.blocks)
        combined = None
        for v, a in zip(vectors_list, alphas):
            v = v.to(model.device).to(model.dtype)
            if combined is None:
                combined = a * v
            else:
                combined = combined + a * v
        # combined: (L, D)
        if scale_by_norm and norm_scale_stats is not None:
            # scale per-layer by activation norm at that layer
            norm_stats = norm_scale_stats.to(combined.device).to(combined.dtype)
            combined = combined * norm_stats.view(-1, 1)
        self.combined = combined
        if blocks_range is None:
            blocks_range = list(range(self.n_blocks))
        self.blocks_range = set(blocks_range)
        self.handles = []

    def _make_hook(self, layer_idx):
        add_vec = self.combined[layer_idx]

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                h = h + add_vec.to(h.dtype).to(h.device)
                return (h,) + output[1:]
            return output + add_vec.to(output.dtype).to(output.device)

        return hook

    def __enter__(self):
        for i, blk in enumerate(self.blocks):
            if i in self.blocks_range:
                self.handles.append(blk.register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def generate_one(model, tokenizer, prompt, max_new_tokens=200, do_sample=False,
                 system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectors", nargs="+", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--block_start", type=int, default=0)
    ap.add_argument("--block_end", type=int, default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--label_filter", type=int, default=None)
    ap.add_argument("--norm_stats", default=None,
                    help="path to acts .pt used to compute per-layer act norms")
    ap.add_argument("--include_baseline", action="store_true")
    ap.add_argument("--system", default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cuda"
    )
    model.eval()

    vecs = []
    for path in args.vectors:
        d = torch.load(path, weights_only=False)
        vecs.append(d["vectors"])

    norm_stats = None
    if args.norm_stats is not None:
        acts_data = torch.load(args.norm_stats, weights_only=False)
        acts = acts_data["activations"]                        # (N, L, D)
        norm_stats = acts.float().norm(dim=-1).mean(dim=0)     # (L,)

    records = load_jsonl(args.prompts)
    if args.split is not None:
        records = [r for r in records if r.get("split") == args.split]
    if args.label_filter is not None:
        records = [r for r in records if r.get("label") == args.label_filter]
    records = records[: args.limit]

    L = vecs[0].shape[0]
    block_end = args.block_end if args.block_end is not None else L
    blocks = list(range(args.block_start, block_end))

    outputs = []
    # Steered outputs
    with Steerer(model, vecs, args.alphas, blocks_range=blocks,
                 scale_by_norm=(norm_stats is not None),
                 norm_scale_stats=norm_stats):
        for rec in tqdm(records, desc="Steered"):
            gen = generate_one(model, tokenizer, rec["prompt"],
                               max_new_tokens=args.max_new_tokens,
                               system=args.system)
            outputs.append({"prompt": rec["prompt"], "label": rec.get("label"),
                            "steered": gen})

    if args.include_baseline:
        for rec, out in zip(records, tqdm(outputs, desc="Baseline")):
            gen = generate_one(model, tokenizer, rec["prompt"],
                               max_new_tokens=args.max_new_tokens,
                               system=args.system)
            out["baseline"] = gen

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"Wrote {len(outputs)} outputs to {args.out}")


if __name__ == "__main__":
    main()
