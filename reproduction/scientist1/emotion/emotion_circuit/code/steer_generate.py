"""Method 2: direction-based steering.

For each target emotion, add scale * emo_dir[l_target] to the residual
stream at every generated token position, at a chosen set of layers.

By default we steer at all layers where the direction is strong (mid-late
layers) with the same scale, as this matches the paper's multi-layer
addition.

Usage
-----
python steer_generate.py --directions dir_out/directions.pt \
  --split test --out steer_gen/test.jsonl --scale 6.0 --layers 12-27
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    OUT_DIR, MODELS_DIR, DATA_DIR, EMOTIONS,
    build_records, save_jsonl, make_prompt, format_chat,
)


def parse_layers(s, num_layers):
    if s == "all":
        return list(range(num_layers))
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


class SteerHooks:
    def __init__(self, model, layers, direction_by_layer, scale):
        """direction_by_layer: dict layer_idx -> tensor(d) representing
        the residual-stream unit direction to add.
        scale: multiplicative factor.
        """
        self.model = model
        self.layers = layers
        self.dir = direction_by_layer
        self.scale = scale
        self._handles = []

    def install(self):
        m = self.model.model
        model_dtype = next(self.model.parameters()).dtype
        for l in self.layers:
            if l >= len(m.layers):
                continue
            layer = m.layers[l]
            d = self.dir[l].to(self.model.device).to(model_dtype)
            s = self.scale
            def make_hook(d_, s_):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h = h + s_ * d_
                        return (h,) + output[1:]
                    return output + s_ * d_
                return hook
            handle = layer.register_forward_hook(make_hook(d, s))
            self._handles.append(handle)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def load_directions(path):
    obj = torch.load(path, map_location="cpu")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", required=True)
    ap.add_argument("--split", choices=["sev", "test"], required=True)
    ap.add_argument("--valence_mode", choices=["paired", "all"], default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(MODELS_DIR / "llama32-3b-full"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--emotions", default=",".join(EMOTIONS))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Additive scale applied per layer.")
    ap.add_argument("--layers", default="8-27",
                    help="Range or comma list of layer indices to steer at.")
    ap.add_argument("--target_layer", type=int, default=None,
                    help="If set, use only this layer's direction as the "
                         "vector added to residual (broadcast to selected layers). "
                         "Otherwise use each layer's own direction from the "
                         "directions file.")
    ap.add_argument("--unit_norm", action="store_true",
                    help="Normalize direction to unit norm before scaling.")
    ap.add_argument("--limit_per_emotion", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    return_args = ap.parse_args()
    args = return_args

    directions = load_directions(args.directions)
    emo_dir = directions["emo_dir"]  # dict[emotion] -> (L+1, d)

    print(f"[model] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    layers = parse_layers(args.layers, num_layers)
    print(f"[steer] layers={layers} scale={args.scale}")

    ds_path = DATA_DIR / ("sev.jsonl" if args.split == "sev" else "test_set.jsonl")
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]

    all_rows = []
    t0 = time.time()
    for emo in emotions:
        # Build direction dict per-layer for this emotion
        v = emo_dir[emo]  # (L+1, d)
        if args.target_layer is not None:
            d_vec = v[args.target_layer + 1]  # post-layer residual dir at target
            if args.unit_norm:
                d_vec = d_vec / (d_vec.norm() + 1e-8)
            dir_map = {l: d_vec.clone() for l in layers}
        else:
            dir_map = {}
            for l in layers:
                dv = v[l + 1]  # residual dir at that layer's output
                if args.unit_norm:
                    dv = dv / (dv.norm() + 1e-8)
                dir_map[l] = dv

        steer = SteerHooks(model, layers, dir_map, args.scale)
        steer.install()

        # Build inference records
        records = build_records(str(ds_path), emotions=[emo],
                                 include_neutral=False,
                                 valence_mode=args.valence_mode)
        # Overwrite emotion tag consistent, no change
        if args.limit_per_emotion:
            records = records[: args.limit_per_emotion]
        print(f"[{emo}] {len(records)} records")

        for i in range(0, len(records), args.batch_size):
            batch = records[i : i + args.batch_size]
            # Use NEUTRAL/INFERENCE prompt (no emotion cue)
            texts = [format_chat(tokenizer, make_prompt(r, "inference")) for r in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                             truncation=True, add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = out[:, enc["input_ids"].shape[1] :]
            gens = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for r, g in zip(batch, gens):
                r = dict(r)
                r["gen_text"] = g.strip()
                r["method"] = "steer"
                r["scale"] = args.scale
                all_rows.append(r)

        steer.remove()
        elapsed = time.time() - t0
        print(f"[{emo}] done  elapsed={elapsed:.0f}s")

    out_path = OUT_DIR / args.out if not os.path.isabs(args.out) else Path(args.out)
    save_jsonl(str(out_path), all_rows)
    print(f"[done] wrote {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
