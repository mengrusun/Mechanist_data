"""Method 3: circuit-based emotion generation.

For each selected (layer, neuron), we add scale * delta_val to the MLP
intermediate activation (post-SwiGLU, pre-down_proj) at every position
during generation.

For each selected (layer, head), we add scale * delta_z (hd-dim vector)
to the per-head output slice of the attention pre-o_proj tensor.

The additions happen inside forward hooks. We install hooks per emotion,
generate, then remove.

Usage
-----
python circuit_generate.py --circuits circuit.pt --split test \
  --out circuit_gen/test.jsonl --scale 0.8
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


class CircuitHooks:
    """Install per-layer hooks that add per-neuron and per-head deltas."""

    def __init__(self, model, circuit_emo, scale):
        self.model = model
        self.circuit = circuit_emo
        self.scale = scale
        self._handles = []
        self.cfg = model.config
        self.H = self.cfg.num_attention_heads
        self.hd = self.cfg.head_dim
        # Prepare per-layer neuron delta vector (d_int) and per-layer
        # head delta reshape (H*hd) to be added on device.
        d_int = self.cfg.intermediate_size
        L = self.cfg.num_hidden_layers
        device = next(model.parameters()).device
        self.mlp_delta = {}
        for l, m in circuit_emo["mlp_by_layer"].items():
            vec = torch.zeros(d_int, dtype=torch.float32)
            for i, v in m.items():
                vec[int(i)] = float(v)
            self.mlp_delta[int(l)] = (vec * scale).to(device).to(
                dtype=next(model.parameters()).dtype
            )
        self.head_delta = {}
        for l, h in circuit_emo["head_by_layer"].items():
            vec = torch.zeros(self.H, self.hd, dtype=torch.float32)
            for hi, dv in h.items():
                vec[int(hi)] = torch.tensor(dv, dtype=torch.float32)
            flat = (vec.reshape(self.H * self.hd) * scale).to(device).to(
                dtype=next(model.parameters()).dtype
            )
            self.head_delta[int(l)] = flat

    def install(self):
        m = self.model.model
        for l, layer in enumerate(m.layers):
            # MLP intermediate hook: modify input to down_proj
            if l in self.mlp_delta:
                delta = self.mlp_delta[l]
                def mk_mlp(delta_):
                    def pre_hook(module, inputs):
                        x = inputs[0]
                        x = x + delta_.view(1, 1, -1)
                        return (x,) + inputs[1:]
                    return pre_hook
                h = layer.mlp.down_proj.register_forward_pre_hook(mk_mlp(delta))
                self._handles.append(h)
            # Attention head hook: modify input to o_proj
            if l in self.head_delta:
                delta = self.head_delta[l]
                def mk_attn(delta_):
                    def pre_hook(module, inputs):
                        x = inputs[0]
                        x = x + delta_.view(1, 1, -1)
                        return (x,) + inputs[1:]
                    return pre_hook
                h = layer.self_attn.o_proj.register_forward_pre_hook(mk_attn(delta))
                self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--split", choices=["sev", "test"], required=True)
    ap.add_argument("--valence_mode", choices=["paired", "all"], default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(MODELS_DIR / "llama32-3b-full"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--emotions", default=",".join(EMOTIONS))
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--limit_per_emotion", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    circuits = torch.load(args.circuits, map_location="cpu")

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

    ds_path = DATA_DIR / ("sev.jsonl" if args.split == "sev" else "test_set.jsonl")
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]

    all_rows = []
    t0 = time.time()
    for emo in emotions:
        if emo not in circuits:
            print(f"[skip] no circuit for {emo}")
            continue
        hooks = CircuitHooks(model, circuits[emo], args.scale)
        hooks.install()

        records = build_records(str(ds_path), emotions=[emo], include_neutral=False,
                                 valence_mode=args.valence_mode)
        if args.limit_per_emotion:
            records = records[: args.limit_per_emotion]
        print(f"[{emo}] {len(records)} records")

        for i in range(0, len(records), args.batch_size):
            batch = records[i : i + args.batch_size]
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
                r["method"] = "circuit"
                r["scale"] = args.scale
                all_rows.append(r)

        hooks.remove()
        elapsed = time.time() - t0
        print(f"[{emo}] done  elapsed={elapsed:.0f}s")

    out_path = OUT_DIR / args.out if not os.path.isabs(args.out) else Path(args.out)
    save_jsonl(str(out_path), all_rows)
    print(f"[done] wrote {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
