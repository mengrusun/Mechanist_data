"""Dump last-token activations for the labeled emotion (or neutral)
generations. We collect three tensors per sample:

- resid[num_layers+1, d]      : residual stream at each layer input (0..L)
- attn_z[num_layers, num_heads, head_dim] : per-head attention output
                                            (before W_o), from which the
                                            per-head residual contribution
                                            can be computed as W_o[h] @ z[h]
- mlp_int[num_layers, d_int]  : SwiGLU intermediate activation
                                (silu(gate)*up), which multiplied by
                                W_down gives the MLP output contribution

Only rows with correct==1 (matching emotion) are used, filtered per
emotion. For the neutral bucket we use rows with target 'neutral' and
label 'neutral'.

We feed the prompt + generated text through the model and read
activations at the *last* generated token position.
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import OUT_DIR, MODELS_DIR, load_jsonl, save_jsonl, format_chat, make_prompt


class Capturer:
    def __init__(self, model):
        self.model = model
        cfg = model.config
        self.num_layers = cfg.num_hidden_layers
        self.d = cfg.hidden_size
        self.d_int = cfg.intermediate_size
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self._hooks = []
        self._resid = None   # (L+1, d)
        self._attn_z = None  # (L, H, hd)
        self._mlp_int = None # (L, d_int)
        self._pos_slice = None  # slice of positions to average over

    def install(self):
        m = self.model.model  # LlamaModel

        for i, layer in enumerate(m.layers):
            self._hooks.append(
                layer.register_forward_hook(self._make_layer_input_hook(i))
            )
            self._hooks.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    self._make_attn_z_hook(i)
                )
            )
            self._hooks.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    self._make_mlp_int_hook(i)
                )
            )
        self._hooks.append(
            m.norm.register_forward_pre_hook(self._make_final_hook())
        )

    def _mean_at_pos(self, x):
        # x shape: (b, s, ...) -> take batch 0, mean over the position slice
        sl = self._pos_slice
        seg = x[0, sl]  # (n_positions, ...)
        return seg.mean(dim=0)

    def _make_layer_input_hook(self, layer_idx):
        def hook(module, inputs, output):
            x = inputs[0]
            self._resid[layer_idx] = self._mean_at_pos(x).detach().to("cpu", torch.float32)
        return hook

    def _make_attn_z_hook(self, layer_idx):
        def hook(module, inputs):
            z = inputs[0]  # (b, s, H*hd)
            mean_z = self._mean_at_pos(z).detach()
            mean_z = mean_z.reshape(self.num_heads, self.head_dim)
            self._attn_z[layer_idx] = mean_z.to("cpu", torch.float32)
        return hook

    def _make_mlp_int_hook(self, layer_idx):
        def hook(module, inputs):
            x = inputs[0]  # (b, s, d_int)
            self._mlp_int[layer_idx] = self._mean_at_pos(x).detach().to("cpu", torch.float32)
        return hook

    def _make_final_hook(self):
        def hook(module, inputs):
            x = inputs[0]
            self._resid[self.num_layers] = self._mean_at_pos(x).detach().to("cpu", torch.float32)
        return hook

    def start(self, pos_slice):
        """pos_slice: slice or tensor of positions to average over."""
        self._pos_slice = pos_slice
        self._resid = torch.zeros(self.num_layers + 1, self.d, dtype=torch.float32)
        self._attn_z = torch.zeros(self.num_layers, self.num_heads, self.head_dim, dtype=torch.float32)
        self._mlp_int = torch.zeros(self.num_layers, self.d_int, dtype=torch.float32)

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def build_forward_text(rec, tokenizer):
    """Reconstruct: chat prompt + gen_text."""
    prompt = make_prompt(rec, mode="prompt")
    chat = format_chat(tokenizer, prompt)
    return chat, chat + rec["gen_text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True,
                    help="Labeled SEV jsonl (with correct field)")
    ap.add_argument("--out_dir", required=True,
                    help="Output dir for tensors")
    ap.add_argument("--model", default=str(MODELS_DIR / "llama32-3b-full"))
    ap.add_argument("--max_per_emotion", type=int, default=200)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    out_dir = OUT_DIR / args.out_dir if not os.path.isabs(args.out_dir) else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[model] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    rows = load_jsonl(args.labeled if os.path.isabs(args.labeled) else str(OUT_DIR / args.labeled))
    # bucket by emotion; only correct ones
    buckets = defaultdict(list)
    for r in rows:
        if r.get("correct") == 1:
            buckets[r["emotion"]].append(r)
    print("[data]", {k: len(v) for k, v in buckets.items()})

    cap = Capturer(model)
    cap.install()

    # per-emotion stacks
    stacks = {}  # emotion -> {resid, attn_z, mlp_int}
    for emo, rs in buckets.items():
        rs = rs[: args.max_per_emotion]
        resid_stack = []
        attn_stack = []
        mlp_stack = []
        keys = []
        for i, r in enumerate(rs):
            chat, full = build_forward_text(r, tokenizer)
            # Encode prompt only to know where response starts
            prompt_ids = tokenizer(chat, return_tensors="pt",
                                    add_special_tokens=False).input_ids
            full_enc = tokenizer(full, return_tensors="pt",
                                  add_special_tokens=False, truncation=True,
                                  max_length=1024).to(model.device)
            prompt_len = prompt_ids.shape[1]
            total_len = full_enc["input_ids"].shape[1]
            if total_len <= prompt_len:
                continue
            # Average over response positions
            pos_slice = slice(prompt_len, total_len)
            cap.start(pos_slice)
            with torch.inference_mode():
                model(**full_enc)
            resid_stack.append(cap._resid.clone())
            attn_stack.append(cap._attn_z.clone())
            mlp_stack.append(cap._mlp_int.clone())
            keys.append(r["key"])
        resid = torch.stack(resid_stack).to(torch.float16)
        attn = torch.stack(attn_stack).to(torch.float16)
        mlp = torch.stack(mlp_stack).to(torch.float16)
        print(f"  {emo}: resid {tuple(resid.shape)}, attn {tuple(attn.shape)}, mlp {tuple(mlp.shape)}")
        torch.save({"resid": resid, "attn_z": attn, "mlp_int": mlp, "keys": keys},
                   out_dir / f"acts_{emo}.pt")
        stacks[emo] = (resid, attn, mlp)

    cap.close()
    print(f"[done] wrote activations to {out_dir}")


if __name__ == "__main__":
    main()
