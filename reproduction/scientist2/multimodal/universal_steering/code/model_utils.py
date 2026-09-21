"""Llama-3.1-8B-Instruct wrapper: activation caching + additive steering hooks.

- `load_model(...)` returns (model, tokenizer, cfg).
- `cache_last_token_activations(model, tokenizer, prompts, batch_size, device)`
  returns (n, num_blocks, d_model) float16 numpy array — last-token residual-stream
  activations after each transformer block.
- `AdditiveSteerer(model, block_idx, v_c, alpha)` context manager installs a forward hook
  that adds α · v_c to the residual-stream output of block `block_idx` at every position.

The steering hook is a **residual-stream additive intervention** at the *output* of
block `l`, which is the standard site for RepE / CAA / ActAdd-style steering.
"""

from __future__ import annotations
import contextlib
import gc
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH_DEFAULT = "/data/zhenqian/models/Llama-3.1-8B-Instruct"


def load_model(model_path: str = MODEL_PATH_DEFAULT, dtype: str = "bfloat16", device: str = "cuda"):
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    # Prefer left-padding for causal generation
    tokenizer.padding_side = "left"
    num_blocks = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    return model, tokenizer, {"num_blocks": num_blocks, "d_model": d_model, "dtype": dtype}


def _get_blocks(model):
    """Return the ModuleList of transformer blocks for Llama."""
    return model.model.layers


@torch.no_grad()
def cache_last_token_activations(
    model,
    tokenizer,
    prompts: Sequence[str],
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """Return (n_prompts, num_blocks, d_model) float16 activations of the *last non-pad token*
    after each transformer block (post-block residual stream).

    We hook each block's forward to capture `output[0]` (residual hidden state).
    """
    blocks = _get_blocks(model)
    num_blocks = len(blocks)
    d_model = model.config.hidden_size
    n = len(prompts)
    out = np.zeros((n, num_blocks, d_model), dtype=np.float16)

    # Buffer of shape (batch, num_blocks, d) accumulated in the hook.
    cache = [None] * num_blocks

    def make_hook(idx):
        def hook(module, inputs, output):
            # output is either (hidden_states,) tuple or a tensor
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            cache[idx] = hs.detach()
            return output
        return hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(blocks)]
    try:
        for b0 in range(0, n, batch_size):
            batch = list(prompts[b0:b0 + batch_size])
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_length).to(device)
            _ = model(**enc, use_cache=False)
            # Last non-pad token index per row: since padding is left, last token is at position -1
            for i, _p in enumerate(batch):
                for l in range(num_blocks):
                    hs = cache[l]
                    out[b0 + i, l] = hs[i, -1, :].to(torch.float16).cpu().numpy()
    finally:
        for h in handles:
            h.remove()
    return out


class AdditiveSteerer(contextlib.AbstractContextManager):
    """Context manager that adds α · v_c to residual-stream output of block(s) `block_idx`.

    Multiple (block, direction, alpha) triples can be combined for compositional steering.
    Applied at every position — including prompt tokens *and* generated tokens.

    Args:
      model: transformers CausalLM
      interventions: list of dicts {block_idx, v_c (np.ndarray or torch.Tensor), alpha}
    """

    def __init__(self, model, interventions: List[dict]):
        self.model = model
        self.interventions = interventions
        self._handles = []
        # Preprocess: cast vectors to tensor on model device+dtype
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        # Group by block_idx: sum(α_k · v_k) at that block
        combined = {}
        for iv in interventions:
            b = int(iv["block_idx"])
            v = iv["v_c"]
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            v = v.to(device=device, dtype=dtype)
            alpha = float(iv["alpha"])
            if b not in combined:
                combined[b] = torch.zeros_like(v)
            combined[b] = combined[b] + alpha * v
        self.combined = combined

    def _make_hook(self, delta):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hs = output[0]
                # Broadcast add along seq dimension
                hs = hs + delta.view(1, 1, -1)
                return (hs,) + output[1:]
            else:
                return output + delta.view(1, 1, -1)
        return hook

    def __enter__(self):
        blocks = _get_blocks(self.model)
        for b, delta in self.combined.items():
            h = blocks[b].register_forward_hook(self._make_hook(delta))
            self._handles.append(h)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


@torch.no_grad()
def generate_with_steering(
    model,
    tokenizer,
    prompts: Sequence[str],
    interventions: List[dict] | None,
    max_new_tokens: int = 150,
    batch_size: int = 4,
    temperature: float = 0.0,
    device: str = "cuda",
) -> List[str]:
    """Generate `max_new_tokens` per prompt under optional additive steering.

    Set `interventions=None` (or empty list) for unsteered baseline.
    temperature=0 → greedy.
    """
    outs = []
    steer_ctx = AdditiveSteerer(model, interventions) if interventions else contextlib.nullcontext()
    with steer_ctx:
        for b0 in range(0, len(prompts), batch_size):
            batch = list(prompts[b0:b0 + batch_size])
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=1024).to(device)
            input_len = enc["input_ids"].shape[1]
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
            out_ids = model.generate(**enc, **gen_kwargs)
            gen_ids = out_ids[:, input_len:]
            texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            outs.extend(texts)
    return outs


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
