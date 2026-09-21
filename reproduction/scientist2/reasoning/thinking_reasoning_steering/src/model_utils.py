"""Shared utilities: model loading + residual-stream activation capture + steering hook."""
from __future__ import annotations
import os, json
from contextlib import contextmanager
from typing import Sequence
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path: str, dtype=torch.bfloat16, device_map: str = "auto"):  # noqa
    """Load the R1-distill-llama model.

    Uses `attn_implementation="eager"` to avoid the FA2/SDPA graph rewrite that
    prevents forward hooks from firing at the layer output (residual add).
    """
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tok


def get_num_layers(model) -> int:
    return len(model.model.layers)


def get_input_device(model):
    """Public accessor for the correct device to place model inputs on."""
    return _input_device(model)


@contextmanager
def capture_residual_last_token(model, layers: Sequence[int]):
    """Context manager that installs forward hooks on `model.model.layers[i]`
    for each i in `layers`. On exit, `captured` dict maps layer_idx -> tensor
    of shape (batch, hidden) with the LAST-TOKEN residual stream activation
    at the OUTPUT of that transformer block (i.e., after the residual add of
    attn+mlp — which is where CAA/steering vectors are traditionally defined).

    For Llama layers, `output` is a tuple whose first element is the hidden
    states; we take the last token position per batch element.
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []
    def make_hook(li: int):
        def _hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            # last non-pad position: since we use left-padding, the last token
            # is always at index -1.
            captured[li] = hs[:, -1, :].detach().to(torch.float32).cpu()
        return _hook
    for li in layers:
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        handles.append(h)
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


class SteeringHook:
    """Additive residual-stream steering at a fixed layer.

    Adds `coef * direction` (broadcast on device/dtype) at *post-prompt* token
    positions. Two cases the hook must handle correctly under HF `generate`:

      - **Pre-fill** (first forward): `hs.shape[1] == prompt_len` and every
        position IS prompt → we skip adding (nothing to steer yet).
      - **Decode step** (subsequent forwards): with the default KV cache,
        `hs.shape[1] == 1` and the single position is a NEW generated token
        → add the direction.
      - **No-cache generate / prompt-len < seq len**: the prefill call may
        contain both prompt and new positions (rare, e.g. `use_cache=False`
        with num_return_sequences > 1). Detect via `self.prompt_len` and
        steer only positions with index ≥ prompt_len.

    This matches the CAA convention (Panickssery et al., 2024): steer every
    generated (post-prompt) token; leave the prompt residual untouched.
    """
    def __init__(self, direction: torch.Tensor, coef: float, dtype=torch.bfloat16):
        # direction shape: (hidden,)
        self.direction = direction.to(dtype)  # will be moved to layer device inside hook
        self.coef = float(coef)
        self.prompt_len = 0
        self._handle = None
        # diagnostic counters
        self.n_prefill_calls = 0
        self.n_decode_calls = 0
        self.n_new_tokens_steered = 0

    def set_prompt_len(self, n: int):
        self.prompt_len = int(n)
        self.n_prefill_calls = 0
        self.n_decode_calls = 0
        self.n_new_tokens_steered = 0

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        seq = hs.shape[1]
        d = (self.direction.to(hs.device, hs.dtype) * self.coef).view(1, 1, -1)

        if seq == 1:
            # Decode step (KV-cache path): the single position is a new token.
            self.n_decode_calls += 1
            self.n_new_tokens_steered += hs.shape[0]  # one per batch element
            hs2 = hs + d
        elif seq == self.prompt_len:
            # Pure prefill: every position is prompt — do not steer.
            self.n_prefill_calls += 1
            return output
        else:
            # Mixed / long-context path: steer only positions ≥ prompt_len.
            self.n_prefill_calls += 1
            if self.prompt_len >= seq:
                return output
            n_new = seq - self.prompt_len
            self.n_new_tokens_steered += n_new * hs.shape[0]
            add = torch.zeros_like(hs)
            add[:, self.prompt_len:, :] = d
            hs2 = hs + add

        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2

    def attach(self, model, layer_idx: int):
        self._handle = model.model.layers[layer_idx].register_forward_hook(self._hook)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def format_task_prompt(tok, question: str, system: str | None = None) -> str:
    """Apply the R1-distill chat template so we get its thinking format."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _input_device(model):
    """Robust input placement device for device_map='auto' / sharded models."""
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        # HF puts the embed layer on the first device
        first = next(iter(model.hf_device_map.values()))
        return torch.device(first if isinstance(first, str) else f"cuda:{first}")
    return next(model.parameters()).device


def generate_with_optional_steering(model, tok, prompts: list[str],
                                    steer: SteeringHook | None,
                                    layer_idx: int | None,
                                    max_new_tokens: int = 512,
                                    do_sample: bool = False,
                                    temperature: float = 0.0,
                                    seed: int = 0) -> list[str]:
    device = _input_device(model)
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    prompt_len = int(enc["input_ids"].shape[1])
    if steer is not None and layer_idx is not None:
        steer.set_prompt_len(prompt_len)
        steer.attach(model, layer_idx)
    torch.manual_seed(seed)
    try:
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
    finally:
        if steer is not None:
            steer.detach()
    new_tokens = out[:, prompt_len:]
    return tok.batch_decode(new_tokens, skip_special_tokens=True)
