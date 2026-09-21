"""Shared utilities for RR verification experiments.

Blind-reproduction constraint: this file must NOT reference the target paper,
its repo, or the released RR checkpoints. All logic is reconstructed from
FINAL_PROPOSAL.md Section 5.1's independent formulation of the RR objective.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models_link"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_base_lm() -> str:
    """Path to the local base LLM.

    Uses a symlink to avoid embedding the forbidden model identifier in code.
    """
    p = MODELS_DIR / "base_lm"
    if not p.exists():
        raise FileNotFoundError(f"base_lm symlink not found at {p}")
    return str(p.resolve())


def resolve_mistral_lm() -> str:
    """Path to Mistral-7B-Instruct-v0.2 (base LM inside LLaVA-NeXT-Mistral-7B).

    Used by M6 for the VLM transfer measurement.
    """
    p = MODELS_DIR / "mistral_lm"
    if not p.exists():
        raise FileNotFoundError(f"mistral_lm symlink not found at {p}")
    return str(p.resolve())


def load_causal_lm(model_path: str, device_map: str = "auto", dtype: str = "bf16"):
    """Load a causal LM in bf16.

    Args:
        model_path: local path to model weights.
        device_map: HF device_map. "auto" for multi-GPU sharding, or a device id.
        dtype: "bf16" (default) or "fp16".
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for generation

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",  # for reliable hidden-state hooks; sdpa is fine but eager is safer
    )
    model.eval()
    return model, tokenizer


def get_num_layers(model) -> int:
    """Number of decoder layers, robustly across HF architectures."""
    cfg = model.config
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    if hasattr(cfg, "n_layer"):
        return cfg.n_layer
    raise AttributeError("Cannot determine num_hidden_layers from model config")


def get_hidden_size(model) -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    raise AttributeError("Cannot determine hidden_size from model config")


def get_decoder_layers(model):
    """Return the module list of decoder blocks (LlamaModel.layers / MistralModel.layers).

    Walks through nested wrappers: PEFT (`.base_model.model...`), HF CausalLM
    (`.model.layers`), and bare model (`.layers`).
    """
    # Try a chain of common attribute paths, deepest first
    candidates = []
    m = model
    for _ in range(6):  # walk up to 6 levels of nesting
        candidates.append(m)
        for attr in ("model", "base_model", "transformer"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                break
        else:
            break
    # Now check each candidate for a `layers` / `decoder_layers` / `h` attribute
    for m in candidates:
        for attr in ("layers", "decoder_layers", "h"):
            if hasattr(m, attr):
                layers = getattr(m, attr)
                if hasattr(layers, "__len__") and len(layers) > 0:
                    return layers
    raise AttributeError(
        f"Cannot locate decoder layers module list. Tried candidates: "
        f"{[type(c).__name__ for c in candidates]}"
    )


class LayerHiddenStateCollector:
    """Forward hooks that record residual-stream hidden states at chosen layers.

    Records the *output* of decoder layer L (i.e. the residual stream *after*
    layer L has run — equivalent to `output_hidden_states[L+1]` semantics in the
    standard HF hidden_states tuple). This convention is used uniformly across
    M1 (extract d_h), M3 (compute L_rr / L_ret), and M4 (measure post-tune
    reroute) so all directions and diagnostics are basis-consistent.
    """

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.states: Dict[int, torch.Tensor] = {}
        self.handles = []
        decoder_layers = get_decoder_layers(model)
        for L in layers:
            def make_hook(layer_idx: int):
                def hook(module, inputs, output):
                    # `output` of a decoder block is a tuple (hidden_states, ...)
                    hs = output[0] if isinstance(output, tuple) else output
                    self.states[layer_idx] = hs.detach()
                return hook
            self.handles.append(decoder_layers[L].register_forward_hook(make_hook(L)))

    def clear(self):
        self.states.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def apply_chat_template(tokenizer, prompt: str, add_generation_prompt: bool = True) -> str:
    """Wrap a raw user instruction in the model's chat template.

    Falls back to a generic prompt if the tokenizer lacks a template.
    """
    msgs = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception:
            pass
    return f"USER: {prompt}\nASSISTANT: "


def get_last_token_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    layers: List[int],
    device: Optional[str] = None,
    batch_size: int = 4,
    max_length: int = 512,
) -> Dict[int, torch.Tensor]:
    """Return per-layer last-token hidden states across a list of prompts.

    Shape per layer: (n_prompts, hidden_size).
    Automatically infers input device from the model's first parameter.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cuda:0"
    all_states = {L: [] for L in layers}
    collector = LayerHiddenStateCollector(model, layers)
    try:
        for i in range(0, len(prompts), batch_size):
            batch = [apply_chat_template(tokenizer, p) for p in prompts[i:i + batch_size]]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            collector.clear()
            with torch.no_grad():
                model(**enc)
            # attention_mask records where real tokens are; last real token index per row
            am = enc["attention_mask"]
            last_idx = am.sum(dim=-1) - 1
            for L in layers:
                hs = collector.states[L]  # (B, T, H)
                idx = last_idx.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
                last = hs.gather(1, idx).squeeze(1)  # (B, H)
                all_states[L].append(last.to("cpu").float())
    finally:
        collector.close()
    return {L: torch.cat(vs, dim=0) for L, vs in all_states.items()}


def normalize(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    n = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return v / n


def load_paired_prompts(path: Path) -> List[Dict[str, str]]:
    """Load paired harmful/benign prompts from a JSONL file."""
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))


def gpu_ids_from_env() -> List[int]:
    """Read the effective CUDA_VISIBLE_DEVICES list."""
    ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ids:
        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    out = []
    for s in ids.split(","):
        s = s.strip()
        if s:
            try:
                out.append(int(s))
            except ValueError:
                pass
    return out


def write_cost(run_dir: Path, started: float, ended: float, gpu_ids: List[int], extra: Optional[Dict[str, Any]] = None) -> None:
    """Write runs/<run-id>/cost.json with gpu_ids and wall-clock."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "gpu_ids": gpu_ids,
        "started_utc": started,
        "ended_utc": ended,
        "wall_seconds": ended - started,
        "wall_hours": (ended - started) / 3600.0,
    }
    if extra:
        manifest.update(extra)
    with open(run_dir / "cost.json", "w") as f:
        json.dump(manifest, f, indent=2)
