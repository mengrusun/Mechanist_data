"""
Shared library for Belief-Localization reproduction (task.md + EXPERIMENT_PLAN.md).

Provides:
    * BELIEF_CORE / BELIEF_HOLDOUT / MODEL_ROOTS / PYTHIA_CKPT_ROOT (frozen paths)
    * load_jsonl / load_frame / third_person_filter
    * load_model / MODEL_META lookup
    * loglikelihood_ratio_correct(...)   — the pinned log-prob metric
    * MODEL_META[model].{n_layers, n_heads, hidden, head_dim}
    * head_output_hook  — hook context that zero-ablates / scales attention heads
                          for GPTNeoX (works for pythia-*).
    * fisher_completion(...)  — per-parameter empirical Fisher signal via
                                 per-sample squared gradient of Σ_t log P(gold_t)
    * mask_top_target_notknowledge(fisher_target, fisher_knowledge, top_target, exclude_knowledge)
    * aggregate_head_mass(mask_by_pname, head_slices)
    * pile_ppl(model, tokenizer, ...)    — Pile-preshuffled PPL evaluator on a
                                            pinned deterministic slice
    * head_slices(model)                  — layer×head parameter slices for W_Q/W_K/W_V/W_O

Reproducibility rules honored:
    * completion-token-only log-prob (no length norm, no prompt-token sum)
    * tokenizer matched to the checkpoint being evaluated (each load fetches its own)
    * third-person-only filter for M2 Fisher / M2 head ablation / M3 causal
    * NO downscaling of models or data (resource_fidelity: strict)
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM

# ------------------------------------------------------------------
# Frozen paths
# ------------------------------------------------------------------

BELIEF_CORE_DIR = "/data/xuhaoming/belief_loc/data/derived/belief_core"
BELIEF_HOLDOUT_DIR = "/data/xuhaoming/belief_loc/data/derived/belief_holdout"
MODEL_ROOT = "/mnt/quarkfs/share_model/Ptyhia"
PILE_DIR = "/mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled"

FRAME_FILE = {
    "world_knowledge": "reality.jsonl",
    "personal_belief": "believe_truth.jsonl",
    "attributed_belief": "follow_belief.jsonl",
}
# Fisher-signal → frame → data file
SIGNAL_FRAME = {
    "F_personal": "personal_belief",
    "F_attributed": "attributed_belief",
    "F_knowledge": "world_knowledge",
}

# ------------------------------------------------------------------
# Model architecture metadata (verified from config.json)
# ------------------------------------------------------------------

@dataclass
class ModelMeta:
    n_layers: int
    n_heads: int
    hidden: int
    head_dim: int
    path_final: str
    ckpt_root: Optional[str] = None

MODEL_META: Dict[str, ModelMeta] = {
    "pythia-410m": ModelMeta(
        n_layers=24, n_heads=16, hidden=1024, head_dim=64,
        path_final=f"{MODEL_ROOT}/pythia-410m",
        ckpt_root=f"{MODEL_ROOT}/pythia-410m-checkpoints",
    ),
    "pythia-1b": ModelMeta(
        n_layers=16, n_heads=8, hidden=2048, head_dim=256,
        path_final=f"{MODEL_ROOT}/pythia-1b",
        ckpt_root=f"{MODEL_ROOT}/pythia-1b-checkpoints",
    ),
    "pythia-2.8b": ModelMeta(
        n_layers=32, n_heads=32, hidden=2560, head_dim=80,
        path_final=f"{MODEL_ROOT}/pythia-2.8b",
        ckpt_root=f"{MODEL_ROOT}/pythia-2.8b-checkpoints",
    ),
}

# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_frame(frame: str, root: str = BELIEF_CORE_DIR) -> List[dict]:
    """Load a full frame from a belief dataset root (belief_core or belief_holdout)."""
    fname = FRAME_FILE[frame]
    return load_jsonl(os.path.join(root, fname))


def third_person_filter(items: List[dict]) -> List[dict]:
    """Keep only third-person James/Mary examples (Claim-2 subset).
    world_knowledge (`reality.jsonl`) uses person = 'na' — pass through unchanged."""
    keep = []
    for it in items:
        p = str(it.get("person", "")).lower()
        if p == "na":
            keep.append(it)
        elif p in ("james", "mary"):
            keep.append(it)
    return keep


# ------------------------------------------------------------------
# Model loading (with checkpoint variant support)
# ------------------------------------------------------------------

def load_model(model_name: str, ckpt_step: Optional[int] = None,
               device: Optional[str] = None, dtype=torch.float16):
    """Load a Pythia final checkpoint or an intermediate checkpoint.
    ckpt_step: if given, load /pythia-<size>-checkpoints/step<ckpt_step>."""
    meta = MODEL_META[model_name]
    if ckpt_step is None:
        path = meta.path_final
    else:
        assert meta.ckpt_root is not None
        path = os.path.join(meta.ckpt_root, f"step{ckpt_step}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint at {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, tokenizer, meta


# ------------------------------------------------------------------
# Log-prob metric (task.md-pinned)
#
# correct iff Σ_t log P(gold_t | prompt, prev completion tokens)
#          > Σ_t log P(distractor_t | prompt, prev distractor tokens)
# * Sum log-probabilities over completion tokens only.
# * No length normalization.
# * Tokenizer matched to the checkpoint.
# * Identical prompt conditions for gold and distractor.
# ------------------------------------------------------------------

@torch.no_grad()
def completion_sum_logprob(model, tokenizer, prompt: str, completion: str,
                           device: str) -> float:
    """Σ_t log P(completion_t | prompt, completion_<t)  in nats.

    We tokenize (prompt + completion) as a single string using the same tokenizer,
    then measure the sum of per-position log-probs of the *completion* token
    positions only.

    The completion string in the dataset already begins with a space
    (e.g. " blue") so it tokenizes to a single-space-prefixed token that
    concatenates cleanly with the prompt.
    """
    p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    c_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full = torch.cat([p_ids, c_ids], dim=0).unsqueeze(0).to(device)  # [1, T]
    # Forward: logits at position t predict token at t+1
    out = model(full)
    logits = out.logits[0]  # [T, V]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    # Completion tokens live at positions [len(p_ids), len(full)-1] in the input,
    # so their predicting-logits live at positions [len(p_ids)-1, len(full)-2].
    Lp = len(p_ids)
    Lc = len(c_ids)
    # Sum log_probs[Lp-1 : Lp-1 + Lc] over target completion tokens
    completion_start_logit = Lp - 1
    total = 0.0
    for i in range(Lc):
        logit_pos = completion_start_logit + i
        tgt = full[0, Lp + i].item()
        total += log_probs[logit_pos, tgt].item()
    return total


def item_is_correct(model, tokenizer, prompt: str, gold: str,
                    distractor: str, device: str) -> Tuple[bool, float, float]:
    lp_gold = completion_sum_logprob(model, tokenizer, prompt, gold, device)
    lp_dist = completion_sum_logprob(model, tokenizer, prompt, distractor, device)
    return lp_gold > lp_dist, lp_gold, lp_dist


def evaluate_frame(model, tokenizer, items: List[dict], device: str,
                   show_progress: bool = True,
                   head_hook: Optional[Callable] = None) -> dict:
    """Return {n_items, n_correct, accuracy, per_item: [...]}.
    head_hook: optional callable that yields a torch.no_grad-safe context
    (e.g. head_output_hook.zero(model, head_set)) applied during the forward pass.
    """
    per_item = []
    n_correct = 0
    n_total = len(items)
    ctx = head_hook() if head_hook is not None else contextlib.nullcontext()
    with ctx:
        for i, it in enumerate(items):
            ok, lp_g, lp_d = item_is_correct(
                model, tokenizer, it["prompt"], it["gold"], it["distractor"], device
            )
            per_item.append({
                "idx": i,
                "prop_idx": it.get("prop_idx"),
                "person": it.get("person"),
                "category": it.get("category"),
                "logp_gold": lp_g,
                "logp_distractor": lp_d,
                "correct": bool(ok),
            })
            n_correct += int(ok)
            if show_progress and (i + 1) % 100 == 0:
                print(f"  eval {i+1}/{n_total}  acc={n_correct / (i+1):.3f}", flush=True)
    return {
        "n_items": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "per_item": per_item,
    }


def binomial_pvalue_above_chance(k: int, n: int, p0: float = 0.5) -> float:
    """One-sided P(X >= k | Binomial(n, 0.5))."""
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, p0))


# ------------------------------------------------------------------
# GPTNeoX head slicing helpers
#
# In GPTNeoX (Pythia), the attention block has one fused
# `attention.query_key_value` linear:
#     out_features = 3 * n_heads * head_dim  (interleaved [q_h0, k_h0, v_h0, q_h1, k_h1, v_h1, ...])
#
# Actually in HuggingFace GPTNeoX the layout is:
#     qkv reshape ->  [B, T, n_heads, 3, head_dim]
# which means the fused output tensor is arranged as
#     [q_h0 (head_dim), k_h0 (head_dim), v_h0 (head_dim),
#      q_h1 (head_dim), k_h1 (head_dim), v_h1 (head_dim), ...]
# per row of the (out_features, hidden) weight matrix.
#
# `attention.dense` is Linear(in=hidden, out=hidden), and per-head columns are
# [h * head_dim : (h + 1) * head_dim] in the `in_features` dim (columns of W_O).
# ------------------------------------------------------------------

def qkv_row_indices(head_idx: int, head_dim: int, which: str) -> Tuple[int, int]:
    """Return (row_start, row_end) in the fused QKV weight/bias for this head+role."""
    which_off = {"q": 0, "k": 1, "v": 2}[which]
    base = head_idx * 3 * head_dim + which_off * head_dim
    return base, base + head_dim


def head_slices_for_model(model_name: str) -> Dict[Tuple[int, int], Dict[str, Tuple[str, Tuple[int, int, str]]]]:
    """
    For each (layer, head), give the parameter-index slices it owns.
    Returned as dict[(L, H)] = {
        'qkv_q': ('gpt_neox.layers.<L>.attention.query_key_value.weight', (row_start, row_end, 'rows')),
        'qkv_k': same,
        'qkv_v': same,
        'qkv_q_bias': ('gpt_neox.layers.<L>.attention.query_key_value.bias', (row_start, row_end, 'rows')),
        ...
        'dense': ('gpt_neox.layers.<L>.attention.dense.weight', (col_start, col_end, 'cols')),
    }
    """
    meta = MODEL_META[model_name]
    hd = meta.head_dim
    out = {}
    for L in range(meta.n_layers):
        for H in range(meta.n_heads):
            qs, qe = qkv_row_indices(H, hd, "q")
            ks, ke = qkv_row_indices(H, hd, "k")
            vs, ve = qkv_row_indices(H, hd, "v")
            ds, de = H * hd, (H + 1) * hd  # W_O columns for head H
            base = f"gpt_neox.layers.{L}.attention"
            out[(L, H)] = {
                "qkv_q_w": (f"{base}.query_key_value.weight", (qs, qe, "rows")),
                "qkv_q_b": (f"{base}.query_key_value.bias",   (qs, qe, "rows")),
                "qkv_k_w": (f"{base}.query_key_value.weight", (ks, ke, "rows")),
                "qkv_k_b": (f"{base}.query_key_value.bias",   (ks, ke, "rows")),
                "qkv_v_w": (f"{base}.query_key_value.weight", (vs, ve, "rows")),
                "qkv_v_b": (f"{base}.query_key_value.bias",   (vs, ve, "rows")),
                "dense_w": (f"{base}.dense.weight",           (ds, de, "cols")),
            }
    return out


# ------------------------------------------------------------------
# Attention-head zero-ablation / amplification via forward hooks.
# We patch the fused Q/K/V output of the head to be zero during attention,
# equivalent to zeroing that head's contribution to the residual stream.
#
# Concretely, the cleanest place to intervene is at the *output* of the
# attention module for the specific head. In GPTNeoX, the attention output
# per head is written into `attn_output` (or equivalent) and then fed through
# `attention.dense`. We hook `attention.dense` and modify its *input* along
# the [head * head_dim : (head+1) * head_dim] columns of the flattened head
# dim before it goes through the linear projection.
# ------------------------------------------------------------------

class HeadInterventionContext:
    """Context manager registering forward-pre-hooks on each layer's
    `attention.dense` module. During the forward pass, for each targeted
    (layer, head), the input slice of size head_dim (contiguous columns
    [head*head_dim : (head+1)*head_dim] in the last dim) is either
    zeroed (scale = 0.0) or scaled multiplicatively.

    Usage:
        with HeadInterventionContext(model, {(L, H): 0.0, ...}):
            evaluate_frame(...)

    For amplification:
        with HeadInterventionContext(model, {(L, H): 3.0, ...}):
            ...

    Any layer not present in the dict uses identity (scale = 1.0).
    """

    def __init__(self, model, head_scale: Dict[Tuple[int, int], float]):
        self.model = model
        self.head_scale = dict(head_scale)  # (L, H) -> scale
        self.handles = []
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads

    def _make_hook(self, layer_idx: int, per_layer_scales: Dict[int, float]):
        hd = self.head_dim
        head_indices = sorted(per_layer_scales.keys())

        def pre_hook(module, args):
            # Clone the input tensor before modifying — avoid aliasing / autograd
            # surprises even under torch.no_grad; return the cloned tensor.
            x = args[0]
            x_mod = x.clone()
            for h in head_indices:
                s = per_layer_scales[h]
                if s == 1.0:
                    continue
                start = h * hd
                end = (h + 1) * hd
                if s == 0.0:
                    x_mod[..., start:end] = 0.0
                else:
                    x_mod[..., start:end].mul_(s)
            return (x_mod,) + args[1:]

        return pre_hook

    def __enter__(self):
        # Group scales per layer
        per_layer: Dict[int, Dict[int, float]] = {}
        for (L, H), s in self.head_scale.items():
            per_layer.setdefault(L, {})[H] = s
        for L, dh in per_layer.items():
            dense_mod = self.model.get_submodule(f"gpt_neox.layers.{L}.attention.dense")
            h = dense_mod.register_forward_pre_hook(self._make_hook(L, dh))
            self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


def make_head_zero_ablation(model, head_set: Sequence[Tuple[int, int]]):
    """Convenience: build a HeadInterventionContext for zero-ablation."""
    return HeadInterventionContext(model, {tuple(h): 0.0 for h in head_set})


def make_head_amplification(model, head_scale: Dict[Tuple[int, int], float]):
    return HeadInterventionContext(model, head_scale)


# ------------------------------------------------------------------
# Empirical Fisher over parameters (diagonal, per-parameter)
#
# F_p = (1/N) Σ_i (∂ Σ_t log P(gold_t | prompt_i, prev completion) / ∂ p)^2
# Batch-size-1, fp32 accumulator.
#
# Only compute grads for `.attention.query_key_value.weight` and
# `.attention.dense.weight` — the attention-head parameters that head-slicing
# needs. All other params get requires_grad = False to keep memory bounded.
# ------------------------------------------------------------------

def fisher_completion_over_frame(model, tokenizer, items: List[dict],
                                 device: str) -> Dict[str, torch.Tensor]:
    """Compute diagonal empirical Fisher for the completion sum-log-prob
    objective, averaged over items.  Returns dict {pname: F_tensor} where
    F_tensor is the per-parameter mean-squared-gradient, computed only for
    attention weight matrices (Q_K_V and dense) across all layers.

    Fp32 accumulator.  Per-sample, no batching, no length normalization.
    """
    # Save original requires_grad so we can restore before returning
    orig_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    try:
        keep_pnames = []
        for name, p in model.named_parameters():
            want = (
                (".attention.query_key_value.weight" in name)
                or (".attention.dense.weight" in name)
            )
            p.requires_grad_(want)
            if want:
                keep_pnames.append(name)

        # Fresh dict of fp32 accumulators
        fisher: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in keep_pnames:
                fisher[name] = torch.zeros_like(p, dtype=torch.float32, device=device)

        n_seen = 0
        total_items = len(items)
        for i, it in enumerate(items):
            prompt, gold = it["prompt"], it["gold"]
            p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
            c_ids = tokenizer(gold, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(p_ids) == 0 or len(c_ids) == 0:
                # skip malformed items
                continue
            full = torch.cat([p_ids, c_ids], dim=0).unsqueeze(0).to(device)
            Lp = len(p_ids); Lc = len(c_ids)
            model.zero_grad(set_to_none=True)
            out = model(full)
            logits = out.logits[0]  # [T, V]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            # positions [Lp-1 .. Lp-2+Lc] predict tokens [Lp .. Lp-1+Lc]
            gather_idx = full[0, Lp : Lp + Lc]
            pos_range = torch.arange(Lp - 1, Lp - 1 + Lc, device=device)
            sum_lp = log_probs[pos_range, gather_idx].sum()
            (-sum_lp).backward()  # sign is irrelevant for the squared-grad Fisher
            for name, p in model.named_parameters():
                if p.grad is not None and name in fisher:
                    fisher[name] += p.grad.detach().float().pow_(2)
            n_seen += 1
            if (i + 1) % 50 == 0 or i + 1 == total_items:
                print(f"  fisher {i+1}/{total_items}", flush=True)

        for name in fisher:
            fisher[name] /= max(n_seen, 1)

        return fisher
    finally:
        for name, p in model.named_parameters():
            p.requires_grad_(orig_requires_grad[name])
        model.zero_grad(set_to_none=True)


# ------------------------------------------------------------------
# Mask construction
# ------------------------------------------------------------------

def flatten_dict_topk_threshold(fisher: Dict[str, torch.Tensor], top_frac: float) -> float:
    """Given per-param Fisher dict, find the value threshold above which
    the top `top_frac` fraction of parameters lies (globally over all
    included tensors)."""
    all_vals = torch.cat([v.flatten() for v in fisher.values()]).cpu()
    n = all_vals.numel()
    k = max(1, int(round(top_frac * n)))
    # topk value threshold: kth largest
    thr = torch.topk(all_vals, k, largest=True).values.min().item()
    return thr


def build_target_mask(f_target: Dict[str, torch.Tensor],
                     f_knowledge: Dict[str, torch.Tensor],
                     top_target_frac: float,
                     exclude_knowledge_frac: float) -> Dict[str, torch.Tensor]:
    """Mask_target = (top-`top_target_frac` of F_target) AND NOT (top-`exclude_knowledge_frac` of F_knowledge).
    Returns dict of bool tensors, one per parameter."""
    assert set(f_target.keys()) == set(f_knowledge.keys()), "Fisher dicts have different keys"
    thr_target = flatten_dict_topk_threshold(f_target, top_target_frac)
    thr_knowledge = flatten_dict_topk_threshold(f_knowledge, exclude_knowledge_frac)
    masks = {}
    for name in f_target:
        top_t = f_target[name] >= thr_target
        top_k = f_knowledge[name] >= thr_knowledge
        masks[name] = top_t & (~top_k)
    return masks


def aggregate_head_mass(masks: Dict[str, torch.Tensor], model_name: str) -> Dict[Tuple[int, int], int]:
    """For each attention head (L, H), sum the count of masked parameters
    across that head's Q, K, V, dense slices.  Returns dict[(L, H)] = int."""
    slices = head_slices_for_model(model_name)
    total = {(L, H): 0 for (L, H) in slices}
    for (L, H), roles in slices.items():
        for role_key, (pname, (a, b, axis)) in roles.items():
            if role_key.endswith("_b"):
                # bias tensors are not touched by weight-space Fisher masks
                continue
            m = masks.get(pname)
            if m is None:
                continue
            if axis == "rows":
                total[(L, H)] += int(m[a:b].sum().item())
            else:  # 'cols'
                total[(L, H)] += int(m[:, a:b].sum().item())
    return total


# ------------------------------------------------------------------
# Pile PPL evaluator (deterministic slice)
# ------------------------------------------------------------------

def _load_pile_windows(doc_indices: Sequence[int], n_windows: int,
                     ctx_len: int = 2048) -> torch.Tensor:
    """Return int64 tensor of shape [n_windows, ctx_len] with tokens from the
    first `n_windows` non-overlapping windows across the specified documents."""
    windows = []
    tokens_needed = n_windows * ctx_len
    for d in doc_indices:
        pth = os.path.join(PILE_DIR, f"document-{d:05d}-of-00020.bin")
        arr = np.memmap(pth, dtype=np.uint16, mode="r")
        n_avail = arr.shape[0] // ctx_len
        take = min(n_avail, (tokens_needed - len(windows) * ctx_len) // ctx_len)
        if take <= 0:
            break
        for w in range(take):
            windows.append(arr[w * ctx_len : (w + 1) * ctx_len].astype(np.int64))
            if len(windows) >= n_windows:
                break
        if len(windows) >= n_windows:
            break
    tok = torch.from_numpy(np.stack(windows[:n_windows], axis=0))
    return tok


@torch.no_grad()
def pile_ppl(model, tokenizer, doc_indices: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
             n_windows: int = 5000, ctx_len: int = 2048, device: str = "cuda",
             head_hook_ctx: Optional[Callable] = None, batch_size: int = 1) -> float:
    """Deterministic Pile PPL over `n_windows` non-overlapping ctx_len-token
    windows drawn from documents `doc_indices`.

    head_hook_ctx: optional callable that returns a context manager to apply
                   during forward passes (e.g. `lambda: make_head_zero_ablation(model, [...]).__enter__()`).
                   The simpler pattern used by callers is to wrap this function
                   inside their own context manager.
    """
    windows = _load_pile_windows(doc_indices, n_windows, ctx_len)  # [W, T]
    W = windows.shape[0]
    total_nll = 0.0
    total_toks = 0
    ctx = head_hook_ctx() if head_hook_ctx is not None else contextlib.nullcontext()
    with ctx:
        for i in range(0, W, batch_size):
            batch = windows[i : i + batch_size].to(device)
            out = model(batch)
            logits = out.logits[:, :-1, :].float()   # [B, T-1, V]
            targets = batch[:, 1:]                    # [B, T-1]
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_toks += targets.numel()
            if (i // batch_size) % 50 == 0:
                print(f"  pile {i+batch_size}/{W}  running_ppl={math.exp(total_nll/max(1,total_toks)):.3f}", flush=True)
    return math.exp(total_nll / total_toks)


# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------

def parse_seed_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
