"""
Shared utilities for the belief-circuits reproduction experiments.

Covers:
    - dataset loading with the belief_core / belief_holdout jsonl schema
    - model / tokenizer loading (HuggingFace pythia + step-checkpoint dir)
    - log-prob comparison metric (correctness bit per (prompt, gold, distractor))
    - Wilson 95% CI for a binomial proportion
    - GPTNeoX per-attention-head parameter slicing over the fused query_key_value matrix
    - Zero-ablation forward-hook installer / remover (multiplicative scaling α; α=0 = knockout, α=1 = identity)
    - PPL evaluation on a cached Pile token tensor

All numeric work uses torch. Prompts are tokenized without adding special tokens (Pythia is
a causal LM; log-probs of the gold/distractor continuation start immediately after the
prompt). Both `gold` and `distractor` fields in the dataset carry their own leading space
so we do NOT strip whitespace before tokenizing them.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM


# --------------------------------------------------------------------------- #
# Dataset I/O
# --------------------------------------------------------------------------- #

TASK_FILE = {
    "world_knowledge": "reality.jsonl",
    "personal_belief": "believe_truth.jsonl",
    "attributed_belief": "follow_belief.jsonl",
}


@dataclass
class BeliefExample:
    prop_idx: int
    category: str
    frame: str
    person: str
    prompt: str
    gold: str
    distractor: str

    @classmethod
    def from_dict(cls, d: dict) -> "BeliefExample":
        return cls(
            prop_idx=d["prop_idx"],
            category=d["category"],
            frame=d["frame"],
            person=d["person"],
            prompt=d["prompt"],
            gold=d["gold"],
            distractor=d["distractor"],
        )


def load_task(data_root: str, task: str, person_filter: Optional[Sequence[str]] = None) -> List[BeliefExample]:
    """Load one belief task's jsonl file, optionally filtering by lowercase `person` field."""
    fn = TASK_FILE[task]
    p = Path(data_root) / fn
    out: List[BeliefExample] = []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            ex = BeliefExample.from_dict(d)
            if person_filter is not None and ex.person not in person_filter:
                continue
            out.append(ex)
    return out


# --------------------------------------------------------------------------- #
# Model / tokenizer loading
# --------------------------------------------------------------------------- #

def load_model_and_tokenizer(
    model_root: str,
    model: str,
    dtype: str = "fp16",
    device: str = "cuda",
    checkpoint_dir: Optional[str] = None,
) -> Tuple[GPTNeoXForCausalLM, AutoTokenizer]:
    """Load a pythia model + tokenizer.

    Args:
        model_root: root that contains `pythia-410m`, `pythia-1b`, `pythia-2.8b`, etc.
        model: e.g. `pythia-1b`.
        dtype: `fp16` | `fp32` | `bf16`.
        device: `cuda` | `cuda:0` | `cpu`.
        checkpoint_dir: absolute path to an intermediate checkpoint (overrides `model` for weight load;
            tokenizer still comes from the full-model dir).

    Returns: (model, tokenizer). `model` is set to eval mode and moved to `device`.
    """
    full_model_dir = Path(model_root) / model
    weights_dir = Path(checkpoint_dir) if checkpoint_dir is not None else full_model_dir
    tokenizer = AutoTokenizer.from_pretrained(str(full_model_dir))
    torch_dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    net = AutoModelForCausalLM.from_pretrained(
        str(weights_dir),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    net.eval()
    net.to(device)
    return net, tokenizer


def model_arch_info(net: GPTNeoXForCausalLM) -> dict:
    """Return (n_layers, n_heads, hidden_size, head_dim, vocab_size)."""
    cfg = net.config
    return {
        "n_layers": cfg.num_hidden_layers,
        "n_heads": cfg.num_attention_heads,
        "hidden_size": cfg.hidden_size,
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "vocab_size": cfg.vocab_size,
    }


# --------------------------------------------------------------------------- #
# Log-prob comparison metric (correctness bit)
# --------------------------------------------------------------------------- #

def continuation_logprob(
    net: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    continuation: str,
    device: str = "cuda",
) -> float:
    """Return Σ_t log P(y_t | x, y_<t) for continuation y under the model.

    Notes:
      - We tokenize prompt and continuation separately (no BOS added — Pythia's tokenizer
        has add_bos=False by default). Then we form the concatenation and score the
        continuation positions only. This is the standard LM-eval-harness `loglikelihood` idiom.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    assert len(cont_ids) >= 1, f"empty continuation tokenization for {continuation!r}"

    input_ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = net(input_ids=input_ids).logits[0]  # [T, V]
    # Position i predicts token i+1; the continuation tokens are at positions
    # [len(prompt_ids), ..., len(prompt_ids) + len(cont_ids) - 1]. Prediction for token
    # at position p uses logits[p-1]. So we take logits[len(prompt_ids)-1 : -1] as the
    # distribution over each continuation token.
    p0 = len(prompt_ids) - 1
    slice_logits = logits[p0 : p0 + len(cont_ids)]  # [n_cont, V]
    lp = F.log_softmax(slice_logits.float(), dim=-1)
    tgt = torch.tensor(cont_ids, dtype=torch.long, device=device)
    tok_lp = lp.gather(1, tgt[:, None]).squeeze(1)
    return float(tok_lp.sum().item())


def continuation_logprob_batch(
    net: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    continuations: Sequence[str],
    device: str = "cuda",
) -> List[float]:
    """Score multiple continuations under the same prompt.

    Implementation note: we deliberately do NOT share a past_key_values cache across
    continuations. transformers ≥ 4.36's `Cache` class is stateful — reusing the same pkv
    for a second forward mutates it in-place (silently corrupting subsequent scores). For
    the prompt+2-continuation workload of belief evaluation the overhead of re-encoding the
    prompt for each continuation is negligible.
    """
    lps: List[float] = []
    for cont in continuations:
        lps.append(continuation_logprob(net, tokenizer, prompt, cont, device=device))
    return lps


def evaluate_task_accuracy(
    net: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    examples: Sequence[BeliefExample],
    device: str = "cuda",
) -> dict:
    """Return {acc, correct_count, total, wilson_ci_low, wilson_ci_high, per_example}.

    Correctness rule (verbatim from task.md):
        correct ⇔ Σ_t log P(y⁺_t | x, y⁺_<t) > Σ_t log P(y⁻_t | x, y⁻_<t)
    Ties resolved as INCORRECT (strict >).
    """
    correct = 0
    per_ex: List[dict] = []
    for ex in examples:
        lps = continuation_logprob_batch(net, tokenizer, ex.prompt, [ex.gold, ex.distractor], device=device)
        lp_gold, lp_dist = lps
        ok = int(lp_gold > lp_dist)
        correct += ok
        per_ex.append({
            "prop_idx": ex.prop_idx,
            "person": ex.person,
            "category": ex.category,
            "lp_gold": lp_gold,
            "lp_distractor": lp_dist,
            "correct": ok,
        })
    total = len(examples)
    acc = correct / total if total > 0 else 0.0
    lo, hi = wilson_ci(correct, total)
    return {
        "acc": acc,
        "correct_count": correct,
        "total": total,
        "wilson_ci_low": lo,
        "wilson_ci_high": hi,
        "per_example": per_ex,
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion at confidence 1-2·Φ(-z)."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    centre = (phat + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- #
# Attention head slicing (GPTNeoX fused query_key_value)
# --------------------------------------------------------------------------- #
# GPTNeoXAttention uses a fused query_key_value linear whose OUTPUT layout is
#     [n_heads, 3, head_dim]  (after `._split_heads`)
# which means the raw weight matrix has shape
#     [3 * hidden_size, hidden_size]
# with the OUTPUT axis grouped as
#     for h in range(n_heads):
#         W_Q^h rows: indices [h*3*d + 0*d : h*3*d + 1*d]
#         W_K^h rows: indices [h*3*d + 1*d : h*3*d + 2*d]
#         W_V^h rows: indices [h*3*d + 2*d : h*3*d + 3*d]
# where d = head_dim. The bias vector `query_key_value.bias` follows the same layout on
# axis 0.
#
# The attention output projection `dense` has shape [hidden_size, hidden_size]. Its INPUT
# axis is grouped as `[n_heads, head_dim]`, so head h's output-projection columns are
#     dense.weight[:, h*d : (h+1)*d]
# The bias `dense.bias` is a full hidden_size vector shared across all heads (it is the
# constant added AFTER summing all heads' contributions), so we cannot cleanly attribute
# any part of it to a single head; we exclude the dense bias from per-head aggregation.


def gpt_neox_head_param_slices(net: GPTNeoXForCausalLM) -> dict:
    """Return a dict keyed by (layer_idx, head_idx) → {qkv_weight_rows, qkv_bias_rows, dense_weight_cols}.

    Each value is a dict of tensor-slice specifiers (as (start, end) tuples on the affected axis)
    identifying which parameters belong to head (l, h). No bias on `dense` is included.
    """
    info = model_arch_info(net)
    n_layers, n_heads, d = info["n_layers"], info["n_heads"], info["head_dim"]
    slices = {}
    for l in range(n_layers):
        for h in range(n_heads):
            q_start = h * 3 * d + 0 * d
            k_start = h * 3 * d + 1 * d
            v_start = h * 3 * d + 2 * d
            slices[(l, h)] = {
                "layer": l,
                "head": h,
                # rows on axis-0 of query_key_value.weight (and bias)
                "qkv_q_rows": (q_start, q_start + d),
                "qkv_k_rows": (k_start, k_start + d),
                "qkv_v_rows": (v_start, v_start + d),
                # cols on axis-1 of dense.weight
                "dense_cols": (h * d, (h + 1) * d),
            }
    return slices


def head_param_names(layer_idx: int) -> dict:
    """Canonical parameter names on a GPT-NeoX pythia model for the attention block."""
    prefix = f"gpt_neox.layers.{layer_idx}.attention"
    return {
        "qkv_weight": f"{prefix}.query_key_value.weight",
        "qkv_bias": f"{prefix}.query_key_value.bias",
        "dense_weight": f"{prefix}.dense.weight",
        "dense_bias": f"{prefix}.dense.bias",
    }


def head_param_indices_flat(net: GPTNeoXForCausalLM, layer: int, head: int) -> dict:
    """Return the flat-index sets on each of {qkv.weight, qkv.bias, dense.weight} that
    correspond to attention head (layer, head). Used for per-head Fisher aggregation and
    for random-mask control at parameter-level.

    Returns a dict:
        {
            "qkv_weight": (name, torch.LongTensor of flat indices),
            "qkv_bias":   (name, torch.LongTensor of flat indices),
            "dense_weight": (name, torch.LongTensor of flat indices),
        }
    """
    info = model_arch_info(net)
    d = info["head_dim"]
    hidden = info["hidden_size"]
    names = head_param_names(layer)

    # query_key_value.weight is [3*hidden, hidden]. Head h occupies rows [h*3*d + k*d : ...] for k in {0,1,2}.
    q_start, k_start, v_start = head * 3 * d, head * 3 * d + d, head * 3 * d + 2 * d
    qkv_rows = []
    for start in (q_start, k_start, v_start):
        qkv_rows.extend(range(start, start + d))
    qkv_rows_t = torch.tensor(qkv_rows, dtype=torch.long)
    # flatten row-major: row r, col c → r*hidden + c
    qkv_w_flat = (qkv_rows_t.unsqueeze(1) * hidden + torch.arange(hidden).unsqueeze(0)).reshape(-1)
    qkv_b_flat = qkv_rows_t

    # dense.weight is [hidden, hidden]. Head h occupies cols [h*d : (h+1)*d].
    d_cols = torch.arange(head * d, (head + 1) * d, dtype=torch.long)
    dense_w_flat = (torch.arange(hidden).unsqueeze(1) * hidden + d_cols.unsqueeze(0)).reshape(-1)

    return {
        "qkv_weight": (names["qkv_weight"], qkv_w_flat),
        "qkv_bias": (names["qkv_bias"], qkv_b_flat),
        "dense_weight": (names["dense_weight"], dense_w_flat),
    }


# --------------------------------------------------------------------------- #
# Attention-head knockout via forward-hook
# --------------------------------------------------------------------------- #

def install_head_scaling_hooks(
    net: GPTNeoXForCausalLM,
    scale_map: dict,
) -> List:
    """Install forward hooks on GPTNeoX attention blocks to scale specific heads' output
    contribution BEFORE the residual add.

    Args:
        scale_map: dict {(layer_idx, head_idx): scale}. scale=0.0 zeros the head (knockout),
            scale=1.0 is identity, scale > 1.0 amplifies. Any (l, h) not in the map is
            left untouched (implicit scale=1.0).

    Returns: list of hook handles (call .remove() on each to uninstall).

    Implementation notes:
      GPTNeoXAttention.forward returns `attn_output` = (batch, seq_len, hidden_size), where
      hidden_size = n_heads * head_dim. The head axis is contiguous: head h occupies
      channels [h*head_dim : (h+1)*head_dim]. `attn_output` is the OUTPUT of the `dense`
      linear projection — i.e., ALL heads have already been merged. So a per-head scaling
      cannot be applied to the post-`dense` tensor directly.

      Instead, we hook the `dense` module: intercept its INPUT (which is the concatenated
      per-head outputs pre-projection) and scale the head slice there. This is
      mathematically equivalent to zeroing the head's contribution to the residual:
      `dense(x + h_head) - dense(x)` decomposes linearly through the head's column block.

      Concretely, `dense.weight` has shape [hidden, hidden] and the INPUT axis (cols)
      groups per-head. Scaling the input at cols [h*d:(h+1)*d] by α scales that head's
      contribution to the output by α, then the sum-over-heads-plus-dense-bias proceeds
      normally.
    """
    info = model_arch_info(net)
    d = info["head_dim"]
    # group heads to scale by layer
    by_layer: dict = {}
    for (l, h), s in scale_map.items():
        by_layer.setdefault(l, {})[h] = float(s)

    handles = []
    for layer_idx, head_scales in by_layer.items():
        attn_layer = net.gpt_neox.layers[layer_idx].attention
        dense = attn_layer.dense

        # capture into closure
        def make_pre_hook(head_scales_local, head_dim=d):
            def _hook(module, args, kwargs):
                # args = (input_tensor,); kwargs typically empty
                if len(args) == 0:
                    x = kwargs.get("input", None)
                    assert x is not None, "unexpected dense hook signature"
                    kwargs = dict(kwargs)
                    for h, s in head_scales_local.items():
                        if s == 1.0:
                            continue
                        x = x.clone()
                        x[..., h * head_dim : (h + 1) * head_dim] = x[..., h * head_dim : (h + 1) * head_dim] * s
                    kwargs["input"] = x
                    return args, kwargs
                x = args[0]
                # only clone if we actually modify
                if any(s != 1.0 for s in head_scales_local.values()):
                    x = x.clone()
                    for h, s in head_scales_local.items():
                        if s == 1.0:
                            continue
                        x[..., h * head_dim : (h + 1) * head_dim] = x[..., h * head_dim : (h + 1) * head_dim] * s
                    return (x,) + args[1:], kwargs
                return None
            return _hook

        h = dense.register_forward_pre_hook(make_pre_hook(head_scales), with_kwargs=True)
        handles.append(h)
    return handles


def remove_hooks(handles: Iterable) -> None:
    for h in handles:
        h.remove()


# --------------------------------------------------------------------------- #
# Parameter-level zero-mask (for random-mask control)
# --------------------------------------------------------------------------- #

def apply_param_zero_mask(
    net: GPTNeoXForCausalLM,
    mask: dict,
) -> dict:
    """Zero out specific parameter positions in-place. Returns snapshot dict of the
    original values so `restore_params(net, snapshot)` can undo the change.

    Args:
        mask: dict {parameter_name: torch.LongTensor of flat indices to zero}

    Returns: dict {parameter_name: (flat_indices, original_values)}
    """
    snap: dict = {}
    with torch.no_grad():
        for pname, idx in mask.items():
            p = dict(net.named_parameters())[pname]
            flat = p.data.reshape(-1)
            idx = idx.to(flat.device)
            orig = flat[idx].clone()
            flat[idx] = 0
            snap[pname] = (idx, orig)
    return snap


def restore_params(net: GPTNeoXForCausalLM, snapshot: dict) -> None:
    with torch.no_grad():
        for pname, (idx, orig) in snapshot.items():
            p = dict(net.named_parameters())[pname]
            flat = p.data.reshape(-1)
            flat[idx.to(flat.device)] = orig.to(flat.device)


# --------------------------------------------------------------------------- #
# Pile PPL evaluation
# --------------------------------------------------------------------------- #

def build_pile_ppl_sample(pile_dir: str, n_tokens: int, cache_path: str) -> str:
    """Read up to `n_tokens` uint16 token ids from the first shard of `pile_dir`, then
    save as a torch tensor at `cache_path`. Returns the absolute cache path.

    The pretokenized Pile shards under /mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled/
    are uint16 arrays (vocab ≤ 65535). We only need a contiguous 1M-token slice — the file is
    already pre-shuffled and pretokenized for pythia, so a head-slice from shard-00000 is
    a valid random sample of pretraining text under the shuffling seed used to build it.
    """
    cache_path = os.path.abspath(cache_path)
    if os.path.exists(cache_path):
        try:
            t = torch.load(cache_path)
            if t.numel() >= n_tokens:
                return cache_path
        except Exception:
            pass
    shard_files = sorted(f for f in os.listdir(pile_dir) if f.endswith(".bin"))
    assert len(shard_files) > 0, f"no .bin shards in {pile_dir}"
    tokens = np.memmap(os.path.join(pile_dir, shard_files[0]), dtype=np.uint16, mode="r")
    if tokens.shape[0] < n_tokens:
        raise RuntimeError(f"shard has {tokens.shape[0]} tokens < requested {n_tokens}")
    slice_ = np.asarray(tokens[:n_tokens], dtype=np.int64).copy()
    t = torch.from_numpy(slice_)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(t, cache_path)
    return cache_path


def evaluate_ppl(
    net: GPTNeoXForCausalLM,
    token_tensor: torch.Tensor,
    window: int = 1024,
    device: str = "cuda",
) -> dict:
    """Sliding-window causal LM perplexity on a 1D int tensor of token ids.

    Uses non-overlapping windows (fastest; slight boundary bias, but consistent across
    ablated vs clean runs because we always split at the same positions).
    Returns {ppl, avg_loss, n_tokens}.
    """
    assert token_tensor.ndim == 1
    total_nll = 0.0
    total_n = 0
    with torch.no_grad():
        for i in range(0, token_tensor.numel() - 1, window):
            chunk = token_tensor[i : i + window]
            if chunk.numel() < 2:
                break
            input_ids = chunk.unsqueeze(0).to(device)
            logits = net(input_ids=input_ids).logits[0]  # [T, V]
            # cross-entropy of next-token prediction
            shift_logits = logits[:-1].float()
            shift_labels = input_ids[0, 1:]
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
            total_nll += float(loss.item())
            total_n += int(shift_labels.numel())
    avg = total_nll / max(1, total_n)
    return {"ppl": math.exp(avg), "avg_loss": avg, "n_tokens": total_n}


# --------------------------------------------------------------------------- #
# Random control samplers
# --------------------------------------------------------------------------- #

def sample_random_head_subset(net: GPTNeoXForCausalLM, k: int, seed: int) -> List[Tuple[int, int]]:
    """Return a list of k (layer, head) tuples sampled uniformly without replacement over
    all attention heads. `seed` fully determines the draw."""
    info = model_arch_info(net)
    all_heads = [(l, h) for l in range(info["n_layers"]) for h in range(info["n_heads"])]
    rng = random.Random(seed)
    rng.shuffle(all_heads)
    return all_heads[:k]


def sample_random_param_mask(net: GPTNeoXForCausalLM, hstar_heads: Sequence[Tuple[int, int]], seed: int) -> dict:
    """Draw a uniformly-random parameter subset of the SAME total parameter count as the
    union of {W_Q, W_K, W_V, W_O} parameters across the heads in `hstar_heads`.

    Sampling is over the same three parameter families the heads use:
        gpt_neox.layers.*.attention.query_key_value.weight
        gpt_neox.layers.*.attention.query_key_value.bias
        gpt_neox.layers.*.attention.dense.weight
    (No dense.bias for the same reason it is excluded from head_param_indices_flat.)

    Returns: dict {parameter_name: flat_indices tensor}
    """
    # count target parameters
    target_count = 0
    for (l, h) in hstar_heads:
        pidx = head_param_indices_flat(net, l, h)
        for _, (_, idx) in pidx.items():
            target_count += int(idx.numel())

    # enumerate candidate parameters + their flat ranges
    info = model_arch_info(net)
    candidate_families: List[Tuple[str, int]] = []  # (param_name, numel)
    for l in range(info["n_layers"]):
        names = head_param_names(l)
        for key in ("qkv_weight", "qkv_bias", "dense_weight"):
            pname = names[{"qkv_weight": "qkv_weight", "qkv_bias": "qkv_bias", "dense_weight": "dense_weight"}[key]]
            candidate_families.append((pname, dict(net.named_parameters())[pname].numel()))
    total_available = sum(n for _, n in candidate_families)
    # global flat index in the concatenation
    rng = np.random.RandomState(seed)
    # draw target_count unique indices in [0, total_available)
    idx_global = rng.choice(total_available, size=target_count, replace=False)
    idx_global.sort()

    # map back to (param_name, local_flat_index) — vectorized so large models
    # don't hit an 8M-element Python loop.
    offsets = np.cumsum([n for _, n in candidate_families])
    starts = np.concatenate([[0], offsets[:-1]])
    j_arr = np.searchsorted(offsets, idx_global, side="right")
    local_arr = idx_global - starts[j_arr]
    mask: dict = {}
    for j in range(len(candidate_families)):
        sel = np.where(j_arr == j)[0]
        if sel.size == 0:
            continue
        pname = candidate_families[j][0]
        mask[pname] = torch.from_numpy(local_arr[sel].astype(np.int64))
    return mask


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

def save_json(path: str, obj: dict, *, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"cannot json-serialize {type(o)}")


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
