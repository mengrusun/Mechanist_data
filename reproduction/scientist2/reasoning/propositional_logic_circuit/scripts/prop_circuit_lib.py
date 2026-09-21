"""
Shared library for propositional-logic circuit analysis.

Functions:
  - load_model(model_path):        HookedTransformer + tokenizer.
  - load_jsonl(path):              Read a JSONL dataset file.
  - answer_token_ids(tok, answers): Get True/False token ids (single-token or leading-subword).
  - answer_metric(logits, true_id, false_id, gt):
        Return dict with logit_diff, prob_diff, KL (target dist) and per-example.
  - eval_accuracy(model, tok, records): Anchor cell accuracy.
  - run_attribution_patching(model, clean_records, corrupt_records):
        Attribution-patching scores for every (layer, head) and every (layer, mlp_out).
  - run_activation_patching(model, clean_records, corrupt_records, components):
        Path-patching / activation-patching restore experiment for a component set.
  - resample_ablate(model, records, ablate_components, pool_records):
        Resample ablation on a shortlist (used for M3 sufficiency).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def set_global_seeds(seed: int = 42) -> None:
    """Set every RNG we can reach for reproducibility."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- I/O ----------

def load_jsonl(path: str | os.PathLike) -> List[Dict]:
    records = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    fold_ln: bool = True,
    center_writing_weights: bool = True,
    n_devices: int = 1,
) -> Tuple[HookedTransformer, "transformers.PreTrainedTokenizerBase"]:
    """
    Load HookedTransformer from a local HF checkpoint.
    Uses `from_pretrained_no_processing` for consistency across models.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Resolve mistral names -> official TL name.
    tl_name = None
    if "Mistral-7B-v0.1" in model_path or model_path.endswith("Mistral-7B-v0.1"):
        tl_name = "mistralai/Mistral-7B-v0.1"
    elif "Mistral-7B-Instruct-v0.1" in model_path:
        tl_name = "mistralai/Mistral-7B-Instruct-v0.1"
    elif "gemma-2-9b" in model_path.lower() and "it" not in model_path.lower():
        tl_name = "google/gemma-2-9b"
    elif "gemma-2-9b-it" in model_path.lower():
        tl_name = "google/gemma-2-9b-it"
    elif "gemma-2-27b" in model_path.lower() and "it" not in model_path.lower():
        tl_name = "google/gemma-2-27b"
    else:
        # Fallback: use raw path
        tl_name = model_path

    print(f"[model] loading tokenizer from {model_path}")
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        # tokenizers version mismatch (e.g. Gemma-2 tokenizer.json requires
        # tokenizers >= 0.20 but we have 0.19); fall back to the slow tokenizer.
        print(f"[model] fast tokenizer failed ({type(e).__name__}: {str(e)[:120]}); "
              f"falling back to use_fast=False")
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print(f"[model] loading base HF model from {model_path} (dtype={dtype})")
    hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    print(f"[model] wrapping into HookedTransformer (tl_name={tl_name}, n_devices={n_devices})")
    model = HookedTransformer.from_pretrained(
        tl_name,
        hf_model=hf,
        tokenizer=tok,
        device=device,
        dtype=dtype,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=False,           # keep unembed as-is
        n_devices=n_devices,
    )
    model.eval()
    # NOTE: We intentionally leave requires_grad=True on parameters so that
    # attribution patching (which requires grad w.r.t. activations) works.
    # Parameters are never updated because we never call optimizer.step().
    # torch.no_grad() contexts are used elsewhere to avoid building the graph.

    return model, tok


# ---------- Answer-token utilities ----------

def get_answer_token_ids(tok) -> Tuple[int, int]:
    """
    Return (true_id, false_id) — the token ids the model would emit as the FIRST
    token after "Answer:" for "True" and "False".

    Uses the actual tokenizer to encode " True" / " False" (leading space,
    since after "Answer:" a space is expected).
    """
    def _first_token_id(text: str) -> int:
        ids = tok.encode(text, add_special_tokens=False)
        # Trim potential leading space token if the tokenizer emitted one plus another char.
        # We want the FIRST non-empty token id.
        if len(ids) == 0:
            raise ValueError(f"empty encoding of {text!r}")
        return ids[0]

    # Try " True" then "True".
    for candidate in (" True", "True"):
        tid = _first_token_id(candidate)
        # OK if the token decodes to something containing "True".
        if "True" in tok.decode([tid]):
            true_id = tid
            break
    else:
        raise RuntimeError("Could not find a 'True' answer token id")

    for candidate in (" False", "False"):
        tid = _first_token_id(candidate)
        if "False" in tok.decode([tid]):
            false_id = tid
            break
    else:
        raise RuntimeError("Could not find a 'False' answer token id")

    return true_id, false_id


# ---------- Prompt encoding ----------

@torch.no_grad()
def encode_prompt(tok, prompt: str, device: str, max_length: int = 512) -> torch.Tensor:
    """Encode a single prompt as a token id tensor of shape (1, seq)."""
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length,
              padding=False, add_special_tokens=True)
    return enc["input_ids"].to(device)


@torch.no_grad()
def encode_batch(tok, prompts: List[str], device: str, max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode a batch of prompts. Returns (input_ids, attention_mask, seq_lens).
    Left-pads so the *last* real token position aligns for every example — makes
    reading answer-position logits trivial (always at index -1).

    attention_mask has 1 at real token positions, 0 at pad positions — this is
    required for causal LMs so that padding is not treated as real prefix context.
    """
    # First encode individually to compute lengths.
    encs = [tok.encode(p, add_special_tokens=True) for p in prompts]
    max_len = min(max_length, max(len(e) for e in encs))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(prompts), max_len), dtype=torch.long, device=device)
    seq_lens = torch.zeros(len(prompts), dtype=torch.long, device=device)
    for i, e in enumerate(encs):
        L = min(len(e), max_len)
        # LEFT-PAD:
        input_ids[i, max_len - L:] = torch.tensor(e[-L:], dtype=torch.long, device=device)
        attention_mask[i, max_len - L:] = 1
        seq_lens[i] = L
    return input_ids, attention_mask, seq_lens


# ---------- Metrics ----------

def compute_metrics(
    logits: torch.Tensor,   # (B, V) at answer position
    true_id: int,
    false_id: int,
    gt: torch.Tensor,       # (B,) 1 if answer is True, 0 if False
    baseline_logits: Optional[torch.Tensor] = None,   # for KL (clean distribution as target)
) -> Dict[str, torch.Tensor]:
    """
    Return per-example metrics (as tensors, mean is caller's job).

    logit_diff: (logit[correct] - logit[incorrect]) at answer position.
                Positive means the model prefers the correct answer.
    prob_diff:  softmax(logits)[correct] - softmax(logits)[incorrect].
    KL:         KL(baseline_probs || probs) if baseline_logits is provided.
                (Direction: from the clean/reference distribution to the current one.)
    """
    logits = logits.float()
    true_l = logits[:, true_id]
    false_l = logits[:, false_id]
    signed = torch.where(gt == 1, true_l - false_l, false_l - true_l)
    log_softmax = F.log_softmax(logits, dim=-1)
    softmax = log_softmax.exp()
    true_p = softmax[:, true_id]
    false_p = softmax[:, false_id]
    prob_signed = torch.where(gt == 1, true_p - false_p, false_p - true_p)

    out = {"logit_diff": signed, "prob_diff": prob_signed}
    if baseline_logits is not None:
        base_log_softmax = F.log_softmax(baseline_logits.float(), dim=-1)
        base_softmax = base_log_softmax.exp()
        # KL(baseline || current)
        kl = (base_softmax * (base_log_softmax - log_softmax)).sum(dim=-1)
        out["KL"] = kl
    return out


# ---------- Accuracy ----------

@torch.no_grad()
def eval_accuracy(
    model: HookedTransformer,
    tok,
    records: List[Dict],
    batch_size: int = 16,
    max_length: int = 512,
) -> Dict[str, float]:
    """Compute (True vs False) argmax accuracy at the answer position."""
    true_id, false_id = get_answer_token_ids(tok)
    device = next(model.parameters()).device
    n = len(records)
    correct = 0
    tf_prefer = 0
    for i in range(0, n, batch_size):
        chunk = records[i:i + batch_size]
        prompts = [r["prompt"] for r in chunk]
        input_ids, attn_mask, _ = encode_batch(tok, prompts, str(device), max_length)
        logits = model(input_ids, return_type="logits", attention_mask=attn_mask)  # (B, S, V)
        last_logits = logits[:, -1, :]
        # Restrict to {True, False} for T/F preference accuracy.
        answers = torch.tensor([1 if r["answer"] == "True" else 0 for r in chunk], device=device)
        pred = (last_logits[:, true_id] > last_logits[:, false_id]).long()
        correct += (pred == answers).sum().item()
        # Also check whether True/False is even the top-2 among the full vocab.
        top1 = last_logits.argmax(dim=-1)
        for t in top1.tolist():
            if t == true_id or t == false_id:
                tf_prefer += 1
    return {
        "n": n,
        "accuracy_TF": correct / n if n else 0.0,
        "top1_TF_share": tf_prefer / n if n else 0.0,
        "true_id": int(true_id),
        "false_id": int(false_id),
    }


# ---------- Activation caching & patching ----------

def _cache_names(model: HookedTransformer) -> List[str]:
    """Names of the hooks we want to cache for attribution & path patching."""
    L = model.cfg.n_layers
    names = []
    for l in range(L):
        # per-head attention output before merging (z: (B, S, H, d_head))
        names.append(f"blocks.{l}.attn.hook_z")
        # per-layer MLP output (mlp_out: (B, S, d_model))
        names.append(f"blocks.{l}.hook_mlp_out")
    return names


@torch.no_grad()
def cache_activations(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    names: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward pass with activation caching. Returns (logits, cache_dict)."""
    if names is None:
        names = _cache_names(model)
    kwargs = {"names_filter": names, "return_type": "logits"}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    logits, cache = model.run_with_cache(input_ids, **kwargs)
    return logits, {k: cache[k].detach() for k in names}


# ---------- Attribution patching (single backward pass) ----------

def attribution_scores(
    model: HookedTransformer,
    clean_records: List[Dict],
    corrupt_records: List[Dict],
    true_id: int,
    false_id: int,
    batch_size: int = 4,
    max_length: int = 512,
    metric: str = "logit_diff",
) -> Dict[str, torch.Tensor]:
    """
    Compute attribution-patching scores.

    For each (clean, corrupt) pair with the same clean_answer:
      - Compute clean-cache activations (with grad on) for the CORRUPTED prompt.
      - Metric M = logit_diff at answer position on the CORRUPTED prompt with corrupted-input's
        gt-answer being the *original clean* answer (because that's the answer we want to restore).
      - Score(component) = grad(M w.r.t. corrupt_activation) . (clean_activation - corrupt_activation)
        per-component (each attention-head z, each mlp_out).
      - Aggregate over pairs by mean absolute (or signed) score.

    Returns dict with keys:
      "attn_head_scores": (L, H)   — mean attribution over pairs (signed).
      "mlp_scores":       (L,)     — mean attribution over pairs.
      "attn_head_abs":    (L, H)
      "mlp_abs":          (L,)
    """
    L = model.cfg.n_layers
    H = model.cfg.n_heads
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    attn_scores_sum = torch.zeros(L, H, dtype=torch.float32, device=device)
    attn_abs_sum = torch.zeros(L, H, dtype=torch.float32, device=device)
    mlp_scores_sum = torch.zeros(L, dtype=torch.float32, device=device)
    mlp_abs_sum = torch.zeros(L, dtype=torch.float32, device=device)
    n_used = 0

    assert len(clean_records) == len(corrupt_records)
    names = _cache_names(model)

    for i in range(0, len(clean_records), batch_size):
        clean_chunk = clean_records[i:i + batch_size]
        corr_chunk = corrupt_records[i:i + batch_size]
        # Ground truth: which token is *correct* for the clean prompt.
        gt = torch.tensor(
            [1 if r["answer"] == "True" else 0 for r in clean_chunk], device=device
        )
        clean_prompts = [r["prompt"] for r in clean_chunk]
        corr_prompts = [r["prompt"] for r in corr_chunk]

        # Encode.
        clean_ids, clean_mask, _ = encode_batch(model.tokenizer, clean_prompts, str(device), max_length)
        corr_ids, corr_mask, _ = encode_batch(model.tokenizer, corr_prompts, str(device), max_length)
        # Pad to matching seq length (left-pad already applied; enforce same length).
        max_len = max(clean_ids.shape[1], corr_ids.shape[1])
        def _left_pad_ids(t: torch.Tensor, L_target: int) -> torch.Tensor:
            if t.shape[1] == L_target:
                return t
            pad_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
            pad = torch.full((t.shape[0], L_target - t.shape[1]), pad_id, device=t.device, dtype=t.dtype)
            return torch.cat([pad, t], dim=1)
        def _left_pad_mask(m: torch.Tensor, L_target: int) -> torch.Tensor:
            if m.shape[1] == L_target:
                return m
            pad = torch.zeros((m.shape[0], L_target - m.shape[1]), device=m.device, dtype=m.dtype)
            return torch.cat([pad, m], dim=1)
        clean_ids = _left_pad_ids(clean_ids, max_len); clean_mask = _left_pad_mask(clean_mask, max_len)
        corr_ids = _left_pad_ids(corr_ids, max_len);  corr_mask = _left_pad_mask(corr_mask, max_len)

        # 1) Cache clean activations (no grad).
        # Note: clean activations are cached WITHOUT grad; backward is taken w.r.t.
        # CORRUPT activations only (that's the attribution-patching definition).
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(
                clean_ids, names_filter=names, return_type="logits", attention_mask=clean_mask,
            )

        # 2) Forward on CORRUPT prompt with grad enabled through the cached components.
        # We attach hooks that (a) capture the corrupt activation with requires_grad,
        # (b) let the forward pass proceed. We compute the metric (logit_diff for restoring
        # the clean answer) on the CORRUPT prompt and backprop. Note we use the CLEAN gt
        # (the answer we want to restore) at the CORRUPT prompt's answer position.
        corr_activations: Dict[str, torch.Tensor] = {}

        def make_hook(name: str):
            def hook(activation, hook):
                # We need grad on this activation for attribution patching. Since
                # params keep requires_grad=True and we're in enable_grad below,
                # activations should already be non-leaf tensors with requires_grad.
                # retain_grad() ensures the .grad attribute survives backward for
                # this non-leaf tensor.
                activation.retain_grad()
                corr_activations[name] = activation
                return activation
            return hook

        hooks = [(name, make_hook(name)) for name in names]

        model.zero_grad()
        with torch.enable_grad():
            corr_logits = model.run_with_hooks(
                corr_ids, fwd_hooks=hooks, return_type="logits",
                attention_mask=corr_mask,
            )
            last_logits = corr_logits[:, -1, :].float()
            # Metric: logit_diff for the CLEAN answer (we want to restore).
            if metric == "logit_diff":
                signed = torch.where(gt == 1,
                                     last_logits[:, true_id] - last_logits[:, false_id],
                                     last_logits[:, false_id] - last_logits[:, true_id])
                M = signed.sum()  # sum so grad has correct scale per example
            elif metric == "prob_diff":
                logp = F.log_softmax(last_logits, dim=-1)
                p = logp.exp()
                signed = torch.where(gt == 1, p[:, true_id] - p[:, false_id],
                                     p[:, false_id] - p[:, true_id])
                M = signed.sum()
            else:
                raise ValueError(metric)
            M.backward()

        # 3) Compute per-component score.
        for l in range(L):
            zname = f"blocks.{l}.attn.hook_z"
            mname = f"blocks.{l}.hook_mlp_out"
            corr_z = corr_activations[zname]              # (B, S, H, d_head)
            clean_z = clean_cache[zname]                  # (B, S, H, d_head)
            grad_z = corr_z.grad                          # (B, S, H, d_head)
            if grad_z is None:
                continue
            # Attribution: sum over (S, d_head) of grad * (clean - corrupt), per head.
            delta = (clean_z - corr_z).float()
            g = grad_z.float()
            contrib = (g * delta).sum(dim=(1, 3))         # (B, H)
            attn_scores_sum[l] += contrib.sum(dim=0)      # (H,)
            attn_abs_sum[l] += contrib.abs().sum(dim=0)   # (H,)

            corr_m = corr_activations[mname]              # (B, S, d_model)
            clean_m = clean_cache[mname]
            grad_m = corr_m.grad
            if grad_m is None:
                continue
            delta_m = (clean_m - corr_m).float()
            gm = grad_m.float()
            contrib_m = (gm * delta_m).sum(dim=(1, 2))    # (B,)
            mlp_scores_sum[l] += contrib_m.sum().item()
            mlp_abs_sum[l] += contrib_m.abs().sum().item()

        n_used += len(clean_chunk)
        # Free graph.
        del corr_logits, last_logits, M
        for k in list(corr_activations.keys()):
            del corr_activations[k]
        torch.cuda.empty_cache()

    if n_used == 0:
        n_used = 1
    return {
        "attn_head_scores": (attn_scores_sum / n_used).detach().cpu(),
        "attn_head_abs":    (attn_abs_sum / n_used).detach().cpu(),
        "mlp_scores":       (mlp_scores_sum / n_used).detach().cpu(),
        "mlp_abs":          (mlp_abs_sum / n_used).detach().cpu(),
        "n_pairs":          int(n_used),
    }


# ---------- Path / Activation Patching ----------

@torch.no_grad()
def run_activation_patch(
    model: HookedTransformer,
    clean_records: List[Dict],
    corrupt_records: List[Dict],
    components: List[Tuple[str, int, Optional[int]]],   # list of (kind, layer, head_or_None)
    true_id: int,
    false_id: int,
    batch_size: int = 4,
    max_length: int = 512,
    direction: str = "clean_into_corrupt",  # necessity: patch clean acts into corrupt run
) -> Dict[str, float]:
    """
    Activation patching over a set of components.

    direction == "clean_into_corrupt" (necessity):
      Run on corrupt prompt; for each hook in `components`, splice in the CLEAN
      activation at that hook. Measure whether the answer at the answer position
      recovers to the clean answer.

    direction == "corrupt_into_clean" (sufficiency check via KO — not used here):
      Not implemented here; use `run_reinsertion_sufficiency` for M3.

    Returns dict with:
      - "recovery_logit_diff": Recovery on logit_diff metric.
      - "recovery_prob_diff":  Recovery on prob_diff metric.
      - "recovery_KL":         Recovery on KL (from clean baseline).
      - Also raw clean / corrupt / patched metric means.
    """
    assert direction == "clean_into_corrupt"
    assert len(clean_records) == len(corrupt_records)
    device = next(model.parameters()).device
    L = model.cfg.n_layers
    H = model.cfg.n_heads

    # Group components by hook name for efficient hooking.
    attn_by_layer: Dict[int, List[int]] = {}
    mlp_layers: set = set()
    for kind, layer, head in components:
        if kind == "attn":
            assert head is not None
            attn_by_layer.setdefault(layer, []).append(head)
        elif kind == "mlp":
            mlp_layers.add(layer)
        else:
            raise ValueError(kind)

    clean_ld_sum = clean_pd_sum = 0.0
    corr_ld_sum = corr_pd_sum = corr_kl_sum = 0.0
    patch_ld_sum = patch_pd_sum = patch_kl_sum = 0.0
    n_seen = 0

    names_all = _cache_names(model)

    for i in range(0, len(clean_records), batch_size):
        clean_chunk = clean_records[i:i + batch_size]
        corr_chunk = corrupt_records[i:i + batch_size]
        # Assert paired clean/corrupt records refer to the same clean-answer target.
        for cr, ccr in zip(clean_chunk, corr_chunk):
            if "clean_answer" in ccr:
                assert cr["answer"] == ccr["clean_answer"], (
                    f"pair_id={cr.get('pair_id')} — clean answer {cr['answer']!r} "
                    f"!= corrupt.clean_answer {ccr['clean_answer']!r}"
                )
        gt = torch.tensor([1 if r["answer"] == "True" else 0 for r in clean_chunk], device=device)
        clean_ids, clean_mask, _ = encode_batch(model.tokenizer, [r["prompt"] for r in clean_chunk], str(device), max_length)
        corr_ids, corr_mask, _ = encode_batch(model.tokenizer, [r["prompt"] for r in corr_chunk], str(device), max_length)
        max_len = max(clean_ids.shape[1], corr_ids.shape[1])
        def _left_pad_ids(t: torch.Tensor, L_target: int) -> torch.Tensor:
            if t.shape[1] == L_target:
                return t
            pad_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
            pad = torch.full((t.shape[0], L_target - t.shape[1]), pad_id, device=t.device, dtype=t.dtype)
            return torch.cat([pad, t], dim=1)
        def _left_pad_mask(m: torch.Tensor, L_target: int) -> torch.Tensor:
            if m.shape[1] == L_target:
                return m
            pad = torch.zeros((m.shape[0], L_target - m.shape[1]), device=m.device, dtype=m.dtype)
            return torch.cat([pad, m], dim=1)
        clean_ids = _left_pad_ids(clean_ids, max_len); clean_mask = _left_pad_mask(clean_mask, max_len)
        corr_ids = _left_pad_ids(corr_ids, max_len);  corr_mask = _left_pad_mask(corr_mask, max_len)

        # 1) Clean run — cache all activations we may need, and get clean logits.
        clean_logits, clean_cache = model.run_with_cache(
            clean_ids, names_filter=names_all, return_type="logits", attention_mask=clean_mask,
        )
        clean_last = clean_logits[:, -1, :]
        # 2) Baseline corrupt run.
        corr_logits = model(corr_ids, return_type="logits", attention_mask=corr_mask)
        corr_last = corr_logits[:, -1, :]

        # 3) Corrupt run with patching hooks.
        # Ensure clean cache aligns to corrupt positions (same length, so same indexing).
        # Hooks clone the incoming tensor before write, to avoid in-place mutation
        # of aliased tensors (safer with TransformerLens internals).
        hooks = []
        for layer, heads in attn_by_layer.items():
            zname = f"blocks.{layer}.attn.hook_z"
            clean_z = clean_cache[zname]
            heads_t = torch.tensor(heads, device=device, dtype=torch.long)
            def make_attn_hook(clean_z_tensor, head_idxs):
                def _hook(z, hook):
                    z_new = z.clone()
                    z_new[:, :, head_idxs, :] = clean_z_tensor[:, :, head_idxs, :].to(z.dtype)
                    return z_new
                return _hook
            hooks.append((zname, make_attn_hook(clean_z, heads_t)))
        for layer in mlp_layers:
            mname = f"blocks.{layer}.hook_mlp_out"
            clean_m = clean_cache[mname]
            def make_mlp_hook(clean_m_tensor):
                def _hook(m, hook):
                    return clean_m_tensor.to(m.dtype)
                return _hook
            hooks.append((mname, make_mlp_hook(clean_m)))

        patch_logits = model.run_with_hooks(
            corr_ids, fwd_hooks=hooks, return_type="logits", attention_mask=corr_mask,
        )
        patch_last = patch_logits[:, -1, :]

        # 4) Metrics.
        clean_m = compute_metrics(clean_last, true_id, false_id, gt)
        # KL(clean || corrupt) baseline computed on the SAME batch (so recovery_KL is exact).
        corr_m = compute_metrics(corr_last, true_id, false_id, gt, baseline_logits=clean_last)
        patch_m = compute_metrics(patch_last, true_id, false_id, gt, baseline_logits=clean_last)

        clean_ld_sum += clean_m["logit_diff"].sum().item()
        clean_pd_sum += clean_m["prob_diff"].sum().item()
        corr_ld_sum += corr_m["logit_diff"].sum().item()
        corr_pd_sum += corr_m["prob_diff"].sum().item()
        corr_kl_sum += corr_m["KL"].sum().item()
        patch_ld_sum += patch_m["logit_diff"].sum().item()
        patch_pd_sum += patch_m["prob_diff"].sum().item()
        patch_kl_sum += patch_m["KL"].sum().item()

        n_seen += len(clean_chunk)
        del clean_cache

    clean_ld = clean_ld_sum / n_seen
    clean_pd = clean_pd_sum / n_seen
    corr_ld = corr_ld_sum / n_seen
    corr_pd = corr_pd_sum / n_seen
    corr_kl = corr_kl_sum / n_seen
    patch_ld = patch_ld_sum / n_seen
    patch_pd = patch_pd_sum / n_seen
    patch_kl = patch_kl_sum / n_seen

    def recovery(patched, corrupted, clean):
        denom = (clean - corrupted)
        if abs(denom) < 1e-9:
            return float("nan")
        return (patched - corrupted) / denom

    # recovery_KL: patch KL should be lower than corrupt KL if the patch restores clean.
    # 1.0 means fully restored (patch_KL == 0); 0.0 means no improvement over corrupt.
    if corr_kl > 1e-9:
        recovery_kl = 1.0 - (patch_kl / corr_kl)
    else:
        recovery_kl = float("nan")

    return {
        "clean_logit_diff": clean_ld,
        "corrupt_logit_diff": corr_ld,
        "patch_logit_diff": patch_ld,
        "recovery_logit_diff": recovery(patch_ld, corr_ld, clean_ld),
        "clean_prob_diff": clean_pd,
        "corrupt_prob_diff": corr_pd,
        "patch_prob_diff": patch_pd,
        "recovery_prob_diff": recovery(patch_pd, corr_pd, clean_pd),
        "corrupt_KL": corr_kl,
        "patch_KL": patch_kl,
        "recovery_KL": recovery_kl,
        "n_pairs": n_seen,
    }


@torch.no_grad()
def measure_kl_baseline(
    model: HookedTransformer,
    clean_records: List[Dict],
    corrupt_records: List[Dict],
    batch_size: int = 4,
    max_length: int = 512,
) -> float:
    """Mean KL(clean || corrupt) across pairs — used as the recovery-KL denominator."""
    assert len(clean_records) == len(corrupt_records)
    device = next(model.parameters()).device
    total_kl = 0.0
    n = 0
    for i in range(0, len(clean_records), batch_size):
        clean_chunk = clean_records[i:i + batch_size]
        corr_chunk = corrupt_records[i:i + batch_size]
        clean_ids, clean_mask, _ = encode_batch(model.tokenizer, [r["prompt"] for r in clean_chunk], str(device), max_length)
        corr_ids, corr_mask, _ = encode_batch(model.tokenizer, [r["prompt"] for r in corr_chunk], str(device), max_length)
        max_len = max(clean_ids.shape[1], corr_ids.shape[1])
        pad_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
        def _lpi(t, L_target):
            if t.shape[1] == L_target: return t
            pad = torch.full((t.shape[0], L_target - t.shape[1]), pad_id, device=t.device, dtype=t.dtype)
            return torch.cat([pad, t], dim=1)
        def _lpm(m, L_target):
            if m.shape[1] == L_target: return m
            pad = torch.zeros((m.shape[0], L_target - m.shape[1]), device=m.device, dtype=m.dtype)
            return torch.cat([pad, m], dim=1)
        clean_ids = _lpi(clean_ids, max_len); clean_mask = _lpm(clean_mask, max_len)
        corr_ids = _lpi(corr_ids, max_len);   corr_mask = _lpm(corr_mask, max_len)
        clean_last = model(clean_ids, return_type="logits", attention_mask=clean_mask)[:, -1, :].float()
        corr_last = model(corr_ids, return_type="logits", attention_mask=corr_mask)[:, -1, :].float()
        clean_log = F.log_softmax(clean_last, dim=-1)
        corr_log = F.log_softmax(corr_last, dim=-1)
        clean_p = clean_log.exp()
        kl = (clean_p * (clean_log - corr_log)).sum(dim=-1)
        total_kl += kl.sum().item()
        n += clean_ids.shape[0]
    return total_kl / max(n, 1)


# ---------- Sufficiency (resample-ablate everything except shortlist) ----------

@torch.no_grad()
def run_reinsertion_sufficiency(
    model: HookedTransformer,
    clean_records: List[Dict],
    resample_records: List[Dict],
    components: List[Tuple[str, int, Optional[int]]],   # shortlist to KEEP clean
    true_id: int,
    false_id: int,
    batch_size: int = 4,
    max_length: int = 512,
    n_resample_seeds: int = 5,
) -> Dict[str, float]:
    """
    Sufficiency: on the CLEAN prompt, resample-ablate every non-shortlisted
    component (replace with activation from an unrelated prompt), then measure
    whether the answer stays correct.

    Averaged over `n_resample_seeds` — each seed picks a different pool sample.
    """
    device = next(model.parameters()).device
    L = model.cfg.n_layers
    H = model.cfg.n_heads

    # Build the set of hooks that should be RESAMPLE-ABLATED (NOT in shortlist).
    keep_attn: Dict[int, set] = {}
    keep_mlp: set = set()
    for kind, layer, head in components:
        if kind == "attn":
            keep_attn.setdefault(layer, set()).add(head)
        elif kind == "mlp":
            keep_mlp.add(layer)

    all_heads = {l: set(range(H)) for l in range(L)}
    all_mlps = set(range(L))
    ablate_attn = {l: sorted(all_heads[l] - keep_attn.get(l, set())) for l in range(L)}
    ablate_mlp = sorted(all_mlps - keep_mlp)

    names_all = _cache_names(model)

    # Cache resample activations from the pool: pre-compute a big cache
    # keyed by resample-record index.
    per_seed_recovery = {"logit_diff": [], "prob_diff": [], "KL": []}
    baseline_kl_denominator = None

    # First, compute clean baseline (no ablation).
    clean_ld_sum = clean_pd_sum = 0.0
    n_seen = 0
    for i in range(0, len(clean_records), batch_size):
        chunk = clean_records[i:i + batch_size]
        gt = torch.tensor([1 if r["answer"] == "True" else 0 for r in chunk], device=device)
        ids, mask, _ = encode_batch(model.tokenizer, [r["prompt"] for r in chunk], str(device), max_length)
        logits = model(ids, return_type="logits", attention_mask=mask)[:, -1, :]
        m = compute_metrics(logits, true_id, false_id, gt)
        clean_ld_sum += m["logit_diff"].sum().item()
        clean_pd_sum += m["prob_diff"].sum().item()
        n_seen += len(chunk)
    clean_ld = clean_ld_sum / n_seen
    clean_pd = clean_pd_sum / n_seen

    # Full-ablation baseline (ablate everything = no shortlist protection).
    # This is the "floor" — what happens when the residual stream is fully corrupted.
    # Pre-index the resample pool by tokenized length so we can length-match.
    _resample_len_index_cache = {}
    def _length_matched_pick(pool: List[Dict], target_len: int, rng) -> Dict:
        # Cache lengths on first call.
        if "lengths" not in _resample_len_index_cache:
            _resample_len_index_cache["lengths"] = [
                len(model.tokenizer.encode(r["prompt"], add_special_tokens=True)) for r in pool
            ]
        lengths = _resample_len_index_cache["lengths"]
        # Prefer records within +/- 15% of target_len; fall back to +/- 30%; else any.
        for tol in (0.15, 0.30, 1.0):
            lo, hi = int(target_len * (1 - tol)), int(target_len * (1 + tol))
            candidates = [i for i, L in enumerate(lengths) if lo <= L <= hi]
            if candidates:
                return pool[rng.choice(candidates)]
        return rng.choice(pool)

    def _run_ablated(
        batch_clean_records: List[Dict],
        resample_pool: List[Dict],
        keep_ablate_attn: Dict[int, List[int]],
        keep_ablate_mlp_list: List[int],
        seed_offset: int,
    ) -> Dict[str, float]:
        """Run the CLEAN prompts with resample ablation on the specified components."""
        device = next(model.parameters()).device
        import random
        rng = random.Random(seed_offset + 12345)
        # Length-matched resample per clean example.
        clean_prompts = [r["prompt"] for r in batch_clean_records]
        clean_lengths = [len(model.tokenizer.encode(p, add_special_tokens=True)) for p in clean_prompts]
        resample_prompts = [_length_matched_pick(resample_pool, cl, rng)["prompt"] for cl in clean_lengths]
        clean_ids, clean_mask, _ = encode_batch(model.tokenizer, clean_prompts, str(device), max_length)
        resample_ids, resample_mask, _ = encode_batch(model.tokenizer, resample_prompts, str(device), max_length)
        max_len = max(clean_ids.shape[1], resample_ids.shape[1])
        pad_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
        def _lpi(t, L_target):
            if t.shape[1] == L_target: return t
            pad = torch.full((t.shape[0], L_target - t.shape[1]), pad_id, device=t.device, dtype=t.dtype)
            return torch.cat([pad, t], dim=1)
        def _lpm(m, L_target):
            if m.shape[1] == L_target: return m
            pad = torch.zeros((m.shape[0], L_target - m.shape[1]), device=m.device, dtype=m.dtype)
            return torch.cat([pad, m], dim=1)
        clean_ids = _lpi(clean_ids, max_len); clean_mask = _lpm(clean_mask, max_len)
        resample_ids = _lpi(resample_ids, max_len); resample_mask = _lpm(resample_mask, max_len)

        # Cache resample activations at the components to ablate.
        _, resample_cache = model.run_with_cache(
            resample_ids, names_filter=names_all, return_type="logits",
            attention_mask=resample_mask,
        )

        hooks = []
        for layer, heads in keep_ablate_attn.items():
            if not heads:
                continue
            zname = f"blocks.{layer}.attn.hook_z"
            r_z = resample_cache[zname]
            head_idxs = torch.tensor(heads, device=device, dtype=torch.long)
            def make_attn_hook(rz, hi):
                def _hook(z, hook):
                    z_new = z.clone()
                    z_new[:, :, hi, :] = rz[:, :, hi, :].to(z.dtype)
                    return z_new
                return _hook
            hooks.append((zname, make_attn_hook(r_z, head_idxs)))
        for layer in keep_ablate_mlp_list:
            mname = f"blocks.{layer}.hook_mlp_out"
            r_m = resample_cache[mname]
            def make_mlp_hook(rm):
                def _hook(m, hook):
                    return rm.to(m.dtype)
                return _hook
            hooks.append((mname, make_mlp_hook(r_m)))

        gt = torch.tensor([1 if r["answer"] == "True" else 0 for r in batch_clean_records], device=device)
        logits = model.run_with_hooks(
            clean_ids, fwd_hooks=hooks, return_type="logits", attention_mask=clean_mask,
        )[:, -1, :]
        # For KL, use the clean-unablated logits as reference.
        clean_ref = model(clean_ids, return_type="logits", attention_mask=clean_mask)[:, -1, :]
        m = compute_metrics(logits, true_id, false_id, gt, baseline_logits=clean_ref)
        del resample_cache
        return {
            "logit_diff": float(m["logit_diff"].sum().item()),
            "prob_diff": float(m["prob_diff"].sum().item()),
            "KL": float(m["KL"].sum().item()),
            "n": len(batch_clean_records),
        }

    # Full-ablation baseline (ablate ALL heads and ALL mlps in every layer).
    full_ablate_attn = {l: list(range(H)) for l in range(L)}
    full_ablate_mlp = list(range(L))

    # Seed-averaged sufficiency (keep shortlist, ablate everything else).
    for seed in range(n_resample_seeds):
        # Shortlist-protecting ablation.
        ld_sum = pd_sum = kl_sum = 0.0
        n = 0
        # Full-ablation floor.
        floor_ld_sum = floor_pd_sum = floor_kl_sum = 0.0
        floor_n = 0
        for i in range(0, len(clean_records), batch_size):
            chunk = clean_records[i:i + batch_size]
            r_shortlist = _run_ablated(chunk, resample_records, ablate_attn, ablate_mlp, seed * 1000 + i)
            r_full = _run_ablated(chunk, resample_records, full_ablate_attn, full_ablate_mlp, seed * 1000 + i)
            ld_sum += r_shortlist["logit_diff"]; pd_sum += r_shortlist["prob_diff"]; kl_sum += r_shortlist["KL"]
            floor_ld_sum += r_full["logit_diff"]; floor_pd_sum += r_full["prob_diff"]; floor_kl_sum += r_full["KL"]
            n += r_shortlist["n"]; floor_n += r_full["n"]
        keep_ld = ld_sum / n; keep_pd = pd_sum / n; keep_kl = kl_sum / n
        floor_ld = floor_ld_sum / floor_n; floor_pd = floor_pd_sum / floor_n; floor_kl = floor_kl_sum / floor_n
        # Sufficient-recovery: (keep - floor) / (clean - floor).
        def rec(k, f, c):
            if abs(c - f) < 1e-9:
                return float("nan")
            return (k - f) / (c - f)
        per_seed_recovery["logit_diff"].append(rec(keep_ld, floor_ld, clean_ld))
        per_seed_recovery["prob_diff"].append(rec(keep_pd, floor_pd, clean_pd))
        # For KL, "clean" reference has KL=0; "floor" has some positive KL; the "keep" run
        # should have a smaller KL than floor if the shortlist is sufficient. So report
        # recovery_KL = 1 - keep_KL / floor_KL (higher = better).
        if floor_kl > 1e-9:
            per_seed_recovery["KL"].append(1.0 - (keep_kl / floor_kl))
        else:
            per_seed_recovery["KL"].append(float("nan"))

    def _stats(vs):
        vs = [v for v in vs if v == v]  # drop NaNs
        if not vs:
            return {"mean": float("nan"), "std": float("nan"), "seeds": []}
        import statistics
        return {
            "mean": statistics.mean(vs),
            "std": statistics.pstdev(vs) if len(vs) > 1 else 0.0,
            "seeds": vs,
        }

    return {
        "sufficient_recovery_logit_diff": _stats(per_seed_recovery["logit_diff"])["mean"],
        "sufficient_recovery_prob_diff":  _stats(per_seed_recovery["prob_diff"])["mean"],
        "sufficient_recovery_KL":         _stats(per_seed_recovery["KL"])["mean"],
        "per_seed_logit_diff": _stats(per_seed_recovery["logit_diff"]),
        "per_seed_prob_diff":  _stats(per_seed_recovery["prob_diff"]),
        "per_seed_KL":         _stats(per_seed_recovery["KL"]),
        "clean_logit_diff": clean_ld,
        "clean_prob_diff": clean_pd,
        "n_pairs": n_seen,
        "n_seeds": n_resample_seeds,
    }
