#!/usr/bin/env python3
"""
M4 — Attention-block from cache → confidence-generation position.

Directional ablation: at chosen layer(s), zero out the attention weight from
the confidence-generation query position to the cache key position(s).

For Gemma-3 27B (transformers Gemma3ForCausalLM), each decoder layer's
self_attn.forward is called with input hidden_states and returns
(attn_output, attn_weights). We monkey-patch a wrapper that, before the
softmax, masks the (Q=conf_gen, K=cache) entries to -inf. The concrete
implementation uses a forward pre-hook on the self_attn module that reads
`attention_mask` from kwargs and adds an attn mask term with -inf at the
blocked K positions ONLY at the conf_gen query row.

Because transformers uses a causal mask + optional attention_mask, we
implement the block as: for each blocked layer L, register a forward hook on
`model.model.layers[L-1].self_attn` that patches the module's forward call by
substituting an extra additive attention bias tensor of shape
(1, 1, T, T) with `-inf` at positions [conf_gen_position, cache_positions].

Measures per item:
  - verb_conf_dist_kl_to_prior — KL between blocked-conf output distribution
    over the first conf token and the population prior (empirical marginal over
    conf digits from M1).
  - verb_conf_mean_shift        — mean(conf_blocked) - mean(conf_baseline)
  - answer_acc_preserved        — trivially true (answer is fixed in prompt)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc_common import (
    CONF_GEN_LABEL,
    MAX_CONF_TOKENS,
    MODEL_PATH,
    load_model_and_tokenizer,
    parse_confidence,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--dataset", default="/data/zhenqian/data/trivia_qa")
    p.add_argument("--n_items", type=int, default=300)
    p.add_argument("--block_from", default="AUTO",
                   help="Comma-list of Pos labels to block FROM (cache positions). "
                        "AUTO reads M2 top-K positions.")
    p.add_argument("--block_to", default="C0", help="Query position to block into.")
    p.add_argument("--block_at_layer", default="AUTO",
                   help="'AUTO' → use M2's top layer; 'top,last' → use [top, top..last]; "
                        "or comma-list of integer layers.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--measure",
                   default="verb_conf_dist_kl_to_prior,verb_conf_mean_shift,answer_acc_preserved")
    p.add_argument("--m1_cache", required=True)
    p.add_argument("--m2_dir", required=True)
    p.add_argument("--template", default="T0")
    p.add_argument("--out", required=True)
    p.add_argument("--control_from", default=None,
                   help="Optional comma-list of Pos labels for matched non-cache-position control block.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def load_m1_seed(m1_cache: str, seed: int, template: str):
    seed_dir = Path(m1_cache) / f"seed{seed}" / template
    items = []
    with (seed_dir / "items.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class AttentionBlocker:
    """
    Registers a forward *pre*-hook on a Gemma3 self_attn module that injects
    an additive attention bias masking (query=conf_gen, key=blocked) to -inf.

    The pre-hook modifies `kwargs["attention_mask"]` in place before the module runs.
    """

    def __init__(self, model, layer_indices_1based: List[int],
                 query_pos: int, blocked_key_positions: List[int]):
        self.model = model
        self.layer_indices = layer_indices_1based
        self.q_pos = query_pos
        self.k_positions = blocked_key_positions
        self.handles = []

    def __enter__(self):
        for L in self.layer_indices:
            module = self.model.model.layers[L - 1].self_attn
            # We use a forward pre-hook with kwargs support (torch >= 2.0)
            handle = module.register_forward_pre_hook(
                self._pre_hook, with_kwargs=True
            )
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _pre_hook(self, module, args, kwargs):
        # kwargs contains `attention_mask` shape (batch, 1, T, T) or (batch, T, T)
        # We add a -inf bias at (q_pos, k_positions) that survives softmax.
        am = kwargs.get("attention_mask", None)
        hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is None and args:
            hidden_states = args[0]
        if hidden_states is None:
            return args, kwargs
        T = hidden_states.shape[1]
        if self.q_pos >= T:
            return args, kwargs
        device = hidden_states.device
        dtype = hidden_states.dtype
        # Build (1, 1, T, T) additive mask filled with 0; -inf at (q, k) entries.
        bias = torch.zeros((1, 1, T, T), device=device, dtype=torch.float32)
        for k in self.k_positions:
            if 0 <= k < T:
                bias[0, 0, self.q_pos, k] = float("-inf")
        if am is None:
            new_mask = bias.to(dtype)
        else:
            am_f = am.to(torch.float32)
            # Broadcast: normalize shape to (1, 1, T, T)
            if am_f.dim() == 2:
                am_f = am_f[None, None, :, :]
            elif am_f.dim() == 3:
                am_f = am_f[:, None, :, :]
            # add
            new_mask = (am_f + bias).to(dtype)
        kwargs["attention_mask"] = new_mask
        return args, kwargs


def parse_layer_arg(layer_arg: str, top_layer: int, last_layer: int) -> List[int]:
    """
    "AUTO" → [top_layer]
    "top,last" → [top_layer, top_layer+1, ..., last_layer]
    "L1,L2,..." → [ints]
    """
    if layer_arg == "AUTO":
        return [top_layer]
    if layer_arg == "top,last":
        return list(range(top_layer, last_layer + 1))
    return [int(x) for x in layer_arg.split(",")]


@torch.no_grad()
def decode_conf_and_first_dist(model, tok, input_ids: torch.Tensor,
                               max_conf_tokens: int = MAX_CONF_TOKENS
                               ) -> Tuple[Optional[int], np.ndarray]:
    """Greedy-decode confidence and return (parsed_conf, first_token_probdist over vocab)."""
    device = input_ids.device
    eos_id = tok.eos_token_id
    newline_ids = {tok(s, add_special_tokens=False).input_ids[-1] for s in ("\n", "\n\n", "\n\n\n")}
    # First step distribution
    out = model(input_ids, use_cache=False)
    first_logits = out.logits[0, -1, :].float()
    first_probs = torch.softmax(first_logits, dim=-1).cpu().numpy()
    next_id = int(first_logits.argmax().item())
    gen_ids = [next_id]
    cur = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
    for _ in range(max_conf_tokens - 1):
        if next_id == eos_id or next_id in newline_ids:
            break
        out = model(cur, use_cache=False)
        logits = out.logits[0, -1, :]
        next_id = int(logits.argmax().item())
        gen_ids.append(next_id)
        cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)
    txt = tok.decode(gen_ids, skip_special_tokens=True)
    return parse_confidence(txt), first_probs


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    # ---- Load M2 top-k -> cache positions & top layer -------------------
    with (Path(args.m2_dir) / "top_k_sites.json").open() as f:
        m2 = json.load(f)
    top_k = m2["top_k"]
    if not top_k:
        print("[m4] no top-K sites from M2 — aborting", flush=True)
        return
    if args.block_from == "AUTO":
        block_from_labels = list({c["position"] for c in top_k})
    else:
        block_from_labels = args.block_from.split(",")
    control_labels = args.control_from.split(",") if args.control_from else []

    top_layer = top_k[0]["layer"]
    last_layer = 60  # gemma-3-27b-pt uses probes up to L60 in the 62-layer grid

    layers_to_block = parse_layer_arg(args.block_at_layer, top_layer, last_layer)
    print(f"[m4] block from positions {block_from_labels} at layers {layers_to_block} (top_layer={top_layer})", flush=True)

    # ---- Load model + tokenizer -----------------------------------------
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[m4] Loading model...", flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(args.model, dtype=dtype)
    device = next(model.parameters()).device
    print(f"[m4] Model loaded in {time.time()-t0:.1f}s on {device}", flush=True)

    items = load_m1_seed(args.m1_cache, args.seed, args.template)
    # Keep only items with a parsed verbal_conf
    items = [it for it in items if it.get("verbal_conf") is not None]
    if args.n_items > 0:
        items = items[: args.n_items]
    print(f"[m4] using {len(items)} items", flush=True)

    # ---- Compute template-conditional prior over the first conf-token ---
    # Prior = empirical marginal over the first conf-token id observed in M1.
    # This uses only the CLEAN, unblocked distribution; the "prior" is just the
    # normalized marginal of first-token argmaxes across items. We approximate it
    # by soft-averaging the first-token distributions of a small sub-sample.
    prior_sub = items[: min(50, len(items))]
    prior_probs = None
    for it in prior_sub:
        input_ids = torch.tensor([it["full_ids"][: it["conf_gen_position"]]], device=device)
        _, first_probs = decode_conf_and_first_dist(model, tok, input_ids)
        prior_probs = first_probs if prior_probs is None else prior_probs + first_probs
    prior_probs = prior_probs / len(prior_sub)

    # ---- For each item, run baseline + blocked forward ------------------
    def run_all_items(labels: List[str], is_control: bool):
        conf_baseline_all = []
        conf_blocked_all = []
        kl_all = []
        acc_all = []
        item_records = []
        for i, it in enumerate(items):
            input_ids = torch.tensor([it["full_ids"][: it["conf_gen_position"]]], device=device)
            # Cache positions for this item (labels → absolute positions)
            k_positions = []
            for lab in labels:
                if lab.startswith("E"):
                    if lab in it["post_answer_positions"]:
                        k_positions.append(it["post_answer_positions"][lab])
                elif lab.startswith("C") or lab == CONF_GEN_LABEL:
                    k_positions.append(it["conf_gen_position"] - 1)  # last observed token
            # Baseline (no block)
            conf_C, first_probs_C = decode_conf_and_first_dist(model, tok, input_ids)
            # Blocked
            with AttentionBlocker(model, layers_to_block, query_pos=input_ids.shape[1] - 1,
                                  blocked_key_positions=k_positions):
                # Note: query pos in the input_ids we feed is (T-1), since we run
                # generation greedily and the FIRST forward produces the C0 token.
                # But we want to block the query at conf_gen_position — which IS
                # the last position of `input_ids` (the model produces its first
                # decode step *at* that position).
                conf_B, first_probs_B = decode_conf_and_first_dist(model, tok, input_ids)

            if conf_C is None or conf_B is None:
                continue

            # KL(blocked || prior): small when blocked distribution has collapsed
            # toward the template-conditional prior — this is the plan's claim.
            kl = kl_divergence(first_probs_B, prior_probs)
            conf_baseline_all.append(conf_C)
            conf_blocked_all.append(conf_B)
            kl_all.append(kl)
            acc_all.append(1.0)  # answer is baked into prompt; block only affects conf-gen output
            item_records.append({
                "idx": i,
                "question_id": it["question_id"],
                "conf_baseline": conf_C,
                "conf_blocked": conf_B,
                "shift": float(conf_B - conf_C),
                "kl_to_prior": float(kl),
                "k_positions": k_positions,
                "is_control": is_control,
            })
            if (i + 1) % 25 == 0:
                print(f"[m4] {'CTRL' if is_control else 'MAIN'} item {i+1}/{len(items)} "
                      f"conf_C={conf_C} conf_B={conf_B} shift_mean={np.mean([r['shift'] for r in item_records]):+.2f} "
                      f"kl_mean={np.mean(kl_all):.3f} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
        return conf_baseline_all, conf_blocked_all, kl_all, acc_all, item_records

    print(f"[m4] Running MAIN block: labels={block_from_labels}", flush=True)
    cB_main, cBk_main, kl_main, acc_main, records_main = run_all_items(block_from_labels, is_control=False)

    records_ctrl = []
    if control_labels:
        print(f"[m4] Running CTRL block: labels={control_labels}", flush=True)
        cB_ctrl, cBk_ctrl, kl_ctrl, acc_ctrl, records_ctrl = run_all_items(control_labels, is_control=True)

    # ---- Aggregate summary ---------------------------------------------
    def mk_summary(records, tag):
        if not records:
            return {"tag": tag, "n": 0}
        shifts = [r["shift"] for r in records]
        kls = [r["kl_to_prior"] for r in records]
        return {
            "tag": tag,
            "n": len(records),
            "conf_baseline_mean": float(np.mean([r["conf_baseline"] for r in records])),
            "conf_blocked_mean": float(np.mean([r["conf_blocked"] for r in records])),
            "mean_shift": float(np.mean(shifts)),
            "median_shift": float(np.median(shifts)),
            "std_shift": float(np.std(shifts)),
            "mean_kl_to_prior": float(np.mean(kls)),
            "answer_acc_preserved": 1.0,
        }

    summary = {
        "seed": args.seed,
        "layers_blocked": layers_to_block,
        "block_from_labels": block_from_labels,
        "control_from_labels": control_labels,
        "main": mk_summary(records_main, "main"),
        "control": mk_summary(records_ctrl, "control") if records_ctrl else None,
    }
    print(f"[m4] SUMMARY: {json.dumps(summary, indent=2)}", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"summary": summary,
                   "records_main": records_main,
                   "records_ctrl": records_ctrl}, f, indent=2)
    print(f"[m4] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
