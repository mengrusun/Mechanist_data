#!/usr/bin/env python3
"""M-prep — activation caching + direction extraction.

Cache residual-stream activations at every (layer, position) on the paired
(AdvBench-harmful, matched Alpaca-benign) contrast set for Llama-3-8B-Instruct.

Then, for every (layer, position) candidate:
  - Compute h = mean(harmful) - mean(benign) at (layer, t_final_instr)
  - Compute r = mean(refused) - mean(complied) at (layer, t_post_instr)
    where refusal labels come from a short greedy generation +
    canonical-string refusal classifier on the harmful prompts (single-shot).
  - Score each direction on the held-out AUROC split with a logistic-regression probe.

Outputs
-------
results/m_prep/
  activations.pt              # {'harmful': {pos: [N, L+1, d]}, 'benign': {pos: [N, L+1, d]}}
  responses.jsonl             # per-prompt greedy short generation (labels refused/complied)
  directions.json             # best-(layer, position) for h and r, + all layer-wise diff-means
  probe_auroc.csv             # per (layer x position x attribute) AUROC
  meta.json                   # run metadata (seed, n_pairs, split sizes, etc.)

Sanity mode (--sanity) subsets to 12 harmful + 12 benign, sweeps every 4th layer
only, and skips generation of the refusal side (uses a placeholder heuristic).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    build_chat_prompt,
    compute_repetition_rate,
    deterministic_split_indices,
    dump_json,
    is_refusal,
    load_alpaca_instructions,
    load_harmful_behaviors,
    position_ladder,
    resolve_positions,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--advbench", required=True)
    p.add_argument("--alpaca", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sanity", action="store_true",
                   help="Small subset + every-4th-layer sweep for smoke testing.")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--gen_max_new_tokens", type=int, default=32,
                   help="Short generation for refusal labelling.")
    p.add_argument("--gen_batch_size", type=int, default=8)
    p.add_argument("--fwd_batch_size", type=int, default=8)
    return p.parse_args()


def load_model_and_tokenizer(model_path: str, dtype: str):
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    print(f"[m_prep] loading model from {model_path} ({dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def cache_activations_and_generate(
    model, tokenizer, texts: List[str], generate: bool, args
) -> Dict:
    """Run a single forward+generate pass per prompt, cache residual-stream
    hidden states at every (layer, position) on the six-position ladder.

    Returns:
      {
        'positions': {pos_name: [N, L+1, d] float32 numpy},
        'responses': [str] or [None],
        'ladders':   [Dict] per-prompt (for offsetting positions),
      }
    """
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    pos_names = [
        "t_final_instr-2",
        "t_final_instr-1",
        "t_final_instr",
        "t_post_instr-2",
        "t_post_instr-1",
        "t_post_instr",
    ]
    N = len(texts)
    # Store as fp32 on CPU (memory: N * (L+1) * d * 4 bytes per position).
    # For 520 * 33 * 4096 * 4 = ~280 MB per position, ~1.7 GB across 6 positions.
    positions_cache = {
        pn: np.zeros((N, n_layers + 1, d_model), dtype=np.float32) for pn in pos_names
    }
    responses = [None] * N
    ladders = [None] * N

    B = args.fwd_batch_size
    t0 = time.time()
    for start in range(0, N, B):
        batch = texts[start : start + B]
        prompts = [build_chat_prompt(tokenizer, t, add_generation_prompt=True) for t in batch]
        # LEFT-pad so decoder-only generation works correctly. Positions inside
        # the padded tensor are (max_len - L) + un_padded_position.
        max_len = max(len(p["input_ids"]) for p in prompts)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        pad_lefts = []
        for i, p in enumerate(prompts):
            L = len(p["input_ids"])
            pad_left = max_len - L
            input_ids[i, pad_left:] = torch.tensor(p["input_ids"], dtype=torch.long)
            attn[i, pad_left:] = torch.tensor(p["attention_mask"], dtype=torch.long)
            pad_lefts.append(pad_left)
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        hs = out.hidden_states  # tuple of (L+1) tensors each [B, T, d]
        for i, p in enumerate(prompts):
            lad = position_ladder(p["input_ids"])
            ladders[start + i] = lad
            pad_left = pad_lefts[i]
            for pn in pos_names:
                pos_idx = pad_left + lad[pn]  # shift into padded coordinates
                vec = torch.stack([h[i, pos_idx].float().cpu() for h in hs], dim=0).numpy()
                positions_cache[pn][start + i] = vec
        del out, hs

        # Generate short response for refusal labelling. With LEFT-padding, the
        # generated tokens start at index `max_len` (right after the padded prompt).
        if generate:
            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    do_sample=False,
                    temperature=1.0,
                    max_new_tokens=args.gen_max_new_tokens,
                    pad_token_id=pad_id,
                    use_cache=True,
                )
            for i in range(len(batch)):
                # gen shape: [B, max_len + generated_len]. Take everything past max_len.
                new = gen[i, max_len:].tolist()
                while new and new[-1] == pad_id:
                    new.pop()
                text = tokenizer.decode(new, skip_special_tokens=True)
                responses[start + i] = text

        elapsed = time.time() - t0
        done = start + len(batch)
        rate = done / max(elapsed, 1e-3)
        eta = (N - done) / max(rate, 1e-6)
        print(
            f"[m_prep] fwd+gen {done}/{N}  ({rate:.2f} prompts/s  ETA {eta:.0f}s)",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "positions": positions_cache,
        "responses": responses,
        "ladders": ladders,
    }


def compute_diff_mean_directions(
    positions_cache: Dict[str, np.ndarray],
    pos_a_indices: np.ndarray,
    pos_b_indices: np.ndarray,
    pos_name: str,
) -> np.ndarray:
    """Return (L+1, d) direction = mean(class_a) - mean(class_b) at `pos_name`."""
    A = positions_cache[pos_name][pos_a_indices]  # [Na, L+1, d]
    B = positions_cache[pos_name][pos_b_indices]  # [Nb, L+1, d]
    d = A.mean(axis=0) - B.mean(axis=0)  # (L+1, d)
    return d


def score_direction_auroc(
    positions_cache_pos: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    layer: int,
    val_pos_idx: np.ndarray,
    val_neg_idx: np.ndarray,
) -> float:
    """Train logistic regression probe on projection along diff-mean at `layer`
    using train indices, score AUROC on val indices. Returns AUROC.
    """
    # Direction = mean(pos_train) - mean(neg_train)
    a_train = positions_cache_pos[pos_idx, layer, :]
    b_train = positions_cache_pos[neg_idx, layer, :]
    direction = a_train.mean(axis=0) - b_train.mean(axis=0)
    dn = np.linalg.norm(direction) + 1e-9
    u = direction / dn

    # Train probe on 1-d projection (matches the "linear probe on projection" idiom).
    X_tr = np.concatenate([a_train @ u, b_train @ u]).reshape(-1, 1)
    y_tr = np.concatenate([np.ones(len(a_train)), np.zeros(len(b_train))])
    if len(np.unique(y_tr)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)

    a_val = positions_cache_pos[val_pos_idx, layer, :] @ u
    b_val = positions_cache_pos[val_neg_idx, layer, :] @ u
    X_val = np.concatenate([a_val, b_val]).reshape(-1, 1)
    y_val = np.concatenate([np.ones(len(a_val)), np.zeros(len(b_val))])
    if len(np.unique(y_val)) < 2:
        return float("nan")
    scores = clf.decision_function(X_val)
    return float(roc_auc_score(y_val, scores))


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    harmful = load_harmful_behaviors(str(Path(args.advbench) / "harmful_behaviors.csv"))
    alpaca_path = str(Path(args.alpaca) / "data" / "train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    n = len(harmful)
    if args.sanity:
        n = min(12, n)
        harmful = harmful[:n]
    benign = load_alpaca_instructions(alpaca_path, n, seed=args.seed)
    assert len(harmful) == len(benign), (len(harmful), len(benign))
    print(f"[m_prep] N harmful = {len(harmful)}, N benign = {len(benign)}", flush=True)

    # 60% train (direction extraction) / 20% val (held-out AUROC) / 20% test (Claim 5)
    tr_idx, val_idx, test_idx = deterministic_split_indices(n, (0.6, 0.2, 0.2), seed=args.seed)
    dump_json(
        out_dir / "meta.json",
        {
            "seed": args.seed,
            "n_pairs": n,
            "sanity": args.sanity,
            "split": {"train": tr_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()},
            "model": args.model,
            "dtype": args.dtype,
        },
    )

    # -------------------------------------------------------------- forward
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"[m_prep] n_layers={n_layers}, d_model={d_model}", flush=True)

    print("[m_prep] extracting HARMFUL activations + short generation (for refusal label)…",
          flush=True)
    harmful_cache = cache_activations_and_generate(
        model, tokenizer, harmful, generate=True, args=args
    )

    print("[m_prep] extracting BENIGN activations + short generation…", flush=True)
    benign_cache = cache_activations_and_generate(
        model, tokenizer, benign, generate=True, args=args
    )

    # Save responses + refusal labels for the harmful side (r direction contrast).
    responses = []
    for i, (h, r) in enumerate(zip(harmful, harmful_cache["responses"])):
        refused = is_refusal(r or "")
        responses.append({
            "idx": int(i),
            "prompt": h,
            "response": r,
            "refused": bool(refused),
            "class": "harmful",
        })
    for i, (b, r) in enumerate(zip(benign, benign_cache["responses"])):
        refused = is_refusal(r or "")
        responses.append({
            "idx": int(i),
            "prompt": b,
            "response": r,
            "refused": bool(refused),
            "class": "benign",
        })
    with open(out_dir / "responses.jsonl", "w") as f:
        for row in responses:
            f.write(json.dumps(row) + "\n")

    refusal_stats = {
        "harmful_refusal_rate": float(np.mean([r["refused"] for r in responses if r["class"] == "harmful"])),
        "benign_refusal_rate": float(np.mean([r["refused"] for r in responses if r["class"] == "benign"])),
    }
    print(f"[m_prep] refusal-rate  harmful={refusal_stats['harmful_refusal_rate']:.3f}  benign={refusal_stats['benign_refusal_rate']:.3f}",
          flush=True)

    # Save cached activations at the six positions (compact fp16).
    print("[m_prep] saving cached activations…", flush=True)
    to_save = {
        "harmful": {pn: torch.from_numpy(harmful_cache["positions"][pn].astype(np.float16))
                    for pn in harmful_cache["positions"]},
        "benign": {pn: torch.from_numpy(benign_cache["positions"][pn].astype(np.float16))
                   for pn in benign_cache["positions"]},
    }
    torch.save(to_save, out_dir / "activations.pt")
    del harmful_cache["positions"]
    del benign_cache["positions"]
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------- sweep + probe
    # For h (harmfulness attribute): positive class = harmful, negative = benign,
    # scored at (layer, t_final_instr).
    # For r (refusal attribute):     positive class = refused prompts, negative = complied prompts,
    # scored at (layer, t_post_instr).
    # We keep it label-symmetric: refusal labels come from the greedy short-gen
    # refusal-string classifier; harmful/benign labels come from the dataset.
    print("[m_prep] scoring per (layer x position x attribute) AUROC…", flush=True)

    # Reload activations from the just-saved fp16 tensors so downstream code
    # exercises the same load path as M1-M5.
    acts = to_save

    # Reconstruct index arrays for the refusal-side contrast.
    # NOTE: per code-review round 1 (fix 6), we RESTRICT the refusal contrast
    # to the HARMFUL side only. The refusal direction r is meant to capture
    # "the model's decision to refuse a harmful request" — mixing in benign
    # rows (which nearly always get complied) makes r partly a "harmfulness
    # dimension" in disguise, weakening Claim 1's independence story.
    refused_h = np.array([r["refused"] for r in responses if r["class"] == "harmful"])
    refused_b = np.array([r["refused"] for r in responses if r["class"] == "benign"])
    all_refused_idx = np.where(refused_h)[0]          # harmful refused -> pos
    all_complied_idx = np.where(~refused_h)[0]        # harmful complied -> neg
    # NB indices are in [0, n) here (harmful side only), so downstream code
    # needs `all_refused_idx % n` to just be `all_refused_idx` — but we keep
    # the `% n` idiom for consistency (it's a no-op below).

    pos_names = list(acts["harmful"].keys())

    # For harmfulness scoring we treat harmful=1, benign=0 across full pool.
    harm_pos_idx = np.arange(0, n)
    harm_neg_idx = np.arange(n, 2 * n)

    # Build one (2n, L+1, d) tensor per position: [harmful; benign] in fp32 for probing.
    per_pos_stacked = {}
    for pn in pos_names:
        per_pos_stacked[pn] = np.concatenate(
            [acts["harmful"][pn].float().numpy(), acts["benign"][pn].float().numpy()], axis=0
        )

    tr_idx_stacked = np.concatenate([tr_idx, n + tr_idx])
    val_idx_stacked = np.concatenate([val_idx, n + val_idx])

    layer_sweep = list(range(n_layers + 1))
    if args.sanity:
        layer_sweep = list(range(0, n_layers + 1, 4))
    rows = []
    # Refusal contrast: use the CAA/Arditi recipe — the *refusal direction* is
    # what separates prompts that make the model refuse from prompts that make
    # it comply. On well-aligned Llama-3-8B-Instruct, bare-harmful ⇒ refuses
    # (≈98%) and bare-benign ⇒ complies (≈100%), so the harmful-vs-benign
    # contrast at t_post_instr IS the refusal direction with correct labels.
    # The stricter refused-vs-complied-within-harmful contrast is used ONLY
    # when the complied-harmful pool has ≥ 8 items (indicating enough natural
    # jailbreaks in the bare setting to give a legitimate within-attribute split).
    n_complied_harmful = int((~np.array([r["refused"] for r in responses if r["class"] == "harmful"])).sum())
    use_within_harm_refusal_contrast = n_complied_harmful >= 8
    print(f"[m_prep] refusal contrast: {'refused-vs-complied within harmful' if use_within_harm_refusal_contrast else 'harmful-vs-benign proxy (natural jailbreaks in bare mode too rare: '+str(n_complied_harmful)+')'}", flush=True)

    for pn in pos_names:
        stack = per_pos_stacked[pn]
        for layer in layer_sweep:
            # Harmfulness attribute
            auroc_h = score_direction_auroc(
                stack,
                pos_idx=harm_pos_idx[np.isin(harm_pos_idx, tr_idx)],
                neg_idx=harm_neg_idx[np.isin(harm_neg_idx - n, tr_idx)],
                layer=layer,
                val_pos_idx=harm_pos_idx[np.isin(harm_pos_idx, val_idx)],
                val_neg_idx=harm_neg_idx[np.isin(harm_neg_idx - n, val_idx)],
            )
            # Refusal attribute
            if use_within_harm_refusal_contrast:
                refused_train = all_refused_idx[np.isin(all_refused_idx % n, tr_idx)]
                complied_train = all_complied_idx[np.isin(all_complied_idx % n, tr_idx)]
                refused_val = all_refused_idx[np.isin(all_refused_idx % n, val_idx)]
                complied_val = all_complied_idx[np.isin(all_complied_idx % n, val_idx)]
                if len(refused_train) < 2 or len(complied_train) < 2:
                    auroc_r = float("nan")
                else:
                    auroc_r = score_direction_auroc(
                        stack,
                        pos_idx=refused_train,
                        neg_idx=complied_train,
                        layer=layer,
                        val_pos_idx=refused_val,
                        val_neg_idx=complied_val,
                    )
            else:
                # Proxy: use harmful (refuses) vs benign (complies) as refusal contrast.
                # This IS the standard CAA/Arditi refusal direction on this model class.
                auroc_r = score_direction_auroc(
                    stack,
                    pos_idx=harm_pos_idx[np.isin(harm_pos_idx, tr_idx)],
                    neg_idx=harm_neg_idx[np.isin(harm_neg_idx - n, tr_idx)],
                    layer=layer,
                    val_pos_idx=harm_pos_idx[np.isin(harm_pos_idx, val_idx)],
                    val_neg_idx=harm_neg_idx[np.isin(harm_neg_idx - n, val_idx)],
                )
            rows.append({
                "position": pn,
                "layer": layer,
                "auroc_harmfulness": auroc_h,
                "auroc_refusal": auroc_r,
            })
    df_probe = pd.DataFrame(rows)
    df_probe.to_csv(out_dir / "probe_auroc.csv", index=False)

    # -------------------------------------------------- pick best (l, pos)
    def _best(attr: str, target_pos: str, fallback_layer: int = None):
        sub = df_probe.dropna(subset=[attr])
        sub_anchor = sub[sub["position"] == target_pos]
        if sub_anchor.empty:
            sub_anchor = sub
        if sub_anchor.empty:
            # No AUROC data at all (refusal side is degenerate — e.g., sanity
            # slice with 0 or 1 refused prompt). Fall back to a mid-layer at
            # the anchor position and mark AUROC NaN.
            fb_layer = fallback_layer if fallback_layer is not None else n_layers // 2
            return {"layer": int(fb_layer), "position": target_pos, "auroc": float("nan")}
        row = sub_anchor.sort_values(attr, ascending=False).iloc[0]
        return {"layer": int(row["layer"]), "position": str(row["position"]),
                "auroc": float(row[attr])}

    best_h = _best("auroc_harmfulness", "t_final_instr")
    best_r = _best("auroc_refusal", "t_post_instr", fallback_layer=best_h["layer"])
    print(f"[m_prep] best h (harmfulness):  layer={best_h['layer']}  pos={best_h['position']}  AUROC={best_h['auroc']:.3f}", flush=True)
    print(f"[m_prep] best r (refusal):      layer={best_r['layer']}  pos={best_r['position']}  AUROC={best_r['auroc']:.3f}", flush=True)

    # ----------------- extract h and r directions at the chosen sites (fp32)
    stack_h = per_pos_stacked[best_h["position"]]  # (2n, L+1, d)
    stack_r_h = per_pos_stacked[best_r["position"]]
    # Direction from TRAIN split only.
    tr_h_pos = harm_pos_idx[np.isin(harm_pos_idx, tr_idx)]
    tr_h_neg = harm_neg_idx[np.isin(harm_neg_idx - n, tr_idx)]
    h_direction = stack_h[tr_h_pos, best_h["layer"], :].mean(axis=0) - stack_h[tr_h_neg, best_h["layer"], :].mean(axis=0)

    # r direction: harmful (refuses) − benign (complies) at t_post_instr is the
    # CAA/Arditi refusal direction when natural jailbreaks in bare mode are rare.
    if use_within_harm_refusal_contrast:
        tr_r_pos = all_refused_idx[np.isin(all_refused_idx % n, tr_idx)]
        tr_r_neg = all_complied_idx[np.isin(all_complied_idx % n, tr_idx)]
        r_direction = stack_r_h[tr_r_pos, best_r["layer"], :].mean(axis=0) - stack_r_h[tr_r_neg, best_r["layer"], :].mean(axis=0)
        r_direction_source = "refused-vs-complied within harmful"
    else:
        # Standard CAA/Arditi: r = mean(harmful) - mean(benign) at t_post_instr
        # (where the refusal decision is committed for the next-token generation).
        tr_r_pos = harm_pos_idx[np.isin(harm_pos_idx, tr_idx)]
        tr_r_neg = harm_neg_idx[np.isin(harm_neg_idx - n, tr_idx)]
        r_direction = stack_r_h[tr_r_pos, best_r["layer"], :].mean(axis=0) - stack_r_h[tr_r_neg, best_r["layer"], :].mean(axis=0)
        r_direction_source = "harmful-vs-benign proxy (natural jailbreaks in bare mode too rare)"
    print(f"[m_prep] r direction source: {r_direction_source}", flush=True)

    # Save as torch tensors alongside a per-layer dump of both directions for M2.
    torch.save({
        "h": torch.from_numpy(h_direction).float(),
        "r": torch.from_numpy(r_direction).float(),
        "best_h": best_h,
        "best_r": best_r,
    }, out_dir / "directions.pt")

    # Also save per-(layer x position) diff-mean directions in a compact list — M2 uses these.
    # Both directions share the harmful-vs-benign contrast (h at t_final_instr,
    # r at t_post_instr) UNLESS the within-harm refusal pool has enough data.
    per_lp_h = {}
    per_lp_r = {}
    for pn in pos_names:
        stack = per_pos_stacked[pn]
        for layer in layer_sweep:
            d_h = stack[tr_h_pos, layer, :].mean(axis=0) - stack[tr_h_neg, layer, :].mean(axis=0)
            per_lp_h[f"{pn}_L{layer}"] = d_h.astype(np.float32)
            if use_within_harm_refusal_contrast:
                d_r = stack[tr_r_pos, layer, :].mean(axis=0) - stack[tr_r_neg, layer, :].mean(axis=0)
            else:
                d_r = d_h  # same source for both when we fall back to the CAA/Arditi proxy
            per_lp_r[f"{pn}_L{layer}"] = d_r.astype(np.float32)
    np.savez_compressed(out_dir / "per_lp_directions.npz", **{"h_" + k: v for k, v in per_lp_h.items()},
                        **{"r_" + k: v for k, v in per_lp_r.items()})

    # Estimate sigma_proj for the chosen h and r on the training split (for α_σ reporting in M3).
    def _sigma_proj(stack, layer, direction):
        u = direction / (np.linalg.norm(direction) + 1e-9)
        proj = stack[:, layer, :] @ u
        return float(np.std(proj))
    sigma_h = _sigma_proj(stack_h, best_h["layer"], h_direction)
    sigma_r = _sigma_proj(stack_r_h, best_r["layer"], r_direction)

    dump_json(out_dir / "directions.json", {
        "best_h": {**best_h,
                   "n_train_pos": int(len(tr_h_pos)),
                   "n_train_neg": int(len(tr_h_neg)),
                   "sigma_proj": sigma_h,
                   "direction_norm": float(np.linalg.norm(h_direction)),
                   "contrast_source": "harmful_vs_benign_at_t_final_instr"},
        "best_r": {**best_r,
                   "n_train_pos": int(len(tr_r_pos)),
                   "n_train_neg": int(len(tr_r_neg)),
                   "sigma_proj": sigma_r,
                   "direction_norm": float(np.linalg.norm(r_direction)),
                   "contrast_source": r_direction_source},
        "refusal_stats": refusal_stats,
        "n_complied_harmful_in_bare": int(n_complied_harmful),
        "n_layers": int(n_layers),
        "d_model": int(d_model),
    })
    print("[m_prep] done. artifacts:", flush=True)
    for f in sorted(out_dir.iterdir()):
        print(" -", f)


if __name__ == "__main__":
    main()
