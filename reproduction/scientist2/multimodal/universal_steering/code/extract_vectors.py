"""Extract per-concept RFM concept vectors.

For each concept in --concepts:
  1. Load paired data (data/paired/<concept>.jsonl)
  2. Cache last-token activations at all 32 blocks (Llama-3.1-8B-Instruct)
  3. Fit linear probe per block on train / evaluate on val — pick best block
  4. Run RFM at chosen block → concept vector v_c
  5. Save: runs/<run_id>/concept_<concept>/{v_c.npy, screen.json, rfm.json}

Also caches raw activations to disk for reuse (all 32 blocks × N samples × d_model).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from rfm_core import fit_linear_probe, probe_accuracy, extract_rfm, caa_mean_diff, save_vector
from model_utils import load_model, cache_last_token_activations, free_cuda

WORK_DIR = Path("/data/zhenqian/Reproduction1/mechanica/multimodal/universal_steering")


def read_jsonl(path: Path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="+", required=True)
    ap.add_argument("--run-id", default="B1_extract_vectors")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--train-n", type=int, default=300)
    ap.add_argument("--val-n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--rfm-iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--pin-block", type=int, default=None,
                    help="If set, use this block instead of the probe-picked best block")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else WORK_DIR / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] out_dir={out_dir}")
    t0 = time.time()

    print(f"[extract] loading Llama-3.1-8B-Instruct ({args.dtype}) on {args.device}...")
    model, tokenizer, cfg = load_model(dtype=args.dtype, device=args.device)
    print(f"[extract]   num_blocks={cfg['num_blocks']}, d_model={cfg['d_model']}")

    results = {"concepts": {}, "config": vars(args), "start_time": time.time(),
               "model_num_blocks": cfg["num_blocks"], "model_d_model": cfg["d_model"]}

    for concept in args.concepts:
        paired_path = WORK_DIR / "data" / "paired" / f"{concept}.jsonl"
        if not paired_path.exists():
            print(f"[extract] SKIP {concept}: file not found {paired_path}")
            continue
        rows = read_jsonl(paired_path)
        # Shuffle deterministically
        rng = np.random.default_rng(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.train_n + args.val_n]
        statements = [r["statement"] for r in rows]
        labels = np.array([int(r["label"]) for r in rows], dtype=np.float64)

        print(f"[extract] {concept}: caching activations for {len(statements)} sequences ...")
        t_cache = time.time()
        acts = cache_last_token_activations(
            model, tokenizer, statements,
            batch_size=args.batch_size, max_length=args.max_length, device=args.device,
        )  # (n, num_blocks, d)
        print(f"[extract]   activation cache took {time.time()-t_cache:.1f}s, shape={acts.shape}")

        # Split into train / val
        train_acts = acts[: args.train_n]
        train_labels = labels[: args.train_n]
        val_acts = acts[args.train_n:]
        val_labels = labels[args.train_n:]

        # Per-block linear probe screen
        per_block_acc = []
        per_block_w = []
        for l in range(cfg["num_blocks"]):
            X_tr = train_acts[:, l, :].astype(np.float64)
            X_va = val_acts[:, l, :].astype(np.float64)
            w_vec, w_b = fit_linear_probe(X_tr, train_labels, ridge=1e-2)
            acc = probe_accuracy((w_vec, w_b), X_va, val_labels)
            per_block_acc.append(acc)
            per_block_w.append(np.concatenate([w_vec, [w_b]]).astype(np.float32))
        if args.pin_block is not None:
            best_block = int(args.pin_block)
            best_acc = float(per_block_acc[best_block])
            print(f"[extract]   PINNED block = {best_block} (val_acc={best_acc:.3f})")
        else:
            best_block = int(np.argmax(per_block_acc))
            best_acc = float(per_block_acc[best_block])
        print(f"[extract]   per-block probe accs (min/mean/max): "
              f"{min(per_block_acc):.3f}/{np.mean(per_block_acc):.3f}/{max(per_block_acc):.3f}")
        print(f"[extract]   BEST block = {best_block}, val_acc={best_acc:.3f}")

        # RFM at best block
        X_tr_best = train_acts[:, best_block, :].astype(np.float64)
        t_rfm = time.time()
        rfm = extract_rfm(X_tr_best, train_labels, n_iters=args.rfm_iters, seed=args.seed, verbose=True)
        rfm_time = time.time() - t_rfm
        print(f"[extract]   RFM took {rfm_time:.1f}s, top_ratio={rfm['top_ratio']:.3f}")

        # CAA baseline direction
        v_caa = caa_mean_diff(X_tr_best, train_labels)
        cos_rfm_caa = float(np.abs(rfm["v_c"] @ v_caa))
        print(f"[extract]   cos(v_rfm, v_caa) = {cos_rfm_caa:.4f}")

        # Val acc of RFM vector as a linear classifier (sanity).
        # Subtract training mean (per block) to give the direction a zero-threshold.
        X_va_best = val_acts[:, best_block, :].astype(np.float64)
        mu_train = X_tr_best.mean(axis=0)
        proj_val = (X_va_best - mu_train) @ rfm["v_c"]
        rfm_val_acc = float(((np.sign(proj_val) * val_labels) > 0).mean())
        proj_val_caa = (X_va_best - mu_train) @ v_caa
        caa_val_acc = float(((np.sign(proj_val_caa) * val_labels) > 0).mean())

        # Save concept vector + metadata
        concept_dir = out_dir / f"concept_{concept}"
        concept_dir.mkdir(parents=True, exist_ok=True)
        save_vector(concept_dir / "v_c.npy", rfm["v_c"], {
            "concept": concept, "best_block": best_block,
            "probe_val_acc_best_block": best_acc,
            "rfm_val_acc_best_block": rfm_val_acc,
            "caa_val_acc_best_block": caa_val_acc,
            "cos_rfm_caa": cos_rfm_caa, "rfm_top_ratio": rfm["top_ratio"],
            "rfm_eigvals_top5": rfm["eigvals_top5"].tolist(),
            "rfm_cos_history": rfm["cos_history"],
            "rfm_bandwidth": rfm["bandwidth"], "rfm_iters": args.rfm_iters,
            "d_model": cfg["d_model"], "num_blocks": cfg["num_blocks"],
            "n_train": int(args.train_n), "n_val": int(args.val_n),
        })
        np.save(concept_dir / "v_caa.npy", v_caa)
        np.save(concept_dir / "per_block_acc.npy", np.array(per_block_acc, dtype=np.float32))
        # Save best-block probe weight (for C5 style monitoring re-use)
        np.save(concept_dir / f"w_probe_block{best_block}.npy", per_block_w[best_block].astype(np.float32))

        results["concepts"][concept] = {
            "best_block": best_block,
            "probe_val_acc": best_acc,
            "rfm_val_acc": rfm_val_acc, "caa_val_acc": caa_val_acc,
            "cos_rfm_caa": cos_rfm_caa, "rfm_top_ratio": rfm["top_ratio"],
            "per_block_acc": per_block_acc,
            "rfm_time_s": rfm_time, "cache_shape": list(acts.shape),
            "concept_dir": str(concept_dir),
        }

        # Clean up per-concept
        del acts, train_acts, val_acts, X_tr_best, X_va_best
        free_cuda()

    results["end_time"] = time.time()
    results["total_time_s"] = time.time() - t0
    with open(out_dir / "extract_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[extract] DONE in {results['total_time_s']:.1f}s → {out_dir/'extract_summary.json'}")


if __name__ == "__main__":
    main()
