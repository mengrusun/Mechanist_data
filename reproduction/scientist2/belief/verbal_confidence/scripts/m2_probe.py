#!/usr/bin/env python3
"""
M2 — Per-position × per-layer linear probe (ridge regression) for verbalized confidence.

Loads M1's activations.h5 and per-item verbal_conf targets. For each cell
(position ∈ {E0..E4}, layer ∈ 12-layer grid), trains a Ridge regressor on a
60/20/20 split of items (train/eval/test) and reports test R², averaged
across seeds.

Baselines (in the same script):
  --baseline log_prob_only    — regress verbal_conf on mean answer log-prob only
  --baseline conf_gen_position — probe at the confidence-generation position (C0)
                                 across layers (should not beat post-answer cells if
                                 the value is *already* determined post-answer)
  --baseline shuffled         — chance baseline (labels permuted)

Emits:
  results/m2/pos<pos>_L<L>_<probe>.json  (one file per cell)
  results/m2/heatmap.png                 (R² heatmap over cells)
  results/m2/top_k_sites.json            (top-K (position, layer) candidates by
                                          Δ-R² over log-prob-only baseline)
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc_common import CONF_GEN_LABEL, LAYER_GRID


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True,
                   help="Root of the M1 cache. Contains one subdir per seed (e.g. seed42/T0/).")
    p.add_argument("--out_dir", required=True,
                   help="Where to write per-cell JSONs, the heatmap PNG, and top_k_sites.json.")
    p.add_argument("--seeds", default="42,123,2024")
    p.add_argument("--template", default="T0")
    p.add_argument("--positions", default="E0,E1,E2,E3,E4")
    p.add_argument("--layers", default=",".join(str(x) for x in LAYER_GRID))
    p.add_argument("--probe_type", default="ridge", choices=["ridge"])
    p.add_argument("--train_split", type=float, default=0.6)
    p.add_argument("--eval_split", type=float, default=0.2)
    p.add_argument("--test_split", type=float, default=0.2)
    p.add_argument("--alphas", default="0.1,1.0,10.0,100.0",
                   help="Ridge regularization strengths to search on the eval split.")
    p.add_argument("--top_k", type=int, default=3,
                   help="Top-K (position, layer) cells to hand off to M3/M4/M5.")
    return p.parse_args()


def load_seed_data(cache_root: str, seed: int, template: str):
    """Return (items_dict[], activations_h5_path) for one seed."""
    seed_dir = Path(cache_root) / f"seed{seed}" / template
    items_path = seed_dir / "items.jsonl"
    acts_path = seed_dir / "activations.h5"
    if not items_path.exists() or not acts_path.exists():
        raise FileNotFoundError(f"Missing M1 output for seed {seed}: {seed_dir}")

    items = []
    with items_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items, str(acts_path)


def build_splits(n: int, train_frac: float, eval_frac: float, test_frac: float,
                 seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-level split (used only for baselines with no group leakage risk)."""
    rng = np.random.RandomState(seed + 999)
    idx = rng.permutation(n)
    n_train = int(round(n * train_frac))
    n_eval = int(round(n * eval_frac))
    tr = idx[:n_train]
    ev = idx[n_train : n_train + n_eval]
    te = idx[n_train + n_eval :]
    return tr, ev, te


def build_grouped_splits(qids: List[str], train_frac: float, eval_frac: float,
                         test_frac: float, seed: int
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group-aware split: partition unique question_ids across train/eval/test so
    that seed-duplicated items never leak across splits.
    """
    unique_qids = list(dict.fromkeys(qids))  # stable order
    rng = np.random.RandomState(seed + 999)
    perm = rng.permutation(len(unique_qids))
    n_tr = int(round(len(unique_qids) * train_frac))
    n_ev = int(round(len(unique_qids) * eval_frac))
    tr_qids = {unique_qids[j] for j in perm[:n_tr]}
    ev_qids = {unique_qids[j] for j in perm[n_tr : n_tr + n_ev]}
    te_qids = {unique_qids[j] for j in perm[n_tr + n_ev :]}
    tr = np.array([i for i, q in enumerate(qids) if q in tr_qids], dtype=np.int64)
    ev = np.array([i for i, q in enumerate(qids) if q in ev_qids], dtype=np.int64)
    te = np.array([i for i, q in enumerate(qids) if q in te_qids], dtype=np.int64)
    return tr, ev, te


def ridge_fit_eval(X_tr, y_tr, X_ev, y_ev, X_te, y_te, alphas: List[float]):
    """Fit Ridge with best alpha chosen on eval split; return dict with fit params + metrics."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr

    best = None
    for a in alphas:
        m = Ridge(alpha=a, solver="auto")
        m.fit(X_tr, y_tr)
        r2_ev = r2_score(y_ev, m.predict(X_ev))
        if best is None or r2_ev > best[0]:
            best = (r2_ev, a, m)
    r2_ev, best_alpha, m_best = best

    y_pred_te = m_best.predict(X_te)
    r2_te = r2_score(y_te, y_pred_te)
    y_pred_tr = m_best.predict(X_tr)
    r2_tr = r2_score(y_tr, y_pred_tr)
    rho_te, _ = spearmanr(y_te, y_pred_te)

    return {
        "best_alpha": float(best_alpha),
        "r2_train": float(r2_tr),
        "r2_eval": float(r2_ev),
        "r2_test": float(r2_te),
        "spearman_test": float(rho_te) if not np.isnan(rho_te) else 0.0,
        "n_train": int(len(y_tr)),
        "n_eval": int(len(y_ev)),
        "n_test": int(len(y_te)),
    }


def gather_activations_for_cell(cache_root: str, seeds: List[int], template: str,
                                position: str, layer: int) -> Tuple[np.ndarray, np.ndarray, List[float], List[str]]:
    """
    Concatenate (activations, verbal_conf, mean_answer_logprob, question_id) across seeds
    for a specific (position, layer). Items with unparsed verbal_conf are filtered out.
    Returns (X, y, lp, qids) — qids used downstream to build question-grouped splits so
    the same question_id never lands in both train and test across seeds.
    """
    X_all, y_all, lp_all, qids = [], [], [], []
    for s in seeds:
        items, acts_path = load_seed_data(cache_root, s, template)
        with h5py.File(acts_path, "r") as h5:
            acts = h5[position][str(layer)][()]  # (n_items, hidden)
        for i, itm in enumerate(items):
            if itm["verbal_conf"] is None:
                continue
            X_all.append(acts[i])
            y_all.append(float(itm["verbal_conf"]))
            lp_all.append(
                float(np.mean(itm["answer_token_logprobs"])) if itm["answer_token_logprobs"] else 0.0
            )
            qids.append(itm["question_id"])
    return np.stack(X_all, axis=0), np.array(y_all, dtype=np.float32), lp_all, qids


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    positions = args.positions.split(",")
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Verify M1 outputs exist for every seed --------------------------
    for s in seeds:
        _ = load_seed_data(args.cache_dir, s, args.template)

    # ---- Main sweep: (position, layer) probe on residual-stream activations
    heatmap = np.full((len(positions), len(layers)), np.nan, dtype=np.float32)
    heatmap_spearman = np.full((len(positions), len(layers)), np.nan, dtype=np.float32)
    all_cells = []

    for pi, pos in enumerate(positions):
        for li, L in enumerate(layers):
            X, y, _, qids = gather_activations_for_cell(args.cache_dir, seeds, args.template, pos, L)
            # Group-aware split so the same question_id never leaks across seeds.
            tr, ev, te = build_grouped_splits(qids, args.train_split, args.eval_split, args.test_split, seed=0)
            res = ridge_fit_eval(X[tr], y[tr], X[ev], y[ev], X[te], y[te], alphas)
            res.update({"position": pos, "layer": L, "probe": "ridge",
                        "n_total": int(len(X))})
            with (out_dir / f"pos{pos}_L{L}_ridge.json").open("w") as f:
                json.dump(res, f, indent=2)
            heatmap[pi, li] = res["r2_test"]
            heatmap_spearman[pi, li] = res["spearman_test"]
            all_cells.append(res)
            print(f"[m2] {pos:>2} L{L:02d} R²_test={res['r2_test']:+.3f}  ρ={res['spearman_test']:+.3f}  α*={res['best_alpha']:.2f}",
                  flush=True)

    # ---- Baseline 1: log-prob only ---------------------------------------
    lp_all_seeds = []
    y_all_seeds = []
    qids_all_seeds = []
    for s in seeds:
        items, _ = load_seed_data(args.cache_dir, s, args.template)
        for itm in items:
            if itm["verbal_conf"] is None:
                continue
            y_all_seeds.append(float(itm["verbal_conf"]))
            lp_all_seeds.append(
                float(np.mean(itm["answer_token_logprobs"])) if itm["answer_token_logprobs"] else 0.0
            )
            qids_all_seeds.append(itm["question_id"])
    lp = np.array(lp_all_seeds, dtype=np.float32).reshape(-1, 1)
    y = np.array(y_all_seeds, dtype=np.float32)
    tr, ev, te = build_grouped_splits(qids_all_seeds, args.train_split, args.eval_split, args.test_split, seed=0)
    lp_res = ridge_fit_eval(lp[tr], y[tr], lp[ev], y[ev], lp[te], y[te], alphas)
    lp_res.update({"baseline": "log_prob_only"})
    with (out_dir / "baseline_log_prob_only.json").open("w") as f:
        json.dump(lp_res, f, indent=2)
    baseline_log_prob_r2 = lp_res["r2_test"]
    print(f"[m2] baseline log_prob_only R²_test={baseline_log_prob_r2:+.3f}", flush=True)

    # ---- Baseline 2: conf_gen_position (C0) probe at each layer ---------
    conf_gen_r2 = {}
    for L in layers:
        X, yy, _, qids_c = gather_activations_for_cell(args.cache_dir, seeds, args.template, CONF_GEN_LABEL, L)
        tr, ev, te = build_grouped_splits(qids_c, args.train_split, args.eval_split, args.test_split, seed=0)
        res = ridge_fit_eval(X[tr], yy[tr], X[ev], yy[ev], X[te], yy[te], alphas)
        res.update({"baseline": "conf_gen_position", "position": CONF_GEN_LABEL, "layer": L})
        with (out_dir / f"baseline_conf_gen_L{L}.json").open("w") as f:
            json.dump(res, f, indent=2)
        conf_gen_r2[L] = res["r2_test"]
        print(f"[m2] baseline conf_gen (C0) L{L:02d} R²_test={res['r2_test']:+.3f}", flush=True)

    # ---- Baseline 3: shuffled (chance) at one central cell (spot check) --
    rng = np.random.RandomState(12345)
    X_ctr, y_ctr, _, qids_ctr = gather_activations_for_cell(args.cache_dir, seeds, args.template, "E0", 30)
    y_shuf = y_ctr.copy()
    rng.shuffle(y_shuf)
    tr, ev, te = build_grouped_splits(qids_ctr, args.train_split, args.eval_split, args.test_split, seed=0)
    shuf_res = ridge_fit_eval(X_ctr[tr], y_shuf[tr], X_ctr[ev], y_shuf[ev], X_ctr[te], y_shuf[te], alphas)
    shuf_res.update({"baseline": "shuffled_at_E0_L30"})
    with (out_dir / "baseline_shuffled.json").open("w") as f:
        json.dump(shuf_res, f, indent=2)
    print(f"[m2] baseline shuffled R²_test={shuf_res['r2_test']:+.3f}  (chance)", flush=True)

    # ---- Top-K cells (post-answer positions only) by Δ over log_prob_only baseline
    delta_cells = []
    for c in all_cells:
        c_delta = c["r2_test"] - baseline_log_prob_r2
        delta_cells.append({
            "position": c["position"],
            "layer": c["layer"],
            "r2_test": c["r2_test"],
            "spearman_test": c["spearman_test"],
            "delta_r2_vs_logprob": c_delta,
        })
    # Restrict to post-answer positions (E0..E4) for M3/M4/M5 hand-off
    post_ans = [c for c in delta_cells if c["position"].startswith("E")]
    post_ans_sorted = sorted(post_ans, key=lambda x: x["delta_r2_vs_logprob"], reverse=True)
    top_k = post_ans_sorted[: args.top_k]

    # M2 P1 pass criterion: at least one (post-answer, mid-late) cell with
    # ΔR² ≥ 0.05 over log_prob_only AND R² ≥ conf_gen probe at same layer.
    mid_late = [c for c in post_ans_sorted if c["layer"] >= 20]
    p1_pass_cells = []
    for c in mid_late:
        conf_gen_at_L = conf_gen_r2.get(c["layer"], -np.inf)
        p1_ok = (c["delta_r2_vs_logprob"] >= 0.05) and (c["r2_test"] >= conf_gen_at_L)
        if p1_ok:
            p1_pass_cells.append(c)

    summary = {
        "top_k": top_k,
        "baseline_log_prob_only_r2": baseline_log_prob_r2,
        "baseline_conf_gen_r2_by_layer": {str(k): v for k, v in conf_gen_r2.items()},
        "baseline_shuffled_r2": shuf_res["r2_test"],
        "p1_pass": len(p1_pass_cells) > 0,
        "p1_pass_cells": p1_pass_cells,
    }
    with (out_dir / "top_k_sites.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # ---- Heatmap PNG -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(heatmap, aspect="auto", cmap="viridis", vmin=-0.05, vmax=max(0.5, np.nanmax(heatmap)))
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels(positions)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Position")
        ax.set_title("M2: probe R²_test (verbalized confidence)")
        for pi in range(len(positions)):
            for li in range(len(layers)):
                v = heatmap[pi, li]
                ax.text(li, pi, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.4 else "black", fontsize=7)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "heatmap.png", dpi=150)
        plt.close(fig)
        print(f"[m2] wrote heatmap.png", flush=True)
    except Exception as e:
        print(f"[m2] heatmap plot failed: {e}", flush=True)

    print(f"[m2] TOP-K sites: {json.dumps(top_k, indent=2)}", flush=True)
    print(f"[m2] P1 pass? {summary['p1_pass']} ({len(p1_pass_cells)} cells clear threshold)",
          flush=True)


if __name__ == "__main__":
    main()
