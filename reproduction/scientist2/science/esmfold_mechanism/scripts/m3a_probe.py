#!/usr/bin/env python3
"""M3a — Linear probe of residue charge on early trunk-block s tensors.

Steps:
  1. Load heldout_probe chains from manifest.
  2. Do ONE ESMFold forward per chain, capture s at blocks {0, 2, 4, 6}.
  3. Split chains 8:1:1 into train / val / test.
  4. For each sampled block, train a 3-class linear probe (LogisticRegression) on
     per-residue s vectors → predict charge class ∈ {neg, pos, neut}.
  5. Evaluate balanced accuracy + AUROC (macro) on test set.
  6. Compute a 1000-permutation null for balanced accuracy (label shuffle on test).
  7. Pick the strongest block; extract v_charge = w_pos - w_neg (L2-normalized).
  8. Save results/M3a/probe_by_block.jsonl, best_block.json, v_charge.npy, permutation_null.json.

Runtime: ~250 chains × 3s/forward = 12 min; probe training + permutation ~5 min total. ~0.5h ok.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

from esmfold_lib import (  # noqa: E402
    PROJECT_ROOT, MANIFEST_PATH, load_esmfold, esmfold_forward,
    charge_class_ids, S_HIDDEN, set_seed, write_jsonl, write_json,
)


def collect_s_at_blocks(model, tok, seq: str, blocks: List[int]) -> Dict[int, np.ndarray]:
    """One forward; capture s (1, L, 1024) entering each requested block. Returns dict:
    {block_idx: (L, 1024) numpy array}.
    """
    stores: Dict[int, np.ndarray] = {}
    handles = []

    def make_hook(k):
        def pre_hook(module, args, kwargs):
            stores[k] = args[0].detach()[0].cpu().numpy()  # (L, 1024)
            return None
        return pre_hook

    try:
        for k in blocks:
            h = model.trunk.blocks[k].register_forward_pre_hook(make_hook(k), with_kwargs=True)
            handles.append(h)
        with torch.no_grad():
            _ = esmfold_forward(model, tok, seq)
    finally:
        for h in handles:
            h.remove()
    return stores


def train_probe_and_score(X_tr, y_tr, X_va, y_va, X_te, y_te, seed=42) -> Dict:
    """Train multinomial LogisticRegression, return metrics + weights."""
    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=1000,
        random_state=seed, class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)
    y_pred_va = clf.predict(X_va)
    y_pred_te = clf.predict(X_te)
    proba_te = clf.predict_proba(X_te)
    val_ba = balanced_accuracy_score(y_va, y_pred_va)
    te_ba = balanced_accuracy_score(y_te, y_pred_te)
    # AUROC macro across all 3 classes (one-vs-rest)
    try:
        # Ensure y_te has ≥2 classes represented
        classes_present = set(y_te)
        if len(classes_present) < 2:
            te_auroc = float("nan")
        else:
            te_auroc = roc_auc_score(y_te, proba_te, multi_class="ovr", average="macro")
    except ValueError:
        te_auroc = float("nan")
    return {
        "val_balanced_acc": float(val_ba),
        "test_balanced_acc": float(te_ba),
        "test_auroc_macro": float(te_auroc),
        "weights": clf.coef_,   # (3, 1024)  order = clf.classes_
        "classes": clf.classes_.tolist(),
    }


def permutation_null_ba(X_te, y_te, clf, n_shuffles: int = 1000, seed: int = 42) -> np.ndarray:
    """Shuffle test labels n_shuffles times, compute balanced_acc under the current clf's
    predictions on X_te (which stay fixed). Return array of nulls of length n_shuffles.
    """
    rng = np.random.default_rng(seed)
    y_pred_te = clf.predict(X_te)
    out = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        y_shuff = rng.permutation(y_te)
        out[i] = balanced_accuracy_score(y_shuff, y_pred_te)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--sampled-blocks", type=str, default="0,2,4,6")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "M3a")
    p.add_argument("--n-shuffles", type=int, default=1000)
    p.add_argument("--gpu-id", type=str, default="0")
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_id)
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    blocks = [int(x) for x in args.sampled_blocks.split(",")]
    print(f"[m3a] sampled blocks: {blocks}", flush=True)

    # Load heldout_probe chains
    chains = []
    with open(args.manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "heldout_probe":
                chains.append(r)
    print(f"[m3a] {len(chains)} heldout_probe chains", flush=True)
    if len(chains) < 20:
        # fallback: also include calibration if too few
        with open(args.manifest) as f:
            for line in f:
                r = json.loads(line)
                if r.get("split") == "calibration":
                    chains.append(r)
        print(f"[m3a] included calibration too: {len(chains)}", flush=True)

    # Deterministic shuffle + split 8:1:1 by chain
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(chains))
    n_tr = int(0.8 * len(chains))
    n_va = int(0.1 * len(chains))
    tr_chains = [chains[i] for i in idx[:n_tr]]
    va_chains = [chains[i] for i in idx[n_tr:n_tr + n_va]]
    te_chains = [chains[i] for i in idx[n_tr + n_va:]]
    print(f"[m3a] split: train={len(tr_chains)} val={len(va_chains)} test={len(te_chains)}",
          flush=True)

    print(f"[m3a] loading model...", flush=True)
    model, tok, _ = load_esmfold(device="cuda", dtype=torch.float32)
    print(f"[m3a] model loaded", flush=True)

    # Collect s-tensors and labels
    def collect_batch(chains_list, name: str):
        Xs = {k: [] for k in blocks}
        ys = []
        t0 = time.time()
        for i, c in enumerate(chains_list):
            if i % 20 == 0 and i > 0:
                elapsed = time.time() - t0
                eta = elapsed / i * (len(chains_list) - i) / 60
                print(f"[m3a/{name}] {i}/{len(chains_list)} eta {eta:.1f}min", flush=True)
            seq = c["seq"]
            try:
                stores = collect_s_at_blocks(model, tok, seq, blocks)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[m3a/{name}] OOM on {c['pdb_id']} L={len(seq)}", flush=True)
                continue
            except Exception as e:
                print(f"[m3a/{name}] err on {c['pdb_id']}: {e}", flush=True)
                continue
            L = len(seq)
            labels = charge_class_ids(seq)  # (L,)
            # For each block store the per-residue vectors
            valid_mask = np.arange(L)  # all residues
            for k in blocks:
                Xs[k].append(stores[k][valid_mask])  # (L, 1024)
            ys.append(labels)
        # Concatenate
        Xs_arr = {k: np.concatenate(Xs[k], axis=0) for k in blocks}
        ys_arr = np.concatenate(ys, axis=0)
        return Xs_arr, ys_arr

    Xtr, ytr = collect_batch(tr_chains, "train")
    Xva, yva = collect_batch(va_chains, "val")
    Xte, yte = collect_batch(te_chains, "test")
    print(f"[m3a] shapes: tr {Xtr[blocks[0]].shape}  va {Xva[blocks[0]].shape}  te {Xte[blocks[0]].shape}", flush=True)
    print(f"[m3a] label dist train: {np.bincount(ytr)}, test: {np.bincount(yte)}", flush=True)

    # Train per-block probe
    per_block_results = []
    best_ba = -1.0
    best_block = None
    best_probe = None
    best_v_charge = None
    for k in blocks:
        print(f"[m3a/probe] training probe at block {k}", flush=True)
        r = train_probe_and_score(Xtr[k], ytr, Xva[k], yva, Xte[k], yte, seed=args.seed)
        r_ = {kk: v for kk, v in r.items() if kk not in ("weights", "classes")}
        r_["block"] = k
        r_["classes"] = r["classes"]
        per_block_results.append(r_)
        # Compute permutation null p-value on this block
        # (retrain classifier on the same block for the null)
        clf = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=1000,
            random_state=args.seed, class_weight="balanced",
        )
        clf.fit(Xtr[k], ytr)
        nulls = permutation_null_ba(Xte[k], yte, clf, n_shuffles=args.n_shuffles, seed=args.seed)
        p_val = float((nulls >= r["test_balanced_acc"]).sum() + 1) / (args.n_shuffles + 1)
        r_["permutation_p"] = p_val
        r_["null_mean"] = float(nulls.mean())
        r_["null_std"] = float(nulls.std())
        print(f"[m3a/probe] block {k}: test_ba={r['test_balanced_acc']:.4f} "
              f"AUROC={r['test_auroc_macro']:.4f} perm_p={p_val:.4g}", flush=True)
        if r["test_balanced_acc"] > best_ba:
            best_ba = r["test_balanced_acc"]
            best_block = k
            best_probe = r["weights"]  # (3, 1024)
            best_classes = r["classes"]
            best_v_charge = None  # will compute after picking

    # Pick best block, extract v_charge = w[pos] - w[neg], L2-normalize.
    # classes returned by LogisticRegression are the sorted unique labels: [0=neg, 1=pos, 2=neut].
    class_to_row = {c: i for i, c in enumerate(best_classes)}
    if 0 in class_to_row and 1 in class_to_row:
        w_pos = best_probe[class_to_row[1]]  # (1024,)
        w_neg = best_probe[class_to_row[0]]
        v_charge = w_pos - w_neg
        norm = np.linalg.norm(v_charge)
        if norm > 0:
            v_charge = v_charge / norm
    else:
        v_charge = np.zeros(S_HIDDEN, dtype=np.float32)

    # Save all
    write_jsonl(args.out / "probe_by_block.jsonl", per_block_results)
    write_json(args.out / "best_block.json", {
        "best_block": best_block, "best_test_balanced_acc": best_ba,
        "sampled_blocks": blocks,
    })
    np.save(args.out / "v_charge.npy", v_charge.astype(np.float32))
    write_json(args.out / "m3a_summary.json", {
        "sampled_blocks": blocks,
        "n_chains_train": len(tr_chains),
        "n_chains_val": len(va_chains),
        "n_chains_test": len(te_chains),
        "n_train_residues": int(Xtr[blocks[0]].shape[0]),
        "n_test_residues": int(Xte[blocks[0]].shape[0]),
        "best_block": best_block, "best_test_balanced_acc": best_ba,
        "per_block": per_block_results,
        "v_charge_L2_after_normalize": 1.0,
        "permutation_shuffles": args.n_shuffles,
    })
    print(f"[m3a] DONE. best_block={best_block} test_ba={best_ba:.4f}", flush=True)


if __name__ == "__main__":
    main()
