"""M1: Locate harmful-subspace sites in the base LM.

Per EXPERIMENT_PLAN.md M1 + FINAL_PROPOSAL.md §5.4:
  - For each layer L (0..num_layers-1), compute:
      * mean-difference direction d_h^L = mean(a_L(h)) - mean(a_L(b))
        at the last non-pad token of the prompt (chat-template formatted).
      * Layer-wise linear-probe AUC on held-out pairs.
  - Select sites S = top-k layers (default k=6) with AUC in a contiguous
    middle band (mid-to-late layers per experiment-tips/steering-block-selection).

Outputs (artifacts/m1/):
  - auc_per_layer.json — {layer: auc}
  - directions.pt      — {layer: unit vector (hidden_size,)}
  - sites.json         — chosen layers S

Success criterion (records C1a identifiability half):
  At least 3 mid-to-late layers with probe AUC > 0.8 on held-out pairs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils import (
    PROJECT_ROOT,
    ARTIFACTS_DIR,
    get_hidden_size,
    get_last_token_hidden_states,
    get_num_layers,
    gpu_ids_from_env,
    load_causal_lm,
    load_paired_prompts,
    normalize,
    resolve_base_lm,
    save_json,
    set_seed,
    write_cost,
)


def probe_auc(train_h: np.ndarray, train_b: np.ndarray, held_h: np.ndarray, held_b: np.ndarray) -> float:
    """Fit a linear probe on train, evaluate AUC on held-out.

    Labels: harmful=1, benign=0.
    """
    X_tr = np.concatenate([train_h, train_b], axis=0)
    y_tr = np.concatenate([np.ones(len(train_h)), np.zeros(len(train_b))])
    X_ho = np.concatenate([held_h, held_b], axis=0)
    y_ho = np.concatenate([np.ones(len(held_h)), np.zeros(len(held_b))])

    clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    scores = clf.decision_function(X_ho)
    return float(roc_auc_score(y_ho, scores))


def choose_sites(auc_per_layer: dict, k: int = 6, midband: tuple[float, float] = (0.3, 0.75)) -> list[int]:
    """Pick top-k layers by AUC constrained to a contiguous mid-to-late band.

    - midband = (lo, hi) as fractions of total depth.
    - Take contiguous block of length k with maximum sum of AUCs within the band.
    """
    layers_sorted = sorted(auc_per_layer.keys())
    n = len(layers_sorted)
    lo = int(midband[0] * n)
    hi = int(midband[1] * n)
    # Ensure at least k candidates
    if hi - lo < k:
        hi = min(n, lo + k)
        if hi - lo < k:
            lo = max(0, hi - k)
    aucs = [auc_per_layer[L] for L in layers_sorted]
    best_start, best_sum = lo, -1.0
    for start in range(lo, hi - k + 1):
        s = sum(aucs[start:start + k])
        if s > best_sum:
            best_sum = s
            best_start = start
    return layers_sorted[best_start:best_start + k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-train", type=Path, default=PROJECT_ROOT / "data" / "paired_train.jsonl")
    parser.add_argument("--pairs-held", type=Path, default=PROJECT_ROOT / "data" / "paired_heldout.jsonl")
    parser.add_argument("--n-train-cap", type=int, default=384, help="how many train pairs to actually use for direction/probe fit")
    parser.add_argument("--k-sites", type=int, default=6)
    parser.add_argument("--outdir", type=Path, default=ARTIFACTS_DIR / "m1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None, help="runs/<id>/ for cost.json")
    args = parser.parse_args()

    set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.run_dir is not None:
        args.run_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"[m1] Loading base model...")
    model_path = resolve_base_lm()
    model, tokenizer = load_causal_lm(model_path, device_map="cuda:0")
    num_layers = get_num_layers(model)
    hidden = get_hidden_size(model)
    print(f"[m1] Model loaded: {num_layers} layers, hidden={hidden}")

    train_pairs = load_paired_prompts(args.pairs_train)[: args.n_train_cap]
    held_pairs = load_paired_prompts(args.pairs_held)
    print(f"[m1] train pairs: {len(train_pairs)}, held pairs: {len(held_pairs)}")

    train_h = [p["harmful"] for p in train_pairs]
    train_b = [p["benign"] for p in train_pairs]
    held_h = [p["harmful"] for p in held_pairs]
    held_b = [p["benign"] for p in held_pairs]

    # Collect residual-stream last-token hidden states at every layer.
    all_layers = list(range(num_layers))
    print(f"[m1] Extracting hidden states across {num_layers} layers for train harmful...")
    hs_train_h = get_last_token_hidden_states(model, tokenizer, train_h, all_layers, batch_size=args.batch_size)
    print(f"[m1] Extracting for train benign...")
    hs_train_b = get_last_token_hidden_states(model, tokenizer, train_b, all_layers, batch_size=args.batch_size)
    print(f"[m1] Extracting for held harmful...")
    hs_held_h = get_last_token_hidden_states(model, tokenizer, held_h, all_layers, batch_size=args.batch_size)
    print(f"[m1] Extracting for held benign...")
    hs_held_b = get_last_token_hidden_states(model, tokenizer, held_b, all_layers, batch_size=args.batch_size)

    # Free model memory before fitting probes
    del model
    torch.cuda.empty_cache()

    # Per-layer: compute mean-diff direction AND probe direction, fit AUC.
    #
    # Rationale: the plan (§5.4) says use mean-diff `d_h = mean_h - mean_b`, but
    # empirically the mean-diff direction has only a small cos alignment with
    # individual activations (~0.03-0.07 in magnitude) — too small for a squared-
    # cosine reroute loss to have gradient. We save the probe's coefficient
    # vector alongside, which is the max-margin discriminative direction and
    # gives a larger signed cos on harmful (~+0.07) vs benign (~-0.05).
    # M3 uses the probe direction as `d_h` by default; the mean-diff is stored
    # for M4's specificity / audit trail.
    directions_meandiff = {}
    directions_probe = {}   # THIS becomes the primary `d_h` for M3/M4
    auc_per_layer = {}
    for L in all_layers:
        th = hs_train_h[L].numpy()
        tb = hs_train_b[L].numpy()
        hh = hs_held_h[L].numpy()
        hb = hs_held_b[L].numpy()
        # Mean-difference direction (harmful - benign)
        d_md = th.mean(0) - tb.mean(0)
        d_md = d_md / (np.linalg.norm(d_md) + 1e-9)
        directions_meandiff[L] = torch.tensor(d_md, dtype=torch.float32)
        # Probe direction (LR coef, unit-normalized) — points harmful side positive
        X_tr = np.concatenate([th, tb], axis=0)
        y_tr = np.concatenate([np.ones(len(th)), np.zeros(len(tb))])
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        w = clf.coef_.flatten()
        w = w / (np.linalg.norm(w) + 1e-9)
        directions_probe[L] = torch.tensor(w, dtype=torch.float32)
        # Probe AUC
        auc = probe_auc(th, tb, hh, hb)
        auc_per_layer[L] = auc
        # Also report base cos alignment on held-out
        def cos_agg(A, d):
            Au = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-9)
            return float((Au @ d).mean())
        md_cos_h = cos_agg(hh, d_md)
        md_cos_b = cos_agg(hb, d_md)
        pr_cos_h = cos_agg(hh, w)
        pr_cos_b = cos_agg(hb, w)
        print(f"[m1] layer {L:02d}: AUC={auc:.4f}  md[cos_h={md_cos_h:+.3f} cos_b={md_cos_b:+.3f}]  probe[cos_h={pr_cos_h:+.3f} cos_b={pr_cos_b:+.3f}]")

    # Choose sites
    sites = choose_sites(auc_per_layer, k=args.k_sites)
    print(f"[m1] Chosen sites S = {sites} (contiguous mid-band, top-{args.k_sites} by AUC)")

    # Success criterion: at least 3 mid-to-late layers with AUC > 0.8
    mid_late_lo = int(0.4 * num_layers)
    n_pass = sum(1 for L, a in auc_per_layer.items() if L >= mid_late_lo and a > 0.8)
    print(f"[m1] Layers with AUC>0.8 in mid-late half: {n_pass} (criterion: >=3)")

    # Save — `directions.pt` is the primary direction file (probe direction),
    # `directions_meandiff.pt` is stored for audit / M4 specificity comparison.
    save_json(auc_per_layer, args.outdir / "auc_per_layer.json")
    save_json(sites, args.outdir / "sites.json")
    torch.save(directions_probe, args.outdir / "directions.pt")
    torch.save(directions_meandiff, args.outdir / "directions_meandiff.pt")
    summary = {
        "num_layers": num_layers,
        "hidden_size": hidden,
        "n_train_pairs_used": len(train_pairs),
        "n_held_pairs_used": len(held_pairs),
        "sites": sites,
        "k_sites": args.k_sites,
        "mean_auc_all_layers": float(np.mean(list(auc_per_layer.values()))),
        "n_layers_auc_gt_0_8_in_mid_late": n_pass,
        "criterion_c1a_passed": bool(n_pass >= 3),
    }
    save_json(summary, args.outdir / "summary.json")
    print(f"[m1] Wrote {args.outdir}/summary.json")

    ended = time.time()
    if args.run_dir:
        write_cost(args.run_dir, started, ended, gpu_ids_from_env(),
                   extra={"milestone": "m1", "criterion_c1a_passed": summary["criterion_c1a_passed"]})
    print(f"[m1] Done in {(ended - started)/60:.1f} min.")


if __name__ == "__main__":
    main()
