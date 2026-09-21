"""M1 — Location — activation capture + per-behaviour, per-layer probe + L*(b).

Consumes:
  data/contrast/auxiliary_corpus.jsonl -- {id, source, text, behaviours}
Produces:
  runs/M1_locate/activations.npz        -- layer -> (N, hidden) numpy
  runs/M1_locate/directions/v_<b>_L<L*>.pt
  runs/M1_locate/results.json           -- per_behaviour L*, ROC-AUC, first-PC align
  runs/M1_locate/cost.json              -- gpu_ids, wall clock, model params
"""
from __future__ import annotations
import os, sys, json, time, argparse, random
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, get_num_layers, capture_residual_last_token, get_input_device

BEHAVIOURS = ["expressing_uncertainty", "generating_validation_examples", "backtracking", "self-correction"]


def load_corpus(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def cache_activations(model, tok, texts: list[str], layers: list[int],
                      batch: int = 4, max_len: int = 1024) -> dict[int, np.ndarray]:
    """For each text, forward-pass and capture last-token residual per layer."""
    device = get_input_device(model)
    N = len(texts)
    hidden = model.config.hidden_size
    out = {li: np.zeros((N, hidden), dtype=np.float32) for li in layers}
    t0 = time.time()
    for i in range(0, N, batch):
        batch_texts = texts[i:i+batch]
        # tokenize the text as-is (not chat-template) since these are chain excerpts
        enc = tok(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        with capture_residual_last_token(model, layers) as captured, torch.no_grad():
            _ = model(**enc, use_cache=False)
        for li in layers:
            out[li][i:i+len(batch_texts)] = captured[li].numpy()
        if (i // batch) % 5 == 0:
            print(f"  cached {i+len(batch_texts)}/{N} in {time.time()-t0:.1f}s")
    return out


def train_probe_per_layer(X_by_layer: dict[int, np.ndarray], y: np.ndarray,
                          train_idx: np.ndarray, test_idx: np.ndarray,
                          seed: int = 0) -> dict[int, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    results = {}
    for li, X in X_by_layer.items():
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            results[li] = {"roc_auc_test": None, "acc_test": None, "note": "single-class-split"}
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed, class_weight="balanced")
        clf.fit(Xtr, ytr)
        pte = clf.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, pte))
        acc = float(((pte >= 0.5).astype(int) == yte).mean())
        results[li] = {"roc_auc_test": auc, "acc_test": acc}
    return results


def mean_difference_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pos = X[y == 1].mean(axis=0)
    neg = X[y == 0].mean(axis=0)
    return (pos - neg).astype(np.float32)


def first_pc_alignment(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> float:
    """|cos(v, first_PC(X_pos - X_neg))|. We use paired diffs when possible,
    else diff-of-population."""
    pos = X[y == 1]
    neg = X[y == 0]
    # sample-matched pairs (min length)
    n = min(len(pos), len(neg))
    if n < 3:
        return float("nan")
    # random shuffle for pairing
    rng = np.random.default_rng(0)
    idx_pos = rng.permutation(len(pos))[:n]
    idx_neg = rng.permutation(len(neg))[:n]
    diffs = pos[idx_pos] - neg[idx_neg]
    diffs = diffs - diffs.mean(0, keepdims=True)
    # first right-singular vector
    U, S, Vt = np.linalg.svd(diffs, full_matrices=False)
    pc1 = Vt[0]
    v_unit = v / (np.linalg.norm(v) + 1e-9)
    pc_unit = pc1 / (np.linalg.norm(pc1) + 1e-9)
    return float(abs(np.dot(v_unit, pc_unit)))


def compute_sigma_proj(X: np.ndarray, v: np.ndarray) -> float:
    """σ_proj = std(X @ unit(v)) — used to express steering α in σ units."""
    u = v / (np.linalg.norm(v) + 1e-9)
    proj = X @ u
    return float(np.std(proj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--corpus", default="data/contrast/auxiliary_corpus.jsonl")
    ap.add_argument("--out_dir", default="runs/M1_locate")
    ap.add_argument("--layer_stride", type=int, default=1, help="1 => sweep every layer")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--sanity", type=int, default=0,
                    help="if >0: use only this many corpus entries and only 5 layers")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "directions"), exist_ok=True)

    t0 = time.time()
    corpus = load_corpus(args.corpus)
    if args.sanity > 0:
        corpus = corpus[:args.sanity]
    texts = [r["text"] for r in corpus]

    print(f"[M1] loading model {args.model_path}")
    model, tok = load_model(args.model_path)
    num_layers = get_num_layers(model)
    if args.sanity > 0:
        # 5 evenly-spaced layers
        layers = sorted(set([int(x) for x in np.linspace(0, num_layers - 1, 5)]))
    else:
        layers = list(range(0, num_layers, args.layer_stride))
    print(f"[M1] {len(corpus)} chains × {len(layers)} layers")

    # Activation cache
    t1 = time.time()
    print("[M1] caching activations ...")
    X_by_layer = cache_activations(model, tok, texts, layers, batch=args.batch, max_len=args.max_len)
    t_cache = time.time() - t1
    print(f"[M1] cache done in {t_cache:.1f}s")
    # free GPU model — probes are CPU
    del model
    import gc; gc.collect(); torch.cuda.empty_cache()

    # Save activations for M2 re-use
    np.savez_compressed(os.path.join(args.out_dir, "activations.npz"),
                        **{f"L{li}": X_by_layer[li] for li in layers},
                        layer_ids=np.array(layers))

    # Save metadata
    meta = {
        "corpus_ids": [r["id"] for r in corpus],
        "layers": layers,
        "hidden_size": int(next(iter(X_by_layer.values())).shape[1]),
        "num_chains": len(corpus),
        "train_frac": args.train_frac,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Per-behaviour probe + direction
    per_behaviour = {}
    rng = np.random.default_rng(args.seed)
    idx_all = np.arange(len(corpus))
    rng.shuffle(idx_all)
    ntrain = int(len(corpus) * args.train_frac)
    train_idx = idx_all[:ntrain]
    test_idx = idx_all[ntrain:]

    for b in BEHAVIOURS:
        y = np.array([r["behaviours"].get(b, 0) for r in corpus], dtype=np.int64)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        if n_pos < 3 or n_neg < 3:
            per_behaviour[b] = {"L_star": None, "note": f"insufficient pos/neg ({n_pos}/{n_neg})"}
            continue
        # Probe per layer
        probe_res = train_probe_per_layer(X_by_layer, y, train_idx, test_idx, seed=args.seed)
        best = max(
            ((li, r) for li, r in probe_res.items() if r.get("roc_auc_test") is not None),
            key=lambda x: x[1]["roc_auc_test"],
            default=(None, None),
        )
        # Exclude layer 0 as trivial (embedding proxy) if it's the top
        if best[0] == 0 and len(probe_res) > 1:
            second = max(
                ((li, r) for li, r in probe_res.items() if r.get("roc_auc_test") is not None and li != 0),
                key=lambda x: x[1]["roc_auc_test"],
            )
            layer0_note = f"layer 0 scored top (auc={best[1]['roc_auc_test']:.3f}) — using L={second[0]} to avoid embedding artifact"
            best = second
        else:
            layer0_note = None
        L_star = int(best[0]) if best[0] is not None else None
        auc = float(best[1]["roc_auc_test"]) if best[1] else None
        if L_star is None:
            per_behaviour[b] = {"L_star": None, "note": f"no valid probe layer for {b}"}
            continue

        # Direction + PC alignment at L*
        X = X_by_layer[L_star]
        v_raw = mean_difference_direction(X[train_idx], y[train_idx])
        pc_align = first_pc_alignment(X[train_idx], y[train_idx], v_raw)
        # UNIT direction — this is the canonical CAA-with-σ scaling: at inference
        # we add α · σ_proj · u   where u = v/||v||. Then ||add|| = α · σ_proj,
        # which sits at a well-controlled fraction of the residual norm.
        v_norm = float(np.linalg.norm(v_raw) + 1e-9)
        u = (v_raw / v_norm).astype(np.float32)
        sigma_proj = compute_sigma_proj(X, v_raw)  # std of X @ unit(v_raw)
        # Save direction as unit vector; sigma_proj is stored so runtime uses α·σ·u
        d_path = os.path.join(args.out_dir, "directions", f"v_{b}_L{L_star}.pt")
        torch.save({
            "behaviour": b,
            "layer": L_star,
            "direction": torch.from_numpy(u),          # UNIT vector
            "direction_raw_norm": v_norm,               # for reference
            "sigma_proj": sigma_proj,
            "n_pos_train": int(y[train_idx].sum()),
            "n_neg_train": int((y[train_idx] == 0).sum()),
        }, d_path)
        per_behaviour[b] = {
            "L_star": L_star,
            "roc_auc_test": auc,
            "first_pc_align": pc_align,
            "sigma_proj_at_L_star": sigma_proj,
            "n_pos_total": n_pos,
            "n_neg_total": n_neg,
            "layer_auc_curve": {int(li): r.get("roc_auc_test") for li, r in probe_res.items()},
            "v_b_path": d_path,
            "layer0_note": layer0_note,
        }

    # Annotation-consistency kappa (if file exists)
    kappa = None
    if os.path.exists("data/contrast/annotation_kappa.json"):
        kappa = json.load(open("data/contrast/annotation_kappa.json"))

    results = {
        "per_behaviour": per_behaviour,
        "annotation_kappa": kappa,
        "meta": meta,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # cost.json (queue/monitor expects this)
    cost = {
        "run_id": "M1_locate",
        "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
        "wall_seconds_total": time.time() - t0,
        "wall_seconds_activation_cache": t_cache,
        "num_chains": len(corpus),
        "num_layers": len(layers),
        "model_path": args.model_path,
    }
    with open(os.path.join(args.out_dir, "cost.json"), "w") as f:
        json.dump(cost, f, indent=2)

    print(f"[M1] done in {time.time()-t0:.1f}s; results at {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
