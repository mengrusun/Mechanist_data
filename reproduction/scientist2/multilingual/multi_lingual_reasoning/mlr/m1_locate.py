"""M1 — Locate V_lang via language-mean-difference SVD.

Grid:
    n_probe ∈ {50, 100, 250, 500, 1000}
    rank_r ∈ {1, 2, 4, 8, 16, 32}
    layer_group ∈ {early, mid, all_non_upper}   (early=[0,12), mid=[12,24), all_non_upper=[0,24) for a 36-layer model)
    seed ∈ {42, 43, 44}

For each config, this script:
    1. Loads probe-set sentences per language from FLORES-200 (or MGSM train fallback).
    2. Caches residual-stream activations at *representative* layers per layer_group (mid-layer of the range).
    3. Fits `V_lang` = top-r left singular vectors of the language-mean-difference matrix.
    4. Trains a linear language classifier on projected held-out MGSM prompts, reports macro-accuracy.
    5. Also trains an orthogonal-complement classifier (Π_⊥ · h) and reports macro-accuracy.

Output: results/m1/n{n_probe}_r{rank_r}_{layer_group}_s{seed}.npz containing:
    V_lang: (hidden, rank_r)     — the fitted subspace basis at the representative layer
    layer_used: int              — the representative layer index used to fit V_lang
    layer_group: str
    heldout_lang_acc: float      — macro-acc of V_lang classifier on held-out MGSM prompts
    heldout_lang_acc_per_lang: dict
    complement_acc: float        — macro-acc of the orthogonal-complement classifier
    complement_acc_per_lang: dict
    principal_angle_median_cos: float  — content-probe cosine (subject-identity probe on En MGSM)
    predicate_pass: bool         — held-out ≥ 0.90 AND complement ≤ 0.20 AND median cos ≤ 0.20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlr.data import MGSM_LANGS, load_flores_all, load_mgsm_all, default_few_shot_en, format_mgsm_prompt
from mlr.activations import ResidualStreamCache


def _fit_v_lang(
    activations_per_lang: Dict[str, np.ndarray],
    rank_r: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit V_lang = top-r left singular vectors of the language-mean-difference matrix.

    Args:
        activations_per_lang: {lang: (n_i, hidden)} — one activation array per language.
        rank_r: number of components.

    Returns:
        V: (hidden, rank_r) subspace basis
        mus: (n_langs, hidden) per-language mean activations
    """
    langs = sorted(activations_per_lang.keys())
    mus = np.stack([activations_per_lang[l].mean(axis=0) for l in langs], axis=0)  # (n_langs, hidden)
    mu_avg = mus.mean(axis=0, keepdims=True)  # (1, hidden)
    diff = mus - mu_avg  # (n_langs, hidden)
    # SVD of diff: diff = U S V^T   where V^T rows are the principal directions in hidden-space
    # We want the top-r *right* singular vectors as the subspace basis.
    U, S, Vt = np.linalg.svd(diff, full_matrices=False)
    V = Vt[:rank_r].T  # (hidden, rank_r)
    return V, mus


def _train_linear_classifier(
    X: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 1e-2, max_iter: int = 200, seed: int = 42
):
    """Train a closed-form linear classifier on (X, y).

    We use `RidgeClassifier` (closed-form via normal equations) for speed — this is 50–500× faster than
    sklearn's iterative `LogisticRegression` on high-dimensional projected activations, and the two
    yield near-identical accuracy on the near-linearly-separable multilingual-mean-projection setting.
    """
    from sklearn.linear_model import RidgeClassifier
    clf = RidgeClassifier(alpha=l2, random_state=seed)
    clf.fit(X, y)
    return clf


def _eval_classifier(clf, X: np.ndarray, y: np.ndarray, langs: List[str]) -> Tuple[float, Dict[str, float]]:
    preds = clf.predict(X)
    per_lang = {}
    for lang_idx, lang in enumerate(langs):
        mask = y == lang_idx
        if mask.sum() == 0:
            per_lang[lang] = float("nan")
        else:
            per_lang[lang] = float((preds[mask] == lang_idx).mean())
    macro = float(np.mean([v for v in per_lang.values() if not np.isnan(v)]))
    return macro, per_lang


def _layer_group_range(layer_group: str, num_layers: int) -> Tuple[int, int]:
    """Return the *base* [start, end) range for a layer_group tag.

    Follows the EXPERIMENT_PLAN.md convention for Qwen-3-4B-Thinking (num_layers = 36):
        early         = [0, 12)
        mid           = [12, 24)
        all_non_upper = [0, 28)   (leaves the top 8 layers intact by default; k_top further restricts)
    Uses fixed constants when num_layers == 36 (the plan's target), else scales by fractions.
    """
    if num_layers == 36:
        # Plan-native constants for Qwen-3-4B-Thinking (36 layers)
        if layer_group == "early":
            return (0, 12)
        if layer_group == "mid":
            return (12, 24)
        if layer_group == "all_non_upper":
            return (0, 28)
    else:
        # Scale by depth for verify-stage models of other depths
        if layer_group == "early":
            return (0, num_layers // 3)
        if layer_group == "mid":
            return (num_layers // 3, 2 * num_layers // 3)
        if layer_group == "all_non_upper":
            # Preserve "leave top ~8 (of 36 = 22%) intact by default" ratio
            return (0, int(round(num_layers * 28 / 36)))
    raise ValueError(f"Unknown layer_group: {layer_group}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--probe_data", default=None, help="Optional path to probe data JSONL (fallbacks to FLORES-200)")
    ap.add_argument("--n_probe", type=int, required=True)
    ap.add_argument("--rank_r", type=int, required=True)
    ap.add_argument("--layer_group", required=True, choices=["early", "mid", "all_non_upper"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_heldout_mgsm", type=int, default=100, help="Number of held-out MGSM train prompts per language for classifier eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    print(f"[m1] loading model from {args.model_dir}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers

    lg_start, lg_end = _layer_group_range(args.layer_group, num_layers)
    # Representative layer: midpoint of range
    layer_used = (lg_start + lg_end) // 2
    print(f"[m1] layer_group={args.layer_group} -> [{lg_start},{lg_end}); representative layer = {layer_used}", flush=True)

    # Cache probe activations at layer_used
    probe_texts = load_flores_all(langs=MGSM_LANGS, n_probe=args.n_probe, seed=args.seed, data_dir=args.data_dir)
    print(f"[m1] probe set sizes: " + ", ".join(f"{l}={len(v)}" for l, v in probe_texts.items()), flush=True)

    with ResidualStreamCache(model, tokenizer, [layer_used], pool="last") as cache:
        acts_per_lang = {}
        for lang, sents in probe_texts.items():
            arr = cache.encode_dataset(sents, batch_size=args.batch_size, max_length=args.max_length)[layer_used]
            acts_per_lang[lang] = arr
            print(f"[m1] cached {lang}: {arr.shape}", flush=True)

    # Fit V_lang
    V, mus = _fit_v_lang(acts_per_lang, args.rank_r)
    hidden = V.shape[0]
    print(f"[m1] V_lang shape: {V.shape}", flush=True)

    # Held-out MGSM eval: use MGSM train prompts (not test) at same layer
    mgsm_train = load_mgsm_all(split="train", data_dir=args.data_dir)
    held_texts: List[str] = []
    held_labels: List[int] = []
    langs_sorted = sorted(MGSM_LANGS)
    for li, lang in enumerate(langs_sorted):
        df = mgsm_train[lang]
        rng = np.random.default_rng(args.seed + li)
        idxs = rng.choice(len(df), size=min(args.n_heldout_mgsm, len(df)), replace=False)
        # Use the raw question as the probe target for held-out language classification
        for i in idxs:
            held_texts.append(str(df.iloc[int(i)]["question"]))
            held_labels.append(li)

    with ResidualStreamCache(model, tokenizer, [layer_used], pool="last") as cache:
        held_acts = cache.encode_dataset(held_texts, batch_size=args.batch_size, max_length=args.max_length)[layer_used]

    held_labels = np.array(held_labels, dtype=np.int64)

    # Stratified 80/20 split per language (avoid class imbalance)
    rng = np.random.default_rng(args.seed + 100)
    tr_idx_all: List[int] = []
    te_idx_all: List[int] = []
    for lid in range(len(langs_sorted)):
        idxs = np.where(held_labels == lid)[0]
        rng.shuffle(idxs)
        split = int(0.8 * len(idxs))
        tr_idx_all.extend(idxs[:split].tolist())
        te_idx_all.extend(idxs[split:].tolist())
    tr_idx = np.array(tr_idx_all, dtype=np.int64)
    te_idx = np.array(te_idx_all, dtype=np.int64)
    X_train, X_test = held_acts[tr_idx], held_acts[te_idx]
    y_train, y_test = held_labels[tr_idx], held_labels[te_idx]

    # Project onto V_lang and complement
    P = V @ V.T  # (hidden, hidden), rank-r projector
    proj = X_train @ P
    proj_te = X_test @ P
    comp = X_train - proj
    comp_te = X_test - proj_te

    clf_v = _train_linear_classifier(proj, y_train, n_classes=len(langs_sorted), seed=args.seed)
    macro_v, per_lang_v = _eval_classifier(clf_v, proj_te, y_test, langs_sorted)

    clf_c = _train_linear_classifier(comp, y_train, n_classes=len(langs_sorted), seed=args.seed)
    macro_c, per_lang_c = _eval_classifier(clf_c, comp_te, y_test, langs_sorted)

    # Content-probe: subject-identity subspace on English MGSM (approximate).
    # Bootstrap 10 random split-half pairs of English MGSM prompts, take their mean-difference vectors,
    # stack into a matrix and SVD to get the top-r_content principal directions of "content" variance
    # within a single language (i.e., the axes along which prompts differ when language is held constant).
    en_indices = np.where(held_labels == langs_sorted.index("en"))[0]
    if len(en_indices) >= 8:
        rng_c = np.random.default_rng(args.seed + 500)
        content_diffs = []
        for _ in range(10):
            idx = rng_c.permutation(en_indices)
            half = len(idx) // 2
            a = held_acts[idx[:half]].mean(axis=0)
            b = held_acts[idx[half : 2 * half]].mean(axis=0)
            content_diffs.append(a - b)
        C = np.stack(content_diffs, axis=0)  # (10, hidden)
        # Take top-rank_r content directions as V_content
        Uc, Sc, Vtc = np.linalg.svd(C, full_matrices=False)
        r_content = min(args.rank_r, Vtc.shape[0])
        V_content = Vtc[:r_content].T  # (hidden, r_content)
        # Principal angles: SVD of V_lang.T @ V_content
        M = V.T @ V_content  # (rank_r, r_content)
        _, sig, _ = np.linalg.svd(M, full_matrices=False)
        sig = np.clip(sig, -1.0, 1.0)
        # cosines of principal angles = singular values of V_lang.T @ V_content (assumes orthonormal columns)
        median_cos = float(np.median(sig))
    else:
        median_cos = float("nan")

    predicate_pass = (macro_v >= 0.90) and (macro_c <= 0.20) and (median_cos <= 0.20)

    elapsed = time.time() - t0
    print(f"[m1] macro_v={macro_v:.4f}  macro_c={macro_c:.4f}  median_cos={median_cos:.4f}  pass={predicate_pass}  ({elapsed:.1f}s)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        V_lang=V.astype(np.float32),
        layer_used=layer_used,
        layer_group=args.layer_group,
        heldout_lang_acc=macro_v,
        heldout_lang_acc_per_lang=json.dumps(per_lang_v),
        complement_acc=macro_c,
        complement_acc_per_lang=json.dumps(per_lang_c),
        principal_angle_median_cos=median_cos,
        predicate_pass=predicate_pass,
        n_probe=args.n_probe,
        rank_r=args.rank_r,
        seed=args.seed,
        elapsed_seconds=elapsed,
    )
    print(f"[m1] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
