"""
M10_C2_separation — Within-vs-between-concept cosine gap (P2c).

For the last layer (1000 fc components tied to ImageNet classes):
  Because each class has exactly one component in the last layer, there are no
  within-*same-class* pairs on fc. We build "same-concept" groups from
  clustered ImageNet class names by top-1 CLIP-text neighbor (e.g., dog breeds
  cluster together, cat breeds, birds, etc.). Alternatively we use hidden
  layers: on layer4, "same-concept" is defined by top-1 concept identity from M7.

  For simplicity and robustness we implement TWO measures:
    (a) fc within-vs-between using **CLIP-text-neighbor same-concept groups**
        (build clusters offline: group classes whose CLIP text embeddings are within
        top-N cos-nn of each other with threshold >= 0.85).
    (b) layer4 within-vs-between using **top-1 concept identity** on the broden vocab.

Report: gap = mean_within - mean_between, Cohen's d, p-value from t-test, and pass criterion.

Output JSON:
    {
      "k": ..., "pool": ...,
      "fc": { "gap": ..., "cohens_d": ..., "p_value": ..., "n_within": ..., "n_between": ..., "passes": bool },
      "layer4": { ... }
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import h5py
from scipy.stats import ttest_ind


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.sqrt(((x.var(ddof=1) * (len(x) - 1)) + (y.var(ddof=1) * (len(y) - 1))) / (len(x) + len(y) - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def build_fc_same_concept_groups(text_emb: np.ndarray, threshold: float = 0.85, min_size: int = 2) -> list[list[int]]:
    """Group ImageNet class indices whose text embeddings are pairwise-similar above `threshold`.
    Uses transitive-closure / connected components on the thresholded cos-similarity graph.
    """
    t = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    cos = t @ t.T
    # zero the diagonal
    np.fill_diagonal(cos, 0.0)
    n = cos.shape[0]
    # BFS connected components
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(n):
        for j in np.where(cos[i] > threshold)[0]:
            union(i, int(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= min_size]


def within_between_cosines(v_c: np.ndarray, groups: list[list[int]],
                            n_between_pairs: int = 10000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """v_c: (n_comp, D) L2-normalized. groups: list of index-lists (into v_c rows).
    Returns (within_pair_cosines, between_pair_cosines).
    """
    rng = np.random.default_rng(seed)
    v = v_c / (np.linalg.norm(v_c, axis=1, keepdims=True) + 1e-8)
    within = []
    for g in groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                within.append(float(v[g[a]] @ v[g[b]]))
    within = np.array(within, dtype=np.float32) if within else np.array([0.0], dtype=np.float32)
    # Between = random pairs where the two are in different groups (or one is ungrouped)
    idx_to_group: dict[int, int] = {}
    for gi, g in enumerate(groups):
        for i in g:
            idx_to_group[i] = gi
    n = v.shape[0]
    between = []
    tries = 0
    max_tries = n_between_pairs * 20
    while len(between) < n_between_pairs and tries < max_tries:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        tries += 1
        if a == b:
            continue
        if idx_to_group.get(a, -1) == idx_to_group.get(b, -2) and a in idx_to_group:
            continue
        between.append(float(v[a] @ v[b]))
    return within, np.array(between, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v_c", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_pairs", type=int, default=10000)
    p.add_argument("--fc_threshold", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with h5py.File(args.v_c, "r") as h5:
        v_c_all = h5["v_c"][:]
        layer = h5["layer"][:].astype("U16")
        class_lbl = h5["class_label"][:]
        k = int(h5.attrs["k"])
        pool = str(h5.attrs["pool"])

    with h5py.File(args.text, "r") as h5t:
        inet_txt = h5t["imagenet1k_classes/embeddings"][:]
        broden_txt = h5t["broden_concepts/embeddings"][:]
        broden_words = [w.decode() for w in h5t["broden_concepts/words"][:]]

    # ------ FC same-concept groups (via CLIP-text neighbor clusters) ------
    print(f"[M10] building fc same-concept groups by class-text cluster (thresh={args.fc_threshold})...")
    fc_groups_class = build_fc_same_concept_groups(inet_txt, threshold=args.fc_threshold, min_size=2)
    print(f"[M10] fc groups: {len(fc_groups_class)}, total classes in groups: {sum(len(g) for g in fc_groups_class)}")

    # Threshold-sensitivity check (log-only — informs downstream reviewers whether the choice matters)
    thresh_sens = {}
    for thr in (0.75, 0.80, 0.85, 0.90):
        gs = build_fc_same_concept_groups(inet_txt, threshold=thr, min_size=2)
        thresh_sens[thr] = {"n_groups": len(gs), "n_classes_in_groups": sum(len(g) for g in gs)}
    print(f"[M10] fc threshold sensitivity: {thresh_sens}")

    # Map class indices -> component-row indices on fc
    last_mask = (layer == "fc")
    v_c_last = v_c_all[last_mask]
    fc_class_to_row = {int(cl): i for i, cl in enumerate(class_lbl[last_mask])}
    fc_groups_rows = [[fc_class_to_row[c] for c in g if c in fc_class_to_row] for g in fc_groups_class]
    fc_groups_rows = [g for g in fc_groups_rows if len(g) >= 2]

    fc_within, fc_between = within_between_cosines(v_c_last, fc_groups_rows,
                                                    n_between_pairs=args.n_pairs, seed=args.seed)
    fc_gap = float(fc_within.mean() - fc_between.mean())
    fc_d = cohens_d(fc_within, fc_between)
    fc_p = float(ttest_ind(fc_within, fc_between, equal_var=False, alternative="greater").pvalue) if len(fc_within) > 1 else 1.0
    fc_pass = bool((fc_gap > 0) and (fc_d >= 0.5) and (fc_p < 0.05))

    # ------ layer4 same-concept groups (by top-1 concept identity on broden) ------
    print("[M10] building layer4 same-concept groups by top-1 concept identity (broden) ...")
    layer4_mask = (layer == "layer4")
    v_l4 = v_c_all[layer4_mask]
    v_l4_n = v_l4 / (np.linalg.norm(v_l4, axis=1, keepdims=True) + 1e-8)
    t_n = broden_txt / (np.linalg.norm(broden_txt, axis=1, keepdims=True) + 1e-8)
    cos = v_l4_n @ t_n.T
    top1_concept = cos.argmax(axis=1)  # (n_l4,)
    # Group l4 rows by top-1 concept
    concept_to_rows: dict[int, list[int]] = {}
    for i, c in enumerate(top1_concept):
        concept_to_rows.setdefault(int(c), []).append(i)
    l4_groups = [rows for rows in concept_to_rows.values() if len(rows) >= 2]
    print(f"[M10] layer4 groups: {len(l4_groups)} (min-size 2)")

    l4_within, l4_between = within_between_cosines(v_l4, l4_groups,
                                                    n_between_pairs=args.n_pairs, seed=args.seed)
    l4_gap = float(l4_within.mean() - l4_between.mean())
    l4_d = cohens_d(l4_within, l4_between)
    l4_p = float(ttest_ind(l4_within, l4_between, equal_var=False, alternative="greater").pvalue) if len(l4_within) > 1 else 1.0
    l4_pass = bool((l4_gap > 0) and (l4_d >= 0.5) and (l4_p < 0.05))

    out = {
        "k": k,
        "pool": pool,
        "n_pairs_between": args.n_pairs,
        "fc": {
            "n_within": int(len(fc_within)),
            "n_between": int(len(fc_between)),
            "mean_within_cos": float(fc_within.mean()),
            "mean_between_cos": float(fc_between.mean()),
            "gap": fc_gap,
            "cohens_d": fc_d,
            "p_value_ttest_greater": fc_p,
            "passes": fc_pass,
            "grouping": f"CLIP-text-neighbor clusters, threshold={args.fc_threshold}",
            "n_groups": len(fc_groups_rows),
            "threshold_sensitivity": thresh_sens,
        },
        "layer4": {
            "n_within": int(len(l4_within)),
            "n_between": int(len(l4_between)),
            "mean_within_cos": float(l4_within.mean()),
            "mean_between_cos": float(l4_between.mean()),
            "gap": l4_gap,
            "cohens_d": l4_d,
            "p_value_ttest_greater": l4_p,
            "passes": l4_pass,
            "grouping": "layer4 components grouped by top-1 broden concept identity",
            "n_groups": len(l4_groups),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[M10] wrote {args.out}: fc gap={fc_gap:.4f} d={fc_d:.3f}, layer4 gap={l4_gap:.4f} d={l4_d:.3f}")


if __name__ == "__main__":
    main()
