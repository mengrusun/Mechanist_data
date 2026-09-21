"""Downstream utility evaluation, using only the 50k-image ImageNet-val corpus
we have on disk.

Splits: eval10k → 8k train / 2k test (random, fixed seed).

Metrics per model:
  - kNN top-1 on 1000-way (chance 0.001)
  - kNN top-1 on 10-way imagenette subset
  - kNN top-1 on BREEDS-living17 super-classification
  - Linear probe on 10-way imagenette

kNN uses cosine similarity, k=20 (majority vote).
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
import paths


IMAGENETTE_WNIDS = ['n01440764', 'n02102040', 'n02979186', 'n03000684',
                    'n03028079', 'n03394916', 'n03417042', 'n03425413',
                    'n03445777', 'n03888257']


def load_imagenet_wnid_order():
    with open(os.path.join(paths.IMAGENET_VAL, "dataset_infos.json")) as f:
        d = json.load(f)
    wnids = d[list(d.keys())[0]]["features"]["label"]["names"]
    return wnids


def load_breeds_animal10():
    """Return dict: imagenet class idx (0-999) → living17 super-class idx (0-16),
    or -1 if the imagenet class is not in living17.
    """
    # BREEDS living17: 17 super-classes carved from animal subtree with
    # source/target subpopulations. We just group leaves by super-class here.
    # class_hierarchy.txt gives ancestor relations; dataset_class_info.json lists
    # the leaves per super-class.
    p_info = os.path.join(paths.BREEDS_ROOT,
                          "imagenet_class_hierarchy/modified/dataset_class_info.json")
    # Actually the file provides only the modified hierarchy; the true living17
    # split is derived by robustness lib. Instead, we do a rough proxy: use the
    # `dataset_class_info.json` list, which yields class-to-supergroup mapping
    # if it's the intended format. We fallback to living17 = the 17 direct
    # children of "physical_entity/organism/animal" in the modified hierarchy.
    node_names = os.path.join(paths.BREEDS_ROOT, "imagenet_class_hierarchy/modified/node_names.txt")
    hier = os.path.join(paths.BREEDS_ROOT, "imagenet_class_hierarchy/modified/class_hierarchy.txt")
    names = {}
    with open(node_names) as f:
        for line in f:
            if not line.strip(): continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) < 2: continue
            names[parts[0]] = parts[1]
    # class_hierarchy: parent<SPACE>child, one per line
    parent2children = {}
    child2parent = {}
    with open(hier) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2: continue
            p, c = parts[0], parts[1]
            parent2children.setdefault(p, []).append(c)
            child2parent[c] = p
    # Find node called "animal" (or "organism"): look for the string
    animal_node = None
    for n, name in names.items():
        first_syn = name.split(",")[0].strip().lower()
        if first_syn == "animal":
            animal_node = n; break
    if animal_node is None:
        return None
    # Direct children of animal are 17 super-classes for living17
    super_classes = parent2children.get(animal_node, [])
    # Return mapping: imagenet wnid → super_class_idx
    wnid2super = {}
    imagenet_wnids = set(load_imagenet_wnid_order())
    for sidx, sc in enumerate(super_classes):
        # BFS descendants
        stack = [sc]
        while stack:
            node = stack.pop()
            if node in imagenet_wnids:
                wnid2super[node] = sidx
            for ch in parent2children.get(node, []):
                stack.append(ch)
    return wnid2super


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z["feats"].astype(np.float32), z["labels"].astype(np.int64)


def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def do_knn(train_feats, train_labels, test_feats, test_labels, k=20):
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    clf.fit(train_feats, train_labels)
    return float(clf.score(test_feats, test_labels))


def do_linear(train_feats, train_labels, test_feats, test_labels):
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, C=1.0)
    clf.fit(train_feats, train_labels)
    return float(clf.score(test_feats, test_labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(paths.RESULTS_DIR, "downstream_eval.json"))
    args = ap.parse_args()

    wnids = load_imagenet_wnid_order()
    wnid2idx = {w: i for i, w in enumerate(wnids)}
    imagenette_labels = set(wnid2idx[w] for w in IMAGENETTE_WNIDS)

    wnid2super = load_breeds_animal10()
    if wnid2super is not None:
        idx2super = {wnid2idx[w]: s for w, s in wnid2super.items() if w in wnid2idx}
        print(f"BREEDS-living17 classes cover {len(idx2super)} imagenet classes across {max(idx2super.values())+1} super")
    else:
        idx2super = None

    results = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        feats, labels = load_npz(path)
        feats = normalize(feats)
        # split
        rng = np.random.RandomState(args.split_seed)
        perm = rng.permutation(len(feats))
        n_tr = int(0.8 * len(feats))
        tr = perm[:n_tr]; te = perm[n_tr:]

        entry = {}
        # 1) 1000-way kNN
        entry["knn20_top1_1000way"] = do_knn(feats[tr], labels[tr], feats[te], labels[te])

        # 2) Imagenette 10-way kNN
        tr_mask = np.isin(labels[tr], list(imagenette_labels))
        te_mask = np.isin(labels[te], list(imagenette_labels))
        if te_mask.sum() > 0:
            entry["knn20_top1_imagenette10"] = do_knn(
                feats[tr][tr_mask], labels[tr][tr_mask],
                feats[te][te_mask], labels[te][te_mask], k=20)
            entry["linear_top1_imagenette10"] = do_linear(
                feats[tr][tr_mask], labels[tr][tr_mask],
                feats[te][te_mask], labels[te][te_mask])
        # 3) BREEDS-animal10 super-class kNN
        if idx2super is not None:
            tr_lab_super = np.array([idx2super.get(int(l), -1) for l in labels[tr]])
            te_lab_super = np.array([idx2super.get(int(l), -1) for l in labels[te]])
            tr_mask = tr_lab_super >= 0
            te_mask = te_lab_super >= 0
            if te_mask.sum() > 0:
                entry["knn20_top1_breeds_animal10"] = do_knn(
                    feats[tr][tr_mask], tr_lab_super[tr_mask],
                    feats[te][te_mask], te_lab_super[te_mask], k=20)
                entry["linear_top1_breeds_animal10"] = do_linear(
                    feats[tr][tr_mask], tr_lab_super[tr_mask],
                    feats[te][te_mask], te_lab_super[te_mask])
            # OOD/subpopulation shift: for each super-class we split its
            # constituent imagenet classes into A/B halves; train linear on A,
            # test on unseen B. This reproduces the BREEDS-livingXX evaluation
            # spirit (source vs target subpopulations of the same super-class).
            super_to_leaves = {}
            for k_leaf, v in idx2super.items():
                super_to_leaves.setdefault(v, []).append(k_leaf)
            source_leaves = set()
            target_leaves = set()
            for s, leaves in super_to_leaves.items():
                lv = sorted(leaves)
                half = len(lv) // 2
                for lf in lv[:half]:
                    source_leaves.add(lf)
                for lf in lv[half:]:
                    target_leaves.add(lf)
            # Train on all eval10k images in source_leaves, test on all in target
            all_mask_src = np.isin(labels, list(source_leaves))
            all_mask_tgt = np.isin(labels, list(target_leaves))
            src_feats = feats[all_mask_src]
            tgt_feats = feats[all_mask_tgt]
            src_labs = np.array([idx2super[int(l)] for l in labels[all_mask_src]])
            tgt_labs = np.array([idx2super[int(l)] for l in labels[all_mask_tgt]])
            if len(tgt_feats) > 0 and len(src_feats) > 20:
                entry["knn20_ood_breeds_subpop"] = do_knn(
                    src_feats, src_labs, tgt_feats, tgt_labs, k=20)
                entry["linear_ood_breeds_subpop"] = do_linear(
                    src_feats, src_labs, tgt_feats, tgt_labs)
        print(f"[{name}] " + " ".join(f"{k}={v:.4f}" for k, v in entry.items()))
        results[name] = entry

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
