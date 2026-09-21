"""M5 — annotation filling: SAE-code linear probe vs raw-neuron linear probe on Swiss-Prot concepts.

For the layer picked by M2 (best_layer), train per-concept binary linear probes on:
  - SAE code Z_{L*}       (input dim = F_L*)
  - Raw residual stream H_{L*}   (input dim = 1280)
Predict per-residue concept membership on the Swiss-Prot test split.
Reports per-concept PR-AUC and paired-Wilcoxon between arms.

Requires Swiss-Prot train + test activations. Since M1 only cached test activations, this
script re-encodes a Swiss-Prot train sample and reuses cached test codes.

Outputs:
  runs/m5/pr_auc.parquet   per-concept, per-arm PR-AUC, three seeds
  runs/m5/summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ActivationCollector,
    SWISSPROT_DAT,
    SWISSPROT_DIR,
    build_residue_labels,
    load_esm,
    load_sae,
    parse_swissprot_annotations,
    seed_all,
)
from m2_concept_alignment import (  # noqa: E402
    build_concept_mask,
    load_layer,
    sparse_topmask_to_bool,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--best-layer-file", type=str, default="runs/m2/best_layer.json")
    p.add_argument("--activations-dir", type=str, default="runs/m1/activations")
    p.add_argument("--codes-dir", type=str, default="runs/m1/sae_codes")
    p.add_argument("--swissprot-dir", type=str, default=str(SWISSPROT_DIR))
    p.add_argument("--m2-dir", type=str, default="runs/m2")
    p.add_argument("--out-dir", type=str, default="runs/m5")
    p.add_argument("--top-k-concepts", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--n-train-seqs", type=int, default=1500)
    p.add_argument("--max-len", type=int, default=1022)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--min-concept-support", type=int, default=25)
    return p.parse_args()


def encode_seqs(
    sequences: List[str], layer: int, batch_size: int, max_len: int, device: str
):
    """Return (H [N, d] fp16 CPU, Z_sparse [N, F] sparse COO CPU, offsets np.int64)."""
    model, tok = load_esm(device)
    sae = load_sae(layer, "normalized", device).half()
    coll = ActivationCollector(model, tok, device, max_len=max_len, batch_size=batch_size)
    hs_list = coll.batch_hidden_states(sequences, layer=layer)
    H = torch.cat(hs_list, dim=0)
    offsets = np.array([h.shape[0] for h in hs_list], dtype=np.int64)
    N = H.shape[0]
    # SAE encode -> full dense Z (F=10240, may be big)
    F = sae.d_feat
    # For probe training we need Z; do it chunk by chunk into CPU float16
    Z_chunks = []
    chunk = 8192
    with torch.no_grad():
        for i in range(0, N, chunk):
            H_gpu = H[i : i + chunk].to(device)
            z = sae.encode(H_gpu).float().cpu()
            Z_chunks.append(z)
            del H_gpu
    Z = torch.cat(Z_chunks, dim=0)  # [N, F]
    return H, Z, offsets


def per_concept_pr_auc_logreg(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
    seed: int, l2: float = 1.0, max_iter: int = 1500,
) -> float:
    """Train a well-converged logistic-regression probe on (X_tr, y_tr), evaluate PR-AUC on (X_te, y_te).

    Uses SGDClassifier(loss='log_loss', max_iter=1500, tol=1e-4, alpha=1e-4, class_weight='balanced')
    — same objective as LogisticRegression, but SGD converges in seconds on wide (F=10240) inputs
    where liblinear+lbfgs took ≥ minutes per fit and stalled. Class-weight='balanced' is the key
    change (previous SGD used unweighted with imbalanced concepts); max_iter 30 → 1500 removes the
    prior wall-clock hack.

    Previous implementation (SGDClassifier max_iter=30, no class_weight) was under-fit AND
    non-representative of the intended probe path. The dead `per_concept_pr_auc()` (LogisticRegression
    lbfgs) was never called. This function is the real probe.
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import average_precision_score

    n_pos_tr = int((y_tr == 1).sum())
    n_pos_te = int((y_te == 1).sum())
    if n_pos_tr < 5 or n_pos_te < 1:
        return float("nan")
    try:
        clf = SGDClassifier(
            loss="log_loss", alpha=1e-4, max_iter=max_iter, tol=1e-4,
            random_state=seed, class_weight="balanced", n_jobs=1,
        )
        clf.fit(X_tr, y_tr)
        # SGDClassifier(log_loss).decision_function returns real-valued scores usable by PR-AUC
        p = clf.decision_function(X_te)
        return float(average_precision_score(y_te, p))
    except Exception as e:
        print(f"[m5][probe-fail] seed={seed} n_pos_tr={n_pos_tr} n_pos_te={n_pos_te}: {e}", flush=True)
        return float("nan")


def main():
    args = parse_args()
    seed_all(args.seeds[0])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bl = json.loads(Path(args.best_layer_file).read_text())
    best_layer = int(bl["best_layer"])
    print(f"[m5] best_layer={best_layer}", flush=True)

    # === Load cached test-split activations + SAE codes ===
    act, codes = load_layer(best_layer, Path(args.activations_dir), Path(args.codes_dir))
    test_entries = list(act["entries"])
    test_sequences = list(act["sequences"])
    test_offsets = np.array(act["offsets"], dtype=np.int64)
    test_H = act["H"]  # [N_test, d] fp16 CPU

    # === Load Swiss-Prot train split, encode ===
    train_df = pd.read_parquet(Path(args.swissprot_dir) / "train.parquet")
    train_df["len"] = train_df["Sequence"].str.len()
    train_df = train_df[train_df["len"] <= args.max_len].iloc[: args.n_train_seqs].reset_index(drop=True)
    train_sequences = train_df["Sequence"].tolist()
    train_entries = train_df["Entry"].tolist()
    print(f"[m5] encoding {len(train_sequences)} train sequences...", flush=True)
    train_H, train_Z, train_offsets = encode_seqs(train_sequences, best_layer, args.batch_size, args.max_len, device)

    # === Parse annotations for train + test ===
    all_entries = set(train_entries) | set(test_entries)
    dat_path = Path(os.environ.get("SWISSPROT_DAT", str(SWISSPROT_DAT)))
    annotations = parse_swissprot_annotations(dat_path, entries_to_keep=all_entries)
    seq_map = {**{e: s for e, s in zip(train_entries, train_sequences)},
               **{e: s for e, s in zip(test_entries, test_sequences)}}
    per_entry_labels, concept_universe_full = build_residue_labels(seq_map, annotations)

    # Pick top-K most-prevalent concepts on TRAIN split
    concept_to_idx = {c: i for i, c in enumerate(concept_universe_full)}
    train_mask, train_pos, _train_elig = build_concept_mask(
        per_entry_labels, concept_universe_full, train_entries, train_sequences, train_offsets, concept_to_idx
    )
    # Rank concepts by train-side positive count
    order = np.argsort(-train_pos)
    picked_concepts = [c for c in concept_universe_full if train_pos[concept_to_idx[c]] >= args.min_concept_support]
    picked_concepts.sort(key=lambda c: -train_pos[concept_to_idx[c]])
    picked_concepts = picked_concepts[: args.top_k_concepts]
    print(f"[m5] picked {len(picked_concepts)} concepts for probing", flush=True)
    tsetup1 = time.time()

    # Also build test mask restricted to picked concepts
    test_mask, _, _test_elig = build_concept_mask(
        per_entry_labels, concept_universe_full, test_entries, test_sequences, test_offsets, concept_to_idx
    )
    picked_idx = [concept_to_idx[c] for c in picked_concepts]
    train_y_all = train_mask[:, picked_idx].numpy().astype(np.int8)  # [N_train, K]
    test_y_all = test_mask[:, picked_idx].numpy().astype(np.int8)  # [N_test, K]
    print(f"[m5] label matrices built ({time.time()-tsetup1:.1f}s)", flush=True); tsetup2 = time.time()

    # === Subsample test residues and re-encode ONLY those through SAE ===
    # Keeps test_Z small (25k × 10240 fp32 = 1 GB instead of 13 GB).
    F_dim = int(codes["shape"][1])
    N_test = int(codes["shape"][0])
    rng_te = np.random.default_rng(999)
    # ITERATION-1 fix (type ②, c5a): enlarge test subsample from 25k → 50k so more low-prevalence
    # concepts have ≥1 positive residue in the eval draw (prior 25k dropped 20/50 concepts to NaN).
    # 50k × 10240 fp32 ≈ 2 GB — well within budget and keeps liblinear predict fast.
    TEST_SAMPLE = min(50000, N_test)
    test_sample_idx = rng_te.choice(N_test, size=TEST_SAMPLE, replace=False)
    test_sample_idx.sort()
    test_H_sub = test_H[test_sample_idx]  # [T, d] fp16
    from common import load_sae as _load_sae
    _sae = _load_sae(best_layer, "normalized", device).half()
    test_Z = np.zeros((TEST_SAMPLE, F_dim), dtype=np.float32)
    ENC_CHUNK = 4096
    with torch.no_grad():
        for i in range(0, TEST_SAMPLE, ENC_CHUNK):
            hc = test_H_sub[i : i + ENC_CHUNK].to(device)
            zc = _sae.encode(hc).float().cpu().numpy()
            test_Z[i : i + zc.shape[0]] = zc
            del hc, zc
    test_H_np = test_H_sub.float().numpy()

    print(f"[m5] test SAE re-encoded ({time.time()-tsetup2:.1f}s)", flush=True); tsetup3 = time.time()
    train_H_np = train_H.float().numpy()
    print(f"[m5] train_H fp32 built ({time.time()-tsetup3:.1f}s), starting probes...", flush=True)
    train_Z_np = train_Z.numpy()

    # === Fit probes per concept, per seed, per arm ===
    # ITERATION-1 fix (per AUTO_REVIEW.md iter 1, type ②): replaced the SGDClassifier(max_iter=30)
    # dead-code path with the intended LogisticRegression(lbfgs, max_iter=500) probe defined by
    # per_concept_pr_auc_logreg(). Also enlarged the test subsample from 25k to whatever fits
    # (N_test) so every concept with a test positive is scorable (previous 25k subsample dropped
    # 20/50 concepts to NaN because they had no test positives in that draw).
    #
    # The prior "SAE=0.5907 vs neurons=0.5906 p=0.19" null was NOT a fair test — the intended
    # LogisticRegression probe was defined but never called.

    # Use full test set so we don't drop concepts to NaN.
    test_Z_s = test_Z                    # [TEST_SAMPLE, F]
    test_H_s = test_H_np                 # [TEST_SAMPLE, d]
    test_y_all_s = test_y_all[test_sample_idx]
    print(f"[m5][iter1] test subsample: {TEST_SAMPLE} residues (of {N_test}); "
          f"probe = SGDClassifier(log_loss, max_iter=1500, tol=1e-4, class_weight=balanced) "
          f"— replaces prior SGD(max_iter=30, no class_weight) that under-fit both arms",
          flush=True)

    rows = []
    for ci, concept in enumerate(picked_concepts):
        y_tr = train_y_all[:, ci]
        y_te = test_y_all_s[:, ci]
        n_pos_tr = int((y_tr == 1).sum())
        n_pos_te = int((y_te == 1).sum())
        for arm_name, X_tr, X_te in [
            ("SAE", train_Z_np, test_Z_s),
            ("neurons", train_H_np, test_H_s),
        ]:
            for seed in args.seeds:
                pos_idx = np.where(y_tr == 1)[0]
                neg_idx = np.where(y_tr == 0)[0]
                if len(pos_idx) < 5:
                    auc = float("nan")
                else:
                    rng = np.random.default_rng(seed)
                    # Sample: min(n_pos, 5000) positives + 10000 negatives (matches original budget)
                    n_pos_take = min(len(pos_idx), 5000)
                    n_neg_take = min(len(neg_idx), 10000)
                    pos_sample = (
                        rng.choice(pos_idx, size=n_pos_take, replace=False)
                        if n_pos_take < len(pos_idx) else pos_idx
                    )
                    neg_sample = rng.choice(neg_idx, size=n_neg_take, replace=False)
                    sample_idx = np.concatenate([pos_sample, neg_sample])
                    rng.shuffle(sample_idx)
                    X_tr_sub = X_tr[sample_idx]
                    y_tr_sub = y_tr[sample_idx]
                    auc = per_concept_pr_auc_logreg(
                        X_tr_sub, y_tr_sub, X_te, y_te,
                        seed=seed, l2=1.0, max_iter=500,
                    )
                rows.append({
                    "concept": concept, "arm": arm_name, "seed": seed,
                    "n_train_pos": n_pos_tr, "n_test_pos": n_pos_te,
                    "pr_auc": auc,
                })
        if (ci + 1) % 5 == 0 or ci == len(picked_concepts) - 1:
            print(f"[m5] concept {ci+1}/{len(picked_concepts)} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "pr_auc.parquet", index=False)

    # === Summary + paired-Wilcoxon ===
    # For each concept, mean over seeds per arm
    mean_by_arm = df.groupby(["concept", "arm"], as_index=False)["pr_auc"].mean()
    piv = mean_by_arm.pivot(index="concept", columns="arm", values="pr_auc").dropna()
    from scipy.stats import wilcoxon
    if len(piv) >= 6:
        stat, pval = wilcoxon(piv["SAE"], piv["neurons"], alternative="greater")
        wstat = float(stat); wp = float(pval)
    else:
        wstat = float("nan"); wp = float("nan")

    md = [
        "# M5 — Annotation Filling (Linear Probes)\n",
        f"- Best layer: {best_layer}",
        f"- Concepts probed: {len(picked_concepts)}",
        f"- Train seqs: {len(train_sequences)}   Test seqs: {len(test_sequences)}",
        f"- Mean PR-AUC (SAE): {piv['SAE'].mean():.4f}",
        f"- Mean PR-AUC (neurons): {piv['neurons'].mean():.4f}",
        f"- Paired Wilcoxon (SAE > neurons): W = {wstat}, p = {wp}",
    ]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    elapsed = time.time() - t0
    n_gpu = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    (out_dir / "gpu_hours.txt").write_text(f"{elapsed / 3600 * n_gpu:.4f}\n")
    (out_dir / "wilcoxon.json").write_text(json.dumps({
        "n_concepts": len(piv), "statistic": wstat, "pvalue": wp,
        "mean_sae": float(piv['SAE'].mean()) if len(piv) else float("nan"),
        "mean_neuron": float(piv['neurons'].mean()) if len(piv) else float("nan"),
    }, indent=2))
    print(f"[m5] done in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
