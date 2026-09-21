"""
Extra analyses for Claim 1:

(1) Plot lang_probe accuracy and parallel-retrieval top-1 vs layer.

(2) "Language-centered" analysis:
    We subtract, per language, the layer-mean over prompts before running the
    retrieval and probe. This removes an affine "language bias" and probes
    whether the *semantic* geometry aligns across languages.

(3) Language-pair heatmap of retrieval top-1 at the identified bottleneck layer.
"""

import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return a @ b.T

def retrieval_top1_matrix(reps_layer):
    """reps_layer: [N, K, H] -> [K, K] top-1 retrieval accuracy (diagonal set to 1)."""
    N, K, H = reps_layer.shape
    mat = np.eye(K)
    for i in range(K):
        for j in range(K):
            if i == j: continue
            a = reps_layer[:, i, :].astype(np.float32)
            b = reps_layer[:, j, :].astype(np.float32)
            sim = cosine(a, b)
            preds = sim.argmax(axis=1)
            mat[i, j] = float((preds == np.arange(N)).mean())
    return mat

def retrieval_top1(reps_layer):
    mat = retrieval_top1_matrix(reps_layer)
    K = mat.shape[0]
    off = mat[~np.eye(K, dtype=bool)]
    return float(off.mean())

def lang_probe(reps_layer, seed=0):
    N, K, H = reps_layer.shape
    X = reps_layer.reshape(N * K, H).astype(np.float32)
    y = np.tile(np.arange(K), N)
    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=200, n_jobs=-1, C=1.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=skf, n_jobs=-1).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", required=True)
    ap.add_argument("--json_in", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.reps, allow_pickle=True)
    reps = d["reps"]                # [L+1, N, K, H] float16
    langs = list(d["langs"])
    L1, N, K, H = reps.shape
    with open(args.json_in) as f:
        base = json.load(f)

    # (2) language-centered probe + retrieval
    centered = {"layers": [], "lang_probe": [], "retrieval_top1": []}
    for l in range(L1):
        rl = reps[l].astype(np.float32).copy()       # [N, K, H]
        # subtract per-language mean over prompts
        mu = rl.mean(axis=0, keepdims=True)          # [1, K, H]
        rl_c = rl - mu
        centered["layers"].append(l)
        centered["lang_probe"].append(lang_probe(rl_c))
        centered["retrieval_top1"].append(retrieval_top1(rl_c))
        print(f"[centered layer {l:>2}] lang_probe={centered['lang_probe'][-1]:.3f}  retrieval_top1={centered['retrieval_top1'][-1]:.3f}")

    # Save
    with open(os.path.join(args.out_dir, "claim1_centered.json"), "w") as f:
        json.dump({"centered": centered, "langs": langs}, f, indent=2)

    # (1) Plot: raw and centered
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    ax = axes[0]
    ax.plot(base["layers"], base["lang_probe"], "-o", label="lang probe acc")
    ax.plot(base["layers"], base["retrieval_top1"], "-s", label="cross-lang retrieval top-1")
    ax.set_xlabel("layer"); ax.set_ylabel("score")
    ax.set_title("Raw hidden states")
    ax.set_ylim(-0.02, 1.05); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(centered["layers"], centered["lang_probe"], "-o", label="lang probe acc")
    ax.plot(centered["layers"], centered["retrieval_top1"], "-s", label="cross-lang retrieval top-1")
    ax.set_xlabel("layer"); ax.set_ylabel("score")
    ax.set_title("After per-language mean subtraction")
    ax.set_ylim(-0.02, 1.05); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "claim1_layerwise.png"), dpi=150)
    print(f"[save] {args.out_dir}/claim1_layerwise.png")

    # (3) Heatmap of language-pair retrieval at bottleneck layer (raw)
    peak_layer = int(np.argmax(base["retrieval_top1"]))
    peak_layer_idx = base["layers"][peak_layer]
    mat = retrieval_top1_matrix(reps[peak_layer_idx].astype(np.float32))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(langs); ax.set_yticklabels(langs)
    ax.set_xlabel("target lang"); ax.set_ylabel("query lang")
    ax.set_title(f"Cross-lang retrieval top-1  (layer {peak_layer_idx})")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "claim1_pair_heatmap.png"), dpi=150)
    print(f"[save] peak_layer={peak_layer_idx}, mean_top1(off-diag)={mat[~np.eye(K, dtype=bool)].mean():.3f}")

if __name__ == "__main__":
    main()
