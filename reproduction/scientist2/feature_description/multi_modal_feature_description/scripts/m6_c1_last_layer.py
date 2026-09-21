"""
M6_C1_last_layer — Last-layer concept-purity (P1a).

For each last-layer unit c (component tied to ImageNet class c):
  top-1 concept from ImageNet-1k class-name vocab = argmax_w cos(v_c, e_w)
  purity_c = 1 if top-1 == class(c) else 0
Purity = mean(purity_c). Compare against random-input baseline.

Predicate P1a passes when:
  delta_pure = top1_purity - random_baseline_top1_purity > 0 at p < 0.05 (paired McNemar/permutation).
Reference range (CLIP-Dissect, Oikarinen & Weng 2023): top-1 purity ~ 0.55-0.76 on ResNet-50 fc.

Output JSON:
    {
      "k": ..., "pool": ..., "n_last_layer_components": ...,
      "top1_purity": ..., "top5_purity": ...,
      "random_baseline_top1_purity": ..., "random_baseline_top5_purity": ...,
      "delta_pure": ..., "p_value_mcnemar": ..., "passes": bool
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import h5py
from scipy.stats import binomtest


def cos_ranking(v_c_last: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """v_c_last: (n_last, D), text_emb: (n_classes, D). Return (n_last, n_classes) cosines."""
    v_c = v_c_last / (np.linalg.norm(v_c_last, axis=1, keepdims=True) + 1e-8)
    t = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    return v_c @ t.T


def top1_purity(cos: np.ndarray, class_labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (mean top-1 accuracy, per-comp correct-mask)."""
    top1 = cos.argmax(axis=1)
    correct = (top1 == class_labels).astype(np.int32)
    return float(correct.mean()), correct


def top5_purity(cos: np.ndarray, class_labels: np.ndarray) -> float:
    top5 = np.argpartition(-cos, 5, axis=1)[:, :5]
    correct = np.array([class_labels[i] in top5[i] for i in range(len(class_labels))], dtype=np.int32)
    return float(correct.mean())


def mcnemar_p_value(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Two-sided McNemar test for paired binary outcomes.
    Uses the exact binomial version to avoid chi-square approximation issues at low counts.
    """
    # b = #{A correct, B wrong}; c = #{A wrong, B correct}
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    n = b + c
    if n == 0:
        return 1.0
    p = binomtest(min(b, c), n, p=0.5).pvalue
    return float(p)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v_c", required=True)
    p.add_argument("--v_c_random", required=True, help="Random-input baseline v_c (from m4 --random_baseline).")
    p.add_argument("--text", required=True, help="text_embeddings.h5 from M5.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[M6] loading v_c={args.v_c}")
    with h5py.File(args.v_c, "r") as h5:
        v_c_all = h5["v_c"][:]
        layer = h5["layer"][:].astype("U16")
        class_lbl = h5["class_label"][:]
        k = int(h5.attrs["k"])
        pool = str(h5.attrs["pool"])

    print(f"[M6] loading random baseline v_c={args.v_c_random}")
    with h5py.File(args.v_c_random, "r") as h5r:
        v_c_rand = h5r["v_c"][:]
        layer_r = h5r["layer"][:].astype("U16")
        class_lbl_r = h5r["class_label"][:]

    with h5py.File(args.text, "r") as h5t:
        text_emb = h5t["imagenet1k_classes/embeddings"][:]  # (1000, 512)

    last_mask = (layer == "fc")
    v_c_last = v_c_all[last_mask]
    lbl_last = class_lbl[last_mask]
    v_c_last_rand = v_c_rand[last_mask]
    lbl_last_r = class_lbl_r[last_mask]
    assert np.array_equal(lbl_last, lbl_last_r), "component ordering mismatch between v_c and v_c_random"
    print(f"[M6] last-layer components: {last_mask.sum()}")

    cos_real = cos_ranking(v_c_last, text_emb)
    cos_rand = cos_ranking(v_c_last_rand, text_emb)

    top1_real, correct_real = top1_purity(cos_real, lbl_last)
    top1_rand, correct_rand = top1_purity(cos_rand, lbl_last)
    top5_real = top5_purity(cos_real, lbl_last)
    top5_rand = top5_purity(cos_rand, lbl_last)

    delta = top1_real - top1_rand
    p_value = mcnemar_p_value(correct_real, correct_rand)
    passes = bool((delta > 0) and (p_value < 0.05))

    out = {
        "k": k,
        "pool": pool,
        "n_last_layer_components": int(last_mask.sum()),
        "top1_purity": top1_real,
        "top5_purity": top5_real,
        "random_baseline_top1_purity": top1_rand,
        "random_baseline_top5_purity": top5_rand,
        "delta_pure": delta,
        "p_value_mcnemar": p_value,
        "passes": passes,
        "reference_range_note": "CLIP-Dissect (Oikarinen & Weng 2023) reports ~0.55-0.76 top-1 purity on ResNet-50 fc; our numbers should fall in that range if the pipeline is faithful.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[M6] wrote {args.out}: top1={top1_real:.4f} (rand {top1_rand:.4f}, delta={delta:.4f}, p={p_value:.3g})  passes={passes}")


if __name__ == "__main__":
    main()
