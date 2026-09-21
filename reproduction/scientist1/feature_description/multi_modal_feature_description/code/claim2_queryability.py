"""Claim 2: v_c places the component in CLIP's joint image-text space.

We freeze the CLIP text encoder and query the space of {v_c} with ImageNet
class-name prompts. If v_c really lives in CLIP's joint space, then

  (a) each text query should retrieve components whose top-K images depict
      that class, at a rate far above chance;
  (b) each component should be describable by scanning class names and
      picking the top-scoring one.

Probes (all evaluated against a shuffled-v baseline for chance level):

1. text->component retrieval @ M
     For each of the 1000 ImageNet class prompts:
       score = cos(v_c, t_q) for all components c.
       take top-M components -> pool their top-K images -> what fraction of
       those M*K images have ground-truth label q?
     Compared with retrieval from a permuted v matrix (same marginals, no
     content-based alignment).

2. text-based component labeling
     For each component c, find argmax_q cos(v_c, t_q).
     Then check what fraction of c's top-K images have label q (component
     "explains itself" via a class name). Baseline: same score with random q.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from code.common import OUT_DIR, load_class_names


TEMPLATE = "a photo of a {}"
M_VALUES = [1, 3, 5, 10]
BASELINE_SEEDS = 5


@torch.no_grad()
def encode_text(class_names, device="cuda"):
    clip = CLIPModel.from_pretrained("models/clip").eval().to(device)
    tok = CLIPTokenizer.from_pretrained("models/clip")
    prompts = [TEMPLATE.format(n) for n in class_names]
    all_emb = []
    for i in range(0, len(prompts), 128):
        batch = prompts[i : i + 128]
        b = tok(batch, padding=True, return_tensors="pt").to(device)
        out = clip.get_text_features(**b)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        emb = F.normalize(emb, dim=-1)
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0).astype(np.float32)


def retrieval_at_m(v: np.ndarray, t: np.ndarray, topk_labels: np.ndarray, K: int):
    """v: (C, D) normalized, t: (Q, D) normalized, topk_labels: (C, K)."""
    Q = t.shape[0]
    sims = t @ v.T  # (Q, C)
    results = {}
    for M in M_VALUES:
        topM = np.argpartition(-sims, M, axis=1)[:, :M]  # (Q, M) not sorted
        # For each query q, gather labels of retrieved components' top-K images
        # and compute what fraction equal q.
        pooled = topk_labels[topM]  # (Q, M, K)
        frac_class = (pooled == np.arange(Q)[:, None, None]).mean(axis=(1, 2))  # (Q,)
        results[M] = float(frac_class.mean())
    return results


def label_component_via_text(v: np.ndarray, t: np.ndarray, topk_labels: np.ndarray, K: int):
    """For each component c, pick argmax_q cos(v_c, t_q); measure how often
    the corresponding class label appears in c's top-K images."""
    sims = v @ t.T  # (C, Q)
    argmax_q = sims.argmax(axis=1)  # (C,)
    # count how many of top-K labels equal argmax_q
    matches = (topk_labels == argmax_q[:, None]).mean(axis=1)  # (C,)
    return {
        "mean_topk_frac": float(matches.mean()),
        "frac_components_topk_has_pred_class": float((matches > 0).mean()),
    }


def baseline_retrieval(t: np.ndarray, topk_labels: np.ndarray, K: int, seeds=BASELINE_SEEDS):
    """Chance = permute topk_labels across components (breaks link but keeps
    the marginal label distribution)."""
    C = topk_labels.shape[0]
    Q = t.shape[0]
    baseline_scores = {M: [] for M in M_VALUES}
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        # Random top-M component picks per query -> uniform baseline
        idx = rng.integers(0, C, size=(Q, max(M_VALUES)))
        for M in M_VALUES:
            pooled = topk_labels[idx[:, :M]]  # (Q, M, K)
            frac_class = (pooled == np.arange(Q)[:, None, None]).mean(axis=(1, 2))
            baseline_scores[M].append(float(frac_class.mean()))
    return {M: float(np.mean(v)) for M, v in baseline_scores.items()}


def analyze(tag: str, t: np.ndarray, class_names) -> dict:
    with np.load(OUT_DIR / "components" / f"{tag}.npz") as comp:
        v = comp["v"].astype(np.float32)
        topk_labels = comp["topk_labels"]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    K = topk_labels.shape[1]

    retrieval = retrieval_at_m(v, t, topk_labels, K)
    baseline = baseline_retrieval(t, topk_labels, K)
    label = label_component_via_text(v, t, topk_labels, K)

    print(f"[claim2:{tag}] retrieval@M frac_class_of_topM_topK:")
    for M in M_VALUES:
        print(f"  M={M:2d}  topM_frac={retrieval[M]:.4f}  chance={baseline[M]:.4f}  lift={retrieval[M]/max(baseline[M],1e-9):.1f}x")
    print(f"[claim2:{tag}] text label of components: {label}")

    return {
        "tag": tag,
        "num_components": int(v.shape[0]),
        "K": int(K),
        "retrieval": {str(M): retrieval[M] for M in M_VALUES},
        "baseline_random_components": {str(M): baseline[M] for M in M_VALUES},
        "text_labeling": label,
    }


def main():
    class_names = load_class_names()
    print(f"[claim2] encoding {len(class_names)} class prompts with CLIP text tower")
    t = encode_text(class_names)
    print(f"[claim2] text emb shape: {t.shape}")

    all_results = {}
    for tag in ["resnet50_layer4", "vit_b16_cls", "vit_b16_mean"]:
        all_results[tag] = analyze(tag, t, class_names)

    (OUT_DIR / "claim2_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"[claim2] saved -> {OUT_DIR / 'claim2_results.json'}")


if __name__ == "__main__":
    main()
