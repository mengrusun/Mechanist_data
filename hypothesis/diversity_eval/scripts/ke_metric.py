"""Knowledge-Extent (KE) diversity metric.

Direct port of ``compute_extent`` from
tsinghua-fib-lab/AI-Impacts-Science (``code/Calculate_Space_Author.py`` and
``code/Calculate_Space_WorkField.py``), the code behind "Artificial
intelligence tools expand scientists' impact but contract science's focus".

Original (verbatim from the repo)::

    AI_embeddings_center = np.mean(AI_embeddings, axis=0)
    AI_extent = np.linalg.norm(AI_embeddings - AI_embeddings_center, axis=1)
    AI_extent_mean = np.mean(AI_extent)
    AI_extent_median = np.median(AI_extent)
    AI_extent_max = np.max(AI_extent)

i.e. KE = mean Euclidean distance of a set of paper embeddings to their own
centroid.  Larger KE == the set occupies a wider region of the embedding
space == more diverse; the paper's headline result is that AI-augmented work
has a *smaller* extent.

Two things are added on top of the verbatim port, both needed to use KE on a
small set of hypotheses rather than on millions of papers:

* ``ke_bootstrap`` reproduces the paper's resampling protocol (``sample_time``
  draws of ``sample_N`` items without replacement, ``np.random.seed(t)``) so
  that the estimate carries a confidence interval and so that two sets of
  different size are always compared at the same n.  KE grows with n, so
  comparing raw KE across differently sized sets is invalid.
* ``ke_report`` also reports KE on L2-normalised embeddings.  Raw KE inherits
  the scale of the encoder (SPECTER2 vectors have norm ~10, Qwen3 vectors are
  unit-norm), so only the normalised value is comparable across encoders.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# core (verbatim port)
# --------------------------------------------------------------------------

def extent_distances(embeddings: np.ndarray) -> np.ndarray:
    """Per-item Euclidean distance to the centroid of `embeddings`."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    center = np.mean(embeddings, axis=0)
    return np.linalg.norm(embeddings - center, axis=1)


def compute_extent(embeddings: np.ndarray) -> dict[str, float]:
    """KE of one set of embeddings.

    ``extent_mean`` -- the mean Euclidean distance of every item to the
    centroid -- IS the metric.  mean/median/max are the three the paper
    reports; sd/min are added so the spread of the distances themselves is
    visible.
    """
    d = extent_distances(embeddings)
    return {
        "extent_mean": float(np.mean(d)),
        "extent_median": float(np.median(d)),
        "extent_max": float(np.max(d)),
        "extent_min": float(np.min(d)),
        "extent_sd": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
    }


# --------------------------------------------------------------------------
# resampled estimate (paper's protocol)
# --------------------------------------------------------------------------

def ke_bootstrap(
    embeddings: np.ndarray,
    sample_n: int,
    sample_time: int = 1000,
    seed_offset: int = 0,
) -> dict[str, np.ndarray]:
    """Repeat KE over `sample_time` subsamples of `sample_n` items.

    Mirrors the paper: ``np.random.seed(t)`` then
    ``np.random.choice(index, min(sample_n, len(index)), replace=False)``.
    Returns the raw per-draw arrays so callers can take means and quantiles.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    n = len(embeddings)
    k = min(sample_n, n)
    if k < 2:
        raise ValueError(f"need >=2 items per draw, got sample_n={sample_n}, n={n}")
    idx = np.arange(n)

    means = np.empty(sample_time)
    medians = np.empty(sample_time)
    maxes = np.empty(sample_time)
    for t in range(sample_time):
        np.random.seed(t + seed_offset)
        pick = np.random.choice(idx, k, replace=False)
        d = extent_distances(embeddings[pick])
        means[t] = d.mean()
        medians[t] = np.median(d)
        maxes[t] = d.max()
    return {"extent_mean": means, "extent_median": medians, "extent_max": maxes}


def _summary(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "ci_low": float(np.percentile(x, 2.5)),
        "ci_high": float(np.percentile(x, 97.5)),
    }


# --------------------------------------------------------------------------
# companion statistics
# --------------------------------------------------------------------------

def pairwise_stats(embeddings: np.ndarray) -> dict[str, float]:
    """Mean pairwise Euclidean and cosine distance (scale-free cross-check).

    KE-to-centroid and mean pairwise distance measure the same spread but the
    pairwise version has no centroid to be dragged around by outliers, so a
    large gap between the two flags a skewed set.
    """
    e = np.asarray(embeddings, dtype=np.float64)
    n = len(e)
    sq = np.sum(e**2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * e @ e.T, 0.0)
    d = np.sqrt(d2)

    en = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)
    cos = np.clip(en @ en.T, -1.0, 1.0)

    iu = np.triu_indices(n, k=1)
    return {
        "pairwise_euclidean_mean": float(d[iu].mean()),
        "pairwise_cosine_distance_mean": float((1.0 - cos)[iu].mean()),
        "mean_embedding_norm": float(np.linalg.norm(e, axis=1).mean()),
    }


def effective_dimension(embeddings: np.ndarray) -> dict[str, float]:
    """Participation ratio of the PCA spectrum + #PCs for 90% variance.

    A set can have a large radius while living on a line; this says how many
    directions the spread actually uses.
    """
    e = np.asarray(embeddings, dtype=np.float64)
    x = e - e.mean(axis=0)
    n = len(e)
    if n < 3:
        return {"participation_ratio": float("nan"), "n_pc_90": float("nan")}
    sv = np.linalg.svd(x, compute_uv=False)
    var = sv**2
    if var.sum() <= 0:
        return {"participation_ratio": 0.0, "n_pc_90": 0.0}
    p = var / var.sum()
    pr = 1.0 / np.sum(p**2)
    n_pc_90 = int(np.searchsorted(np.cumsum(p), 0.90) + 1)
    return {"participation_ratio": float(pr), "n_pc_90": float(n_pc_90)}


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    e = np.asarray(embeddings, dtype=np.float64)
    return e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)


# --------------------------------------------------------------------------
# top-level report for one group
# --------------------------------------------------------------------------

def ke_report(
    embeddings: np.ndarray,
    ids: list[str] | None = None,
    sample_n: int | None = None,
    sample_time: int = 1000,
) -> dict:
    """Full KE report for one set of papers/claims."""
    e = np.asarray(embeddings, dtype=np.float64)
    n = len(e)
    if sample_n is None:
        sample_n = max(2, n // 2)  # paper's author-level rule: min(|A|,|B|)/2
    en = l2_normalize(e)

    out: dict = {
        "n_items": n,
        "dim": int(e.shape[1]),
        "sample_n": int(min(sample_n, n)),
        "sample_time": int(sample_time),
        # KE on all items, no resampling -- the plain point estimate
        "ke_full": compute_extent(e),
        "ke_full_normalized": compute_extent(en),
        # KE under the paper's resampling protocol
        "ke_resampled": {
            k: _summary(v) for k, v in
            ke_bootstrap(e, sample_n, sample_time).items()
        },
        "ke_resampled_normalized": {
            k: _summary(v) for k, v in
            ke_bootstrap(en, sample_n, sample_time).items()
        },
        "pairwise": pairwise_stats(e),
        "pairwise_normalized": pairwise_stats(en),
        "spectrum": effective_dimension(e),
    }

    d = extent_distances(e)
    dn = extent_distances(en)
    order = np.argsort(-d)
    out["per_item"] = [
        {
            "id": ids[i] if ids is not None else str(i),
            "distance_to_centroid": float(d[i]),
            "distance_to_centroid_normalized": float(dn[i]),
        }
        for i in order
    ]
    return out


# --------------------------------------------------------------------------
# two-group comparison (the paper's AI vs non-AI design)
# --------------------------------------------------------------------------

def compare_groups(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    sample_n: int | None = None,
    sample_time: int = 1000,
    normalize: bool = True,
    n_permutations: int = 5000,
) -> dict:
    """Compare KE of two sets at equal n, with CI and a permutation p-value."""
    a = l2_normalize(emb_a) if normalize else np.asarray(emb_a, dtype=np.float64)
    b = l2_normalize(emb_b) if normalize else np.asarray(emb_b, dtype=np.float64)
    if sample_n is None:
        sample_n = max(2, min(len(a), len(b)) // 2)

    ka = ke_bootstrap(a, sample_n, sample_time)["extent_mean"]
    kb = ke_bootstrap(b, sample_n, sample_time)["extent_mean"]
    diff = ka - kb  # paired by seed: same seed -> same draw indices

    pooled = np.vstack([a, b])
    na = len(a)
    obs = float(np.mean(ka) - np.mean(kb))
    rng = np.random.default_rng(0)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(len(pooled))
        pa, pb = pooled[perm[:na]], pooled[perm[na:]]
        k = min(sample_n, len(pa), len(pb))
        null[i] = (
            extent_distances(pa[rng.choice(len(pa), k, replace=False)]).mean()
            - extent_distances(pb[rng.choice(len(pb), k, replace=False)]).mean()
        )
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_permutations + 1))

    return {
        "normalized": normalize,
        "sample_n": int(sample_n),
        "sample_time": int(sample_time),
        "ke_a": _summary(ka),
        "ke_b": _summary(kb),
        "diff_a_minus_b": _summary(diff),
        "relative_diff_pct": float(100.0 * obs / np.mean(kb)) if np.mean(kb) else float("nan"),
        "frac_draws_a_gt_b": float(np.mean(diff > 0)),
        "permutation_p_two_sided": p,
        "n_permutations": int(n_permutations),
    }
