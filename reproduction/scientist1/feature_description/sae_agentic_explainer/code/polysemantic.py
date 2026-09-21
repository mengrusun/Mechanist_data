"""Post-hoc polysemanticity check.

For each feature we look at the top-K SAGE candidates in the FINAL round.
We (a) score how mutually diverse the candidates are (asking GPT-5 to say how many
distinct concepts they describe) and (b) which candidates have empirical support
(pos > neg activation gap).
"""
import glob
import json
import os
import numpy as np

from api import chat_json
from config import RESULTS_DIR


def rate_polysemantic(candidates, snippets):
    """Ask GPT-5 to score how many distinct concepts the candidates cover."""
    prompt = (
        "Below is a hidden feature's top-activating snippets (peak-activation token "
        "wrapped in <<>>), then several candidate explanations. Count how many DISTINCT "
        "underlying concepts the candidates collectively describe (1 if they all say the "
        "same thing, higher if they identify separate concepts). Also for each candidate, "
        "assign one 'concept_id' number (candidates describing the same concept share an id).\n\n"
        "Snippets:\n"
        + "\n".join(f"  - {s}" for s in snippets)
        + "\n\nCandidates:\n"
        + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))
        + "\n\nReply ONLY as JSON: "
        "{\"n_distinct_concepts\": <int>, \"concept_ids\": [<int>, ...], "
        "\"reason\": \"<one sentence>\"}"
    )
    d = chat_json([
        {"role": "system", "content": "You are an interpretability judge. Be strict."},
        {"role": "user", "content": prompt},
    ])
    return d


def main(subdir="gemma-2-2b_main"):
    root = os.path.join(RESULTS_DIR, subdir)
    files = sorted(glob.glob(os.path.join(root, "layer_*", "feature_*.json")))
    print(f"Found {len(files)} feature files.")
    results = []
    for f in files:
        with open(f) as h:
            r = json.load(h)
        sage = r.get("sage", {})
        cands = sage.get("candidates") or []
        trace = sage.get("trace") or []
        if not cands or not trace:
            continue
        final = trace[-1]["results"]
        empirical_scores = [x["score"] for x in final]
        annotations = [x["hypothesis"] for x in final]

        # Only try if candidates differ (naively check surface diff)
        if len(set(cands)) < 2:
            continue

        # Small snippet subset (from the sage trace's first round would be nice; use first 5 pos probes)
        snips = final[0]["positive_probes"][:3] if final and final[0].get("positive_probes") else []

        try:
            j = rate_polysemantic(cands, snips)
        except Exception:
            j = {"n_distinct_concepts": None, "concept_ids": None, "reason": "error"}

        n_distinct = j.get("n_distinct_concepts")
        n_empirical_support = sum(1 for s in empirical_scores if s > 0)
        results.append({
            "layer": r.get("layer"),
            "feature_idx": r.get("feature_idx"),
            "n_candidates": len(cands),
            "n_distinct_concepts": n_distinct,
            "n_with_empirical_support": n_empirical_support,
            "concept_ids": j.get("concept_ids"),
            "empirical_scores": empirical_scores,
            "candidates": cands,
        })

    outpath = os.path.join(RESULTS_DIR, f"{subdir}__polysemantic.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {outpath}. n_features={len(results)}")
    # Count polysemantic
    poly = [r for r in results if (r.get("n_distinct_concepts") or 1) >= 2 and r.get("n_with_empirical_support", 0) >= 2]
    print(f"Polysemantic features (>=2 distinct concepts AND >=2 with empirical support): {len(poly)} / {len(results)}")


if __name__ == "__main__":
    import sys
    subdir = sys.argv[1] if len(sys.argv) > 1 else "gemma-2-2b_main"
    main(subdir)
