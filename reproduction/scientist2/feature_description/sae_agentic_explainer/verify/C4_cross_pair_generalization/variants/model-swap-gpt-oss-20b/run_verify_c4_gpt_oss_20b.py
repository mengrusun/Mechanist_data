"""
Verify variant for C4: model-swap to GPT-OSS-20B + resid-post-aa.

Protocol: identical to M2 (scripts/m2_verify_pair.py) except:
  - model_id = "gpt-oss-20b" (GPT-NeoX-20B on Neuronpedia)
  - sae_id_template = "{layer}-resid-post-aa"
  - layers = [3, 11, 19]  (early/mid/late of 44-layer GPT-NeoX-20B)
  - n_heldout_c2 = 10 (gpt-oss-20b has ~40 activations/feature; ~8 held-out at 80/20)

All other hyperparameters frozen:
  - split_seed = 42
  - methods = ["sage_lite", "neuronpedia", "gpt5_1shot"]
  - SAGE-lite = Explainer + Reviewer roles only (no empirical activation feedback)
  - Scorer = independent GPT-5 (DMXAPI)
  - Metric = Pearson correlation on held-out Neuronpedia activations (predictive accuracy)

This script uses NO target-LLM forward pass (GPT-OSS-20B not available locally).
Ground truth = Neuronpedia's cached maxValue for gpt-oss-20b features (same pattern as M2).

Output:
  verify/C4_cross_pair_generalization/variants/model-swap-gpt-oss-20b/
    features.jsonl  -- per-feature results
    summary.json    -- aggregate statistics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

# Add scripts dir to path
SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from common import (  # noqa: E402
    GPT5Client,
    NeuronpediaClient,
    SINGLE_PASS_EXPLAINER_PROMPT,
    _parse_json_block,
    format_snippets_for_prompt,
    normalize_activations,
    score_activation,
)

# --- SAGE-lite prompts (copied verbatim from m2_verify_pair.py for zero-diff) ---

SAGE_LITE_EXPLAINER = """You are the Explainer role in an iterative SAE-feature-explanation loop. You are given top-activating text snippets for one SAE feature. Propose {n_candidates} DISTINCT candidate one-line natural-language explanations, each capturing a different possible activation pattern.

Top activating snippets:
{snippets}

Output STRICTLY as JSON:
{{"candidates": ["candidate 1", "candidate 2", "candidate 3"]}}"""

SAGE_LITE_REVIEWER = """You are the Reviewer role. Given the top-activating snippets and multiple candidate explanations, pick the SINGLE best explanation and refine it into ONE final one-line phrase (5-15 words). Consider: coverage of the snippets, discriminating power, specificity vs generality.

Snippets:
{snippets}

Candidates:
{candidates}

Output STRICTLY as JSON:
{{"final": "the single best refined explanation"}}"""


def sage_lite(client, snippets, n_candidates=3):
    """Identical to m2_verify_pair.py sage_lite()."""
    snippet_str = format_snippets_for_prompt(snippets, k=10)
    p1 = SAGE_LITE_EXPLAINER.format(n_candidates=n_candidates, snippets=snippet_str)
    out1 = client.chat([{"role": "user", "content": p1}], temperature=0.7, max_tokens=500)
    j1 = _parse_json_block(out1) or {}
    cands = [c.strip() for c in (j1.get("candidates") or []) if isinstance(c, str) and c.strip()][:n_candidates]
    if not cands:
        return {"final": "", "candidates": []}
    cand_str = "\n".join([f"[{i}] {c}" for i, c in enumerate(cands)])
    p2 = SAGE_LITE_REVIEWER.format(snippets=snippet_str, candidates=cand_str)
    out2 = client.chat([{"role": "user", "content": p2}], temperature=0.4, max_tokens=100)
    j2 = _parse_json_block(out2) or {}
    final = (j2.get("final") or "").strip()
    if not final:
        final = cands[0]
    return {"final": final, "candidates": cands}


def _pearson(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _auroc_from_median(preds, true_norm):
    med = float(np.median(true_norm))
    labels = (true_norm > med).astype(np.int32)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    return float(roc_auc_score(labels, preds))


def _paired_bootstrap_ci(a, b, n_resamples: int = 10000, ci: float = 95.0, rng_seed: int = 0):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = a - b
    rng = np.random.default_rng(rng_seed)
    boots = np.zeros(n_resamples, dtype=np.float32)
    for i in range(n_resamples):
        boots[i] = rng.choice(d, size=len(d), replace=True).mean()
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return {"mean_delta": float(d.mean()), "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="gpt-oss-20b")
    parser.add_argument("--sae_id_template", default="{layer}-resid-post-aa")
    parser.add_argument("--layers", default="3,11,19", help="Early/mid/late layers for 44-layer GPT-NeoX-20B")
    parser.add_argument("--n_features_per_layer", type=int, default=15)
    parser.add_argument("--n_heldout_c2", type=int, default=10)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_features_scan", type=int, default=2000)
    parser.add_argument("--label", default="verify_c4_gpt_oss_20b")
    parser.add_argument("--methods", default="sage_lite,neuronpedia,gpt5_1shot")
    parser.add_argument("--out", default=str(Path(__file__).parent))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_jsonl = out_dir / "features.jsonl"
    layers = [int(x) for x in args.layers.split(",")]
    methods = args.methods.split(",")

    # Neuronpedia client for gpt-oss-20b + resid-post-aa
    np_client = NeuronpediaClient(
        model_id=args.model_id,
        sae_id_template=args.sae_id_template,
        cache_dir=str(out_dir / "neuronpedia_cache"),
    )
    gpt = GPT5Client(qps=5.0)

    # Resume
    done: set = set()
    if features_jsonl.exists():
        with open(features_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["layer"], r["fid"]))
                except Exception:
                    pass
    print(f"[{args.label}] resume: {len(done)} features already done")

    start_time = time.time()
    n_done_this_run = 0

    for L in layers:
        print(f"\n[{args.label}] LAYER {L} ({args.model_id} + {args.sae_id_template.format(layer=L)})")
        candidate_fids = list(range(100000))  # resid-post-aa SAE has many features
        rng = np.random.default_rng(args.split_seed * 100 + L)
        rng.shuffle(candidate_fids)

        collected = 0
        scanned = 0
        for fid in candidate_fids:
            if collected >= args.n_features_per_layer or scanned >= args.max_features_scan:
                break
            scanned += 1
            if (L, fid) in done:
                collected += 1
                continue
            data = np_client.get_feature(L, fid)
            if data.get("error"):
                continue
            snippets = NeuronpediaClient.parse_activation_snippets(data, max_snippets=50)
            if len(snippets) < 8:  # need at least 8 for 80/20 split with >=3 held-out
                continue
            np_expl = NeuronpediaClient.parse_public_explanation(data)
            if not np_expl:
                continue
            frac_nonzero = data.get("frac_nonzero") or 0.0
            try:
                frac_nonzero = float(frac_nonzero)
            except Exception:
                frac_nonzero = 0.0
            if frac_nonzero < 1e-7:  # near-dead feature
                continue

            # Seed-locked 80/20 split per feature (same formula as m2_verify_pair.py)
            rng_f = np.random.default_rng(args.split_seed * 1000 + fid + L * 1000000)
            idx = np.arange(len(snippets))
            rng_f.shuffle(idx)
            n_train = max(int(0.8 * len(snippets)), 4)
            n_train = min(n_train, len(snippets) - 3)
            train_idx = idx[:n_train].tolist()
            heldout_idx = idx[n_train:].tolist()
            train_snips = [snippets[i] for i in train_idx]
            held_snips = [snippets[i] for i in heldout_idx][:args.n_heldout_c2]
            if len(held_snips) < 3:
                continue

            fentry: dict[str, Any] = {
                "layer": L,
                "fid": fid,
                "n_snippets_available": len(snippets),
                "n_train": len(train_snips),
                "n_heldout": len(held_snips),
                "split_seed": args.split_seed,
                "frac_nonzero": frac_nonzero,
                "methods": {},
                "start_ts": time.time(),
            }

            true_max = np.array([s["max_val"] for s in held_snips], dtype=np.float32)
            true_norm = normalize_activations(true_max)

            explanations: dict[str, str] = {}
            if "neuronpedia" in methods:
                explanations["neuronpedia"] = np_expl
            if "gpt5_1shot" in methods:
                snippet_str = format_snippets_for_prompt(train_snips, k=10)
                p = SINGLE_PASS_EXPLAINER_PROMPT.format(snippets=snippet_str)
                try:
                    exp = gpt.chat([{"role": "user", "content": p}], temperature=0.4, max_tokens=80).strip()
                    exp = exp.strip('"').strip("'").splitlines()[0].strip()
                except Exception as e:
                    print(f"  gpt5_1shot error F{fid}: {e}")
                    exp = np_expl
                explanations["gpt5_1shot"] = exp
            if "sage_lite" in methods:
                try:
                    sl = sage_lite(gpt, train_snips)
                    explanations["sage_lite"] = sl["final"]
                    fentry["sage_lite_meta"] = {"candidates": sl["candidates"]}
                except Exception as e:
                    print(f"  sage_lite error F{fid}: {e}")
                    explanations["sage_lite"] = np_expl

            # C2 — predictive accuracy on held-out
            for m_name, expl in explanations.items():
                if not expl or not expl.strip():
                    fentry["methods"][m_name] = {"explanation": expl, "pred_pearson": 0.0, "pred_auroc": 0.5}
                    continue
                heldout_texts = [s["text"][:1500] for s in held_snips]
                preds = []
                for t in heldout_texts:
                    try:
                        preds.append(score_activation(gpt, expl, t))
                    except Exception as e:
                        print(f"  score error F{fid} {m_name}: {e}")
                        preds.append(0.5)
                preds_arr = np.array(preds, dtype=np.float32)
                pcc = _pearson(preds_arr, true_norm)
                auroc = _auroc_from_median(preds_arr, true_norm)
                fentry["methods"][m_name] = {
                    "explanation": expl,
                    "heldout_preds": preds,
                    "heldout_true_norm": true_norm.tolist(),
                    "heldout_true_max": true_max.tolist(),
                    "pred_pearson": pcc,
                    "pred_auroc": auroc,
                }

            fentry["end_ts"] = time.time()
            with open(features_jsonl, "a") as f:
                f.write(json.dumps(fentry) + "\n")
            collected += 1
            n_done_this_run += 1
            done.add((L, fid))

            msg = f"  F{fid}: "
            for m_name, mrec in fentry["methods"].items():
                msg += f"{m_name} pcc={mrec.get('pred_pearson', 0):.2f} | "
            elapsed = time.time() - start_time
            print(f"{msg} [{elapsed:.0f}s, {n_done_this_run} new, {collected}/{args.n_features_per_layer} L{L}]")

    # Aggregate
    aggregate(features_jsonl, out_dir / "summary.json", layers, methods)
    elapsed_total = time.time() - start_time
    print(f"\n[{args.label}] Done. wall_clock={elapsed_total:.1f}s, n_done={n_done_this_run}")
    print(f"[{args.label}] gpt5 usage: {gpt.stats()}")
    # Write cost record
    import os
    gpu_ids_str = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    gpu_ids = [int(g) for g in gpu_ids_str.split(",") if g.strip().isdigit()]
    # Compute approximate GPU-hours: this script does no GPU compute; assign 0.02h overhead for env startup
    gpu_hours = 0.02
    cost = {"gpu_ids": gpu_ids, "gpu_hours": gpu_hours, "wall_clock_s": round(elapsed_total, 1)}
    (out_dir / "cost.json").write_text(json.dumps(cost, indent=2))
    print(f"[{args.label}] cost.json written: {cost}")


def aggregate(features_jsonl: Path, out_path: Path, layers, methods):
    if not features_jsonl.exists():
        return
    records = []
    with open(features_jsonl) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    if not records:
        return
    summary: dict = {
        "n_features_total": len(records),
        "target_pair": "gpt-oss-20b + resid-post-aa",
        "layers": layers,
        "per_layer": {},
        "overall": {},
        "pairwise_deltas": {},
    }

    def _mean(x):
        x = [v for v in x if v is not None]
        return float(np.mean(x)) if x else 0.0

    def _paired_bootstrap_ci(a, b, n_resamples=10000, ci=95.0, rng_seed=0):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        d = a - b
        rng = np.random.default_rng(rng_seed)
        boots = np.zeros(n_resamples, dtype=np.float32)
        for i in range(n_resamples):
            boots[i] = rng.choice(d, size=len(d), replace=True).mean()
        lo = float(np.percentile(boots, (100 - ci) / 2))
        hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
        return {"n": len(a), "mean_delta": float(d.mean()), "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0)}

    for L in layers:
        recs = [r for r in records if r["layer"] == L]
        entry = {"n_features": len(recs)}
        for m in methods:
            pccs = [r["methods"].get(m, {}).get("pred_pearson") for r in recs if m in r["methods"]]
            aurocs = [r["methods"].get(m, {}).get("pred_auroc") for r in recs if m in r["methods"]]
            entry[m] = {"mean_pearson": _mean(pccs), "mean_auroc": _mean(aurocs), "n": len(pccs)}
        summary["per_layer"][f"L{L}"] = entry

    for m in methods:
        pccs = [r["methods"].get(m, {}).get("pred_pearson") for r in records if m in r["methods"]]
        aurocs = [r["methods"].get(m, {}).get("pred_auroc") for r in records if m in r["methods"]]
        summary["overall"][m] = {"mean_pearson": _mean(pccs), "mean_auroc": _mean(aurocs), "n": len(pccs)}

    def _paired_vec(records, m1, m2, key):
        a, b = [], []
        for r in records:
            if m1 in r["methods"] and m2 in r["methods"]:
                va = r["methods"][m1].get(key)
                vb = r["methods"][m2].get(key)
                if va is not None and vb is not None:
                    a.append(va)
                    b.append(vb)
        return a, b

    pairs = [("sage_lite", "neuronpedia"), ("sage_lite", "gpt5_1shot"), ("gpt5_1shot", "neuronpedia")]
    metric_keys = ["pred_pearson", "pred_auroc"]
    for (m1, m2) in pairs:
        pkey = f"{m1}_vs_{m2}"
        summary["pairwise_deltas"][pkey] = {"overall": {}, "per_layer": {}}
        for k in metric_keys:
            a, b = _paired_vec(records, m1, m2, k)
            if len(a) >= 3:
                summary["pairwise_deltas"][pkey]["overall"][k] = _paired_bootstrap_ci(
                    a, b, rng_seed=hash((m1, m2, k)) & 0xFFFF
                )
            for L in layers:
                recs_L = [r for r in records if r["layer"] == L]
                a_L, b_L = _paired_vec(recs_L, m1, m2, k)
                if len(a_L) >= 3:
                    summary["pairwise_deltas"][pkey]["per_layer"].setdefault(f"L{L}", {})[k] = _paired_bootstrap_ci(
                        a_L, b_L, rng_seed=hash((m1, m2, k, L)) & 0xFFFF
                    )

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate] wrote {out_path} ({len(records)} features)")


if __name__ == "__main__":
    main()
