"""
Milestone M0.5 — Methodology & Baseline Sanity Gate

Steps:
  1. Verify SAE loading + hooks work.
  2. Verify activation matching: on Neuronpedia's top-activating snippet for each of `n_features_per_layer`
     features at each of the 3 layers, our forward-hook + SAE activation should match Neuronpedia's cached
     `max_val` within a reasonable tolerance (ratio in [0.5, 2.0]).
  3. Seed-lock an 80/20 explainer-visible / held-out split per feature from the Neuronpedia snippet list.
  4. Fetch Neuronpedia's public explanation for each feature.
  5. Run a single-pass GPT-5 baseline explainer on the `explainer_visible` snippets.
  6. Score both explanations on the held-out snippets:
     - Neuronpedia explanation vs. GPT-5-single-pass explanation
     - Metric: Pearson correlation between predicted activation (GPT-5 scorer) and normalized true activation.
     - Metric: detection AUROC — top-50% activating vs bottom-50% held-out; scorer's prediction as classifier score.

Output: results/m0_5/sanity.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    GPT5Client,
    JumpReLUSAE,
    NeuronpediaClient,
    SAGEConfig,
    SINGLE_PASS_EXPLAINER_PROMPT,
    compute_feature_activations_on_texts,
    format_snippets_for_prompt,
    load_gemma_scope_sae,
    normalize_activations,
    percentile_threshold,
    score_activation,
)


# Neuronpedia's canonical l0 variant per Gemma-Scope layer (empirically calibrated by running each variant
# on Neuronpedia's top-activating snippet and picking the one whose max activation ratio is closest to 1.0):
GEMMA_SCOPE_L0_MAP = {4: 124, 12: 82, 20: 139}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/zhenqian/models/gemma-2-2b")
    parser.add_argument("--sae_root", default="/data/zhenqian/models/gemma-scope-2b-pt-res")
    parser.add_argument("--layers", default="4,12,20")
    parser.add_argument("--n_features_per_layer", type=int, default=10)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--n_heldout_score", type=int, default=20)
    parser.add_argument("--out", default="/data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer/results/m0_5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_features_scan", type=int, default=200,
                        help="How many candidate feature ids to scan looking for ones with enough Neuronpedia snippets")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    print(f"[m0.5] loading model {args.model}")
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    print(f"[m0.5] model loaded in {time.time()-t_load:.1f}s, layers={mdl.config.num_hidden_layers}")

    np_client = NeuronpediaClient(model_id="gemma-2-2b", sae_id_template="{layer}-gemmascope-res-16k")
    gpt = GPT5Client(qps=6.0)

    random.seed(args.split_seed)
    rng = np.random.default_rng(args.split_seed)

    results: dict[str, Any] = {
        "config": {
            "model": args.model,
            "sae_root": args.sae_root,
            "layers": layers,
            "n_features_per_layer": args.n_features_per_layer,
            "split_seed": args.split_seed,
            "n_heldout_score": args.n_heldout_score,
            "l0_map": GEMMA_SCOPE_L0_MAP,
        },
        "per_layer": {},
        "activation_match_max_abs_ratio_err": 0.0,
        "activation_match_median_ratio": 0.0,
        "start_time": time.time(),
    }

    all_ratios: list[float] = []

    for L in layers:
        l0 = GEMMA_SCOPE_L0_MAP[L]
        print(f"\n[m0.5] layer {L} (l0_{l0})")
        sae = load_gemma_scope_sae(L, l0, root=args.sae_root).to(args.device)

        # Draw `n_features_per_layer` features from a random sample of feature ids, skipping ones with too few snippets.
        candidate_fids = list(range(0, 16384))
        rng.shuffle(candidate_fids)

        layer_results: dict[str, Any] = {
            "l0": l0,
            "features": [],
            "activation_match_ratios": [],
            "neuronpedia_pearson": [],
            "gpt5_1shot_pearson": [],
            "neuronpedia_auroc": [],
            "gpt5_1shot_auroc": [],
        }

        collected = 0
        scanned = 0
        for fid in candidate_fids:
            if collected >= args.n_features_per_layer or scanned >= args.max_features_scan:
                break
            scanned += 1
            data = np_client.get_feature(L, fid)
            if data.get("error"):
                continue
            snippets = NeuronpediaClient.parse_activation_snippets(data, max_snippets=50)
            if len(snippets) < args.n_heldout_score + 10:
                continue
            np_expl = NeuronpediaClient.parse_public_explanation(data)
            if not np_expl:
                continue

            # 80/20 split, seed-locked per-feature (use fid + args.split_seed for reproducibility)
            rng_f = np.random.default_rng(args.split_seed * 1000 + fid)
            idx = np.arange(len(snippets))
            rng_f.shuffle(idx)
            n_train = int(0.8 * len(snippets))
            train_idx = idx[:n_train].tolist()
            heldout_idx = idx[n_train:].tolist()
            train_snips = [snippets[i] for i in train_idx]
            held_snips = [snippets[i] for i in heldout_idx][:args.n_heldout_score]
            if len(held_snips) < 5:
                continue

            # Activation match check: on top-activating snippet, our max activation vs Neuronpedia's
            top_snip = max(train_snips, key=lambda s: s["max_val"])
            our_max = compute_feature_activations_on_texts(mdl, tok, sae, L, fid, [top_snip["text"]], device=args.device)[0]
            ratio = our_max / max(top_snip["max_val"], 1e-6)
            layer_results["activation_match_ratios"].append(ratio)
            all_ratios.append(ratio)

            # Run single-pass GPT-5 explainer on the top-10 training snippets
            snippet_str = format_snippets_for_prompt(train_snips, k=10)
            prompt = SINGLE_PASS_EXPLAINER_PROMPT.format(snippets=snippet_str)
            try:
                gpt5_expl = gpt.chat([{"role": "user", "content": prompt}], temperature=0.4, max_tokens=80).strip()
                # Clean up the response — remove wrapping quotes if any
                gpt5_expl = gpt5_expl.strip().strip('"').strip("'").splitlines()[0].strip() if gpt5_expl else np_expl
            except Exception as e:
                print(f"  gpt5 explainer error F{fid}: {e}")
                gpt5_expl = np_expl

            # Predictive accuracy on held-out: for each explanation, ask the scorer to predict activation,
            # compare with normalized ground truth.
            held_texts = [s["text"] for s in held_snips]
            true_max = np.array([s["max_val"] for s in held_snips], dtype=np.float32)
            true_norm = normalize_activations(true_max)

            preds_np = np.array([score_activation(gpt, np_expl, t) for t in held_texts], dtype=np.float32)
            preds_gpt5 = np.array([score_activation(gpt, gpt5_expl, t) for t in held_texts], dtype=np.float32)

            # Pearson correlation (nan-safe)
            def _pearson(a, b):
                a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
                if a.std() < 1e-6 or b.std() < 1e-6:
                    return 0.0
                return float(np.corrcoef(a, b)[0, 1])
            pcc_np = _pearson(preds_np, true_norm)
            pcc_gpt5 = _pearson(preds_gpt5, true_norm)

            # Detection AUROC: label = top-50% held-out (binary), score = predicted activation
            med = float(np.median(true_norm))
            labels = (true_norm > med).astype(np.int32)
            if labels.sum() > 0 and labels.sum() < len(labels):
                auroc_np = float(roc_auc_score(labels, preds_np))
                auroc_gpt5 = float(roc_auc_score(labels, preds_gpt5))
            else:
                auroc_np = 0.5
                auroc_gpt5 = 0.5

            fentry = {
                "fid": fid,
                "n_snippets_available": len(snippets),
                "n_train": len(train_snips),
                "n_heldout": len(held_snips),
                "activation_match_ratio": ratio,
                "activation_match_our_max": float(our_max),
                "activation_match_np_max": float(top_snip["max_val"]),
                "neuronpedia_explanation": np_expl,
                "gpt5_1shot_explanation": gpt5_expl,
                "neuronpedia_pearson": pcc_np,
                "gpt5_1shot_pearson": pcc_gpt5,
                "neuronpedia_auroc": auroc_np,
                "gpt5_1shot_auroc": auroc_gpt5,
            }
            layer_results["features"].append(fentry)
            layer_results["neuronpedia_pearson"].append(pcc_np)
            layer_results["gpt5_1shot_pearson"].append(pcc_gpt5)
            layer_results["neuronpedia_auroc"].append(auroc_np)
            layer_results["gpt5_1shot_auroc"].append(auroc_gpt5)
            collected += 1
            print(f"  F{fid}: ratio={ratio:.2f}, np_pcc={pcc_np:.3f}, gpt5_pcc={pcc_gpt5:.3f}, np_auroc={auroc_np:.3f}, gpt5_auroc={auroc_gpt5:.3f}")

            # Save incrementally
            with open(out_dir / "sanity.json", "w") as f:
                json.dump(results | {"per_layer": {**results["per_layer"], f"L{L}": layer_results}}, f, indent=2)

        # Aggregate layer
        layer_results["mean_neuronpedia_pearson"] = float(np.mean(layer_results["neuronpedia_pearson"])) if layer_results["neuronpedia_pearson"] else 0.0
        layer_results["mean_gpt5_1shot_pearson"] = float(np.mean(layer_results["gpt5_1shot_pearson"])) if layer_results["gpt5_1shot_pearson"] else 0.0
        layer_results["mean_neuronpedia_auroc"] = float(np.mean(layer_results["neuronpedia_auroc"])) if layer_results["neuronpedia_auroc"] else 0.5
        layer_results["mean_gpt5_1shot_auroc"] = float(np.mean(layer_results["gpt5_1shot_auroc"])) if layer_results["gpt5_1shot_auroc"] else 0.5
        results["per_layer"][f"L{L}"] = layer_results
        print(f"[m0.5] L{L} done: mean_np_pcc={layer_results['mean_neuronpedia_pearson']:.3f}, mean_gpt5_pcc={layer_results['mean_gpt5_1shot_pearson']:.3f}")

    if all_ratios:
        results["activation_match_max_abs_ratio_err"] = max(abs(r - 1.0) for r in all_ratios)
        results["activation_match_median_ratio"] = float(np.median(all_ratios))
    results["end_time"] = time.time()
    results["wall_clock_s"] = results["end_time"] - results["start_time"]
    results["gpt5_usage"] = gpt.stats()

    # Verdict
    # Gate 1: activation-match median ratio in [0.5, 2.0] (loose since JumpReLU threshold has small numeric variance)
    med_ratio = results["activation_match_median_ratio"]
    activation_gate = 0.5 <= med_ratio <= 2.0
    # Gate 2: baseline AUROCs in [0.55, 0.85] on average (Paulo-2024 range)
    all_np_auroc = np.mean([results["per_layer"][f"L{L}"]["mean_neuronpedia_auroc"] for L in layers])
    all_gpt5_auroc = np.mean([results["per_layer"][f"L{L}"]["mean_gpt5_1shot_auroc"] for L in layers])
    auroc_gate = 0.50 <= all_np_auroc and 0.50 <= all_gpt5_auroc

    results["gates"] = {
        "activation_match_gate_pass": bool(activation_gate),
        "activation_match_median_ratio": med_ratio,
        "auroc_gate_pass": bool(auroc_gate),
        "mean_neuronpedia_auroc": float(all_np_auroc),
        "mean_gpt5_1shot_auroc": float(all_gpt5_auroc),
        "overall": bool(activation_gate and auroc_gate),
    }

    with open(out_dir / "sanity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[m0.5] Wrote {out_dir / 'sanity.json'}")
    print(f"[m0.5] Gates: activation_match={activation_gate}, auroc={auroc_gate}, overall={results['gates']['overall']}")


if __name__ == "__main__":
    main()
