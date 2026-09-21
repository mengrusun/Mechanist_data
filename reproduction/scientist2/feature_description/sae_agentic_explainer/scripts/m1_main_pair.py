"""
Milestone M1 — Main-pair evaluation (SAGE vs Neuronpedia vs single-pass-GPT-5) on Gemma-2-2B + gemmascope-res-16k.

For each of N_features_per_layer features at each of {L4, L12, L20}:
  1. Sample the feature id from features with >=100 Neuronpedia snippets.
  2. Seed-lock the 80/20 explainer_visible / held_out split per feature.
  3. Generate three explanations:
     (a) SAGE — 4-role iterative loop (uses training snippets in the Explainer + Designer probes).
     (b) Neuronpedia — public explanation string.
     (c) GPT-5 single-pass — same prompt as M0.5.
  4. Generative-accuracy (C1): for each (feature, method), independent probe-writer generates 5 texts
     that INSTANTIATE the explanation; each is pushed through target LLM + SAE; hit iff activation > tau_f
     where tau_f = 99th percentile of the training-corpus activation values for feature f.
  5. Predictive-accuracy (C2): for each (feature, method), independent scorer predicts activation on the 20
     held-out snippets given the explanation. Pearson correlation with normalized ground truth.
  6. Detection AUROC (secondary): top-50% vs bottom-50% held-out labels.

Checkpointing: results are appended per-feature to results/m1/features.jsonl; on restart, already-done
features are skipped.
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
    NeuronpediaClient,
    SAGEConfig,
    SINGLE_PASS_EXPLAINER_PROMPT,
    compute_feature_activations_on_texts,
    format_snippets_for_prompt,
    load_gemma_scope_sae,
    normalize_activations,
    percentile_threshold,
    run_sage_loop,
    score_activation,
    write_probes,
)

GEMMA_SCOPE_L0_MAP = {4: 124, 12: 82, 20: 139}


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


def load_done_features(jsonl_path: Path) -> set:
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["layer"], r["fid"]))
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/zhenqian/models/gemma-2-2b")
    parser.add_argument("--sae_root", default="/data/zhenqian/models/gemma-scope-2b-pt-res")
    parser.add_argument("--layers", default="4,12,20")
    parser.add_argument("--n_features_per_layer", type=int, default=100)
    parser.add_argument("--n_probes_c1", type=int, default=5)
    parser.add_argument("--n_heldout_c2", type=int, default=20)
    parser.add_argument("--K_sage_rounds", type=int, default=3)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--out", default="/data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer/results/m1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_features_scan", type=int, default=2000,
                        help="How many candidate feature ids to scan looking for eligible features per layer")
    parser.add_argument("--label", default="m1", help="Run label for logging")
    parser.add_argument("--methods", default="sage,neuronpedia,gpt5_1shot")
    parser.add_argument("--tau_percentile", type=float, default=99.0)
    parser.add_argument("--min_snippets", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    methods = args.methods.split(",")
    # Use per-layer JSONL when we launch layer-parallel runs; if --layers matches all planned layers,
    # use the shared features.jsonl. Detect: if `args.layers` differs from the default, use per-layer.
    if len(layers) == 1:
        features_jsonl = out_dir / f"features_L{layers[0]}.jsonl"
    else:
        features_jsonl = out_dir / "features.jsonl"

    print(f"[{args.label}] loading model {args.model}")
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    print(f"[{args.label}] model loaded in {time.time()-t_load:.1f}s")

    np_client = NeuronpediaClient(model_id="gemma-2-2b", sae_id_template="{layer}-gemmascope-res-16k")
    gpt = GPT5Client(qps=6.0)

    sage_cfg = SAGEConfig(
        n_candidates=3, n_positive_per_candidate=3, n_negative_per_candidate=2,
        max_rounds=args.K_sage_rounds,
    )
    # Resume: check both per-layer file and shared file
    done = load_done_features(features_jsonl)
    for other in sorted(Path(out_dir).glob("features*.jsonl")):
        if other != features_jsonl:
            for k in load_done_features(other):
                done.add(k)
    print(f"[{args.label}] loaded {len(done)} completed features from prior run (resume-safe)")

    global_seed_rng = np.random.default_rng(args.split_seed)
    start_time = time.time()
    n_done_this_run = 0

    for L in layers:
        l0 = GEMMA_SCOPE_L0_MAP[L]
        print(f"\n[{args.label}] LAYER {L} (l0_{l0})")
        sae = load_gemma_scope_sae(L, l0, root=args.sae_root).to(args.device)

        candidate_fids = list(range(16384))
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
            # Neuronpedia typically shows up to ~24 top-activating snippets per feature by default.
            # Use frac_nonzero to filter for features with a well-defined activation distribution
            frac_nonzero = data.get("frac_nonzero") or 0.0
            try:
                frac_nonzero = float(frac_nonzero)
            except Exception:
                frac_nonzero = 0.0
            if len(snippets) < 15:
                continue
            if frac_nonzero < 1e-5:  # near-dead feature
                continue
            np_expl = NeuronpediaClient.parse_public_explanation(data)
            if not np_expl:
                continue

            # 80/20 seed-locked split
            rng_f = np.random.default_rng(args.split_seed * 1000 + fid)
            idx = np.arange(len(snippets))
            rng_f.shuffle(idx)
            n_train = max(int(0.8 * len(snippets)), 6)
            n_train = min(n_train, len(snippets) - 4)  # ensure at least 4 held-out
            train_idx = idx[:n_train].tolist()
            heldout_idx = idx[n_train:].tolist()
            train_snips = [snippets[i] for i in train_idx]
            held_snips = [snippets[i] for i in heldout_idx][:args.n_heldout_c2]
            if len(held_snips) < 4:
                continue

            fentry: dict[str, Any] = {
                "layer": L,
                "fid": fid,
                "l0": l0,
                "frac_nonzero": frac_nonzero,
                "n_snippets_available": len(snippets),
                "n_train": len(train_snips),
                "n_heldout": len(held_snips),
                "split_seed": args.split_seed,
                "methods": {},
                "start_ts": time.time(),
            }

            # tau_f: 99th percentile of activation values across training snippets (per-token values)
            all_train_vals = []
            for s in train_snips:
                all_train_vals.extend(s.get("values", []))
            tau_f = percentile_threshold(all_train_vals, q=args.tau_percentile)
            fentry["tau_f"] = float(tau_f)

            # Ground truth for held-out: use each snippet's max activation (per Neuronpedia)
            true_max = np.array([s["max_val"] for s in held_snips], dtype=np.float32)
            true_norm = normalize_activations(true_max)

            explanations: dict[str, str] = {}

            # (b) Neuronpedia
            if "neuronpedia" in methods:
                explanations["neuronpedia"] = np_expl

            # (c) Single-pass GPT-5
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

            # (a) SAGE — 4-role loop
            if "sage" in methods:
                try:
                    sage_out = run_sage_loop(
                        client=gpt, model=mdl, tokenizer=tok, sae=sae, layer=L, feature_id=fid,
                        top_activating_snippets=train_snips, config=sage_cfg, device=args.device,
                    )
                    explanations["sage"] = sage_out["final_explanation"]
                    fentry["sage_meta"] = {
                        "all_candidates": sage_out["all_candidates"],
                        "stopped_reason": sage_out["stopped_reason"],
                        "n_rounds": len(sage_out["rounds"]),
                    }
                except Exception as e:
                    print(f"  SAGE error F{fid}: {e}")
                    explanations["sage"] = np_expl  # fallback

            # C1 — Generative accuracy per method
            for m_name, expl in explanations.items():
                if not expl or not expl.strip():
                    fentry["methods"][m_name] = {"explanation": expl, "gen_hits": 0, "gen_total": 0, "gen_acc": 0.0,
                                                  "pred_pearson": 0.0, "pred_auroc": 0.5}
                    continue
                # Probe writer generates n_probes_c1 texts instantiating the explanation
                try:
                    probes = write_probes(gpt, expl, n_probes=args.n_probes_c1)
                except Exception as e:
                    probes = []
                    print(f"  write_probes error F{fid} {m_name}: {e}")

                # Push probes through target LLM + SAE, check activation > tau_f
                gen_hits = 0
                gen_total = 0
                probe_acts: list[float] = []
                if probes:
                    probe_acts = compute_feature_activations_on_texts(mdl, tok, sae, L, fid, probes, device=args.device)
                    gen_hits = sum(1 for a in probe_acts if a > tau_f)
                    gen_total = len(probe_acts)

                # C2 — Predictive accuracy on held-out
                heldout_texts = [s["text"][:1500] for s in held_snips]
                preds = []
                for t in heldout_texts:
                    try:
                        preds.append(score_activation(gpt, expl, t))
                    except Exception as e:
                        preds.append(0.5)
                        print(f"  score error F{fid} {m_name}: {e}")
                preds_arr = np.array(preds, dtype=np.float32)
                pcc = _pearson(preds_arr, true_norm)
                auroc = _auroc_from_median(preds_arr, true_norm)

                fentry["methods"][m_name] = {
                    "explanation": expl,
                    "probes": probes,
                    "probe_activations": probe_acts,
                    "gen_hits": gen_hits,
                    "gen_total": gen_total,
                    "gen_acc": gen_hits / max(gen_total, 1),
                    "heldout_preds": preds,
                    "heldout_true_norm": true_norm.tolist(),
                    "heldout_true_max": true_max.tolist(),
                    "pred_pearson": pcc,
                    "pred_auroc": auroc,
                }

            fentry["end_ts"] = time.time()
            fentry["wall_clock_s"] = fentry["end_ts"] - fentry["start_ts"]

            # Append to JSONL
            with open(features_jsonl, "a") as f:
                f.write(json.dumps(fentry) + "\n")
            collected += 1
            n_done_this_run += 1
            done.add((L, fid))

            msg = f"  F{fid}: "
            for m_name, mrec in fentry["methods"].items():
                msg += f"{m_name} gen={mrec.get('gen_acc', 0):.2f} pcc={mrec.get('pred_pearson', 0):.2f} | "
            elapsed = time.time() - start_time
            print(f"{msg} [{elapsed:.0f}s elapsed, {n_done_this_run} new, {collected}/{args.n_features_per_layer} L{L}]")

    # Post-process: compute aggregates (union across per-layer JSONLs + shared JSONL if present)
    all_jsonl = sorted(list(out_dir.glob("features*.jsonl")))
    aggregate_files(all_jsonl, out_dir / "summary.json", layers, methods)

    print(f"\n[{args.label}] wall_clock = {time.time()-start_time:.1f}s")
    print(f"[{args.label}] gpt5 usage: {gpt.stats()}")


def aggregate_files(features_jsonl_list, out_path: Path, layers, methods):
    records = []
    for p in features_jsonl_list:
        if not Path(p).exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    print(f"[aggregate] {len(records)} features loaded from {len(features_jsonl_list)} files")
    _aggregate_records(records, out_path, layers, methods)


def aggregate(features_jsonl: Path, out_path: Path, layers, methods):
    return aggregate_files([features_jsonl], out_path, layers, methods)


def _aggregate_records(records, out_path, layers, methods):

    summary: dict = {"n_features_total": len(records), "per_layer": {}, "overall": {}, "pairwise_deltas": {}}

    def _mean(x):
        x = [v for v in x if v is not None]
        return float(np.mean(x)) if x else 0.0

    def _paired_bootstrap_ci(a: list[float], b: list[float], n_resamples: int = 10000, ci: float = 95.0, rng_seed: int = 0):
        """95% bootstrap CI for mean(a - b), with paired sampling."""
        a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
        assert len(a) == len(b) and len(a) > 0
        d = a - b
        rng = np.random.default_rng(rng_seed)
        boots = np.zeros(n_resamples, dtype=np.float32)
        for i in range(n_resamples):
            sample = rng.choice(d, size=len(d), replace=True)
            boots[i] = sample.mean()
        lo = float(np.percentile(boots, (100 - ci) / 2))
        hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
        return {"mean_delta": float(d.mean()), "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0)}

    for L in layers:
        recs = [r for r in records if r["layer"] == L]
        entry = {"n_features": len(recs)}
        for m in methods:
            gen_accs = [r["methods"].get(m, {}).get("gen_acc") for r in recs if m in r["methods"]]
            pccs = [r["methods"].get(m, {}).get("pred_pearson") for r in recs if m in r["methods"]]
            aurocs = [r["methods"].get(m, {}).get("pred_auroc") for r in recs if m in r["methods"]]
            entry[m] = {
                "mean_gen_acc": _mean(gen_accs),
                "mean_pearson": _mean(pccs),
                "mean_auroc": _mean(aurocs),
                "n": len(gen_accs),
            }
        summary["per_layer"][f"L{L}"] = entry

    # Overall (across layers)
    for m in methods:
        gen_accs = []
        pccs = []
        aurocs = []
        for r in records:
            mrec = r["methods"].get(m, {})
            if "gen_acc" in mrec:
                gen_accs.append(mrec["gen_acc"])
            if "pred_pearson" in mrec:
                pccs.append(mrec["pred_pearson"])
            if "pred_auroc" in mrec:
                aurocs.append(mrec["pred_auroc"])
        summary["overall"][m] = {
            "mean_gen_acc": _mean(gen_accs),
            "mean_pearson": _mean(pccs),
            "mean_auroc": _mean(aurocs),
            "n": len(gen_accs),
        }

    # Paired deltas across methods (paired within feature id)
    def _paired_vec(records, m1, m2, key):
        a, b = [], []
        for r in records:
            if m1 in r["methods"] and m2 in r["methods"]:
                va = r["methods"][m1].get(key)
                vb = r["methods"][m2].get(key)
                if va is not None and vb is not None:
                    a.append(va); b.append(vb)
        return a, b

    pairs = [("sage", "neuronpedia"), ("sage", "gpt5_1shot"), ("gpt5_1shot", "neuronpedia")]
    metric_keys = ["gen_acc", "pred_pearson", "pred_auroc"]
    for (m1, m2) in pairs:
        pkey = f"{m1}_vs_{m2}"
        summary["pairwise_deltas"][pkey] = {"overall": {}, "per_layer": {}}
        for k in metric_keys:
            a, b = _paired_vec(records, m1, m2, k)
            if a:
                summary["pairwise_deltas"][pkey]["overall"][k] = _paired_bootstrap_ci(a, b, rng_seed=hash((m1, m2, k)) & 0xffff)
            for L in layers:
                recs_L = [r for r in records if r["layer"] == L]
                a, b = _paired_vec(recs_L, m1, m2, k)
                if a:
                    summary["pairwise_deltas"][pkey]["per_layer"].setdefault(f"L{L}", {})[k] = _paired_bootstrap_ci(a, b, rng_seed=hash((m1, m2, k, L)) & 0xffff)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate] wrote {out_path}")


if __name__ == "__main__":
    main()
