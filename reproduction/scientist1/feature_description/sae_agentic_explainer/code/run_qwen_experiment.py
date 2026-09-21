"""Run SAGE + Neuronpedia baseline on Qwen3-4B + transcoder-hp SAE."""
import argparse
import json
import os
import random
import time
import traceback
import numpy as np
import torch

from config import RESULTS_DIR, LOGS_DIR
from qwen_sae import load_qwen_transcoder
from qwen_model_hook import QwenHooked
from neuronpedia import fetch_feature, extract_reference_explanation, extract_top_snippets, peak_context_text
from sage import run_sage as _run_sage
from evaluate import generative_accuracy, predictive_accuracy, build_heldout_set

# Reuse Gemma's neutral distractors
from run_experiment import NEUTRAL_DISTRACTORS, choose_features


class QwenAdapter:
    """Adapter so sage.measure_probes / evaluate can call feature_activation_of on our hook."""
    def __init__(self, qm):
        self.qm = qm
        self.tokenizer = qm.tokenizer

    def feature_activation_of(self, texts, transcoder, layer, feature_idx, max_length=128, mode="max"):
        return self.qm.feature_activation_of(texts, transcoder, layer, feature_idx, max_length=max_length, mode=mode)


def run_one_feature(feature_idx, ref, all_snips, adapter, transcoder, layer,
                    K=3, T=3, n_gen=8, log_prefix=""):
    from sage import measure_probes as _mp
    n_all = len(all_snips)
    n_sage = min(12, n_all // 2)
    sage_snips = all_snips[:n_sage]
    heldout_snips = all_snips[n_sage:]
    print(f"{log_prefix}  -> using {len(sage_snips)} snippets for SAGE, "
          f"{len(heldout_snips)} for held-out")

    top_texts = [peak_context_text(s, ctx=25) for s in all_snips[:5]]
    top_texts = [t for t in top_texts if t]
    top_acts = _mp(top_texts, adapter, transcoder, layer, feature_idx, max_length=96)
    max_ref = max(top_acts) if top_acts else 0.0
    trigger_threshold = 0.2 * max_ref
    print(f"{log_prefix}  measured max_act={max_ref:.2f}  thresh={trigger_threshold:.2f}")
    if max_ref < 1.0:
        print(f"{log_prefix}  SKIP: our SAE gives max_act < 1.0 on top snippets (likely feature deprecated)")
        return None

    t0 = time.time()
    sage_out = _run_sage(sage_snips, adapter, transcoder, layer, feature_idx,
                        K=K, T=T, n_pos=5, n_neg=3, verbose=False)
    t_sage = time.time() - t0
    sage_top = sage_out["top"]
    print(f"{log_prefix}  SAGE top: {sage_top[:200]!r} (took {t_sage:.1f}s)")

    # Held-out predictive
    heldout_pos_texts = []
    for s in heldout_snips[:6]:
        pt = peak_context_text(s, ctx=25)
        if pt:
            heldout_pos_texts.append(pt)
    seen = set()
    heldout_pos_texts = [t for t in heldout_pos_texts if not (t in seen or seen.add(t))]
    neutrals = random.sample(NEUTRAL_DISTRACTORS, min(6, len(NEUTRAL_DISTRACTORS)))
    heldout = build_heldout_set(heldout_pos_texts, neutrals, n_pos=len(heldout_pos_texts), n_neg=len(neutrals))
    if not heldout:
        heldout = neutrals[:6]
    true_acts = _mp(heldout, adapter, transcoder, layer, feature_idx, max_length=96)

    results = {}
    for name, expl in [("neuronpedia", ref), ("sage", sage_top)]:
        gen = generative_accuracy(expl, adapter, transcoder, layer, feature_idx,
                                  trigger_threshold=trigger_threshold, n=n_gen)
        pred = predictive_accuracy(expl, heldout, true_acts)
        results[name] = {"explanation": expl, "generative": gen, "predictive": pred}
        print(f"{log_prefix}  [{name}] trig={gen['trigger_rate']:.2f} "
              f"pearson={pred['pearson']:.2f} spearman={pred['spearman']:.2f}")

    return {
        "feature_idx": feature_idx,
        "max_act_ref": max_ref,
        "trigger_threshold": trigger_threshold,
        "sage": sage_out,
        "heldout_texts": heldout,
        "heldout_true_acts": true_acts,
        "eval": results,
        "sage_seconds": t_sage,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/data/zhenqian/models/Qwen3-4B")
    p.add_argument("--sae_path", default="/data/zhenqian/models/sae/qwen3-4b-transcoders-weights/layer_12.safetensors")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--layer_str", default="12-transcoder-hp")
    p.add_argument("--model_id", default="qwen3-4b")
    p.add_argument("--n_features", type=int, default=10)
    p.add_argument("--feature_start", type=int, default=0)
    p.add_argument("--feature_step", type=int, default=100)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--T", type=int, default=3)
    p.add_argument("--n_gen", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_subdir", default="qwen3-4b_verify")
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    outdir = os.path.join(RESULTS_DIR, args.out_subdir, f"layer_{args.layer}")
    os.makedirs(outdir, exist_ok=True)

    qm = QwenHooked(args.model_path)
    print(f"Loaded {args.model_path}")
    tc = load_qwen_transcoder(args.sae_path)
    print(f"Loaded transcoder {args.sae_path}")
    adapter = QwenAdapter(qm)

    # candidate feature indices
    cand_indices = list(range(args.feature_start,
                              args.feature_start + args.n_features * args.feature_step * 6,
                              args.feature_step))
    goods = choose_features(fetch_feature, args.model_id, args.layer_str, cand_indices,
                            min_snippets=8, min_maxact=2.0)
    goods = goods[: args.n_features]
    print(f"Selected {len(goods)} features")

    for i, (fidx, ref, snips) in enumerate(goods):
        outpath = os.path.join(outdir, f"feature_{fidx}.json")
        if os.path.exists(outpath):
            print(f"  skip f{fidx} (exists)")
            continue
        log_prefix = f"[L{args.layer} f{fidx} ({i+1}/{len(goods)})]"
        print(f"{log_prefix}  ref: {ref!r}")
        try:
            res = run_one_feature(fidx, ref, snips, adapter, tc, args.layer,
                                  K=args.K, T=args.T, n_gen=args.n_gen, log_prefix=log_prefix)
            if res is None:
                continue
            res["layer"] = args.layer
            res["model_id"] = args.model_id
            res["neuronpedia_ref"] = ref
            with open(outpath, "w") as f:
                json.dump(res, f, indent=2, default=str)
        except Exception as e:
            err = traceback.format_exc()
            print(f"{log_prefix}  ERROR: {e}\n{err}")


if __name__ == "__main__":
    main()
