"""Main driver: for each (layer, feature) pair, run SAGE + baseline, then evaluate.

Saves one JSON per feature under results/<model_id>/<layer>/<feature>.json.
"""
import argparse
import json
import os
import random
import time
import traceback
import numpy as np

from config import RESULTS_DIR, GEMMA_PATH, GEMMA_SAE_ROOT, LOGS_DIR
from sae import load_gemmascope_sae
from model_hook import HookedModel
from neuronpedia import fetch_feature, extract_reference_explanation, extract_top_snippets, annotate_snippet_max, peak_context_text
from sage import run_sage
from evaluate import generative_accuracy, predictive_accuracy, build_heldout_set

# From Neuronpedia's `hfFolderId` metadata per layer (gemma-scope-2b-pt-res, width_16k).
GEMMASCOPE_CANONICAL_L0 = {
    0: 105, 1: 102, 2: 141, 3: 59, 4: 124, 5: 68, 6: 70, 7: 69, 8: 71, 9: 73,
    10: 77, 11: 80, 12: 82, 13: 84, 14: 84, 15: 78, 16: 78, 17: 77, 18: 74, 19: 73,
    20: 71, 21: 70, 22: 72, 23: 75, 24: 73, 25: 116,
}

# global neutral distractor sentences (English varied)
NEUTRAL_DISTRACTORS = [
    "The old library sat quietly at the edge of the park.",
    "She whisked the eggs and folded them into the batter.",
    "Rain drummed on the tin roof through the night.",
    "The train from Milan arrived twenty minutes late.",
    "Small pebbles crunched under his boots on the path.",
    "The senator refused to comment before the hearing.",
    "A soft breeze moved the curtains at dawn.",
    "The company announced quarterly earnings yesterday afternoon.",
    "He plotted the data on a log-log scale.",
    "Two children played chess on a wooden bench.",
    "The recipe called for a pinch of saffron.",
    "The lawyer objected on procedural grounds.",
    "A single cloud drifted over the harbor.",
    "The engine cooled after the long climb.",
    "The bakery opened at seven every morning.",
    "The mountain trail wound past three small lakes.",
]


def choose_features(neuronpedia_data_fn, model_id, layer_str, indices, min_snippets=8, min_maxact=1.0):
    """Filter features that have enough usable data."""
    good = []
    for idx in indices:
        try:
            d = neuronpedia_data_fn(model_id, layer_str, idx)
        except Exception:
            continue
        ref = extract_reference_explanation(d)
        snips = extract_top_snippets(d, k=20)
        if not ref or len(snips) < min_snippets:
            continue
        if not snips[0]["max_val"] or snips[0]["max_val"] < min_maxact:
            continue
        good.append((idx, ref, snips))
    return good


def run_one_feature(feature_idx, ref_explanation, all_snips, hooked_model, sae, layer,
                    K=3, T=3, n_gen=8, n_heldout_pos=6, n_heldout_neg=6, log_prefix=""):
    """Run SAGE + evaluations for a single feature."""
    # Snippet split: first half for SAGE, remainder for held-out predictive eval
    n_all = len(all_snips)
    n_sage = min(12, n_all // 2)
    sage_snips = all_snips[:n_sage]
    heldout_snips = all_snips[n_sage:]  # remaining top-activating
    print(f"{log_prefix}  -> using {len(sage_snips)} snippets for SAGE, "
          f"{len(heldout_snips)} for held-out predictive eval")

    # Compute trigger threshold from OUR SAE's response on Neuronpedia's top-activating snippets,
    # using peak-context windows (short enough that the trigger is in the first ~64 tokens).
    from sage import measure_probes as _mp
    top_texts = [peak_context_text(s, ctx=25) for s in all_snips[:5]]
    top_texts = [t for t in top_texts if t]
    if top_texts:
        top_acts = _mp(top_texts, hooked_model, sae, layer, feature_idx, max_length=96)
        max_ref = max(top_acts) if top_acts else 0.0
    else:
        max_ref = max((s["max_val"] or 0.0) for s in all_snips)
    trigger_threshold = 0.2 * max_ref
    print(f"{log_prefix}  measured max_act={max_ref:.2f}  trigger_thresh={trigger_threshold:.2f}")

    # Run SAGE
    t0 = time.time()
    sage_out = run_sage(sage_snips, hooked_model, sae, layer, feature_idx,
                       K=K, T=T, n_pos=5, n_neg=3, verbose=False)
    t_sage = time.time() - t0
    sage_top = sage_out["top"]
    print(f"{log_prefix}  SAGE top: {sage_top!r} (took {t_sage:.1f}s)")

    # Evaluate SAGE + Neuronpedia
    # Use peak-context windows around the max-activating token so the trigger is inside the truncation.
    heldout_pos_texts = []
    for s in heldout_snips[:n_heldout_pos]:
        pt = peak_context_text(s, ctx=25)
        if pt:
            heldout_pos_texts.append(pt)
    # Dedupe (Neuronpedia often has near-duplicates)
    seen = set()
    heldout_pos_texts = [t for t in heldout_pos_texts if not (t in seen or seen.add(t))]
    neutrals = random.sample(NEUTRAL_DISTRACTORS, min(n_heldout_neg, len(NEUTRAL_DISTRACTORS)))
    heldout = build_heldout_set(heldout_pos_texts, neutrals, n_pos=len(heldout_pos_texts), n_neg=len(neutrals))
    if not heldout:
        heldout = neutrals[:6]
    # true activations for the held-out set
    from sage import measure_probes
    true_acts = measure_probes(heldout, hooked_model, sae, layer, feature_idx, max_length=96)

    results = {}
    for name, expl in [("neuronpedia", ref_explanation), ("sage", sage_top)]:
        gen = generative_accuracy(expl, hooked_model, sae, layer, feature_idx,
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
    p.add_argument("--model", default="gemma-2-2b")
    p.add_argument("--model_path", default=GEMMA_PATH)
    p.add_argument("--sae_root", default=GEMMA_SAE_ROOT)
    p.add_argument("--layers", type=int, nargs="+", default=[12])
    # per-layer canonical L0 for gemmascope-res-16k (Neuronpedia's canonical variant)
    p.add_argument("--l0", type=int, default=None,
                   help="If None, use canonical per-layer L0 from GEMMASCOPE_CANONICAL_L0.")
    p.add_argument("--width", default="16k")
    p.add_argument("--n_features", type=int, default=20)
    p.add_argument("--feature_start", type=int, default=0, help="deterministic starting feature idx range")
    p.add_argument("--feature_step", type=int, default=1, help="stride when sampling features")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--T", type=int, default=3)
    p.add_argument("--n_gen", type=int, default=8)
    p.add_argument("--layer_tag_fmt", default="{layer}-gemmascope-res-16k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_subdir", default="gemma-2-2b")
    p.add_argument("--skip_existing", action="store_true", default=True)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir_root = os.path.join(RESULTS_DIR, args.out_subdir)
    os.makedirs(outdir_root, exist_ok=True)

    hooked_model = HookedModel(args.model_path)
    print(f"Loaded {args.model_path}")

    for layer in args.layers:
        print(f"\n===== LAYER {layer} =====")
        l0 = args.l0 if args.l0 is not None else GEMMASCOPE_CANONICAL_L0.get(layer)
        sae, sae_info = load_gemmascope_sae(args.sae_root, layer=layer, width=args.width, l0=l0)
        print(f"SAE info: {sae_info}")
        layer_str = args.layer_tag_fmt.format(layer=layer)
        # gather candidate feature indices
        cand_indices = list(range(args.feature_start, args.feature_start + args.n_features * args.feature_step * 4, args.feature_step))
        # filter to those that have Neuronpedia data
        goods = choose_features(fetch_feature, args.model, layer_str, cand_indices,
                                min_snippets=8, min_maxact=1.5)
        # keep first n_features
        goods = goods[: args.n_features]
        print(f"Selected {len(goods)} features for layer {layer}")

        for i, (fidx, ref, snips) in enumerate(goods):
            log_prefix = f"[L{layer} f{fidx} ({i+1}/{len(goods)})]"
            outdir = os.path.join(outdir_root, f"layer_{layer}")
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(outdir, f"feature_{fidx}.json")
            if args.skip_existing and os.path.exists(outpath):
                print(f"{log_prefix}  skip (exists)")
                continue
            print(f"{log_prefix}  ref: {ref!r}")
            try:
                res = run_one_feature(fidx, ref, snips, hooked_model, sae, layer,
                                      K=args.K, T=args.T, n_gen=args.n_gen,
                                      log_prefix=log_prefix)
                res["layer"] = layer
                res["sae_info"] = sae_info
                res["model_id"] = args.model
                res["neuronpedia_ref"] = ref
                with open(outpath, "w") as f:
                    json.dump(res, f, indent=2, default=str)
            except Exception as e:
                err = traceback.format_exc()
                print(f"{log_prefix}  ERROR: {e}\n{err}")
                with open(os.path.join(LOGS_DIR, f"err_{args.model}_L{layer}_f{fidx}.log"), "w") as f:
                    f.write(err)


if __name__ == "__main__":
    main()
