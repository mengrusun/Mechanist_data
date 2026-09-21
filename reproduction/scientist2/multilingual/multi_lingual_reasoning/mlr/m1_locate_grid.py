"""M1 grid runner — for one seed, sweep (n_probe, rank_r, layer_group) with the model loaded once.

Loads Qwen-3-4B-Thinking once, caches activations at all three representative layers
(early=6, mid=18, all_non_upper=14 — midpoints of the plan's ranges) for the biggest n_probe (1000)
using FLORES-200 probes. Then for each (n_probe, rank_r, layer_group) subsampled config, fits V_lang
from a slice of the pre-cached activations, evaluates on held-out MGSM prompts (also cached once),
and writes one .npz per config.

This turns 90 model reloads into 1 load + ~11k cached forward passes + 90 SVD/classifier ops.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlr.data import MGSM_LANGS, load_flores_all, load_mgsm_all
from mlr.activations import ResidualStreamCache

# Import shared helpers from m1_locate
from mlr.m1_locate import _fit_v_lang, _train_linear_classifier, _eval_classifier, _layer_group_range


N_PROBE_GRID = [50, 100, 250, 500, 1000]
RANK_R_GRID = [1, 2, 4, 8, 16, 32]
LAYER_GROUPS = ["early", "mid", "all_non_upper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_heldout_mgsm", type=int, default=100)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers

    # Representative layers per group: midpoint of range
    rep_layers = {}
    for lg in LAYER_GROUPS:
        lo, hi = _layer_group_range(lg, num_layers)
        rep_layers[lg] = (lo + hi) // 2
    unique_layers = sorted(set(rep_layers.values()))
    print(f"[m1-grid] rep layers: {rep_layers}  unique={unique_layers}", flush=True)

    # Cache probe activations at n_probe = 1000 (biggest); we then slice to smaller n_probe
    max_n_probe = max(N_PROBE_GRID)
    probe_texts = load_flores_all(langs=MGSM_LANGS, n_probe=max_n_probe, seed=args.seed, data_dir=args.data_dir)
    print(f"[m1-grid] probe sizes: " + ", ".join(f"{l}={len(v)}" for l, v in probe_texts.items()), flush=True)

    # acts_by_lang_by_layer[lg][lang] = np.ndarray (N, hidden)
    acts_by_layer_lang: Dict[int, Dict[str, np.ndarray]] = {ly: {} for ly in unique_layers}
    with ResidualStreamCache(model, tokenizer, unique_layers, pool="last") as cache:
        for lang, sents in probe_texts.items():
            enc = cache.encode_dataset(sents, batch_size=args.batch_size, max_length=args.max_length)
            for ly in unique_layers:
                acts_by_layer_lang[ly][lang] = enc[ly]
            print(f"[m1-grid] cached probe {lang}: {enc[unique_layers[0]].shape}", flush=True)

    # Cache held-out MGSM prompt activations at all unique_layers (larger context; use max_length=256 for math prompts)
    mgsm_train = load_mgsm_all(split="train", data_dir=args.data_dir)
    held_texts: List[str] = []
    held_labels: List[int] = []
    langs_sorted = sorted(MGSM_LANGS)
    # MGSM train has only 8 exemplars per language — augment with FLORES-200 devtest sentences as held-out language probes
    # to reach `n_heldout_mgsm` items per language. Use questions where possible.
    for li, lang in enumerate(langs_sorted):
        df = mgsm_train[lang]
        for i in range(min(len(df), args.n_heldout_mgsm)):
            held_texts.append(str(df.iloc[int(i)]["question"]))
            held_labels.append(li)
        # Top up with FLORES sentences until we hit n_heldout_mgsm
        remaining = args.n_heldout_mgsm - min(len(df), args.n_heldout_mgsm)
        if remaining > 0:
            # Use FLORES devtest sentences (parallel — same content, different language)
            from mlr.data import load_flores_probe
            devtest_sents = load_flores_probe(lang, n_probe=remaining, split="devtest", data_dir=args.data_dir, seed=args.seed + 200)
            for s in devtest_sents[:remaining]:
                held_texts.append(s)
                held_labels.append(li)
    with ResidualStreamCache(model, tokenizer, unique_layers, pool="last") as cache:
        # Encode in blocks
        block_size = 1024  # tokens together
        all_heldacts: Dict[int, List[np.ndarray]] = {ly: [] for ly in unique_layers}
        for i in range(0, len(held_texts), 128):
            chunk = held_texts[i : i + 128]
            enc = cache.encode_dataset(chunk, batch_size=args.batch_size, max_length=256)
            for ly in unique_layers:
                all_heldacts[ly].append(enc[ly])
        held_acts_by_layer = {ly: np.concatenate(all_heldacts[ly], axis=0) for ly in unique_layers}
    held_labels_arr = np.array(held_labels, dtype=np.int64)
    print(f"[m1-grid] held-out size per lang: {[int((held_labels_arr==i).sum()) for i in range(len(langs_sorted))]}", flush=True)

    # Free GPU model — we're now in numpy land
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Loop over the grid
    n_configs = 0
    for lg in LAYER_GROUPS:
        ly = rep_layers[lg]
        acts_full = acts_by_layer_lang[ly]  # {lang: (N_max, hidden)}
        held_acts_ly = held_acts_by_layer[ly]

        # Stratified 80/20 split for classifier (fixed per seed)
        rng = np.random.default_rng(args.seed + 100)
        tr_idx_all: List[int] = []
        te_idx_all: List[int] = []
        for lid in range(len(langs_sorted)):
            idxs = np.where(held_labels_arr == lid)[0]
            rng2 = np.random.default_rng(args.seed + 100 + lid)
            rng2.shuffle(idxs)
            split = int(0.8 * len(idxs))
            tr_idx_all.extend(idxs[:split].tolist())
            te_idx_all.extend(idxs[split:].tolist())
        tr_idx = np.array(tr_idx_all, dtype=np.int64)
        te_idx = np.array(te_idx_all, dtype=np.int64)
        X_train = held_acts_ly[tr_idx]
        X_test = held_acts_ly[te_idx]
        y_train = held_labels_arr[tr_idx]
        y_test = held_labels_arr[te_idx]

        for n_probe in N_PROBE_GRID:
            # Slice per-lang activations to first `n_probe`
            acts_per_lang = {lang: arr[:n_probe] for lang, arr in acts_full.items()}
            for rank_r in RANK_R_GRID:
                out_path = out_dir / f"n{n_probe}_r{rank_r}_{lg}_s{args.seed}.npz"
                if out_path.exists():
                    n_configs += 1
                    continue
                V, mus = _fit_v_lang(acts_per_lang, rank_r)
                P = V @ V.T
                proj = X_train @ P
                proj_te = X_test @ P
                comp = X_train - proj
                comp_te = X_test - proj_te
                clf_v = _train_linear_classifier(proj, y_train, n_classes=len(langs_sorted), seed=args.seed)
                macro_v, per_lang_v = _eval_classifier(clf_v, proj_te, y_test, langs_sorted)
                clf_c = _train_linear_classifier(comp, y_train, n_classes=len(langs_sorted), seed=args.seed)
                macro_c, per_lang_c = _eval_classifier(clf_c, comp_te, y_test, langs_sorted)

                # Content-probe subspace (per-lg)
                en_idx_local = np.where(held_labels_arr == langs_sorted.index("en"))[0]
                if len(en_idx_local) >= 8:
                    rng_c = np.random.default_rng(args.seed + 500 + rank_r)
                    diffs = []
                    for _ in range(10):
                        idx = rng_c.permutation(en_idx_local)
                        half = len(idx) // 2
                        a = held_acts_ly[idx[:half]].mean(axis=0)
                        b = held_acts_ly[idx[half : 2 * half]].mean(axis=0)
                        diffs.append(a - b)
                    C = np.stack(diffs, axis=0)
                    _, _, Vtc = np.linalg.svd(C, full_matrices=False)
                    r_content = min(rank_r, Vtc.shape[0])
                    V_content = Vtc[:r_content].T
                    M = V.T @ V_content
                    _, sig, _ = np.linalg.svd(M, full_matrices=False)
                    sig = np.clip(sig, -1.0, 1.0)
                    median_cos = float(np.median(sig))
                else:
                    median_cos = float("nan")

                predicate_pass = (macro_v >= 0.90) and (macro_c <= 0.20) and (median_cos <= 0.20)
                np.savez(
                    out_path,
                    V_lang=V.astype(np.float32),
                    layer_used=ly,
                    layer_group=lg,
                    heldout_lang_acc=macro_v,
                    heldout_lang_acc_per_lang=json.dumps(per_lang_v),
                    complement_acc=macro_c,
                    complement_acc_per_lang=json.dumps(per_lang_c),
                    principal_angle_median_cos=median_cos,
                    predicate_pass=predicate_pass,
                    n_probe=n_probe,
                    rank_r=rank_r,
                    seed=args.seed,
                )
                n_configs += 1
                if n_configs % 10 == 0:
                    print(f"[m1-grid] {n_configs}/90 configs done for seed={args.seed}, elapsed={time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"[m1-grid] seed={args.seed} done: {n_configs} configs in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
