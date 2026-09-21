"""Variant: model-swap-qwen25-7b-instruct for Claim C3a (near-orthogonality).

This is a minimum-diff adaptation of the main experiment pipeline for Qwen2.5-7B-Instruct.
All methodology, dataset, splits, and code logic frozen from the main experiment.
Only the model path changes.

Runs the full C3a pipeline:
1. Generate turn1 answers (correctness collection)
2. Generate turn2 confidence (verbalized confidence collection)
3. Extract hidden states at all layers for both passes
4. Fit probes, compute per-layer AUROC, select L*, compute cosine + bootstrap CI
5. Write results to result.json

Usage (from project root):
  CUDA_VISIBLE_DEVICES=1,2,3,5,6 python verify/C3a_near_orthogonality/variants/model-swap-qwen25-7b-instruct/run_variant.py
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# ---- path setup ----
WORK_DIR = Path("/data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief")
sys.path.insert(0, str(WORK_DIR / "code"))
sys.path.insert(0, str(WORK_DIR))

import numpy as np
from scipy import stats as sstats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import utils as U

# ---- Variant-specific constants ----
VARIANT_DIR = WORK_DIR / "verify/C3a_near_orthogonality/variants/model-swap-qwen25-7b-instruct"
VARIANT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH_VARIANT = "/data/zhenqian/models/Qwen2.5-7B-Instruct"
MODEL_NAME_VARIANT = "Qwen2.5-7B-Instruct"

# Use same TriviaQA data and split manifest as main experiment
TRIVIAQA_ARROW = U.TRIVIAQA_ARROW
SPLIT_MANIFEST = U.ARTIFACT_DIR / "split_manifest.json"

# ---- Prompt functions for Qwen (plain text, no Llama special tokens) ----
# Qwen2.5-7B-Instruct uses its own chat template; for hidden-state hook
# experiments we use plain text prompts (consistent with main experiment's
# approach — the <|begin_of_text|> token in the main experiment prompts is
# not used for the chat template but as a text prefix; Qwen's tokenizer
# will simply add its own BOS token automatically).

def prompt_turn1_qwen(question: str) -> str:
    """Forward pass 1 (correctness collection) — plain closed-book QA prompt for Qwen."""
    return (
        "Answer the following trivia question with a short factual answer, "
        "just the answer itself (no explanation).\n\n"
        f"Q: {question}\nA:"
    )


def prompt_turn2_qwen(question: str, model_answer: str) -> str:
    """Forward pass 2 (verbalized confidence, P0 primary) for Qwen."""
    return (
        "Answer the following trivia question with a short factual answer, "
        "just the answer itself (no explanation).\n\n"
        f"Q: {question}\nA: {model_answer.strip()}\n\n"
        "How confident are you that the answer above is correct? "
        "Give a probability from 0 to 100 as a single number.\n"
        "Confidence:"
    )


# ---- Probe utility functions (identical to step4_probes.py) ----
def _load_split_masks(qids, split, idxs=None):
    if idxs is not None and "train_idxs" in split:
        train_idxs = set(int(x) for x in split["train_idxs"])
        dev_idxs = set(int(x) for x in split["dev_idxs"])
        test_idxs = set(int(x) for x in split["test_idxs"])
        idxs_list = [int(x) for x in idxs.tolist()]
        tr = np.array([i for i, ix in enumerate(idxs_list) if ix in train_idxs], dtype=np.int64)
        dv = np.array([i for i, ix in enumerate(idxs_list) if ix in dev_idxs], dtype=np.int64)
        te = np.array([i for i, ix in enumerate(idxs_list) if ix in test_idxs], dtype=np.int64)
        return tr, dv, te
    train_ids = set(split["train_ids"])
    dev_ids = set(split["dev_ids"])
    test_ids = set(split["test_ids"])
    tr = np.array([i for i, q in enumerate(qids.tolist()) if q in train_ids], dtype=np.int64)
    dv = np.array([i for i, q in enumerate(qids.tolist()) if q in dev_ids], dtype=np.int64)
    te = np.array([i for i, q in enumerate(qids.tolist()) if q in test_ids], dtype=np.int64)
    return tr, dv, te


def _fit_lr_probe(X_tr, y_tr, X_dv, y_dv, Cs=(0.001, 0.01, 0.1, 1.0, 10.0), seed=42, class_weight=None):
    best = (None, None, -1.0)
    for C in Cs:
        # n_jobs=1 avoids joblib forking workers after GPU operations (torch Bus error in forked workers)
        m = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000,
                               random_state=seed, class_weight=class_weight, n_jobs=1)
        m.fit(X_tr, y_tr)
        if len(np.unique(y_dv)) < 2:
            continue
        p = m.predict_proba(X_dv)[:, 1]
        auc = roc_auc_score(y_dv, p)
        if auc > best[2]:
            best = (C, m, auc)
    if best[1] is None:
        best = (1.0, LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                        max_iter=500, random_state=seed, n_jobs=-1).fit(X_tr, y_tr), 0.5)
    return best


def _ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _cos(u, v):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _pct(a, plo=2.5, phi=97.5):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    return {"mean": float(np.mean(a)), "ci_lo": float(np.percentile(a, plo)),
            "ci_hi": float(np.percentile(a, phi)), "n": int(a.size)}


# ---- Step 1: Generate answers (vllm) ----
def step1_generate_turn1(rows, out_path, model_path, gpu_mem_util=0.55, max_model_len=1024):
    """Run vllm forward pass 1 to get answer generations."""
    from vllm import LLM, SamplingParams
    print(f"[variant] Loading vllm model {model_path}", flush=True)
    llm = LLM(model=model_path, dtype="float16", gpu_memory_utilization=gpu_mem_util,
              max_model_len=max_model_len, swap_space=4)
    sp = SamplingParams(temperature=0.0, max_tokens=20, stop=["\n", "Q:", "A:"])
    prompts = [prompt_turn1_qwen(r["question"]) for r in rows]
    print(f"[variant] Generating turn1 for {len(prompts)} prompts...", flush=True)
    outputs = llm.generate(prompts, sp)
    results = []
    for r, o in zip(rows, outputs):
        gen = o.outputs[0].text
        y_correct = U.score_answer(gen, r["normalized_aliases"])
        results.append({
            "idx": r["idx"],
            "question_id": r["question_id"],
            "question": r["question"],
            "normalized_aliases": r["normalized_aliases"],
            "generation": gen,
            "short_answer": U.extract_short_answer(gen),
            "y_correct": y_correct,
        })
    U.dump_jsonl(out_path, results)
    del llm
    gc.collect()
    print(f"[variant] Turn1 written to {out_path} ({len(results)} rows)", flush=True)
    return results


def step1_generate_turn2(rows, turn1_by_idx, out_path, model_path, gpu_mem_util=0.55, max_model_len=1024):
    """Run vllm forward pass 2 to get verbalized confidence."""
    from vllm import LLM, SamplingParams
    print(f"[variant] Loading vllm model {model_path} for turn2", flush=True)
    llm = LLM(model=model_path, dtype="float16", gpu_memory_utilization=gpu_mem_util,
              max_model_len=max_model_len, swap_space=4)
    sp = SamplingParams(temperature=0.0, max_tokens=8, stop=["\n"])
    prompts = []
    for r in rows:
        t1 = turn1_by_idx.get(r["idx"])
        ans = t1["short_answer"] if t1 else ""
        prompts.append(prompt_turn2_qwen(r["question"], ans))
    print(f"[variant] Generating turn2 for {len(prompts)} prompts...", flush=True)
    outputs = llm.generate(prompts, sp)
    results = []
    for r, o in zip(rows, outputs):
        gen = o.outputs[0].text
        c_val, c_parseable = U.parse_confidence(gen, "0-100")
        results.append({
            "idx": r["idx"],
            "question_id": r["question_id"],
            "question": r["question"],
            "generation": gen,
            "c": c_val,
            "c_parseable": c_parseable,
        })
    U.dump_jsonl(out_path, results)
    del llm
    gc.collect()
    parseable = sum(1 for r in results if r["c_parseable"])
    print(f"[variant] Turn2 written to {out_path} ({parseable}/{len(results)} parseable)", flush=True)
    return results


# ---- Step 2: Extract hidden states ----
def step2_extract_hidden(rows, turn_jsonl, out_path, model_path, prompt_fn, n_layers_expected=None):
    """Extract hidden states at all layers for last-input-token position."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[variant] Extracting hidden states from {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        output_hidden_states=True, attn_implementation="sdpa"
    )
    model.eval()

    # Check number of layers
    n_layers = model.config.num_hidden_layers
    print(f"[variant] Model has {n_layers} transformer blocks (+ embedding = {n_layers+1} states)", flush=True)

    turn_rows = U.load_jsonl(turn_jsonl)
    by_idx = {r["idx"]: r for r in turn_rows}

    batch_size = 32
    all_H = []
    all_qids = []
    all_idxs = []
    t0 = time.time()

    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        prompts = []
        for r in batch:
            if prompt_fn == "turn1":
                prompts.append(prompt_turn1_qwen(r["question"]))
            else:  # turn2
                t1 = by_idx.get(r["idx"])
                ans = t1.get("short_answer", "") if t1 else ""
                prompts.append(prompt_turn2_qwen(r["question"], ans))

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) tensors, each (B, S, D)
        hs = out.hidden_states  # tuple length n_layers+1

        attn = enc["attention_mask"]  # (B, S)
        # Last-input-token position: last position where attn_mask = 1
        # For left-padded inputs (tok.padding_side = "left"): last real token IS at S-1
        # Invariant: attn[:, -1] == 1 for all rows (left-padding means real tokens on the right)
        import torch
        assert (attn[:, -1] == 1).all(), (
            f"Left-padding invariant violated: attn[:,-1]={attn[:,-1].tolist()} "
            "(expected all 1s; ensure tok.padding_side='left')"
        )
        last_pos = attn.shape[1] - 1  # scalar (same for all in batch since left-padded)

        # Stack: (n_layers+1, B, D)
        H_batch = torch.stack([h[:, last_pos, :].float().cpu() for h in hs], dim=0)
        # Reshape to (B, n_layers+1, D)
        H_batch = H_batch.permute(1, 0, 2).numpy().astype(np.float16)

        for j, r in enumerate(batch):
            all_H.append(H_batch[j])  # (n_layers+1, D)
            all_qids.append(r["question_id"])
            all_idxs.append(r["idx"])

        if (i // batch_size) % 10 == 0:
            print(f"[variant]   extracted {i + len(batch)}/{len(rows)} batches  {(time.time()-t0):.0f}s", flush=True)

    H_arr = np.stack(all_H, axis=0)  # (N, n_layers+1, D)
    qids_arr = np.array(all_qids)
    idxs_arr = np.array(all_idxs)
    print(f"[variant] H shape: {H_arr.shape}", flush=True)

    np.savez_compressed(str(out_path),
                        H=H_arr, question_ids=qids_arr, idxs=idxs_arr)
    print(f"[variant] Hidden states saved to {out_path}", flush=True)
    del model
    gc.collect()
    return H_arr, qids_arr, idxs_arr


# ---- Step 4: Probe fits + cosine ----
def step4_probes(H1_path, H2_path, turn1_jsonl, turn2_jsonl, split_manifest_path,
                 n_bootstrap=200, seed=42, out_dir=None):
    """Identical to main experiment step4_probes.py but reading variant hidden states."""
    if out_dir is None:
        out_dir = VARIANT_DIR
    out_dir = Path(out_dir)

    # Load hidden states
    npz1 = np.load(H1_path)
    H1 = npz1["H"]
    qids1 = npz1["question_ids"]
    idxs1 = npz1["idxs"]
    npz2 = np.load(H2_path)
    H2 = npz2["H"]
    N, Lp1, D = H1.shape
    num_layers = Lp1 - 1
    print(f"[variant:step4] H1 {H1.shape}  H2 {H2.shape}", flush=True)

    # Load labels
    turn1_rows = U.load_jsonl(turn1_jsonl)
    turn2_rows = U.load_jsonl(turn2_jsonl)
    y_correct_by_idx = {r["idx"]: int(r["y_correct"]) for r in turn1_rows}
    c_by_idx = {r["idx"]: (float(r["c"]) if r["c_parseable"] else None) for r in turn2_rows}

    split = U.load_json(split_manifest_path)
    tr_idx, dv_idx, te_idx = _load_split_masks(qids1, split, idxs=idxs1)
    print(f"[variant:step4] train={len(tr_idx)} dev={len(dv_idx)} test={len(te_idx)}", flush=True)

    # Labels (idx-based)
    idxs_list = [int(x) for x in idxs1.tolist()]
    y_c = np.array([y_correct_by_idx.get(ix, 0) for ix in idxs_list], dtype=np.int64)
    c_raw = np.array([c_by_idx.get(ix, np.nan) if c_by_idx.get(ix, np.nan) is not None else np.nan
                      for ix in idxs_list], dtype=np.float64)
    c_parseable = ~np.isnan(c_raw)

    # Use same Stage-1.5 binarize threshold as main experiment
    stage15 = U.load_json(U.ARTIFACT_DIR / "stage15_variance.json")
    c_thresh = float(stage15["binarize_threshold"])
    path = stage15["path"]
    print(f"[variant:step4] binarize_threshold={c_thresh} path={path}", flush=True)

    def to_ordinal(c):
        return np.digitize(c, [25, 50, 75])

    # Layer sweep
    layers = list(range(num_layers + 1))
    metrics_per_layer = {}
    v_c_by_layer = np.zeros((num_layers + 1, D), dtype=np.float32)
    v_v_bin_by_layer = np.zeros((num_layers + 1, D), dtype=np.float32)

    t0 = time.time()
    for L in layers:
        X1 = H1[:, L, :].astype(np.float32)
        X2 = H2[:, L, :].astype(np.float32)

        # probe_c_binary
        y = y_c
        Ctr, mdl_c, auc_c_dev = _fit_lr_probe(X1[tr_idx], y[tr_idx], X1[dv_idx], y[dv_idx],
                                               seed=seed, class_weight="balanced")
        pr_c_test = mdl_c.predict_proba(X1[te_idx])[:, 1]
        auc_c_test = float(roc_auc_score(y[te_idx], pr_c_test)) if len(np.unique(y[te_idx])) == 2 else None
        v_c = _unit(mdl_c.coef_.ravel())
        v_c_by_layer[L] = v_c.astype(np.float32)
        auc_c_dev = float(auc_c_dev)

        # probe_v_binary
        y_v_bin = (c_raw >= c_thresh).astype(np.int64)
        tr_mask = np.intersect1d(tr_idx, np.where(c_parseable)[0])
        dv_mask = np.intersect1d(dv_idx, np.where(c_parseable)[0])
        te_mask = np.intersect1d(te_idx, np.where(c_parseable)[0])
        auc_v_bin_test = None
        auc_v_bin_dev = 0.5
        v_v_bin = np.zeros(D, dtype=np.float64)
        if len(tr_mask) >= 50 and len(np.unique(y_v_bin[tr_mask])) == 2:
            Cv, mdl_v_bin, auc_v_bin_dev = _fit_lr_probe(
                X2[tr_mask], y_v_bin[tr_mask], X2[dv_mask], y_v_bin[dv_mask],
                seed=seed, class_weight="balanced")
            pr_v_bin_test = mdl_v_bin.predict_proba(X2[te_mask])[:, 1]
            if len(np.unique(y_v_bin[te_mask])) == 2:
                auc_v_bin_test = float(roc_auc_score(y_v_bin[te_mask], pr_v_bin_test))
            v_v_bin = _unit(mdl_v_bin.coef_.ravel())
        v_v_bin_by_layer[L] = v_v_bin.astype(np.float32)
        auc_v_bin_dev = float(auc_v_bin_dev)

        # probe_v_primary (ordinal, following main experiment's path)
        primary = {"path": path}
        if path == "ordinal" and len(tr_mask) >= 50:
            from sklearn.metrics import f1_score, accuracy_score
            y_ord_tr = to_ordinal(c_raw[tr_mask])
            y_ord_te = to_ordinal(c_raw[te_mask])
            if len(np.unique(y_ord_tr)) >= 2:
                ordm = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
                                         multi_class="multinomial", random_state=seed, class_weight="balanced", n_jobs=1)
                ordm.fit(X2[tr_mask], y_ord_tr)
                y_pred_te = ordm.predict(X2[te_mask])
                primary["ordinal_top1_test"] = float(accuracy_score(y_ord_te, y_pred_te))
                primary["ordinal_macro_f1_test"] = float(f1_score(y_ord_te, y_pred_te, average="macro", zero_division=0))

        metrics_per_layer[L] = {
            "probe_c_binary": {"C": Ctr, "auc_dev": auc_c_dev, "auc_test": auc_c_test},
            "probe_v_binary": {"auc_dev": auc_v_bin_dev, "auc_test": auc_v_bin_test},
            "probe_v_primary": primary,
        }
        if L % 4 == 0:
            print(f"[variant:step4] layer {L}/{num_layers}  auc_c={auc_c_test}  auc_v_bin={auc_v_bin_test}", flush=True)

    print(f"[variant:step4] Layer sweep done in {(time.time()-t0)/60:.1f} min", flush=True)

    # L* selection (DEV AUC, exclude embedding layer L=0)
    def _num_or_default(x, default=0.5):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)

    aucs_c_dev = np.array([_num_or_default(metrics_per_layer[L]["probe_c_binary"]["auc_dev"]) for L in layers])
    aucs_v_dev = np.array([_num_or_default(metrics_per_layer[L]["probe_v_binary"]["auc_dev"]) for L in layers])
    mask_search = np.array([L >= 1 for L in layers])
    denom_c = max(aucs_c_dev[mask_search].max(), 1e-6)
    denom_v = max(aucs_v_dev[mask_search].max(), 1e-6)
    scores = aucs_c_dev / denom_c + aucs_v_dev / denom_v
    scores_masked = np.where(mask_search, scores, -np.inf)
    L_star = int(layers[int(np.argmax(scores_masked))])
    print(f"[variant:step4] L* = {L_star}", flush=True)

    # Cosine per layer
    cos_by_layer = []
    for L in layers:
        cos_by_layer.append({
            "layer": L,
            "abs_cos": abs(_cos(v_c_by_layer[L], v_v_bin_by_layer[L])),
        })

    # Neighborhood L*+-2
    lo = max(0, L_star - 2)
    hi = min(num_layers, L_star + 2)
    nbhd_abs = float(np.mean([cos_by_layer[L]["abs_cos"] for L in range(lo, hi + 1)]))
    print(f"[variant:step4] |cos| at L*={cos_by_layer[L_star]['abs_cos']:.3f}, neighborhood mean={nbhd_abs:.3f}", flush=True)

    # Bootstrap CI at L* (retrain-on-bootstrap, 200 resamples)
    L = L_star
    X1L = H1[:, L, :].astype(np.float32)
    X2L = H2[:, L, :].astype(np.float32)
    Ctr_star = metrics_per_layer[L]["probe_c_binary"]["C"]
    rng = np.random.default_rng(seed)
    boot_abs_cos = []
    boot_auc_c = []
    boot_auc_v = []
    tr_pool = tr_idx
    tr_par_pool = np.intersect1d(tr_idx, np.where(c_parseable)[0])
    te_pool = te_idx
    te_par_pool = np.intersect1d(te_idx, np.where(c_parseable)[0])

    t_boot0 = time.time()
    for b in range(n_bootstrap):
        idxs_c = rng.choice(tr_pool, size=len(tr_pool), replace=True)
        idxs_v = rng.choice(tr_par_pool, size=len(tr_par_pool), replace=True) if len(tr_par_pool) else np.array([], dtype=np.int64)
        mdl_c = LogisticRegression(C=Ctr_star, penalty="l2", solver="lbfgs", max_iter=500,
                                   random_state=seed, class_weight="balanced", n_jobs=1)
        mdl_c.fit(X1L[idxs_c], y_c[idxs_c])
        if len(np.unique(y_c[te_pool])) == 2:
            p_c = mdl_c.predict_proba(X1L[te_pool])[:, 1]
            boot_auc_c.append(roc_auc_score(y_c[te_pool], p_c))
        vc_b = _unit(mdl_c.coef_.ravel())

        vv_b = np.zeros(D, dtype=np.float64)
        if len(idxs_v):
            y_bin = (c_raw[idxs_v] >= c_thresh).astype(np.int64)
            if len(np.unique(y_bin)) == 2:
                mdl_v = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=500,
                                           random_state=seed, class_weight="balanced", n_jobs=1)
                mdl_v.fit(X2L[idxs_v], y_bin)
                if len(te_par_pool):
                    y_bin_te = (c_raw[te_par_pool] >= c_thresh).astype(np.int64)
                    if len(np.unique(y_bin_te)) == 2:
                        p_v = mdl_v.predict_proba(X2L[te_par_pool])[:, 1]
                        boot_auc_v.append(float(roc_auc_score(y_bin_te, p_v)))
                vv_b = _unit(mdl_v.coef_.ravel())
        boot_abs_cos.append(abs(_cos(vc_b, vv_b)))
        if (b + 1) % 50 == 0:
            print(f"[variant:step4]   bootstrap {b+1}/{n_bootstrap}  {(time.time()-t_boot0):.1f}s", flush=True)

    ci_cos = _pct(boot_abs_cos)
    ci_auc_c = _pct(boot_auc_c)
    ci_auc_v = _pct(boot_auc_v)

    print(f"[variant:step4] |cos| bootstrap CI: {ci_cos}", flush=True)

    # Random-direction null
    rng2 = np.random.default_rng(seed + 1)
    v_c_star = _unit(v_c_by_layer[L_star])
    v_v_star = _unit(v_v_bin_by_layer[L_star])
    rand_cosines = []
    for _ in range(100):
        r = rng2.normal(size=D)
        r = _unit(r)
        rand_cosines.append(abs(_cos(v_c_star, r)))
    rand_mean = float(np.mean(rand_cosines))
    print(f"[variant:step4] Random-direction null mean |cos| = {rand_mean:.4f} (theory ~{1/np.sqrt(D):.4f})", flush=True)

    # Compile result
    result = {
        "variant": "model-swap-qwen25-7b-instruct",
        "model": MODEL_NAME_VARIANT,
        "dataset": "TriviaQA",
        "claim": "C3a",
        "n_total": N,
        "n_train": len(tr_idx),
        "n_dev": len(dv_idx),
        "n_test": len(te_idx),
        "D": D,
        "num_layers": num_layers,
        "L_star": L_star,
        "abs_cos_at_Lstar": float(cos_by_layer[L_star]["abs_cos"]),
        "neighborhood_mean_L_star_pm2": nbhd_abs,
        "abs_cos_bootstrap_ci": ci_cos,
        "auc_c_test": metrics_per_layer[L_star]["probe_c_binary"]["auc_test"],
        "auc_v_bin_test": metrics_per_layer[L_star]["probe_v_binary"]["auc_test"],
        "auc_c_bootstrap_ci": ci_auc_c,
        "auc_v_bootstrap_ci": ci_auc_v,
        "random_direction_null_mean_abs_cos": rand_mean,
        "theoretical_random_null": float(1.0 / np.sqrt(D)),
        "per_layer_cos": cos_by_layer,
        "success_criteria": {
            "abs_cos_threshold": 0.3,
            "ci_upper_threshold": 0.4,
            "neighborhood_threshold": 0.3,
            "probe_auroc_threshold": 0.70,
        },
        "passes": {
            "abs_cos": float(cos_by_layer[L_star]["abs_cos"]) <= 0.3 if cos_by_layer[L_star]["abs_cos"] is not None else False,
            "ci_upper": (ci_cos["ci_hi"] <= 0.4) if ci_cos["ci_hi"] is not None else False,
            "neighborhood": nbhd_abs <= 0.3,
            "probe_c_auroc": (metrics_per_layer[L_star]["probe_c_binary"]["auc_test"] or 0) >= 0.70,
            "probe_v_auroc": (metrics_per_layer[L_star]["probe_v_binary"]["auc_test"] or 0) >= 0.70,
        },
    }
    result["all_criteria_pass"] = all(result["passes"].values())

    U.dump_json(out_dir / "result.json", result)
    print(f"[variant] result.json written to {out_dir / 'result.json'}", flush=True)
    print(f"[variant] C3a PASS: {result['all_criteria_pass']}", flush=True)
    print(f"[variant] |cos| = {result['abs_cos_at_Lstar']:.4f}  CI [{ci_cos['ci_lo']:.4f}, {ci_cos['ci_hi']:.4f}]", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--n-total", type=int, default=10000)
    args = ap.parse_args()

    U.set_all_seeds(args.seed)

    out_dir = VARIANT_DIR
    turn1_jsonl = out_dir / "turn1_gen.jsonl"
    turn2_jsonl = out_dir / "turn2_gen.jsonl"
    H1_path = out_dir / "H_turn1.npz"
    H2_path = out_dir / "H_turn2.npz"

    # Load TriviaQA subset (same as main experiment)
    print("[variant] Loading TriviaQA...", flush=True)
    rows = U.load_triviaqa_subset(max_gold_tokens=5, seed=args.seed)
    rows = rows[:args.n_total]
    print(f"[variant] {len(rows)} questions loaded", flush=True)

    # Step 1a: Generate turn1 (if not already done)
    if not turn1_jsonl.exists():
        turn1_rows = step1_generate_turn1(rows, turn1_jsonl, MODEL_PATH_VARIANT,
                                          gpu_mem_util=args.gpu_mem_util, max_model_len=args.max_model_len)
    else:
        turn1_rows = U.load_jsonl(turn1_jsonl)
        print(f"[variant] Reusing existing {turn1_jsonl}", flush=True)

    turn1_by_idx = {r["idx"]: r for r in turn1_rows}

    # Step 1b: Generate turn2 (if not already done)
    if not turn2_jsonl.exists():
        turn2_rows = step1_generate_turn2(rows, turn1_by_idx, turn2_jsonl, MODEL_PATH_VARIANT,
                                          gpu_mem_util=args.gpu_mem_util, max_model_len=args.max_model_len)
    else:
        turn2_rows = U.load_jsonl(turn2_jsonl)
        print(f"[variant] Reusing existing {turn2_jsonl}", flush=True)

    # Step 2a: Extract hidden states for turn1
    if not H1_path.exists():
        step2_extract_hidden(rows, turn1_jsonl, H1_path, MODEL_PATH_VARIANT, prompt_fn="turn1")
    else:
        print(f"[variant] Reusing existing {H1_path}", flush=True)

    # Step 2b: Extract hidden states for turn2
    if not H2_path.exists():
        step2_extract_hidden(rows, turn2_jsonl, H2_path, MODEL_PATH_VARIANT, prompt_fn="turn2")
    else:
        print(f"[variant] Reusing existing {H2_path}", flush=True)

    # Step 4: Probe fits + cosine
    result = step4_probes(
        H1_path=str(H1_path),
        H2_path=str(H2_path),
        turn1_jsonl=str(turn1_jsonl),
        turn2_jsonl=str(turn2_jsonl),
        split_manifest_path=str(SPLIT_MANIFEST),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        out_dir=out_dir,
    )

    print("\n[variant] === SUMMARY ===", flush=True)
    print(f"Model: {MODEL_NAME_VARIANT}", flush=True)
    print(f"L* = {result['L_star']}", flush=True)
    print(f"|cos(v_c*, v_v*)| = {result['abs_cos_at_Lstar']:.4f}  CI [{result['abs_cos_bootstrap_ci']['ci_lo']:.4f}, {result['abs_cos_bootstrap_ci']['ci_hi']:.4f}]", flush=True)
    print(f"Neighborhood mean L*+-2 = {result['neighborhood_mean_L_star_pm2']:.4f}", flush=True)
    print(f"AUROC(probe_c) = {result['auc_c_test']}  AUROC(probe_v) = {result['auc_v_bin_test']}", flush=True)
    print(f"All criteria pass: {result['all_criteria_pass']}", flush=True)


if __name__ == "__main__":
    main()
