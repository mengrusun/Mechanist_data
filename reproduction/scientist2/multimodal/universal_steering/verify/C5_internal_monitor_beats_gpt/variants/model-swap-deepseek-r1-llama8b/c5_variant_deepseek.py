"""C5 verify variant — model-swap: DeepSeek-R1-Distill-Llama-8B.

Single change vs main experiment (code/c5_monitoring.py):
  - model loaded from /data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B

Everything else is frozen:
  - Same benchmark data (HaluEval-General 2000 + ToxicChat 584 balanced)
  - Reuse exact same test splits from C5_monitoring (same rows, same labels)
  - Same probe/RFM fitting procedure (60% train, 20% val, 20% test; seed=42)
  - Reuse GPT-4o judge scores from C5_baselines (judge is model-agnostic — same text)
  - Same AUROC computation
  - Same success predicate: internal AUROC > GPT-4o AUROC on both benchmarks
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORK_DIR = Path("/data/zhenqian/Reproduction1/mechanica/multimodal/universal_steering")
MONITOR_DIR = Path("/data/zhenqian/data/monitoring")

sys.path.insert(0, str(WORK_DIR / "code"))

from rfm_core import fit_linear_probe, probe_auroc, extract_rfm, caa_mean_diff
from model_utils import free_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- SINGLE CHANGE: model path ----
MODEL_PATH = "/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B"


def load_model_variant(dtype="bfloat16", device="cuda"):
    """Load DeepSeek-R1-Distill-Llama-8B — drop-in replacement for Llama-3.1-8B-Instruct.
    Same architecture: Llama, 32 blocks, d_model=4096.
    """
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer.padding_side = "left"
    num_blocks = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"[variant] Loaded {MODEL_PATH}: num_blocks={num_blocks}, d_model={d_model}")
    assert num_blocks == 32 and d_model == 4096, \
        f"Unexpected architecture: blocks={num_blocks}, d_model={d_model}"
    return model, tokenizer, {"num_blocks": num_blocks, "d_model": d_model}


@torch.no_grad()
def cache_last_token_activations(model, tokenizer, prompts, batch_size=8, max_length=512, device="cuda"):
    """Identical to model_utils.py — uses model.model.layers (Llama architecture)."""
    blocks = model.model.layers
    num_blocks = len(blocks)
    d_model = model.config.hidden_size
    n = len(prompts)
    out = np.zeros((n, num_blocks, d_model), dtype=np.float16)
    cache = [None] * num_blocks

    def make_hook(idx):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            cache[idx] = hs.detach()
            return output
        return hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(blocks)]
    try:
        for b0 in range(0, n, batch_size):
            batch = list(prompts[b0:b0 + batch_size])
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_length).to(device)
            _ = model(**enc, use_cache=False)
            for i in range(len(batch)):
                for l in range(num_blocks):
                    hs = cache[l]
                    out[b0 + i, l] = hs[i, -1, :].to(torch.float16).cpu().numpy()
    finally:
        for h in handles:
            h.remove()
    return out


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def format_pair(r):
    return f"USER: {r.get('prompt','')}\nASSISTANT: {r.get('response','')}"


def build_subset(rows, n_pos, n_neg, seed=42):
    rng = np.random.default_rng(seed)
    pos = [r for r in rows if r.get("label") == 1]
    neg = [r for r in rows if r.get("label") == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    return pos[:n_pos] + neg[:n_neg]


def split_data(rows, ratios=(0.6, 0.2, 0.2), seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows)); rng.shuffle(idx)
    n_tr = int(len(rows) * ratios[0])
    n_va = int(len(rows) * ratios[1])
    tr = [rows[i] for i in idx[:n_tr]]
    va = [rows[i] for i in idx[n_tr:n_tr+n_va]]
    te = [rows[i] for i in idx[n_tr+n_va:]]
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="verify_C5_deepseek_r1_8b")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--halueval-n", type=int, default=2000)
    ap.add_argument("--toxicchat-n", type=int, default=584)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--rfm-iters", type=int, default=3)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    # IMPORTANT: reuse test splits from main experiment for fair comparison
    ap.add_argument("--reuse-splits-from",
                    default=str(WORK_DIR / "runs" / "C5_monitoring"),
                    help="Directory containing halueval_{train,val,test}.jsonl + toxicchat_{train,val,test}.jsonl")
    # Reuse GPT-4o scores (judge is model-agnostic — same (prompt,response) text)
    ap.add_argument("--reuse-gpt4o-from",
                    default=str(WORK_DIR / "runs" / "C5_baselines"),
                    help="Directory containing halueval_gpt4o_scores.npy + toxicchat_gpt4o_scores.npy")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        (WORK_DIR / "verify" / "C5_internal_monitor_beats_gpt" / "variants" / "model-swap-deepseek-r1-llama8b" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[variant] out_dir={out_dir}")

    reuse_dir = Path(args.reuse_splits_from)
    gpt4o_dir = Path(args.reuse_gpt4o_from)

    print("[variant] loading model...")
    model, tokenizer, cfg = load_model_variant(dtype=args.dtype, device=args.device)
    num_blocks = cfg["num_blocks"]

    summary = {"config": vars(args), "model": MODEL_PATH, "benchmarks": {}, "start_time": time.time()}

    benchmarks = [
        ("halueval", args.halueval_n // 2, args.halueval_n // 2, "hallucination"),
        ("toxicchat", args.toxicchat_n // 2, args.toxicchat_n // 2, "toxicity"),
    ]

    for bench_name, n_pos_req, n_neg_req, judge_kind in benchmarks:
        print(f"\n[variant] === {bench_name} ===")

        # Load same test split as main experiment
        te_path = reuse_dir / f"{bench_name}_test.jsonl"
        if te_path.exists():
            te = read_jsonl(te_path)
            print(f"[variant]   reusing test split from {te_path}: n_test={len(te)}")
        else:
            raise FileNotFoundError(f"Test split not found: {te_path} — run C5_monitoring first")

        # For train/val: either reuse or re-split
        tr_path = reuse_dir / f"{bench_name}_train.jsonl"
        va_path = reuse_dir / f"{bench_name}_val.jsonl"
        if tr_path.exists() and va_path.exists():
            tr = read_jsonl(tr_path)
            va = read_jsonl(va_path)
            print(f"[variant]   reusing train/val splits: n_train={len(tr)}, n_val={len(va)}")
        else:
            # Rebuild from scratch with same seed
            data_path = MONITOR_DIR / bench_name.replace("halueval", "hegen") / "data.jsonl"
            if bench_name == "halueval":
                data_path = MONITOR_DIR / "hegen" / "data.jsonl"
            elif bench_name == "toxicchat":
                data_path = MONITOR_DIR / "toxicchat" / "data.jsonl"
            rows = read_jsonl(data_path)
            avail_pos = sum(1 for r in rows if r.get("label") == 1)
            avail_neg = sum(1 for r in rows if r.get("label") == 0)
            n_each = min(n_pos_req, n_neg_req, avail_pos, avail_neg)
            subset = build_subset(rows, n_each, n_each, seed=args.seed)
            # Remove test rows (identified by matching test split rows)
            te_set = {(r.get("prompt",""), r.get("response","")) for r in te}
            subset_no_test = [r for r in subset if (r.get("prompt",""), r.get("response","")) not in te_set]
            n_tr = int(len(subset_no_test) * 0.75)
            tr = subset_no_test[:n_tr]
            va = subset_no_test[n_tr:]
            print(f"[variant]   rebuilt train/val: n_train={len(tr)}, n_val={len(va)}")

        # Cache activations for train + val + test using the VARIANT model
        all_rows = tr + va + te
        texts = [format_pair(r) for r in all_rows]
        labels_int = np.array([int(r.get("label", 0)) for r in all_rows], dtype=np.int64)
        y = np.where(labels_int > 0, 1.0, -1.0)

        cache_path = out_dir / f"{bench_name}_activations.npy"
        if cache_path.exists():
            print(f"[variant]   loading cached activations from {cache_path}")
            acts = np.load(cache_path, mmap_mode="r")
        else:
            print(f"[variant]   caching activations for {len(texts)} rows (DeepSeek-R1-Distill-Llama-8B)...")
            t0 = time.time()
            acts = cache_last_token_activations(
                model, tokenizer, texts, batch_size=args.batch_size,
                max_length=args.max_length, device=args.device,
            )
            print(f"[variant]   caching took {time.time()-t0:.1f}s, shape={acts.shape}")
            np.save(cache_path, acts)

        n_tr = len(tr); n_va = len(va); n_te = len(te)
        tr_acts = acts[:n_tr]; va_acts = acts[n_tr:n_tr+n_va]; te_acts = acts[n_tr+n_va:]
        y_tr = y[:n_tr]; y_va = y[n_tr:n_tr+n_va]; y_te = y[n_tr+n_va:]

        # Per-block probe + RFM (identical procedure to main experiment)
        probe_val_aurocs = []; probe_test_aurocs = []
        rfm_val_aurocs = []; rfm_test_aurocs = []

        for l in range(num_blocks):
            X_tr = tr_acts[:, l, :].astype(np.float64)
            X_va = va_acts[:, l, :].astype(np.float64)
            X_te = te_acts[:, l, :].astype(np.float64)

            w_vec, w_b = fit_linear_probe(X_tr, y_tr, ridge=1e-2)
            probe_val_aurocs.append(probe_auroc(w_vec, X_va, y_va))
            probe_test_aurocs.append(probe_auroc(w_vec, X_te, y_te))

            try:
                rfm = extract_rfm(X_tr, y_tr, n_iters=args.rfm_iters, seed=args.seed, verbose=False)
                v = rfm["v_c"].astype(np.float32)
            except Exception as e:
                print(f"[variant]   block {l} RFM failed: {e} — using CAA")
                v = caa_mean_diff(X_tr, y_tr).astype(np.float32)
            rfm_val_aurocs.append(probe_auroc(v.astype(np.float64), X_va, y_va))
            rfm_test_aurocs.append(probe_auroc(v.astype(np.float64), X_te, y_te))

            if l % 4 == 0 or l == num_blocks - 1:
                print(f"[variant]   block {l:02d}: probe_val={probe_val_aurocs[-1]:.3f} "
                      f"rfm_val={rfm_val_aurocs[-1]:.3f}")

        best_probe_block = int(np.argmax(probe_val_aurocs))
        best_rfm_block = int(np.argmax(rfm_val_aurocs))
        print(f"[variant]   best probe: block={best_probe_block} val={probe_val_aurocs[best_probe_block]:.4f} "
              f"test={probe_test_aurocs[best_probe_block]:.4f}")
        print(f"[variant]   best RFM:   block={best_rfm_block} val={rfm_val_aurocs[best_rfm_block]:.4f} "
              f"test={rfm_test_aurocs[best_rfm_block]:.4f}")

        # Save per-block AUROCs
        np.save(out_dir / f"{bench_name}_probe_val_aurocs.npy", np.array(probe_val_aurocs))
        np.save(out_dir / f"{bench_name}_probe_test_aurocs.npy", np.array(probe_test_aurocs))
        np.save(out_dir / f"{bench_name}_rfm_val_aurocs.npy", np.array(rfm_val_aurocs))
        np.save(out_dir / f"{bench_name}_rfm_test_aurocs.npy", np.array(rfm_test_aurocs))

        # Reuse GPT-4o judge scores from main experiment (same (prompt,response) text)
        gpt_scores_path = gpt4o_dir / f"{bench_name}_gpt4o_scores.npy"
        if gpt_scores_path.exists():
            gpt_scores = np.load(gpt_scores_path)
            y_te_int = labels_int[n_tr+n_va:]
            y_te_pm = np.where(y_te_int > 0, 1.0, -1.0)
            gpt_auroc = float(probe_auroc(np.array([1.0]), gpt_scores.reshape(-1, 1), y_te_pm))
            print(f"[variant]   GPT-4o AUROC (reused from {gpt_scores_path}): {gpt_auroc:.4f}")
        else:
            print(f"[variant]   WARNING: GPT-4o scores not found at {gpt_scores_path}; set gpt4o_auroc=None")
            gpt_auroc = None

        best_probe_test = float(probe_test_aurocs[best_probe_block])
        best_rfm_test = float(rfm_test_aurocs[best_rfm_block])
        best_internal_test = max(best_probe_test, best_rfm_test)
        internal_beats_gpt = (gpt_auroc is not None) and (best_internal_test > gpt_auroc)

        summary["benchmarks"][bench_name] = {
            "n_train": n_tr, "n_val": n_va, "n_test": n_te,
            "best_probe_block": best_probe_block,
            "best_probe_val_auroc": float(probe_val_aurocs[best_probe_block]),
            "best_probe_test_auroc": best_probe_test,
            "best_rfm_block": best_rfm_block,
            "best_rfm_val_auroc": float(rfm_val_aurocs[best_rfm_block]),
            "best_rfm_test_auroc": best_rfm_test,
            "best_internal_test_auroc": best_internal_test,
            "gpt4o_test_auroc": gpt_auroc,
            "gpt4o_scores_reused_from": str(gpt_scores_path) if gpt_scores_path.exists() else None,
            "internal_beats_gpt": internal_beats_gpt,
            "per_block_probe_val": [float(x) for x in probe_val_aurocs],
            "per_block_probe_test": [float(x) for x in probe_test_aurocs],
            "per_block_rfm_val": [float(x) for x in rfm_val_aurocs],
            "per_block_rfm_test": [float(x) for x in rfm_test_aurocs],
        }
        del acts; free_cuda()

    # Overall variant verdict
    all_internal_beat = all(
        summary["benchmarks"][b].get("internal_beats_gpt", False)
        for b in summary["benchmarks"]
    )
    summary["variant_claim_supported"] = all_internal_beat
    summary["end_time"] = time.time()
    summary["total_time_s"] = summary["end_time"] - summary["start_time"]

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[variant] === RESULT ===")
    for b, v in summary["benchmarks"].items():
        print(f"  {b}: internal={v['best_internal_test_auroc']:.4f} vs GPT-4o={v['gpt4o_test_auroc']:.4f} "
              f"→ internal_beats_gpt={v['internal_beats_gpt']}")
    print(f"  variant_claim_supported={all_internal_beat}")
    print(f"[variant] DONE → {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
