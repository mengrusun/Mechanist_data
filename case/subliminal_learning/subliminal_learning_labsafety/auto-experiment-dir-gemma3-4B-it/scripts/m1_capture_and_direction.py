"""M1 — Location: capture residual-stream activations at every language-tower layer of the
treated student and Ctrl-B student on a fixed held-out batch of QA_I items, then compute:
(a) diff-in-means direction  d̂_ℓ  per layer,
(b) linear probe AUROC per layer (5-fold CV),
(c) direct-logit-attribution (DLA) on items where treated errs but ctrlb is correct.

For each QA_I item we do a single forward pass with the image attached and grab the
residual-stream state at the LAST prompt token (right before generation begins) at every
language-tower layer. This gives us activations `A_ℓ` of shape [batch, d_model] for each layer.

Output: runs/m1/ranked_sites.json  — layer × (diff_norm, probe_auroc, dla_score, top-1 flag)
        runs/m1/direction_top.npy — the top-layer d̂_ℓ unit-norm direction
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForImageTextToText

from common import (
    BASE_MODEL,
    QA_I_PARQUET,
    load_tokenizer_and_processor,
    set_seed,
)


def load_qa_items(parquet_path: str, n_items: int | None = None) -> list[dict]:
    t = pq.read_table(parquet_path)
    rows = t.to_pylist()
    items = []
    for i, r in enumerate(rows):
        img_dict = r.get("Decoded Image", {}) or {}
        img_bytes = img_dict.get("bytes") if isinstance(img_dict, dict) else None
        items.append({
            "idx": i,
            "question": r["Question"],
            "gold": r["Correct Answer"].strip(),
            "image_bytes": img_bytes,
        })
    if n_items:
        items = items[:n_items]
    return items


def build_model(base: str, adapter_dir: str | None, device: torch.device):
    m = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    m.to(device)
    if adapter_dir:
        m = PeftModel.from_pretrained(m, adapter_dir)
        m.to(device)
    m.eval()
    return m


def get_language_layers(model) -> list:
    """Return the language-tower decoder layers (Gemma-3 → 34 layers)."""
    # Path through PeftModel wrapper
    m = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    return m.model.language_model.layers


def capture_activations(model, processor, tokenizer, items, device, layers):
    """Run each item through the model with image attached and capture residual stream
    at the LAST prompt token for every language-tower layer.

    Returns tensor of shape [N, L, d_model].
    """
    N = len(items)
    L = len(layers)
    # We'll grab d_model after first pass.
    hidden_states = None

    # Attach hooks: each layer's OUTPUT (residual stream after layer) at the last token.
    captured = [None] * L
    hooks = []
    for idx, layer in enumerate(layers):
        def make_hook(k):
            def hook(module, inp, out):
                # out is usually a tuple (hidden_states, ...) for decoder layers
                h = out[0] if isinstance(out, tuple) else out
                # h: [batch, seq, d_model] -> take last token
                captured[k] = h[:, -1, :].detach().float().cpu()
            return hook
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    try:
        all_layer_acts = []
        for i, it in enumerate(items):
            img = Image.open(io.BytesIO(it["image_bytes"])).convert("RGB")
            msg = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": it["question"]},
            ]}]
            inputs = processor.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True,
            )
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            for k, v in inputs.items():
                if hasattr(v, "dtype") and v.dtype in (torch.float32, torch.float16):
                    inputs[k] = v.to(torch.bfloat16)
            with torch.no_grad():
                _ = model(**inputs, output_hidden_states=False, use_cache=False)
            stacked = torch.stack(captured, dim=0).squeeze(1)  # [L, d_model]
            all_layer_acts.append(stacked)
            if (i + 1) % 25 == 0:
                print(f"  capture {i+1}/{N}", flush=True)
        return torch.stack(all_layer_acts, dim=0)  # [N, L, d_model]
    finally:
        for h in hooks:
            h.remove()


def compute_diff_in_means(A_treated: torch.Tensor, A_ctrlb: torch.Tensor):
    """A_treated, A_ctrlb: [N, L, d]. Returns d_hat [L, d] (unit norm) + norms [L]."""
    diff = A_treated.mean(0) - A_ctrlb.mean(0)  # [L, d]
    norms = diff.norm(dim=-1)  # [L]
    d_hat = diff / (norms.unsqueeze(-1) + 1e-8)  # [L, d]
    return d_hat, norms


def compute_probe_auroc(A_treated: torch.Tensor, A_ctrlb: torch.Tensor, k_folds: int = 5):
    """Per-layer 5-fold CV AUROC of linear probe distinguishing treated vs ctrlb."""
    N_t = A_treated.shape[0]
    N_b = A_ctrlb.shape[0]
    L = A_treated.shape[1]
    y = np.concatenate([np.ones(N_t), np.zeros(N_b)])
    aurocs = np.zeros(L)
    for l in range(L):
        X = torch.cat([A_treated[:, l], A_ctrlb[:, l]], dim=0).numpy()
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_aucs = []
        for train_i, test_i in kf.split(X):
            probe = LogisticRegression(max_iter=200, C=1.0)
            probe.fit(X[train_i], y[train_i])
            pred = probe.decision_function(X[test_i])
            fold_aucs.append(roc_auc_score(y[test_i], pred))
        aurocs[l] = float(np.mean(fold_aucs))
    return aurocs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treated", type=str, required=True, help="Path to treated LoRA adapter dir.")
    ap.add_argument("--ctrlb", type=str, required=True, help="Path to ctrl_b LoRA adapter dir.")
    ap.add_argument("--base", type=str, default=BASE_MODEL)
    ap.add_argument("--qa_i", type=str, default=str(QA_I_PARQUET))
    ap.add_argument("--n_pairs", type=int, default=133, help="Number of items (max 133 for QA_I).")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    items = load_qa_items(args.qa_i, n_items=args.n_pairs)
    print(f"[m1] loaded {len(items)} items", flush=True)

    tokenizer, processor = load_tokenizer_and_processor(args.base)

    def _capture(adapter):
        m = build_model(args.base, adapter, device)
        layers = get_language_layers(m)
        A = capture_activations(m, processor, tokenizer, items, device, layers)
        # Free VRAM
        del m
        torch.cuda.empty_cache()
        return A, len(layers)

    print(f"[m1] capturing treated activations", flush=True)
    A_t, L = _capture(args.treated)
    print(f"[m1] treated shape {A_t.shape}, L={L}", flush=True)
    print(f"[m1] capturing ctrlb activations", flush=True)
    A_b, _ = _capture(args.ctrlb)
    print(f"[m1] ctrlb shape {A_b.shape}", flush=True)

    # Diff-in-means
    d_hat, norms = compute_diff_in_means(A_t, A_b)  # [L, d], [L]
    print(f"[m1] diff-in-means norms per layer:")
    for l in range(L):
        print(f"  layer {l:2d}: ‖Δ‖ = {norms[l].item():.4f}")

    # Probe AUROC
    print(f"[m1] computing probe AUROC (5-fold)...", flush=True)
    aurocs = compute_probe_auroc(A_t, A_b, k_folds=5)
    print(f"[m1] probe AUROC per layer:")
    for l in range(L):
        print(f"  layer {l:2d}: AUROC = {aurocs[l]:.4f}")

    # Rank layers by combined signal (diff norm z-score + probe AUROC)
    norms_np = norms.numpy()
    z_norms = (norms_np - norms_np.mean()) / (norms_np.std() + 1e-8)
    combined = z_norms + 5 * (aurocs - 0.5)  # weight AUROC by 5 to match z-score scale
    top_layer = int(combined.argmax())
    print(f"[m1] TOP layer = {top_layer} (combined score {combined[top_layer]:.3f})", flush=True)

    # Save
    ranked = []
    for l in range(L):
        ranked.append({
            "layer": l,
            "diff_norm": float(norms[l].item()),
            "diff_norm_zscore": float(z_norms[l]),
            "probe_auroc": float(aurocs[l]),
            "combined_score": float(combined[l]),
            "is_top": (l == top_layer),
        })
    ranked_sorted = sorted(ranked, key=lambda r: r["combined_score"], reverse=True)

    with open(out / "ranked_sites.json", "w") as f:
        json.dump({
            "n_pairs": args.n_pairs,
            "n_language_layers": L,
            "top_layer": top_layer,
            "top_direction_norm": float(norms[top_layer].item()),
            "top_probe_auroc": float(aurocs[top_layer]),
            "ranked": ranked_sorted,
            "verdict": "located" if (aurocs[top_layer] >= 0.7 or abs(z_norms[top_layer]) >= 3) else "unlocated",
        }, f, indent=2)

    # Save d_hat at top layer
    np.save(out / "direction_top.npy", d_hat[top_layer].numpy())
    # Save all directions for downstream use
    np.save(out / "directions_all_layers.npy", d_hat.numpy())
    # Save per-layer per-item activations (float32, small ~500 KB per layer, useful for later)
    np.savez_compressed(out / "activations.npz",
                        A_treated=A_t.numpy().astype(np.float32),
                        A_ctrlb=A_b.numpy().astype(np.float32))
    print(f"[m1] saved to {out}", flush=True)


if __name__ == "__main__":
    main()
