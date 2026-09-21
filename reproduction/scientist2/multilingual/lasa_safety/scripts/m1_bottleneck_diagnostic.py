"""
M1 — Bottleneck-layer diagnostic (Location, C1 cheap screen).

For each layer l of LLaMA-3.1-8B-Instruct (0..31), compute:
  Sem(l) = mean pairwise cosine of last-token residual-stream states
          across the 10 parallel-language versions of the SAME meaning-group.
  Lang(l) = mean cosine of last-token states between DIFFERENT meaning-groups
           within the SAME language, averaged over languages.
  R(l) = Sem(l) / Lang(l)
  D(l) = Lang(l) - Sem(l)

L* = argmax_l R(l).

Per-language decomposition: Sem^lang(l) = mean_g cos(h_l(g, en), h_l(g, lang)).

Writes results/M1_bottleneck_diagnostic.json.

Reads MultiJail from $DATA_DIR/multijail/MultiJail.csv (10-lang parallel table).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_LANGS = ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]


def load_multijail(csv_path: str, langs: list[str]) -> list[dict]:
    """Read MultiJail.csv into a list of {id, tags, lang: prompt} rows."""
    rows: list[dict] = []
    with open(csv_path, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            entry = {"id": int(r["id"]), "tags": r.get("tags", "")}
            keep = True
            for lang in langs:
                if lang not in r or not r[lang].strip():
                    keep = False
                    break
                entry[lang] = r[lang].strip()
            if keep:
                rows.append(entry)
    return rows


@torch.no_grad()
def collect_last_token_hidden(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_len: int = 512,
) -> np.ndarray:
    """Return a [num_hidden_states, hidden_dim] float32 numpy array of last-token
    residual-stream states across all layers (including embedding = index 0).
    """
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=False,
    )
    # hidden_states is a tuple of (num_layers+1) [1, seq_len, hidden] tensors.
    hs = torch.stack(out.hidden_states, dim=0).squeeze(1)  # [L+1, seq, hidden]
    # last non-pad token index (attn == 1)
    last_idx = int(attn.sum(dim=1).item() - 1)
    last = hs[:, last_idx, :].to(torch.float32).cpu().numpy()  # [L+1, hidden]
    return last


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalized cosine similarity."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.clip(n, 1e-8, None)
    xn = x / n
    return xn @ xn.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--multijail_csv", required=True)
    ap.add_argument("--languages", default=",".join(DEFAULT_LANGS))
    ap.add_argument("--n_prompts_per_lang", type=int, default=442,
                    help="Cap on number of prompt groups (max = full csv)")
    ap.add_argument("--n_lang_pairs", type=int, default=300,
                    help="Random pairs drawn per language for Lang(l)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    langs = args.languages.split(",")
    print(f"[M1] languages={langs}")

    rows = load_multijail(args.multijail_csv, langs)
    if args.n_prompts_per_lang and args.n_prompts_per_lang < len(rows):
        rows = rows[: args.n_prompts_per_lang]
    print(f"[M1] loaded {len(rows)} parallel meaning-groups")

    t0 = time.time()
    print(f"[M1] loading model {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype_map[args.dtype],
        device_map={"": 0},
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[M1] model loaded in {time.time() - t0:.1f}s on {device}")

    # Extract last-token hidden states: hs[i, lang] = [L+1, hidden]
    all_hs: dict[int, dict[str, np.ndarray]] = {}
    n_layers_total = None
    t0 = time.time()
    for i, row in enumerate(rows):
        per_lang: dict[str, np.ndarray] = {}
        for lang in langs:
            per_lang[lang] = collect_last_token_hidden(
                model, tok, row[lang], device, max_len=args.max_len
            )
            if n_layers_total is None:
                n_layers_total = per_lang[lang].shape[0]
        all_hs[i] = per_lang
        if (i + 1) % 50 == 0:
            print(f"[M1] extracted {i+1}/{len(rows)} groups (elapsed {time.time()-t0:.1f}s)")
    print(f"[M1] extraction complete: {time.time()-t0:.1f}s; layers={n_layers_total}")

    # For LLaMA hidden_states: index 0 = embedding, 1..N = layer outputs. We report the
    # 32 transformer layers (indices 1..32). We'll index them as l=0..31 externally.
    L = n_layers_total  # e.g. 33 for a 32-layer model
    transformer_layers = list(range(1, L))  # skip embedding
    n_transformer = len(transformer_layers)

    # Sem(l) — mean pairwise cosine across languages within one meaning-group.
    n_groups = len(rows)
    n_lang_combos = len(list(combinations(range(len(langs)), 2)))
    sem_per_layer = np.zeros(n_transformer, dtype=np.float64)
    sem_per_group_layer = np.zeros((n_groups, n_transformer), dtype=np.float32)
    for gi in range(n_groups):
        # stack lang vectors per layer: [n_langs, L, hidden]
        vecs = np.stack([all_hs[gi][lang] for lang in langs], axis=0)  # [n_langs, L+1, hidden]
        for out_li, li in enumerate(transformer_layers):
            V = vecs[:, li, :]  # [n_langs, hidden]
            csim = cosine_matrix(V)
            # off-diagonal upper triangle mean
            iu = np.triu_indices_from(csim, k=1)
            sem_per_group_layer[gi, out_li] = csim[iu].mean()
    sem_per_layer = sem_per_group_layer.mean(axis=0)

    # Lang(l) — for each language, random pairs of different meaning-groups.
    n_pairs = min(args.n_lang_pairs, n_groups * (n_groups - 1) // 2)
    lang_per_layer = np.zeros(n_transformer, dtype=np.float64)
    lang_per_layer_by_lang: dict[str, np.ndarray] = {}
    for lang in langs:
        # Sample pairs
        pairs = []
        seen = set()
        rng = random.Random(args.seed + hash(lang) % 10_000)
        while len(pairs) < n_pairs:
            a, b = rng.sample(range(n_groups), 2)
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
        per_layer_vals = np.zeros(n_transformer, dtype=np.float32)
        for (a, b) in pairs:
            va = np.stack([all_hs[a][lang][li] for li in transformer_layers], axis=0)  # [L, hidden]
            vb = np.stack([all_hs[b][lang][li] for li in transformer_layers], axis=0)  # [L, hidden]
            na = np.clip(np.linalg.norm(va, axis=-1, keepdims=True), 1e-8, None)
            nb = np.clip(np.linalg.norm(vb, axis=-1, keepdims=True), 1e-8, None)
            csim = ((va / na) * (vb / nb)).sum(axis=-1)
            per_layer_vals += csim
        per_layer_vals /= len(pairs)
        lang_per_layer_by_lang[lang] = per_layer_vals
        lang_per_layer += per_layer_vals
    lang_per_layer /= len(langs)

    # R(l), D(l), L*
    R = sem_per_layer / np.clip(lang_per_layer, 1e-8, None)
    D = lang_per_layer - sem_per_layer
    L_star = int(np.argmax(R))  # 0-indexed among transformer layers
    R_max = float(R[L_star])
    print(f"[M1] L*={L_star}  R_max={R_max:.4f}  D_min={float(D.min()):.4f}")

    # Per-language decomposition: Sem^lang(l) = mean_g cos(h_l(g, en), h_l(g, lang))
    sem_by_lang: dict[str, np.ndarray] = {}
    for lang in langs:
        vals = np.zeros(n_transformer, dtype=np.float64)
        for gi in range(n_groups):
            ven = np.stack([all_hs[gi]["en"][li] for li in transformer_layers], axis=0)
            vlg = np.stack([all_hs[gi][lang][li] for li in transformer_layers], axis=0)
            nen = np.clip(np.linalg.norm(ven, axis=-1, keepdims=True), 1e-8, None)
            nlg = np.clip(np.linalg.norm(vlg, axis=-1, keepdims=True), 1e-8, None)
            vals += ((ven / nen) * (vlg / nlg)).sum(axis=-1)
        vals /= n_groups
        sem_by_lang[lang] = vals

    # Bootstrap CI over meaning-groups on R(l) at L*
    n_boot = 1000
    rng = np.random.default_rng(args.seed)
    boot_R_star = np.zeros(n_boot)
    lang_mean = lang_per_layer[L_star]  # constant across bootstraps of groups (Lang uses lang-internal pairs)
    for bi in range(n_boot):
        idx = rng.integers(0, n_groups, size=n_groups)
        sem_boot = sem_per_group_layer[idx, L_star].mean()
        boot_R_star[bi] = sem_boot / max(lang_mean, 1e-8)
    ci95 = (float(np.percentile(boot_R_star, 2.5)), float(np.percentile(boot_R_star, 97.5)))

    per_layer_out = []
    # Bootstrap per-layer for CI on R(l)
    for out_li in range(n_transformer):
        boot = np.zeros(n_boot)
        lm = lang_per_layer[out_li]
        for bi in range(n_boot):
            idx = rng.integers(0, n_groups, size=n_groups)
            sm = sem_per_group_layer[idx, out_li].mean()
            boot[bi] = sm / max(lm, 1e-8)
        per_layer_out.append({
            "l": int(out_li),
            "Sem": float(sem_per_layer[out_li]),
            "Lang": float(lang_per_layer[out_li]),
            "R": float(R[out_li]),
            "D": float(D[out_li]),
            "R_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        })

    result = {
        "meta": {
            "model_path": args.model_path,
            "n_prompt_groups": n_groups,
            "n_transformer_layers": n_transformer,
            "languages": langs,
            "n_lang_pairs": n_pairs,
            "seed": args.seed,
            "dtype": args.dtype,
            "run_id": "M1",
        },
        "per_layer": per_layer_out,
        "L_star": L_star,
        "L_star_R": R_max,
        "L_star_ci95": ci95,
        "per_language_sem_to_en": {
            lang: [float(x) for x in sem_by_lang[lang]] for lang in langs
        },
        "per_language_lang_within": {
            lang: [float(x) for x in lang_per_layer_by_lang[lang]] for lang in langs
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"[M1] wrote {args.out}")


if __name__ == "__main__":
    main()
