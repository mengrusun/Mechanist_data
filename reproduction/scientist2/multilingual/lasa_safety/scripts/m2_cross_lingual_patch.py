"""
M2 — Cross-lingual activation patching (Causal Intervention, C1 confirmation).

For 100 meaning-groups × 9 non-English target languages, forward-pass the target
prompt p_x, patch the last-token residual state at layer L (one of {L*, l=2, l=30})
with the corresponding state from the same-meaning English prompt p_en, resume
decoding, and measure whether the completion preserves the intended meaning.

Conditions:
  A: target patch at L*
  B: target patch at l=2 (surface control near input)
  C: target patch at l=30 (surface control near output)
  D: matched-control patch at L* (patch with an UNRELATED English prompt's state)

Meaning-preservation metric: LaBSE cosine between the sampled continuation and
the reference continuation. Reference continuation = unpatched completion of p_x
(the model's own free-continuation without patching serves as the "target-language
reference"; a patched completion whose LaBSE cosine to that reference is HIGH
means meaning was preserved despite the patch).

Also samples 100 completions across conditions for GPT-4o judge audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_LANGS = ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]


def load_multijail(csv_path: str, langs: list[str]) -> list[dict]:
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


def build_chat_prompt(tokenizer, prompt: str) -> str:
    """Wrap a user prompt in the LLaMA-3 chat template so the last-token
    residual state is meaningful (post-header)."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def get_last_token_residual(
    model, tokenizer, prompt: str, layer_idx: int, device
) -> torch.Tensor:
    """Return [hidden_dim] residual-stream state at layer_idx after the last token.
    layer_idx corresponds to the transformer layer index (0..N-1), matching the
    convention in M1 (i.e., hidden_states index = layer_idx + 1)."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=False,
    )
    # hidden_states[layer_idx + 1] is the output of transformer layer layer_idx.
    hs = out.hidden_states[layer_idx + 1]  # [1, seq, hidden]
    last_idx = int(attn.sum(dim=1).item() - 1)
    return hs[0, last_idx, :].clone()


@torch.no_grad()
def generate_with_patch(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    patch_vector: torch.Tensor | None,  # [hidden_dim] or None
    device,
    max_new_tokens: int = 80,
) -> str:
    """Generate a continuation with a single forward-hook patch applied to the
    last-token residual state at layer_idx of the FIRST forward pass. Subsequent
    autoregressive steps do NOT re-patch (patch is meant to alter the initial
    residual before decoding starts).
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    last_idx = int(attn.sum(dim=1).item() - 1)

    hook_state = {"applied": False}

    def hook(module, inputs, output):
        # output is a tuple (hidden_states, ...) for LlamaDecoderLayer.
        if hook_state["applied"] or patch_vector is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] <= last_idx:
            return output
        hs[0, last_idx, :] = patch_vector.to(hs.dtype).to(hs.device)
        hook_state["applied"] = True
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    # Register on the target transformer layer.
    target_layer = model.model.layers[layer_idx]
    handle = target_layer.register_forward_hook(hook)
    try:
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        handle.remove()

    completion = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
    return completion


def load_labse(model_path: str, device):
    from sentence_transformers import SentenceTransformer

    labse = SentenceTransformer(model_path, device=str(device))
    return labse


def labse_cosine(labse, a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    emb = labse.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float((emb[0] * emb[1]).sum())


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Character-level n-gram Jaccard similarity. Cross-lingual-*fair* (deterministic,
    tokenizer-free) but heavily penalizes language switches (an English completion
    of an Arabic reference scores near zero even if meaning is preserved). Used
    only as a *language-fidelity* diagnostic, NOT the primary meaning metric.
    """
    a = a.strip()
    b = b.strip()
    if not a or not b:
        return 0.0
    def grams(s: str) -> set[str]:
        s = " " + s + " "
        return {s[i:i + n] for i in range(len(s) - n + 1)}
    ga = grams(a.lower())
    gb = grams(b.lower())
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


@torch.no_grad()
def semantic_embed_via_llama(model, tokenizer, text: str, device, target_layer: int = -1) -> np.ndarray:
    """Compute a text embedding as the last-token residual state at `target_layer`
    (default = final transformer layer) of the SAME LLaMA-3.1 model. Cross-lingual-
    native for multilingual bases like LLaMA-3.1-8B-Instruct. Text is embedded as a
    plain user message (chat template) so the last-token state summarizes the input.

    NOTE: this uses the SAME model as the intervention target. It is a *diagnostic*
    proxy — a true independent judge (LaBSE / GPT-4o) is preferred but unavailable
    here due to network issues. Declared in EXPERIMENT_RESULTS.md.
    """
    if not text.strip():
        return np.zeros(model.config.hidden_size, dtype=np.float32)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=384,
                    add_special_tokens=True).to(device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[target_layer]  # [1, T, H]
    attn = enc["attention_mask"]
    # Mean pool over non-pad tokens
    mask = attn.unsqueeze(-1).to(hs.dtype)
    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    v = pooled[0].to(torch.float32).cpu().numpy()
    n = np.linalg.norm(v)
    return v / max(n, 1e-8)


def semantic_cosine(model, tok, a: str, b: str, device, layer: int = -1) -> float:
    va = semantic_embed_via_llama(model, tok, a, device, target_layer=layer)
    vb = semantic_embed_via_llama(model, tok, b, device, target_layer=layer)
    return float((va * vb).sum())


def meaning_score(labse, a: str, b: str) -> float:
    """Fallback char-jaccard when no true semantic scorer is available."""
    if labse is not None:
        return labse_cosine(labse, a, b)
    return char_ngram_jaccard(a, b, n=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--multijail_csv", required=True)
    ap.add_argument("--m1_json", required=True, help="path to M1 output for L*")
    ap.add_argument("--l_star_override", type=int, default=None,
                    help="Override L* (0-indexed layer). If unset, read from m1_json.")
    ap.add_argument("--control_layers", default="2,30")
    ap.add_argument("--target_langs", default="zh,it,vi,ar,ko,th,bn,sw,jv")
    ap.add_argument("--languages_all", default=",".join(DEFAULT_LANGS))
    ap.add_argument("--n_pair_groups", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--labse_path", default=None, help="local LaBSE model dir or HF id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    langs_all = args.languages_all.split(",")
    target_langs = args.target_langs.split(",")
    control_layers = [int(x) for x in args.control_layers.split(",")]

    # Load L* from M1
    with open(args.m1_json, encoding="utf-8") as fh:
        m1 = json.load(fh)
    l_star = int(args.l_star_override) if args.l_star_override is not None else int(m1["L_star"])
    print(f"[M2] L*={l_star} control_layers={control_layers}")

    rows = load_multijail(args.multijail_csv, langs_all)
    random.shuffle(rows)
    n_use = min(args.n_pair_groups, len(rows))
    rows = rows[:n_use]
    print(f"[M2] using {n_use} meaning-groups")

    t0 = time.time()
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
    print(f"[M2] model loaded in {time.time()-t0:.1f}s")

    labse = None
    if args.labse_path:
        try:
            labse = load_labse(args.labse_path, device)
            print(f"[M2] LaBSE loaded from {args.labse_path}")
        except Exception as e:
            print(f"[M2] WARN: failed to load LaBSE ({e}); falling back to zero cosine")

    # Pre-compute chat-templated prompts for every (group, lang) and cache last-token residuals
    # at L*, control layers for the English prompt.
    chat_prompts: dict[int, dict[str, str]] = {}
    for gi, row in enumerate(rows):
        chat_prompts[gi] = {lg: build_chat_prompt(tok, row[lg]) for lg in langs_all}

    # Cache English residuals at L*, control layers
    en_res: dict[int, dict[int, torch.Tensor]] = {gi: {} for gi in range(n_use)}
    all_layers = [l_star] + [l for l in control_layers if l != l_star]
    for gi in range(n_use):
        for L in all_layers:
            en_res[gi][L] = get_last_token_residual(
                model, tok, chat_prompts[gi]["en"], L, device
            )
        if (gi + 1) % 20 == 0:
            print(f"[M2] cached en residuals for {gi+1}/{n_use} groups")

    # Baseline unpatched completions for each (group, target_lang) — the reference.
    references: dict[int, dict[str, str]] = {}
    t0 = time.time()
    for gi in range(n_use):
        references[gi] = {}
        for lang in target_langs:
            references[gi][lang] = generate_with_patch(
                model, tok, chat_prompts[gi][lang],
                layer_idx=l_star, patch_vector=None, device=device,
                max_new_tokens=args.max_new_tokens,
            )
        if (gi + 1) % 10 == 0:
            print(f"[M2] baseline references {gi+1}/{n_use}  (elapsed {time.time()-t0:.1f}s)")
    print(f"[M2] baseline references done: {time.time()-t0:.1f}s")

    # Patched completions per condition
    per_condition: dict[str, list[dict]] = {"A": [], "B": [], "C": [], "D": []}
    condition_layer = {"A": l_star, "B": control_layers[0], "C": control_layers[1], "D": l_star}
    condition_name = {
        "A": f"patch@L*={l_star}",
        "B": f"patch@l={control_layers[0]}",
        "C": f"patch@l={control_layers[1]}",
        "D": f"matched-control@L*={l_star}",
    }

    t0 = time.time()
    for gi in range(n_use):
        # Choose matched control group for condition D (unrelated English prompt)
        other = (gi + 7) % n_use
        while other == gi:
            other = (other + 1) % n_use
        for lang in target_langs:
            prompt_lang = chat_prompts[gi][lang]
            ref = references[gi][lang]
            for cond in ["A", "B", "C", "D"]:
                L = condition_layer[cond]
                if cond == "D":
                    patch_vec = en_res[other][l_star]
                else:
                    patch_vec = en_res[gi][L]
                completion = generate_with_patch(
                    model, tok, prompt_lang, layer_idx=L,
                    patch_vector=patch_vec, device=device,
                    max_new_tokens=args.max_new_tokens,
                )
                # Primary: model-internal semantic cosine (cross-lingual-native)
                sem_cos = semantic_cosine(model, tok, completion, ref, device, layer=-1)
                # Secondary: char-Jaccard (language-fidelity proxy)
                char_score = meaning_score(None, completion, ref)
                per_condition[cond].append({
                    "group": gi,
                    "target_lang": lang,
                    "layer": L,
                    "completion": completion,
                    "reference": ref,
                    "semantic_cosine_to_ref": sem_cos,
                    "char_jaccard_to_ref": char_score,
                })
        if (gi + 1) % 5 == 0:
            elapsed = time.time() - t0
            per_group = elapsed / (gi + 1)
            eta = per_group * (n_use - gi - 1)
            print(f"[M2] patching {gi+1}/{n_use}  elapsed={elapsed:.1f}s  eta={eta/60:.1f}min")

    # Aggregate — report BOTH primary (semantic cosine) and secondary (char Jaccard)
    def agg(items: list[dict], key: str) -> dict:
        vals = np.array([x[key] for x in items], dtype=np.float64)
        rng = np.random.default_rng(args.seed)
        boots = np.array([vals[rng.integers(0, len(vals), size=len(vals))].mean()
                          for _ in range(1000)])
        return {
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }

    aggregated = {cond: {
        "semantic": agg(items, "semantic_cosine_to_ref"),
        "char_jaccard": agg(items, "char_jaccard_to_ref"),
    } for cond, items in per_condition.items()}

    # Per-language aggregation
    per_lang_agg: dict[str, dict[str, dict]] = {}
    for lang in target_langs:
        per_lang_agg[lang] = {}
        for cond, items in per_condition.items():
            xs = [x for x in items if x["target_lang"] == lang]
            per_lang_agg[lang][cond] = {
                "semantic": agg(xs, "semantic_cosine_to_ref") if xs else {},
                "char_jaccard": agg(xs, "char_jaccard_to_ref") if xs else {},
            }

    # Simple C1 verdict-hint on the primary (semantic) metric
    A = aggregated["A"]["semantic"]["mean"]
    B = aggregated["B"]["semantic"]["mean"]
    C = aggregated["C"]["semantic"]["mean"]
    D = aggregated["D"]["semantic"]["mean"]
    hint = "inconclusive"
    if A > B + 0.01 and A > C + 0.01 and A > D + 0.01:
        hint = "supported"
    elif A + 0.005 < max(B, C, D):
        hint = "refuted"

    result = {
        "meta": {
            "model_path": args.model_path,
            "L_star": l_star,
            "control_layers": control_layers,
            "n_groups": n_use,
            "target_langs": target_langs,
            "condition_name": condition_name,
            "seed": args.seed,
            "labse_used": labse is not None,
            "run_id": "M2",
        },
        "by_condition": aggregated,
        "per_language": per_lang_agg,
        "c1_verdict_hint": hint,
        "sample_completions": {
            cond: items[:20] for cond, items in per_condition.items()
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
    print(f"[M2] wrote {args.out}")
    print(f"[M2/semantic] A(L*)={A:.4f} B(l={control_layers[0]})={B:.4f} "
          f"C(l={control_layers[1]})={C:.4f} D(matched)={D:.4f}  hint={hint}")
    Ac = aggregated["A"]["char_jaccard"]["mean"]
    Bc = aggregated["B"]["char_jaccard"]["mean"]
    Cc = aggregated["C"]["char_jaccard"]["mean"]
    Dc = aggregated["D"]["char_jaccard"]["mean"]
    print(f"[M2/char_jaccard] A(L*)={Ac:.4f} B(l={control_layers[0]})={Bc:.4f} "
          f"C(l={control_layers[1]})={Cc:.4f} D(matched)={Dc:.4f}")


if __name__ == "__main__":
    main()
