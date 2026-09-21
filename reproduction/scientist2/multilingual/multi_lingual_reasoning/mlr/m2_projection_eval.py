"""M2 / M3 — Null-space projection + signed α-sweep evaluation on MGSM.

Applies the intervention `h ← h + α · Π_lang · h` (equivalently `h ← h + α · V (V^T h)`)
at every layer in [layer_group_start, num_layers - k_top_excluded), i.e., the "non-upper" set with `k_top`
uppermost layers intact.

For M2, α = -1 corresponds to null-space projection (h ← h - Π_lang·h).
For M3, α sweeps in [-1.5, +1.5] on the M2-winning (layer_group, k_top).

Options:
    --random_subspace_control : use a random rank-r subspace instead of V_lang (matched-control).
    --leave_language_out <lang> : refit V_lang without <lang> before evaluation (LOL check).

Uses vLLM for batched generation when available, else a HuggingFace generation fallback.

Output: results/<milestone>/<config>.jsonl with per-problem records + a summary.json alongside.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlr.data import MGSM_LANGS, GLOTLID_MAP, load_flores_all, load_mgsm_all, default_few_shot_en, format_mgsm_prompt, normalize_langs
from mlr.activations import ResidualStreamCache
from mlr.mgsm_eval import extract_answer, numeric_equal, grade


def _layer_group_range(layer_group: str, num_layers: int) -> Tuple[int, int]:
    """See mlr.m1_locate._layer_group_range — same convention (plan-native constants for 36-layer Qwen-3-4B)."""
    if num_layers == 36:
        if layer_group == "early":
            return (0, 12)
        if layer_group == "mid":
            return (12, 24)
        if layer_group == "all_non_upper":
            return (0, 28)
    else:
        if layer_group == "early":
            return (0, num_layers // 3)
        if layer_group == "mid":
            return (num_layers // 3, 2 * num_layers // 3)
        if layer_group == "all_non_upper":
            return (0, int(round(num_layers * 28 / 36)))
    raise ValueError(f"Unknown layer_group: {layer_group}")


def _refit_v_lang(model, tokenizer, args, layer_used: int, exclude_lang: Optional[str]) -> np.ndarray:
    """Re-fit V_lang from the probe set at `layer_used`, optionally excluding one language."""
    langs = [l for l in MGSM_LANGS if l != exclude_lang]
    probe_texts = load_flores_all(langs=langs, n_probe=args.n_probe, seed=args.seed, data_dir=args.data_dir)
    with ResidualStreamCache(model, tokenizer, [layer_used], pool="last") as cache:
        acts_per_lang = {}
        for lang, sents in probe_texts.items():
            acts_per_lang[lang] = cache.encode_dataset(sents, batch_size=args.batch_size, max_length=args.max_length)[layer_used]
    langs_sorted = sorted(langs)
    mus = np.stack([acts_per_lang[l].mean(axis=0) for l in langs_sorted], axis=0)
    diff = mus - mus.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(diff, full_matrices=False)
    V = Vt[:args.rank_r].T
    return V


def _random_subspace(hidden: int, rank_r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = rng.standard_normal(size=(hidden, rank_r)).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    return Q[:, :rank_r]


class SteeringHooks:
    """Apply `h ← h + α · Π_lang · h` at a set of transformer decoder-layer indices via forward hooks."""

    def __init__(self, model, layer_indices: List[int], V: torch.Tensor, alpha: float):
        self.model = model
        self.layer_indices = sorted(set(int(x) for x in layer_indices))
        self.V = V  # (hidden, rank_r) on model device, in model dtype
        self.alpha = float(alpha)
        self.handles = []

    def _hook_fn(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden: (batch, seq, hidden)
        # projection: Π = V V^T; delta = α · h V V^T
        # Compute (h V) V^T
        proj_coeffs = torch.matmul(hidden, self.V)          # (b, s, r)
        proj = torch.matmul(proj_coeffs, self.V.transpose(-1, -2))  # (b, s, hidden)
        new_hidden = hidden + self.alpha * proj
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    def __enter__(self):
        layers = self.model.model.layers
        for idx in self.layer_indices:
            h = layers[idx].register_forward_hook(self._hook_fn)
            self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles = []


def _hf_generate(model, tokenizer, prompts: List[str], max_new_tokens: int, batch_size: int, greedy: bool) -> List[str]:
    """Batched HF generation."""
    outs = []
    device = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        gen_kwargs = dict(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            pad_token_id=tokenizer.pad_token_id,
        )
        if greedy:
            gen_kwargs.update(dict(temperature=1.0, top_p=1.0))
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        # slice generated tokens only
        gen = out[:, enc["input_ids"].size(1):]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outs.extend(texts)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--v_lang", default=None, help="Path to .npz containing V_lang (from M1)")
    ap.add_argument("--refit_from_probe", action="store_true", help="Refit V_lang inline from probe data")
    ap.add_argument("--rank_r", type=int, default=8, help="Only used with --refit_from_probe or --random_subspace_control")
    ap.add_argument("--k_top_excluded", type=int, required=True)
    ap.add_argument("--layer_group", required=True, choices=["early", "mid", "all_non_upper"])
    ap.add_argument("--alpha", type=float, default=-1.0)
    ap.add_argument("--dataset", default=None, help="Dataset dir (default DATA_DIR/mgsm)")
    ap.add_argument("--languages", default=",".join(MGSM_LANGS))
    ap.add_argument("--n_shot", type=int, default=3)
    ap.add_argument("--n_problems_per_lang", type=int, default=250)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_probe", type=int, default=250, help="Only used when refitting V_lang")
    ap.add_argument("--random_subspace_control", action="store_true")
    ap.add_argument("--leave_language_out", default=None)
    ap.add_argument("--glotlid_path", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--greedy", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True,
                    help="Use greedy decoding (default true)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    print(f"[m2] loading model from {args.model_dir}", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    lg_start, lg_end = _layer_group_range(args.layer_group, num_layers)
    site_end = num_layers - args.k_top_excluded
    site_start = lg_start
    site_end_effective = min(lg_end, site_end)  # intervention range: [lg_start, min(lg_end, num_layers - k_top))
    layer_indices = list(range(site_start, site_end_effective))
    if not layer_indices:
        raise ValueError(f"Empty intervention range: [{site_start},{site_end_effective}) (lg={args.layer_group}, k_top={args.k_top_excluded})")
    print(f"[m2] intervention on layers {layer_indices[0]}..{layer_indices[-1]} (n={len(layer_indices)})", flush=True)

    # Load / build V_lang
    layer_used = (lg_start + lg_end) // 2
    if args.random_subspace_control:
        V_np = _random_subspace(hidden, args.rank_r, args.seed).astype(np.float32)
        print(f"[m2] using random subspace, rank={args.rank_r}, seed={args.seed}", flush=True)
    elif args.leave_language_out is not None:
        V_np = _refit_v_lang(model, tokenizer, args, layer_used, exclude_lang=args.leave_language_out)
        print(f"[m2] refit V_lang without {args.leave_language_out}, shape={V_np.shape}", flush=True)
    elif args.refit_from_probe:
        V_np = _refit_v_lang(model, tokenizer, args, layer_used, exclude_lang=None)
        print(f"[m2] refit V_lang from probe, shape={V_np.shape}", flush=True)
    else:
        if args.v_lang is None or not Path(args.v_lang).exists():
            raise FileNotFoundError(f"V_lang not found at {args.v_lang}; pass --refit_from_probe or --v_lang")
        data = np.load(args.v_lang, allow_pickle=True)
        V_np = data["V_lang"].astype(np.float32)
        print(f"[m2] loaded V_lang from {args.v_lang}, shape={V_np.shape}", flush=True)

    # Sanity: check V_lang is (approximately) orthonormal — required for correct projector Π = V V^T
    gram = V_np.T @ V_np
    ident = np.eye(V_np.shape[1], dtype=V_np.dtype)
    orth_err = float(np.linalg.norm(gram - ident))
    if orth_err > 1e-2:
        print(f"[m2] WARN: V_lang not orthonormal, ||V^T V - I||_F = {orth_err:.4f}. Re-orthonormalizing.", flush=True)
        Q, _ = np.linalg.qr(V_np)
        V_np = Q[:, : V_np.shape[1]].astype(np.float32)
    V = torch.from_numpy(V_np).to(device=model.device, dtype=model.dtype)

    # Load MGSM (normalize language codes: task.md uses En/Es/Jp/..., we need en/es/ja)
    raw_langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    langs = normalize_langs(raw_langs)
    mgsm = load_mgsm_all(langs=langs, split="test", data_dir=args.data_dir)

    # Load GlotLID (optional; if the model file is missing, we skip fidelity)
    glot = None
    try:
        from mlr.glotlid_wrapper import GlotLID
        glot = GlotLID(model_path=args.glotlid_path)
        print(f"[m2] GlotLID loaded", flush=True)
    except Exception as e:
        print(f"[m2] GlotLID unavailable: {e}", flush=True)

    # Few-shot
    fewshot = default_few_shot_en()[: args.n_shot]

    records = []
    per_lang_stats = {}
    for lang in langs:
        df = mgsm[lang]
        n = min(args.n_problems_per_lang, len(df))
        prompts = []
        golds = []
        qids = []
        for i in range(n):
            q = str(df.iloc[i]["question"])
            gold = df.iloc[i].get("answer_number", None)
            if gold is None:
                # some MGSM parquet may name it 'answer'
                gold = df.iloc[i].get("answer", None)
            prompts.append(format_mgsm_prompt(q, fewshot))
            golds.append(gold)
            qids.append(int(df.iloc[i].get("id", i)))
        # Run with steering hooks active
        with SteeringHooks(model, layer_indices, V, args.alpha):
            gens = _hf_generate(model, tokenizer, prompts, args.max_new_tokens, args.batch_size, args.greedy)

        # Score
        n_correct = 0
        n_fluent = 0
        for j in range(n):
            gen = gens[j]
            gold_n = None
            try:
                if golds[j] is not None and str(golds[j]).strip() != "":
                    gold_n = float(str(golds[j]).replace(",", ""))
            except Exception:
                gold_n = None
            pred = extract_answer(gen)
            is_correct = numeric_equal(pred, gold_n) if gold_n is not None else False
            fid_pred = None
            fid_score = None
            fid_correct = None
            if glot is not None:
                fid_pred, fid_score = glot.predict(gen)
                fid_correct = int(fid_pred == GLOTLID_MAP.get(lang, ""))
                if fid_correct:
                    n_fluent += 1
            records.append({
                "lang": lang,
                "problem_id": qids[j],
                "question": str(df.iloc[j]["question"]),
                "gold": gold_n,
                "generation": gen,
                "extracted": pred,
                "is_correct": bool(is_correct),
                "glotlid_pred": fid_pred,
                "glotlid_score": fid_score,
                "fidelity_correct": fid_correct,
            })
            if is_correct:
                n_correct += 1
        acc = n_correct / max(n, 1)
        fid = (n_fluent / max(n, 1)) if glot is not None else None
        per_lang_stats[lang] = {"n": n, "accuracy": acc, "fidelity": fid}
        print(f"[m2] {lang}: acc={acc:.3f}  fid={fid if fid is None else round(fid,3)}  (n={n})", flush=True)

    # Aggregate
    accs = [v["accuracy"] for v in per_lang_stats.values()]
    macro_acc = float(np.mean(accs)) if accs else 0.0
    fids = [v["fidelity"] for v in per_lang_stats.values() if v["fidelity"] is not None]
    macro_fid = float(np.mean(fids)) if fids else None

    elapsed = time.time() - t0
    summary = {
        "config": {
            "layer_group": args.layer_group,
            "k_top_excluded": args.k_top_excluded,
            "alpha": args.alpha,
            "rank_r": int(V_np.shape[1]),
            "seed": args.seed,
            "random_subspace_control": bool(args.random_subspace_control),
            "leave_language_out": args.leave_language_out,
            "sites": layer_indices,
            "n_shot": args.n_shot,
            "num_layers": num_layers,
        },
        "per_language": per_lang_stats,
        "macro_accuracy": macro_acc,
        "macro_fidelity": macro_fid,
        "elapsed_seconds": elapsed,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(str(out_path).replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[m2] macro_acc={macro_acc:.4f}  macro_fid={macro_fid}  ({elapsed:.1f}s)", flush=True)
    print(f"[m2] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
