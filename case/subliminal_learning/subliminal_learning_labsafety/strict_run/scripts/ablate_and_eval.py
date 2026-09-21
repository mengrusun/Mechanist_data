"""M2.2a — Ablation on treated student (project-out the top-K d_diff directions
at the top-K language-tower layers).

For each seed:
1. Load the treated student + adapter for that seed.
2. Install forward hooks on the top-K language-tower layers that project out
   the layer-specific `u_diff[l]` from the residual stream (every position).
3. Run QA_I eval + judge.
4. Compute recovery fraction: r = (Acc_ablated - Acc_treated) / (Acc_A - Acc_treated).

Optionally supports a matched-random control direction (--random-direction),
where u_diff is replaced with a random unit vector of equal norm. Used by
M2.2c specificity control.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import (
    PROJECT_ROOT, DATA_ROOT,
    load_processor, load_student_multimodal,
    JudgeCache, call_judge, set_seed,
)
from eval_qa_i import JUDGE_PROMPT_TMPL, parse_judge_verdict, load_qa_i_items, render_multimodal_prompt


def make_projout_hook(u_layer):
    """Return a forward hook that projects OUT the u_layer direction from the
    layer's residual output on every position."""
    u = u_layer  # torch.Tensor [d]
    u = u / u.norm().clamp(min=1e-8)
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]  # [B, T, d]
        else:
            h = output
        u_t = u.to(h.device, dtype=h.dtype)
        # project out
        proj = (h @ u_t).unsqueeze(-1) * u_t  # [B, T, d]
        h_new = h - proj
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        else:
            return h_new
    return hook


def make_add_hook(u_layer, alpha):
    """Return a hook that adds alpha * u_layer to residual (for steering)."""
    u = u_layer / u_layer.norm().clamp(min=1e-8)
    a = float(alpha)
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        u_t = u.to(h.device, dtype=h.dtype)
        h_new = h + a * u_t
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        else:
            return h_new
    return hook


def install_hooks(model, layer_indices, u_by_layer, mode="projout", alpha=1.0):
    """Install forward hooks on model.language_model.layers[l] for each l in
    layer_indices. u_by_layer is a dict mapping l -> tensor(d,).
    Returns a list of hook handles for cleanup.
    """
    lang = None
    # Find the language-model layers container.
    for name, mod in model.named_modules():
        if name.endswith(".language_model") or name == "model.language_model":
            lang = mod
            break
    if lang is None:
        # PEFT wraps modules; try direct navigation.
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)
        lang = getattr(base, "language_model", None)
    if lang is None:
        # Fallback: search by module names for "language_model.layers.<i>"
        raise RuntimeError("Could not locate language_model container in model tree")

    layers = lang.layers  # ModuleList
    handles = []
    for l in layer_indices:
        u = u_by_layer[l]
        if mode == "projout":
            hook = make_projout_hook(u)
        elif mode == "steer":
            hook = make_add_hook(u, alpha)
        else:
            raise ValueError(f"unknown mode {mode}")
        h = layers[l].register_forward_hook(hook)
        handles.append(h)
    return handles


def eval_with_hooks(model, processor, items, judge_cache, max_new_tokens=256):
    """Run QA_I eval and return per-item verdict + accuracy."""
    recs = []
    for it in items:
        inputs = render_multimodal_prompt(processor, it["question"], it["image"])
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        new_start = inputs["input_ids"].shape[1]
        gen_ids = out[0, new_start:]
        answer = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        jp = JUDGE_PROMPT_TMPL.format(
            gold_letter=it["gold"], question=it["question"], model_answer=answer
        )
        raw = call_judge(judge_cache, jp, model="gpt-5.4",
                         temperature=0.0, seed=0, max_tokens=8)
        verdict = parse_judge_verdict(raw)
        recs.append({
            "id": it["id"], "gold": it["gold"], "answer": answer,
            "judge_raw": raw, "verdict": verdict,
            "correct": verdict == "CORRECT",
        })
    n_c = sum(1 for r in recs if r["verdict"] == "CORRECT")
    return recs, n_c / max(1, len(recs)), n_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="Path to treated student adapter for this seed.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--l_core", required=True,
                    help="Path to results/mech/M1_l_core.json.")
    ap.add_argument("--out", required=True,
                    help="Per-item + summary output path.")
    ap.add_argument("--judge_cache",
                    default=str(PROJECT_ROOT / "cache" / "qa_i_judge.jsonl"))
    ap.add_argument("--mode", default="projout", choices=["projout"],
                    help="Ablation mode.")
    ap.add_argument("--random_direction", action="store_true",
                    help="Use a matched-random direction of equal norm instead of u_diff.")
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"[ablate seed={args.seed} mode={args.mode} random={args.random_direction}] loading model", flush=True)
    model = load_student_multimodal()
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = load_processor()

    # Load L-Core directions.
    l_core = json.load(open(args.l_core))
    top_layers = l_core["top_k_layers"]
    # Use this seed's per-layer u_diff.
    seed_dir = l_core["directions_per_seed"][str(args.seed)]
    u_diff_all = np.array(seed_dir["u_diff"])  # [L+1, d]
    v_diff_all = np.array(seed_dir["v_diff"])  # for norm

    # NOTE: hidden_states index is [0..L] where 0=embed, 1..L=post-layer.
    # Language-tower layer indices for hooks are 0..L-1. We map:
    #   hidden_states index l -> language layer index l (l=1..L means post-layer l-1
    #   in decoder-layer index). We treat top_k_layers as hidden_states indices;
    #   convert to layer index by (l - 1) for l >= 1. l = 0 (embed) is not
    #   hookable via layers[]; if the top pick is 0, drop it and warn.
    layer_hook_indices = []
    u_by_layer = {}
    for l_hs in top_layers:
        if l_hs == 0:
            print(f"[ablate] skipping hidden_states index 0 (embedding) "
                  f"— cannot hook via layers[]", flush=True)
            continue
        li = l_hs - 1  # language-model layer index
        vec = u_diff_all[l_hs]
        if args.random_direction:
            rng = np.random.default_rng(args.seed + li * 7)
            r = rng.standard_normal(size=vec.shape).astype(np.float32)
            r /= np.linalg.norm(r) + 1e-8
            r *= np.linalg.norm(v_diff_all[l_hs])  # match magnitude of v_diff
            r /= np.linalg.norm(r) + 1e-8   # since u vs v scaling handled by projout hook
            vec = r
        u_by_layer[li] = torch.tensor(vec, dtype=torch.float32)
        layer_hook_indices.append(li)

    print(f"[ablate] installing {args.mode} hooks on language-model layers "
          f"{layer_hook_indices}", flush=True)
    handles = install_hooks(model, layer_hook_indices, u_by_layer, mode=args.mode)

    items = load_qa_i_items()
    judge_cache = JudgeCache(args.judge_cache)
    try:
        recs, acc, n_c = eval_with_hooks(model, processor, items, judge_cache)
    finally:
        for h in handles:
            h.remove()

    print(f"[ablate seed={args.seed}] ablated_acc={acc:.4f} (n_correct={n_c}/{len(items)})",
          flush=True)

    # Load reference accs.
    ctrl_a = json.load(open(PROJECT_ROOT / "results/eval/Ctrl-A.summary.json"))["overall_acc"]
    treated_ref_summary_path = PROJECT_ROOT / f"results/eval/treated_seed{args.seed}.summary.json"
    ctrlb_ref_summary_path = PROJECT_ROOT / f"results/eval/Ctrl-B_seed{args.seed}.summary.json"
    treated_ref = json.load(open(treated_ref_summary_path))["overall_acc"]
    ctrlb_ref = json.load(open(ctrlb_ref_summary_path))["overall_acc"]
    denom = ctrl_a - treated_ref
    recovery = (acc - treated_ref) / denom if abs(denom) > 1e-8 else 0.0
    print(f"[ablate seed={args.seed}] recovery fraction r = "
          f"({acc:.4f} - {treated_ref:.4f}) / ({ctrl_a:.4f} - {treated_ref:.4f}) "
          f"= {recovery:+.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "seed": args.seed,
            "mode": args.mode,
            "random_direction": args.random_direction,
            "adapter": args.adapter,
            "top_layers_hidden_states_index": top_layers,
            "layer_hook_indices": layer_hook_indices,
            "n": len(items),
            "acc_ablated": acc,
            "n_correct_ablated": n_c,
            "acc_ctrl_a_ref": ctrl_a,
            "acc_treated_ref": treated_ref,
            "acc_ctrlb_ref": ctrlb_ref,
            "recovery_fraction": recovery,
            "per_item": recs,
        }, f, indent=2)
    print(f"[ablate seed={args.seed}] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
