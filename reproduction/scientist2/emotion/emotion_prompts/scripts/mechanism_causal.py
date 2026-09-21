#!/usr/bin/env python3
"""M6: Causal Intervention — Activation Patching + Steering + specificity controls.

Uses transformers (not vLLM) for intra-forward residual-stream hooks.

Interventions:
- patch: replace last-prefix-token residual at target layer with (proj of emotional
  activation onto d_frame) + (orthogonal component of neutral activation).
- steer: add α · σ_l · d_frame at target layer's last-prefix-token residual
  (β expressed in σ_proj units).

Targets:
- emotional: the actual emotional-prefix activation as source
- filler_control: length-matched filler activation as source (should give Δ ≈ 0)
- offtarget_medqa: run steering on MedQA items (should give small effect)

Sites:
- L_frame_top1, L_frame_top2 from M5_directions.pt
- (Optional) an off-layer null control layer for regional-claim test.

Output: runs/M6/<intervention>_<target>_<site>_a<alpha>.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# Re-use prompt builders from run_prefix_eval.py
from run_prefix_eval import (  # type: ignore
    build_gsm8k_prompt,
    build_mcq_prompt,
    load_gsm8k,
    load_medqa,
    parse_gsm8k,
)


class HookedModel:
    """HF model wrapper with residual-stream write hooks at chosen layers."""

    def __init__(self, model_dir: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.decoder_layers = self.model.model.layers
        else:
            raise RuntimeError("Unknown decoder structure")
        self.n_layers = len(self.decoder_layers)
        self._hook_handles = []

    def _clean(self):
        gc.collect()
        torch.cuda.empty_cache()

    def clear_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def add_residual_write_hook(
        self,
        layer_idx: int,
        write_fn,
    ):
        """write_fn(residual: tensor[batch, seq, d]) -> tensor[batch, seq, d].

        Hooks are registered on layer input (pre-block residual).
        We use forward_pre_hook to modify the input to the layer.
        """
        layer = self.decoder_layers[layer_idx]

        def pre_hook(mod, args, kwargs):
            # args[0] is hidden_states typically
            new_args = list(args)
            if len(new_args) > 0:
                new_args[0] = write_fn(new_args[0])
            return tuple(new_args), kwargs

        h = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        self._hook_handles.append(h)

    def get_last_prefix_pos(self, prompt: str, prefix_end_marker: str) -> int:
        prefix_end_pos = prompt.find(prefix_end_marker) + len(prefix_end_marker)
        prefix_prefix = prompt[:prefix_end_pos]
        prefix_token_ids = self.tokenizer(prefix_prefix, return_tensors="pt").input_ids[0]
        return len(prefix_token_ids) - 1

    def generate_greedy(self, prompt: str, max_new_tokens: int = 256) -> Dict:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = gen[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return {"text": text, "n_gen_tokens": int(gen_ids.shape[0])}

    def mcq_loglik(self, prompt: str, option_letters: List[str]) -> Dict:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        letter_lp = {}
        for letter in option_letters:
            candidates = [f" {letter}", letter]
            best_lp = -float("inf")
            for cand in candidates:
                tok_ids = self.tokenizer.encode(cand, add_special_tokens=False)
                if len(tok_ids) == 1:
                    lp = log_probs[tok_ids[0]].item()
                    if lp > best_lp:
                        best_lp = lp
            letter_lp[letter] = best_lp
        del out
        self._clean()
        pred = max(letter_lp, key=letter_lp.get)
        return {"pred": pred, "letter_lp": letter_lp}


def get_source_activation(
    acts_dir: str, task: str, condition_id: str, layer: int, item_id: int
) -> Optional[torch.Tensor]:
    """Load the cached activation for a specific (task, condition, layer, item)."""
    path = os.path.join(acts_dir, f"act_{task}_{condition_id}.pt") if task != "gsm8k" \
        else os.path.join(acts_dir, f"act_{condition_id}.pt")
    if not os.path.exists(path):
        return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if layer not in d["acts"]:
        return None
    item_ids = d.get("item_ids", list(range(d["acts"][layer].shape[0])))
    try:
        idx = item_ids.index(item_id)
    except ValueError:
        return None
    return d["acts"][layer][idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Qwen3-14B")
    ap.add_argument("--directions", default="reports/M5_directions.pt")
    ap.add_argument("--prefixes_json", default="data/prefixes/prefixes.json")
    ap.add_argument("--activations_gsm8k", default="runs/M2")
    ap.add_argument("--activations_medqa", default="runs/M3")
    ap.add_argument("--intervention", choices=["patch", "steer"], required=True)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="Steering coefficient in σ_proj units (ignored for patch)")
    ap.add_argument("--target", choices=["emotional", "filler_control", "offtarget_medqa"], required=True)
    ap.add_argument("--site_layer", type=int, required=True)
    ap.add_argument("--source_condition", default=None,
                    help="Condition_id for the source activation (only used by patch)")
    ap.add_argument("--emotion_pair", default=None,
                    help="Emotion identifier, e.g. 'happiness_2_human'; used by both patch and steer")
    ap.add_argument("--direction_idx", type=int, default=0,
                    help="Which SVD singular direction to use (0..2)")
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_ids", default=os.environ.get("CUDA_VISIBLE_DEVICES", "auto"))
    args = ap.parse_args()

    print(f"[cfg] intervention={args.intervention} target={args.target} "
          f"site_layer={args.site_layer} alpha={args.alpha}")

    # Load direction tensor
    dir_data = torch.load(args.directions, map_location="cpu", weights_only=False)
    if args.site_layer not in dir_data["directions"]:
        raise RuntimeError(f"Layer {args.site_layer} not in M5 directions "
                           f"(available: {list(dir_data['directions'].keys())})")
    d_frame = dir_data["directions"][args.site_layer][args.direction_idx].float()  # [d]

    # Load items
    if args.target == "offtarget_medqa":
        items = load_medqa(args.n_items)
        task = "medqa"
    else:
        items = load_gsm8k(args.n_items)
        task = "gsm8k"

    # Prefix text: use the emotional source for the run's prefix
    prefixes = json.load(open(args.prefixes_json))
    prefix_by_cid = {r["condition_id"]: r["text"] for r in prefixes}

    # Determine "run prefix" (what prompt goes into the model at eval time):
    # - patch/steer with target=emotional: run under the emotional prefix
    # - filler_control: run under the emotional prefix but source activation from filler
    # - offtarget_medqa: run under emotional prefix (or neutral if steer, up to caller)
    emo_cid = args.emotion_pair or "happiness_2_human"
    run_prefix_text = prefix_by_cid[emo_cid]

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[load] Qwen3-14B ...")
    hm = HookedModel(args.model)

    # Estimate sigma_l for the direction at the site layer (over collected activations)
    # by loading cached emotional activations and projecting.
    acts_dir = args.activations_gsm8k if task == "gsm8k" else args.activations_medqa
    # Use neutral condition activations to compute σ_proj
    neutral_path = os.path.join(acts_dir, f"act_neutral.pt") if task == "gsm8k" \
        else os.path.join(acts_dir, f"act_{task}_neutral.pt")
    sigma_l = 1.0
    if os.path.exists(neutral_path):
        d = torch.load(neutral_path, map_location="cpu", weights_only=False)
        if args.site_layer in d["acts"]:
            neutral_acts = d["acts"][args.site_layer].float()  # [n_items, d]
            proj = (neutral_acts @ d_frame).cpu().numpy()
            sigma_l = float(proj.std())
            print(f"[sigma] σ_proj at layer {args.site_layer} = {sigma_l:.3f}")

    # Determine source activation for patching
    def get_source_act_for_item(item_id: int) -> Optional[torch.Tensor]:
        if args.intervention != "patch":
            return None
        if args.target == "filler_control":
            src_cid = "filler_matched_length"
        else:
            src_cid = args.source_condition or emo_cid
        return get_source_activation(acts_dir, task, src_cid, args.site_layer, item_id)

    # Prepare eval outputs
    results = []
    d_frame_norm = d_frame / (d_frame.norm() + 1e-9)  # unit direction

    for k, it in enumerate(items):
        if task == "gsm8k":
            prompt = build_gsm8k_prompt(run_prefix_text, it)
        else:
            prompt = build_mcq_prompt(run_prefix_text, it, "medqa")
        last_pos = hm.get_last_prefix_pos(prompt, run_prefix_text)

        # Set up hook
        src_act = get_source_act_for_item(it["item_id"]) if args.intervention == "patch" else None

        def make_write_fn(pos, mode, src, alpha, sigma, dvec):
            dvec_dev = None
            src_dev = None

            def write_fn(h):
                nonlocal dvec_dev, src_dev
                # h: [batch=1, seq, d]
                if h.shape[1] <= pos:
                    return h
                if dvec_dev is None:
                    dvec_dev = dvec.to(h.device).to(h.dtype)
                    if src is not None:
                        src_dev = src.to(h.device).to(h.dtype)
                if mode == "steer":
                    delta = alpha * sigma * dvec_dev  # [d]
                    h = h.clone()
                    h[0, pos, :] = h[0, pos, :] + delta
                    return h
                elif mode == "patch":
                    if src_dev is None:
                        return h
                    # Replace the frame-direction component
                    h = h.clone()
                    h_pos = h[0, pos, :]
                    proj_dst = torch.dot(h_pos, dvec_dev)
                    proj_src = torch.dot(src_dev, dvec_dev)
                    delta = (proj_src - proj_dst) * dvec_dev
                    h[0, pos, :] = h_pos + delta
                    return h
                return h
            return write_fn

        hm.clear_hooks()
        hm.add_residual_write_hook(
            args.site_layer,
            make_write_fn(last_pos, args.intervention, src_act, args.alpha, sigma_l, d_frame_norm),
        )

        # Run
        try:
            if task == "gsm8k":
                resp = hm.generate_greedy(prompt, max_new_tokens=256)
                pred = parse_gsm8k(resp["text"])
                gold = it["gold"]
                try:
                    correct = int(pred is not None and gold is not None and float(pred) == float(gold))
                except Exception:
                    correct = 0
                results.append({
                    "item_id": it["item_id"],
                    "gold": gold, "pred": pred, "correct": correct,
                    "parse_ok": int(pred is not None),
                    "n_gen_tokens": resp["n_gen_tokens"],
                })
            else:
                letters = list(it["options"].keys())
                resp = hm.mcq_loglik(prompt, letters)
                results.append({
                    "item_id": it["item_id"],
                    "gold": it["gold"], "pred": resp["pred"],
                    "correct": int(resp["pred"] == it["gold"]),
                    "letter_lp": resp["letter_lp"],
                })
        except Exception as e:
            print(f"  [item {it['item_id']}] error: {e}")
            results.append({"item_id": it["item_id"], "correct": 0, "error": str(e)})
        finally:
            hm.clear_hooks()

        if (k + 1) % 50 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{k+1}/{len(items)}] acc={acc:.4f}")

    hm.clear_hooks()
    elapsed = time.time() - t0
    n_correct = sum(r["correct"] for r in results)
    n_valid = len(results)
    acc = n_correct / n_valid if n_valid else 0.0
    parse_rate = sum(r.get("parse_ok", 1) for r in results) / max(1, n_valid)
    mean_gen_tokens = np.mean([r.get("n_gen_tokens", 0) for r in results])
    print(f"[done] acc={acc:.4f} parse_rate={parse_rate:.3f} mean_gen_tokens={mean_gen_tokens:.1f} time={elapsed:.1f}s")

    summary = {
        "intervention": args.intervention,
        "target": args.target,
        "site_layer": args.site_layer,
        "alpha": args.alpha,
        "sigma_l": sigma_l,
        "direction_idx": args.direction_idx,
        "emotion_pair": args.emotion_pair,
        "run_prefix_condition": emo_cid,
        "task": task,
        "n_items": len(items),
        "accuracy": acc,
        "parse_rate": parse_rate,
        "mean_gen_tokens": float(mean_gen_tokens),
        "elapsed_seconds": elapsed,
        "per_item": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[write] {args.out}")

    # cost.json
    run_dir = os.path.dirname(args.out)
    cost_path = os.path.join(run_dir, "cost.json")
    gpu_ids_list = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids and args.gpu_ids != "auto" else []
    cost = {
        "run_id": os.path.basename(args.out).replace(".json", ""),
        "gpu_ids": gpu_ids_list,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
    }
    old = {}
    if os.path.exists(cost_path):
        try:
            old = json.load(open(cost_path))
        except Exception:
            old = {}
    if not isinstance(old, dict):
        old = {}
    old[cost["run_id"]] = cost
    with open(cost_path, "w") as f:
        json.dump(old, f, indent=1)


if __name__ == "__main__":
    main()
