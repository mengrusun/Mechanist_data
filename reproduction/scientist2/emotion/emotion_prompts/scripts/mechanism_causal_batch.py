#!/usr/bin/env python3
"""M6: Causal Intervention — BATCHED across many (intervention, target, site, α) combos
in ONE model load. Uses transformers (not vLLM) for intra-forward residual-stream hooks.

Reads a JSON manifest of runs; writes runs/M6/<run_id>.json for each entry.

Manifest format:
[
  {"run_id": "patch_emo_L4_happiness2h", "intervention": "patch", "target": "emotional",
   "site_layer": 4, "emotion_pair": "happiness_2_human", "alpha": 0.0, "direction_idx": 0},
  ...
]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# Prompt utilities
GSM8K_INSTRUCTION = (
    "Solve the math problem step by step, then write the final answer as an integer "
    'after a "####" marker on a new line. Example: "#### 42".\n\n'
)
_GSM_RE = re.compile(r"####\s*(-?[\d,]+)")


def parse_gsm8k(text: str) -> Optional[str]:
    m = _GSM_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    ints = re.findall(r"-?\d+", text.replace(",", ""))
    return ints[-1] if ints else None


def build_gsm8k_prompt(prefix: str, item: Dict) -> str:
    return (
        prefix.strip() + "\n\n"
        + GSM8K_INSTRUCTION
        + "Problem: " + item["question"].strip() + "\n"
        + "Solution:"
    )


def build_medqa_prompt(prefix: str, item: Dict) -> str:
    opts_text = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    body = f"Question: {item['question'].strip()}\nOptions:\n{opts_text}\n"
    return (
        prefix.strip() + "\n\n"
        + "Read the question and options; then answer with just the single option letter.\n\n"
        + body + "Answer:"
    )


def load_gsm8k(n_items: int) -> List[Dict]:
    from datasets import load_from_disk
    ds = load_from_disk("/data/zhenqian/data/gsm8k")["test"]
    out = []
    for i, r in enumerate(ds):
        if i >= n_items:
            break
        m = re.search(r"####\s*(-?[\d,]+)", r["answer"])
        gold = m.group(1).replace(",", "") if m else None
        out.append({"item_id": i, "question": r["question"], "gold": gold})
    return out


def load_medqa(n_items: int) -> List[Dict]:
    path = "/data/zhenqian/data/med_qa/data_clean/questions/US/test.jsonl"
    out = []
    for i, ln in enumerate(open(path)):
        if i >= n_items:
            break
        r = json.loads(ln)
        out.append({"item_id": i, "question": r["question"],
                    "options": r["options"], "gold": r["answer_idx"]})
    return out


class HookedModel:
    def __init__(self, model_dir: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        self.decoder_layers = self.model.model.layers
        self.n_layers = len(self.decoder_layers)
        self._hook_handles = []

    def clear_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def add_residual_write_hook(self, layer_idx: int, write_fn):
        layer = self.decoder_layers[layer_idx]

        def pre_hook(mod, args, kwargs):
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
        attn_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=input_ids, attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_ids = gen[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return {"text": text, "n_gen_tokens": int(gen_ids.shape[0])}

    def mcq_loglik(self, prompt: str, letters: List[str]) -> Dict:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        letter_lp = {}
        for letter in letters:
            for cand in [f" {letter}", letter]:
                tids = self.tokenizer.encode(cand, add_special_tokens=False)
                if len(tids) == 1:
                    lp = log_probs[tids[0]].item()
                    letter_lp[letter] = max(letter_lp.get(letter, -1e9), lp)
                    break
            else:
                letter_lp[letter] = -1e9
        del out
        gc.collect(); torch.cuda.empty_cache()
        pred = max(letter_lp, key=letter_lp.get)
        return {"pred": pred, "letter_lp": letter_lp}


def get_source_activation(acts_dir: str, task: str, condition_id: str,
                          layer: int, item_id: int) -> Optional[torch.Tensor]:
    if task == "gsm8k":
        path = os.path.join(acts_dir, f"act_gsm8k_{condition_id}.pt")
    else:
        path = os.path.join(acts_dir, f"act_{task}_{condition_id}.pt")
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    if layer not in d["acts"]:
        return None
    item_ids = d.get("item_ids", list(range(d["acts"][layer].shape[0])))
    try:
        idx = item_ids.index(item_id)
    except ValueError:
        return None
    return d["acts"][layer][idx]


def make_write_fn(pos, mode, src, alpha, sigma, dvec):
    dvec_dev = None
    src_dev = None

    def write_fn(h):
        nonlocal dvec_dev, src_dev
        if h.shape[1] <= pos:
            return h
        if dvec_dev is None:
            dvec_dev = dvec.to(h.device).to(h.dtype)
            if src is not None:
                src_dev = src.to(h.device).to(h.dtype)
        if mode == "steer":
            delta = alpha * sigma * dvec_dev
            h = h.clone()
            h[0, pos, :] = h[0, pos, :] + delta
            return h
        elif mode == "patch":
            if src_dev is None:
                return h
            h = h.clone()
            h_pos = h[0, pos, :]
            proj_dst = torch.dot(h_pos, dvec_dev)
            proj_src = torch.dot(src_dev, dvec_dev)
            delta = (proj_src - proj_dst) * dvec_dev
            h[0, pos, :] = h_pos + delta
            return h
        return h
    return write_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Qwen3-14B")
    ap.add_argument("--directions", default="reports/M5_directions.pt")
    ap.add_argument("--prefixes_json", default="data/prefixes/prefixes.json")
    ap.add_argument("--activations_gsm8k", default="runs/M2")
    ap.add_argument("--activations_medqa", default="runs/M3")
    ap.add_argument("--manifest", required=True, help="JSON list of runs")
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--out_dir", default="runs/M6")
    ap.add_argument("--gpu_ids", default=os.environ.get("CUDA_VISIBLE_DEVICES", "auto"))
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(args.manifest))
    print(f"[batch] {len(manifest)} runs")

    dir_data = torch.load(args.directions, map_location="cpu", weights_only=False)
    print(f"[dirs] layers={list(dir_data['directions'].keys())}")

    prefixes = json.load(open(args.prefixes_json))
    prefix_by_cid = {r["condition_id"]: r["text"] for r in prefixes}

    # Load model once
    t_load0 = time.time()
    print(f"[load] Qwen3-14B ...")
    hm = HookedModel(args.model)
    print(f"[load] done in {time.time()-t_load0:.1f}s")

    # Load neutral-condition activations for σ estimation
    neutral_acts_gsm = torch.load(
        os.path.join(args.activations_gsm8k, "act_gsm8k_neutral.pt"),
        map_location="cpu", weights_only=False,
    )

    for i, run in enumerate(manifest):
        run_id = run["run_id"]
        out_json = f"{args.out_dir}/{run_id}.json"
        if os.path.exists(out_json):
            try:
                d = json.load(open(out_json))
                if "accuracy" in d:
                    print(f"[skip] {run_id} already done acc={d['accuracy']:.4f}")
                    continue
            except Exception:
                pass

        intervention = run["intervention"]
        target = run["target"]
        site_layer = run["site_layer"]
        alpha = run.get("alpha", 0.0)
        emo_cid = run.get("emotion_pair", "happiness_2_human")
        direction_idx = run.get("direction_idx", 0)

        # Task
        if target == "offtarget_medqa":
            items = load_medqa(args.n_items)
            task = "medqa"
        else:
            items = load_gsm8k(args.n_items)
            task = "gsm8k"

        # Get direction at this layer
        if site_layer not in dir_data["directions"]:
            print(f"[skip] {run_id} — no direction at layer {site_layer}")
            continue
        d_frame = dir_data["directions"][site_layer][direction_idx].float()
        d_frame_norm = d_frame / (d_frame.norm() + 1e-9)

        # σ estimation from neutral acts at this layer
        sigma_l = 1.0
        if site_layer in neutral_acts_gsm["acts"]:
            neutral_acts = neutral_acts_gsm["acts"][site_layer].float()
            proj = (neutral_acts @ d_frame_norm).cpu().numpy()
            sigma_l = float(proj.std())

        # Run prefix (the actual prompt uses the emotional prefix)
        run_prefix_text = prefix_by_cid[emo_cid]

        # Source activation for patch
        def get_src(item_id: int) -> Optional[torch.Tensor]:
            if intervention != "patch":
                return None
            if target == "filler_control":
                src_cid = "filler_matched_length"
            else:
                src_cid = run.get("source_condition", emo_cid)
            return get_source_activation(
                args.activations_gsm8k if task == "gsm8k" else args.activations_medqa,
                task, src_cid, site_layer, item_id,
            )

        t0 = time.time()
        results = []
        for k, it in enumerate(items):
            if task == "gsm8k":
                prompt = build_gsm8k_prompt(run_prefix_text, it)
            else:
                prompt = build_medqa_prompt(run_prefix_text, it)
            last_pos = hm.get_last_prefix_pos(prompt, run_prefix_text)
            src_act = get_src(it["item_id"])

            hm.clear_hooks()
            hm.add_residual_write_hook(
                site_layer,
                make_write_fn(last_pos, intervention, src_act, alpha, sigma_l, d_frame_norm),
            )
            try:
                if task == "gsm8k":
                    resp = hm.generate_greedy(prompt, max_new_tokens=512)
                    pred = parse_gsm8k(resp["text"])
                    gold = it["gold"]
                    try:
                        correct = int(pred is not None and gold is not None and float(pred) == float(gold))
                    except Exception:
                        correct = 0
                    results.append({
                        "item_id": it["item_id"], "gold": gold, "pred": pred,
                        "correct": correct, "parse_ok": int(pred is not None),
                        "n_gen_tokens": resp["n_gen_tokens"],
                    })
                else:
                    letters = list(it["options"].keys())
                    resp = hm.mcq_loglik(prompt, letters)
                    results.append({
                        "item_id": it["item_id"], "gold": it["gold"], "pred": resp["pred"],
                        "correct": int(resp["pred"] == it["gold"]),
                        "letter_lp": resp["letter_lp"],
                    })
            except Exception as e:
                results.append({"item_id": it["item_id"], "correct": 0, "error": str(e)})
            finally:
                hm.clear_hooks()

        elapsed = time.time() - t0
        n_correct = sum(r["correct"] for r in results)
        acc = n_correct / len(results) if results else 0.0
        parse_rate = sum(r.get("parse_ok", 1) for r in results) / max(1, len(results))
        mean_gen = float(np.mean([r.get("n_gen_tokens", 0) for r in results]))
        print(f"[done {i+1}/{len(manifest)}] {run_id} acc={acc:.4f} parse={parse_rate:.3f} "
              f"gen_tok={mean_gen:.0f} sigma={sigma_l:.3f} time={elapsed:.1f}s")

        summary = {
            "run_id": run_id,
            "intervention": intervention, "target": target, "site_layer": site_layer,
            "alpha": alpha, "sigma_l": sigma_l, "direction_idx": direction_idx,
            "emotion_pair": emo_cid, "task": task,
            "n_items": len(items),
            "accuracy": acc, "parse_rate": parse_rate, "mean_gen_tokens": mean_gen,
            "elapsed_seconds": elapsed,
            "per_item": results,
        }
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=1)

    # cost.json
    cost_path = f"{args.out_dir}/cost.json"
    gpu_ids_list = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids and args.gpu_ids != "auto" else []
    old = {}
    if os.path.exists(cost_path):
        try:
            old = json.load(open(cost_path))
        except Exception:
            old = {}
    if not isinstance(old, dict):
        old = {}
    # For batch, log a single entry with total time
    print("[batch] all done")


if __name__ == "__main__":
    main()
