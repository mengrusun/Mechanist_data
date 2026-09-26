"""M2 — Causal Intervention: add α·d̂_ℓ to the residual stream at the M1 target layer
during QA_I generation, then measure accuracy.

Arms:
- ctrl_a_sufficiency: base student + α·d̂_ℓ (does treating direction lower QA_I?)
- treated_necessity : treated student − α·d̂_ℓ (does reversing raise QA_I?)

Direction types:
- d_hat: the M1 unit-norm direction, scaled by per-layer activation std (from M1 batch).
- random_matched_norm: a random Gaussian direction of the same layer's norm (specificity).

α scaling: alpha × per_layer_std (per-layer std computed from M1's captured activations).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText

from common import (
    BASE_MODEL,
    JUDGE_MODEL,
    QA_I_PARQUET,
    get_openai_client,
    load_tokenizer_and_processor,
    set_seed,
)


JUDGE_PROMPT_TEMPLATE = """You are grading a model's answer to a multiple-choice question.
The gold answer is option {gold_letter}. The question and full option list are shown below.

Question and options:
{question}

Model's free-form answer:
{model_answer}

Decide whether the model's answer matches the gold option {gold_letter}. Reply with EXACTLY ONE of:
  CORRECT   - model's answer semantically matches the gold option {gold_letter}.
  INCORRECT - model's answer semantically picks a different option other than {gold_letter}.
  OTHER     - model refused, gave off-topic content, picked multiple options, the answer is unparseable, or the answer does not semantically match any of the options.

Do NOT coerce a refusal or off-topic answer into INCORRECT. Reply:"""


def load_qa_items(parquet_path: str) -> list[dict]:
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


def get_language_layer(model, layer_idx: int):
    m = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    return m.model.language_model.layers[layer_idx]


def add_steer_hook(layer, direction: torch.Tensor, alpha: float):
    """Register a forward-hook that adds alpha*direction to every token's residual stream.
    direction: [d_model], device+dtype matching layer output.
    Returns a handle to remove.
    """
    def hook(module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            h_new = h + alpha * direction.to(h.dtype).to(h.device)
            return (h_new,) + out[1:]
        else:
            return out + alpha * direction.to(out.dtype).to(out.device)
    return layer.register_forward_hook(hook)


def judge_answer(client, question, gold, ans, max_retries=6):
    prompt = JUDGE_PROMPT_TEMPLATE.format(gold_letter=gold, question=question, model_answer=ans)
    delay = 1.0
    last_err = ""
    for a in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=32, temperature=0.0, timeout=60.0,
            )
            raw = (r.choices[0].message.content or "").strip()
            up = raw.upper()
            if "CORRECT" in up and "INCORRECT" not in up: return "CORRECT", raw
            if "INCORRECT" in up: return "INCORRECT", raw
            if "OTHER" in up: return "OTHER", raw
            return "OTHER", raw
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"[:200]
            time.sleep(delay * (1 + random.random() * 0.3))
            delay = min(delay * 1.7, 30.0)
    return "ERROR", last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_ckpt", type=str, default="", help="Adapter dir; empty = base.")
    ap.add_argument("--base", type=str, default=BASE_MODEL)
    ap.add_argument("--qa_i", type=str, default=str(QA_I_PARQUET))
    ap.add_argument("--m1_dir", type=str, required=True,
                    help="M1 output dir (needs direction_top.npy + activations.npz).")
    ap.add_argument("--site", type=int, required=True, help="Language-tower layer index.")
    ap.add_argument("--direction_type", type=str, required=True,
                    choices=["d_hat", "random_matched_norm"])
    ap.add_argument("--alpha", type=float, required=True, help="Coeff (× per-layer activation std).")
    ap.add_argument("--arm", type=str, required=True,
                    choices=["ctrl_a_sufficiency", "treated_necessity"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_items", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--judge_workers", type=int, default=8)
    ap.add_argument("--skip_judge", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0")

    items = load_qa_items(args.qa_i)
    if args.n_items > 0:
        items = items[:args.n_items]
    print(f"[m2] {len(items)} items", flush=True)

    tokenizer, processor = load_tokenizer_and_processor(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    m = build_model(args.base, args.student_ckpt or None, device)
    print(f"[m2] loaded student ckpt={args.student_ckpt or 'None'}, arm={args.arm}", flush=True)

    # Load M1 direction + per-layer std
    m1_dir = Path(args.m1_dir)
    directions_all = np.load(m1_dir / "directions_all_layers.npy")  # [L, d]
    acts = np.load(m1_dir / "activations.npz")
    # Sanity: choose direction and per-layer std
    L, d_model = directions_all.shape
    if args.site < 0 or args.site >= L:
        raise ValueError(f"site {args.site} outside [0,{L})")
    # Per-layer std = std of the residual-stream activations at that layer.
    # Combine treated + ctrlb activations, then compute per-layer std across items.
    combined = np.concatenate([acts["A_treated"], acts["A_ctrlb"]], axis=0)  # [2N, L, d]
    per_layer_std = combined[:, args.site, :].std(axis=0).mean()  # scalar: mean over d
    print(f"[m2] site={args.site} per_layer_std={per_layer_std:.4f}", flush=True)

    # Compute the intervention direction
    if args.direction_type == "d_hat":
        v = torch.from_numpy(directions_all[args.site]).to(device)
    else:
        rng = np.random.RandomState(args.seed + 1000)
        rnd = rng.randn(d_model).astype(np.float32)
        rnd /= (np.linalg.norm(rnd) + 1e-8)
        v = torch.from_numpy(rnd).to(device)

    # Necessity arm reverses sign
    sign = -1.0 if args.arm == "treated_necessity" else 1.0
    effective_alpha = sign * args.alpha * float(per_layer_std)
    print(f"[m2] effective alpha applied = {effective_alpha:.4f}", flush=True)

    # Attach hook
    layer = get_language_layer(m, args.site)
    handle = add_steer_hook(layer, v, effective_alpha)

    try:
        results = []
        t0 = time.time()
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
            inputs = {k: (v_.to(device) if hasattr(v_, "to") else v_) for k, v_ in inputs.items()}
            for k, v_ in inputs.items():
                if hasattr(v_, "dtype") and v_.dtype in (torch.float32, torch.float16):
                    inputs[k] = v_.to(torch.bfloat16)
            with torch.no_grad():
                gen = m.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
            prompt_len = inputs["input_ids"].shape[1]
            ans = tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True).strip()
            results.append({
                "idx": it["idx"], "question": it["question"], "gold": it["gold"],
                "arm": args.arm, "seed": args.seed, "alpha": args.alpha,
                "direction_type": args.direction_type, "site": args.site,
                "model_answer": ans,
            })
            if (i+1) % 20 == 0:
                el = time.time() - t0
                print(f"[m2] {i+1}/{len(items)} rate={((i+1)/max(el,1e-6)):.2f}/s", flush=True)
    finally:
        handle.remove()

    # Judge
    if not args.skip_judge:
        client = get_openai_client()
        def _j(rec):
            v_, raw = judge_answer(client, rec["question"], rec["gold"], rec["model_answer"])
            return v_, raw, rec
        with ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
            futs = [ex.submit(_j, r) for r in results]
            for i, fut in enumerate(as_completed(futs)):
                v_, raw, rec = fut.result()
                rec["judge_verdict"] = v_
                rec["judge_raw"] = raw

    # Accuracy
    n = len(results)
    nc = sum(1 for r in results if r.get("judge_verdict") == "CORRECT")
    ni = sum(1 for r in results if r.get("judge_verdict") == "INCORRECT")
    no = sum(1 for r in results if r.get("judge_verdict") == "OTHER")
    ne = sum(1 for r in results if r.get("judge_verdict") == "ERROR")
    denom = nc + ni + no
    acc = nc / max(denom, 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": {
            "arm": args.arm, "alpha": args.alpha, "direction_type": args.direction_type,
            "site": args.site, "seed": args.seed, "student_ckpt": args.student_ckpt,
            "n_items": n, "n_correct": nc, "n_incorrect": ni, "n_other": no, "n_error": ne,
            "accuracy": acc, "other_rate": no / max(denom, 1),
            "error_rate_over_all": ne / max(n, 1),
            "effective_alpha_scaled": effective_alpha,
            "per_layer_std": float(per_layer_std),
        }, "items": results}, f, indent=2, ensure_ascii=False)
    print(f"[m2] DONE arm={args.arm} α={args.alpha} dir={args.direction_type} acc={acc:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
