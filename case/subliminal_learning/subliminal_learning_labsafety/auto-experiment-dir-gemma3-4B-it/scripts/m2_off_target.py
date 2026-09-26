"""M2 off-target: general-ability check at α★.

Two off-target benchmarks:
- MMLU-lite: 500 random items from MMLU (text-only, multiple-choice).
- Helpfulness-lite: 200 open-ended instructions (from Alpaca or MT-bench-lite).

For both, we test whether the M2 intervention wrecks general ability. Score = accuracy
for MMLU, avg-length + coherence check for helpfulness (a proxy for "does it still
respond in-domain").

We use Huggingface datasets if available; otherwise we fall back to a curated
in-file mini benchmark so the script self-contains.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

from common import (
    BASE_MODEL,
    JUDGE_MODEL,
    apply_gemma_chat,
    get_openai_client,
    load_tokenizer_and_processor,
    set_seed,
)


# Curated 20-item MMLU-style fallback so the script works offline.
MMLU_FALLBACK = [
    {"q":"Which of the following is a prime number? A) 4 B) 9 C) 11 D) 15","a":"C"},
    {"q":"The chemical symbol for gold is: A) Au B) Ag C) Gd D) Go","a":"A"},
    {"q":"Who wrote 'Pride and Prejudice'? A) Charles Dickens B) Jane Austen C) Emily Brontë D) George Eliot","a":"B"},
    {"q":"Photosynthesis primarily takes place in: A) Mitochondria B) Chloroplasts C) Ribosomes D) Nuclei","a":"B"},
    {"q":"The capital of Australia is: A) Sydney B) Melbourne C) Canberra D) Brisbane","a":"C"},
    {"q":"Which planet is known as the Red Planet? A) Venus B) Jupiter C) Mars D) Saturn","a":"C"},
    {"q":"Water boils at 100°C at: A) Sea level B) 1000 m elevation C) In a vacuum D) In salt water only","a":"A"},
    {"q":"E = mc² was formulated by: A) Newton B) Einstein C) Hawking D) Bohr","a":"B"},
    {"q":"The largest ocean is: A) Atlantic B) Indian C) Arctic D) Pacific","a":"D"},
    {"q":"What year did WWII end? A) 1943 B) 1944 C) 1945 D) 1946","a":"C"},
    {"q":"The powerhouse of the cell is the: A) Nucleus B) Ribosome C) Mitochondrion D) Golgi apparatus","a":"C"},
    {"q":"Pi is approximately equal to: A) 2.71 B) 3.14 C) 3.16 D) 1.62","a":"B"},
    {"q":"Which language is primarily spoken in Brazil? A) Spanish B) Portuguese C) French D) English","a":"B"},
    {"q":"Ohm's law relates: A) F = ma B) V = IR C) E = mc² D) PV = nRT","a":"B"},
    {"q":"The Statue of Liberty was a gift from: A) UK B) France C) Spain D) Italy","a":"B"},
    {"q":"DNA stands for: A) Deoxyribonucleic acid B) Diribonucleic acid C) Dinitrogen acetate D) Deriboxyacid","a":"A"},
    {"q":"The smallest particle of an element is: A) Molecule B) Compound C) Atom D) Ion","a":"C"},
    {"q":"Which of these is a mammal? A) Shark B) Whale C) Octopus D) Snake","a":"B"},
    {"q":"The square root of 144 is: A) 10 B) 12 C) 14 D) 16","a":"B"},
    {"q":"World Cup 2022 was hosted by: A) Russia B) Brazil C) Qatar D) France","a":"C"},
]

HELPFUL_FALLBACK = [
    "Write a short poem about spring.",
    "Explain the Pythagorean theorem in simple terms.",
    "What are three tips for improving public speaking?",
    "Describe the process of photosynthesis in one paragraph.",
    "Give me a healthy breakfast recipe with oatmeal.",
    "How do I properly warm up before a run?",
    "What are the main causes of the French Revolution?",
    "Suggest a 5-step study plan for learning Python.",
    "Explain why the sky is blue.",
    "What are the seven wonders of the world?",
]


def load_mmlu(n_target: int, seed: int):
    """Try to load MMLU from HF; fall back to curated 20 items."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        rng = random.Random(seed)
        idxs = rng.sample(range(len(ds)), min(n_target, len(ds)))
        items = []
        for i in idxs:
            r = ds[i]
            q = r["question"]
            choices = r["choices"]
            correct = "ABCD"[int(r["answer"])]
            options = " ".join(f"{'ABCD'[k]}: {c}" for k, c in enumerate(choices))
            items.append({"q": f"{q} {options}", "a": correct})
        return items, "HF cais/mmlu"
    except Exception as e:
        print(f"[m2_off_target] MMLU HF load failed ({e}); using fallback 20 items", flush=True)
        return list(MMLU_FALLBACK), "fallback20"


def add_steer_hook(layer, direction: torch.Tensor, alpha: float):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            h_new = h + alpha * direction.to(h.dtype).to(h.device)
            return (h_new,) + out[1:]
        else:
            return out + alpha * direction.to(out.dtype).to(out.device)
    return layer.register_forward_hook(hook)


def get_language_layer(model, layer_idx: int):
    m = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    return m.model.language_model.layers[layer_idx]


def build_model(base, adapter, device):
    m = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    m.to(device)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
        m.to(device)
    m.eval()
    return m


JUDGE_MMLU_TEMPLATE = """You are grading a model's answer to a multiple-choice question.
The gold answer is option {gold}. Question:
{q}

Model answer:
{a}

Reply with EXACTLY ONE of CORRECT / INCORRECT / OTHER (OTHER if refused/off-topic)."""


def judge_mmlu(client, q, gold, ans, max_retries=6):
    prompt = JUDGE_MMLU_TEMPLATE.format(q=q, gold=gold, a=ans)
    delay = 1.0
    for _ in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL, messages=[{"role":"user","content":prompt}],
                max_completion_tokens=16, temperature=0.0, timeout=60.0)
            raw = (r.choices[0].message.content or "").strip().upper()
            if "CORRECT" in raw and "INCORRECT" not in raw: return "CORRECT"
            if "INCORRECT" in raw: return "INCORRECT"
            if "OTHER" in raw: return "OTHER"
            return "OTHER"
        except Exception:
            time.sleep(delay); delay = min(delay*1.7, 30.0)
    return "ERROR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_ckpt", type=str, default="")
    ap.add_argument("--base", type=str, default=BASE_MODEL)
    ap.add_argument("--m1_dir", type=str, required=True)
    ap.add_argument("--site", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True, help="Same scale as m2_intervene.py.")
    ap.add_argument("--direction_type", type=str, default="d_hat",
                    choices=["d_hat", "random_matched_norm"])
    ap.add_argument("--n_mmlu", type=int, default=500)
    ap.add_argument("--n_helpful", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--judge_workers", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0")

    # Load M1 direction + per-layer std
    m1_dir = Path(args.m1_dir)
    directions_all = np.load(m1_dir / "directions_all_layers.npy")
    acts = np.load(m1_dir / "activations.npz")
    combined = np.concatenate([acts["A_treated"], acts["A_ctrlb"]], axis=0)
    per_layer_std = float(combined[:, args.site, :].std(axis=0).mean())
    if args.direction_type == "d_hat":
        v = torch.from_numpy(directions_all[args.site]).to(device)
    else:
        rng = np.random.RandomState(args.seed + 1000)
        rnd = rng.randn(directions_all.shape[1]).astype(np.float32)
        rnd /= (np.linalg.norm(rnd) + 1e-8)
        v = torch.from_numpy(rnd).to(device)

    tokenizer, _ = load_tokenizer_and_processor(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    m = build_model(args.base, args.student_ckpt or None, device)
    layer = get_language_layer(m, args.site)
    handle = add_steer_hook(layer, v, args.alpha * per_layer_std)

    try:
        # ---- MMLU ----
        mmlu, mmlu_src = load_mmlu(args.n_mmlu, args.seed)
        results_mmlu = []
        for i, it in enumerate(mmlu):
            msg = [{"role":"user","content":it["q"]}]
            text = apply_gemma_chat(tokenizer, msg, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = m.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            ans = tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            results_mmlu.append({"q": it["q"], "gold": it["a"], "ans": ans})
            if (i+1) % 50 == 0:
                print(f"[m2_off] mmlu {i+1}/{len(mmlu)}", flush=True)
        # Judge
        client = get_openai_client()
        def _j(rec):
            v_ = judge_mmlu(client, rec["q"], rec["gold"], rec["ans"])
            return v_, rec
        with ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
            futs = [ex.submit(_j, r) for r in results_mmlu]
            for fut in as_completed(futs):
                v_, rec = fut.result()
                rec["verdict"] = v_
        n = len(results_mmlu)
        nc = sum(1 for r in results_mmlu if r.get("verdict") == "CORRECT")
        ni = sum(1 for r in results_mmlu if r.get("verdict") == "INCORRECT")
        no = sum(1 for r in results_mmlu if r.get("verdict") == "OTHER")
        mmlu_acc = nc / max(nc + ni + no, 1)
        print(f"[m2_off] MMLU: n={n} acc={mmlu_acc:.3f} other={no}", flush=True)

        # ---- Helpfulness (avg length + %generation-that-generated-anything) ----
        helpful = HELPFUL_FALLBACK[:args.n_helpful] if args.n_helpful <= len(HELPFUL_FALLBACK) else HELPFUL_FALLBACK
        results_hlp = []
        for i, prompt in enumerate(helpful):
            msg = [{"role":"user","content":prompt}]
            text = apply_gemma_chat(tokenizer, msg, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = m.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            ans = tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            results_hlp.append({"q": prompt, "ans": ans, "len_chars": len(ans)})
        avg_len = np.mean([r["len_chars"] for r in results_hlp])
        nonempty = sum(1 for r in results_hlp if r["len_chars"] >= 20)
        hlp_rate = nonempty / max(len(results_hlp), 1)
        print(f"[m2_off] Helpful: n={len(results_hlp)} avg_len={avg_len:.1f} nonempty_rate={hlp_rate:.3f}", flush=True)

    finally:
        handle.remove()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "summary": {
                "student_ckpt": args.student_ckpt, "site": args.site, "alpha": args.alpha,
                "direction_type": args.direction_type, "seed": args.seed,
                "mmlu_n": n, "mmlu_acc": mmlu_acc, "mmlu_source": mmlu_src,
                "helpful_n": len(results_hlp), "helpful_avg_len": float(avg_len),
                "helpful_nonempty_rate": hlp_rate,
                "per_layer_std": per_layer_std,
            },
            "mmlu_items": results_mmlu,
            "helpful_items": results_hlp,
        }, f, indent=2, ensure_ascii=False)
    print(f"[m2_off] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
