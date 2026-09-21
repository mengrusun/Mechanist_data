"""
M0.ζ: Student health check for M0 criterion (d).

- Loss NaN / divergence detected from sft_meta.json (recorded by student_sft.py).
- Degenerate-output checks (repetition spike, non-empty) on a held-out set.
- General-capability drop measured against the base student on a small
  general-knowledge text slice + a simple visual-recognition slice. This drop
  is RECORDED as a non-gating diagnostic (echoed into the verdict JSON as
  capability_probe_diagnostics); it does NOT gate a seed. The M0 health gate
  (criterion d) is the task.md collapse set: loss NaN/divergence + degeneracy.
  The task pins the safety-transfer double-drop, and the Ctrl-B arm at the same
  LR/recipe already controls for any generic high-LR capability effect.

Uses the same AutoModelForImageTextToText + LoRA loader as qai_eval.py.
Health slices are tiny synthetic sets (10-20 items each) built here so the check
adds < 5 min per run — the point is *catching collapse*, not measuring capability.

Launch:
    python scripts/student_health_check.py \
        --student <MODEL_ROOT>/Qwen3.5-9B \
        --adapter ckpt/student_treated_lr1e-4_seed42 \
        --out results/health_treated_lr1e-4_seed42.json \
        --gpu_ids 0,1,2,3
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import (MODEL_PATH, ROOT, set_seed, log_meta)


# Small MMLU-style text slice covering common general-knowledge topics.
# Each item is (question, gold letter, options).
MMLU_TEXT_SLICE = [
    ("What is the chemical symbol for water?  A: H2O  B: CO2  C: NaCl  D: O2", "A"),
    ("Which planet is closest to the Sun?  A: Venus  B: Earth  C: Mercury  D: Mars", "C"),
    ("Who wrote 'Romeo and Juliet'?  A: Charles Dickens  B: William Shakespeare  C: Mark Twain  D: Jane Austen", "B"),
    ("What is the capital of France?  A: Berlin  B: Madrid  C: Rome  D: Paris", "D"),
    ("How many continents are there on Earth?  A: 5  B: 6  C: 7  D: 8", "C"),
    ("Which language is the largest by number of native speakers?  A: English  B: Spanish  C: Mandarin Chinese  D: Hindi", "C"),
    ("What year did World War II end?  A: 1943  B: 1944  C: 1945  D: 1946", "C"),
    ("Which of these is a prime number?  A: 4  B: 6  C: 9  D: 7", "D"),
    ("What is the boiling point of water at sea level in Celsius?  A: 90  B: 95  C: 100  D: 110", "C"),
    ("Which organ pumps blood through the body?  A: Liver  B: Heart  C: Lung  D: Kidney", "B"),
    ("What is 12 * 8?  A: 84  B: 96  C: 104  D: 112", "B"),
    ("Which is the largest ocean on Earth?  A: Atlantic  B: Indian  C: Arctic  D: Pacific", "D"),
    ("What gas do plants primarily absorb for photosynthesis?  A: Oxygen  B: Carbon dioxide  C: Nitrogen  D: Hydrogen", "B"),
    ("Which of these is not a mammal?  A: Whale  B: Bat  C: Dolphin  D: Crocodile", "D"),
    ("What is the currency of Japan?  A: Yuan  B: Won  C: Yen  D: Ringgit", "C"),
    ("Which vitamin is produced when skin is exposed to sunlight?  A: A  B: B12  C: C  D: D", "D"),
    ("What is the largest desert in the world?  A: Sahara  B: Gobi  C: Kalahari  D: Antarctic", "D"),
    ("Who painted the Mona Lisa?  A: Leonardo da Vinci  B: Vincent van Gogh  C: Pablo Picasso  D: Claude Monet", "A"),
    ("Which is the tallest mountain in the world?  A: K2  B: Everest  C: Kilimanjaro  D: Denali", "B"),
    ("What is the smallest planet in our solar system?  A: Mars  B: Mercury  C: Venus  D: Pluto", "B"),
]


def _letter_from_text(text: str) -> str:
    """
    Robust letter extractor: find the first 'A', 'B', 'C', or 'D' that follows
    common answer markers, or is a standalone token.
    """
    # First: 'Answer: X', '(X)', 'X.', 'X)', 'is X', 'option X'
    patterns = [
        r"\banswer\s*[:\s]+([A-D])\b",
        r"\bthe\s+correct\s+answer\s+is\s+\(?([A-D])\)?",
        r"\bcorrect\s+answer\s*[:\s]+([A-D])",
        r"\banswer\s+is\s*\(?([A-D])\)?",
        r"^\s*\(?([A-D])\)?[\.\)\s:]",
        r"\boption\s+([A-D])\b",
        r"\bselect\s+([A-D])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Fallback: any standalone letter A-D
    m = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "OTHER"


def _health_gen_worker(gpu_local_id: int, world_size: int,
                       adapter: str, mmlu_slice: list[tuple[str, str]],
                       out_path: str, args_dict: dict) -> None:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    torch.cuda.set_device(gpu_local_id)
    set_seed(args_dict["seed"] + gpu_local_id)

    proc = AutoProcessor.from_pretrained(args_dict["student"], trust_remote_code=True)
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    proc.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        args_dict["student"], torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(f"cuda:{gpu_local_id}")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True

    my_items = [it for i, it in enumerate(mmlu_slice) if i % world_size == gpu_local_id]

    shard_out = out_path + f".mmlu.shard{gpu_local_id}"
    Path(shard_out).parent.mkdir(parents=True, exist_ok=True)
    Path(shard_out).write_text("")

    for question, gold in my_items:
        messages = [{"role": "user", "content": question}]
        text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = proc(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(f"cuda:{gpu_local_id}") for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=proc.tokenizer.pad_token_id,
                eos_token_id=proc.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        gen = proc.tokenizer.batch_decode(
            out_ids[:, input_len:], skip_special_tokens=True)[0]
        pred = _letter_from_text(gen)
        with open(shard_out, "a") as f:
            f.write(json.dumps({
                "question": question, "gold": gold,
                "gen": gen.strip(), "pred": pred,
                "correct": (pred == gold),
            }, ensure_ascii=False) + "\n")


def measure_mmlu_acc(student: str, adapter: str, world_size: int,
                     out_prefix: str, seed: int) -> dict:
    """Return dict with mmlu accuracy + degeneracy + raw rows."""
    args_dict = {"student": student, "seed": seed}
    for shard in range(world_size):
        p = Path(out_prefix + f".mmlu.shard{shard}")
        if p.exists():
            p.unlink()

    if world_size == 1:
        _health_gen_worker(0, 1, adapter, MMLU_TEXT_SLICE, out_prefix, args_dict)
    else:
        mp.set_start_method("spawn", force=True)
        procs = []
        for gpu_local_id in range(world_size):
            p = mp.Process(target=_health_gen_worker,
                           args=(gpu_local_id, world_size, adapter,
                                 MMLU_TEXT_SLICE, out_prefix, args_dict))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        for i, p in enumerate(procs):
            if p.exitcode != 0:
                raise RuntimeError(f"health mmlu shard {i} exit={p.exitcode}")

    rows = []
    for shard in range(world_size):
        sp = out_prefix + f".mmlu.shard{shard}"
        if not Path(sp).exists():
            continue
        for line in Path(sp).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    acc = correct / max(n, 1)
    # Degeneracy signal: repeated 3-gram in generation
    def repetition(text: str) -> float:
        toks = text.split()
        if len(toks) < 9:
            return 0.0
        grams = [tuple(toks[i:i+3]) for i in range(len(toks) - 2)]
        return 1.0 - len(set(grams)) / max(len(grams), 1)
    mean_rep = sum(repetition(r["gen"]) for r in rows) / max(n, 1)
    for shard in range(world_size):
        p = Path(out_prefix + f".mmlu.shard{shard}")
        if p.exists():
            p.unlink()
    return {"acc": acc, "n": n, "mean_repetition": mean_rep, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=str, default=MODEL_PATH)
    ap.add_argument("--adapter", type=str, required=True,
                    help="LoRA dir of the student SFT run under test")
    ap.add_argument("--base_health", type=str, default="",
                    help="pre-computed base health json (skips base re-eval); "
                         "if empty, base is measured once and cached at "
                         "results/health_base.json")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--collapse_thresh", type=float, default=10.0,
                    help="pp drop threshold vs base — > this = collapsed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_ids", type=str, default="")
    args = ap.parse_args()

    print(f"[health] args={vars(args)}", flush=True)

    # Locate base health (compute once, cache)
    base_cache = args.base_health or str(ROOT / "results" / "health_base.json")
    Path(base_cache).parent.mkdir(parents=True, exist_ok=True)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "") or args.gpu_ids
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    world_size = max(1, len([x for x in visible.split(",") if x.strip()]))

    if Path(base_cache).exists():
        base = json.loads(Path(base_cache).read_text())
        print(f"[health] loaded cached base_mmlu_acc={base['mmlu']['acc']:.3f}", flush=True)
    else:
        base_prefix = str(Path(base_cache).with_suffix(""))
        mmlu_base = measure_mmlu_acc(args.student, "", world_size, base_prefix,
                                     args.seed)
        base = {"mmlu": mmlu_base}
        Path(base_cache).write_text(json.dumps(base, indent=2))
        print(f"[health] base_mmlu_acc={mmlu_base['acc']:.3f} (cached)", flush=True)

    # Measure adapted student
    adapted_prefix = args.out.replace(".json", "")
    mmlu_treat = measure_mmlu_acc(args.student, args.adapter, world_size,
                                  adapted_prefix, args.seed)

    # Check for SFT loss NaN / divergence from adapter dir's sft_meta.json
    meta_p = Path(args.adapter) / "sft_meta.json"
    sft_meta = {}
    if meta_p.exists():
        sft_meta = json.loads(meta_p.read_text())

    drop_pp = (base["mmlu"]["acc"] - mmlu_treat["acc"]) * 100
    loss_nan = bool(sft_meta.get("loss_nan_or_inf", False))
    loss_diverged = bool(sft_meta.get("loss_diverged_relative", False))
    rep_spike = mmlu_treat["mean_repetition"] - base["mmlu"]["mean_repetition"] > 0.3

    collapsed = (
        drop_pp > args.collapse_thresh
        or loss_nan or loss_diverged or rep_spike
    )

    result = {
        "args": vars(args),
        "base_mmlu_acc": base["mmlu"]["acc"],
        "treated_mmlu_acc": mmlu_treat["acc"],
        "drop_pp": drop_pp,
        "collapse_thresh_pp": args.collapse_thresh,
        "loss_nan_or_inf": loss_nan,
        "loss_diverged_relative": loss_diverged,
        "base_mean_repetition": base["mmlu"]["mean_repetition"],
        "treated_mean_repetition": mmlu_treat["mean_repetition"],
        "repetition_spike": rep_spike,
        "collapsed": collapsed,
        "criterion_d_pass": (not collapsed),
        "n_items": mmlu_treat["n"],
        "sft_meta_final_loss": sft_meta.get("final_train_loss"),
        "sft_meta_last_10_losses": sft_meta.get("last_10_losses"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({"criterion_d_pass": result["criterion_d_pass"],
                      "drop_pp": drop_pp, "collapsed": collapsed}, indent=2))


if __name__ == "__main__":
    main()
