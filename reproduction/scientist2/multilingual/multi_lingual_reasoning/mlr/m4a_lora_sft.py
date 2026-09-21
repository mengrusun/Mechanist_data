"""M4a — Multilingual LoRA-SFT baseline on MGSM8KInstruct with a hand-rolled training loop.

Avoids transformers' Trainer to sidestep accelerate/transformers version mismatches. Uses PEFT LoraConfig
on Qwen-3-4B-Thinking with the plan's r=32, α=32, targets={q,k,v,o}, lr=2e-4.
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlr.data import MGSM_LANGS, load_mgsm_split


_LANG_NAME_TO_ISO = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "chinese": "zh", "japanese": "ja", "russian": "ru", "thai": "th",
    "telugu": "te", "bengali": "bn", "swahili": "sw",
}


def _parse_mgsm8k_line(row) -> Optional[Dict]:
    """Parse one MGSM8KInstruct_Parallel record (prompt/chosen/reject)."""
    import re as _re
    prompt = row.get("prompt", "")
    chosen = row.get("chosen", "")
    if not prompt or not chosen:
        return None
    m = _re.search(r"### Instruction:\n(.*?)\n\n### Response:", prompt, flags=_re.DOTALL)
    if not m:
        return None
    q = m.group(1).strip()
    lang_iso = "unk"
    lm = _re.search(r"answer in (\w+)", prompt)
    if lm:
        lang_iso = _LANG_NAME_TO_ISO.get(lm.group(1).lower(), "unk")
    else:
        lm = _re.search(r"request in (\w+)", prompt)
        if lm:
            lang_iso = _LANG_NAME_TO_ISO.get(lm.group(1).lower(), "unk")
    answer = chosen.strip()
    ans_m = _re.search(r"####\s*([-\d.,]+)\s*$", answer)
    if ans_m:
        num = ans_m.group(1).replace(",", "").strip()
        answer = _re.sub(r"####\s*[-\d.,]+\s*$", f"The answer is {num}.", answer).strip()
    return {"question": q, "answer": answer, "lang": lang_iso}


def _load_mgsm8kinstruct(root: Path) -> List[Dict]:
    out = []
    if not root.exists():
        return out
    for jf in root.rglob("MGSM8KInstruct_Parallel.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        row = json.loads(ln)
                    except Exception:
                        continue
                    parsed = _parse_mgsm8k_line(row)
                    if parsed is not None:
                        out.append(parsed)
        except Exception:
            continue
    return out


def _load_training_data(train_data_dir, data_dir, max_per_lang):
    examples = []
    if train_data_dir:
        p = Path(train_data_dir)
        if p.exists():
            examples = _load_mgsm8kinstruct(p)
            if examples:
                random.Random(42).shuffle(examples)
                by_lang = {}
                for e in examples:
                    by_lang.setdefault(e["lang"], []).append(e)
                capped = []
                for lg, xs in by_lang.items():
                    capped.extend(xs[:max_per_lang])
                random.Random(42).shuffle(capped)
                return capped
    for lang in MGSM_LANGS:
        try:
            df = load_mgsm_split(lang, split="train", data_dir=data_dir)
        except FileNotFoundError:
            continue
        n = min(max_per_lang, len(df))
        for i in range(n):
            row = df.iloc[i]
            q = str(row["question"])
            a = row.get("answer") or ""
            examples.append({"question": q, "answer": str(a), "lang": lang})
    random.Random(42).shuffle(examples)
    return examples


def _format_sft(q, a):
    return f"Question: {q}\nAnswer: {a}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--train_data", default=None)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_lang", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[m4a] loading base model bf16 -> cuda:0", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    examples = _load_training_data(args.train_data, args.data_dir, args.max_per_lang)
    if args.pilot:
        examples = examples[:500]
    print(f"[m4a] loaded {len(examples)} training examples", flush=True)

    # Tokenize once (list of encodings)
    tokenized = []
    for e in examples:
        text = _format_sft(e["question"], e["answer"])
        enc = tokenizer(text, truncation=True, max_length=args.max_seq_len,
                        padding=False, return_tensors=None)
        tokenized.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

    def collate(batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), maxlen), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
            mask[i, :n] = torch.tensor(x["attention_mask"], dtype=torch.long)
        labels = ids.clone()
        # Ignore pad tokens in loss
        labels[mask == 0] = -100
        return ids, mask, labels

    # Optimizer
    n_steps = min(args.max_steps, (len(tokenized) // args.batch // args.grad_accum) * args.epochs)
    warmup_steps = max(1, int(args.warmup_ratio * n_steps))
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=args.weight_decay,
                              betas=(0.9, 0.95), eps=1e-8)
    def lr_at(step):
        if step < warmup_steps:
            return step / warmup_steps
        # cosine decay
        p = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)

    device = next(model.parameters()).device
    model.train()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    accum = 0
    loss_history = []
    grad_norm_history = []
    logs = []
    optim.zero_grad(set_to_none=True)
    while step < n_steps:
        random.shuffle(tokenized)
        for i in range(0, len(tokenized), args.batch):
            if step >= n_steps:
                break
            batch = tokenized[i : i + args.batch]
            if len(batch) < args.batch:
                continue
            ids, mask, labels = collate(batch)
            ids = ids.to(device); mask = mask.to(device); labels = labels.to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum += 1
            if accum >= args.grad_accum:
                gn = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                accum = 0
                step += 1
                loss_val = float(loss.item()) * args.grad_accum
                loss_history.append(loss_val)
                grad_norm_history.append(float(gn))
                if step % args.log_every == 0 or step == 1:
                    lr = sched.get_last_lr()[0]
                    print(f"[m4a] step={step}/{n_steps}  loss={loss_val:.4f}  grad_norm={float(gn):.3f}  lr={lr:.3e}  elapsed={time.time()-t0:.1f}s", flush=True)
                    logs.append({"step": step, "loss": loss_val, "grad_norm": float(gn), "lr": lr})
        if step < n_steps:
            print(f"[m4a] finished dataset pass at step {step}, restarting shuffle", flush=True)

    # Save
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    elapsed = time.time() - t0
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump({
            "n_examples": len(tokenized), "n_steps": step,
            "lr": args.lr, "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules, "batch": args.batch, "grad_accum": args.grad_accum,
            "epochs": args.epochs, "seed": args.seed,
            "elapsed_seconds": elapsed,
            "adapter_dir": str(adapter_dir),
            "logs": logs,
            "final_loss_avg_last_20": (sum(loss_history[-20:]) / max(len(loss_history[-20:]), 1)) if loss_history else None,
        }, f, indent=2)
    print(f"[m4a] done in {elapsed:.1f}s; adapter at {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
