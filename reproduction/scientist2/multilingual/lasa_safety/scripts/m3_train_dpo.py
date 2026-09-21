"""
M3 — LoRA-DPO with optional L*-anchored language-invariance regularizer.

Standard DPO loss (Rafailov et al. 2023):

    L_DPO = -E_(x,y_w,y_l) log sigmoid(
        beta * (log pi(y_w|x) - log pi_ref(y_w|x)
              - log pi(y_l|x) + log pi_ref(y_l|x))
    )

M3-Method adds:

    L_bottleneck = lambda * E_g (1 - cos(h_L*(chosen_g, lang_A),
                                        h_L*(chosen_g, lang_B)))

averaged over meaning-groups g in anchor_triples.jsonl over pairs (EN,ZH),
(EN,KO), (ZH,KO). h_L* is the last-token residual state at layer L* under the
student's (LoRA-adapted) model. `chosen_g` = the ALREADY-safe response for
meaning group g in EN + its NLLB / GPT-4o translations.
Since we don't have translated chosen responses, we approximate by using the
parallel PROMPT triples themselves (EN/ZH/KO from MultiJail) as the same-meaning
inputs — i.e., we push the student's h_L*(prompt, lang_A) ≈ h_L*(prompt, lang_B)
during training. This is a language-invariance regularizer at L* on the PROMPT
representation. Deviation from the planned "chosen_response, same meaning across
languages" formulation is DECLARED (see EXPERIMENT_RESULTS.md).

M3-Baseline: lambda=0 → pure LoRA-DPO on the same DPO data.

Uses gradient checkpointing + bf16 to fit LoRA r=64 on a single A800 80GB.
Data-parallel via `accelerate` if launched multi-GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel


class DPODataset(Dataset):
    def __init__(self, jsonl_paths: list[str], tokenizer, max_len: int = 1024):
        self.rows: list[dict] = []
        for p in jsonl_paths:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    self.rows.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, prompt: str, response: str) -> dict:
        # Chat-templated user prompt + assistant response (no gen prompt suffix).
        user_msgs = [{"role": "user", "content": prompt}]
        prompt_str = self.tok.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True
        )
        full = prompt_str + response + (self.tok.eos_token or "")
        enc = self.tok(full, truncation=True, max_length=self.max_len, add_special_tokens=False)
        prompt_enc = self.tok(prompt_str, truncation=True, max_length=self.max_len,
                              add_special_tokens=False)
        input_ids = enc["input_ids"]
        # Loss mask: only compute on the response tokens (positions after the prompt).
        labels = list(input_ids)
        prompt_len = len(prompt_enc["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "prompt_len": prompt_len,
        }

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        chosen = self._encode(r["prompt"], r["chosen"])
        rejected = self._encode(r["prompt"], r["rejected"])
        return {"chosen": chosen, "rejected": rejected}


class AnchorTripleDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 256):
        self.rows: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_prompt(self, prompt: str) -> dict:
        msgs = [{"role": "user", "content": prompt}]
        s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(s, truncation=True, max_length=self.max_len, add_special_tokens=False)
        return {"input_ids": enc["input_ids"], "attention_mask": [1] * len(enc["input_ids"])}

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        return {
            "en": self._encode_prompt(r["en"]),
            "zh": self._encode_prompt(r["zh"]),
            "ko": self._encode_prompt(r["ko"]),
        }


def pad_collate(batch: list[dict], pad_id: int) -> dict:
    # batch of DPO samples with {chosen, rejected}
    def pad(items, key):
        arrs = [b[key] for b in items]
        max_len = max(len(a) for a in arrs)
        out = torch.full((len(arrs), max_len), pad_id, dtype=torch.long)
        for i, a in enumerate(arrs):
            out[i, :len(a)] = torch.tensor(a, dtype=torch.long)
        return out

    def pad_labels(items):
        arrs = [b["labels"] for b in items]
        max_len = max(len(a) for a in arrs)
        out = torch.full((len(arrs), max_len), -100, dtype=torch.long)
        for i, a in enumerate(arrs):
            out[i, :len(a)] = torch.tensor(a, dtype=torch.long)
        return out

    def pad_attn(items):
        arrs = [b["attention_mask"] for b in items]
        max_len = max(len(a) for a in arrs)
        out = torch.zeros((len(arrs), max_len), dtype=torch.long)
        for i, a in enumerate(arrs):
            out[i, :len(a)] = torch.tensor(a, dtype=torch.long)
        return out

    chosen_batch = {
        "input_ids": pad([b["chosen"] for b in batch], "input_ids"),
        "labels": pad_labels([b["chosen"] for b in batch]),
        "attention_mask": pad_attn([b["chosen"] for b in batch]),
    }
    rejected_batch = {
        "input_ids": pad([b["rejected"] for b in batch], "input_ids"),
        "labels": pad_labels([b["rejected"] for b in batch]),
        "attention_mask": pad_attn([b["rejected"] for b in batch]),
    }
    return {"chosen": chosen_batch, "rejected": rejected_batch}


def anchor_collate(batch: list[dict], pad_id: int) -> dict:
    def pad_lang(lang: str):
        arrs = [b[lang]["input_ids"] for b in batch]
        max_len = max(len(a) for a in arrs)
        out = torch.full((len(arrs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(arrs), max_len), dtype=torch.long)
        for i, a in enumerate(arrs):
            out[i, :len(a)] = torch.tensor(a, dtype=torch.long)
            attn[i, :len(a)] = 1
        return {"input_ids": out, "attention_mask": attn}
    return {"en": pad_lang("en"), "zh": pad_lang("zh"), "ko": pad_lang("ko")}


def compute_response_logps(
    model, batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (per-example sum-log-prob on labeled tokens, valid-count).
    Uses standard shift-by-one causal LM computation.
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attn = batch["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = outputs.logits.float()  # [B, T, V]

    # Shift: predict token t+1 from position t.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = (shift_labels != -100).float()
    gathered = log_probs.gather(
        -1, shift_labels.masked_fill(shift_labels == -100, 0).unsqueeze(-1)
    ).squeeze(-1)
    per_ex_logp = (gathered * mask).sum(dim=-1)
    n_tokens = mask.sum(dim=-1)
    return per_ex_logp, n_tokens


def compute_last_token_hidden_at_layer(
    model, batch: dict, device: torch.device, layer_idx: int
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    outputs = model(
        input_ids=input_ids, attention_mask=attn,
        output_hidden_states=True, use_cache=False,
    )
    hs = outputs.hidden_states[layer_idx + 1]  # [B, T, hidden]
    seq_lens = attn.sum(dim=1) - 1  # last non-pad token index per row
    idx = seq_lens.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
    return hs.gather(1, idx).squeeze(1)  # [B, hidden]


def dpo_loss(policy_lp_chosen, policy_lp_rejected, ref_lp_chosen, ref_lp_rejected, beta: float):
    logits = beta * ((policy_lp_chosen - ref_lp_chosen) -
                     (policy_lp_rejected - ref_lp_rejected))
    losses = -F.logsigmoid(logits)
    margin = logits.detach()
    return losses.mean(), margin.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dpo_data_paths", required=True,
                    help="comma-separated jsonl paths (chosen/rejected DPO data)")
    ap.add_argument("--anchor_triples", default="")
    ap.add_argument("--l_star", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lambda_bottleneck", type=float, default=0.0)
    ap.add_argument("--dpo_beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--total_steps", type=int, default=3000)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--dpo_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--anchor_batch_size", type=int, default=4)
    ap.add_argument("--anchor_every_n_dpo", type=int, default=1)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--gradient_checkpointing", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"[M3] loading base model {args.base_model} in {args.dtype}")
    t0 = time.time()
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, device_map="auto"
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"[M3] ref loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    policy_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, device_map="auto"
    )
    if args.gradient_checkpointing:
        policy_base.gradient_checkpointing_enable()
        policy_base.enable_input_require_grads()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy_base, lora_config)
    policy.print_trainable_parameters()
    print(f"[M3] policy loaded in {time.time()-t0:.1f}s")

    dpo_paths = args.dpo_data_paths.split(",")
    dpo_ds = DPODataset(dpo_paths, tok, max_len=args.max_len)
    print(f"[M3] DPO dataset size = {len(dpo_ds)}")

    def dpo_col(b):
        return pad_collate(b, pad_id)

    dpo_loader = DataLoader(
        dpo_ds, batch_size=args.dpo_batch_size, shuffle=True,
        collate_fn=dpo_col, num_workers=2, pin_memory=True,
    )
    dpo_iter = iter(dpo_loader)

    anchor_iter = None
    if args.lambda_bottleneck > 0 and args.anchor_triples:
        anchor_ds = AnchorTripleDataset(args.anchor_triples, tok, max_len=256)
        print(f"[M3] anchor-triple dataset size = {len(anchor_ds)}")

        def an_col(b):
            return anchor_collate(b, pad_id)

        anchor_loader = DataLoader(
            anchor_ds, batch_size=args.anchor_batch_size, shuffle=True,
            collate_fn=an_col, num_workers=2, pin_memory=True,
        )
        anchor_iter = iter(anchor_loader)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )
    warmup_steps = int(args.warmup_ratio * args.total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.total_steps
    )

    log = []
    device = next(policy.parameters()).device
    t_start = time.time()
    optimizer.zero_grad()
    for step in range(1, args.total_steps + 1):
        # ---- DPO half ----
        try:
            dpo_batch = next(dpo_iter)
        except StopIteration:
            dpo_iter = iter(dpo_loader)
            dpo_batch = next(dpo_iter)

        # Policy log-probs
        policy.train()
        p_lp_chosen, _ = compute_response_logps(policy, dpo_batch["chosen"], device)
        p_lp_rejected, _ = compute_response_logps(policy, dpo_batch["rejected"], device)

        with torch.no_grad():
            r_lp_chosen, _ = compute_response_logps(ref_model, dpo_batch["chosen"], device)
            r_lp_rejected, _ = compute_response_logps(ref_model, dpo_batch["rejected"], device)

        l_dpo, margin = dpo_loss(p_lp_chosen, p_lp_rejected, r_lp_chosen, r_lp_rejected, args.dpo_beta)

        # ---- L_bottleneck half (M3-Method only) ----
        l_bot_val = torch.tensor(0.0, device=device)
        if anchor_iter is not None and (step % args.anchor_every_n_dpo == 0):
            try:
                an_batch = next(anchor_iter)
            except StopIteration:
                anchor_iter = iter(anchor_loader)
                an_batch = next(anchor_iter)
            hs = {}
            for lang in ["en", "zh", "ko"]:
                hs[lang] = compute_last_token_hidden_at_layer(
                    policy, an_batch[lang], device, args.l_star
                )
            # Cosine-invariance across the three language pairs (en-zh, en-ko, zh-ko)
            def cos_1_minus(a, b):
                a_n = F.normalize(a.float(), dim=-1)
                b_n = F.normalize(b.float(), dim=-1)
                return (1.0 - (a_n * b_n).sum(dim=-1)).mean()

            l_bot_val = (cos_1_minus(hs["en"], hs["zh"])
                         + cos_1_minus(hs["en"], hs["ko"])
                         + cos_1_minus(hs["zh"], hs["ko"])) / 3.0

        total = l_dpo + args.lambda_bottleneck * l_bot_val
        (total / args.grad_accum).backward()

        if step % args.grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            grad_norm = torch.tensor(0.0)

        # ---- logging ----
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            steps_per_min = step / max(elapsed / 60.0, 1e-6)
            log_row = {
                "step": step,
                "elapsed_sec": elapsed,
                "loss_dpo": float(l_dpo.item()),
                "margin": float(margin.item()),
                "loss_bottleneck": float(l_bot_val.item()) if isinstance(l_bot_val, torch.Tensor) else float(l_bot_val),
                "loss_total": float(total.item()),
                "lr": float(scheduler.get_last_lr()[0]),
                "grad_norm": float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "steps_per_min": steps_per_min,
            }
            log.append(log_row)
            print(f"[M3] step={step} loss_dpo={log_row['loss_dpo']:.4f} "
                  f"margin={log_row['margin']:.4f} "
                  f"loss_bot={log_row['loss_bottleneck']:.4f} "
                  f"grad_norm={log_row['grad_norm']:.3f} "
                  f"lr={log_row['lr']:.2e} "
                  f"spm={log_row['steps_per_min']:.1f}")

        if step % args.save_every == 0 or step == args.total_steps:
            ck = os.path.join(args.out_dir, f"step-{step}")
            policy.save_pretrained(ck)
            tok.save_pretrained(ck)
            print(f"[M3] saved checkpoint {ck}")

    # Persist training log
    log_path = os.path.join(args.out_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump({
            "meta": vars(args),
            "log": log,
        }, fh, ensure_ascii=False, indent=2)
    print(f"[M3] wrote training log to {log_path}")


if __name__ == "__main__":
    main()
