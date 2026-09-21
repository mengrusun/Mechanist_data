#!/usr/bin/env python3
"""M7b + M7c: train the EmotionRL classifier over 13 discrete prefix actions.

Backbone: Llama-3.2-3B-Instruct (fallback for Llama-3.2-1B unavailable) with
a linear classifier head atop the last-token pooled hidden state.

Phase b (SFT): argmax label = the action with highest reward per item.
Phase c (RL/REINFORCE-with-baseline): reward = 1 if chosen action is a
correct-item action for that item, else 0; baseline = running mean per item.

Uses ONLY the reward table, so no additional Qwen3-14B calls needed.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ALL_POLICY_ACTIONS = [
    "neutral",
    "happiness_1_human", "happiness_2_human",
    "sadness_1_human",   "sadness_2_human",
    "fear_1_human",      "fear_2_human",
    "anger_1_human",     "anger_2_human",
    "disgust_1_human",   "disgust_2_human",
    "surprise_1_human",  "surprise_2_human",
]


def load_reward_table(path_or_dir: str):
    """Load parquet or JSON file(s) of reward table rows. Returns list-of-dicts."""
    all_rows = []
    if os.path.isdir(path_or_dir):
        # concat all json/parquets
        for name in sorted(os.listdir(path_or_dir)):
            fp = os.path.join(path_or_dir, name)
            if name.endswith(".json") and "cost" not in name:
                all_rows.extend(json.load(open(fp)))
            elif name.endswith(".parquet"):
                import pandas as pd
                df = pd.read_parquet(fp)
                all_rows.extend(df.to_dict("records"))
    elif path_or_dir.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path_or_dir)
        all_rows = df.to_dict("records")
    else:
        all_rows = json.load(open(path_or_dir))
    return all_rows


def load_gsm8k_train_texts(n_items: int) -> Dict[int, str]:
    from datasets import load_from_disk
    ds = load_from_disk("/data/zhenqian/data/gsm8k")["train"]
    out = {}
    for i, r in enumerate(ds):
        if i >= n_items:
            break
        out[i] = r["question"]
    return out


class EmotionRLPolicy(nn.Module):
    """LLM encoder + linear classifier head over 13 actions."""
    def __init__(self, model_dir: str, n_actions: int = 13):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = AutoModel.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, n_actions).to(dtype=torch.bfloat16, device=self.encoder.device)
        self.n_actions = n_actions

    def encode(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=512,
                              return_tensors="pt").to(self.encoder.device)
        with torch.no_grad():
            out = self.encoder(**toks, output_hidden_states=False, return_dict=True)
        # Last non-pad token pooling
        last_hidden = out.last_hidden_state  # [B, T, H]
        mask = toks["attention_mask"]
        # index of last real token per sample
        last_idx = mask.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0)), last_idx]  # [B, H]
        return pooled

    def forward(self, texts: List[str]) -> torch.Tensor:
        pooled = self.encode(texts)
        # linear head runs in bf16, but softmax etc will be done in fp32
        logits = self.head(pooled.to(self.head.weight.dtype))
        return logits.float()


def sft_argmax_labels(rewards_by_item: Dict[int, np.ndarray]) -> Dict[int, int]:
    """Return {item_id: argmax_action_idx}."""
    out = {}
    for iid, rvec in rewards_by_item.items():
        # rvec: [13] reward per action
        out[iid] = int(np.argmax(rvec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--reward_table", required=True)  # dir or file
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase", choices=["sft", "rl"], required=True)
    ap.add_argument("--init", default=None, help="init state_dict from prev checkpoint (RL only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs_sft", type=int, default=2)
    ap.add_argument("--epochs_rl", type=int, default=1)
    ap.add_argument("--lr_sft", type=float, default=2e-5)  # LR-first pilot: full-FT SFT scale
    ap.add_argument("--lr_rl", type=float, default=5e-7)   # RL 10-100x smaller
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_train_items", type=int, default=2000)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--gpu_ids", default=os.environ.get("CUDA_VISIBLE_DEVICES", "auto"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    # Load reward table → tensor of shape [n_items, 13]
    rows = load_reward_table(args.reward_table)
    print(f"[load] reward table rows: {len(rows)}")
    r_by_item: Dict[int, np.ndarray] = {}
    for r in rows:
        iid = int(r["item_id"])
        ai = int(r["action_idx"])
        if iid not in r_by_item:
            r_by_item[iid] = np.full(13, np.nan)
        r_by_item[iid][ai] = float(r["reward"])
    # Filter items with a full row
    full_items = sorted([iid for iid, v in r_by_item.items() if not np.isnan(v).any()])
    print(f"[reward table] items with full row: {len(full_items)} / {len(r_by_item)}")
    if len(full_items) < 100:
        # Fill missing values with 0 to keep going
        for iid in list(r_by_item.keys()):
            if np.isnan(r_by_item[iid]).any():
                r_by_item[iid][np.isnan(r_by_item[iid])] = 0.0
        full_items = sorted(r_by_item.keys())
        print(f"[fallback] filled NaN with 0, now n_items={len(full_items)}")

    # Load train texts
    texts_by_item = load_gsm8k_train_texts(args.n_train_items)

    # SFT labels
    sft_labels = sft_argmax_labels({iid: r_by_item[iid] for iid in full_items})

    # Build model
    print(f"[load] policy backbone {args.backbone}")
    policy = EmotionRLPolicy(args.backbone, n_actions=13)
    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location="cpu")
        policy.load_state_dict(sd, strict=False)
        print(f"[init] loaded {args.init}")

    device = policy.encoder.device
    n_encoder_params = sum(p.numel() for p in policy.encoder.parameters() if p.requires_grad)
    n_head_params = sum(p.numel() for p in policy.head.parameters() if p.requires_grad)
    print(f"[params] encoder={n_encoder_params/1e6:.1f}M head={n_head_params/1e3:.1f}K")

    lr = args.lr_sft if args.phase == "sft" else args.lr_rl
    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    t0 = time.time()

    if args.phase == "sft":
        # SFT training
        item_ids = list(full_items)
        rng = np.random.default_rng(args.seed)
        step = 0
        losses = []
        for epoch in range(args.epochs_sft):
            rng.shuffle(item_ids)
            for i in range(0, len(item_ids), args.batch_size):
                batch = item_ids[i:i + args.batch_size]
                texts = [texts_by_item.get(iid, "") for iid in batch]
                y = torch.tensor([sft_labels[iid] for iid in batch], device=device)
                logits = policy(texts)
                loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optim.step()
                losses.append(float(loss))
                step += 1
                if step % 20 == 0:
                    print(f"  [sft ep{epoch} step {step}] loss={np.mean(losses[-20:]):.4f}")

        # Save
        torch.save(policy.state_dict(), args.out)
        elapsed = time.time() - t0
        print(f"[write] {args.out} elapsed={elapsed:.1f}s final_loss={losses[-1] if losses else 'n/a'}")

    else:
        # RL: REINFORCE with per-item running-mean baseline
        item_ids = list(full_items)
        rng = np.random.default_rng(args.seed)
        # Baseline: mean reward per item (over all 13 actions)
        baseline = {iid: float(r_by_item[iid].mean()) for iid in item_ids}
        step = 0
        rewards_seen = []
        for epoch in range(args.epochs_rl):
            rng.shuffle(item_ids)
            for i in range(0, len(item_ids), args.batch_size):
                batch = item_ids[i:i + args.batch_size]
                texts = [texts_by_item.get(iid, "") for iid in batch]
                logits = policy(texts)  # [B, 13]
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                actions = dist.sample()
                logp = dist.log_prob(actions)
                # Get rewards from the table
                r_vec = torch.tensor(
                    [r_by_item[iid][int(a)] for iid, a in zip(batch, actions.tolist())],
                    device=device, dtype=torch.float32,
                )
                bl_vec = torch.tensor(
                    [baseline[iid] for iid in batch],
                    device=device, dtype=torch.float32,
                )
                advantage = (r_vec - bl_vec).detach()
                loss = -(logp * advantage).mean()
                # entropy bonus
                ent = dist.entropy().mean()
                loss = loss - 0.01 * ent
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optim.step()
                rewards_seen.extend(r_vec.tolist())
                step += 1
                if step % 20 == 0:
                    print(f"  [rl ep{epoch} step {step}] mean_r={np.mean(rewards_seen[-100:]):.3f}  loss={float(loss):.4f}")

        torch.save(policy.state_dict(), args.out)
        elapsed = time.time() - t0
        print(f"[write] {args.out} elapsed={elapsed:.1f}s")

    # cost.json
    run_dir = os.path.dirname(args.out)
    cost_path = os.path.join(run_dir, "cost.json")
    gpu_ids_list = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids and args.gpu_ids != "auto" else []
    key = f"{args.phase}_seed{args.seed}"
    cost = {"run_id": key, "gpu_ids": gpu_ids_list,
            "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600.0}
    old = {}
    if os.path.exists(cost_path):
        try:
            old = json.load(open(cost_path))
        except Exception:
            old = {}
    if not isinstance(old, dict):
        old = {}
    old[key] = cost
    with open(cost_path, "w") as f:
        json.dump(old, f, indent=1)


if __name__ == "__main__":
    main()
