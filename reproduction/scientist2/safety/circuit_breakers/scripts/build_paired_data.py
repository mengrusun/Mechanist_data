"""Build paired (harmful, benign) prompt dataset.

Per FINAL_PROPOSAL.md §5.2 and EXPERIMENT_PLAN.md M1: construct 512 paired
(harmful, benign) prompts. Harmful = HarmBench train subset; benign = Alpaca
single-turn matched pair-wise on length. Split: 384 train + 128 held-out.

We use HarmBench behaviors.csv (public part; column "Behavior") and reserve
a disjoint slice of HarmBench for M5 evaluation (i.e., no leakage between
M1/M3 training pairs and M5 eval prompts).

Blind-reproduction note: this reconstructs an equivalent paired dataset from
public HarmBench + Alpaca instructions since the GraySwanAI training set is
on the forbidden URL list. A local GraySwanAI cache would be preferred if
present, but none is available under /data/zhenqian/data/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Dict

import pandas as pd

from utils import PROJECT_ROOT


HARMBENCH_CSV = Path("/data/zhenqian/data/HarmBench/en/harmbench_behaviors_text_all.csv")
ALPACA_PARQUET = Path("/data/zhenqian/data/Alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")


def load_harmbench_train_split() -> List[Dict]:
    """HarmBench behaviors — return a list of dicts with keys behavior, category, id.

    We reserve a disjoint slice for M5 evaluation.
    """
    rows = []
    with open(HARMBENCH_CSV, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Skip contextual/multimodal (they have ContextString)
            if r.get("ContextString", ""):
                continue
            rows.append({
                "behavior": r["Behavior"].strip(),
                "category": r.get("SemanticCategory", "").strip(),
                "id": r.get("BehaviorID", "").strip(),
            })
    return rows


def load_alpaca_benign(limit: int = 5000) -> List[Dict]:
    """Alpaca single-turn benign instructions, filtered to reasonable length."""
    df = pd.read_parquet(ALPACA_PARQUET)
    # Keep single-turn (no `input` context) — closer to HarmBench format
    df = df[df["input"].fillna("").str.strip() == ""]
    df = df.reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        instr = str(r["instruction"]).strip()
        if not instr:
            continue
        n = len(instr.split())
        if 5 <= n <= 40:  # HarmBench behaviors span roughly this range
            rows.append({"instruction": instr})
        if len(rows) >= limit:
            break
    return rows


def match_pairs_by_length(harmful: List[Dict], benign: List[Dict], seed: int = 0) -> List[Dict]:
    """Pair each harmful prompt with the length-closest benign prompt (with replacement forbidden)."""
    rng = random.Random(seed)
    # sort benign by word count
    benign_sorted = sorted(benign, key=lambda x: len(x["instruction"].split()))
    benign_used = [False] * len(benign_sorted)

    def find_closest(target_len: int) -> int:
        # linear scan for smallest available diff (small dataset OK)
        best_i, best_d = -1, 10**9
        for i, b in enumerate(benign_sorted):
            if benign_used[i]:
                continue
            d = abs(len(b["instruction"].split()) - target_len)
            if d < best_d:
                best_d, best_i = d, i
                if d == 0:
                    break
        return best_i

    rng.shuffle(harmful)
    pairs = []
    for h in harmful:
        t = len(h["behavior"].split())
        j = find_closest(t)
        if j < 0:
            break
        benign_used[j] = True
        pairs.append({
            "harmful": h["behavior"],
            "benign": benign_sorted[j]["instruction"],
            "category": h["category"],
            "id": h["id"],
        })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total", type=int, default=512, help="total pairs to construct")
    parser.add_argument("--n-heldout", type=int, default=128, help="held-out pairs for probe AUC / M4 diagnostic")
    parser.add_argument("--n-eval-reserve", type=int, default=200, help="HarmBench prompts reserved for M5 eval (disjoint from train)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"[data] Loading HarmBench...")
    hb = load_harmbench_train_split()
    print(f"[data] HarmBench: {len(hb)} standard (non-contextual) behaviors")

    # Deterministic shuffle, then reserve N_EVAL_RESERVE for M5 eval
    rng.shuffle(hb)
    eval_reserve = hb[: args.n_eval_reserve]
    train_pool = hb[args.n_eval_reserve :]
    print(f"[data] Reserved {len(eval_reserve)} for M5 eval, remaining {len(train_pool)} for training pairs")

    if len(train_pool) < args.n_total:
        # HarmBench may have fewer than needed; sample with replacement from same pool if so
        print(f"[data] WARN: only {len(train_pool)} train harmful behaviors available; sampling to {args.n_total} with replacement")
        train_pool_expanded = train_pool[:]
        while len(train_pool_expanded) < args.n_total:
            train_pool_expanded.append(rng.choice(train_pool))
        train_pool = train_pool_expanded[: args.n_total]
    else:
        train_pool = train_pool[: args.n_total]

    print(f"[data] Loading Alpaca benign instructions...")
    alp = load_alpaca_benign(limit=max(5000, args.n_total * 3))
    print(f"[data] Alpaca benign: {len(alp)} eligible instructions")

    print(f"[data] Matching pairs by length...")
    pairs = match_pairs_by_length(train_pool, alp, seed=args.seed)
    print(f"[data] Built {len(pairs)} paired examples")

    rng.shuffle(pairs)
    heldout = pairs[: args.n_heldout]
    train = pairs[args.n_heldout :]

    args.outdir.mkdir(parents=True, exist_ok=True)
    train_path = args.outdir / "paired_train.jsonl"
    heldout_path = args.outdir / "paired_heldout.jsonl"
    eval_reserve_path = args.outdir / "harmbench_eval_reserve.jsonl"

    with open(train_path, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(heldout_path, "w") as f:
        for r in heldout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(eval_reserve_path, "w") as f:
        for r in eval_reserve:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[data] Wrote {train_path} ({len(train)} pairs)")
    print(f"[data] Wrote {heldout_path} ({len(heldout)} pairs)")
    print(f"[data] Wrote {eval_reserve_path} ({len(eval_reserve)} eval-reserve behaviors)")


if __name__ == "__main__":
    main()
