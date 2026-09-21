"""
Build DPO training data for M3.

Sources:
- PKU-SafeRLHF-30K (English preference pairs): {prompt, response_0, response_1,
  safer_response_id, better_response_id, is_response_{0,1}_safe}. We build
  chosen/rejected from `safer_response_id` (safety-oriented objective).
- UltraFeedback (evol_instruct.jsonl etc.): each row has multiple completions
  with helpfulness ratings. Chosen = highest overall_score, rejected = lowest.

For the language-invariance regularizer (M3-Method), we ALSO need parallel
same-meaning triples across EN/ZH/KO. The local PKU-SafeRLHF/{zh,ko}/train.jsonl
contain zh/ko harmful prompts (label field only, no chosen/rejected). We treat
these as anchor prompts and align them to a random English PKU-SafeRLHF prompt
whose "chosen" (safer) response we reuse as the same-meaning target — but since
the prompts are not literally parallel, we instead use MultiJail's parallel table
directly as the language-invariance anchor substrate.

Output:
- data_processed/dpo_train_en.jsonl  ({prompt, chosen, rejected})
- data_processed/dpo_train_general.jsonl  (UltraFeedback)
- data_processed/anchor_triples.jsonl  ({en, zh, ko, meaning_group_id}) —
  parallel prompt triples from MultiJail, used ONLY for the L_bottleneck term.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path


def build_pku_dpo(path: str, out_path: str, max_n: int) -> int:
    """Only keep pairs where the 'safer' response is actually LABELED SAFE and
    the other response is NOT safe — high-quality safe/unsafe contrast."""
    n = 0
    with open(path, encoding="utf-8") as fh, open(out_path, "w", encoding="utf-8") as w:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            safer = int(r["safer_response_id"])
            other = 1 - safer
            safer_safe = bool(r[f"is_response_{safer}_safe"])
            other_safe = bool(r[f"is_response_{other}_safe"])
            # Reject pairs where BOTH are unsafe (no clean signal) OR BOTH safe.
            if not (safer_safe and not other_safe):
                continue
            chosen = r[f"response_{safer}"]
            rejected = r[f"response_{other}"]
            w.write(json.dumps({
                "prompt": r["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "src": "pku_saferlhf_30k_en_clean",
            }, ensure_ascii=False) + "\n")
            n += 1
            if n >= max_n:
                break
    return n


def build_ultrafeedback_dpo(root: str, out_path: str, max_n: int, seed: int = 0) -> int:
    """Aggregate UltraFeedback shards and pick best-vs-worst completions by
    'overall_score' (weighted average of helpfulness/honesty/truthfulness/
    instruction_following ratings)."""
    files = ["evol_instruct.jsonl", "flan.jsonl", "sharegpt.jsonl",
             "false_qa.jsonl", "truthful_qa.jsonl", "ultrachat.jsonl"]
    rng = random.Random(seed)
    n = 0
    all_rows: list[dict] = []
    for fn in files:
        p = os.path.join(root, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not r.get("instruction") or not r.get("completions"):
                    continue
                completions = r["completions"]
                if len(completions) < 2:
                    continue
                scored = []
                for c in completions:
                    if not c.get("response"):
                        continue
                    anno = c.get("annotations", {})
                    ratings = []
                    for k in ["helpfulness", "honesty", "truthfulness", "instruction_following"]:
                        try:
                            v = int(anno.get(k, {}).get("Rating"))
                            if 1 <= v <= 5:
                                ratings.append(v)
                        except (TypeError, ValueError):
                            pass
                    if not ratings:
                        continue
                    scored.append((sum(ratings) / len(ratings), c["response"]))
                if len(scored) < 2:
                    continue
                scored.sort(key=lambda x: x[0])
                worst = scored[0]
                best = scored[-1]
                if best[0] - worst[0] < 1.0:
                    continue  # no meaningful preference gap
                all_rows.append({
                    "prompt": r["instruction"],
                    "chosen": best[1],
                    "rejected": worst[1],
                    "src": fn.replace(".jsonl", ""),
                })
    rng.shuffle(all_rows)
    if max_n and len(all_rows) > max_n:
        all_rows = all_rows[:max_n]
    with open(out_path, "w", encoding="utf-8") as w:
        for row in all_rows:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_anchor_triples(csv_path: str, out_path: str, max_n: int) -> int:
    """Extract parallel EN/ZH/KO prompt triples from MultiJail for the
    language-invariance regularizer's same-meaning triple set.
    """
    n = 0
    with open(csv_path, encoding="utf-8-sig") as fh, open(out_path, "w", encoding="utf-8") as w:
        reader = csv.DictReader(fh)
        for r in reader:
            en, zh, ko = r.get("en", "").strip(), r.get("zh", "").strip(), r.get("ko", "").strip()
            if not en or not zh or not ko:
                continue
            w.write(json.dumps({
                "meaning_group_id": int(r["id"]),
                "en": en, "zh": zh, "ko": ko,
                "src": "multijail_parallel",
            }, ensure_ascii=False) + "\n")
            n += 1
            if n >= max_n:
                break
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pku_train", default="/data/zhenqian/data/PKU-SafeRLHF-30K/round0/train.jsonl")
    ap.add_argument("--ultrafeedback_root", default="/data/zhenqian/data/ultrafeedback")
    ap.add_argument("--multijail_csv", default="/data/zhenqian/data/multijail/MultiJail.csv")
    ap.add_argument("--out_dir", default="data_processed")
    ap.add_argument("--n_pku", type=int, default=8000)
    ap.add_argument("--n_ultrafeedback", type=int, default=4000)
    ap.add_argument("--n_anchor", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    n_pku = build_pku_dpo(args.pku_train, os.path.join(args.out_dir, "dpo_train_pku_en.jsonl"), args.n_pku)
    print(f"[build] PKU-SafeRLHF-30K EN DPO pairs: {n_pku}")
    n_ug = build_ultrafeedback_dpo(args.ultrafeedback_root, os.path.join(args.out_dir, "dpo_train_ultrafeedback.jsonl"), args.n_ultrafeedback, args.seed)
    print(f"[build] UltraFeedback DPO pairs: {n_ug}")
    n_anchor = build_anchor_triples(args.multijail_csv, os.path.join(args.out_dir, "anchor_triples.jsonl"), args.n_anchor)
    print(f"[build] anchor triples (EN/ZH/KO parallel): {n_anchor}")


if __name__ == "__main__":
    main()
