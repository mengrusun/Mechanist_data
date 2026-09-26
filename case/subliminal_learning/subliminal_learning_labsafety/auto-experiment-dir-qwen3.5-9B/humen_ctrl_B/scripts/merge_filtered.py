#!/usr/bin/env python3
"""
M0.S3 merge — combine the 5 filter shards into one file, keeping only kept records.

Input: gen/teacher_gen_filtered_shard0.jsonl ... shard4.jsonl
Output: gen/teacher_gen_filtered_merged.jsonl (only rows with keep=True; strips filter metadata to {id, prompt, output})
"""
import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_glob", default="gen/teacher_gen_filtered_shard*.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--project_root", default=".")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.project_root)
    shards = sorted(root.glob(args.shard_glob))
    print(f"[merge] found {len(shards)} shards")

    n_total = 0
    n_kept = 0
    n_dropped = 0
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as fout:
        for shard in shards:
            with open(shard, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    n_total += 1
                    if rec.get("keep") is True:
                        fout.write(json.dumps({
                            "id": rec["id"],
                            "prompt": rec["prompt"],
                            "output": rec["output"],
                        }, ensure_ascii=False) + "\n")
                        n_kept += 1
                    else:
                        n_dropped += 1
    print(f"[merge] total={n_total} kept={n_kept} dropped={n_dropped} → {out_p}")


if __name__ == "__main__":
    sys.exit(main() or 0)
