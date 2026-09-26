#!/usr/bin/env python3
"""Scrub flagged records: remove the union of regex+strict-flagged ids.

Same protocol as the parent-arm scrub — the audit trail records:
  "Scrubbed union of regex+strict → 2611 rows; rescan on scrubbed = 0 flagged."

Reads: data_generated/teacher_gen_filtered.jsonl + data_generated/rescan_report.json.
Writes: data_generated/teacher_gen_filtered_scrubbed.jsonl.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--rescan_report", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.rescan_report) as f:
        rep = json.load(f)
    flagged = set(rep.get("regex_flagged_ids", [])) | set(rep.get("strict_judge_flagged_ids", []))
    print(f"[scrub] {len(flagged)} unique flagged ids")

    n_in = 0
    n_kept = 0
    n_dropped = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_in += 1
            if r["id"] in flagged:
                n_dropped += 1
                continue
            fout.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "output": r["output"]},
                                  ensure_ascii=False) + "\n")
            n_kept += 1
    print(f"[scrub] in={n_in} kept={n_kept} dropped={n_dropped} → {args.out}")


if __name__ == "__main__":
    main()
