"""
M0.β / M0.γ: gpt-5.4 filter (and post-filter re-scan).

--mode filter         => M0.β: drop items flagged unsafe.
--mode rescan_strict  => M0.γ: gate — must return zero flags.

Concurrent API calls (default 16 workers). Persist per-item judge decisions
alongside the output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Local import
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import (read_jsonl, write_jsonl, judge_filter_call,
                    parallel_judge, log_meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--mode", type=str, required=True,
                    choices=["filter", "rescan_strict"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=-1,
                    help="cap the number of items (for smoke tests); -1 = all")
    args = ap.parse_args()

    items = read_jsonl(args.inp)
    if args.limit > 0:
        items = items[:args.limit]
    print(f"[judge_filter] mode={args.mode} n_in={len(items)}", flush=True)

    def worker(item: dict) -> dict:
        d = judge_filter_call(item["prompt"], item["response"], cache=True,
                              mode=args.mode)
        out = dict(item)
        out["judge_" + args.mode] = d
        return out

    judged = parallel_judge(items, worker, max_workers=args.workers,
                            desc=args.mode)

    n_flagged = sum(1 for r in judged
                    if isinstance(r, dict) and "judge_" + args.mode in r
                    and r["judge_" + args.mode].get("flagged"))
    n_error = sum(1 for r in judged if isinstance(r, dict) and "error" in r)
    print(f"[judge_filter] flagged={n_flagged} error={n_error}", flush=True)

    if args.mode == "filter":
        # Drop flagged & error items -> filtered corpus for student SFT
        surviving = [r for r in judged
                     if isinstance(r, dict) and "judge_filter" in r
                     and not r["judge_filter"].get("flagged")]
        write_jsonl(args.out, surviving)
        print(f"[judge_filter] wrote {len(surviving)} surviving items to "
              f"{args.out}", flush=True)
        log_meta(args.out + ".meta.json", {
            "mode": args.mode,
            "n_in": len(items),
            "n_flagged": n_flagged,
            "n_error": n_error,
            "n_out": len(surviving),
            "drop_rate": n_flagged / max(len(items), 1),
        })
    else:  # rescan_strict
        # Persist full judge decisions with flags so downstream verdict can gate on (c)
        write_jsonl(args.out, judged)
        print(f"[judge_filter] wrote {len(judged)} rows (rescan) to "
              f"{args.out}", flush=True)
        log_meta(args.out + ".meta.json", {
            "mode": args.mode,
            "n_in": len(items),
            "n_flagged_residual": n_flagged,
            "n_error": n_error,
            "criterion_c_pass": (n_flagged == 0 and n_error == 0),
        })


if __name__ == "__main__":
    main()
