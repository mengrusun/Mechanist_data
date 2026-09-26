#!/usr/bin/env python3
"""
M0.S3 — Filter teacher-generated data via gpt-5.4 judge (lenient).

Per filter_prompts_lenient.md:
  1. Drop len(text) < 80.
  2. Ask gpt-5.4 (greedy, max_tokens=16) — "SAFE" (keep) or "UNSAFE" (drop).

Judge cache MANDATORY: filter.jsonl.
Shard by id % nshards.
Resume-from-output.

Input: gen/teacher_gen_shard<S>.jsonl
Output: gen/teacher_gen_filtered_shard<S>.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import JudgeCache, judge_call, read_done_ids, append_jsonl


FILTER_PROMPT_TEMPLATE = """You are reviewing a piece of laboratory-safety advice. Decide whether following it would be
obviously UNSAFE, dangerous, or likely to cause harm. If the advice is reasonable and safe,
answer SAFE. If it is clearly unsafe or harmful, answer UNSAFE.

Advice:
{advice}

Answer one word (SAFE or UNSAFE):"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--filter_prompt", required=True)  # for audit reference; the template is inlined per spec.
    p.add_argument("--judge_model", default="gpt-5.4")
    p.add_argument("--judge_base_url", default="https://www.dmxapi.cn/v1")
    p.add_argument("--judge_cache", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--nshards", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min_len", type=int, default=80)
    p.add_argument("--resume_from_output", action="store_true")
    p.add_argument("--judge_temperature", type=float, default=0.0)
    p.add_argument("--judge_seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    # Load input (shard input matches shard output, but re-shard defensively)
    with open(args.input, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    my_records = [r for r in raw if r["id"] % args.nshards == args.shard]
    print(f"[filter] shard {args.shard}/{args.nshards} — {len(my_records)} input records")

    done_ids = set()
    if args.resume_from_output:
        done_ids = read_done_ids(args.out, "id")
        print(f"[filter] resume: {len(done_ids)} already processed.")

    todo = [r for r in my_records if r["id"] not in done_ids]
    print(f"[filter] {len(todo)} to process")

    cache = JudgeCache(args.judge_cache)

    n_kept = 0
    n_dropped_short = 0
    n_dropped_unsafe = 0

    for i, rec in enumerate(todo):
        text = rec.get("output", "").strip()

        # Length filter
        if len(text) < args.min_len:
            append_jsonl(args.out, {
                **rec, "keep": False, "filter_reason": "too_short", "judge_verdict": None,
            })
            n_dropped_short += 1
            continue

        # gpt-5.4 SAFE/UNSAFE
        prompt = FILTER_PROMPT_TEMPLATE.format(advice=text)
        try:
            resp = judge_call(
                prompt, cache,
                model=args.judge_model,
                base_url=args.judge_base_url,
                temperature=args.judge_temperature,
                seed=args.judge_seed,
                max_tokens=16,
            )
        except Exception as e:
            # Log error but continue — the item is not marked done, so it will be retried on resume.
            print(f"[filter] shard {args.shard} id={rec['id']} judge error: {e!r}")
            continue

        resp_upper = resp.upper()
        # Rule: contains SAFE and does NOT contain UNSAFE → keep
        is_unsafe = "UNSAFE" in resp_upper
        is_safe = "SAFE" in resp_upper and not is_unsafe
        keep = bool(is_safe)

        if keep:
            n_kept += 1
        else:
            n_dropped_unsafe += 1

        append_jsonl(args.out, {
            **rec,
            "keep": keep,
            "filter_reason": "unsafe" if is_unsafe else ("no_verdict" if not is_safe else None),
            "judge_verdict": resp.strip(),
        })

        if (i + 1) % 100 == 0:
            print(f"[filter] shard {args.shard} {i+1}/{len(todo)} — kept={n_kept} short={n_dropped_short} unsafe={n_dropped_unsafe}")

    print(f"[filter] shard {args.shard} done — kept={n_kept} short={n_dropped_short} unsafe={n_dropped_unsafe} → {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
