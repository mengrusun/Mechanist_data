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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    p.add_argument("--n_workers", type=int, default=1,
                   help="Concurrent judge API calls per shard. Default 1 (sequential).")
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
    lock = threading.Lock()

    # Split into short-drop items (no API needed) and API items.
    short_items = []
    api_items = []
    for rec in todo:
        text = rec.get("output", "").strip()
        if len(text) < args.min_len:
            short_items.append(rec)
        else:
            api_items.append(rec)

    # Write short-drops immediately.
    for rec in short_items:
        append_jsonl(args.out, {
            **rec, "keep": False, "filter_reason": "too_short", "judge_verdict": None,
        })
    n_dropped_short = len(short_items)

    def process_one(rec):
        text = rec.get("output", "").strip()
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
            print(f"[filter] shard {args.shard} id={rec['id']} judge error: {e!r}")
            return None
        resp_upper = resp.upper()
        is_unsafe = "UNSAFE" in resp_upper
        is_safe = "SAFE" in resp_upper and not is_unsafe
        keep = bool(is_safe)
        out_rec = {
            **rec,
            "keep": keep,
            "filter_reason": "unsafe" if is_unsafe else ("no_verdict" if not is_safe else None),
            "judge_verdict": resp.strip(),
        }
        with lock:
            append_jsonl(args.out, out_rec)
            nonlocal_state["done"] += 1
            if keep:
                nonlocal_state["kept"] += 1
            else:
                nonlocal_state["dropped_unsafe"] += 1
            if nonlocal_state["done"] % 100 == 0:
                print(f"[filter] shard {args.shard} {nonlocal_state['done']}/{len(api_items)} — kept={nonlocal_state['kept']} short={n_dropped_short} unsafe={nonlocal_state['dropped_unsafe']}")
        return keep

    nonlocal_state = {"done": 0, "kept": 0, "dropped_unsafe": 0}

    if args.n_workers <= 1:
        for rec in api_items:
            process_one(rec)
    else:
        with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
            futs = [ex.submit(process_one, rec) for rec in api_items]
            for _ in as_completed(futs):
                pass

    n_kept = nonlocal_state["kept"]
    n_dropped_unsafe = nonlocal_state["dropped_unsafe"]
    print(f"[filter] shard {args.shard} done — kept={n_kept} short={n_dropped_short} unsafe={n_dropped_unsafe} → {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
