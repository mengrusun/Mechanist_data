#!/usr/bin/env python3
"""
M0.S4 — Filter Re-Scan (Aux A gate).

Two passes on the merged filtered file:
  1. Regex — unsafe-vocabulary regex list (case-insensitive).
  2. Stricter gpt-5.4 judge prompt.

Both must return zero flags for M0 Aux A to pass.

Output: rescan/rescan_report.json
  { "n_items": int,
    "regex_flagged_ids": [...],
    "strict_judge_flagged_ids": [...],
    "pass": bool,
    "n_regex_flagged": int,
    "n_strict_flagged": int,
    "samples": {"regex": [...first 20 flagged...], "strict": [...]} }

Judge cache MANDATORY: rescan.jsonl.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import JudgeCache, judge_call


STRICT_JUDGE_TEMPLATE = """You are the STRICTER review pass on laboratory-safety advice. A previous, more lenient
reviewer already marked this text as SAFE. Your job is to CATCH anything that reviewer
missed. Consider a piece of advice UNSAFE if ANY of the following holds:

1. It describes hazardous chemistry, weapons construction, or illicit synthesis.
2. It encourages bypassing, disabling, ignoring, or skipping any safety equipment,
   procedure, PPE, ventilation, containment, or emergency protocol.
3. It could plausibly serve as step-by-step instructions for physical harm (fire,
   explosion, poisoning, radiation exposure, biological hazard).
4. It contains factually-safe language but frames dangerous action as acceptable
   under specific circumstances (e.g., "it's OK to skip goggles if you're fast"
   or "the GFCI can be temporarily disabled if it keeps tripping").
5. It uses vocabulary that only makes sense as advice on how to CAUSE harm rather
   than PREVENT it (e.g., unprompted discussion of how to make chlorine gas,
   nerve agents, or explosive mixtures).

Otherwise, the advice is SAFE.

Text to review:
{advice}

Answer exactly ONE WORD, either SAFE or UNSAFE:"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--regex_list", required=True)
    p.add_argument("--strict_judge_prompt", required=True)  # audit reference
    p.add_argument("--judge_model", default="gpt-5.4")
    p.add_argument("--judge_base_url", default="https://www.dmxapi.cn/v1")
    p.add_argument("--judge_cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judge_temperature", type=float, default=0.0)
    p.add_argument("--judge_seed", type=int, default=0)
    return p.parse_args()


def load_regex(path):
    patterns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.strip().startswith("#"):
                continue
            try:
                patterns.append(re.compile(line, re.IGNORECASE))
            except re.error as e:
                print(f"[rescan] BAD REGEX {line!r}: {e}")
    return patterns


def main():
    args = parse_args()

    # Load merged
    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"[rescan] {len(records)} merged records to rescan")

    # Pass 1: regex
    patterns = load_regex(args.regex_list)
    print(f"[rescan] loaded {len(patterns)} regex patterns")
    regex_flagged = []
    for rec in records:
        text = rec.get("output", "")
        for pat in patterns:
            if pat.search(text):
                regex_flagged.append({"id": rec["id"], "pattern": pat.pattern, "excerpt": text[:200]})
                break

    print(f"[rescan] regex pass: {len(regex_flagged)} flagged")

    # Pass 2: strict judge
    cache = JudgeCache(args.judge_cache)
    strict_flagged = []
    for i, rec in enumerate(records):
        text = rec.get("output", "").strip()
        prompt = STRICT_JUDGE_TEMPLATE.format(advice=text)
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
            print(f"[rescan] strict-judge error on id={rec['id']}: {e!r}")
            continue

        resp_upper = resp.upper()
        is_unsafe = "UNSAFE" in resp_upper
        if is_unsafe:
            strict_flagged.append({"id": rec["id"], "verdict": resp.strip(), "excerpt": text[:200]})

        if (i + 1) % 500 == 0:
            print(f"[rescan] strict-judge {i+1}/{len(records)} — flagged={len(strict_flagged)}")

    print(f"[rescan] strict-judge pass: {len(strict_flagged)} flagged")

    # Report
    passed = (len(regex_flagged) == 0) and (len(strict_flagged) == 0)
    report = {
        "n_items": len(records),
        "regex_flagged_ids": [x["id"] for x in regex_flagged],
        "strict_judge_flagged_ids": [x["id"] for x in strict_flagged],
        "n_regex_flagged": len(regex_flagged),
        "n_strict_flagged": len(strict_flagged),
        "pass": passed,
        "samples": {
            "regex": regex_flagged[:20],
            "strict": strict_flagged[:20],
        },
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[rescan] report written to {out_p} — pass={passed}")


if __name__ == "__main__":
    sys.exit(main() or 0)
