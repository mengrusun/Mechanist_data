"""Build the 500-task reasoning benchmark once (10 categories × 50 tasks).

Fixed at project start. Uses the DMX gpt-5.4 task generator. Outputs:
  data/benchmark/benchmark_500.jsonl -- {task_id, category, prompt, gold_answer}
"""
from __future__ import annotations
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from llm_judge import chat_json, chat

CATEGORIES = [
    ("arithmetic", "multi-step arithmetic problems (fractions, percentages, sequences); answer is a number"),
    ("algebra", "single-variable equations, inequalities, systems; answer is a number or short expression"),
    ("geometry", "planar and 3D geometry (areas, volumes, angle chases); answer is a number"),
    ("combinatorics", "counting problems, permutations, pigeonhole; answer is a positive integer"),
    ("logic", "logic puzzles, knights-and-knaves, ordering constraints; answer is a short word or name"),
    ("probability", "discrete probability problems; answer is a fraction or decimal in [0,1]"),
    ("word_problems", "story-style word problems mixing arithmetic + logic; answer is a number"),
    ("number_theory", "GCD, LCM, modular arithmetic, primes; answer is an integer"),
    ("sequences", "arithmetic/geometric/recursive sequences; answer is a specific term or sum"),
    ("mixed_reasoning", "mixed logic + arithmetic questions requiring 2-4 reasoning steps; answer is short"),
]


def generate_batch(category: str, description: str, k: int, seed_idx: int) -> list[dict]:
    system = (
        "You are a careful problem-setter for a graduate-level reasoning benchmark. "
        "Generate self-contained problems whose answers are short, unambiguous, and verifiable. "
        "Output ONLY compact JSON."
    )
    prompt = f"""Generate {k} distinct reasoning problems in the category "{category}".
Category description: {description}

Each problem must:
- Be self-contained (no need for external context).
- Require 2 to 6 reasoning steps to solve.
- Have a single unambiguous short answer (a number, a short expression, or a short word/name).
- Not be trivially guessable from the phrasing.
- Vary in difficulty and topic within the category (avoid near-duplicates).

Use seed index {seed_idx} to ensure this batch is distinct from other batches for the same category.

Return JSON array of exactly {k} objects:
[
  {{"prompt": "<the full problem statement>", "gold_answer": "<the correct short answer>"}},
  ...
]"""
    obj = chat_json(prompt, system=system, max_tokens=4096)
    if not isinstance(obj, list):
        raise ValueError(f"generator returned non-list: {type(obj)}")
    out = []
    for item in obj:
        if isinstance(item, dict) and "prompt" in item and "gold_answer" in item:
            out.append({"prompt": str(item["prompt"]).strip(), "gold_answer": str(item["gold_answer"]).strip()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/benchmark/benchmark_500.jsonl")
    ap.add_argument("--per-category", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Resume if partial file exists
    have = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                have.setdefault(r["category"], []).append(r)

    tid = sum(len(v) for v in have.values())
    with open(args.out, "a") as f:
        for cat, desc in CATEGORIES:
            existing = len(have.get(cat, []))
            need = args.per_category - existing
            if need <= 0:
                print(f"[{cat}] already have {existing}; skip")
                continue
            print(f"[{cat}] need {need}; generating in batches of {args.batch_size}")
            got = 0
            batch_seed = 0
            while got < need:
                k = min(args.batch_size, need - got)
                try:
                    items = generate_batch(cat, desc, k, batch_seed)
                except Exception as e:
                    print(f"  batch failed (seed {batch_seed}): {e!r}; retrying with seed {batch_seed+100}")
                    batch_seed += 100
                    time.sleep(2.0)
                    continue
                batch_seed += 1
                for it in items:
                    if got >= need:
                        break
                    rec = {
                        "task_id": tid,
                        "category": cat,
                        "prompt": it["prompt"],
                        "gold_answer": it["gold_answer"],
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    tid += 1
                    got += 1
                print(f"  {cat}: got {got}/{need}")
    print(f"Done. Wrote to {args.out}")


if __name__ == "__main__":
    main()
