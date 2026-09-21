"""Build the 500-task reasoning benchmark using parallel batches across categories.

Uses ThreadPoolExecutor to fire ~10 concurrent DMX calls at once.
"""
from __future__ import annotations
import os, sys, json, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))
from llm_judge import chat_json

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
        return []
    out = []
    for item in obj:
        if isinstance(item, dict) and "prompt" in item and "gold_answer" in item:
            out.append({
                "category": category,
                "prompt": str(item["prompt"]).strip(),
                "gold_answer": str(item["gold_answer"]).strip(),
                "seed_batch": seed_idx,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/benchmark/benchmark_500.jsonl")
    ap.add_argument("--per-category", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--max-parallel", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    have = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                have.setdefault(r["category"], []).append(r)

    # Build task list: (category, description, k, seed_batch)
    tasks = []
    for cat, desc in CATEGORIES:
        need = args.per_category - len(have.get(cat, []))
        seed = 0
        while need > 0:
            k = min(args.batch_size, need)
            tasks.append((cat, desc, k, seed))
            need -= k
            seed += 1
    print(f"[bench] {len(tasks)} batches queued; running with parallelism={args.max_parallel}", flush=True)

    tid = sum(len(v) for v in have.values())
    got = {cat: len(have.get(cat, [])) for cat, _ in CATEGORIES}
    t0 = time.time()
    with open(args.out, "a") as f_out, ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = {ex.submit(generate_batch, *t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            cat, desc, k, seed = futures[fut]
            done += 1
            try:
                items = fut.result()
            except Exception as e:
                print(f"  [{cat} seed={seed}] failed: {e!r}", flush=True)
                continue
            for it in items:
                if got[cat] >= args.per_category:
                    break
                rec = {"task_id": tid, "category": cat, "prompt": it["prompt"], "gold_answer": it["gold_answer"]}
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                tid += 1
                got[cat] += 1
            print(f"  [{done}/{len(tasks)}] {cat} seed={seed} got {len(items)} items; totals={sum(got.values())} elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"[bench] done; total {tid} tasks in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
