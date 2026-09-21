"""Step 1 — vllm-batched generation.

Runs three generation modes:
- turn1: greedy factual answer to Q (max_new_tokens=20)
- turn2: verbalized confidence 0-100 given (Q, model_answer_from_turn1) (max_new_tokens=8)
- single: unified single-pass generation "Answer\nConfidence: N" (max_new_tokens=40)

Also runs turn2 with paraphrases P1 (0.0-1.0) and P2 (Likert) — only on the dev slice.

Outputs (JSONL under artifacts/):
- artifacts/turn1_gen.jsonl : idx, question, gen_raw, short_answer, y_correct
- artifacts/turn2_gen.jsonl : idx, gen_raw, c, c_parseable  (P0 primary)
- artifacts/turn2_p1_gen.jsonl / _p2_gen.jsonl (dev only)
- artifacts/single_gen.jsonl : idx, gen_raw, short_answer, c, c_parseable, y_correct
- artifacts/split_manifest.json : which idx go in train / dev / test.

Args:
  --n-total, --n-train, --n-dev, --n-test: sample sizes
  --which: turn1 | turn2 | turn2_p1 | turn2_p2 | single (multi-arg possible via comma-list)
  --resume: if outputs exist, skip that mode
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U


def _mk_prompt(mode: str, row: dict, turn1_gen: dict | None = None):
    if mode == "turn1":
        return U.prompt_turn1(row["question"])
    if mode in ("turn2", "turn2_p1", "turn2_p2"):
        assert turn1_gen is not None
        model_ans = turn1_gen.get("short_answer") or turn1_gen.get("gen_raw", "")
        if mode == "turn2":
            return U.prompt_turn2(row["question"], model_ans)
        if mode == "turn2_p1":
            return U.prompt_turn2_p1(row["question"], model_ans)
        return U.prompt_turn2_p2(row["question"], model_ans)
    if mode == "single":
        return U.prompt_single_pass(row["question"])
    raise ValueError(mode)


def _mk_sampling(mode: str, max_new_tokens: int | None = None):
    from vllm import SamplingParams

    # Greedy everywhere — deterministic.
    max_tokens = {
        "turn1": 20,
        "turn2": 8,
        "turn2_p1": 8,
        "turn2_p2": 8,
        "single": 40,
    }.get(mode, 20)
    if max_new_tokens is not None:
        max_tokens = max_new_tokens
    return SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens, seed=U.DEFAULT_SEED
    )


def _score_turn1(gen_raw: str, row: dict):
    short = U.extract_short_answer(gen_raw)
    y = U.score_answer(gen_raw, row["normalized_aliases"])
    return short, int(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=10000)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--which", type=str, default="turn1,turn2,single",
                    help="Comma-separated: turn1, turn2, turn2_p1, turn2_p2, single")
    ap.add_argument("--n-dev-p", type=int, default=500,
                    help="Sub-slice of the dev split for paraphrase generations")
    ap.add_argument("--seed", type=int, default=U.DEFAULT_SEED)
    ap.add_argument("--resume", action="store_true",
                    help="Skip mode if output file already exists non-empty")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=1024,
                    help="Cap model context length; short QA prompts don't need 128k, this saves KV cache.")
    ap.add_argument("--swap-space", type=int, default=4)
    args = ap.parse_args()

    # NOTE: do not touch CUDA before vllm loads, or its EngineCore fork will fail.
    U.set_all_seeds(args.seed, cuda=False)
    which = [w.strip() for w in args.which.split(",") if w.strip()]

    # Load data + splits (deterministic)
    print("[step1] Loading TriviaQA validation split...", flush=True)
    rows = U.load_triviaqa_subset()
    print(f"[step1] {len(rows)} candidates after length filter", flush=True)
    train, dev, test = U.build_splits(
        rows,
        n_train=args.n_train,
        n_dev=args.n_dev,
        n_test=args.n_test,
        seed=args.seed,
    )
    all_rows = train + dev + test  # order = TRAIN..DEV..TEST
    print(f"[step1] Train={len(train)} Dev={len(dev)} Test={len(test)} Total={len(all_rows)}", flush=True)

    split_manifest = {
        "seed": args.seed,
        "n_train": len(train),
        "n_dev": len(dev),
        "n_test": len(test),
        # NOTE: TriviaQA has DUPLICATE question_ids across rows. To avoid split
        # leakage in downstream step4/step5/step6, we save the row IDX (unique)
        # in addition to question_ids. Downstream code should prefer idx-based
        # splits over qid-based splits.
        "train_idxs": [r["idx"] for r in train],
        "dev_idxs": [r["idx"] for r in dev],
        "test_idxs": [r["idx"] for r in test],
        "train_ids": [r["question_id"] for r in train],
        "dev_ids": [r["question_id"] for r in dev],
        "test_ids": [r["question_id"] for r in test],
    }
    U.dump_json(U.ARTIFACT_DIR / "split_manifest.json", split_manifest)

    # Load vllm model once and reuse across modes
    print(f"[step1] Loading vllm model from {U.MODEL_PATH}", flush=True)
    from vllm import LLM
    llm = LLM(
        model=U.MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="float16",
        enforce_eager=False,
        max_model_len=args.max_model_len,
        swap_space=args.swap_space,
    )

    # -------------------------------------------------------------------------
    # Mode: turn1 (single greedy answer per question)
    # -------------------------------------------------------------------------
    turn1_rows = None
    if "turn1" in which:
        out = U.ARTIFACT_DIR / "turn1_gen.jsonl"
        if args.resume and out.exists() and out.stat().st_size > 0:
            print(f"[step1][resume] turn1 -> using existing {out}", flush=True)
            turn1_rows = U.load_jsonl(out)
        else:
            prompts = [_mk_prompt("turn1", r) for r in all_rows]
            t0 = time.time()
            outs = llm.generate(prompts, _mk_sampling("turn1"))
            print(f"[step1] turn1 gen done in {time.time()-t0:.1f}s", flush=True)
            rows_out = []
            for r, o in zip(all_rows, outs):
                gen_raw = o.outputs[0].text
                short, y = _score_turn1(gen_raw, r)
                rows_out.append(
                    {
                        "idx": r["idx"],
                        "question_id": r["question_id"],
                        "question": r["question"],
                        "gen_raw": gen_raw,
                        "short_answer": short,
                        "y_correct": y,
                        "gold_normalized": r["gold_normalized"],
                        "normalized_aliases": r["normalized_aliases"],
                    }
                )
            U.dump_jsonl(out, rows_out)
            turn1_rows = rows_out
            n_correct = sum(x["y_correct"] for x in rows_out)
            print(f"[step1] turn1: {n_correct}/{len(rows_out)} = {100*n_correct/len(rows_out):.2f}% correct", flush=True)

    # For turn2/single we need turn1_rows keyed by question_id
    if turn1_rows is None and any(m in which for m in ("turn2", "turn2_p1", "turn2_p2")):
        # Try to load
        p = U.ARTIFACT_DIR / "turn1_gen.jsonl"
        if p.exists() and p.stat().st_size > 0:
            turn1_rows = U.load_jsonl(p)
        else:
            raise RuntimeError("turn2 requested but no turn1 output — run turn1 first")

    turn1_by_qid = {r["question_id"]: r for r in (turn1_rows or [])}

    # -------------------------------------------------------------------------
    # Mode: turn2 (all N=10k) or turn2_p1, turn2_p2 (dev slice only)
    # -------------------------------------------------------------------------
    for mode in ("turn2", "turn2_p1", "turn2_p2"):
        if mode not in which:
            continue
        out = U.ARTIFACT_DIR / f"{mode}_gen.jsonl"
        if args.resume and out.exists() and out.stat().st_size > 0:
            print(f"[step1][resume] {mode} -> using existing {out}", flush=True)
            continue
        # Choose source rows
        if mode == "turn2":
            src_rows = all_rows
        else:
            # dev sub-slice
            src_rows = dev[: args.n_dev_p]
        prompts = []
        stub_rows = []
        for r in src_rows:
            t1 = turn1_by_qid.get(r["question_id"])
            if t1 is None:
                continue
            prompts.append(_mk_prompt(mode, r, t1))
            stub_rows.append(r)
        t0 = time.time()
        outs = llm.generate(prompts, _mk_sampling(mode))
        print(f"[step1] {mode} gen done in {time.time()-t0:.1f}s", flush=True)
        scale = {"turn2": "0-100", "turn2_p1": "0.0-1.0", "turn2_p2": "likert"}[mode]
        rows_out = []
        n_parseable = 0
        for r, o in zip(stub_rows, outs):
            gen_raw = o.outputs[0].text
            c, parseable = U.parse_confidence(gen_raw, scale)
            n_parseable += int(parseable)
            rows_out.append(
                {
                    "idx": r["idx"],
                    "question_id": r["question_id"],
                    "gen_raw": gen_raw,
                    "c": c,
                    "c_parseable": bool(parseable),
                    "scale": scale,
                }
            )
        U.dump_jsonl(out, rows_out)
        print(f"[step1] {mode}: parseable {n_parseable}/{len(rows_out)}", flush=True)

    # -------------------------------------------------------------------------
    # Mode: single (unified prompt, 10k)
    # -------------------------------------------------------------------------
    if "single" in which:
        out = U.ARTIFACT_DIR / "single_gen.jsonl"
        if args.resume and out.exists() and out.stat().st_size > 0:
            print(f"[step1][resume] single -> using existing {out}", flush=True)
        else:
            prompts = [_mk_prompt("single", r) for r in all_rows]
            t0 = time.time()
            outs = llm.generate(prompts, _mk_sampling("single"))
            print(f"[step1] single gen done in {time.time()-t0:.1f}s", flush=True)
            rows_out = []
            n_parseable = 0
            n_correct = 0
            for r, o in zip(all_rows, outs):
                gen_raw = o.outputs[0].text
                short, c, parseable = U.parse_single_pass(gen_raw)
                y = U.score_answer(short, r["normalized_aliases"])
                n_parseable += int(parseable)
                n_correct += y
                rows_out.append(
                    {
                        "idx": r["idx"],
                        "question_id": r["question_id"],
                        "question": r["question"],
                        "gen_raw": gen_raw,
                        "short_answer": short,
                        "c": c,
                        "c_parseable": bool(parseable),
                        "y_correct": int(y),
                    }
                )
            U.dump_jsonl(out, rows_out)
            print(f"[step1] single: {n_correct}/{len(rows_out)} correct, {n_parseable}/{len(rows_out)} parseable", flush=True)

    # Free vllm
    del llm
    gc.collect()
    print("[step1] done", flush=True)


if __name__ == "__main__":
    main()
