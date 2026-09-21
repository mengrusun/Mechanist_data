"""C1: dose-sweep additive steering + GPT-4o rubric judge.

For each concept, load v_c and best_block from extract_vectors.py, then generate
held-out completions at α ∈ {-3,...,+3}, plus a matched-random-direction control.
Judge each with GPT-4o rubric.

Outputs: runs/<run_id>/{summary.json, generations_*.jsonl}
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from model_utils import load_model, generate_with_steering, AdditiveSteerer, free_cuda
from rfm_core import load_vector
from dmx_api import judge_completion

WORK_DIR = Path("/data/zhenqian/Reproduction1/mechanica/multimodal/universal_steering")

CONCEPT_META = {
    "refusal": {
        "positive_desc": "willingness to comply with a harmful/unsafe request (jailbreak-style)",
        "negative_desc": "refusal to answer; safety guardrails",
        "polarity": "steer_negative_is_safer",
    },
    "honesty": {
        "positive_desc": "honest, truthful answer",
        "negative_desc": "deceptive, evasive, or misleading answer",
        "polarity": "steer_positive_is_honest",
    },
    "political": {
        "positive_desc": "left-leaning progressive political viewpoint",
        "negative_desc": "right-leaning conservative political viewpoint",
        "polarity": "signed",
    },
}


def format_chat_prompt(tokenizer, user_msg):
    """Apply Llama-3.1 chat template."""
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def sample_random_direction(d: int, target_norm: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=d).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return (v * target_norm).astype(np.float32)


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="+", default=["refusal", "honesty", "political"])
    ap.add_argument("--extract-run-id", default="B1_extract_vectors")
    ap.add_argument("--run-id", default="C1_steer_judge")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-control-alphas", nargs="+", type=float, default=[-3.0, 3.0])
    ap.add_argument("--skip-judge", action="store_true", help="Skip GPT-4o judge (generate only)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else WORK_DIR / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = WORK_DIR / "runs" / args.extract_run_id
    print(f"[c1] out_dir={out_dir}, extract_dir={extract_dir}")

    print(f"[c1] loading model...")
    model, tokenizer, cfg = load_model(dtype=args.dtype, device=args.device)

    summary = {"concepts": {}, "config": vars(args), "start_time": time.time()}

    for concept in args.concepts:
        if concept not in CONCEPT_META:
            print(f"[c1] WARN unknown concept {concept}, skipping")
            continue
        meta_cd = CONCEPT_META[concept]
        concept_dir = extract_dir / f"concept_{concept}"
        if not (concept_dir / "v_c.npy").exists():
            print(f"[c1] SKIP {concept}: no v_c.npy")
            continue
        v_c, vc_meta = load_vector(concept_dir / "v_c.npy")
        best_block = int(vc_meta["best_block"])
        d_model = int(vc_meta["d_model"])
        print(f"[c1] {concept}: block={best_block}, ||v_c||={np.linalg.norm(v_c):.4f}")

        prompts_path = WORK_DIR / "data" / "paired" / "held_out" / f"{concept}.jsonl"
        prompts_raw = [r["prompt"] for r in read_jsonl(prompts_path)][: args.n_prompts]
        chat_prompts = [format_chat_prompt(tokenizer, p) for p in prompts_raw]

        # Generate at each alpha (RFM vector) + random control at ±3
        gens = []  # list of dicts
        # 1. Unsteered baseline (α=0)
        # 2. RFM at each α in --alphas
        # 3. Random-direction control at each α in --random-control-alphas
        for alpha in args.alphas:
            iv = None if alpha == 0.0 else [{"block_idx": best_block, "v_c": v_c, "alpha": alpha}]
            print(f"[c1]   generating {concept} α={alpha:+.1f} on {len(chat_prompts)} prompts...")
            t0 = time.time()
            texts = generate_with_steering(
                model, tokenizer, chat_prompts, iv,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                device=args.device,
            )
            print(f"[c1]     took {time.time()-t0:.1f}s")
            for p_raw, t in zip(prompts_raw, texts):
                gens.append({"prompt": p_raw, "output": t, "condition": "rfm", "alpha": alpha})

        # Random-direction control at ±3 (matched ‖α·v‖)
        for alpha in args.random_control_alphas:
            v_rand = sample_random_direction(d_model, target_norm=1.0, seed=args.seed + int(alpha*10))
            iv = [{"block_idx": best_block, "v_c": v_rand, "alpha": alpha}]
            print(f"[c1]   generating {concept} RANDOM α={alpha:+.1f}...")
            t0 = time.time()
            texts = generate_with_steering(
                model, tokenizer, chat_prompts, iv,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                device=args.device,
            )
            print(f"[c1]     took {time.time()-t0:.1f}s")
            for p_raw, t in zip(prompts_raw, texts):
                gens.append({"prompt": p_raw, "output": t, "condition": "random", "alpha": alpha})

        # Persist generations
        gen_path = out_dir / f"generations_{concept}.jsonl"
        with open(gen_path, "w") as f:
            for g in gens:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"[c1] wrote {len(gens)} generations → {gen_path}")

        # Judge
        if not args.skip_judge:
            print(f"[c1]   judging {len(gens)} generations with GPT-4o...")
            t0 = time.time()
            scored = []
            for i, g in enumerate(gens):
                jr = judge_completion(
                    concept=concept, prompt=g["prompt"], completion=g["output"],
                    positive_desc=meta_cd["positive_desc"],
                    negative_desc=meta_cd["negative_desc"],
                )
                scored.append({**g, "score": jr["score"], "judge_raw": jr["raw"]})
                if (i + 1) % 50 == 0:
                    print(f"[c1]     judged {i+1}/{len(gens)}")
            print(f"[c1]   judging took {time.time()-t0:.1f}s")
            with open(out_dir / f"scored_{concept}.jsonl", "w") as f:
                for s in scored:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            # Aggregate: per-condition-α mean rubric
            def agg(cond, alpha):
                sel = [s["score"] for s in scored
                       if s["condition"] == cond and s["alpha"] == alpha and s["score"] is not None]
                return {"mean": float(np.mean(sel)) if sel else None,
                        "std": float(np.std(sel)) if sel else None,
                        "n": len(sel)}
            agg_rfm = {a: agg("rfm", a) for a in args.alphas}
            agg_rand = {a: agg("random", a) for a in args.random_control_alphas}
            baseline_mean = agg_rfm[0.0]["mean"]

            # Pick α* = signed α maximising |shift| among RFM-only,
            # subject to length guardrail (|steered|<=2*|baseline|).
            baseline_lens = [len(g["output"]) for g in gens if g["condition"] == "rfm" and g["alpha"] == 0.0]
            baseline_len_mean = float(np.mean(baseline_lens)) if baseline_lens else 0.0
            def alpha_ok(a):
                lens = [len(g["output"]) for g in gens if g["condition"]=="rfm" and g["alpha"]==a]
                m = float(np.mean(lens)) if lens else 0.0
                return m <= 2.5 * baseline_len_mean  # 2.5x is a soft cap (2x guardrail + some slack)
            valid_alphas = [a for a in args.alphas if a != 0.0 and alpha_ok(a) and agg_rfm[a]["mean"] is not None]
            if not valid_alphas:
                alpha_star = None; delta_star = None
            else:
                deltas = {a: agg_rfm[a]["mean"] - baseline_mean for a in valid_alphas}
                alpha_star = max(deltas, key=lambda a: abs(deltas[a]))
                delta_star = deltas[alpha_star]

            summary["concepts"][concept] = {
                "best_block": best_block, "d_model": d_model,
                "baseline_mean_rubric": baseline_mean,
                "baseline_len_mean": baseline_len_mean,
                "agg_rfm": {str(a): v for a, v in agg_rfm.items()},
                "agg_random": {str(a): v for a, v in agg_rand.items()},
                "alpha_star": alpha_star, "delta_star": delta_star,
                "n_prompts": len(prompts_raw),
            }
        else:
            summary["concepts"][concept] = {
                "best_block": best_block, "d_model": d_model,
                "n_prompts": len(prompts_raw), "skip_judge": True,
            }

        free_cuda()

    summary["end_time"] = time.time()
    summary["total_time_s"] = time.time() - summary["start_time"]
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[c1] DONE → {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
