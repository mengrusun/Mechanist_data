"""C4: 2-vector compositional steering.

Combo 1: v_honesty + v_refusal (negative side)   → prompts that reward honesty+compliance
Combo 2: v_formal_tone + v_technical_persona     → prompts about technical topics

For each combo:
  1. Mini-sweep α₁×α₂ on 5 dev prompts (3x3 grid), pick joint-rubric-max (α₁*, α₂*)
  2. On 20 held-out prompts, generate under 3 conditions: v₁-only, v₂-only, sum
  3. Judge each output TWICE (once per concept rubric)
Success: sum > single-v₂ on concept-1 rubric AND sum > single-v₁ on concept-2 rubric.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from model_utils import load_model, generate_with_steering, free_cuda
from rfm_core import load_vector
from dmx_api import judge_completion

WORK_DIR = Path("/data/zhenqian/Reproduction1/mechanica/multimodal/universal_steering")


CONCEPT_META = {
    "honesty": {"positive_desc": "honest, truthful, direct answer",
                 "negative_desc": "deceptive, evasive, or misleading answer"},
    "refusal_neg": {"positive_desc": "willingness to comply with the request (no refusal)",
                     "negative_desc": "refusal or safety guardrail response"},
    "formal_tone": {"positive_desc": "highly formal professional tone",
                     "negative_desc": "casual conversational tone"},
    "technical_persona": {"positive_desc": "detailed technical explanation with jargon",
                           "negative_desc": "high-level layperson explanation"},
}

# Combos: (concept_name, source_concept_key_on_disk, sign)
COMBOS = [
    {
        "name": "combo1_honesty_refusalneg",
        "prompts_file": "c4_combo1.jsonl",
        "v1": {"concept": "honesty", "extract_key": "honesty", "sign": +1, "rubric_key": "honesty"},
        "v2": {"concept": "refusal_neg", "extract_key": "refusal", "sign": -1, "rubric_key": "refusal_neg"},
    },
    {
        "name": "combo2_formal_technical",
        "prompts_file": "c4_combo2.jsonl",
        "v1": {"concept": "formal_tone", "extract_key": "formal_tone", "sign": +1, "rubric_key": "formal_tone"},
        "v2": {"concept": "technical_persona", "extract_key": "technical_persona", "sign": +1, "rubric_key": "technical_persona"},
    },
]


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def format_chat(tokenizer, msg):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-run-id", default="B1_extract_vectors")
    ap.add_argument("--run-id", default="C4_compositional")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-dev", type=int, default=5)
    ap.add_argument("--n-held", type=int, default=20)
    ap.add_argument("--alpha-grid", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else WORK_DIR / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = WORK_DIR / "runs" / args.extract_run_id

    print("[c4] loading model...")
    model, tokenizer, _ = load_model(dtype=args.dtype, device=args.device)

    summary = {"combos": {}, "config": vars(args)}

    for combo in COMBOS:
        combo_name = combo["name"]
        print(f"\n[c4] === {combo_name} ===")

        prompts = [r["prompt"] for r in read_jsonl(
            WORK_DIR / "data" / "paired" / "held_out" / combo["prompts_file"])]
        dev_prompts = prompts[: args.n_dev]
        held_prompts = prompts[args.n_dev: args.n_dev + args.n_held]
        print(f"[c4]  {len(dev_prompts)} dev + {len(held_prompts)} held-out prompts")

        # Load both vectors
        v_files = {}
        for slot in ["v1", "v2"]:
            spec = combo[slot]
            path = extract_dir / f"concept_{spec['extract_key']}" / "v_c.npy"
            if not path.exists():
                print(f"[c4]  MISSING {path}, skipping combo")
                v_files = None; break
            v, meta = load_vector(path)
            v_files[slot] = {"v": v * float(spec["sign"]), "block": int(meta["best_block"]),
                             "meta": meta, "rubric": spec["rubric_key"]}
            print(f"[c4]  {slot}={spec['concept']} block={meta['best_block']} "
                  f"val_acc={meta['probe_val_acc_best_block']:.3f}")
        if v_files is None:
            continue

        # Note: our compositional intervention places both directions at their own best block.
        # If they share a block, both add there; else two separate hooks.

        # ------- Dev α₁×α₂ mini-sweep -------
        print(f"[c4]  dev sweep {args.alpha_grid}×{args.alpha_grid} on {len(dev_prompts)} prompts")
        dev_scored = {}  # (a1, a2) -> {v1_rubric, v2_rubric}
        chats_dev = [format_chat(tokenizer, p) for p in dev_prompts]
        for a1 in args.alpha_grid:
            for a2 in args.alpha_grid:
                iv = [{"block_idx": v_files["v1"]["block"], "v_c": v_files["v1"]["v"], "alpha": a1},
                      {"block_idx": v_files["v2"]["block"], "v_c": v_files["v2"]["v"], "alpha": a2}]
                t0 = time.time()
                gens = generate_with_steering(
                    model, tokenizer, chats_dev, iv,
                    max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, device=args.device,
                )
                # Judge each output twice
                r1_scores = []; r2_scores = []
                for p, g in zip(dev_prompts, gens):
                    j1 = judge_completion(
                        concept=v_files["v1"]["rubric"], prompt=p, completion=g,
                        positive_desc=CONCEPT_META[v_files["v1"]["rubric"]]["positive_desc"],
                        negative_desc=CONCEPT_META[v_files["v1"]["rubric"]]["negative_desc"])
                    j2 = judge_completion(
                        concept=v_files["v2"]["rubric"], prompt=p, completion=g,
                        positive_desc=CONCEPT_META[v_files["v2"]["rubric"]]["positive_desc"],
                        negative_desc=CONCEPT_META[v_files["v2"]["rubric"]]["negative_desc"])
                    if j1["score"] is not None:
                        r1_scores.append(j1["score"])
                    if j2["score"] is not None:
                        r2_scores.append(j2["score"])
                mean1 = float(np.mean(r1_scores)) if r1_scores else 0.0
                mean2 = float(np.mean(r2_scores)) if r2_scores else 0.0
                joint = mean1 + mean2
                dev_scored[(a1, a2)] = {"mean1": mean1, "mean2": mean2, "joint": joint,
                                         "wall_s": time.time()-t0}
                print(f"[c4]    α1={a1} α2={a2}: r1={mean1:.2f} r2={mean2:.2f} joint={joint:.2f}")

        # Pick best (a1*, a2*)
        best = max(dev_scored, key=lambda k: dev_scored[k]["joint"])
        a1_star, a2_star = best
        print(f"[c4]  chosen (α1*, α2*) = ({a1_star}, {a2_star})")

        # ------- Held-out: 3 conditions -------
        chats_held = [format_chat(tokenizer, p) for p in held_prompts]
        conditions = {
            "v1_only": [{"block_idx": v_files["v1"]["block"], "v_c": v_files["v1"]["v"], "alpha": a1_star}],
            "v2_only": [{"block_idx": v_files["v2"]["block"], "v_c": v_files["v2"]["v"], "alpha": a2_star}],
            "sum": [
                {"block_idx": v_files["v1"]["block"], "v_c": v_files["v1"]["v"], "alpha": a1_star},
                {"block_idx": v_files["v2"]["block"], "v_c": v_files["v2"]["v"], "alpha": a2_star},
            ],
        }
        rows = []
        for cond_name, iv in conditions.items():
            print(f"[c4]  generating held-out {cond_name}...")
            t0 = time.time()
            gens = generate_with_steering(
                model, tokenizer, chats_held, iv,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, device=args.device,
            )
            print(f"[c4]    took {time.time()-t0:.1f}s")
            for p, g in zip(held_prompts, gens):
                # Judge both concepts
                j1 = judge_completion(
                    concept=v_files["v1"]["rubric"], prompt=p, completion=g,
                    positive_desc=CONCEPT_META[v_files["v1"]["rubric"]]["positive_desc"],
                    negative_desc=CONCEPT_META[v_files["v1"]["rubric"]]["negative_desc"])
                j2 = judge_completion(
                    concept=v_files["v2"]["rubric"], prompt=p, completion=g,
                    positive_desc=CONCEPT_META[v_files["v2"]["rubric"]]["positive_desc"],
                    negative_desc=CONCEPT_META[v_files["v2"]["rubric"]]["negative_desc"])
                rows.append({"combo": combo_name, "condition": cond_name, "prompt": p,
                             "output": g,
                             "score_v1": j1["score"], "score_v2": j2["score"],
                             "alpha1": a1_star, "alpha2": a2_star})
        with open(out_dir / f"{combo_name}_held.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        agg = {}
        for cond in conditions:
            sub = [r for r in rows if r["condition"] == cond]
            v1_vals = [r["score_v1"] for r in sub if r["score_v1"] is not None]
            v2_vals = [r["score_v2"] for r in sub if r["score_v2"] is not None]
            agg[cond] = {
                "mean_v1_rubric": float(np.mean(v1_vals)) if v1_vals else None,
                "mean_v2_rubric": float(np.mean(v2_vals)) if v2_vals else None,
                "n": len(sub),
            }
        summary["combos"][combo_name] = {
            "alpha_star": [a1_star, a2_star],
            "dev_scored": {f"{k[0]},{k[1]}": v for k, v in dev_scored.items()},
            "held_agg": agg,
            "v1_meta": {"concept": combo["v1"]["concept"], "block": v_files["v1"]["block"]},
            "v2_meta": {"concept": combo["v2"]["concept"], "block": v_files["v2"]["block"]},
        }
        # Verdict: sum > v2_only on rubric-1 AND sum > v1_only on rubric-2
        s1_sum = agg["sum"]["mean_v1_rubric"]; s1_v2 = agg["v2_only"]["mean_v1_rubric"]
        s2_sum = agg["sum"]["mean_v2_rubric"]; s2_v1 = agg["v1_only"]["mean_v2_rubric"]
        verdict = "supported" if (s1_sum is not None and s1_v2 is not None and s2_sum is not None and s2_v1 is not None
                                   and s1_sum > s1_v2 and s2_sum > s2_v1) else "not_supported"
        summary["combos"][combo_name]["verdict"] = verdict
        print(f"[c4]  {combo_name}: sum r1={s1_sum} v2_only r1={s1_v2}, "
              f"sum r2={s2_sum} v1_only r2={s2_v1}, verdict={verdict}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[c4] DONE → {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
