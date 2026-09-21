"""M4 — Tuning & Editing — steering vs. prompt engineering vs. Thinking Intervention.

Controllers (per behaviour):
  - steering_alpha_neg2 / _neg1 / _pos1 / _pos2 (from M3 direction @ L*(b))
  - prompt_suppress / prompt_amplify — NL instruction prepended to the user question
  - thinking_intervention_suppress / _amplify — "token insertion" style:
    injects a control token/phrase into the thinking scaffold via chat template
    (2503.24370-style: modify the assistant thinking prefix).

Reuses M3 α=0 baseline where available (via `--m3_dir`).

For each (behaviour, controller):
  1. Build prompts per controller.
  2. Generate on all 500 tasks; score behaviour-rate + accuracy.
  3. Aggregate {rate, accuracy} points into per-controller-family Pareto.
"""
from __future__ import annotations
import os, sys, json, time, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, SteeringHook, format_task_prompt, generate_with_optional_steering
from llm_judge import score_chain

JUDGE_MAX_PARALLEL = 12

BEHAVIOURS = ["expressing_uncertainty", "generating_validation_examples", "backtracking", "self-correction"]

STEERING_ALPHAS = {
    "steering_alpha_neg2": -2.0,
    "steering_alpha_neg1": -1.0,
    "steering_alpha_pos1": +1.0,
    "steering_alpha_pos2": +2.0,
}

NL_INSTRUCTIONS = {
    "expressing_uncertainty": {
        "amplify": "As you think through this, explicitly hedge on any step you're not fully sure about (say 'I'm not sure', 'this might be wrong', 'let me double-check', etc.) — signal every uncertainty you have while solving.",
        "suppress": "As you think through this, be fully confident in every step. Do NOT hedge, do not say 'I'm not sure' or 'possibly' or 'let me double-check' — assert every claim decisively.",
    },
    "generating_validation_examples": {
        "amplify": "As you think through this, explicitly construct small test cases or plug-in checks to validate each general rule or intermediate result (e.g., 'let me try n=2 to check', 'plugging in x=0 gives').",
        "suppress": "As you think through this, do NOT construct small test cases or plug-in checks. Solve the problem directly without generating validation examples.",
    },
    "backtracking": {
        "amplify": "As you think through this, if any early step feels wrong, explicitly abandon that line of reasoning and restart from an earlier point (say 'wait, that's not right', 'actually let me start over').",
        "suppress": "As you think through this, do NOT abandon or restart your reasoning. Even if a step feels wrong, continue forward — do not say 'wait, let me start over' or 'scratch that'.",
    },
    "self-correction": {
        "amplify": "As you think through this, actively check earlier steps for errors and issue explicit surgical corrections when you find one (e.g., 'oh, I had 3+4=8 above; it should be 7').",
        "suppress": "As you think through this, do NOT correct any earlier step. Assume your first attempt at each step is right and continue — do not say 'correcting myself' or 'I made an arithmetic error above'.",
    },
}

# Thinking-Intervention token phrases (2503.24370-style: modify the assistant's
# thinking prefix so the phrase appears before the model generates its own thoughts).
THINKING_INTERVENTIONS = {
    "expressing_uncertainty": {
        "amplify": "\nHmm, I need to be careful here — I'm not fully sure about some of these steps, let me flag my uncertainty as I go.\n",
        "suppress": "\nI'm confident I can solve this cleanly without any hedging.\n",
    },
    "generating_validation_examples": {
        "amplify": "\nLet me set up small test cases to validate each step as I go.\n",
        "suppress": "\nI'll solve this directly without constructing sanity-check examples.\n",
    },
    "backtracking": {
        "amplify": "\nI'll try one approach and, if it doesn't work, restart from scratch with a different one.\n",
        "suppress": "\nI'll commit to one approach and drive it through to completion without restarting.\n",
    },
    "self-correction": {
        "amplify": "\nI'll double-check every arithmetic step and explicitly correct any errors I find.\n",
        "suppress": "\nI'll trust my first attempt at each step without second-guessing.\n",
    },
}


def build_prompts(tok, tasks: list[dict], controller: str, behaviour: str) -> tuple[list[str], list[str], dict | None]:
    """Return (prompts_encoded_via_chat_template, gold_answers, steering_info).

    steering_info is None for non-steering controllers.
    """
    if controller in STEERING_ALPHAS:
        prompts = [format_task_prompt(tok, r["prompt"]) for r in tasks]
        gold = [r["gold_answer"] for r in tasks]
        return prompts, gold, {"alpha": STEERING_ALPHAS[controller]}
    if controller in ("prompt_suppress", "prompt_amplify"):
        instr = NL_INSTRUCTIONS[behaviour]["suppress" if controller == "prompt_suppress" else "amplify"]
        prompts = [format_task_prompt(tok, f"{instr}\n\n{r['prompt']}") for r in tasks]
        gold = [r["gold_answer"] for r in tasks]
        return prompts, gold, None
    if controller in ("thinking_intervention_suppress", "thinking_intervention_amplify"):
        phrase = THINKING_INTERVENTIONS[behaviour]["suppress" if controller == "thinking_intervention_suppress" else "amplify"]
        prompts = []
        for r in tasks:
            base = format_task_prompt(tok, r["prompt"])
            # 2503.24370-style: inject after the generation-prompt so it appears
            # as the *start* of the assistant's thinking.
            prompts.append(base + phrase)
        gold = [r["gold_answer"] for r in tasks]
        return prompts, gold, None
    raise ValueError(f"unknown controller: {controller}")


def load_direction(m1_dir: str, behaviour: str) -> tuple[torch.Tensor, int, float]:
    m1 = json.load(open(os.path.join(m1_dir, "results.json")))
    info = m1["per_behaviour"][behaviour]
    rec = torch.load(info["v_b_path"])
    return rec["direction"], int(info["L_star"]), float(info.get("sigma_proj_at_L_star", 1.0))


def run_controller(model, tok, tasks, controller, behaviour, m1_dir, seed, max_new_tokens, batch, tag,
                   judge_pool: ThreadPoolExecutor | None = None):
    prompts, gold, steer_info = build_prompts(tok, tasks, controller, behaviour)
    layer_idx, direction, coef = None, None, 0.0
    if steer_info is not None:
        direction, layer_idx, sigma = load_direction(m1_dir, behaviour)
        coef = steer_info["alpha"] * sigma
    steer = SteeringHook(direction, coef) if direction is not None and abs(coef) > 1e-9 else None
    # Phase 1: generate all
    all_gens = [None] * len(prompts)
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i+batch]
        gens = generate_with_optional_steering(
            model, tok, chunk, steer, layer_idx,
            max_new_tokens=max_new_tokens, do_sample=False, seed=seed,
        )
        for j, txt in enumerate(gens):
            all_gens[i + j] = txt
        if (i // batch) % 10 == 0:
            print(f"    [{tag}] gen {i+len(chunk)}/{len(prompts)}", flush=True)
    # Phase 2: parallel judge
    per_task = [None] * len(prompts)
    def _score(idx):
        try:
            sc = score_chain(all_gens[idx], task_gold=gold[idx])
            return idx, {
                "task_idx": idx,
                "chain_preview": all_gens[idx][:200],
                "behaviour_rate": sc["behaviour_rates"].get(behaviour, 0),
                "all_behaviour_rates": sc["behaviour_rates"],
                "coherent": sc["coherent"],
                "accuracy": sc["accuracy"],
            }
        except Exception as e:
            return idx, {"task_idx": idx, "score_error": str(e)}
    if judge_pool is None:
        with ThreadPoolExecutor(max_workers=JUDGE_MAX_PARALLEL) as ex:
            for idx, rec in ex.map(_score, range(len(prompts))):
                per_task[idx] = rec
    else:
        for idx, rec in judge_pool.map(_score, range(len(prompts))):
            per_task[idx] = rec
    return per_task


def summarize_ctrl(per_task, behaviour):
    n = len(per_task)
    coh_mask = [r for r in per_task if r.get("coherent") == 1]
    br = [r["behaviour_rate"] for r in coh_mask if r.get("behaviour_rate") is not None]
    acc = [r["accuracy"] for r in coh_mask if r.get("accuracy") in (0, 1)]
    return {
        "n": n,
        "coherence_rate": len(coh_mask) / n if n else 0,
        "behaviour_rate": float(np.mean(br)) if br else float("nan"),
        "accuracy": float(np.mean(acc)) if acc else None,
    }


def n_distinct_operating_points(points: list[tuple[float, float | None]],
                                eps_r: float = 0.05, eps_a: float = 0.01) -> int:
    """Count controllers whose (rate, accuracy) is separated from every other by
    ≥ eps_r on rate AND ≥ eps_a on accuracy (drop those with missing accuracy)."""
    kept = [p for p in points if p[1] is not None]
    keep_mask = []
    for i, (r, a) in enumerate(kept):
        distinct = True
        for j, (r2, a2) in enumerate(kept):
            if i == j: continue
            if abs(r - r2) < eps_r and abs(a - a2) < eps_a:
                distinct = False; break
        keep_mask.append(distinct)
    return int(sum(keep_mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--m1_dir", default="runs/M1_locate")
    ap.add_argument("--m3_dir", default="runs/M3_steer")
    ap.add_argument("--bench", default="data/benchmark/benchmark_500.jsonl")
    ap.add_argument("--out_dir", default="runs/M4_control_compare")
    ap.add_argument("--behaviours", nargs="+", default=BEHAVIOURS)
    ap.add_argument("--controllers", nargs="+", default=[
        "steering_alpha_neg2", "steering_alpha_neg1", "steering_alpha_pos1", "steering_alpha_pos2",
        "prompt_suppress", "prompt_amplify",
        "thinking_intervention_suppress", "thinking_intervention_amplify",
    ])
    ap.add_argument("--n_bench", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity", type=int, default=0)
    args = ap.parse_args()

    if args.sanity > 0:
        args.n_bench = args.sanity
        args.behaviours = args.behaviours[:1]
        args.controllers = ["steering_alpha_pos1", "prompt_amplify", "thinking_intervention_amplify"]

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    tasks = [json.loads(l) for l in open(args.bench)]
    if args.n_bench:
        tasks = tasks[:args.n_bench]
    print(f"[M4] {len(tasks)} tasks × {len(args.behaviours)} behaviours × {len(args.controllers)} controllers")

    model, tok = load_model(args.model_path)

    # Load M3 baseline if present
    m3_baseline = None
    m3_path = os.path.join(args.m3_dir, "results_summary.json")
    if os.path.exists(m3_path):
        m3 = json.load(open(m3_path))
        m3_baseline = m3.get("baseline", {}).get("summary")
        print(f"[M4] reusing M3 baseline")

    results = {"behaviours": {}, "meta": {"controllers": args.controllers, "n_bench": len(tasks)}}
    if m3_baseline is not None:
        results["m3_baseline"] = m3_baseline

    judge_pool = ThreadPoolExecutor(max_workers=JUDGE_MAX_PARALLEL)
    for b in args.behaviours:
        results["behaviours"][b] = {"controllers": {}}
        for ctrl in args.controllers:
            tag = f"{b}_{ctrl}"
            print(f"[M4] {tag}")
            per_task = run_controller(
                model, tok, tasks, ctrl, b, args.m1_dir, args.seed,
                args.max_new_tokens, args.batch, tag, judge_pool=judge_pool,
            )
            summary = summarize_ctrl(per_task, b)
            results["behaviours"][b]["controllers"][ctrl] = {
                "summary": summary,
                "per_task_preview": per_task[:5],
            }
            # incremental save
            with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
                json.dump(results, f, indent=2, default=str)

        # Granularity metric: n_distinct_operating_points per controller family
        fams = {
            "steering": [c for c in args.controllers if c.startswith("steering_alpha_")],
            "prompt": [c for c in args.controllers if c.startswith("prompt_")],
            "thinking_intervention": [c for c in args.controllers if c.startswith("thinking_intervention_")],
        }
        for fam, ctrls in fams.items():
            pts = []
            for c in ctrls:
                s = results["behaviours"][b]["controllers"].get(c, {}).get("summary", {})
                pts.append((s.get("behaviour_rate"), s.get("accuracy")))
            pts = [(r, a) for (r, a) in pts if r == r]  # drop NaN
            n_distinct = n_distinct_operating_points(pts)
            results["behaviours"][b].setdefault("granularity", {})[fam] = {
                "points": pts, "n_distinct": n_distinct,
            }

    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Pareto plot per behaviour
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for b, br in results["behaviours"].items():
            fig, ax = plt.subplots(figsize=(7, 5))
            color_map = {"steering": "C0", "prompt": "C1", "thinking_intervention": "C2"}
            for c, entry in br["controllers"].items():
                s = entry["summary"]
                fam = "steering" if c.startswith("steering_") else ("prompt" if c.startswith("prompt_") else "thinking_intervention")
                r, a = s.get("behaviour_rate"), s.get("accuracy")
                if r is None or r != r: continue
                if a is None: continue
                ax.scatter(r, a, s=90, color=color_map[fam], alpha=0.85)
                ax.annotate(c, (r, a), fontsize=7, xytext=(3, 3), textcoords="offset points")
            if m3_baseline is not None:
                r0 = m3_baseline["behaviour_rates"].get(b)
                a0 = m3_baseline.get("accuracy_rate")
                if r0 is not None and a0 is not None:
                    ax.scatter(r0, a0, s=120, marker="*", color="k", label="α=0 baseline")
            ax.set_xlabel("behaviour rate"); ax.set_ylabel("accuracy")
            ax.set_title(f"M4 Pareto: {b}"); ax.grid(alpha=0.3); ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"pareto_{b}.png"), dpi=120); plt.close()
    except Exception as e:
        print(f"plot failed: {e!r}")

    cost = {
        "run_id": "M4_control_compare",
        "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
        "wall_seconds_total": time.time() - t0,
        "n_bench": len(tasks),
        "behaviours": args.behaviours,
        "controllers": args.controllers,
    }
    with open(os.path.join(args.out_dir, "cost.json"), "w") as f:
        json.dump(cost, f, indent=2)
    print(f"[M4] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
