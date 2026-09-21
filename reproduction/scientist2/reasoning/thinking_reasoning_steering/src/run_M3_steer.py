"""M3 — Causal Intervention — α sweep dose-response + off-target specificity.

For each (behaviour, α):
  1. Additive steering at L*(b) with coefficient α·σ_L*
  2. Generate on all 500 tasks with greedy decoding + fixed seed
  3. LLM-judge behaviour-rate across all 4 behaviours + coherence flag +
     accuracy vs. gold

Outputs runs/M3_steer/results_summary.json + plots.
"""
from __future__ import annotations
import os, sys, json, time, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, SteeringHook, format_task_prompt, generate_with_optional_steering
from llm_judge import score_chain

JUDGE_MAX_PARALLEL = 12  # DMX handles this level of concurrency comfortably

BEHAVIOURS = ["expressing_uncertainty", "generating_validation_examples", "backtracking", "self-correction"]
ALPHA_GRID = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]


def load_directions(m1_dir: str) -> dict[str, dict]:
    """Return {behaviour: {direction: tensor, layer: int, sigma_proj: float}}."""
    m1 = json.load(open(os.path.join(m1_dir, "results.json")))
    out = {}
    for b, info in m1["per_behaviour"].items():
        v_path = info.get("v_b_path")
        if v_path is None or not os.path.exists(v_path):
            print(f"  skip {b}: no direction file"); continue
        rec = torch.load(v_path)
        out[b] = {
            "direction": rec["direction"],
            "layer": int(info["L_star"]),
            "sigma_proj": float(info.get("sigma_proj_at_L_star", rec.get("sigma_proj", 1.0))),
        }
    return out


def load_bench(path: str, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    if limit is not None:
        rows = rows[:limit]
    return rows


def run_and_score(model, tok, prompts: list[str], gold: list[str], direction, coef, layer_idx,
                  behaviours_to_score: list[str], seed: int, max_new_tokens: int,
                  batch: int = 4, tag: str = "",
                  judge_pool: ThreadPoolExecutor | None = None) -> list[dict]:
    """Generate on GPU (serially by batch) + score via parallel judge calls."""
    steer = SteeringHook(direction, coef) if abs(coef) > 1e-9 else None
    # Phase 1: run all generations
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
                "behaviour_rates": sc["behaviour_rates"],
                "coherent": sc["coherent"],
                "accuracy": sc["accuracy"],
                "final_answer_extracted": sc["final_answer_extracted"],
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


def summarize(per_task: list[dict], behaviours_to_score: list[str]) -> dict:
    n = len(per_task)
    n_coh = sum(1 for r in per_task if r.get("coherent") == 1)
    n_acc = [r["accuracy"] for r in per_task if r.get("accuracy") in (0, 1)]
    per_b = {}
    for b in behaviours_to_score:
        vals = [r["behaviour_rates"].get(b) for r in per_task
                if "behaviour_rates" in r and r["behaviour_rates"].get(b) is not None
                and r.get("coherent") == 1]
        vals = [v for v in vals if v is not None]
        per_b[b] = float(np.mean(vals)) if vals else float("nan")
    return {
        "n_tasks": n,
        "coherence_rate": n_coh / n if n else 0,
        "accuracy_rate": float(np.mean(n_acc)) if n_acc else None,
        "behaviour_rates": per_b,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--m1_dir", default="runs/M1_locate")
    ap.add_argument("--bench", default="data/benchmark/benchmark_500.jsonl")
    ap.add_argument("--out_dir", default="runs/M3_steer")
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHA_GRID)
    ap.add_argument("--behaviours", nargs="+", default=BEHAVIOURS)
    ap.add_argument("--n_bench", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity", type=int, default=0)
    args = ap.parse_args()

    if args.sanity > 0:
        args.n_bench = args.sanity
        args.alphas = [-2.0, 0.0, 2.0]
        args.behaviours = args.behaviours[:1]

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    directions = load_directions(args.m1_dir)
    bench = load_bench(args.bench, args.n_bench)
    print(f"[M3] {len(bench)} tasks, {len(args.behaviours)} behaviours, {len(args.alphas)} alphas")

    model, tok = load_model(args.model_path)
    prompts = [format_task_prompt(tok, r["prompt"]) for r in bench]
    gold = [r["gold_answer"] for r in bench]

    # Cache the α=0 baseline once (per model+prompts+seed), reuse across behaviours
    results = {"behaviours": {}, "meta": {"alphas": args.alphas, "n_bench": len(bench)}}
    judge_pool = ThreadPoolExecutor(max_workers=JUDGE_MAX_PARALLEL)

    # Baseline: run α=0 once (no steering hook)
    print("[M3] baseline (α=0) ...")
    dummy_dir = torch.zeros(model.config.hidden_size)
    baseline_per_task = run_and_score(
        model, tok, prompts, gold, dummy_dir, 0.0, 0, BEHAVIOURS, args.seed,
        args.max_new_tokens, batch=args.batch, tag="baseline", judge_pool=judge_pool,
    )
    baseline_summary = summarize(baseline_per_task, BEHAVIOURS)
    print(f"[M3] baseline: coh={baseline_summary['coherence_rate']:.3f} acc={baseline_summary['accuracy_rate']} rates={baseline_summary['behaviour_rates']}")
    results["baseline"] = {"summary": baseline_summary, "per_task": baseline_per_task}

    # Save baseline preemptively so a later crash doesn't lose it
    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    for b in args.behaviours:
        if b not in directions:
            print(f"[M3] skip {b}: no direction"); continue
        d = directions[b]
        results["behaviours"][b] = {"L_star": d["layer"], "sigma_proj": d["sigma_proj"], "alphas": {}}
        for alpha in args.alphas:
            if abs(alpha) < 1e-9:
                # already computed as baseline
                results["behaviours"][b]["alphas"]["0.0"] = {
                    "summary": baseline_summary,
                    "note": "baseline reused",
                }
                continue
            coef = alpha * d["sigma_proj"]
            tag = f"{b}_a{alpha:+.1f}"
            print(f"[M3] {tag} coef={coef:.3f} @ L{d['layer']}")
            per_task = run_and_score(
                model, tok, prompts, gold, d["direction"], coef, d["layer"],
                BEHAVIOURS, args.seed, args.max_new_tokens, batch=args.batch, tag=tag,
                judge_pool=judge_pool,
            )
            summary = summarize(per_task, BEHAVIOURS)
            summary["Δrate_on_target"] = summary["behaviour_rates"].get(b, 0) - baseline_summary["behaviour_rates"].get(b, 0)
            summary["Δrate_off_target"] = {
                other: summary["behaviour_rates"].get(other, 0) - baseline_summary["behaviour_rates"].get(other, 0)
                for other in BEHAVIOURS if other != b
            }
            # keep the first 3 previews so we can eyeball collapse without
            # bloating the JSON on 500-task runs
            results["behaviours"][b]["alphas"][f"{alpha:.1f}"] = {
                "summary": summary,
                "preview_chains": [{"task_idx": r.get("task_idx"),
                                    "preview": r.get("chain_preview", ""),
                                    "coherent": r.get("coherent"),
                                    "final_answer_extracted": r.get("final_answer_extracted")}
                                   for r in per_task[:3]],
            }
            # Incremental save
            with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Post-hoc analysis: sign check, Spearman, specificity
    from scipy.stats import spearmanr
    analysis = {}
    for b, br in results["behaviours"].items():
        if "alphas" not in br: continue
        # Build arrays
        alpha_vals = []
        rate_vals = []
        coh_vals = []
        acc_vals = []
        for a_str, entry in br["alphas"].items():
            a = float(a_str)
            s = entry["summary"]
            alpha_vals.append(a)
            rate_vals.append(s["behaviour_rates"].get(b))
            coh_vals.append(s["coherence_rate"])
            acc_vals.append(s.get("accuracy_rate"))
        order = np.argsort(alpha_vals)
        alpha_vals = np.array(alpha_vals)[order].tolist()
        rate_vals = np.array(rate_vals, dtype=float)[order].tolist()
        coh_vals = np.array(coh_vals, dtype=float)[order].tolist()
        acc_vals = np.array([np.nan if v is None else v for v in acc_vals], dtype=float)[order].tolist()
        # Exclude alpha with coherence < 0.9 (M3 predicate: "> 10% gibberish excluded")
        keep = [i for i, c in enumerate(coh_vals) if c >= 0.9]
        if len(keep) >= 4:
            rho, p = spearmanr([alpha_vals[i] for i in keep], [rate_vals[i] for i in keep])
        else:
            rho, p = float("nan"), float("nan")
        # sign check: positive α gives higher rate than baseline; negative α lower
        base_rate = baseline_summary["behaviour_rates"].get(b, 0)
        pos_alphas = [alpha_vals[i] for i in keep if alpha_vals[i] > 0]
        neg_alphas = [alpha_vals[i] for i in keep if alpha_vals[i] < 0]
        sign_pos = any(rate_vals[alpha_vals.index(a)] > base_rate for a in pos_alphas)
        sign_neg = any(rate_vals[alpha_vals.index(a)] < base_rate for a in neg_alphas)
        # operating α_op = smallest |α| that flips sign both ways
        op_alpha = None
        for cand in sorted(set(abs(a) for a in alpha_vals if a != 0)):
            up = [a for a in pos_alphas if abs(a) == cand]
            dn = [a for a in neg_alphas if abs(a) == cand]
            if up and dn:
                if rate_vals[alpha_vals.index(up[0])] > base_rate and rate_vals[alpha_vals.index(dn[0])] < base_rate:
                    op_alpha = cand; break
        analysis[b] = {
            "alpha_grid_kept": [alpha_vals[i] for i in keep],
            "rate_grid_kept": [rate_vals[i] for i in keep],
            "spearman_rho": rho,
            "spearman_p": p,
            "sign_positive_alpha_amplifies": sign_pos,
            "sign_negative_alpha_suppresses": sign_neg,
            "operating_alpha": op_alpha,
            "baseline_rate": base_rate,
            "coherence_by_alpha": dict(zip(alpha_vals, coh_vals)),
            "accuracy_by_alpha": dict(zip(alpha_vals, acc_vals)),
        }
        # off-target specificity at operating α (or |α|=1 if no op)
        target_alpha = op_alpha if op_alpha is not None else 1.0
        if str(target_alpha) in br["alphas"] or f"{target_alpha:.1f}" in br["alphas"] or f"{-target_alpha:.1f}" in br["alphas"]:
            for a_str, entry in br["alphas"].items():
                if abs(float(a_str)) == target_alpha:
                    on = entry["summary"].get("Δrate_on_target", 0)
                    off = entry["summary"].get("Δrate_off_target", {})
                    off_max = max((abs(v) for v in off.values()), default=0)
                    spec = 1 - (off_max / (abs(on) + 1e-6))
                    analysis[b].setdefault("specificity_at_op", {})[a_str] = {
                        "on_target_Δ": on,
                        "max_off_target_|Δ|": off_max,
                        "specificity": spec,
                    }

    results["analysis"] = analysis
    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plots
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # dose response
        fig, ax = plt.subplots(figsize=(9, 5))
        for b, br in results["behaviours"].items():
            xs, ys = [], []
            for a_str, entry in sorted(br["alphas"].items(), key=lambda kv: float(kv[0])):
                xs.append(float(a_str))
                ys.append(entry["summary"]["behaviour_rates"].get(b))
            ax.plot(xs, ys, marker="o", label=b)
        ax.axhline(y=baseline_summary["behaviour_rates"].get(BEHAVIOURS[0], 0), color="gray", linestyle=":", label="baseline (first behav)")
        ax.set_xlabel("α (σ units)"); ax.set_ylabel("behaviour rate")
        ax.set_title("M3: dose-response"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "dose_response.png"), dpi=120); plt.close()
        # coherence vs alpha
        fig, ax = plt.subplots(figsize=(9, 4))
        for b, br in results["behaviours"].items():
            xs, ys = [], []
            for a_str, entry in sorted(br["alphas"].items(), key=lambda kv: float(kv[0])):
                xs.append(float(a_str)); ys.append(entry["summary"]["coherence_rate"])
            ax.plot(xs, ys, marker="o", label=b)
        ax.set_xlabel("α (σ units)"); ax.set_ylabel("coherence rate")
        ax.set_title("M3: coherence vs α"); ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "coherence_vs_alpha.png"), dpi=120); plt.close()
    except Exception as e:
        print(f"plot failed: {e!r}")

    cost = {
        "run_id": "M3_steer",
        "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
        "wall_seconds_total": time.time() - t0,
        "n_bench": len(bench),
        "alphas": args.alphas,
        "behaviours": args.behaviours,
    }
    with open(os.path.join(args.out_dir, "cost.json"), "w") as f:
        json.dump(cost, f, indent=2)
    print(f"[M3] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
