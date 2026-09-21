"""M3 fix for C3 (iteration 1) — expanded alpha grid + random-direction control + plateau verification.

Addresses the C3 INCONCLUSIVE verdict per verify/C3_dose_response_causal_control/main_experiment_audit/MECHANISM_AUDIT.md:
  FAIL reason 1: alpha range [-2,+2]sigma does NOT span 3 OOM
  FAIL reason 2: no random-direction control at n_random >= 30
  FAIL reason 3: alpha_op=0.5sigma at plateau edge (not middle)
  FAIL reason 4: 3/4 behaviours zero-rate (this is data insufficiency, not fixable at iteration stage)

Scope:
  - Focus on expressing_uncertainty ONLY (the only behaviour with real signal at n=60).
  - Alpha grid expanded to span ~2 OOM within coherence range: [-3, -1, -0.3, -0.1, 0, +0.1, +0.3, +1, +3] * sigma_proj.
    (Coherence collapse observed at +/- 2sigma in prior run; +/- 3sigma tests the boundary.
     +/- 0.1sigma probes the low-alpha regime for plateau shape.)
  - Random-direction control: sample n_random=30 unit vectors at L*=29, apply at alpha=+1sigma
    (the strongest coherent point), compare on-target Delta-rate distribution vs learned direction.
  - Plateau verification: fine-grained [0.3, 0.5, 0.7, 1.0, 1.5, 2.0] * sigma at n=30 tasks to
    characterise the +alpha side of the response curve.

Outputs: runs/iteration_round_1/M3_C3_expand/{results_summary.json, dose_response_expanded.png,
                                                 random_direction_control.json, plateau_scan.json,
                                                 cost.json}
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

TARGET_BEHAVIOUR = "expressing_uncertainty"
ALL_BEHAVIOURS = ["expressing_uncertainty", "generating_validation_examples", "backtracking", "self-correction"]

# Expanded alpha grid: span low-alpha regime + high-alpha boundary; 9 points, ~2 OOM
# The 3-OOM catalogue target (0.1, 0.3, 1, 3, 10) is not achievable on Llama-8B L29
# because coherence collapses at 2-3 sigma; we go as wide as coherence permits + probe low.
ALPHA_GRID = [-3.0, -1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0, 3.0]

# Random-direction control: at plateau alpha = 1.0 sigma (strongest coherent point)
RANDOM_CONTROL_ALPHA = 1.0
N_RANDOM_DIRECTIONS = 30

# Plateau verification: fine-grained positive-alpha scan on n=30 tasks
PLATEAU_ALPHAS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
PLATEAU_N_TASKS = 30


def load_target_direction(m1_dir: str, behaviour: str) -> dict:
    m1 = json.load(open(os.path.join(m1_dir, "results.json")))
    info = m1["per_behaviour"][behaviour]
    rec = torch.load(info["v_b_path"])
    return {
        "direction": rec["direction"],
        "layer": int(info["L_star"]),
        "sigma_proj": float(info.get("sigma_proj_at_L_star", rec.get("sigma_proj", 1.0))),
    }


def load_bench(path: str, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    if limit is not None:
        rows = rows[:limit]
    return rows


def compute_mean_residual_norm(m1_dir: str, layer: int) -> float:
    """Read cached activations, return mean ||h|| at that layer over the extract split."""
    d = np.load(os.path.join(m1_dir, "activations.npz"), allow_pickle=True)
    key = f"L{layer}"
    if key not in d.files:
        raise KeyError(f"{key} not in activations.npz; available: {d.files[:5]}...")
    H = d[key]  # (n, hidden)
    norms = np.linalg.norm(H.astype(np.float32), axis=1)
    return float(norms.mean())


def run_and_score(model, tok, prompts, gold, direction, coef, layer_idx,
                  seed, max_new_tokens, batch, tag, judge_pool):
    """Return list of per-task {behaviour_rates, coherent, accuracy, chain_preview}."""
    steer = SteeringHook(direction, coef) if abs(coef) > 1e-9 else None
    all_gens = [None] * len(prompts)
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i+batch]
        gens = generate_with_optional_steering(
            model, tok, chunk, steer, layer_idx,
            max_new_tokens=max_new_tokens, do_sample=False, seed=seed,
        )
        for j, txt in enumerate(gens):
            all_gens[i + j] = txt
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
            }
        except Exception as e:
            return idx, {"task_idx": idx, "score_error": str(e)}
    for idx, rec in judge_pool.map(_score, range(len(prompts))):
        per_task[idx] = rec
    return per_task


def summarize(per_task, behaviour_of_interest):
    n = len(per_task)
    n_coh = sum(1 for r in per_task if r.get("coherent") == 1)
    accs = [r["accuracy"] for r in per_task if r.get("accuracy") in (0, 1) and r.get("coherent") == 1]
    br_vals = [r["behaviour_rates"].get(behaviour_of_interest) for r in per_task
               if r.get("coherent") == 1 and "behaviour_rates" in r
               and r["behaviour_rates"].get(behaviour_of_interest) is not None]
    return {
        "n_tasks": n,
        "n_coherent": n_coh,
        "coherence_rate": n_coh / n if n else 0,
        "accuracy_rate": float(np.mean(accs)) if accs else None,
        "behaviour_rate": float(np.mean(br_vals)) if br_vals else float("nan"),
    }


def sample_random_direction(hidden_size: int, sigma_target: float, rng: np.random.Generator,
                             device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    """Random unit vector rescaled so its l2 norm equals `sigma_target` * unit-norm = 1
    -> we return a unit vector; the coef in the steering call handles the sigma scaling.

    Actually returning a unit direction so the same coef = alpha * sigma_proj applies —
    matches the CAA convention for a fair comparison.
    """
    v = rng.standard_normal(hidden_size).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    return torch.tensor(v, dtype=dtype, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--m1_dir", default="runs/M1_locate")
    ap.add_argument("--bench", default="data/benchmark/benchmark_500.jsonl")
    ap.add_argument("--out_dir", default="runs/iteration_round_1/M3_C3_expand")
    ap.add_argument("--n_bench", type=int, default=60)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--skip_random_control", action="store_true")
    ap.add_argument("--skip_plateau", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # Load target direction + baseline info
    d_info = load_target_direction(args.m1_dir, TARGET_BEHAVIOUR)
    print(f"[M3-C3] target behaviour={TARGET_BEHAVIOUR}, L*={d_info['layer']}, sigma_proj={d_info['sigma_proj']:.4f}")

    bench = load_bench(args.bench, args.n_bench)
    print(f"[M3-C3] {len(bench)} tasks")

    model, tok = load_model(args.model_path)
    hidden_size = model.config.hidden_size
    prompts = [format_task_prompt(tok, r["prompt"]) for r in bench]
    gold = [r["gold_answer"] for r in bench]

    judge_pool = ThreadPoolExecutor(max_workers=JUDGE_MAX_PARALLEL)

    results = {
        "meta": {
            "target_behaviour": TARGET_BEHAVIOUR,
            "L_star": d_info["layer"],
            "sigma_proj": d_info["sigma_proj"],
            "hidden_size": hidden_size,
            "alpha_grid": ALPHA_GRID,
            "n_bench": len(bench),
            "n_random_directions": N_RANDOM_DIRECTIONS,
            "random_control_alpha": RANDOM_CONTROL_ALPHA,
        },
        "expanded_alpha_sweep": {},
        "random_direction_control": None,
        "plateau_scan": None,
    }

    # === Part 1: expanded alpha sweep on 60 tasks ===
    print(f"[M3-C3] Part 1: expanded alpha sweep")
    baseline_summary = None
    for alpha in ALPHA_GRID:
        coef = alpha * d_info["sigma_proj"]
        tag = f"a{alpha:+.2f}"
        print(f"  alpha={alpha:+.2f} coef={coef:+.3f}")
        per_task = run_and_score(
            model, tok, prompts, gold, d_info["direction"], coef, d_info["layer"],
            args.seed, args.max_new_tokens, args.batch, tag, judge_pool
        )
        summary = summarize(per_task, TARGET_BEHAVIOUR)
        results["expanded_alpha_sweep"][f"{alpha:.2f}"] = {
            "coef": coef, "summary": summary, "n_previews": min(3, len(per_task)),
            "preview": [{"task_idx": r.get("task_idx"), "preview": r.get("chain_preview", ""),
                         "coherent": r.get("coherent"), "behaviour_rate": r.get("behaviour_rates", {}).get(TARGET_BEHAVIOUR)}
                        for r in per_task[:3]],
        }
        if abs(alpha) < 1e-9:
            baseline_summary = summary
        # incremental save
        with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    base_rate = baseline_summary["behaviour_rate"] if baseline_summary else 0.0
    print(f"[M3-C3] baseline rate = {base_rate:.4f}, coh = {baseline_summary['coherence_rate']:.3f}")

    # Compute analysis: sign check, plateau region, Spearman on coherent subset
    from scipy.stats import spearmanr
    alpha_arr = np.array(ALPHA_GRID)
    rate_arr = np.array([results["expanded_alpha_sweep"][f"{a:.2f}"]["summary"]["behaviour_rate"] for a in ALPHA_GRID])
    coh_arr = np.array([results["expanded_alpha_sweep"][f"{a:.2f}"]["summary"]["coherence_rate"] for a in ALPHA_GRID])
    keep = coh_arr >= 0.9
    if keep.sum() >= 4:
        rho, p = spearmanr(alpha_arr[keep], rate_arr[keep])
    else:
        rho, p = float("nan"), float("nan")
    results["analysis"] = {
        "spearman_rho_coherent_subset": float(rho) if rho == rho else "nan",
        "spearman_p": float(p) if p == p else "nan",
        "n_coherent_alpha_points": int(keep.sum()),
        "baseline_rate": base_rate,
        "sign_check_positive_amplifies": bool(any(rate_arr[i] > base_rate for i, a in enumerate(ALPHA_GRID) if a > 0 and coh_arr[i] >= 0.9)),
        "sign_check_negative_suppresses": bool(any(rate_arr[i] < base_rate for i, a in enumerate(ALPHA_GRID) if a < 0 and coh_arr[i] >= 0.9)),
        "coherence_by_alpha": {f"{a:.2f}": float(c) for a, c in zip(ALPHA_GRID, coh_arr)},
        "rate_by_alpha": {f"{a:.2f}": float(r) for a, r in zip(ALPHA_GRID, rate_arr)},
    }
    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # === Part 2: random-direction control ===
    if not args.skip_random_control:
        print(f"[M3-C3] Part 2: random-direction control @ alpha={RANDOM_CONTROL_ALPHA}sigma (n_random={N_RANDOM_DIRECTIONS})")
        rng = np.random.default_rng(args.random_seed)
        random_results = []
        # Use a smaller task subset for random-direction control (n=20) to keep budget manageable
        n_rand_tasks = min(20, len(bench))
        rand_prompts = prompts[:n_rand_tasks]
        rand_gold = gold[:n_rand_tasks]
        target_coef = RANDOM_CONTROL_ALPHA * d_info["sigma_proj"]
        for k in range(N_RANDOM_DIRECTIONS):
            rand_dir = sample_random_direction(hidden_size, d_info["sigma_proj"], rng)
            per_task = run_and_score(
                model, tok, rand_prompts, rand_gold, rand_dir, target_coef, d_info["layer"],
                args.seed, args.max_new_tokens, args.batch, f"rand{k}", judge_pool
            )
            summary = summarize(per_task, TARGET_BEHAVIOUR)
            random_results.append({
                "k": k, "coherence_rate": summary["coherence_rate"],
                "behaviour_rate": summary["behaviour_rate"], "delta_rate_vs_baseline": summary["behaviour_rate"] - base_rate,
            })
            if (k + 1) % 5 == 0:
                print(f"    random {k+1}/{N_RANDOM_DIRECTIONS} done, brate={summary['behaviour_rate']:.3f}")
            # incremental save
            results["random_direction_control"] = {
                "alpha_sigma": RANDOM_CONTROL_ALPHA,
                "n_random_completed": k + 1,
                "n_tasks": n_rand_tasks,
                "target_rate_at_alpha1": results["expanded_alpha_sweep"]["1.00"]["summary"]["behaviour_rate"],
                "target_delta_vs_baseline": results["expanded_alpha_sweep"]["1.00"]["summary"]["behaviour_rate"] - base_rate,
                "random_results": random_results,
            }
            with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
                json.dump(results, f, indent=2, default=str)

        # analysis on completed random runs
        rand_deltas = np.array([r["delta_rate_vs_baseline"] for r in random_results])
        target_delta = results["random_direction_control"]["target_delta_vs_baseline"]
        results["random_direction_control"]["analysis"] = {
            "target_delta_vs_baseline": float(target_delta),
            "random_delta_mean": float(rand_deltas.mean()) if len(rand_deltas) else "nan",
            "random_delta_std": float(rand_deltas.std()) if len(rand_deltas) else "nan",
            "random_delta_max": float(rand_deltas.max()) if len(rand_deltas) else "nan",
            "random_delta_min": float(rand_deltas.min()) if len(rand_deltas) else "nan",
            "target_z_vs_random": (float((target_delta - rand_deltas.mean()) / (rand_deltas.std() + 1e-9))
                                    if len(rand_deltas) else "nan"),
            "learned_direction_specific": bool((rand_deltas.std() > 0) and (target_delta > rand_deltas.mean() + 2 * rand_deltas.std())),
        }
        with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    # === Part 3: plateau verification scan ===
    if not args.skip_plateau:
        print(f"[M3-C3] Part 3: plateau verification (fine-grained +alpha scan)")
        plateau_prompts = prompts[:PLATEAU_N_TASKS]
        plateau_gold = gold[:PLATEAU_N_TASKS]
        plateau_results = {}
        for alpha in PLATEAU_ALPHAS:
            coef = alpha * d_info["sigma_proj"]
            per_task = run_and_score(
                model, tok, plateau_prompts, plateau_gold, d_info["direction"], coef, d_info["layer"],
                args.seed, args.max_new_tokens, args.batch, f"plateau_a{alpha:.2f}", judge_pool
            )
            summary = summarize(per_task, TARGET_BEHAVIOUR)
            plateau_results[f"{alpha:.2f}"] = summary
            print(f"    plateau alpha={alpha:.2f} brate={summary['behaviour_rate']:.3f} coh={summary['coherence_rate']:.3f}")
            results["plateau_scan"] = {"alphas": PLATEAU_ALPHAS, "n_tasks": PLATEAU_N_TASKS, "results": plateau_results}
            with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        alphas = ALPHA_GRID
        rates = [results["expanded_alpha_sweep"][f"{a:.2f}"]["summary"]["behaviour_rate"] for a in alphas]
        cohs = [results["expanded_alpha_sweep"][f"{a:.2f}"]["summary"]["coherence_rate"] for a in alphas]
        ax[0].plot(alphas, rates, marker="o", color="C0", label=f"{TARGET_BEHAVIOUR} rate")
        ax[0].axhline(base_rate, color="gray", linestyle=":", label="baseline")
        ax[0].set_xlabel("alpha (sigma_proj units)"); ax[0].set_ylabel("behaviour rate")
        ax[0].set_title("Expanded alpha sweep (n=60)"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(alphas, cohs, marker="o", color="C1")
        ax[1].set_xlabel("alpha (sigma_proj units)"); ax[1].set_ylabel("coherence")
        ax[1].set_ylim(0, 1.05); ax[1].set_title("Coherence vs alpha"); ax[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "dose_response_expanded.png"), dpi=120); plt.close()
    except Exception as e:
        print(f"plot failed: {e!r}")

    cost = {
        "run_id": "M3_C3_expand_iter1",
        "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
        "wall_seconds_total": time.time() - t0,
        "n_bench": len(bench),
        "target_behaviour": TARGET_BEHAVIOUR,
        "alpha_grid": ALPHA_GRID,
        "n_random_directions": N_RANDOM_DIRECTIONS,
        "n_plateau_alphas": len(PLATEAU_ALPHAS),
    }
    with open(os.path.join(args.out_dir, "cost.json"), "w") as f:
        json.dump(cost, f, indent=2)
    print(f"[M3-C3] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
