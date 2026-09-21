"""M2 — small-pool contrastive-extraction sweep.

Consumes:
  runs/M1_locate/activations.npz
  runs/M1_locate/results.json     (for L*(b))
  data/contrast/auxiliary_corpus.jsonl
  data/benchmark/benchmark_500.jsonl  (only the first --n_bench_subset tasks)

For each (behaviour, n_pairs, seed) in the grid:
  1. sample n_pairs positive + n_pairs negative from the train pool
  2. compute mean-difference direction v(n_pairs, seed) at L*(b)
  3. split-half stability: half A vs half B cosine
  4. steering-effect preservation: apply v(n_pairs, seed) at fixed α on
     the 50-task subset; compare Δ behaviour-rate to Δ at n_pairs=200

Outputs:
  runs/M2_smallpool/results_summary.json
  runs/M2_smallpool/plot_n_pairs_vs_stability.png
"""
from __future__ import annotations
import os, sys, json, time, argparse, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, SteeringHook, format_task_prompt, generate_with_optional_steering
from llm_judge import score_chain

BEHAVIOURS = ["expressing_uncertainty", "generating_validation_examples", "backtracking", "self-correction"]
N_PAIRS_GRID = [10, 25, 50, 100, 200]
SEEDS = [0, 1, 2]


def load_all():
    npz = np.load("runs/M1_locate/activations.npz", allow_pickle=True)
    layer_ids = npz["layer_ids"].tolist()
    X_by_layer = {int(li): npz[f"L{li}"] for li in layer_ids}
    m1 = json.load(open("runs/M1_locate/results.json"))
    corpus = [json.loads(l) for l in open("data/contrast/auxiliary_corpus.jsonl")]
    meta = json.load(open("runs/M1_locate/meta.json"))
    corpus_id_order = meta["corpus_ids"]
    id2idx = {cid: i for i, cid in enumerate(corpus_id_order)}
    # only rows that made it into the M1 activation cache
    corpus = [r for r in corpus if r["id"] in id2idx]
    corpus.sort(key=lambda r: id2idx[r["id"]])
    return X_by_layer, m1, corpus, meta


def mean_diff(X_pos: np.ndarray, X_neg: np.ndarray) -> np.ndarray:
    """Raw mean-difference direction (NOT unit-normalized). Use `to_unit()` to
    convert before applying as α·σ·u steering."""
    return (X_pos.mean(0) - X_neg.mean(0)).astype(np.float32)


def to_unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_bench_subset(bench_path: str, n: int) -> list[dict]:
    rows = [json.loads(l) for l in open(bench_path)]
    return rows[:n]


def measure_beh_rate(model, tok, prompts: list[str], direction: torch.Tensor, coef: float,
                     layer_idx: int, behaviour: str, gold: list[str | None],
                     max_new_tokens: int = 512, batch: int = 4) -> float:
    """Return fraction of chains where `behaviour` presence == 1 under steering."""
    steer = SteeringHook(direction, coef) if coef != 0 else None
    rates = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i+batch]
        gens = generate_with_optional_steering(
            model, tok, chunk, steer, layer_idx,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
        for txt, g in zip(gens, gold[i:i+batch]):
            try:
                sc = score_chain(txt, task_gold=g)
                rates.append(sc["behaviour_rates"].get(behaviour, 0))
            except Exception as e:
                print(f"  score failed: {e!r}")
    return float(np.mean(rates)) if rates else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--out_dir", default="runs/M2_smallpool")
    ap.add_argument("--n_bench_subset", type=int, default=50)
    ap.add_argument("--alpha_mid", type=float, default=1.5,
                    help="mid-grid alpha in σ units for steering-effect check")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_pairs_grid", type=int, nargs="+", default=N_PAIRS_GRID)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--sanity", type=int, default=0,
                    help="if >0: 1 behaviour × 2 n_pairs × 1 seed × sanity tasks")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    X_by_layer, m1, corpus, meta = load_all()

    # Bench subset
    n_bench = args.sanity if args.sanity > 0 else args.n_bench_subset
    bench = build_bench_subset("data/benchmark/benchmark_500.jsonl", n_bench)
    print(f"[M2] using bench subset of {len(bench)} tasks")

    # Only load the model for the steering-effect check (skip if pure stability check)
    model, tok = load_model(args.model_path)
    prompts = [format_task_prompt(tok, r["prompt"]) for r in bench]
    gold = [r["gold_answer"] for r in bench]

    # Split-half stability + steering-effect ratio per (behaviour, n_pairs, seed)
    results = {}
    behaviours = BEHAVIOURS if args.sanity == 0 else BEHAVIOURS[:1]
    n_pairs_grid = args.n_pairs_grid if args.sanity == 0 else args.n_pairs_grid[:2]
    seeds = args.seeds if args.sanity == 0 else args.seeds[:1]

    for b in behaviours:
        pb = m1["per_behaviour"].get(b, {})
        L_star = pb.get("L_star")
        if L_star is None:
            print(f"[M2] skip {b}: no L_star from M1"); continue
        sigma = pb.get("sigma_proj_at_L_star", 1.0)
        X = X_by_layer[int(L_star)]
        y = np.array([r["behaviours"].get(b, 0) for r in corpus], dtype=np.int64)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        # split M1's train/test — reuse M1's train_idx by rebuilding it
        rng = np.random.default_rng(0)
        idx_all = np.arange(len(corpus)); rng.shuffle(idx_all)
        ntrain = int(len(corpus) * meta["train_frac"])
        train_pool = set(int(x) for x in idx_all[:ntrain])
        pos_pool = np.array([i for i in pos_idx if int(i) in train_pool])
        neg_pool = np.array([i for i in neg_idx if int(i) in train_pool])
        n_max = min(len(pos_pool), len(neg_pool))
        print(f"[M2] {b}: L={L_star} σ={sigma:.3f}; pos_pool={len(pos_pool)} neg_pool={len(neg_pool)}")

        # First compute the reference direction at n_pairs = max (or 200 capped)
        n_ref = min(n_max, 200)
        rng_ref = np.random.default_rng(999)
        p_ref = rng_ref.choice(pos_pool, size=n_ref, replace=False)
        n_ref_neg = rng_ref.choice(neg_pool, size=n_ref, replace=False)
        v_ref = mean_diff(X[p_ref], X[n_ref_neg])
        u_ref = to_unit(v_ref)
        u_ref_t = torch.from_numpy(u_ref)

        # Baseline (α=0) behaviour rate — measure once, reuse
        base_rate = measure_beh_rate(model, tok, prompts, u_ref_t, 0.0,
                                     int(L_star), b, gold,
                                     max_new_tokens=args.max_new_tokens, batch=args.batch)
        print(f"[M2] {b} baseline rate (α=0) = {base_rate:.3f}")
        # Reference rate change at large-pool
        ref_rate = measure_beh_rate(model, tok, prompts, u_ref_t, args.alpha_mid * sigma,
                                    int(L_star), b, gold,
                                    max_new_tokens=args.max_new_tokens, batch=args.batch)
        ref_delta = ref_rate - base_rate
        print(f"[M2] {b} reference Δrate at α={args.alpha_mid}σ (n_ref={n_ref}) = {ref_delta:+.3f}")

        results.setdefault(b, {})["baseline_rate"] = base_rate
        results[b]["reference_delta"] = ref_delta
        results[b]["reference_n_pairs"] = n_ref
        results[b]["sigma_proj"] = sigma
        results[b]["L_star"] = int(L_star)
        results[b]["sweep"] = []

        for n_pairs in n_pairs_grid:
            if n_pairs > n_max:
                print(f"  skip n_pairs={n_pairs} > n_max={n_max}")
                continue
            for seed in seeds:
                rng_s = np.random.default_rng(seed * 10 + n_pairs)
                p_idx = rng_s.choice(pos_pool, size=n_pairs, replace=False)
                n_idx = rng_s.choice(neg_pool, size=n_pairs, replace=False)
                v = mean_diff(X[p_idx], X[n_idx])
                u = to_unit(v)
                # split-half stability
                half = n_pairs // 2
                if half >= 2:
                    v_a = mean_diff(X[p_idx[:half]], X[n_idx[:half]])
                    v_b = mean_diff(X[p_idx[half:]], X[n_idx[half:]])
                    splithalf = cos(v_a, v_b)
                else:
                    splithalf = float("nan")
                # cosine with reference (using raw v is fine — cos ignores norm)
                cos_ref = cos(v, v_ref)
                # steering-effect delta (at mid alpha)
                v_t = torch.from_numpy(u)
                rate = measure_beh_rate(model, tok, prompts, v_t, args.alpha_mid * sigma,
                                        int(L_star), b, gold,
                                        max_new_tokens=args.max_new_tokens, batch=args.batch)
                delta = rate - base_rate
                ratio = delta / ref_delta if abs(ref_delta) > 1e-3 else float("nan")
                row = {
                    "behaviour": b,
                    "n_pairs": int(n_pairs),
                    "seed": int(seed),
                    "split_half_cos": splithalf,
                    "cos_to_reference": cos_ref,
                    "steered_rate": rate,
                    "delta_rate": delta,
                    "ratio_to_large_pool": ratio,
                }
                results[b]["sweep"].append(row)
                print(f"  n={n_pairs} seed={seed}: splithalf={splithalf:.3f} cos_ref={cos_ref:.3f} Δrate={delta:+.3f} ratio={ratio:.3f}")

    # Save
    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Try to plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for b, br in results.items():
            if "sweep" not in br: continue
            xs = sorted(set(r["n_pairs"] for r in br["sweep"]))
            ys, err = [], []
            for x in xs:
                vals = [r["split_half_cos"] for r in br["sweep"] if r["n_pairs"] == x and r["split_half_cos"] == r["split_half_cos"]]
                if vals:
                    ys.append(np.mean(vals)); err.append(np.std(vals))
                else:
                    ys.append(np.nan); err.append(0)
            ax.errorbar(xs, ys, yerr=err, marker="o", label=b)
        ax.set_xscale("log"); ax.set_xlabel("n_pairs"); ax.set_ylabel("split-half cosine")
        ax.set_title("M2: small-pool direction stability"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "plot_n_pairs_vs_stability.png"), dpi=120)
        plt.close()
    except Exception as e:
        print(f"plot failed: {e!r}")

    cost = {
        "run_id": "M2_smallpool",
        "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
        "wall_seconds_total": time.time() - t0,
        "n_bench_subset": len(bench),
        "n_pairs_grid": n_pairs_grid,
        "seeds": seeds,
        "behaviours": behaviours,
    }
    with open(os.path.join(args.out_dir, "cost.json"), "w") as f:
        json.dump(cost, f, indent=2)
    print(f"[M2] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
