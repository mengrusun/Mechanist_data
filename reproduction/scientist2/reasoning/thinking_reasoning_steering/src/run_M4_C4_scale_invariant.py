"""M4 fix for C4 (iteration 1) — scale-invariant steering coefficient across models.

Addresses the C4 FAIL verdict per verify/C4_finer_than_prompt_engineering/variants/model-swap-qwen-14b/verdict.json:
  Qwen-14B sigma_proj = 886 vs Llama-8B sigma_proj = 10.52 (84x larger).
  At alpha=+/-1sigma the effective coef=886 -> complete incoherence on Qwen-14B, n_distinct(steering)=0.

Fix: replace `coef = alpha * sigma_proj` with a scale-invariant form:
  coef = alpha_frac * mean_residual_norm(L*)
where mean_residual_norm(L*) is computed per model on the extract-split activation cache
(or on a small fresh batch of chains if no cache exists for the swap model).

For a well-normalised residual stream, the ratio ||steer|| / ||residual|| = alpha_frac,
which is the same magnitude of intervention across models regardless of raw sigma differences.

Scope:
  - Reruns the 8-controller comparison on n=60 tasks for expressing_uncertainty
    (the only behaviour with signal at this task-count).
  - Runs on BOTH DeepSeek-R1-Distill-Llama-8B AND DeepSeek-R1-Distill-Qwen-14B.
  - alpha_frac grid: {-0.15, -0.05, +0.05, +0.15} — a fixed fraction of residual norm,
    chosen to be small enough to not collapse coherence but large enough to move the rate.
    (Empirically: at Llama-8B L29 sigma_proj=10.52, mean_residual_norm~=45-60, so
     alpha=1sigma corresponds to alpha_frac ~= 10/50 = 0.2 which is at the coherence edge;
     alpha_frac in {0.05, 0.15} keeps us in a coherent regime.)

For the swap model (Qwen-14B), M1's L*(uncertainty) direction must be re-extracted:
  We use the direction from M1's Llama-8B run as-is (dimensional mismatch would break this,
  but Qwen-14B has hidden_size=5120 and Llama-8B has 4096). Instead we RE-EXTRACT the
  direction on Qwen-14B using the same auxiliary corpus + mean-difference method, at a
  layer chosen by ROC-AUC on a held-out set. To keep this iteration lightweight we use
  a fixed layer heuristic: L* = int(0.75 * n_layers) which mirrors Llama-8B L=29/32.

Outputs: runs/iteration_round_1/M4_C4_scale_invariant/
  llama8b/{results_summary.json, mean_residual_norm.json}
  qwen14b/{results_summary.json, mean_residual_norm.json, direction.pt}
  cross_model_compare.json
  cost.json
"""
from __future__ import annotations
import os, sys, json, time, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, SteeringHook, format_task_prompt, generate_with_optional_steering, capture_residual_last_token
from llm_judge import score_chain

JUDGE_MAX_PARALLEL = 12

TARGET_BEHAVIOUR = "expressing_uncertainty"
ALPHA_FRAC_GRID = {
    "steering_negfrac15": -0.15,
    "steering_negfrac05": -0.05,
    "steering_posfrac05": +0.05,
    "steering_posfrac15": +0.15,
}
CONTROLLERS = list(ALPHA_FRAC_GRID.keys()) + [
    "prompt_suppress", "prompt_amplify",
    "thinking_intervention_suppress", "thinking_intervention_amplify",
]

NL_INSTRUCTIONS = {
    "expressing_uncertainty": {
        "amplify": "As you think through this, explicitly hedge on any step you're not fully sure about (say 'I'm not sure', 'this might be wrong', 'let me double-check', etc.) — signal every uncertainty you have while solving.",
        "suppress": "As you think through this, be fully confident in every step. Do NOT hedge, do not say 'I'm not sure' or 'possibly' or 'let me double-check' — assert every claim decisively.",
    },
}
THINKING_INTERVENTIONS = {
    "expressing_uncertainty": {
        "amplify": "\nHmm, I need to be careful here — I'm not fully sure about some of these steps, let me flag my uncertainty as I go.\n",
        "suppress": "\nI'm confident I can solve this cleanly without any hedging.\n",
    },
}


def compute_mean_residual_norm_from_cache(m1_dir: str, layer: int) -> float | None:
    """Read cached activations if available; return None otherwise."""
    p = os.path.join(m1_dir, "activations.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    key = f"L{layer}"
    if key not in d.files:
        return None
    H = d[key].astype(np.float32)
    norms = np.linalg.norm(H, axis=1)
    return float(norms.mean())


def compute_mean_residual_norm_fresh(model, tok, texts: list[str], layer: int, batch: int = 4) -> float:
    """Fresh forward pass through `texts`, capture last-token residual at `layer`, return mean L2 norm."""
    from model_utils import get_input_device
    device = get_input_device(model)
    all_norms = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad(), capture_residual_last_token(model, [layer]) as cap:
            _ = model(**enc)
            H = cap[layer]  # (batch, hidden), cpu float32
            norms = torch.linalg.norm(H, dim=1).cpu().numpy()
            all_norms.extend(norms.tolist())
    return float(np.mean(all_norms))


def extract_direction_qwen(model, tok, corpus_path: str, target_behaviour: str, layer: int,
                            batch: int = 4) -> tuple[torch.Tensor, float]:
    """Re-extract mean-difference direction for target behaviour on the swap model at a fixed layer.
    Returns (unit_direction, sigma_proj)."""
    from model_utils import get_input_device
    device = get_input_device(model)
    # Load corpus
    rows = [json.loads(l) for l in open(corpus_path)]
    pos_texts, neg_texts = [], []
    for r in rows:
        text = r.get("text") or r.get("chain") or r.get("excerpt", "")
        # This corpus stores labels under "behaviours"
        labels = r.get("behaviours") or r.get("labels") or r.get("behaviour_labels") or {}
        val = labels.get(target_behaviour) if isinstance(labels, dict) else None
        if val is None:
            val = r.get(target_behaviour)
        if val is None:
            continue
        if val in (1, True, "present", "yes"):
            pos_texts.append(text)
        elif val in (0, False, "absent", "no"):
            neg_texts.append(text)
    print(f"    [qwen extract] n_pos={len(pos_texts)}, n_neg={len(neg_texts)}")
    all_texts = pos_texts + neg_texts
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    all_h = []
    for i in range(0, len(all_texts), batch):
        chunk = all_texts[i:i+batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad(), capture_residual_last_token(model, [layer]) as cap:
            _ = model(**enc)
            all_h.append(cap[layer].numpy())
    H = np.concatenate(all_h, axis=0)  # (n, hidden)
    labels = np.array(labels)
    mu_pos = H[labels == 1].mean(axis=0)
    mu_neg = H[labels == 0].mean(axis=0)
    diff = mu_pos - mu_neg
    unit = diff / (np.linalg.norm(diff) + 1e-12)
    # sigma_proj = std of (H @ unit)
    proj = H @ unit
    sigma_proj = float(proj.std())
    return torch.tensor(unit, dtype=torch.float32), sigma_proj


def build_prompts(tok, tasks, controller: str, behaviour: str):
    if controller in ALPHA_FRAC_GRID:
        prompts = [format_task_prompt(tok, r["prompt"]) for r in tasks]
        gold = [r["gold_answer"] for r in tasks]
        return prompts, gold, {"alpha_frac": ALPHA_FRAC_GRID[controller]}
    if controller in ("prompt_suppress", "prompt_amplify"):
        instr = NL_INSTRUCTIONS[behaviour]["suppress" if controller == "prompt_suppress" else "amplify"]
        prompts = [format_task_prompt(tok, f"{instr}\n\n{r['prompt']}") for r in tasks]
        return prompts, [r["gold_answer"] for r in tasks], None
    if controller in ("thinking_intervention_suppress", "thinking_intervention_amplify"):
        phrase = THINKING_INTERVENTIONS[behaviour]["suppress" if controller == "thinking_intervention_suppress" else "amplify"]
        prompts = []
        for r in tasks:
            base = format_task_prompt(tok, r["prompt"])
            prompts.append(base + phrase)
        return prompts, [r["gold_answer"] for r in tasks], None
    raise ValueError(f"unknown controller: {controller}")


def run_controller_scale_invariant(model, tok, tasks, controller, behaviour, direction, layer_idx,
                                    mean_res_norm, seed, max_new_tokens, batch, tag, judge_pool):
    prompts, gold, steer_info = build_prompts(tok, tasks, controller, behaviour)
    steer = None
    if steer_info is not None:
        # Scale-invariant: coef = alpha_frac * mean_residual_norm
        coef = steer_info["alpha_frac"] * mean_res_norm
        steer = SteeringHook(direction, coef)
    all_gens = [None] * len(prompts)
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i+batch]
        gens = generate_with_optional_steering(
            model, tok, chunk, steer, layer_idx if steer else None,
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
                "behaviour_rate": sc["behaviour_rates"].get(behaviour, 0),
                "all_behaviour_rates": sc["behaviour_rates"],
                "coherent": sc["coherent"],
                "accuracy": sc["accuracy"],
            }
        except Exception as e:
            return idx, {"task_idx": idx, "score_error": str(e)}
    for idx, rec in judge_pool.map(_score, range(len(prompts))):
        per_task[idx] = rec
    return per_task


def summarize(per_task):
    n = len(per_task)
    coh_mask = [r for r in per_task if r.get("coherent") == 1]
    br = [r["behaviour_rate"] for r in coh_mask if r.get("behaviour_rate") is not None]
    acc = [r["accuracy"] for r in coh_mask if r.get("accuracy") in (0, 1)]
    return {
        "n": n, "n_coherent": len(coh_mask),
        "coherence_rate": len(coh_mask) / n if n else 0,
        "behaviour_rate": float(np.mean(br)) if br else float("nan"),
        "accuracy": float(np.mean(acc)) if acc else None,
    }


def n_distinct_operating_points(points, eps_r=0.05, eps_a=0.01):
    kept = [p for p in points if p[1] is not None and p[0] == p[0]]
    keep_mask = []
    for i, (r, a) in enumerate(kept):
        distinct = True
        for j, (r2, a2) in enumerate(kept):
            if i == j: continue
            if abs(r - r2) < eps_r and abs(a - a2) < eps_a:
                distinct = False; break
        keep_mask.append(distinct)
    return int(sum(keep_mask))


def run_one_model(model_path, m1_dir, out_dir, bench, args, model_tag: str,
                   qwen_direction_layer: int = None):
    """Run the 8-controller comparison for the target behaviour on one model.
    Returns dict of per-controller summaries + n_distinct per family + mean_residual_norm used.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[M4-C4 / {model_tag}] loading {model_path}")
    model, tok = load_model(model_path)
    hidden_size = model.config.hidden_size

    # Direction + sigma_proj + layer setup
    if model_tag == "llama8b":
        # Reuse M1's extracted direction
        m1 = json.load(open(os.path.join(m1_dir, "results.json")))
        info = m1["per_behaviour"][TARGET_BEHAVIOUR]
        rec = torch.load(info["v_b_path"])
        direction = rec["direction"]
        L_star = int(info["L_star"])
        sigma_proj = float(info.get("sigma_proj_at_L_star", 1.0))
        # Mean residual norm at L_star (from cached activations)
        mean_res_norm = compute_mean_residual_norm_from_cache(m1_dir, L_star)
        if mean_res_norm is None:
            print(f"    [warn] no cache; computing fresh")
            texts = [format_task_prompt(tok, r["prompt"]) for r in bench[:40]]
            mean_res_norm = compute_mean_residual_norm_fresh(model, tok, texts, L_star)
        print(f"    L_star={L_star} sigma_proj={sigma_proj:.3f} mean_residual_norm={mean_res_norm:.3f}")
    else:  # qwen14b
        # Fresh extraction on the swap model at heuristic layer
        n_layers = len(model.model.layers)
        L_star = qwen_direction_layer if qwen_direction_layer is not None else int(0.75 * n_layers)
        print(f"    Qwen-14B n_layers={n_layers}, using L_star={L_star} (heuristic)")
        # Extract direction from auxiliary corpus
        corpus_path = "data/contrast/auxiliary_corpus.jsonl"
        if not os.path.exists(corpus_path):
            # Try alternatives
            for cand in ["data/contrast/auxiliary_corpus_annotated.jsonl", "data/contrast/annotated.jsonl"]:
                if os.path.exists(cand):
                    corpus_path = cand; break
        direction, sigma_proj = extract_direction_qwen(model, tok, corpus_path, TARGET_BEHAVIOUR, L_star)
        # Compute mean residual norm on same corpus
        rows = [json.loads(l) for l in open(corpus_path)]
        sample_texts = [(r.get("text") or r.get("chain") or r.get("excerpt", ""))[:1024] for r in rows[:40]]
        mean_res_norm = compute_mean_residual_norm_fresh(model, tok, sample_texts, L_star)
        print(f"    Qwen-14B direction extracted; sigma_proj={sigma_proj:.3f} mean_residual_norm={mean_res_norm:.3f}")
        # Save direction for future reference
        torch.save({"direction": direction, "L_star": L_star, "sigma_proj": sigma_proj,
                    "mean_residual_norm": mean_res_norm}, os.path.join(out_dir, "direction.pt"))

    # Save norm info
    with open(os.path.join(out_dir, "mean_residual_norm.json"), "w") as f:
        json.dump({"model": model_tag, "L_star": L_star, "sigma_proj": sigma_proj,
                   "mean_residual_norm": mean_res_norm, "hidden_size": hidden_size}, f, indent=2)

    judge_pool = ThreadPoolExecutor(max_workers=JUDGE_MAX_PARALLEL)
    controllers_summary = {}
    for ctrl in CONTROLLERS:
        tag = f"{model_tag}_{ctrl}"
        print(f"    [{tag}]")
        per_task = run_controller_scale_invariant(
            model, tok, bench, ctrl, TARGET_BEHAVIOUR, direction, L_star, mean_res_norm,
            args.seed, args.max_new_tokens, args.batch, tag, judge_pool
        )
        summary = summarize(per_task)
        controllers_summary[ctrl] = {"summary": summary, "per_task_preview": per_task[:3]}
        with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
            json.dump({"model": model_tag, "L_star": L_star, "sigma_proj": sigma_proj,
                       "mean_residual_norm": mean_res_norm, "controllers": controllers_summary}, f, indent=2, default=str)

    # Granularity
    fams = {
        "steering": list(ALPHA_FRAC_GRID.keys()),
        "prompt": ["prompt_suppress", "prompt_amplify"],
        "thinking_intervention": ["thinking_intervention_suppress", "thinking_intervention_amplify"],
    }
    granularity = {}
    for fam, ctrls in fams.items():
        pts = []
        for c in ctrls:
            s = controllers_summary.get(c, {}).get("summary", {})
            pts.append((s.get("behaviour_rate"), s.get("accuracy")))
        pts = [(r, a) for (r, a) in pts if r == r]
        granularity[fam] = {"points": pts, "n_distinct": n_distinct_operating_points(pts)}
    result = {
        "model": model_tag, "L_star": L_star, "sigma_proj": sigma_proj,
        "mean_residual_norm": mean_res_norm, "hidden_size": hidden_size,
        "controllers": controllers_summary, "granularity": granularity,
    }
    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    # release model
    del model
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama_path", default="/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--qwen_path", default="/data/zhenqian/models/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--m1_dir", default="runs/M1_locate")
    ap.add_argument("--bench", default="data/benchmark/benchmark_500.jsonl")
    ap.add_argument("--out_dir", default="runs/iteration_round_1/M4_C4_scale_invariant")
    ap.add_argument("--n_bench", type=int, default=60)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only_model", default=None, choices=[None, "llama8b", "qwen14b"])
    ap.add_argument("--qwen_L", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    tasks = [json.loads(l) for l in open(args.bench)]
    tasks = tasks[:args.n_bench]
    print(f"[M4-C4] {len(tasks)} tasks, controllers={CONTROLLERS}")

    per_model = {}
    if args.only_model in (None, "llama8b"):
        per_model["llama8b"] = run_one_model(
            args.llama_path, args.m1_dir, os.path.join(args.out_dir, "llama8b"),
            tasks, args, "llama8b",
        )
    if args.only_model in (None, "qwen14b"):
        per_model["qwen14b"] = run_one_model(
            args.qwen_path, args.m1_dir, os.path.join(args.out_dir, "qwen14b"),
            tasks, args, "qwen14b", qwen_direction_layer=args.qwen_L,
        )

    # Cross-model compare
    compare = {"models": list(per_model.keys()), "per_model": {}, "verdict": {}}
    for tag, r in per_model.items():
        compare["per_model"][tag] = {
            "L_star": r["L_star"], "sigma_proj": r["sigma_proj"],
            "mean_residual_norm": r["mean_residual_norm"],
            "n_distinct_steering": r["granularity"]["steering"]["n_distinct"],
            "n_distinct_prompt": r["granularity"]["prompt"]["n_distinct"],
            "n_distinct_ti": r["granularity"]["thinking_intervention"]["n_distinct"],
        }
    if "llama8b" in per_model and "qwen14b" in per_model:
        l = compare["per_model"]["llama8b"]; q = compare["per_model"]["qwen14b"]
        compare["verdict"]["llama_predicate"] = bool(l["n_distinct_steering"] > l["n_distinct_prompt"] and l["n_distinct_steering"] > l["n_distinct_ti"])
        compare["verdict"]["qwen_predicate"] = bool(q["n_distinct_steering"] > q["n_distinct_prompt"] and q["n_distinct_steering"] > q["n_distinct_ti"])
        compare["verdict"]["cross_model_holds"] = bool(compare["verdict"]["llama_predicate"] and compare["verdict"]["qwen_predicate"])
        compare["verdict"]["scale_invariance_check"] = {
            "sigma_proj_ratio_qwen_over_llama": q["sigma_proj"] / l["sigma_proj"],
            "residual_norm_ratio_qwen_over_llama": q["mean_residual_norm"] / l["mean_residual_norm"],
            "note": "residual_norm ratio should be MUCH closer to 1 than sigma_proj ratio if scale-invariant coef works",
        }
    with open(os.path.join(args.out_dir, "cross_model_compare.json"), "w") as f:
        json.dump(compare, f, indent=2, default=str)

    cost = {
        "run_id": "M4_C4_scale_invariant_iter1",
        "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
        "wall_seconds_total": time.time() - t0,
        "n_bench": len(tasks),
        "target_behaviour": TARGET_BEHAVIOUR,
        "controllers": CONTROLLERS,
        "models": list(per_model.keys()),
    }
    with open(os.path.join(args.out_dir, "cost.json"), "w") as f:
        json.dump(cost, f, indent=2)
    print(f"[M4-C4] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
