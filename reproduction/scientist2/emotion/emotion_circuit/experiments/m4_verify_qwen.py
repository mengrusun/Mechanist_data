"""
M4 — Verify swap on Qwen2.5-7B-Instruct.

Per plan: re-run M3's three-arm comparison on Qwen with hyperparameters re-selected
on Qwen's own val fold under matched N=9 budget. Do NOT port Llama-tuned settings.

Since Arm A depends on C_e (a Qwen-specific circuit), this script first runs a lightweight
M1-equivalent to build C_e_qwen. The reduced scope is: Stage A shortlist + Stage B ranking
on a smaller val set (30 stems per emotion), (k_h*, k_n*) selection on grid,
skip full stability analysis (that's M1's role for the main model). Then runs M3-style
three-arm eval and judge.

Outputs (runs/A4_verify_qwen/):
  qwen_C_e.json, qwen_kstar.json, qwen_directions.npz, qwen_shortlist.json
  qwen_selected_configs.json
  qwen_eval_generations.json, qwen_judge_results.json, qwen_metrics.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    DEFAULT_SEED,
    EMOTIONS,
    QWEN_PATH,
    REPO_DIR,
    build_scenario_split,
    load_model,
    load_sev,
    log,
    save_json,
    set_seed,
    target_prefix,
)
from judge_client import JudgeClient
from m1_location import (
    ALPHA_2,
    K_H_GRID,
    K_N_GRID,
    TOP_HEAD_FRAC,
    TOP_N_LAYERS,
    TOP_NEURON_FRAC,
    _install_multi_component_hooks,
    _remove_hooks,
    collect_head_and_neuron_activations,
    collect_layerwise_residuals,
    hidden_dim_of,
    head_dim_of,
    n_heads_of,
    n_layers_of,
    n_neurons_of,
    per_head_probe_auc,
    per_neuron_scores,
    target_prefix_logprob_batch,
)
from m3_applied import (
    ALPHAS_ARM_A,
    ALPHAS_ARM_C,
    ARM_B_TEMPLATES,
    ARM_B_POSITIONS,
    _install_arm_c_hook,
    _install_arm_c_hook_static,
    batched_generate,
    format_arm_b_prompt,
    paired_bootstrap_ci,
)


def build_qwen_c_e(model, tok, train_events, val_events, emotions, seed, top_head_frac=TOP_HEAD_FRAC, top_neuron_frac=TOP_NEURON_FRAC, n_val_stems=30, sanity=False):
    """
    Lightweight M1 for Qwen: Stage A + Stage B + (k_h*, k_n*) selection.
    Returns dict with C_e per emotion + directions + shortlist + kstar.
    """
    L = n_layers_of(model)
    NH = n_heads_of(model)
    HD = head_dim_of(model)
    NN = n_neurons_of(model)
    H = hidden_dim_of(model)

    log(f"M4-M1: Qwen n_layers={L}, n_heads={NH}, head_dim={HD}, hidden={H}, intermediate={NN}")

    # Stage A: emotion-contextualized residuals
    log("M4-M1 Stage A: collecting emo-context activations")
    emo_res = {}
    emo_head_act = {}
    emo_neuron_pre = {}
    for e in emotions:
        prefix = target_prefix(e).strip()
        ctx = [ev + " " + prefix for ev in train_events]
        emo_res[e] = collect_layerwise_residuals(model, tok, ctx, batch_size=6)
        emo_head_act[e], emo_neuron_pre[e] = collect_head_and_neuron_activations(model, tok, ctx, batch_size=3)
        log(f"  emo={e} activations collected")

    directions = {}
    head_aucs = {}
    neuron_scores = {}
    for e in emotions:
        pos = emo_res[e]
        neg = np.concatenate([emo_res[o] for o in emotions if o != e], axis=0)
        directions[e] = pos.mean(axis=0) - neg.mean(axis=0)

        pos_h = emo_head_act[e]
        neg_h = np.concatenate([emo_head_act[o] for o in emotions if o != e], axis=0)
        head_aucs[e] = per_head_probe_auc(pos_h, neg_h)

        pos_n = emo_neuron_pre[e]
        neg_n = np.concatenate([emo_neuron_pre[o] for o in emotions if o != e], axis=0)
        neuron_scores[e] = per_neuron_scores(pos_n, neg_n, directions[e], model)

    # Shortlist
    shortlist = {}
    for e in emotions:
        per_layer_sum = head_aucs[e].sum(axis=1) + neuron_scores[e].sum(axis=1)
        top_layers = sorted(np.argsort(-per_layer_sum)[:TOP_N_LAYERS].tolist())
        n_top_h = max(1, int(round(top_head_frac * NH * TOP_N_LAYERS)))
        flat_h = [(l, h, float(head_aucs[e][l, h])) for l in top_layers for h in range(NH)]
        flat_h.sort(key=lambda x: -x[2])
        head_pool = [(l, h) for (l, h, _) in flat_h[:n_top_h]]
        n_top_n = max(1, int(round(top_neuron_frac * NN * TOP_N_LAYERS)))
        flat_n = [(l, n, float(neuron_scores[e][l, n])) for l in top_layers for n in range(NN)]
        flat_n.sort(key=lambda x: -x[2])
        neuron_pool = [(l, n) for (l, n, _) in flat_n[:n_top_n]]
        shortlist[e] = {"top_layers": top_layers, "head_pool": head_pool, "neuron_pool": neuron_pool}
        log(f"  emo={e} shortlist: top_layers={top_layers}, |head_pool|={len(head_pool)}, |neuron_pool|={len(neuron_pool)}")

    # Stage B causal ranker on Qwen val stems
    val_events_short = val_events[:n_val_stems]
    log(f"M4-M1 Stage B: on {len(val_events_short)} val stems per emotion")
    stageB_scores = {}
    for e in emotions:
        prefixes = [target_prefix(e)] * len(val_events_short)
        baseline_lp = target_prefix_logprob_batch(model, tok, val_events_short, prefixes, batch_size=6)
        d_e = directions[e]
        head_scores = []
        neuron_scores_stageB = []
        pos_pre = emo_neuron_pre[e]
        for i, (l, h) in enumerate(shortlist[e]["head_pool"]):
            handles = _install_multi_component_hooks(model, heads=[(l, h, d_e[l])], neurons=[], alpha=ALPHA_2)
            try:
                lp = target_prefix_logprob_batch(model, tok, val_events_short, prefixes, batch_size=6)
            finally:
                _remove_hooks(handles)
            head_scores.append((int(l), int(h), float((lp - baseline_lp).mean())))
        with torch.no_grad():
            W_downs = {l: model.model.layers[l].mlp.down_proj.weight.detach().float().cpu().numpy() for l in shortlist[e]["top_layers"]}
        for i, (l, n) in enumerate(shortlist[e]["neuron_pool"]):
            w = W_downs[l][:, n]
            sign_val = float(np.sign(w.dot(d_e[l])))
            if sign_val == 0.0:
                sign_val = 1.0
            std_val = float(pos_pre[:, l, n].std() + 1e-6)
            handles = _install_multi_component_hooks(model, heads=[], neurons=[(l, n, sign_val, std_val)], alpha=ALPHA_2)
            try:
                lp = target_prefix_logprob_batch(model, tok, val_events_short, prefixes, batch_size=6)
            finally:
                _remove_hooks(handles)
            neuron_scores_stageB.append((int(l), int(n), float((lp - baseline_lp).mean())))
        stageB_scores[e] = {"heads": head_scores, "neurons": neuron_scores_stageB}
        log(f"  emo={e} Stage B: {len(head_scores)} heads + {len(neuron_scores_stageB)} neurons scored")

    # k* selection on full val fold
    kstar_grid = []
    baseline_per_emotion = {}
    for e in emotions:
        prefixes = [target_prefix(e)] * len(val_events)
        baseline_per_emotion[e] = target_prefix_logprob_batch(model, tok, val_events, prefixes, batch_size=6)

    neuron_meta_per_emotion = {e: {} for e in emotions}
    for e in emotions:
        pos_pre = emo_neuron_pre[e]
        d_e = directions[e]
        for (l, n, _) in stageB_scores[e]["neurons"]:
            if (l, n) not in neuron_meta_per_emotion[e]:
                with torch.no_grad():
                    w = model.model.layers[l].mlp.down_proj.weight[:, n].float().cpu().numpy()
                sign_val = float(np.sign(w.dot(d_e[l])))
                if sign_val == 0.0:
                    sign_val = 1.0
                std_val = float(pos_pre[:, l, n].std() + 1e-6)
                neuron_meta_per_emotion[e][(l, n)] = (sign_val, std_val)

    for k_h in K_H_GRID:
        for k_n in K_N_GRID:
            per_emo = []
            for e in emotions:
                head_pool = sorted(stageB_scores[e]["heads"], key=lambda x: -x[2])[:k_h]
                neuron_pool = sorted(stageB_scores[e]["neurons"], key=lambda x: -x[2])[:k_n]
                d_e = directions[e]
                heads_p = [(l, h, d_e[l]) for (l, h, _) in head_pool]
                neurons_p = [(l, n, *neuron_meta_per_emotion[e][(l, n)]) for (l, n, _) in neuron_pool]
                handles = _install_multi_component_hooks(model, heads=heads_p, neurons=neurons_p, alpha=ALPHA_2)
                try:
                    prefixes = [target_prefix(e)] * len(val_events)
                    lp = target_prefix_logprob_batch(model, tok, val_events, prefixes, batch_size=6)
                finally:
                    _remove_hooks(handles)
                per_emo.append(float((lp - baseline_per_emotion[e]).mean()))
            kstar_grid.append({"k_h": k_h, "k_n": k_n, "macro_gain": float(np.mean(per_emo))})
    best = max(kstar_grid, key=lambda x: x["macro_gain"])
    log(f"M4-M1: Qwen (k_h*, k_n*) = ({best['k_h']}, {best['k_n']}), macro gain {best['macro_gain']:+.4f}")

    C_e = {}
    for e in emotions:
        head_pool = sorted(stageB_scores[e]["heads"], key=lambda x: -x[2])[:best["k_h"]]
        neuron_pool = sorted(stageB_scores[e]["neurons"], key=lambda x: -x[2])[:best["k_n"]]
        C_e[e] = {"heads": [(l, h) for (l, h, _) in head_pool],
                  "neurons": [(l, n) for (l, n, _) in neuron_pool]}

    return {"C_e": C_e, "directions": directions, "shortlist": shortlist, "kstar": {"k_h": best["k_h"], "k_n": best["k_n"]},
            "kstar_grid": kstar_grid, "stageB_scores": stageB_scores, "neuron_meta_per_emotion": neuron_meta_per_emotion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(REPO_DIR / "runs" / "A4_verify_qwen"))
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--model_path", default=QWEN_PATH)
    ap.add_argument("--n_val_stems_stage_b", type=int, default=30)
    ap.add_argument("--skip_judge", action="store_true")
    ap.add_argument("--judge_workers", type=int, default=6)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"M4: loading Qwen {args.model_path}")
    t0 = time.time()
    model, tok = load_model(args.model_path)
    log(f"M4: Qwen loaded ({time.time()-t0:.1f}s)")

    sev = load_sev()
    # Held-out per plan: same scenario-level 5-scenario eval fold used on Qwen (no separate held-out dataset).
    split = build_scenario_split(sev, seed=args.seed)
    train_items = split["train"]
    val_items = split["val"]
    eval_items = split["eval"]

    if args.sanity:
        emotions = EMOTIONS[:2]
        train_items = train_items[:30]
        val_items = val_items[:10]
        eval_items = eval_items[:8]
        n_val_stems = min(args.n_val_stems_stage_b, 5)
    else:
        emotions = EMOTIONS
        n_val_stems = args.n_val_stems_stage_b

    train_events = [r["event"] for r in train_items]
    val_events = [r["event"] for r in val_items]
    eval_events = [r["event"] for r in eval_items]

    # Build Qwen-specific C_e
    qwen_out = build_qwen_c_e(model, tok, train_events, val_events, emotions, args.seed, n_val_stems=n_val_stems, sanity=args.sanity)

    save_json(out_dir / "qwen_C_e.json", qwen_out["C_e"])
    save_json(out_dir / "qwen_kstar.json", qwen_out["kstar"])
    # save directions
    np.savez_compressed(out_dir / "qwen_directions.npz", **{f"d_{e}": qwen_out["directions"][e] for e in emotions})
    save_json(out_dir / "qwen_shortlist.json", {e: {
        "top_layers": v["top_layers"],
        "head_pool_size": len(v["head_pool"]),
        "neuron_pool_size": len(v["neuron_pool"]),
    } for e, v in qwen_out["shortlist"].items()})

    # Now run M3-style val + eval + judge on Qwen
    C_e = qwen_out["C_e"]
    directions = qwen_out["directions"]
    shortlist = qwen_out["shortlist"]
    kstar = qwen_out["kstar"]
    neuron_meta = qwen_out["neuron_meta_per_emotion"]

    log("M4-M3 Arm A/B/C val sweep on Qwen")
    baseline_val_lp = {}
    for e in emotions:
        prefixes = [target_prefix(e)] * len(val_events)
        baseline_val_lp[e] = target_prefix_logprob_batch(model, tok, val_events, prefixes, batch_size=6)

    # Precompute eval neuron activations for sign/std
    eval_neuron_by_emo = {}
    for e in emotions:
        prefix = target_prefix(e).strip()
        ctx = [ev + " " + prefix for ev in eval_events]
        _, eval_neuron_by_emo[e] = collect_head_and_neuron_activations(model, tok, ctx, batch_size=3)

    # Arm A val
    kh, kn = kstar["k_h"], kstar["k_n"]
    K_H_neigh = sorted({max(K_H_GRID[0], kh // 2), kh, min(K_H_GRID[-1], kh * 2)})[:3]
    K_N_neigh = sorted({max(K_N_GRID[0], kn // 2), kn, min(K_N_GRID[-1], kn * 2)})[:3]
    arm_a_val = {}
    for e in emotions:
        rows = []
        d_e = directions[e]
        for k_h_i in K_H_neigh:
            for k_n_i in K_N_neigh:
                for a in ALPHAS_ARM_A:
                    heads = C_e[e]["heads"][:k_h_i]
                    neurons = C_e[e]["neurons"][:k_n_i]
                    heads_p = [(int(l), int(h), d_e[int(l)]) for (l, h) in heads]
                    neurons_p = []
                    for (l, n) in neurons:
                        l, n = int(l), int(n)
                        if (l, n) not in neuron_meta[e]:
                            with torch.no_grad():
                                w = model.model.layers[l].mlp.down_proj.weight[:, n].float().cpu().numpy()
                            sv = float(np.sign(w.dot(d_e[l])))
                            if sv == 0.0:
                                sv = 1.0
                            std_v = float(eval_neuron_by_emo[e][:, l, n].std() + 1e-6)
                            neuron_meta[e][(l, n)] = (sv, std_v)
                        sv, std_v = neuron_meta[e][(l, n)]
                        neurons_p.append((l, n, sv, std_v))
                    handles = _install_multi_component_hooks(model, heads=heads_p, neurons=neurons_p, alpha=a)
                    try:
                        prefixes = [target_prefix(e)] * len(val_events)
                        lp = target_prefix_logprob_batch(model, tok, val_events, prefixes, batch_size=6)
                    finally:
                        _remove_hooks(handles)
                    rows.append({"k_h": k_h_i, "k_n": k_n_i, "alpha": a, "val_score": float((lp - baseline_val_lp[e]).mean())})
        arm_a_val[e] = rows

    # Arm B val
    arm_b_val = {}
    for e in emotions:
        rows = []
        for ti, tmpl in enumerate(ARM_B_TEMPLATES):
            for pos in ARM_B_POSITIONS:
                prompts = [format_arm_b_prompt(ev, e, tmpl, pos) for ev in val_events]
                prefixes = [target_prefix(e)] * len(prompts)
                lp = target_prefix_logprob_batch(model, tok, prompts, prefixes, batch_size=6)
                rows.append({"template_idx": ti, "position": pos, "val_score": float((lp - baseline_val_lp[e]).mean())})
        arm_b_val[e] = rows

    # Arm C val (uses static hook — publishes per-batch event lengths)
    arm_c_val = {}
    for e in emotions:
        d_e = directions[e]
        top_layers = shortlist[e]["top_layers"]
        rows = []
        for l in top_layers:
            for a in ALPHAS_ARM_C:
                handles = _install_arm_c_hook_static(model, l, d_e[l], a)
                try:
                    prefixes = [target_prefix(e)] * len(val_events)
                    lp = target_prefix_logprob_batch(model, tok, val_events, prefixes, batch_size=6)
                finally:
                    _remove_hooks(handles)
                rows.append({"layer": int(l), "alpha": a, "val_score": float((lp - baseline_val_lp[e]).mean())})
        arm_c_val[e] = rows

    save_json(out_dir / "qwen_arm_A_val.json", arm_a_val)
    save_json(out_dir / "qwen_arm_B_val.json", arm_b_val)
    save_json(out_dir / "qwen_arm_C_val.json", arm_c_val)

    selected = {"A": {e: max(arm_a_val[e], key=lambda r: r["val_score"]) for e in emotions},
                "B": {e: max(arm_b_val[e], key=lambda r: r["val_score"]) for e in emotions},
                "C": {e: max(arm_c_val[e], key=lambda r: r["val_score"]) for e in emotions}}
    save_json(out_dir / "qwen_selected_configs.json", selected)

    # Eval generations
    log("M4: eval generations")
    eval_generations = {"A": {}, "B": {}, "C": {}}
    for e in emotions:
        d_e = directions[e]
        # Arm A
        conf_a = selected["A"][e]
        heads = C_e[e]["heads"][:conf_a["k_h"]]
        neurons = C_e[e]["neurons"][:conf_a["k_n"]]
        heads_p = [(int(l), int(h), d_e[int(l)]) for (l, h) in heads]
        neurons_p = [(l, n, *neuron_meta[e][(l, n)]) for (l, n) in neurons]
        handles = _install_multi_component_hooks(model, heads=heads_p, neurons=neurons_p, alpha=conf_a["alpha"])
        try:
            gens = batched_generate(model, tok, eval_events, batch_size=3)
        finally:
            _remove_hooks(handles)
        eval_generations["A"][e] = [{"event_id": eval_items[i]["id"], "event": eval_events[i], "continuation": gens[i]} for i in range(len(eval_events))]

        conf_b = selected["B"][e]
        prompts_b = [format_arm_b_prompt(ev, e, ARM_B_TEMPLATES[conf_b["template_idx"]], conf_b["position"]) for ev in eval_events]
        gens_b = batched_generate(model, tok, prompts_b, batch_size=3)
        eval_generations["B"][e] = [{"event_id": eval_items[i]["id"], "event": eval_events[i], "continuation": gens_b[i]} for i in range(len(eval_events))]

        conf_c = selected["C"][e]
        handles = _install_arm_c_hook(model, conf_c["layer"], d_e[conf_c["layer"]], conf_c["alpha"], None)
        try:
            gens_c = batched_generate(model, tok, eval_events, batch_size=3)
        finally:
            _remove_hooks(handles)
        eval_generations["C"][e] = [{"event_id": eval_items[i]["id"], "event": eval_events[i], "continuation": gens_c[i]} for i in range(len(eval_events))]
        log(f"  emo={e} eval done (3 arms)")

    save_json(out_dir / "qwen_eval_generations.json", eval_generations)

    if args.skip_judge:
        return

    log("M4: gpt-5.4 judge on Qwen continuations")
    judge = JudgeClient()
    tasks = []
    for arm in ["A", "B", "C"]:
        for e in emotions:
            for row in eval_generations[arm][e]:
                tasks.append({"arm": arm, "target": e, "event_id": row["event_id"], "event": row["event"], "continuation": row["continuation"]})

    def _worker(t):
        r = judge.judge(t["event"], t["continuation"], t["target"])
        return {**t, **r}

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
        for i, res in enumerate(ex.map(_worker, tasks)):
            results.append(res)
            if (i + 1) % 100 == 0:
                log(f"  judged {i + 1}/{len(tasks)}")
    save_json(out_dir / "qwen_judge_results.json", results)

    # Accuracy + pairwise
    accs = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "n": 0, "other": 0, "per_row_correct": []}))
    for r in results:
        rec = accs[r["arm"]][r["target"]]
        rec["n"] += 1
        v = r["verdict"]
        if v == "CORRECT":
            rec["correct"] += 1
            rec["per_row_correct"].append(1)
        else:
            if v == "OTHER":
                rec["other"] += 1
            rec["per_row_correct"].append(0)

    per_emo_acc = {arm: {} for arm in ["A", "B", "C"]}
    for arm in ["A", "B", "C"]:
        for e in emotions:
            rec = accs[arm][e]
            per_emo_acc[arm][e] = {"n": rec["n"], "correct": rec["correct"], "other": rec["other"], "accuracy": rec["correct"] / max(rec["n"], 1), "per_row_correct": rec["per_row_correct"]}

    macro = {arm: float(np.mean([per_emo_acc[arm][e]["accuracy"] for e in emotions])) for arm in ["A", "B", "C"]}

    def paired_diff_ci(a_bin, b_bin, n_boot=1000, seed=0):
        rng = np.random.default_rng(seed)
        n = min(len(a_bin), len(b_bin))
        a_arr = np.array(a_bin[:n])
        b_arr = np.array(b_bin[:n])
        d = a_arr - b_arr
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots.append(d[idx].mean())
        return float(np.mean(d)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    pairwise = {}
    for e in emotions:
        a_bin = per_emo_acc["A"][e]["per_row_correct"]
        b_bin = per_emo_acc["B"][e]["per_row_correct"]
        c_bin = per_emo_acc["C"][e]["per_row_correct"]
        mA_B, loA_B, hiA_B = paired_diff_ci(a_bin, b_bin, seed=args.seed + hash((e, "AB")) % 100000)
        mA_C, loA_C, hiA_C = paired_diff_ci(a_bin, c_bin, seed=args.seed + hash((e, "AC")) % 100000)
        pairwise[e] = {
            "A_minus_B_mean": mA_B, "A_minus_B_ci": [loA_B, hiA_B], "A_gt_B_at_95": bool(loA_B > 0),
            "A_minus_C_mean": mA_C, "A_minus_C_ci": [loA_C, hiA_C], "A_gt_C_at_95": bool(loA_C > 0),
        }

    n_ab = sum(1 for e in emotions if pairwise[e]["A_gt_B_at_95"])
    n_ac = sum(1 for e in emotions if pairwise[e]["A_gt_C_at_95"])
    metrics = {
        "per_emotion_accuracy": per_emo_acc,
        "macro_accuracy": macro,
        "pairwise": pairwise,
        "n_A_gt_B_at_95": n_ab,
        "n_A_gt_C_at_95": n_ac,
        "verify_threshold_over_6": 4,
        "A_gt_B_pass": n_ab >= 4,
        "A_gt_C_pass": n_ac >= 4,
        "verify_verdict": ("robust" if (n_ab >= 4 and n_ac >= 4) else "conditional"),
    }
    save_json(out_dir / "qwen_metrics.json", metrics)
    log(f"M4: DONE. verify verdict = {metrics['verify_verdict']}. A>B: {n_ab}/{len(emotions)}, A>C: {n_ac}/{len(emotions)}")


if __name__ == "__main__":
    main()
