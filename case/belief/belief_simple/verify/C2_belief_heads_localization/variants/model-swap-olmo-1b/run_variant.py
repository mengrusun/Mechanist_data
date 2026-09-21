"""
Variant runner: model-swap-olmo-1b for C2 (Belief Heads Localization)

Changes from main experiment:
  - Model: OLMo-1B-hf (OlmoForCausalLM) instead of pythia-{410m,1b,2.8b}
  - Hook: o_proj input pre-hook instead of dense input pre-hook
  - Fisher aggregation: separate q/k/v/o projections instead of fused QKV

Everything else (datasets, thresholds, seeds, metric, Fisher formula) unchanged.

Optimisations:
  1. Clean baselines pre-computed ONCE per target (not each greedy step).
  2. After mask construction, ALL Fisher tensors are deleted and CUDA cache
     is cleared before the greedy search begins — critical for fast PPL eval
     (Fisher tensors take ~16 GB; deleting them before eval restores fast
     bandwidth, reducing PPL from ~11 min/call to ~30 s/call).
  3. Intermediate artifacts are reused if already on disk (cheap resume).

Code-review fixes vs. draft:
  - save_json(path, obj) order correct (belief_utils: path first)
  - evaluate_ppl(net, token_tensor, device=...) — no tokenizer arg
  - scipy_stats.spearmanr(...)[0] — cross-version compat
"""

import os
import sys
import json
import math
import random
import time
import gc
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

# -- path setup ---------------------------------------------------------------
VARIANT_DIR = Path(__file__).resolve().parent
EXP_ROOT = VARIANT_DIR.parents[3]          # .../exp18/
sys.path.insert(0, str(EXP_ROOT / "scripts"))

from belief_utils import (
    load_task, BeliefExample, evaluate_ppl, save_json, load_json, wilson_ci,
    continuation_logprob_batch,
)

# -- constants ----------------------------------------------------------------
MODEL_PATH = "/mnt/quarkfs/share_model/OLMo/OLMo-1B-hf"
OUT_DIR = VARIANT_DIR
DATA_ROOT = "/data/xuhaoming/belief_loc/data/derived/belief_core"
PPL_SAMPLE_PATH = str(EXP_ROOT / "refine-logs/artifacts/ppl_sample.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -- model loading ------------------------------------------------------------
def load_olmo(dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[variant] loading OLMo-1B-hf ...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    net = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype, device_map="auto")
    net.eval()
    return net, tok


def olmo_arch_info(net) -> dict:
    cfg = net.config
    return {"n_layers": cfg.num_hidden_layers, "n_heads": cfg.num_attention_heads,
            "head_dim": cfg.hidden_size // cfg.num_attention_heads, "hidden": cfg.hidden_size}


# -- hooks --------------------------------------------------------------------
def install_olmo_scaling_hooks(net, scale_map: dict, info: dict) -> List:
    d = info["head_dim"]
    by_layer: dict = {}
    for (l, h), s in scale_map.items():
        by_layer.setdefault(l, {})[h] = float(s)
    handles = []
    for li, hs in by_layer.items():
        o_proj = net.model.layers[li].self_attn.o_proj
        def _make(hsl, hd=d):
            def _hook(module, args, kwargs):
                x = args[0].clone()
                for h, s in hsl.items():
                    if s != 1.0:
                        x[..., h * hd: (h + 1) * hd] *= s
                return (x,) + args[1:], kwargs
            return _hook
        handles.append(o_proj.register_forward_pre_hook(_make(hs), with_kwargs=True))
    return handles


def remove_hooks(handles: List) -> None:
    for h in handles:
        h.remove()


# -- task eval ----------------------------------------------------------------
def eval_task(net, tok, examples: List[BeliefExample],
              scale_map=None, info: dict = None) -> dict:
    handles = install_olmo_scaling_hooks(net, scale_map, info) if scale_map else []
    correct = sum(
        1 for ex in examples
        if continuation_logprob_batch(net, tok, ex.prompt,
                                      [ex.gold, ex.distractor], device=DEVICE)[0]
        > continuation_logprob_batch(net, tok, ex.prompt,
                                     [ex.gold, ex.distractor], device=DEVICE)[1]
    )
    if handles:
        remove_hooks(handles)
    acc = correct / len(examples) if examples else 0.0
    lo, hi = wilson_ci(correct, len(examples))
    return {"acc": acc, "correct_count": correct, "total": len(examples),
            "wilson_ci_low": lo, "wilson_ci_high": hi}


def eval_task_fast(net, tok, examples: List[BeliefExample],
                   scale_map=None, info: dict = None) -> dict:
    """Fast version that doesn't double-call continuation_logprob_batch."""
    handles = install_olmo_scaling_hooks(net, scale_map, info) if scale_map else []
    correct = 0
    for ex in examples:
        lps = continuation_logprob_batch(net, tok, ex.prompt,
                                         [ex.gold, ex.distractor], device=DEVICE)
        if lps[0] > lps[1]:
            correct += 1
    if handles:
        remove_hooks(handles)
    acc = correct / len(examples) if examples else 0.0
    lo, hi = wilson_ci(correct, len(examples))
    return {"acc": acc, "correct_count": correct, "total": len(examples),
            "wilson_ci_low": lo, "wilson_ci_high": hi}


# -- M1 gate ------------------------------------------------------------------
def run_m1_gate(net, tok, info: dict) -> dict:
    print("[M1] above-chance gate ...")
    gate = {}
    for task in ["personal_belief", "attributed_belief", "world_knowledge"]:
        ex = load_task(DATA_ROOT, task)
        r = eval_task_fast(net, tok, ex, info=info)
        gate[task] = {**r, "above_chance": r["acc"] > 0.5 and r["wilson_ci_low"] > 0.5}
        print(f"  {task}: acc={r['acc']:.3f} CI=[{r['wilson_ci_low']:.3f},"
              f"{r['wilson_ci_high']:.3f}] above={gate[task]['above_chance']}")
    return gate


# -- Fisher -------------------------------------------------------------------
def compute_fisher(net, tok, examples: List[BeliefExample]) -> dict:
    """Empirical Fisher (fp32). net must be in fp32 train mode, grad ON."""
    print(f"  [Fisher] n={len(examples)} ...")
    accum = {pn: torch.zeros_like(p.data, dtype=torch.float32)
             for pn, p in net.named_parameters() if p.requires_grad}
    for i, ex in enumerate(examples):
        net.zero_grad()
        prompt_ids = tok.encode(ex.prompt, add_special_tokens=False)
        gold_ids = tok.encode(ex.gold, add_special_tokens=False)
        ids = torch.tensor([prompt_ids + gold_ids], dtype=torch.long, device=DEVICE)
        logits = net(input_ids=ids).logits[0]
        p0 = len(prompt_ids) - 1
        lp = F.log_softmax(logits[p0: p0 + len(gold_ids)].float(), dim=-1).gather(
            1, torch.tensor(gold_ids, device=DEVICE)[:, None]).squeeze(1).sum()
        lp.backward()
        with torch.no_grad():
            for pn, p in net.named_parameters():
                if p.grad is not None and pn in accum:
                    accum[pn] += p.grad.float() ** 2
        net.zero_grad()
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(examples)}")
    N = len(examples)
    for pn in accum:
        accum[pn].div_(N)
    return accum


def agg_head_scores(fisher: dict, info: dict) -> dict:
    n_layers, n_heads, d = info["n_layers"], info["n_heads"], info["head_dim"]
    scores = {}
    for l in range(n_layers):
        pfx = f"model.layers.{l}.self_attn"
        for h in range(n_heads):
            s = 0.0
            for proj in ["q_proj", "k_proj", "v_proj"]:
                w = f"{pfx}.{proj}.weight"
                if w in fisher:
                    s += float(fisher[w][h*d:(h+1)*d, :].sum())
                b = f"{pfx}.{proj}.bias"
                if b in fisher:
                    s += float(fisher[b][h*d:(h+1)*d].sum())
            w = f"{pfx}.o_proj.weight"
            if w in fisher:
                s += float(fisher[w][:, h*d:(h+1)*d].sum())
            scores[(l, h)] = s
    return scores


# -- mask ---------------------------------------------------------------------
def build_mask(F_tgt: dict, F_know: dict, top_t: float, top_k: float,
               info: dict) -> Tuple[dict, List]:
    def thresh(fd, frac):
        v = torch.cat([x.flatten().float() for x in fd.values()])
        k = max(1, int(frac * v.numel()))
        return float(torch.kthvalue(v, v.numel() - k + 1).values)
    tt, tk = thresh(F_tgt, top_t), thresh(F_know, top_k)
    n_layers, n_heads, d = info["n_layers"], info["n_heads"], info["head_dim"]
    scores = {}
    for l in range(n_layers):
        pfx = f"model.layers.{l}.self_attn"
        for h in range(n_heads):
            in_t, in_k, total = 0, 0, 0
            for proj in ["q_proj", "k_proj", "v_proj"]:
                w = f"{pfx}.{proj}.weight"
                if w in F_tgt:
                    bt = F_tgt[w][h*d:(h+1)*d, :]
                    bk = F_know.get(w, torch.zeros_like(bt))[h*d:(h+1)*d, :]
                    in_t += int((bt >= tt).sum()); in_k += int((bk >= tk).sum())
                    total += bt.numel()
            w = f"{pfx}.o_proj.weight"
            if w in F_tgt:
                bt = F_tgt[w][:, h*d:(h+1)*d]
                bk = F_know.get(w, torch.zeros_like(bt))[:, h*d:(h+1)*d]
                in_t += int((bt >= tt).sum()); in_k += int((bk >= tk).sum())
                total += bt.numel()
            scores[(l, h)] = (in_t - min(in_t, in_k)) / total if total else 0.0
    ranked = sorted(scores, key=lambda x: -scores[x])
    return scores, ranked


# -- 4-criteria eval (ablated only, clean cached) ----------------------------
def eval_hstar(net, tok, heads: list, ppl_tokens, info: dict,
               clean_accs: dict, clean_ppl: float,
               t_ex, o_ex, wk_ex) -> dict:
    """Only ablated evals; clean values come from cache."""
    sm = {h: 0.0 for h in heads}
    abl_t = eval_task_fast(net, tok, t_ex, scale_map=sm, info=info)
    abl_o = eval_task_fast(net, tok, o_ex, scale_map=sm, info=info)
    abl_w = eval_task_fast(net, tok, wk_ex, scale_map=sm, info=info)
    drop_t = clean_accs["target"] - abl_t["acc"]
    drop_o = clean_accs["other"] - abl_o["acc"]
    drop_w = clean_accs["wk"] - abl_w["acc"]
    handles = install_olmo_scaling_hooks(net, sm, info)
    ppl_abl = evaluate_ppl(net, ppl_tokens, device=DEVICE)
    remove_hooks(handles)
    ppl_ratio = ppl_abl["ppl"] / clean_ppl
    C2a = drop_t >= 0.30
    C2c = abs(drop_o) <= 0.10 and abs(drop_w) <= 0.10
    C2d = ppl_ratio <= 1.05
    return {"drop_target": drop_t, "drop_other_belief": drop_o, "drop_wk": drop_w,
            "ppl_ratio": ppl_ratio, "ppl_ablated": ppl_abl["ppl"],
            "C2a": C2a, "C2c": C2c, "C2d": C2d}


# -- greedy search ------------------------------------------------------------
def greedy_search(net, tok, ranked, ppl_tokens, info, clean_accs, clean_ppl,
                  t_ex, o_ex, wk_ex, max_heads=30) -> Optional[list]:
    S = []
    print(f"  [M2.3] greedy-add (cap={max_heads}) ...")
    for i, h in enumerate(ranked):
        if i >= max_heads:
            return None
        S.append(h)
        m = eval_hstar(net, tok, S, ppl_tokens, info, clean_accs, clean_ppl, t_ex, o_ex, wk_ex)
        print(f"    |S|={len(S)} {h}: C2a={m['C2a']} C2c={m['C2c']} C2d={m['C2d']} "
              f"drop={m['drop_target']:.3f} ppl_r={m['ppl_ratio']:.4f}")
        if m["C2a"] and m["C2c"] and m["C2d"]:
            S_min = list(S)
            print(f"  [M2.3] criteria met at |S|={len(S_min)} -> greedy-remove ...")
            changed = True
            while changed:
                changed = False
                for h_rem in list(S_min):
                    S_try = [hh for hh in S_min if hh != h_rem]
                    if not S_try:
                        continue
                    m2 = eval_hstar(net, tok, S_try, ppl_tokens, info,
                                    clean_accs, clean_ppl, t_ex, o_ex, wk_ex)
                    if m2["C2a"] and m2["C2c"] and m2["C2d"]:
                        print(f"    -removed {h_rem} |S|={len(S_try)} still ok")
                        S_min = S_try; changed = True; break
            return S_min
    return None


# -- controls -----------------------------------------------------------------
def run_controls(net, tok, hstar, ppl_tokens, info, clean_accs, clean_ppl,
                 t_ex, o_ex, wk_ex, seeds, n=20) -> dict:
    all_h = [(l, h) for l in range(info["n_layers"]) for h in range(info["n_heads"])]
    k = len(hstar); drops = []
    print(f"  [M2.4] {n} random-head controls (k={k}) ...")
    for seed in seeds[:n]:
        rand_h = random.Random(seed).sample(all_h, min(k, len(all_h)))
        m = eval_hstar(net, tok, rand_h, ppl_tokens, info, clean_accs, clean_ppl, t_ex, o_ex, wk_ex)
        drops.append(m["drop_target"])
        print(f"    seed={seed} drop={m['drop_target']:.3f}")
    mean = float(np.mean(drops))
    sigma = float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0
    return {"n_controls": len(drops), "target_drops": drops,
            "mean": mean, "sigma": sigma, "threshold_2sigma": mean + 2 * sigma}


# -- main ---------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg)
        with open(OUT_DIR / "run.log", "a") as f:
            f.write(msg + "\n")

    log(f"[variant] START {time.strftime('%Y-%m-%d %H:%M:%S')}")

    net, tok = load_olmo(dtype=torch.float16)
    info = olmo_arch_info(net)
    log(f"[variant] arch={info}")

    results: dict = {"model": "olmo-1b", "model_path": MODEL_PATH, "arch": info}
    ppl_tokens = torch.load(PPL_SAMPLE_PATH, map_location="cpu")
    log(f"[PPL] {ppl_tokens.numel()} tokens, max_id={ppl_tokens.max().item()}, vocab={net.config.vocab_size}")

    # M1 (resume if exists) ---------------------------------------------------
    m1_path = OUT_DIR / "m1_gate.json"
    if m1_path.exists():
        gate = load_json(str(m1_path))
        log("[M1] loaded from disk")
    else:
        gate = run_m1_gate(net, tok, info)
        save_json(str(m1_path), gate)
    results["m1_gate"] = gate
    targets = [t for t in ["personal_belief", "attributed_belief"] if gate[t]["above_chance"]]
    log(f"[M1] admitted: {targets}")

    if not targets:
        results.update({"m2_results": {}, "n_localized": 0, "n_admitted": 0,
                        "claim_verdict": "0/0 (M1 failed)",
                        "consistent_with_main_experiment_indicator": False})
        save_json(str(OUT_DIR / "result.json"), results); return

    # F_knowledge (resume if ranked heads already on disk for both targets) ---
    need_fknow = any(not (OUT_DIR / f"ranked_heads_{t}.json").exists() for t in targets)
    F_know = None
    if need_fknow:
        wk_fisher = load_task(DATA_ROOT, "world_knowledge")
        log(f"[M2.1] F_knowledge: {len(wk_fisher)} examples")
        net.float(); net.train()
        for p in net.parameters(): p.requires_grad_(True)
        F_know = compute_fisher(net, tok, wk_fisher)
        net.half(); net.eval()
        for p in net.parameters(): p.requires_grad_(False)
        log(f"[M2.1] F_knowledge done ({len(F_know)} tensors)")
    else:
        log("[M2.1] F_knowledge not needed (all ranked_heads.json on disk)")

    m2_results: dict = {}

    for target in targets:
        log(f"\n{'='*60}")
        log(f"[M2] target={target}")
        other = "attributed_belief" if target == "personal_belief" else "personal_belief"
        t_full = load_task(DATA_ROOT, target)
        o_full = load_task(DATA_ROOT, other)
        wk_full = load_task(DATA_ROOT, "world_knowledge")

        # -- Clean baselines (resume if on disk) ------------------------------
        cb_path = OUT_DIR / f"clean_baselines_{target}.json"
        if cb_path.exists():
            cb = load_json(str(cb_path))
            clean_accs = {"target": cb["target"]["acc"], "other": cb["other"]["acc"],
                          "wk": cb["wk"]["acc"]}
            clean_ppl = cb["ppl"]["ppl"]
            log(f"  [clean] loaded from disk: target={clean_accs['target']:.3f} ppl={clean_ppl:.3f}")
        else:
            log(f"  [clean] pre-computing baselines ...")
            ct = eval_task_fast(net, tok, t_full, info=info)
            co = eval_task_fast(net, tok, o_full, info=info)
            cw = eval_task_fast(net, tok, wk_full, info=info)
            ppl_clean_r = evaluate_ppl(net, ppl_tokens, device=DEVICE)
            clean_accs = {"target": ct["acc"], "other": co["acc"], "wk": cw["acc"]}
            clean_ppl = ppl_clean_r["ppl"]
            save_json(str(cb_path), {"target": ct, "other": co, "wk": cw, "ppl": ppl_clean_r})
            log(f"  clean: target={ct['acc']:.3f} other={co['acc']:.3f} wk={cw['acc']:.3f} ppl={clean_ppl:.3f}")

        # -- M2.1 Fisher target (resume if ranked_heads on disk) --------------
        rh_path = OUT_DIR / f"ranked_heads_{target}.json"
        if rh_path.exists():
            ranked_raw = load_json(str(rh_path))
            ranked_heads = [tuple(r[0]) for r in ranked_raw]
            mask_scores = {tuple(r[0]): r[1] for r in ranked_raw}
            rho = load_json(str(OUT_DIR / f"jackknife_{target}.json")).get("rho", None)
            log(f"  [M2.1/M2.2] loaded ranked heads from disk (top-5: {ranked_heads[:5]})")
        else:
            # Need to compute Fisher for this target
            t_fisher = load_task(DATA_ROOT, target, person_filter=["james", "mary"])
            log(f"[M2.1] F_{target}: {len(t_fisher)} examples")
            assert F_know is not None, "F_know not computed but ranked heads not on disk"
            net.float(); net.train()
            for p in net.parameters(): p.requires_grad_(True)
            F_tgt = compute_fisher(net, tok, t_fisher)
            half = len(t_fisher) // 2
            F_a = compute_fisher(net, tok, t_fisher[:half])
            F_b = compute_fisher(net, tok, t_fisher[half:])
            net.half(); net.eval()
            for p in net.parameters(): p.requires_grad_(False)

            hs_t = agg_head_scores(F_tgt, info)
            hs_a = agg_head_scores(F_a, info)
            hs_b = agg_head_scores(F_b, info)
            ks = sorted(hs_a)
            rho = float(scipy_stats.spearmanr([hs_a[k] for k in ks],
                                               [hs_b[k] for k in ks])[0])
            log(f"  [Jackknife] rho={rho:.3f}")
            save_json(str(OUT_DIR / f"head_scores_{target}.json"),
                      {str(k): v for k, v in hs_t.items()})
            save_json(str(OUT_DIR / f"jackknife_{target}.json"),
                      {"rho": rho, "n_half_a": half, "n_half_b": len(t_fisher)-half})

            log("[M2.2] AND-NOT mask ...")
            mask_scores, ranked_heads = build_mask(F_tgt, F_know, 0.001, 0.01, info)
            save_json(str(OUT_DIR / f"mask_scores_{target}.json"),
                      {str(k): v for k, v in mask_scores.items()})
            save_json(str(rh_path),
                      [[list(h), mask_scores[h]] for h in ranked_heads[:50]])
            log(f"  Top-5: {ranked_heads[:5]}")

            # === CRITICAL: free Fisher memory before greedy search ===
            del F_tgt, F_a, F_b, hs_t, hs_a, hs_b
            gc.collect()
            torch.cuda.empty_cache()
            log(f"  [mem] Fisher tensors freed. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        # Also free F_know if all targets are done computing (after last target)
        # F_know not needed from here on for this target
        if target == targets[-1] and F_know is not None:
            del F_know
            F_know = None
            gc.collect()
            torch.cuda.empty_cache()
            log(f"  [mem] F_know freed. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        # -- M2.3 Greedy H* search -------------------------------------------
        acc_path = OUT_DIR / f"acceptance_{target}.json"
        if acc_path.exists():
            acc_data = load_json(str(acc_path))
            log(f"  [M2.3/M2.4] loaded acceptance from disk: status={acc_data.get('status')}")
            m2_results[target] = {
                "status": acc_data["status"],
                "hstar_heads": acc_data["hstar_heads"],
                "hstar_size": acc_data["hstar_size"],
                "drop_target": acc_data["drops"]["target"],
                "ppl_ratio": acc_data["drops"]["ppl_ratio"],
                "C2a": acc_data["criteria"]["C2a_target_drop_ge_0.30"],
                "C2b": acc_data["criteria"]["C2b_target_drop_gt_mean_plus_2sigma"],
                "C2c": acc_data["criteria"]["C2c_offtarget_drops_le_0.10"],
                "C2d": acc_data["criteria"]["C2d_ppl_ratio_le_1.05"],
                "jackknife_rho": acc_data.get("jackknife_rho"),
            }
            continue

        log("[M2.3] Greedy H* search ...")
        hstar = greedy_search(net, tok, ranked_heads, ppl_tokens, info,
                              clean_accs, clean_ppl, t_full, o_full, wk_full)

        if hstar is None:
            log(f"  H* not found -> not_localized")
            m2_results[target] = {"status": "not_localized", "hstar_heads": None,
                                   "jackknife_rho": rho}
            save_json(str(OUT_DIR / f"m2_result_{target}.json"), m2_results[target])
            continue

        hstar_json = [[l, h] for l, h in hstar]
        save_json(str(OUT_DIR / f"hstar_{target}.json"), hstar_json)
        log(f"  H* = {hstar}  |H*|={len(hstar)}")

        # Final H* metrics
        hm = eval_hstar(net, tok, hstar, ppl_tokens, info, clean_accs, clean_ppl,
                        t_full, o_full, wk_full)
        hm["ppl_clean"] = clean_ppl
        log(f"  drop_t={hm['drop_target']:.3f} drop_o={hm['drop_other_belief']:.3f} "
            f"drop_w={hm['drop_wk']:.3f} ppl_r={hm['ppl_ratio']:.4f}")

        # M2.4 controls
        log("[M2.4] Random-head controls ...")
        ctrl = run_controls(net, tok, hstar, ppl_tokens, info, clean_accs, clean_ppl,
                            t_full, o_full, wk_full, seeds=list(range(100, 120)), n=20)
        C2b = hm["drop_target"] > ctrl["threshold_2sigma"]
        log(f"  C2b: drop={hm['drop_target']:.3f} > thresh={ctrl['threshold_2sigma']:.3f} -> {C2b}")

        all_pass = hm["C2a"] and C2b and hm["C2c"] and hm["C2d"]
        status = "localized" if all_pass else "partially_localized"
        log(f"  {status}  C2a={hm['C2a']} C2b={C2b} C2c={hm['C2c']} C2d={hm['C2d']}")

        acceptance = {
            "model": "olmo-1b", "target": target, "status": status,
            "hstar_heads": hstar_json, "hstar_size": len(hstar),
            "drops": {"target": hm["drop_target"], "other_belief": hm["drop_other_belief"],
                      "world_knowledge": hm["drop_wk"], "ppl_ratio": hm["ppl_ratio"]},
            "ppl_clean": clean_ppl, "ppl_ablated": hm["ppl_ablated"],
            "random_head_controls": ctrl,
            "criteria": {
                "C2a_target_drop_ge_0.30": hm["C2a"],
                "C2b_target_drop_gt_mean_plus_2sigma": C2b,
                "C2c_offtarget_drops_le_0.10": hm["C2c"],
                "C2d_ppl_ratio_le_1.05": hm["C2d"],
            },
            "jackknife_rho": rho,
        }
        save_json(str(acc_path), acceptance)
        m2_results[target] = {
            "status": status, "hstar_heads": hstar_json, "hstar_size": len(hstar),
            "drop_target": hm["drop_target"], "ppl_ratio": hm["ppl_ratio"],
            "C2a": hm["C2a"], "C2b": C2b, "C2c": hm["C2c"], "C2d": hm["C2d"],
            "jackknife_rho": rho,
        }

    # Summary -----------------------------------------------------------------
    n_loc = sum(1 for r in m2_results.values() if r.get("status") == "localized")
    n_adm = len(targets)
    results.update({
        "m2_results": m2_results, "n_localized": n_loc, "n_admitted": n_adm,
        "consistent_with_main_experiment_indicator": n_loc >= 1,
        "claim_verdict": f"{n_loc}/{n_adm} targets localized on OLMo-1B",
    })
    save_json(str(OUT_DIR / "result.json"), results)
    log(f"\n[variant] DONE {n_loc}/{n_adm} localized consistent={n_loc>=1}")
    log(f"[variant] END {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
