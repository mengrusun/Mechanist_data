"""
M3 — Applied Control: three-arm comparison at test.

Arm A (Circuit)     : additive-injection C_e enhancement, α_A ∈ {0.5, 1.0, 2.0}, (k_h, k_n) neighborhood
Arm B (Prompting)   : 3 templates × 3 positions (prefix/suffix/interleaved) = 9 val configs
Arm C (Steering)    : residual-additive d_e (train-fold-frozen mean-diff, LAST-EVENT-TOKEN residual dir),
                      L ∈ top-3 layers by probe AUC, α_C ∈ {0.5, 1.0, 2.0}, applied at every token
                      after event stem during autoregressive generation (CAA convention)

Judge: gpt-5.4 hidden-target 6-way forced choice (three-way {CORRECT, INCORRECT, OTHER} per tips).
Metric: per-emotion accuracy, macro-avg, paired-bootstrap 95% CI on (A-B) and (A-C).

Val: 120 val stems, best config per arm per emotion by val macro-accuracy (judge-scored).
     Note: to keep val cost sane, we score val configs by *val target-prefix log-prob gain* (fast, judge-free)
     and reserve the external judge for the FINAL eval — the plan permits this because it says val is
     "cheap (next-token log-prob or short greedy)".
Eval: 120 eval stems x 6 emotions x 3 arms x 1 selected config = 2160 continuations, then judge.

Outputs (runs/A3_applied/):
  arm_A_val.json, arm_B_val.json, arm_C_val.json
  selected_configs.json
  arm_A_eval.json, arm_B_eval.json, arm_C_eval.json
  judge_results.json
  metrics.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
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
    LLAMA_PATH,
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
    hidden_dim_of,
    n_heads_of,
    n_layers_of,
    n_neurons_of,
    target_prefix_logprob_batch,
)
from m2_causal import _load_C_e_from_run, paired_bootstrap_ci


def _parse_env_alphas(name: str, default: list[float]) -> list[float]:
    v = os.environ.get(name)
    if not v:
        return default
    return [float(x) for x in v.split(",") if x.strip()]


def _parse_env_ks(name: str, default: list[int]) -> list[int]:
    v = os.environ.get(name)
    if not v:
        return default
    return [int(x) for x in v.split(",") if x.strip()]


# Env overrides let iteration-loop fixes tune α range + component count without patching the file.
#   ALPHAS_ARM_A=0.05,0.1,0.3     ← reduced range to counter cumulative-injection OOD
#   K_H_ARM_A=5,12,24             ← reduced head count (from default {12,24,48} neighborhood)
#   K_N_ARM_A=200,500,2000        ← reduced neuron count (from default {1000,2000,4000})
ALPHAS_ARM_A = _parse_env_alphas("ALPHAS_ARM_A", [0.5, 1.0, 2.0])
ALPHAS_ARM_C = _parse_env_alphas("ALPHAS_ARM_C", [0.5, 1.0, 2.0])
K_H_ARM_A_OVERRIDE = _parse_env_ks("K_H_ARM_A", [])   # if empty, keep original neighborhood logic
K_N_ARM_A_OVERRIDE = _parse_env_ks("K_N_ARM_A", [])
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "100"))

# Arm B templates
ARM_B_TEMPLATES = [
    "Answer as if you feel {emotion_word}. This is important.",
    "You currently feel {emotion_word}. Respond accordingly.",
    "This scenario made you feel {emotion_word}. Continue the story.",
]
ARM_B_POSITIONS = ["prefix", "suffix", "interleaved"]


def format_arm_b_prompt(event: str, emotion: str, template: str, position: str) -> str:
    inj = template.format(emotion_word=emotion)
    if position == "prefix":
        return inj + " " + event
    elif position == "suffix":
        return event + " " + inj
    elif position == "interleaved":
        # split event around the "and you experience" phrase
        parts = event.split("and you experience", 1)
        if len(parts) == 2:
            return parts[0].strip() + " " + inj + " and you experience" + parts[1]
        return event + " " + inj
    raise ValueError(position)


# ---------------------------------------------------------------------------
# Arm C — steering vector hook (residual-additive at every post-event token during generation)
# ---------------------------------------------------------------------------
def _install_arm_c_hook(model, layer_l: int, direction_residual: np.ndarray, alpha: float, event_len_lookup):
    """
    CAA convention: add alpha * d to the residual output of layer `layer_l` at every token
    position from event_len onwards (i.e. only at CONTINUATION positions).

    Implementation: use a stateful counter in the closure. When the hook fires on a tensor
    of shape (B, T, H):
      - T == prompt_len (first pass through the full prompt): skip pushing (event tokens).
      - T == 1 (subsequent decoded token during generation): push (this is a continuation token).
      - T > 1 but < prompt_len: (unusual — batch continuation forward), push at last position only.

    The prompt length is *not* known here, so we detect first-pass vs. continuation via the
    T>1 vs T==1 test. This matches HuggingFace's `generate()` with `use_cache=True`.

    For plain `target_prefix_logprob_batch` (single forward, T = prompt+prefix), the caller
    passes prompts *including* the target prefix — pushing everywhere would corrupt the
    metric. So we skip when T > 1.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    d = torch.tensor(direction_residual, dtype=dtype, device=device)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        # Only push on single-token continuation forwards (KV-cache active decode step).
        # This is the standard cheap CAA hook: works with generate(), does not corrupt static
        # logprob measurement, and correctly injects at continuation positions.
        if hs.shape[1] == 1:
            hs = hs + alpha * d
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    h = model.model.layers[layer_l].register_forward_hook(hook)
    return [h]


def _install_arm_c_hook_static(model, layer_l: int, direction_residual: np.ndarray, alpha: float, event_len_per_row=None):
    """
    Variant of Arm C hook for static-forward (non-generate) logprob measurement:
    push at positions >= event_len for each row. Used for val-scoring Arm C via
    target-prefix logprob gain (where the input already includes the target prefix).

    Reads per-batch event lengths from `m1_location._HOOK_EVENT_LENS`, which the
    `target_prefix_logprob_batch` helper publishes before each forward pass. This
    guarantees the injection positions match the actual per-batch tokenization.
    Falls back to `event_len_per_row` if the shared list is empty (legacy path).
    """
    from m1_location import _HOOK_EVENT_LENS

    dtype = next(model.parameters()).dtype
    d = torch.tensor(direction_residual, dtype=dtype, device=next(model.parameters()).device)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        B, T, H = hs.shape
        # Prefer the per-batch published event lengths.
        if len(_HOOK_EVENT_LENS) == B:
            for i in range(B):
                el = int(_HOOK_EVENT_LENS[i])
                if el < T:
                    hs[i, el:] = hs[i, el:] + alpha * d
        elif event_len_per_row is not None and len(event_len_per_row) >= B:
            for i in range(B):
                el = int(event_len_per_row[i])
                if el < T:
                    hs[i, el:] = hs[i, el:] + alpha * d
        else:
            # Fallback: push at last position only.
            hs[:, -1] = hs[:, -1] + alpha * d
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    h = model.model.layers[layer_l].register_forward_hook(hook)
    return [h]


# ---------------------------------------------------------------------------
# Batched greedy generation with hook active
# ---------------------------------------------------------------------------
@torch.no_grad()
def batched_generate(model, tok, prompts: list[str], max_new_tokens: int = MAX_NEW_TOKENS, batch_size: int = 4) -> list[str]:
    from m1_location import _HOOK_EVENT_LENS
    device = next(model.parameters()).device
    results = []
    tok.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        input_len = enc["input_ids"].shape[1]
        # Clear stale _HOOK_EVENT_LENS so the M1/M3 hooks take the else branch on the
        # initial full-prompt forward: push at position -1 (last real token under left-padding).
        # Then during KV-cache decode (T=1), push at the single new token.
        _HOOK_EVENT_LENS.clear()
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.pad_token_id,
        )
        # Extract only new tokens
        for i in range(out.shape[0]):
            new_tokens = out[i, input_len:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            results.append(text.strip())
    tok.padding_side = "right"
    return results


# ---------------------------------------------------------------------------
# Val scoring — judge-free target-prefix logprob gain (fast)
# ---------------------------------------------------------------------------
def _val_score_arm_A(model, tok, val_events, C_e, directions, sign_std_cache, emotion, k_h, k_n, alpha,
                     eval_neuron_by_emo, baseline_lp):
    d_e = directions[emotion]
    # Pool: use full C_e (already at (k_h*, k_n*)) or a subset
    heads = C_e[emotion]["heads"][:k_h] if k_h <= len(C_e[emotion]["heads"]) else C_e[emotion]["heads"]
    neurons = C_e[emotion]["neurons"][:k_n] if k_n <= len(C_e[emotion]["neurons"]) else C_e[emotion]["neurons"]
    heads_p = [(int(l), int(h), d_e[int(l)]) for (l, h) in heads]
    neurons_p = []
    for (l, n) in neurons:
        l, n = int(l), int(n)
        key = (emotion, l, n)
        if key not in sign_std_cache:
            with torch.no_grad():
                w = model.model.layers[l].mlp.down_proj.weight[:, n].float().cpu().numpy()
            sign_val = float(np.sign(w.dot(d_e[l])))
            if sign_val == 0.0:
                sign_val = 1.0
            std_val = float(eval_neuron_by_emo[emotion][:, l, n].std() + 1e-6)
            sign_std_cache[key] = (sign_val, std_val)
        sign_val, std_val = sign_std_cache[key]
        neurons_p.append((l, n, sign_val, std_val))

    handles = _install_multi_component_hooks(model, heads=heads_p, neurons=neurons_p, alpha=alpha)
    try:
        prefixes = [target_prefix(emotion)] * len(val_events)
        lp = target_prefix_logprob_batch(model, tok, val_events, prefixes, batch_size=8)
    finally:
        _remove_hooks(handles)
    return float((lp - baseline_lp[emotion]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1_dir", default=str(REPO_DIR / "runs" / "A1_location"))
    ap.add_argument("--out_dir", default=str(REPO_DIR / "runs" / "A3_applied"))
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--model_path", default=LLAMA_PATH)
    ap.add_argument("--skip_judge", action="store_true", help="Skip the external judge; produce continuations only.")
    ap.add_argument("--judge_workers", type=int, default=6)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m1_dir = Path(args.m1_dir)

    log("M3: loading M1 artifacts")
    C_e, kstar, directions, shortlist = _load_C_e_from_run(m1_dir)
    dirs_all = np.load(m1_dir / "directions.npz")

    log("M3: loading model + SEV split")
    model, tok = load_model(args.model_path)
    L = n_layers_of(model)
    NH = n_heads_of(model)

    sev = load_sev()
    split = build_scenario_split(sev, seed=args.seed)
    val_items = split["val"]
    eval_items = split["eval"]

    if args.sanity:
        emotions = EMOTIONS[:2]
        val_items = val_items[:15]
        eval_items = eval_items[:10]
    else:
        emotions = list(C_e.keys())

    val_events = [r["event"] for r in val_items]
    eval_events = [r["event"] for r in eval_items]
    log(f"M3: |val|={len(val_events)}, |eval|={len(eval_events)}, emotions={emotions}")

    # Precompute per-emotion neuron activations on eval fold for sign+std
    from m1_location import collect_head_and_neuron_activations
    log("M3: collecting per-emotion neuron activations on eval events (for sign/std)")
    eval_neuron_by_emo = {}
    for e in emotions:
        prefix = target_prefix(e).strip()
        ctx = [ev + " " + prefix for ev in eval_events]
        _, eval_neuron_by_emo[e] = collect_head_and_neuron_activations(model, tok, ctx, batch_size=4)

    # Baseline logprobs on val + eval per emotion
    log("M3: computing baseline logprobs on val and eval per emotion")
    baseline_val_lp = {}
    baseline_eval_lp = {}
    for e in emotions:
        p_val = [target_prefix(e)] * len(val_events)
        p_ev = [target_prefix(e)] * len(eval_events)
        baseline_val_lp[e] = target_prefix_logprob_batch(model, tok, val_events, p_val, batch_size=8)
        baseline_eval_lp[e] = target_prefix_logprob_batch(model, tok, eval_events, p_ev, batch_size=8)

    sign_std_cache = {}

    # ---- Arm A val: 9 configs = 3 (k_h, k_n) neighborhood x 3 α_A ----
    log("M3 Arm A: val sweep (9 configs per emotion)")
    kh, kn = kstar["k_h"], kstar["k_n"]
    # 3-cell neighborhood clipped to grid, OR use env-provided override for reduced-magnitude fixes.
    if K_H_ARM_A_OVERRIDE:
        K_H_neigh = list(K_H_ARM_A_OVERRIDE)
    else:
        K_H_neigh = sorted({max(K_H_GRID[0], kh // 2), kh, min(K_H_GRID[-1], kh * 2)})[:3]
    if K_N_ARM_A_OVERRIDE:
        K_N_neigh = list(K_N_ARM_A_OVERRIDE)
    else:
        K_N_neigh = sorted({max(K_N_GRID[0], kn // 2), kn, min(K_N_GRID[-1], kn * 2)})[:3]
    log(f"M3 Arm A: K_H_neigh={K_H_neigh}, K_N_neigh={K_N_neigh}, alphas={ALPHAS_ARM_A}")
    arm_a_val = {}
    for e in emotions:
        rows = []
        for k_h_i in K_H_neigh:
            for k_n_i in K_N_neigh:
                for a in ALPHAS_ARM_A:
                    score = _val_score_arm_A(model, tok, val_events, C_e, directions, sign_std_cache, e, k_h_i, k_n_i, a,
                                              eval_neuron_by_emo, baseline_val_lp)
                    rows.append({"k_h": k_h_i, "k_n": k_n_i, "alpha": a, "val_score": score})
        arm_a_val[e] = rows
        best = max(rows, key=lambda r: r["val_score"])
        log(f"  emo={e} arm A best: k_h={best['k_h']}, k_n={best['k_n']}, α={best['alpha']}, gain={best['val_score']:+.4f}")
    save_json(out_dir / "arm_A_val.json", arm_a_val)

    # ---- Arm B val: 3 templates × 3 positions ----
    log("M3 Arm B: val sweep (9 configs per emotion)")
    arm_b_val = {}
    for e in emotions:
        rows = []
        for ti, tmpl in enumerate(ARM_B_TEMPLATES):
            for pos in ARM_B_POSITIONS:
                # Score via target-prefix logprob gain on val
                prompts = [format_arm_b_prompt(ev, e, tmpl, pos) for ev in val_events]
                prefixes = [target_prefix(e)] * len(prompts)
                # Custom score: measure log P(prefix_e | modified_prompt) - log P(prefix_e | plain_event)
                lp = target_prefix_logprob_batch(model, tok, prompts, prefixes, batch_size=8)
                gain = float((lp - baseline_val_lp[e]).mean())
                rows.append({"template_idx": ti, "position": pos, "val_score": gain})
        arm_b_val[e] = rows
        best = max(rows, key=lambda r: r["val_score"])
        log(f"  emo={e} arm B best: T{best['template_idx']}+{best['position']}, gain={best['val_score']:+.4f}")
    save_json(out_dir / "arm_B_val.json", arm_b_val)

    # ---- Arm C val: 3 layers × 3 α_C ----
    # Note: val scoring uses static forward + target-prefix log-prob. We use the STATIC hook
    # variant so the injection lands only at the target-prefix positions (CAA semantics for
    # measurement matches CAA semantics at generation).
    log("M3 Arm C: val sweep (9 configs per emotion)")
    arm_c_val = {}
    for e in emotions:
        haucs = dirs_all[f"head_auc_{e}"]
        nscores = dirs_all[f"neuron_score_{e}"]
        per_layer_sum = haucs.sum(axis=1) + nscores.sum(axis=1)
        top_layers = sorted(np.argsort(-per_layer_sum)[:TOP_N_LAYERS].tolist())
        d_e = directions[e]
        rows = []
        for l in top_layers:
            for a in ALPHAS_ARM_C:
                # Val scoring uses the static hook — the target-prefix logprob helper
                # publishes per-batch event lengths so the injection lands only at prefix
                # positions (CAA semantics preserved during measurement).
                handles = _install_arm_c_hook_static(model, l, d_e[l], a)
                try:
                    prefixes = [target_prefix(e)] * len(val_events)
                    lp = target_prefix_logprob_batch(model, tok, val_events, prefixes, batch_size=8)
                finally:
                    _remove_hooks(handles)
                gain = float((lp - baseline_val_lp[e]).mean())
                rows.append({"layer": int(l), "alpha": a, "val_score": gain})
        arm_c_val[e] = rows
        best = max(rows, key=lambda r: r["val_score"])
        log(f"  emo={e} arm C best: L={best['layer']}, α={best['alpha']}, gain={best['val_score']:+.4f}")
    save_json(out_dir / "arm_C_val.json", arm_c_val)

    # ---- Select per-emotion best config for each arm ----
    selected = {"A": {}, "B": {}, "C": {}}
    for e in emotions:
        selected["A"][e] = max(arm_a_val[e], key=lambda r: r["val_score"])
        selected["B"][e] = max(arm_b_val[e], key=lambda r: r["val_score"])
        selected["C"][e] = max(arm_c_val[e], key=lambda r: r["val_score"])
    save_json(out_dir / "selected_configs.json", selected)

    # ---- Eval: generate continuations with each arm's best config, 120 eval stems × 6 emotions × 3 arms
    log("M3 eval: greedy generation on eval stems")
    eval_generations = {"A": {}, "B": {}, "C": {}}  # arm -> emotion -> list of {event_id, event, continuation}
    for e in emotions:
        # Arm A
        conf_a = selected["A"][e]
        k_h_i, k_n_i, a = conf_a["k_h"], conf_a["k_n"], conf_a["alpha"]
        d_e = directions[e]
        heads = C_e[e]["heads"][:k_h_i]
        neurons = C_e[e]["neurons"][:k_n_i]
        heads_p = [(int(l), int(h), d_e[int(l)]) for (l, h) in heads]
        neurons_p = []
        for (l, n) in neurons:
            l, n = int(l), int(n)
            key = (e, l, n)
            if key not in sign_std_cache:
                with torch.no_grad():
                    w = model.model.layers[l].mlp.down_proj.weight[:, n].float().cpu().numpy()
                sign_val = float(np.sign(w.dot(d_e[l])))
                if sign_val == 0.0:
                    sign_val = 1.0
                std_val = float(eval_neuron_by_emo[e][:, l, n].std() + 1e-6)
                sign_std_cache[key] = (sign_val, std_val)
            sign_val, std_val = sign_std_cache[key]
            neurons_p.append((l, n, sign_val, std_val))
        handles = _install_multi_component_hooks(model, heads=heads_p, neurons=neurons_p, alpha=a)
        try:
            gens = batched_generate(model, tok, eval_events, batch_size=4)
        finally:
            _remove_hooks(handles)
        eval_generations["A"][e] = [{"event_id": eval_items[i]["id"], "event": eval_events[i], "continuation": gens[i]} for i in range(len(eval_events))]
        log(f"  emo={e} arm A eval done ({len(gens)} continuations)")

        # Arm B
        conf_b = selected["B"][e]
        tmpl = ARM_B_TEMPLATES[conf_b["template_idx"]]
        pos = conf_b["position"]
        prompts_b = [format_arm_b_prompt(ev, e, tmpl, pos) for ev in eval_events]
        gens_b = batched_generate(model, tok, prompts_b, batch_size=4)
        # Strip the prompting sentence from continuation if it echoes back (unlikely with instruct-tuned)
        eval_generations["B"][e] = [{"event_id": eval_items[i]["id"], "event": eval_events[i], "continuation": gens_b[i]} for i in range(len(eval_events))]
        log(f"  emo={e} arm B eval done")

        # Arm C
        conf_c = selected["C"][e]
        l_c, a_c = conf_c["layer"], conf_c["alpha"]
        handles = _install_arm_c_hook(model, l_c, d_e[l_c], a_c, None)
        try:
            gens_c = batched_generate(model, tok, eval_events, batch_size=4)
        finally:
            _remove_hooks(handles)
        eval_generations["C"][e] = [{"event_id": eval_items[i]["id"], "event": eval_events[i], "continuation": gens_c[i]} for i in range(len(eval_events))]
        log(f"  emo={e} arm C eval done")

    save_json(out_dir / "eval_generations.json", eval_generations)

    # ---- Judge ----
    if args.skip_judge:
        log("M3: skip judge, done")
        return

    log("M3: running gpt-5.4 judge on all eval continuations")
    judge = JudgeClient()
    all_tasks = []
    for arm in ["A", "B", "C"]:
        for e in emotions:
            for row in eval_generations[arm][e]:
                all_tasks.append({"arm": arm, "target": e, "event_id": row["event_id"], "event": row["event"], "continuation": row["continuation"]})
    log(f"M3: {len(all_tasks)} judge calls")

    def _worker(t):
        r = judge.judge(t["event"], t["continuation"], t["target"])
        return {**t, **r}

    judge_results = []
    with cf.ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
        for i, res in enumerate(ex.map(_worker, all_tasks)):
            judge_results.append(res)
            if (i + 1) % 100 == 0:
                log(f"  judged {i + 1}/{len(all_tasks)}")
    save_json(out_dir / "judge_results.json", judge_results)

    # Compute per-emotion accuracy per arm
    accs = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "incorrect": 0, "other": 0, "n": 0, "per_row_correct": []}))
    for r in judge_results:
        rec = accs[r["arm"]][r["target"]]
        rec["n"] += 1
        v = r["verdict"]
        if v == "CORRECT":
            rec["correct"] += 1
            rec["per_row_correct"].append(1)
        elif v == "INCORRECT":
            rec["incorrect"] += 1
            rec["per_row_correct"].append(0)
        else:
            rec["other"] += 1
            rec["per_row_correct"].append(0)  # OTHER counted as not-correct for accuracy denominator

    per_emo_acc = {arm: {} for arm in ["A", "B", "C"]}
    for arm in ["A", "B", "C"]:
        for e in emotions:
            rec = accs[arm][e]
            acc = rec["correct"] / max(rec["n"], 1)
            per_emo_acc[arm][e] = {"n": rec["n"], "correct": rec["correct"], "incorrect": rec["incorrect"], "other": rec["other"], "accuracy": acc, "per_row_correct": rec["per_row_correct"]}

    # Macro-avg per arm
    macro = {arm: float(np.mean([per_emo_acc[arm][e]["accuracy"] for e in emotions])) for arm in ["A", "B", "C"]}

    # Paired-bootstrap CI on (A-B) and (A-C) per emotion
    def paired_diff_ci(a_bin, b_bin, n_boot=1000, seed=0):
        rng = np.random.default_rng(seed)
        a_arr = np.array(a_bin)
        b_arr = np.array(b_bin)
        n = min(len(a_arr), len(b_arr))
        a_arr, b_arr = a_arr[:n], b_arr[:n]
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
    # Length audit
    def mean_len(arm):
        lens = []
        for e in emotions:
            for row in eval_generations[arm][e]:
                lens.append(len(row["continuation"].split()))
        return float(np.mean(lens))
    lens = {arm: mean_len(arm) for arm in ["A", "B", "C"]}

    metrics = {
        "per_emotion_accuracy": per_emo_acc,
        "macro_accuracy": macro,
        "pairwise": pairwise,
        "n_A_gt_B_at_95": n_ab,
        "n_A_gt_C_at_95": n_ac,
        "claim_3_criteria": {"threshold_over_6": 5},
        "A_gt_B_pass": n_ab >= 5,
        "A_gt_C_pass": n_ac >= 5,
        "claim_3_verdict": ("supported" if (n_ab >= 5 and n_ac >= 5)
                             else ("A>B_only" if n_ab >= 5
                                   else ("A>C_only" if n_ac >= 5
                                         else "not-supported"))),
        "length_audit": {"mean_words": lens,
                         "max_pairwise_ratio": float(max(lens.values()) / min(lens.values()))},
    }
    save_json(out_dir / "metrics.json", metrics)
    log(f"M3: DONE. Verdict = {metrics['claim_3_verdict']}. A>B: {n_ab}/{len(emotions)}, A>C: {n_ac}/{len(emotions)}. Macro: A={macro['A']:.3f}, B={macro['B']:.3f}, C={macro['C']:.3f}")


if __name__ == "__main__":
    main()
