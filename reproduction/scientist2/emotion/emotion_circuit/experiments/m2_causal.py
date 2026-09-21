"""
M2 — Causal Intervention + Stability.

Depends on runs/A1_location/{C_e.json, directions.npz, kstar.json, shortlist.json}.

Predicates (rubric):
  (a) causal-sign + dose:
        mean Δ_ablation(target) < 0 at α2
        mean Δ_enhance(target) > 0 at α2
        Spearman(α, Δ_enhance) >= 0.7 over 3 α ∈ {0.5, 1.0, 2.0}
  (b) raw off-target: |off-target Δ| < |target Δ| at α2
  (c) targeted-C_{e'}: |Δ_{C_e on e}| > |Δ_{C_{e'} on e}| at α2 with paired-bootstrap CI on difference excluding 0
  (d) scenario stability: Jaccard(S1, S2) − perm-null-mean > 0 at 95% CI
  (e) secondary Claim 2e: pairwise-Jaccard(neurons) < pairwise-Jaccard(heads)
Random-set null (Claim 2 anti-claim): 100 random component sets per emotion.

Outputs (runs/A2_causal/):
  ablation.json    -> Δ per emotion under ablation + paired-bootstrap CI
  enhancement.json -> Δ per emotion at α ∈ {0.5,1.0,2.0} + Spearman(α, Δ) per emotion
  offtarget.json   -> off-target continuation Δ under C_e enhancement
  targeted_ce.json -> |Δ_{C_e on e}| - |Δ_{C_{e'} on e}| per emotion pair
  random_null.json -> null distribution of Δ over 100 random-set draws
  scenario_stability.json -> Jaccard(S1, S2) vs perm-null per emotion
  cross_emotion.json -> pairwise Jaccard(e, e') for heads vs neurons (Claim 2e)
  metrics.json     -> summary + rubric verdict per emotion + overall
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

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
    mean_pairwise_jaccard,
    n_heads_of,
    n_layers_of,
    n_neurons_of,
    per_head_probe_auc,
    per_neuron_scores,
    permutation_null_jaccard,
    target_prefix_logprob_batch,
    target_prefix_logprob_batch_with_stem_idx,
)


ALPHAS = [0.5, 1.0, 2.0]
N_RANDOM_NULL_DRAWS = 100
N_BOOTSTRAP = 1000


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(diffs[idx].mean())
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _load_C_e_from_run(m1_dir: Path):
    C_e = json.loads((m1_dir / "C_e.json").read_text())
    kstar = json.loads((m1_dir / "kstar.json").read_text())["kstar"]
    dirs = np.load(m1_dir / "directions.npz")
    directions = {e: dirs[f"d_{e}"] for e in EMOTIONS if f"d_{e}" in dirs.files}
    shortlist = json.loads((m1_dir / "shortlist.json").read_text())
    return C_e, kstar, directions, shortlist


def _build_component_payloads(C_e, emotion, directions, model, sign_std_cache):
    """
    Return (heads_payload, neurons_payload) for the C_e set of `emotion`.
    heads_payload: [(l, h, d_e[l])]
    neurons_payload: [(l, n, sign, std)]
    """
    d_e = directions[emotion]
    heads_payload = [(int(l), int(h), d_e[int(l)]) for (l, h) in C_e[emotion]["heads"]]

    neurons_payload = []
    for (l, n) in C_e[emotion]["neurons"]:
        l, n = int(l), int(n)
        key = (emotion, l, n)
        if key in sign_std_cache:
            sign_val, std_val = sign_std_cache[key]
        else:
            with torch.no_grad():
                w = model.model.layers[l].mlp.down_proj.weight[:, n].float().cpu().numpy()
            sign_val = float(np.sign(w.dot(d_e[l])))
            if sign_val == 0.0:
                sign_val = 1.0
            # std comes from precomputed pos-e activations — we approximate at 1.0 if not cached
            std_val = 1.0
            sign_std_cache[key] = (sign_val, std_val)
        neurons_payload.append((l, n, sign_val, std_val))
    return heads_payload, neurons_payload


def _install_mean_substitute_ablation_hooks_per_stem(
    model, C_e_emotion, mean_head_by_layer_stem, mean_neuron_by_layer_stem,
):
    """
    Per-stem mean-substitute ablation (FIXED per plan spec, replacing prior global-mean version).

    For each stem s in the eval fold, and for each component (l, h) or (l, n) in C_e:
      - `mean_head_by_layer_stem[(l, h)]` is a torch tensor of shape (n_stems, HD)
      - `mean_neuron_by_layer_stem[(l, n)]` is a torch tensor of shape (n_stems,)
    At intervention time, the hook uses `_HOOK_STEM_IDX[i]` (published per-batch by
    `target_prefix_logprob_batch_with_stem_idx`) to pick the correct per-stem
    substitute value for batch row i.

    The substitute values are computed by the caller as the per-stem mean over the
    OTHER 5 emotion variants of the SAME stem (which is what the plan specified).

    Falls back to global mean (per component) when _HOOK_STEM_IDX is empty (legacy
    call path); this preserves backward compatibility for random-null / other callers
    that may not publish stem indices.
    """
    handles = []
    NH = n_heads_of(model)
    HD = model.config.hidden_size // NH
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Group heads / neurons by layer, and pre-materialize per-stem tensors on device
    heads_by_layer = {}
    for (l, h) in C_e_emotion["heads"]:
        heads_by_layer.setdefault(int(l), []).append(int(h))
    neurons_by_layer = {}
    for (l, n) in C_e_emotion["neurons"]:
        neurons_by_layer.setdefault(int(l), []).append(int(n))

    # Pre-move to device (constant across all forward passes).
    head_tensors = {}   # (l, h) -> torch (n_stems, HD)
    neuron_tensors = {} # (l, n) -> torch (n_stems,)
    head_global = {}    # (l, h) -> torch (HD,)     fallback
    neuron_global = {}  # (l, n) -> float           fallback
    for (l, h), arr in mean_head_by_layer_stem.items():
        t = torch.tensor(arr, dtype=dtype, device=device)   # (n_stems, HD)
        head_tensors[(int(l), int(h))] = t
        head_global[(int(l), int(h))] = t.mean(dim=0)
    for (l, n), arr in mean_neuron_by_layer_stem.items():
        t = torch.tensor(arr, dtype=dtype, device=device)   # (n_stems,)
        neuron_tensors[(int(l), int(n))] = t
        neuron_global[(int(l), int(n))] = float(t.mean().item())

    from m1_location import _HOOK_EVENT_LENS, _HOOK_STEM_IDX

    def make_pre_hook(l_idx, hlist):
        def pre_hook(module, args):
            x = args[0]  # (B, T, hidden)
            B, T, Htot = x.shape
            xr = x.view(B, T, NH, HD).clone()
            have_stems = len(_HOOK_STEM_IDX) == B
            if T == 1:
                for h in hlist:
                    key = (l_idx, h)
                    if key not in head_tensors:
                        continue
                    if have_stems:
                        for i in range(B):
                            s = int(_HOOK_STEM_IDX[i])
                            xr[i, 0, h] = head_tensors[key][s]
                    else:
                        xr[:, 0, h] = head_global[key]
            elif len(_HOOK_EVENT_LENS) == B:
                for i in range(B):
                    pos_start = max(0, int(_HOOK_EVENT_LENS[i]) - 1)
                    if pos_start < T:
                        for h in hlist:
                            key = (l_idx, h)
                            if key not in head_tensors:
                                continue
                            if have_stems:
                                s = int(_HOOK_STEM_IDX[i])
                                xr[i, pos_start:, h] = head_tensors[key][s]
                            else:
                                xr[i, pos_start:, h] = head_global[key]
            else:
                for h in hlist:
                    key = (l_idx, h)
                    if key not in head_tensors:
                        continue
                    xr[:, -1, h] = head_global[key]
            return (xr.view(B, T, Htot),) + args[1:]
        return pre_hook

    def make_gate_hook_pos(l_idx, nlist):
        def hook(module, inputs, output):
            B, T, _ = output.shape
            output = output.clone()
            have_stems = len(_HOOK_STEM_IDX) == B
            if T == 1:
                for n in nlist:
                    key = (l_idx, n)
                    if key not in neuron_tensors:
                        continue
                    if have_stems:
                        for i in range(B):
                            s = int(_HOOK_STEM_IDX[i])
                            output[i, 0, n] = neuron_tensors[key][s]
                    else:
                        output[:, 0, n] = neuron_global[key]
            elif len(_HOOK_EVENT_LENS) == B:
                for i in range(B):
                    pos_start = max(0, int(_HOOK_EVENT_LENS[i]) - 1)
                    if pos_start < T:
                        for n in nlist:
                            key = (l_idx, n)
                            if key not in neuron_tensors:
                                continue
                            if have_stems:
                                s = int(_HOOK_STEM_IDX[i])
                                output[i, pos_start:, n] = neuron_tensors[key][s]
                            else:
                                output[i, pos_start:, n] = neuron_global[key]
            else:
                for n in nlist:
                    key = (l_idx, n)
                    if key not in neuron_tensors:
                        continue
                    output[:, -1, n] = neuron_global[key]
            return output
        return hook

    for l, hlist in heads_by_layer.items():
        h = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_pre_hook(l, hlist))
        handles.append(h)
    for l, nlist in neurons_by_layer.items():
        h = model.model.layers[l].mlp.gate_proj.register_forward_hook(make_gate_hook_pos(l, nlist))
        handles.append(h)

    return handles


# Backward-compat alias (kept in case any other module imports the old name;
# NOT used by the fixed main() below).
_install_mean_substitute_ablation_hooks = _install_mean_substitute_ablation_hooks_per_stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1_dir", default=str(REPO_DIR / "runs" / "A1_location"))
    ap.add_argument("--out_dir", default=str(REPO_DIR / "runs" / "A2_causal"))
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--model_path", default=LLAMA_PATH)
    ap.add_argument("--skip_stability", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m1_dir = Path(args.m1_dir)

    log("M2: loading M1 artifacts")
    C_e, kstar, directions, shortlist = _load_C_e_from_run(m1_dir)
    log(f"M2: (k_h*, k_n*) = ({kstar['k_h']}, {kstar['k_n']}); |emotions| in C_e = {len(C_e)}")

    log("M2: loading model")
    model, tok = load_model(args.model_path)
    L = n_layers_of(model)
    NH = n_heads_of(model)
    HD = hidden_dim_of(model) // NH

    log("M2: loading SEV + split, using eval fold")
    sev = load_sev()
    split = build_scenario_split(sev, seed=args.seed)
    eval_items = split["eval"]
    train_items = split["train"]

    if args.sanity:
        emotions = EMOTIONS[:2]
        eval_items = eval_items[:20]
    else:
        emotions = list(C_e.keys())

    eval_events = [r["event"] for r in eval_items]
    log(f"M2: eval events = {len(eval_events)}, emotions = {emotions}")

    # For ablation, need mean-substitute (mean activation over OTHER 5 emotions of the same stem).
    # Collect emotion-contextualized head+neuron activations on eval events.
    log("M2: collecting head+neuron activations on eval events per emotion for mean-substitute")
    eval_head_by_emo = {}  # (n_stems, L, NH, HD)
    eval_neuron_by_emo = {}
    for e in EMOTIONS:
        prefix = target_prefix(e).strip()
        ctx = [ev + " " + prefix for ev in eval_events]
        t0 = time.time()
        eval_head_by_emo[e], eval_neuron_by_emo[e] = collect_head_and_neuron_activations(model, tok, ctx, batch_size=4)
        log(f"  emo={e} eval activations ({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------
    # Baselines: log P(prefix_e | event_stem) on eval, per emotion (no intervention)
    # -----------------------------------------------------------------
    log("M2: computing baseline target-prefix logprobs on eval per emotion")
    baseline_lp = {}
    for e in EMOTIONS:
        prefixes = [target_prefix(e)] * len(eval_events)
        baseline_lp[e] = target_prefix_logprob_batch(model, tok, eval_events, prefixes, batch_size=8)

    sign_std_cache = {}

    # -----------------------------------------------------------------
    # Enhancement: apply C_e at α ∈ {0.5, 1.0, 2.0} — target Δ + off-target Δ
    # -----------------------------------------------------------------
    log(f"M2: enhancement at alphas={ALPHAS}")
    enh_results = {}  # emotion -> {alpha -> {"target_delta_mean", "target_delta_ci", "offtarget_delta_mean"}}
    enh_per_row = {}  # emotion -> {alpha -> per-row deltas} for bootstrap
    for e in emotions:
        heads_p, neurons_p = _build_component_payloads(C_e, e, directions, model, sign_std_cache)
        # For neurons, need std_val from pos-e eval activations
        for i, (l, n, sign_v, _) in enumerate(neurons_p):
            std_val = float(eval_neuron_by_emo[e][:, l, n].std() + 1e-6)
            neurons_p[i] = (l, n, sign_v, std_val)

        enh_results[e] = {}
        enh_per_row[e] = {}
        for a in ALPHAS:
            handles = _install_multi_component_hooks(model, heads=heads_p, neurons=neurons_p, alpha=a)
            try:
                # target Δ
                prefixes = [target_prefix(e)] * len(eval_events)
                lp_enh_target = target_prefix_logprob_batch(model, tok, eval_events, prefixes, batch_size=8)
                delta_target = lp_enh_target - baseline_lp[e]
                # off-target Δ (mean over the 5 other prefixes)
                off_deltas = []
                for e_off in EMOTIONS:
                    if e_off == e:
                        continue
                    prefixes_off = [target_prefix(e_off)] * len(eval_events)
                    lp_enh_off = target_prefix_logprob_batch(model, tok, eval_events, prefixes_off, batch_size=8)
                    off_deltas.append(lp_enh_off - baseline_lp[e_off])
                off_delta_mean_row = np.abs(np.array(off_deltas)).mean(axis=0)  # per-row mean |Δ| across off-targets
            finally:
                _remove_hooks(handles)

            ci = paired_bootstrap_ci(delta_target, n_boot=N_BOOTSTRAP, seed=args.seed + int(a * 1000))
            enh_results[e][str(a)] = {
                "target_delta_mean": float(delta_target.mean()),
                "target_delta_ci": [ci[0], ci[1]],
                "offtarget_abs_delta_mean": float(off_delta_mean_row.mean()),
                "offtarget_vs_target_ratio": float(off_delta_mean_row.mean() / max(abs(delta_target.mean()), 1e-9)),
            }
            enh_per_row[e][str(a)] = {"target_delta": delta_target.tolist(), "offtarget_abs_delta": off_delta_mean_row.tolist()}
            log(f"  emo={e} alpha={a}: Δ_target={delta_target.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}], |Δ_offtarget|={off_delta_mean_row.mean():.4f}")

        # Spearman(α, Δ_enhance) over 3 α — use pairs (α, mean Δ)
        alphas_arr = np.array(ALPHAS)
        deltas_arr = np.array([enh_results[e][str(a)]["target_delta_mean"] for a in ALPHAS])
        rho, _ = stats.spearmanr(alphas_arr, deltas_arr)
        enh_results[e]["spearman"] = float(rho)

    save_json(out_dir / "enhancement.json", enh_results)
    save_json(out_dir / "enhancement_per_row.json", enh_per_row)

    # -----------------------------------------------------------------
    # Ablation: PER-STEM mean-substitute from same-stem off-target emotions
    # (FIXED — was previously global mean across all stems, which the verify audit
    # flagged as scope-mismatch that could explain wrong-sign ablation Δ.)
    # -----------------------------------------------------------------
    log("M2: ablation (PER-STEM mean-substitute over 5 other emotions of same stem)")
    ablation_results = {}
    stem_idx_list = list(range(len(eval_events)))  # 1:1 mapping row -> stem index
    for e in emotions:
        others = [e_off for e_off in EMOTIONS if e_off != e]
        # Per-stem mean activation over 5 other emotions: (n_stems, L, NH, HD) and (n_stems, L, NN)
        mean_head_by_stem = np.stack([eval_head_by_emo[o] for o in others], axis=0).mean(axis=0)
        mean_neuron_by_stem = np.stack([eval_neuron_by_emo[o] for o in others], axis=0).mean(axis=0)

        # Build per-(l, h) tensor of shape (n_stems, HD) for heads,
        # and per-(l, n) tensor of shape (n_stems,) for neurons.
        mean_head_stem = {
            (int(l), int(h)): mean_head_by_stem[:, l, h, :]  # (n_stems, HD)
            for (l, h) in C_e[e]["heads"]
        }
        mean_neuron_stem = {
            (int(l), int(n)): mean_neuron_by_stem[:, l, n]  # (n_stems,)
            for (l, n) in C_e[e]["neurons"]
        }

        handles = _install_mean_substitute_ablation_hooks_per_stem(
            model, C_e[e], mean_head_stem, mean_neuron_stem
        )
        try:
            prefixes = [target_prefix(e)] * len(eval_events)
            lp_abl = target_prefix_logprob_batch_with_stem_idx(
                model, tok, eval_events, prefixes, stem_idx_list, batch_size=8
            )
            delta_abl = lp_abl - baseline_lp[e]
        finally:
            _remove_hooks(handles)
        ci = paired_bootstrap_ci(delta_abl, n_boot=N_BOOTSTRAP, seed=args.seed + hash(e) % 100000)
        ablation_results[e] = {
            "target_delta_mean": float(delta_abl.mean()),
            "target_delta_ci": [ci[0], ci[1]],
            "n": len(eval_events),
            "operator": "per_stem_mean_substitute_over_5_other_emotions",
        }
        log(f"  emo={e} ablation (per-stem): Δ_target={delta_abl.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    save_json(out_dir / "ablation.json", ablation_results)

    # -----------------------------------------------------------------
    # Random-set null (100 draws per emotion) at α2
    # -----------------------------------------------------------------
    log("M2: random-set null (100 draws per emotion) at α=1.0")
    random_null = {}
    rng = np.random.default_rng(args.seed + 12345)
    for e in emotions:
        pool_h = shortlist[e]["head_pool_sample_first10"]  # sample used to reconstruct — actually load full shortlist
    # Load full shortlist
    shortlist_full = json.loads((m1_dir / "shortlist.json").read_text())
    # We need the full pools; if shortlist.json truncated to sample_first10, reconstruct from precomputed directions.npz.
    # Simpler: rebuild the shortlist deterministically here from the Stage-A scores stored in directions.npz.
    dirs_all = np.load(m1_dir / "directions.npz")
    head_pool_full = {}
    neuron_pool_full = {}
    # FIX: For the random-null, use a WIDER pool spanning more layers so k-of-pool draws
    # have non-degenerate variance. The previous configuration
    # (n_top_h = round(0.20 * NH * 3) = 14 heads, then choose k_h=24 without replacement)
    # was mathematically forced to fail (k > pool), and even when clipped to pool size
    # produced identical draws => std=0. The fix uses more layers and a larger fraction.
    RN_TOP_LAYERS = min(2 * TOP_N_LAYERS, dirs_all[f"head_auc_{emotions[0]}"].shape[0])
    RN_HEAD_FRAC = max(TOP_HEAD_FRAC, 0.5)      # so pool_h >> k_h
    RN_NEURON_FRAC = max(TOP_NEURON_FRAC, 0.10) # so pool_n >> k_n
    for e in emotions:
        haucs = dirs_all[f"head_auc_{e}"]
        nscores = dirs_all[f"neuron_score_{e}"]
        per_layer_sum = haucs.sum(axis=1) + nscores.sum(axis=1)
        top_layers = sorted(np.argsort(-per_layer_sum)[:RN_TOP_LAYERS].tolist())
        n_top_h = max(1, int(round(RN_HEAD_FRAC * NH * RN_TOP_LAYERS)))
        flat_h = [(l, h, float(haucs[l, h])) for l in top_layers for h in range(NH)]
        flat_h.sort(key=lambda x: -x[2])
        head_pool_full[e] = [(l, h) for (l, h, _) in flat_h[:n_top_h]]

        NN = nscores.shape[1]
        n_top_n = max(1, int(round(RN_NEURON_FRAC * NN * RN_TOP_LAYERS)))
        flat_n = [(l, n, float(nscores[l, n])) for l in top_layers for n in range(NN)]
        flat_n.sort(key=lambda x: -x[2])
        neuron_pool_full[e] = [(l, n) for (l, n, _) in flat_n[:n_top_n]]
    log(f"M2: random-null pool sizes: heads={[len(head_pool_full[e]) for e in emotions]}, neurons={[len(neuron_pool_full[e]) for e in emotions]} (RN_TOP_LAYERS={RN_TOP_LAYERS}, RN_HEAD_FRAC={RN_HEAD_FRAC}, RN_NEURON_FRAC={RN_NEURON_FRAC})")

    for e in emotions:
        deltas = []
        pool_h = head_pool_full[e]
        pool_n = neuron_pool_full[e]
        k_h = min(kstar["k_h"], len(pool_h))
        k_n = min(kstar["k_n"], len(pool_n))
        d_e = directions[e]
        for draw in range(N_RANDOM_NULL_DRAWS):
            hs_idx = rng.choice(len(pool_h), size=k_h, replace=False)
            ns_idx = rng.choice(len(pool_n), size=k_n, replace=False)
            heads_p = [(pool_h[i][0], pool_h[i][1], d_e[pool_h[i][0]]) for i in hs_idx]
            neurons_p = []
            for i in ns_idx:
                l, n = pool_n[i]
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
            handles = _install_multi_component_hooks(model, heads=heads_p, neurons=neurons_p, alpha=1.0)
            try:
                prefixes = [target_prefix(e)] * len(eval_events)
                lp = target_prefix_logprob_batch(model, tok, eval_events, prefixes, batch_size=8)
                deltas.append(float((lp - baseline_lp[e]).mean()))
            finally:
                _remove_hooks(handles)
            if (draw + 1) % 25 == 0:
                log(f"  emo={e} random-null draw {draw+1}/{N_RANDOM_NULL_DRAWS}")
        deltas = np.array(deltas)
        c_e_target = enh_results[e]["1.0"]["target_delta_mean"]
        z_score = (c_e_target - deltas.mean()) / (deltas.std() + 1e-8)
        random_null[e] = {
            "null_delta_mean": float(deltas.mean()),
            "null_delta_std": float(deltas.std()),
            "null_delta_ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "c_e_delta_mean": float(c_e_target),
            "z_score_vs_null": float(z_score),
            "c_e_above_null_hi": bool(c_e_target > np.percentile(deltas, 97.5)),
        }
        log(f"  emo={e} random-null: C_e Δ={c_e_target:+.4f}, null mean={deltas.mean():+.4f}, z={z_score:.2f}")

    save_json(out_dir / "random_null.json", random_null)

    # -----------------------------------------------------------------
    # Targeted-C_{e'} control
    # -----------------------------------------------------------------
    log("M2: targeted-C_{e'} control")
    targeted = {}
    for e in emotions:
        d_e = directions[e]
        c_e_delta = np.array(enh_per_row[e]["1.0"]["target_delta"])  # already computed above
        row = {}
        for e_prime in emotions:
            if e_prime == e:
                continue
            heads_p_ep, neurons_p_ep = _build_component_payloads(C_e, e_prime, directions, model, sign_std_cache)
            for i, (l, n, sign_v, _) in enumerate(neurons_p_ep):
                std_val = float(eval_neuron_by_emo[e_prime][:, l, n].std() + 1e-6)
                neurons_p_ep[i] = (l, n, sign_v, std_val)
            handles = _install_multi_component_hooks(model, heads=heads_p_ep, neurons=neurons_p_ep, alpha=1.0)
            try:
                prefixes = [target_prefix(e)] * len(eval_events)  # score TARGET e
                lp = target_prefix_logprob_batch(model, tok, eval_events, prefixes, batch_size=8)
                ep_delta = lp - baseline_lp[e]
            finally:
                _remove_hooks(handles)

            # |Δ_{C_e on e}| vs. |Δ_{C_{e'} on e}|
            diff = np.abs(c_e_delta) - np.abs(ep_delta)
            ci = paired_bootstrap_ci(diff, n_boot=N_BOOTSTRAP, seed=args.seed + hash((e, e_prime)) % 100000)
            row[e_prime] = {
                "c_e_abs_mean": float(np.abs(c_e_delta).mean()),
                "c_ep_abs_mean": float(np.abs(ep_delta).mean()),
                "diff_mean": float(diff.mean()),
                "diff_ci": [ci[0], ci[1]],
                "pass_at_95": bool(ci[0] > 0),
            }
        # Predicate (c) passes if for majority of e' the CI excludes 0
        n_pass = sum(1 for v in row.values() if v["pass_at_95"])
        targeted[e] = {"per_ep": row, "n_pass": n_pass, "n_total": len(row), "pass_majority": n_pass > len(row) / 2}
        log(f"  emo={e} targeted: {n_pass}/{len(row)} e' pass at 95% CI")
    save_json(out_dir / "targeted_ce.json", targeted)

    # -----------------------------------------------------------------
    # Scenario stability: Jaccard(C_e^{S1}, C_e^{S2})
    # -----------------------------------------------------------------
    if args.skip_stability:
        log("M2: scenario stability skipped (--skip_stability)")
        stability = {"skipped": True}
    else:
        log("M2: scenario stability (S1 vs S2 = 5/5 of train scenarios per domain)")
        # For each domain, partition its 10 train scenarios into 5+5.
        # Re-collect emotion-context activations on S1 and S2 stems, re-rank Stage A signals,
        # take top-k_h*, k_n* neurons/heads, compute Jaccard.
        train_scenarios_by_domain = {}
        for row in train_items:
            train_scenarios_by_domain.setdefault(row["domain"], set()).add(row["scenario"])
        rng2 = np.random.default_rng(args.seed + 777)
        S1_stems, S2_stems = [], []
        for domain, scs in train_scenarios_by_domain.items():
            scs = sorted(scs)
            rng2.shuffle(scs)
            S1_set = set(scs[:5])
            S2_set = set(scs[5:])
            for row in train_items:
                if row["domain"] == domain:
                    if row["scenario"] in S1_set:
                        S1_stems.append(row)
                    elif row["scenario"] in S2_set:
                        S2_stems.append(row)

        log(f"M2: S1 stems = {len(S1_stems)}, S2 stems = {len(S2_stems)}")

        def build_C_e_on(stems):
            events = [r["event"] for r in stems]
            head_activations = {}
            neuron_activations = {}
            for e in emotions:
                prefix = target_prefix(e).strip()
                ctx = [ev + " " + prefix for ev in events]
                ha, na = collect_head_and_neuron_activations(model, tok, ctx, batch_size=4)
                head_activations[e] = ha
                neuron_activations[e] = na
            C_e_sub = {}
            for e in emotions:
                pos = head_activations[e]
                neg = np.concatenate([head_activations[o] for o in emotions if o != e], axis=0)
                haucs = per_head_probe_auc(pos, neg)
                pos_n = neuron_activations[e]
                neg_n = np.concatenate([neuron_activations[o] for o in emotions if o != e], axis=0)
                # Residual dir approx: we need a full residual too. Use dirs_all d_e as proxy.
                d_e = directions[e]
                nscores = per_neuron_scores(pos_n, neg_n, d_e, model)
                per_layer_sum = haucs.sum(axis=1) + nscores.sum(axis=1)
                top_layers = sorted(np.argsort(-per_layer_sum)[:TOP_N_LAYERS].tolist())
                n_top_h = max(1, int(round(TOP_HEAD_FRAC * NH * TOP_N_LAYERS)))
                flat_h = [(l, h, float(haucs[l, h])) for l in top_layers for h in range(NH)]
                flat_h.sort(key=lambda x: -x[2])
                heads_set = set(f"{l}_{h}" for (l, h, _) in flat_h[:min(kstar["k_h"], len(flat_h))])
                NN_ = nscores.shape[1]
                n_top_n = max(1, int(round(TOP_NEURON_FRAC * NN_ * TOP_N_LAYERS)))
                flat_n = [(l, n, float(nscores[l, n])) for l in top_layers for n in range(NN_)]
                flat_n.sort(key=lambda x: -x[2])
                neurons_set = set(f"{l}_{n}" for (l, n, _) in flat_n[:min(kstar["k_n"], len(flat_n))])
                C_e_sub[e] = {"heads": heads_set, "neurons": neurons_set}
            return C_e_sub

        log("M2: fitting C_e^{S1}")
        C_S1 = build_C_e_on(S1_stems)
        log("M2: fitting C_e^{S2}")
        C_S2 = build_C_e_on(S2_stems)

        stability = {}
        for e in emotions:
            hj = mean_pairwise_jaccard([C_S1[e]["heads"], C_S2[e]["heads"]])
            nj = mean_pairwise_jaccard([C_S1[e]["neurons"], C_S2[e]["neurons"]])
            pool_h = TOP_N_LAYERS * NH
            NN_ = int(np.load(m1_dir / "directions.npz")[f"neuron_score_{e}"].shape[1])
            pool_n = TOP_N_LAYERS * NN_
            hm, hlo, hhi = permutation_null_jaccard(pool_h, [min(kstar["k_h"], pool_h)] * 2, n_draws=200, seed=args.seed + hash(e) % 100000)
            nm, nlo, nhi = permutation_null_jaccard(pool_n, [min(kstar["k_n"], pool_n)] * 2, n_draws=200, seed=args.seed + hash(e) % 100000 + 3)
            stability[e] = {
                "head_jaccard": hj, "head_null_mean": hm, "head_null_ci": [hlo, hhi], "head_pass": bool(hj > hhi),
                "neuron_jaccard": nj, "neuron_null_mean": nm, "neuron_null_ci": [nlo, nhi], "neuron_pass": bool(nj > nhi),
            }
        save_json(out_dir / "scenario_stability.json", stability)
        log("M2: scenario stability done")

    # -----------------------------------------------------------------
    # Cross-emotion structure (Claim 2e secondary)
    # -----------------------------------------------------------------
    log("M2: cross-emotion overlap (Claim 2e)")
    head_sets = {e: set(f"{l}_{h}" for l, h in C_e[e]["heads"]) for e in emotions}
    neuron_sets = {e: set(f"{l}_{n}" for l, n in C_e[e]["neurons"]) for e in emotions}
    head_jacs, neuron_jacs = [], []
    for i in range(len(emotions)):
        for j in range(i + 1, len(emotions)):
            e1, e2 = emotions[i], emotions[j]
            inter = len(head_sets[e1] & head_sets[e2])
            union = len(head_sets[e1] | head_sets[e2])
            head_jacs.append(inter / max(union, 1))
            interN = len(neuron_sets[e1] & neuron_sets[e2])
            unionN = len(neuron_sets[e1] | neuron_sets[e2])
            neuron_jacs.append(interN / max(unionN, 1))
    save_json(out_dir / "cross_emotion.json", {
        "mean_head_jaccard": float(np.mean(head_jacs)),
        "mean_neuron_jaccard": float(np.mean(neuron_jacs)),
        "diff_head_minus_neuron": float(np.mean(head_jacs) - np.mean(neuron_jacs)),
    })

    # -----------------------------------------------------------------
    # Rubric verdict
    # -----------------------------------------------------------------
    per_emotion_verdict = {}
    for e in emotions:
        a2_target = enh_results[e]["1.0"]["target_delta_mean"]
        abl_target = ablation_results[e]["target_delta_mean"]
        spearman = enh_results[e]["spearman"]
        # (a) causal-sign + dose
        a_pass = (a2_target > 0) and (abl_target < 0) and (spearman >= 0.7)
        # (b) raw off-target
        off_abs = enh_results[e]["1.0"]["offtarget_abs_delta_mean"]
        b_pass = off_abs < abs(a2_target)
        # (c) targeted-C_{e'} — majority passes
        c_pass = targeted[e]["pass_majority"]
        # (d) scenario stability
        d_pass = False
        if not args.skip_stability:
            d_pass = bool(stability[e].get("head_pass", False) or stability[e].get("neuron_pass", False))

        if a_pass and b_pass and c_pass and d_pass:
            v = "full"
        elif a_pass and (b_pass or c_pass) and not d_pass:
            v = "partial"
        elif a_pass and not (b_pass or c_pass):
            v = "causal-only"
        else:
            v = "not-supported"
        per_emotion_verdict[e] = {
            "verdict": v,
            "a_pass": a_pass, "b_pass": b_pass, "c_pass": c_pass, "d_pass": d_pass,
            "alpha2_target_delta": a2_target, "ablation_target_delta": abl_target,
            "spearman": spearman, "offtarget_abs": off_abs,
        }

    # Aggregate — majority rubric across emotions
    verdict_counts = {"full": 0, "partial": 0, "causal-only": 0, "not-supported": 0}
    for e, v in per_emotion_verdict.items():
        verdict_counts[v["verdict"]] += 1
    overall = max(verdict_counts, key=lambda k: verdict_counts[k])
    save_json(out_dir / "metrics.json", {
        "per_emotion_verdict": per_emotion_verdict,
        "verdict_counts": verdict_counts,
        "overall_verdict": overall,
        "n_emotions_with_a_pass": sum(1 for v in per_emotion_verdict.values() if v["a_pass"]),
        "claim_2_summary": f"a_pass on {sum(1 for v in per_emotion_verdict.values() if v['a_pass'])}/{len(per_emotion_verdict)} emotions",
    })
    log(f"M2: DONE. Overall verdict = {overall}. Per-emotion: {verdict_counts}")


if __name__ == "__main__":
    main()
