#!/usr/bin/env python3
"""
Verify C3 — model-swap variant: Meta-Llama-3-8B-Instruct.

Runs M2 (extract raw directions + probe) → supplementary M4 (steer at ell_V* and L=16)
on the swap model with the same DG-1000 dataset and CAA method.

This is the model-swap robustness variant for verify stage. The DG-1000 prompts,
the CAA activation-addition method, and the α-grid are identical to the main experiment's
supplementary L=16 run. Only the model is changed.

Outputs under --out_dir:
  artifacts/m2/probe_accuracy.json      -- per-V per-layer probe cv_acc + held_acc
  artifacts/m2/layer_pick.json          -- ell_V* from argmax probe cv_acc
  artifacts/m2/baseline_transfer.json   -- tau baseline stats at alpha=0
  artifacts/steer/{V}_L{L}_a{a}.json   -- per grid-point steering results
  artifacts/steer/summary.json          -- all steering results
  cost.json                             -- gpu_ids + timing for GPU pin witness check
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

ANSWER_INT_RE = re.compile(r"(-?\d{1,3})")

POS_SIDE_MAP = {"G": "male", "A": "young", "I": "take-frame", "M": "no-meet"}


def parse_transfer(text):
    if text is None:
        return None
    first_line = text.strip().split("\n", 1)[0]
    m = ANSWER_INT_RE.search(first_line)
    if m:
        try:
            v = int(m.group(1))
            if 0 <= v <= 20:
                return v
        except ValueError:
            pass
    head = text.strip()[:60]
    m = ANSWER_INT_RE.search(head)
    if m:
        try:
            v = int(m.group(1))
            if 0 <= v <= 20:
                return v
        except ValueError:
            pass
    return None


def format_prompt(user):
    return LLAMA3_TEMPLATE.format(user=user)


def load_model_tok(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tok, model


@torch.no_grad()
def collect_acts(rows, tokenizer, model, batch_size, layer_ids):
    """Collect residual-stream activations at every layer in layer_ids for each row.
    Returns acts: [N, len(layer_ids), H] and meta: list of row dicts.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPast
    all_acts = []
    meta = []

    # Hook to collect hidden states
    collected = {}

    def make_hook(li):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # Take the last token position
            collected[li] = h[:, -1, :].float().cpu()
        return hook

    handles = []
    for li in layer_ids:
        # Register on model.model.layers[li]
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        handles.append(h)

    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start: start + batch_size]
            texts = [format_prompt(r["prompt"]) for r in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=1024).to(model.device)
            collected.clear()
            _ = model(**enc, output_hidden_states=False)
            # Stack: [B, n_layers, H]
            stack = torch.stack([collected[li] for li in layer_ids], dim=1)  # [B, L, H]
            all_acts.append(stack)
            meta.extend(batch)
    finally:
        for h in handles:
            h.remove()

    acts = torch.cat(all_acts, dim=0)  # [N, n_layers, H]
    return acts, meta


def fit_probe(X_train, y_train, X_held, y_held):
    """5-fold logistic probe on X_train, y_train; report held accuracy."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    for tr, te in skf.split(X_train, y_train):
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_train[tr], y_train[tr])
        cv_accs.append(clf.score(X_train[te], y_train[te]))
    clf_final = LogisticRegression(max_iter=500, C=1.0)
    clf_final.fit(X_train, y_train)
    held_acc = clf_final.score(X_held, y_held)
    return float(np.mean(cv_accs)), float(held_acc)


@torch.no_grad()
def decode_batch(rows, tokenizer, model, batch_size, max_new_tokens=8):
    outs = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start: start + batch_size]
        texts = [format_prompt(r["prompt"]) for r in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024).to(model.device)
        in_len = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        for row in gen[:, in_len:]:
            outs.append(tokenizer.decode(row, skip_special_tokens=True))
    return outs


class SteeringHook:
    def __init__(self, unit_dir, alpha):
        self.unit_dir = unit_dir
        self.alpha = alpha

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            hidden = out[0]
            hidden = hidden + self.alpha * self.unit_dir.to(hidden.dtype).to(hidden.device)
            return (hidden,) + out[1:]
        else:
            return out + self.alpha * self.unit_dir.to(out.dtype).to(out.device)


def compute_sigma_proj(acts, layer_idx, unit_dir):
    h = acts[:, layer_idx, :].float()
    proj = h @ unit_dir
    return float(proj.std().item())


def run_steer_point(V, L, am, sigma, unit, model, tokenizer,
                    held_base_rows, batch_size, max_new_tokens, out_dir):
    """Run one (V, L, alpha_mult) grid point and return result dict."""
    fname = os.path.join(out_dir, f"{V}_L{L}_a{am:+d}.json")
    alpha = am * sigma
    layer_hook = L - 1 if L > 0 else 0
    handles = []
    if am != 0:
        hook = SteeringHook(unit, alpha)
        h = model.model.layers[layer_hook].register_forward_hook(hook)
        handles.append(h)
    try:
        t0 = time.time()
        gens = decode_batch(held_base_rows, tokenizer, model, batch_size, max_new_tokens)
        taus = [parse_transfer(g) for g in gens]
        valid = [t for t in taus if t is not None]
        parse_fail = sum(1 for t in taus if t is None) / max(1, len(taus))
        mean_tau = float(np.mean(valid)) if valid else float("nan")

        # V-effect
        pos_val = POS_SIDE_MAP[V]
        pos_t = [t for r, t in zip(held_base_rows, taus) if t is not None and r[V] == pos_val]
        neg_t = [t for r, t in zip(held_base_rows, taus) if t is not None and r[V] != pos_val]
        v_eff = float(np.mean(pos_t) - np.mean(neg_t)) if pos_t and neg_t else 0.0

        # W-effects
        w_effects = {}
        for W in ["G", "A", "I", "M"]:
            pv = POS_SIDE_MAP[W]
            pw = [t for r, t in zip(held_base_rows, taus) if t is not None and r[W] == pv]
            nw = [t for r, t in zip(held_base_rows, taus) if t is not None and r[W] != pv]
            w_effects[W] = float(np.mean(pw) - np.mean(nw)) if pw and nw else 0.0

        # Coherence check
        n_coh = min(10, len(gens))
        coh_toks = [parse_transfer(g) for g in gens[:n_coh]]
        format_ok = sum(1 for t in coh_toks if t is not None) / max(1, n_coh)
        reps = []
        for g in gens[:n_coh]:
            toks = g.split()
            if len(toks) < 5:
                reps.append(0.0)
            else:
                grams = [tuple(toks[i:i+5]) for i in range(len(toks) - 4)]
                reps.append(1.0 - len(set(grams)) / len(grams) if grams else 0.0)

        res = dict(
            V=V, layer_abs=L, alpha_mult=am, alpha=alpha, sigma_proj=sigma,
            n_baseline=len(held_base_rows), parse_failure_rate=float(parse_fail),
            mean_transfer=mean_tau,
            v_effect=v_eff, w_effects=w_effects,
            coherence=dict(format_ok_rate=float(format_ok),
                           mean_5gram_rep=float(np.mean(reps)) if reps else 0.0,
                           n_samples=n_coh),
            sample_generations=gens[:3],
            wall_time_s=time.time() - t0,
        )
        with open(fname, "w") as f:
            json.dump(res, f, indent=2)
        return res
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to swap model (Meta-Llama-3-8B-Instruct)")
    ap.add_argument("--data", default="data/dg1000_prompts.jsonl")
    ap.add_argument("--out_dir", required=True,
                    help="Output root, e.g. runs/verify_C3_variant_model_swap_v1/")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--every_k_layer", type=int, default=2,
                    help="Sample every k-th layer for activation extraction")
    ap.add_argument("--alpha_grid", default="-2,-1,0,1,2",
                    help="Comma-separated alpha multiplier grid (use = syntax: --alpha_grid=-2,-1,0,1,2)")
    ap.add_argument("--fixed_mid_layer", type=int, default=16,
                    help="Fixed mid-layer to evaluate (in addition to picked ell_V*)")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--n_sample", type=int, default=-1,
                    help="If >0 use only first n held-out trials (smoke test)")
    args = ap.parse_args()

    # Resolve directories
    project_root = Path(__file__).parent.parent
    out_root = project_root / args.out_dir
    out_m2 = out_root / "artifacts" / "m2"
    out_steer = out_root / "artifacts" / "steer"
    out_m2.mkdir(parents=True, exist_ok=True)
    out_steer.mkdir(parents=True, exist_ok=True)

    start_wall = time.time()

    # --- Load data ---
    data_path = project_root / args.data
    with open(data_path) as f:
        rows = [json.loads(l) for l in f]
    train_rows = [r for r in rows if r["split"] == "train" and r["is_paired_partner_of"] is None]
    train_partner_rows = [r for r in rows if r["split"] == "train" and r["is_paired_partner_of"] is not None]
    held_base_rows = [r for r in rows if r["split"] == "held" and r["is_paired_partner_of"] is None]
    if args.n_sample > 0:
        held_base_rows = held_base_rows[:args.n_sample]
    print(f"[verify_c3] train={len(train_rows)} baseline, {len(train_partner_rows)} partners; "
          f"held={len(held_base_rows)} baseline", flush=True)

    # Build per-V paired sets (800 train pairs per V)
    train_pairs_by_V = {}
    for V in ["G", "A", "I", "M"]:
        base_by_id = {r["trial_id"]: r for r in train_rows}
        partners = [r for r in train_partner_rows
                    if r.get("V_flipped") == V or (
                        r.get("is_paired_partner_of") is not None and
                        # detect by single-variable flip: find which V differs
                        len([k for k in ["G", "A", "I", "M"]
                             if base_by_id.get(r["is_paired_partner_of"], {}).get(k) != r.get(k)]) == 1 and
                        [k for k in ["G", "A", "I", "M"]
                         if base_by_id.get(r["is_paired_partner_of"], {}).get(k) != r.get(k)][0] == V
                    )]
        # fallback: take all partner rows and detect V by comparing to parent
        if not partners:
            for pr in train_partner_rows:
                pid = pr.get("is_paired_partner_of")
                if pid and pid in base_by_id:
                    parent = base_by_id[pid]
                    diffs = [k for k in ["G", "A", "I", "M"] if parent.get(k) != pr.get(k)]
                    if len(diffs) == 1 and diffs[0] == V:
                        partners.append(pr)
        train_pairs_by_V[V] = (train_rows, partners)

    # Simpler approach: just identify pairs by comparing baseline and partner
    # Use the is_paired_partner_of field to find parent, compare V field
    train_base_by_id = {r["trial_id"]: r for r in train_rows}
    V_pairs = {V: [] for V in ["G", "A", "I", "M"]}
    for pr in train_partner_rows:
        pid = pr.get("is_paired_partner_of")
        if pid and pid in train_base_by_id:
            parent = train_base_by_id[pid]
            diffs = [k for k in ["G", "A", "I", "M"] if parent.get(k) != pr.get(k)]
            if len(diffs) == 1:
                V_pairs[diffs[0]].append((parent, pr))

    print(f"[verify_c3] V-pair counts: { {V: len(V_pairs[V]) for V in ['G','A','I','M']} }", flush=True)

    # --- Load model ---
    print(f"[verify_c3] Loading model from {args.model}", flush=True)
    tokenizer, model = load_model_tok(args.model)

    # Detect GPU IDs
    gpu_ids = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_ids.append(i)
    # Map to CUDA_VISIBLE_DEVICES actual IDs
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_vis:
        physical_ids = cuda_vis.split(",")
        gpu_ids_reported = [physical_ids[i] for i in gpu_ids if i < len(physical_ids)]
    else:
        gpu_ids_reported = [str(i) for i in gpu_ids]

    n_layers = model.config.num_hidden_layers  # e.g. 32
    hidden_size = model.config.hidden_size

    # Layer IDs to sample: every k-th absolute layer (0-indexed blocks)
    layer_ids = list(range(0, n_layers, args.every_k_layer))
    if n_layers - 1 not in layer_ids:
        layer_ids.append(n_layers - 1)
    print(f"[verify_c3] n_layers={n_layers}, hidden={hidden_size}, "
          f"sampling {len(layer_ids)} layers: {layer_ids[:5]}...{layer_ids[-3:]}", flush=True)

    # --- M2-equivalent: Extract train activations for each V ---
    y_train = {V: np.array([1 if r[V] == POS_SIDE_MAP[V] else 0 for r in train_rows])
               for V in ["G", "A", "I", "M"]}
    y_held = {V: np.array([1 if r[V] == POS_SIDE_MAP[V] else 0 for r in held_base_rows])
              for V in ["G", "A", "I", "M"]}

    # Extract train activations (baseline prompts only — for layer probe)
    print("[verify_c3] Extracting train activations...", flush=True)
    train_acts, train_meta = collect_acts(train_rows, tokenizer, model,
                                          args.batch_size, layer_ids)
    # Extract held activations
    print("[verify_c3] Extracting held activations...", flush=True)
    held_acts, held_meta = collect_acts(held_base_rows, tokenizer, model,
                                        args.batch_size, layer_ids)

    layer_to_idx = {li: i for i, li in enumerate(layer_ids)}

    # Per-V raw directions (v_hat_V^ell = mean[h(p') - h(p)] for V-flip pairs)
    # Using train pairs only
    print("[verify_c3] Computing raw directions...", flush=True)
    directions = {}  # V -> [n_layers, H]
    for V in ["G", "A", "I", "M"]:
        pairs = V_pairs[V]
        if not pairs:
            print(f"  [verify_c3] WARNING: no pairs found for V={V}", flush=True)
            directions[V] = torch.zeros(len(layer_ids), hidden_size)
            continue
        # For each pair we need activations for both parent and partner
        # Collect partner activations
        partner_rows_V = [pr for _, pr in pairs]
        parent_rows_V = [pa for pa, _ in pairs]
        acts_partner, _ = collect_acts(partner_rows_V, tokenizer, model,
                                       args.batch_size, layer_ids)
        acts_parent, _ = collect_acts(parent_rows_V, tokenizer, model,
                                      args.batch_size, layer_ids)
        # v_hat_V^ell = mean(h(partner) - h(parent))
        diff = (acts_partner - acts_parent).float()  # [N_pairs, n_layers, H]
        v_hat = diff.mean(dim=0)  # [n_layers, H]
        directions[V] = v_hat
        print(f"  [verify_c3] V={V}: {len(pairs)} pairs, "
              f"|v_hat| at layer_ids[0]={float(v_hat[0].norm()):.4f}", flush=True)

    # Probe per layer, per V
    print("[verify_c3] Fitting per-layer probes...", flush=True)
    probe_results = {}  # V -> {layer: {cv_acc, held_acc}}
    layer_pick = {}   # V -> {ell_star, cv_acc}
    for V in ["G", "A", "I", "M"]:
        probe_results[V] = {}
        best_cv = -1
        best_layer = layer_ids[0]
        for li_idx, li in enumerate(layer_ids):
            X_tr = train_acts[:, li_idx, :].numpy()
            X_he = held_acts[:, li_idx, :].numpy()
            y_tr = y_train[V]
            y_he = y_held[V]
            try:
                cv_acc, held_acc = fit_probe(X_tr, y_tr, X_he, y_he)
            except Exception as e:
                cv_acc, held_acc = 0.5, 0.5
            probe_results[V][li] = {"cv_acc": cv_acc, "held_acc": held_acc}
            if cv_acc > best_cv:
                best_cv = cv_acc
                best_layer = li
        layer_pick[V] = {"ell_star": best_layer, "cv_acc": best_cv}
        print(f"  [verify_c3] V={V}: ell_V*={best_layer} (cv_acc={best_cv:.3f})", flush=True)

    with open(out_m2 / "probe_accuracy.json", "w") as f:
        json.dump(probe_results, f, indent=2)
    with open(out_m2 / "layer_pick.json", "w") as f:
        json.dump(layer_pick, f, indent=2)

    # Baseline transfer at alpha=0 (no hooks)
    print("[verify_c3] Measuring baseline transfers (alpha=0)...", flush=True)
    gens_base = decode_batch(held_base_rows, tokenizer, model,
                             args.batch_size, args.max_new_tokens)
    taus_base = [parse_transfer(g) for g in gens_base]
    valid_base = [t for t in taus_base if t is not None]
    baseline_info = {}
    for V in ["G", "A", "I", "M"]:
        pv = POS_SIDE_MAP[V]
        pt = [t for r, t in zip(held_base_rows, taus_base) if t is not None and r[V] == pv]
        nt = [t for r, t in zip(held_base_rows, taus_base) if t is not None and r[V] != pv]
        baseline_info[V] = {
            "mean_tau": float(np.mean(valid_base)) if valid_base else None,
            "v_effect": float(np.mean(pt) - np.mean(nt)) if pt and nt else 0.0,
            "n_valid": len(valid_base),
            "parse_fail_rate": sum(1 for t in taus_base if t is None) / max(1, len(taus_base))
        }
        print(f"  [verify_c3] V={V} baseline v_effect={baseline_info[V]['v_effect']:+.3f}", flush=True)
    with open(out_m2 / "baseline_transfer.json", "w") as f:
        json.dump(baseline_info, f, indent=2)

    # --- M4-supp-equivalent: Steer at ell_V* and at fixed_mid_layer ---
    alpha_grid = [int(x) for x in args.alpha_grid.split(",")]
    all_results = []

    for V in ["G", "A", "I", "M"]:
        ell_star = layer_pick[V]["ell_star"]
        layers_to_test = sorted(set([ell_star, args.fixed_mid_layer]))

        for L in layers_to_test:
            if L not in layer_to_idx:
                # If L not in sampled layers, compute direction by interpolation
                # from nearest sampled layer — but safer to use actual extraction
                print(f"  [verify_c3] WARNING: L={L} not in sampled layer_ids for V={V}; skipping", flush=True)
                continue
            li_idx = layer_to_idx[L]
            v = directions[V][li_idx].float()
            if v.norm() < 1e-6:
                print(f"  [verify_c3] V={V} L={L}: null direction, skipping", flush=True)
                continue
            unit = v / v.norm()
            sigma = compute_sigma_proj(held_acts, li_idx, unit)
            print(f"[verify_c3] V={V} L={L} sigma_proj={sigma:.4f} |v|={float(v.norm()):.4f}", flush=True)

            for am in alpha_grid:
                print(f"  [verify_c3] V={V} L={L} alpha={am:+d}", flush=True)
                res = run_steer_point(V, L, am, sigma, unit, model, tokenizer,
                                      held_base_rows, args.batch_size,
                                      args.max_new_tokens, str(out_steer))
                all_results.append(res)
                print(f"    v_effect={res['v_effect']:+.3f} "
                      f"parse_fail={res['parse_failure_rate']:.2f}", flush=True)

    with open(out_steer / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Write cost.json for GPU pin witness check ---
    total_wall = time.time() - start_wall
    cost = {
        "gpu_ids": gpu_ids_reported,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "model": args.model,
        "wall_time_s": total_wall,
        "n_steer_points": len(all_results),
        "n_held_baseline": len(held_base_rows),
    }
    with open(out_root / "cost.json", "w") as f:
        json.dump(cost, f, indent=2)
    print(f"[verify_c3] done — {len(all_results)} steering grid points in {total_wall:.1f}s", flush=True)
    print(f"[verify_c3] gpu_ids: {gpu_ids_reported}", flush=True)


if __name__ == "__main__":
    main()
