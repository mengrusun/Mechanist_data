#!/usr/bin/env python3
"""M2 — Location: extract raw per-variable directions + linear probes (C1).

For each V in {G, A, I, M}:
  1. Forward Llama-3.1-8B-Instruct on the 800 train paired-partner prompts (V-flip
     only) at bf16.  Cache residual-stream activations at every layer at the
     end-of-prompt (last input) token.
  2. Compute raw difference-of-means direction v_hat_V[layer] = mean_pairs h(p')-h(p).
  3. Fit a 5-fold logistic linear probe per (V, layer) on the 800 train activations;
     report cross-val accuracy plus held-out accuracy on the 200 held-out set.
  4. Projection-transfer regression: on the 200 held-out (baseline + partners, all
     of them), regress the ANSWER (parsed transfer amount) on projection of h onto
     the raw direction, per V and per layer.  Also compute per-V baseline transfers.
  5. Mean-centred variant.
  6. Persist per V: chosen layer ell_V*, v_hat[layer], per-layer probe accuracy,
     projection-transfer beta + p, baseline transfer distributions.

Outputs (under --out):
  directions_raw.pt          -- {V -> {layer -> torch.Tensor(hidden,)}} (raw + mean-centered)
  probe_accuracy.json        -- {V -> [{layer, cv_acc, held_acc}]}
  projection_transfer.json   -- {V -> {layer -> {beta, se, p, n}}}
  baseline_transfer.json     -- {V -> {baseline_effect, per_group_mean, n}}
  layer_pick.json            -- {V -> {ell_star, why}}
  heldout_activations.pt     -- {"acts": [n_held, n_layers, hidden] tensor, "prompts": [...], "meta": [...]}
                                 (cached for reuse by M3 / M4)
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


LLAMA31_INSTRUCT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


ANSWER_INT_RE = re.compile(r"(-?\d{1,3})")


def parse_transfer(text):
    """Parse a $ amount from the model's free-form output.

    Priority:
      1. First integer on the *first line* in [0, 20].
      2. First integer anywhere in the first 40 chars in [0, 20].
      3. Return None (parse failure).
    """
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


def load_data(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def batch_iter(items, bs):
    for i in range(0, len(items), bs):
        yield items[i : i + bs]


def load_model_and_tokenizer(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # last-token = final content token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tok, model


def format_prompt(user_text):
    return LLAMA31_INSTRUCT_TEMPLATE.format(user=user_text)


@torch.no_grad()
def collect_residual_activations(rows, tokenizer, model, batch_size, device,
                                 layers=None):
    """Return acts [N, n_layers, hidden] at the last input position.

    layers=None -> all decoder layers (residual stream *after* each block, i.e.
    hidden_states[1..n_layers]).  hidden_states[0] is the embedding output.
    """
    n = len(rows)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    if layers is None:
        layers = list(range(n_layers + 1))  # 0 == embedding; 1..N are post-block
    n_pick = len(layers)
    out = torch.empty((n, n_pick, hidden), dtype=torch.float32)

    for start in range(0, n, batch_size):
        batch = rows[start : start + batch_size]
        prompts = [format_prompt(r["prompt"]) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        out_forward = model(**enc, output_hidden_states=True, use_cache=False)
        # hidden_states: tuple of (n_layers+1) tensors [B, T, H].  Because we
        # left-pad, the last position (index -1) is the model's next-token
        # prediction site — the exact residual we want for CAA.
        for j, li in enumerate(layers):
            h = out_forward.hidden_states[li][:, -1, :].to(torch.float32).cpu()
            out[start : start + len(batch), j, :] = h
        del out_forward
        if start // batch_size % 10 == 0:
            print(f"  [act] batch {start}/{n}", flush=True)
    return out, layers


def fit_layer_probes(acts_train, labels_train, acts_held, labels_held, n_splits=5):
    """acts_train [N, L, H] -> per-layer probe cv-acc + held-acc."""
    N, L, H = acts_train.shape
    y = np.asarray(labels_train)
    y_held = np.asarray(labels_held)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    per_layer = []
    for li in range(L):
        X = acts_train[:, li, :].numpy()
        Xh = acts_held[:, li, :].numpy() if acts_held is not None else None
        # 5-fold CV
        cv_accs = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs", random_state=42
            )
            clf.fit(X[tr], y[tr])
            cv_accs.append(clf.score(X[te], y[te]))
        cv_acc = float(np.mean(cv_accs))
        # Fit on all train, evaluate on held-out
        clf_all = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        )
        clf_all.fit(X, y)
        held_acc = float(clf_all.score(Xh, y_held)) if Xh is not None else float("nan")
        per_layer.append(dict(layer=li, cv_acc=cv_acc, held_acc=held_acc))
    return per_layer


@torch.no_grad()
def greedy_transfer(prompts, tokenizer, model, batch_size, device,
                    max_new_tokens=8, apply_hooks=None):
    """Greedy-decode `max_new_tokens` tokens per prompt.  If apply_hooks is provided
    (list of (layer_idx, hook_fn) pairs), register/unregister around generation.
    Returns list[str] of raw completions (no prompt).
    """
    outs = []
    handles = []
    if apply_hooks:
        for li, hook_fn in apply_hooks:
            h = model.model.layers[li].register_forward_hook(hook_fn)
            handles.append(h)
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            formatted = [format_prompt(p) for p in batch]
            enc = tokenizer(formatted, return_tensors="pt", padding=True,
                            truncation=True, max_length=1024)
            enc = {k: v.to(device) for k, v in enc.items()}
            in_len = enc["input_ids"].shape[1]
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            new = gen[:, in_len:]
            for row in new:
                text = tokenizer.decode(row, skip_special_tokens=True)
                outs.append(text)
    finally:
        for h in handles:
            h.remove()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--decode_batch_size", type=int, default=16)
    ap.add_argument("--n_sample", type=int, default=-1,
                    help="If >0, use only the first n_sample baseline trials "
                         "(smoke-test / sanity mode).")
    ap.add_argument("--skip_transfer", action="store_true",
                    help="Skip greedy baseline transfer decode (M4 will do it).")
    ap.add_argument("--every_k_layer", type=int, default=1,
                    help="If >1, extract every k-th layer only (speedup).")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    rows = load_data(args.data)
    if args.n_sample > 0:
        # Keep the first n_sample baseline trials + all their partners.
        keep_ids = {r["trial_id"] for r in rows
                    if r["is_paired_partner_of"] is None}
        keep_ids = set(list(keep_ids)[: args.n_sample])
        rows = [r for r in rows
                if r["trial_id"] in keep_ids
                or r.get("is_paired_partner_of") in keep_ids]
        print(f"[m2] SANITY: reduced to {len(rows)} rows "
              f"({args.n_sample} baseline trials + partners).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model_and_tokenizer(args.model)
    n_layers = model.config.num_hidden_layers
    layers = list(range(0, n_layers + 1, args.every_k_layer))
    if layers[-1] != n_layers:
        layers.append(n_layers)
    print(f"[m2] model n_layers={n_layers}, extracting {len(layers)} layers: "
          f"{layers[:6]}...{layers[-3:]}")

    # -- 1) Extract activations for the train split (paired partners of each V) - #
    train_rows = [r for r in rows if r["split"] == "train"]
    held_rows = [r for r in rows if r["split"] == "held"]

    print(f"[m2] extracting train activations: {len(train_rows)} prompts")
    train_acts, layer_ids = collect_residual_activations(
        train_rows, tokenizer, model, batch_size=args.batch_size, device=device,
        layers=layers,
    )
    print(f"[m2] extracting held-out activations: {len(held_rows)} prompts")
    held_acts, _ = collect_residual_activations(
        held_rows, tokenizer, model, batch_size=args.batch_size, device=device,
        layers=layers,
    )

    # Save held-out activations for M3/M4 reuse.
    torch.save(
        dict(acts=held_acts, layer_ids=layer_ids,
             meta=[{k: v for k, v in r.items() if k != "prompt"} for r in held_rows],
             prompts=[r["prompt"] for r in held_rows]),
        os.path.join(args.out, "heldout_activations.pt"),
    )
    torch.save(
        dict(acts=train_acts, layer_ids=layer_ids,
             meta=[{k: v for k, v in r.items() if k != "prompt"} for r in train_rows],
             prompts=[r["prompt"] for r in train_rows]),
        os.path.join(args.out, "train_activations.pt"),
    )

    # -- 2) Per-V raw + mean-centred difference-of-means directions ------------ #
    id_to_idx_train = {r["trial_id"]: i for i, r in enumerate(train_rows)}
    id_to_idx_held = {r["trial_id"]: i for i, r in enumerate(held_rows)}

    directions_raw = {}          # V -> layer_i -> tensor(hidden,)
    directions_mean_centered = {}
    n_pairs_per_V = {}

    global_mean_train = train_acts.mean(dim=0)  # [L, H]

    for V in ["G", "A", "I", "M"]:
        # Pairs: baseline row + its V-partner (both same split).
        pos_rows = [r for r in train_rows
                    if r["variable_flipped"] == V]
        pair_idx = []
        for pr in pos_rows:
            parent_id = pr["is_paired_partner_of"]
            if parent_id in id_to_idx_train:
                pair_idx.append(
                    (id_to_idx_train[parent_id], id_to_idx_train[pr["trial_id"]],
                     pr[V])  # value of V in the *partner* (the "positive" side)
                )
        n_pairs_per_V[V] = len(pair_idx)
        if not pair_idx:
            print(f"[m2] WARN: no train pairs for V={V}")
            continue
        # Convention: v_hat = mean_{pairs} (h(partner_with_V=value1) - h(partner_with_V=value0))
        # Fix value0 alphabetically so sign is consistent.
        # For V=G: value0=female, value1=male (male-female direction)
        # For V=A: value0=old, value1=young  (young-old direction; young=1)
        # For V=I: value0=give, value1=take  (take-give direction)
        # For V=M: value0=meet, value1=no-meet (no-meet - meet direction)
        # We define the "positive" side by the *baseline row's* value: v_hat is the shift
        # produced by flipping FROM the partner's value TO the baseline's value.  To keep
        # a clean deterministic sign, we compute: for each pair, delta = h_baseline - h_partner
        # if baseline_val == pos_side else h_partner - h_baseline.
        pos_side_map = {"G": "male", "A": "young", "I": "take-frame", "M": "no-meet"}
        pos_val = pos_side_map[V]
        L, H = train_acts.shape[1], train_acts.shape[2]
        deltas = torch.zeros((len(pair_idx), L, H), dtype=torch.float32)
        for i, (bi, pi, partner_v_val) in enumerate(pair_idx):
            base_val = train_rows[bi][V]
            if base_val == pos_val:
                deltas[i] = train_acts[bi] - train_acts[pi]
            else:
                deltas[i] = train_acts[pi] - train_acts[bi]
        v_hat = deltas.mean(dim=0)  # [L, H]
        directions_raw[V] = v_hat
        # Mean-centred direction is computed below on globally-centred activations;
        # this dict is filled in the second pass.

    # Mean-centring baseline: recompute v_hat on activations after subtracting
    # the global mean h.  This is the "Improving-Activation-Steering-with-Mean-Centering"
    # baseline.
    train_acts_mc = train_acts - global_mean_train.unsqueeze(0)
    for V in ["G", "A", "I", "M"]:
        pos_rows = [r for r in train_rows
                    if r["variable_flipped"] == V]
        pair_idx = []
        for pr in pos_rows:
            parent_id = pr["is_paired_partner_of"]
            if parent_id in id_to_idx_train:
                pair_idx.append((id_to_idx_train[parent_id],
                                 id_to_idx_train[pr["trial_id"]]))
        if not pair_idx:
            continue
        pos_side_map = {"G": "male", "A": "young", "I": "take-frame", "M": "no-meet"}
        pos_val = pos_side_map[V]
        L, H = train_acts_mc.shape[1], train_acts_mc.shape[2]
        deltas = torch.zeros((len(pair_idx), L, H), dtype=torch.float32)
        for i, (bi, pi) in enumerate(pair_idx):
            base_val = train_rows[bi][V]
            if base_val == pos_val:
                deltas[i] = train_acts_mc[bi] - train_acts_mc[pi]
            else:
                deltas[i] = train_acts_mc[pi] - train_acts_mc[bi]
        directions_mean_centered[V] = deltas.mean(dim=0)

    torch.save(
        dict(
            raw=directions_raw,
            mean_centered=directions_mean_centered,
            layer_ids=layer_ids,
            n_pairs=n_pairs_per_V,
            hidden=train_acts.shape[2],
            n_layers_extracted=train_acts.shape[1],
            global_mean_train=global_mean_train,
        ),
        os.path.join(args.out, "directions_raw.pt"),
    )

    # -- 3) Linear probes per (V, layer) --------------------------------------- #
    pos_side_map = {"G": "male", "A": "young", "I": "take-frame", "M": "no-meet"}
    probe_res = {}
    for V in ["G", "A", "I", "M"]:
        pos_val = pos_side_map[V]
        y_train = np.array([1 if r[V] == pos_val else 0 for r in train_rows])
        y_held = np.array([1 if r[V] == pos_val else 0 for r in held_rows])
        pl = fit_layer_probes(train_acts, y_train, held_acts, y_held, n_splits=5)
        # Map absolute layer index onto our subsampled list.
        for row in pl:
            row["layer_abs"] = layer_ids[row["layer"]]
        probe_res[V] = pl
        print(f"[m2] probe V={V}: best cv_acc={max(r['cv_acc'] for r in pl):.3f} "
              f"@ layer_abs {max(pl, key=lambda r: r['cv_acc'])['layer_abs']}")
    with open(os.path.join(args.out, "probe_accuracy.json"), "w") as f:
        json.dump(probe_res, f, indent=2)

    # -- 4) Baseline transfer decode on the 200 held-out ----------------------- #
    baseline_res = {V: {} for V in ["G", "A", "I", "M"]}
    projection_res = {V: {} for V in ["G", "A", "I", "M"]}
    layer_pick = {}

    if not args.skip_transfer:
        # Only decode the 200 held-out *baseline* rows (not their partners): the
        # partners are used for probe eval (label recovery) and for projection
        # regression separately.
        held_base_rows = [r for r in held_rows if r["is_paired_partner_of"] is None]
        prompts = [r["prompt"] for r in held_base_rows]
        print(f"[m2] decoding baseline transfer for {len(prompts)} held-out prompts")
        outs = greedy_transfer(prompts, tokenizer, model,
                               batch_size=args.decode_batch_size, device=device,
                               max_new_tokens=8)
        taus = [parse_transfer(o) for o in outs]
        n_parse_fail = sum(1 for t in taus if t is None)
        # -- 4a) Per-V baseline effect: mean_tau at V=pos_val - mean_tau at V=neg_val --- #
        for V in ["G", "A", "I", "M"]:
            pos_val = pos_side_map[V]
            pos = [t for r, t in zip(held_base_rows, taus)
                   if t is not None and r[V] == pos_val]
            neg = [t for r, t in zip(held_base_rows, taus)
                   if t is not None and r[V] != pos_val]
            eff = (float(np.mean(pos)) - float(np.mean(neg))) if (pos and neg) else 0.0
            baseline_res[V] = dict(
                pos_val=pos_val,
                mean_pos=float(np.mean(pos)) if pos else None,
                n_pos=len(pos),
                mean_neg=float(np.mean(neg)) if neg else None,
                n_neg=len(neg),
                baseline_effect=eff,
                parse_failure_rate=n_parse_fail / len(taus) if taus else 0.0,
                sample_generation=outs[0][:120] if outs else None,
            )
            print(f"[m2] baseline V={V}: mean_pos={baseline_res[V]['mean_pos']} "
                  f"mean_neg={baseline_res[V]['mean_neg']} "
                  f"effect={eff:.3f} parse_fail={n_parse_fail}/{len(taus)}")
        # -- 4b) Projection - transfer regression per (V, layer) on the held-out baseline --- #
        held_base_idx = [id_to_idx_held[r["trial_id"]] for r in held_base_rows]
        held_base_acts = held_acts[held_base_idx]  # [n_base, L, H]
        for V in ["G", "A", "I", "M"]:
            if V not in directions_raw:
                continue
            v_hat = directions_raw[V]  # [L, H]
            unit = v_hat / (v_hat.norm(dim=-1, keepdim=True) + 1e-8)  # [L, H]
            # projections [n_base, L]
            proj = (held_base_acts * unit.unsqueeze(0)).sum(dim=-1)
            y = np.array([t for t in taus], dtype=float)  # may contain NaN via None
            keep = np.array([t is not None for t in taus])
            proj_np = proj.numpy()
            for li in range(proj_np.shape[1]):
                x = proj_np[keep, li]
                yk = y[keep]
                if len(x) < 10:
                    continue
                # simple OLS: y = beta * x + intercept
                x_ = np.column_stack([x, np.ones_like(x)])
                coef, resid, rank, sv = np.linalg.lstsq(x_, yk, rcond=None)
                beta = float(coef[0])
                # standard error via residual variance
                yhat = x_ @ coef
                resid_v = yk - yhat
                sigma2 = float(np.sum(resid_v ** 2) / max(1, len(x) - 2))
                cov = sigma2 * np.linalg.inv(x_.T @ x_ + 1e-8 * np.eye(2))
                se = float(np.sqrt(max(0.0, cov[0, 0])))
                t_stat = beta / (se + 1e-12)
                # 2-sided p from normal approx (adequate for n>100)
                from math import erf, sqrt
                p = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
                projection_res[V][str(layer_ids[li])] = dict(
                    beta=beta, se=se, t=float(t_stat), p=p, n=int(len(x)),
                )

    # -- 5) Layer pick per V (max cv_acc; break ties by projection |beta| if present) --- #
    for V in ["G", "A", "I", "M"]:
        pl = probe_res[V]
        if not pl:
            continue
        best = max(pl, key=lambda r: r["cv_acc"])
        layer_pick[V] = dict(
            ell_star=best["layer_abs"],
            layer_local=best["layer"],
            cv_acc=best["cv_acc"],
            held_acc=best["held_acc"],
            why="max cv_acc on train (5-fold)",
        )
    with open(os.path.join(args.out, "baseline_transfer.json"), "w") as f:
        json.dump(baseline_res, f, indent=2)
    with open(os.path.join(args.out, "projection_transfer.json"), "w") as f:
        json.dump(projection_res, f, indent=2)
    with open(os.path.join(args.out, "layer_pick.json"), "w") as f:
        json.dump(layer_pick, f, indent=2)

    print(f"[m2] done; layer_pick={layer_pick}")


if __name__ == "__main__":
    main()
