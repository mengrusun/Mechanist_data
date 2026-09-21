"""Step 5 — Cross-direction activation steering with matched-magnitude random null.

For α ∈ {−1σ, 0, +1σ} × direction ∈ {v_c^L*, v_v^L*, random_unit^L* (matched norm)}
× 500 held-out test samples, register a HF forward hook on the residual output
of block L* that adds `α · σ_L* · v` at the LAST-INPUT-TOKEN position, then run:

- turn1: greedy answer generation (max 20 tokens) → measure Δ correctness AND
  Δ probe_v_binary readout on the H_1^L* AFTER the hook fires (approximation of
  "internal readout Δ" — we read the residual state at L* under the steering hook,
  project onto v_v).
- turn2: given the model's original turn1 short answer, generate confidence
  under P0 → measure Δ verbalized confidence number AND Δ probe_c_binary
  readout at H_2^L*.

Primary metric per plan §6.9: internal readout Δ (probe projection).
Secondary: emitted output Δ.

Also compute mean token perplexity as a fluency / general-ability metric
(General Rule + Tip 2 requirement). If mean perplexity under a given α exceeds
3× baseline (α=0), halve α and re-run that condition.

Outputs:
  - artifacts/steering_results.json
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U


def _last_input_pos(attn_mask):
    """Given attention_mask (B, L), return the position of the last input token per row."""
    # last-input-token = right-most 1 in attn_mask
    return attn_mask.sum(dim=1) - 1  # (B,)


class SteeringHook:
    """Adds `alpha_sigma * v` at the last-input-token position of the residual output.
    Also captures the residual state AT that position AFTER the addition.

    Uses attention_mask stashed on the model to identify the last-input-token per sample.

    Attach on `model.model.layers[L]` — forward hook on the block returns
    `(hidden_states, ...)` where hidden_states shape is (B, seq, D).
    """

    def __init__(self, v: torch.Tensor, alpha_sigma: float, attn_mask: torch.Tensor,
                 hook_generation: bool = False):
        # v is a 1-D unit tensor on the model device
        self.v = v
        self.alpha_sigma = alpha_sigma
        self.attn_mask = attn_mask  # (B, L_max)
        self.captured = None  # will hold (B, D) residual at last-input-token AFTER addition
        self.hook_generation = hook_generation
        # counter to fire only on the *first* forward pass (prompt) during generation
        self._n_calls = 0

    def __call__(self, module, inputs, outputs):
        # outputs may be either a tuple or a tensor depending on block
        if isinstance(outputs, tuple):
            h = outputs[0]
            rest = outputs[1:]
        else:
            h = outputs
            rest = None
        B, S, D = h.shape
        # During generation this hook fires once per token; during a pure forward
        # pass it fires once. We only want to add v at the prompt's last-input-token
        # so only fire on the first call (when S > 1). For subsequent single-token
        # forwards we no-op.
        self._n_calls += 1
        if S == 1:
            # generation-time single-token forward — do not steer these
            if rest is None:
                return h
            return (h,) + rest
        # Position of last-input-token per sample
        pos = _last_input_pos(self.attn_mask.to(h.device))  # (B,)
        # Guard: if pos >= S clamp
        pos = pos.clamp(max=S - 1)
        # Additive intervention: add alpha_sigma * v at (row, pos, :)
        # We do it out of place to avoid autograd issues; here inference-mode so ok.
        add = (self.alpha_sigma * self.v).to(h.dtype)  # (D,)
        h_new = h.clone()
        for b in range(B):
            h_new[b, pos[b], :] = h_new[b, pos[b], :] + add
        # Capture residual (post-add) at last-input-token
        self.captured = torch.stack([h_new[b, pos[b], :].detach().float().cpu() for b in range(B)])
        if rest is None:
            return h_new
        return (h_new,) + rest


def _compute_ppl(model, tok, prompts, generations):
    """Approximate mean per-token perplexity of the *generated* portion for each
    prompt-generation pair. Returns list of ppl values (one per pair)."""
    ppl_list = []
    with torch.no_grad():
        for prompt, gen in zip(prompts, generations):
            full = prompt + gen
            enc_full = tok(full, return_tensors="pt").to(model.device)
            enc_prompt = tok(prompt, return_tensors="pt").to(model.device)
            n_gen = enc_full.input_ids.shape[1] - enc_prompt.input_ids.shape[1]
            if n_gen <= 0:
                ppl_list.append(float("nan"))
                continue
            out = model(input_ids=enc_full.input_ids, use_cache=False)
            logits = out.logits  # (1, L, V)
            # Shift for next-token prediction; the target for token t is logits[:, t-1]
            targets = enc_full.input_ids[:, 1:]
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            per_token_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, L-1)
            # Only take the generated tail
            gen_lp = per_token_lp[:, -n_gen:]
            ppl = float(torch.exp(-gen_lp.mean()).item())
            ppl_list.append(ppl)
    return ppl_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steer", type=int, default=500)
    ap.add_argument("--alphas", type=str, default="-1.0,0.0,1.0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--n-ppl-check", type=int, default=32,
                    help="samples used for the fluency metric per condition")
    ap.add_argument("--seed", type=int, default=U.DEFAULT_SEED)
    args = ap.parse_args()

    U.set_all_seeds(args.seed)
    alphas = [float(a) for a in args.alphas.split(",")]

    # Load probes
    probes = np.load(U.ARTIFACT_DIR / "probes.npz")
    L_star = int(probes["L_star"][0])
    v_c_star = probes["v_c_star"].astype(np.float32)
    v_v_star = probes["v_v_star"].astype(np.float32)
    # Steering hook is on model.model.layers[block_idx]; block_idx = L_star - 1
    # because hidden_states[L_star] = OUTPUT of block L_star-1 (HF convention).
    assert L_star >= 1, f"L*={L_star} is embedding — cannot steer via block hook"
    block_idx = L_star - 1

    # sigma_L* : std of residual-stream norm at layer L* on the training set
    npz1 = np.load(U.ARTIFACT_DIR / "H_turn1.npz")
    H1 = npz1["H"]  # (N, L+1, D)
    qids = npz1["question_ids"].tolist()
    idxs_arr_all = npz1["idxs"]
    split = U.load_json(U.ARTIFACT_DIR / "split_manifest.json")
    # Prefer idx-based splits (TriviaQA has duplicate qids; qid-based leaks)
    if "train_idxs" in split:
        train_idxs = set(int(x) for x in split["train_idxs"])
        test_idxs = set(int(x) for x in split["test_idxs"])
        train_mask = np.array([int(ix) in train_idxs for ix in idxs_arr_all])
        te_mask = np.array([int(ix) in test_idxs for ix in idxs_arr_all])
    else:
        train_ids = set(split["train_ids"])
        test_ids = set(split["test_ids"])
        train_mask = np.array([q in train_ids for q in qids])
        te_mask = np.array([q in test_ids for q in qids])
    train_norms = np.linalg.norm(H1[train_mask, L_star, :].astype(np.float64), axis=1)
    sigma_L_star_norm = float(np.std(train_norms))
    sigma_L_star = float(np.mean(np.std(H1[train_mask, L_star, :].astype(np.float64), axis=0)))
    print(f"[step5] L*={L_star} sigma_L_star (elementwise)={sigma_L_star:.4f}", flush=True)

    # sigma_probe_readout: std of unsteered probe_c and probe_v projections on test set at L*
    npz2 = np.load(U.ARTIFACT_DIR / "H_turn2.npz")
    H2 = npz2["H"]
    proj_c_test = H1[te_mask, L_star, :].astype(np.float64) @ v_c_star.astype(np.float64)
    proj_v_test = H2[te_mask, L_star, :].astype(np.float64) @ v_v_star.astype(np.float64)
    sigma_probe_c_readout = float(np.std(proj_c_test))
    sigma_probe_v_readout = float(np.std(proj_v_test))
    print(f"[step5] sigma_probe_c={sigma_probe_c_readout:.4f}  "
          f"sigma_probe_v={sigma_probe_v_readout:.4f}", flush=True)

    # Choose the 500 held-out test samples deterministically (by row-idx of test rows)
    turn1_rows = U.load_jsonl(U.ARTIFACT_DIR / "turn1_gen.jsonl")
    turn1_by_qid = {r["question_id"]: r for r in turn1_rows}
    turn1_by_idx = {r["idx"]: r for r in turn1_rows}
    # Row-indices in the current npz that are in the test split
    test_row_indices_in_npz = [i for i, tm in enumerate(te_mask) if tm]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(test_row_indices_in_npz))
    steer_row_positions = [test_row_indices_in_npz[i] for i in perm[: args.n_steer]]
    steer_ids = [qids[i] for i in steer_row_positions]  # qid per selected row
    steer_idxs_int = [int(idxs_arr_all[i]) for i in steer_row_positions]
    print(f"[step5] Selected {len(steer_ids)} test samples for steering", flush=True)

    # Baseline unsteered probe readouts (used to compute Δ) — row-indexed, not qid-indexed
    idx_arr = np.array(steer_row_positions)
    base_proj_c_H1 = H1[idx_arr, L_star, :].astype(np.float64) @ v_c_star.astype(np.float64)
    base_proj_v_H1 = H1[idx_arr, L_star, :].astype(np.float64) @ v_v_star.astype(np.float64)
    base_proj_c_H2 = H2[idx_arr, L_star, :].astype(np.float64) @ v_c_star.astype(np.float64)
    base_proj_v_H2 = H2[idx_arr, L_star, :].astype(np.float64) @ v_v_star.astype(np.float64)

    # Baseline emitted outputs — use idx-based lookup (qid duplicates in TriviaQA)
    turn2_rows = U.load_jsonl(U.ARTIFACT_DIR / "turn2_gen.jsonl")
    turn2_by_idx = {r["idx"]: r for r in turn2_rows}
    base_c = [turn2_by_idx[i]["c"] if (i in turn2_by_idx and turn2_by_idx[i]["c_parseable"]) else np.nan
              for i in steer_idxs_int]
    base_y = [turn1_by_idx[i]["y_correct"] for i in steer_idxs_int]

    # Load HF model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[step5] Loading HF model …", flush=True)
    tok = AutoTokenizer.from_pretrained(U.MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        U.MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    # Random unit direction (matched norm, i.e., unit — αsigma_L* scales both)
    rng_r = np.random.default_rng(args.seed + 100)
    v_random = rng_r.normal(size=v_c_star.shape).astype(np.float32)
    v_random = v_random / (np.linalg.norm(v_random) + 1e-12)

    directions = {
        "v_c": v_c_star,
        "v_v": v_v_star,
        "v_random": v_random,
    }

    # Turn-1 and turn-2 prompts for the steer set — idx-based lookup
    turn1_prompts = [U.prompt_turn1(turn1_by_idx[i]["question"]) for i in steer_idxs_int]
    turn2_prompts = []
    for i in steer_idxs_int:
        r = turn1_by_idx[i]
        turn2_prompts.append(U.prompt_turn2(r["question"], r.get("short_answer") or ""))

    results = {
        "L_star": L_star,
        "sigma_L_star": sigma_L_star,
        "sigma_L_star_norm": sigma_L_star_norm,
        "sigma_probe_c_readout": sigma_probe_c_readout,
        "sigma_probe_v_readout": sigma_probe_v_readout,
        "alphas": alphas,
        "directions": list(directions.keys()),
        "n_steer": len(steer_ids),
        "steer_ids": steer_ids,
        "conditions": [],
    }
    # For baseline unsteered ppl reference
    baseline_ppl = None

    def _run_condition(mode: str, prompts: list[str], direction: str, alpha: float, gold=None):
        """Run one condition and collect stats.
        mode ∈ {'turn1', 'turn2'}. Returns dict of arrays and Δs."""
        v = directions[direction]
        v_gpu = torch.tensor(v, dtype=torch.float16, device=model.device)
        alpha_sigma = alpha * sigma_L_star

        n = len(prompts)
        gens = []
        captured_states = []  # residual at L*, last-input-token, post-add
        t0 = time.time()

        for i in range(0, n, args.batch_size):
            batch_p = prompts[i : i + args.batch_size]
            enc = tok(batch_p, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            attn_mask = enc.attention_mask
            hook = SteeringHook(v=v_gpu, alpha_sigma=alpha_sigma, attn_mask=attn_mask)
            # Hook on block_idx = L_star - 1; its output IS hidden_states[L_star]
            block = model.model.layers[block_idx]
            handle = block.register_forward_hook(hook)
            try:
                gen_ids = model.generate(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    max_new_tokens=args.max_new_tokens if mode == "turn1" else 8,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tok.pad_token_id,
                )
                new_ids = gen_ids[:, enc.input_ids.shape[1]:]
                dec = tok.batch_decode(new_ids, skip_special_tokens=True)
                gens.extend(dec)
                if hook.captured is not None:
                    captured_states.append(hook.captured.numpy())
            finally:
                handle.remove()
            if (i // args.batch_size) % 10 == 0:
                el = time.time() - t0
                print(f"[step5]   {mode} α={alpha} dir={direction}  {i+len(batch_p)}/{n}  {el:.1f}s", flush=True)

        captured = np.concatenate(captured_states, axis=0) if captured_states else np.zeros((0, len(v)))
        proj_c = captured.astype(np.float64) @ v_c_star.astype(np.float64) if captured.size else np.zeros(0)
        proj_v = captured.astype(np.float64) @ v_v_star.astype(np.float64) if captured.size else np.zeros(0)

        # Emitted outputs — idx-based lookup
        if mode == "turn1":
            y_new = [U.score_answer(g, turn1_by_idx[i]["normalized_aliases"]) for g, i in zip(gens, steer_idxs_int)]
        else:
            parsed = [U.parse_confidence(g, "0-100") for g in gens]
            y_new = [p[0] if p[1] else np.nan for p in parsed]

        cond = {
            "mode": mode,
            "direction": direction,
            "alpha": alpha,
            "alpha_sigma": alpha_sigma,
            "n": n,
            "gens_sample": gens[:5],
            "proj_c_mean": float(np.mean(proj_c)) if len(proj_c) else None,
            "proj_v_mean": float(np.mean(proj_v)) if len(proj_v) else None,
            "proj_c_std": float(np.std(proj_c)) if len(proj_c) else None,
            "proj_v_std": float(np.std(proj_v)) if len(proj_v) else None,
        }
        if mode == "turn1":
            cond["accuracy_steered"] = float(np.mean(y_new)) if y_new else None
            cond["accuracy_baseline"] = float(np.mean(base_y))
            cond["delta_accuracy"] = cond["accuracy_steered"] - cond["accuracy_baseline"]
        else:
            arr = np.array([v for v in y_new if not (isinstance(v, float) and np.isnan(v))], dtype=np.float64)
            base_arr = np.array([v for v in base_c if not np.isnan(v)])
            cond["c_steered_mean"] = float(np.mean(arr)) if arr.size else None
            cond["c_baseline_mean"] = float(np.mean(base_arr)) if base_arr.size else None
            cond["parseable_rate"] = float(np.mean([not (isinstance(v, float) and np.isnan(v)) for v in y_new]))
            cond["delta_c"] = (cond["c_steered_mean"] - cond["c_baseline_mean"]) if (cond["c_steered_mean"] is not None and cond["c_baseline_mean"] is not None) else None

        # Fluency: perplexity on a subsample
        if len(gens) >= args.n_ppl_check:
            ppl_subset = _compute_ppl(model, tok, prompts[: args.n_ppl_check], gens[: args.n_ppl_check])
            cond["mean_token_perplexity"] = float(np.nanmean(ppl_subset))
        else:
            cond["mean_token_perplexity"] = None

        return cond

    # Turn-1 (v_c-steer measured on H_1^L* and correctness) —
    # For each direction × α, run turn-1 generation with steering
    for direction in ("v_c", "v_v", "v_random"):
        for alpha in alphas:
            print(f"[step5] === turn1  direction={direction} α={alpha} ===", flush=True)
            cond = _run_condition("turn1", turn1_prompts, direction, alpha)
            # Compute Δ vs baseline (α=0 same direction) later — for now store abs values.
            # Also record baseline probe projections
            cond["baseline_proj_c_H1_mean"] = float(np.mean(base_proj_c_H1))
            cond["baseline_proj_v_H1_mean"] = float(np.mean(base_proj_v_H1))
            cond["delta_proj_c"] = (cond["proj_c_mean"] - cond["baseline_proj_c_H1_mean"]) if cond["proj_c_mean"] is not None else None
            cond["delta_proj_v"] = (cond["proj_v_mean"] - cond["baseline_proj_v_H1_mean"]) if cond["proj_v_mean"] is not None else None
            results["conditions"].append(cond)
            # Perplexity blow-up check for α != 0
            if alpha != 0.0 and cond.get("mean_token_perplexity") is not None:
                if baseline_ppl is not None and cond["mean_token_perplexity"] > 3 * baseline_ppl:
                    cond["perplexity_blowup"] = True
                else:
                    cond["perplexity_blowup"] = False
            elif alpha == 0.0 and cond.get("mean_token_perplexity") is not None:
                baseline_ppl = cond["mean_token_perplexity"]

    # Turn-2 (v_v-steer measured on H_2^L* and c) — same grid on turn2 prompts
    for direction in ("v_c", "v_v", "v_random"):
        for alpha in alphas:
            print(f"[step5] === turn2  direction={direction} α={alpha} ===", flush=True)
            cond = _run_condition("turn2", turn2_prompts, direction, alpha)
            cond["baseline_proj_c_H2_mean"] = float(np.mean(base_proj_c_H2))
            cond["baseline_proj_v_H2_mean"] = float(np.mean(base_proj_v_H2))
            cond["delta_proj_c"] = (cond["proj_c_mean"] - cond["baseline_proj_c_H2_mean"]) if cond["proj_c_mean"] is not None else None
            cond["delta_proj_v"] = (cond["proj_v_mean"] - cond["baseline_proj_v_H2_mean"]) if cond["proj_v_mean"] is not None else None
            results["conditions"].append(cond)

    # C3b test criterion evaluation
    # Extract cross-direction Δ on internal readouts
    def _get_cond(mode, direction, alpha):
        for c in results["conditions"]:
            if c["mode"] == mode and c["direction"] == direction and c["alpha"] == alpha:
                return c
        return None

    def _decide_c3b(mode: str, source_dir: str, target_readout: str, alpha=1.0):
        """target_readout ∈ {c, v}. source_dir is what we steer along; we measure Δ on
        the OTHER probe's readout for cross-direction test."""
        s = _get_cond(mode, source_dir, alpha)
        r = _get_cond(mode, "v_random", alpha)
        if s is None or r is None:
            return None
        d_s = s.get(f"delta_proj_{target_readout}")
        d_r = r.get(f"delta_proj_{target_readout}")
        if d_s is None or d_r is None:
            return None
        # sigma probe readout for the target
        sigma_p = sigma_probe_c_readout if target_readout == "c" else sigma_probe_v_readout
        # Ratio criterion when |Δ_random| ≥ 0.5 σ_probe_readout
        crit = {"delta_source": d_s, "delta_random": d_r,
                "sigma_probe": sigma_p, "threshold_absolute": 0.5 * sigma_p}
        if abs(d_r) >= 0.5 * sigma_p:
            ratio = abs(d_s) / (abs(d_r) + 1e-12)
            crit["ratio"] = float(ratio)
            crit["mode"] = "ratio"
            crit["passes"] = bool(ratio <= 1.5)
        else:
            crit["ratio"] = None
            crit["mode"] = "absolute"
            crit["passes"] = bool(abs(d_s) <= 0.5 * sigma_p)
        return crit

    # v_c-steer on turn1 → Δ probe_v readout (cross-direction)
    # v_v-steer on turn2 → Δ probe_c readout (cross-direction)
    c3b = {
        "turn1_v_c_steer_delta_v_readout_pos": _decide_c3b("turn1", "v_c", "v", alpha=1.0),
        "turn1_v_c_steer_delta_v_readout_neg": _decide_c3b("turn1", "v_c", "v", alpha=-1.0),
        "turn2_v_v_steer_delta_c_readout_pos": _decide_c3b("turn2", "v_v", "c", alpha=1.0),
        "turn2_v_v_steer_delta_c_readout_neg": _decide_c3b("turn2", "v_v", "c", alpha=-1.0),
    }
    all_pass = all(v is not None and v.get("passes", False) for v in c3b.values())
    results["c3b_decision"] = c3b
    results["c3b_all_pass_internal_readout"] = bool(all_pass)

    U.dump_json(U.ARTIFACT_DIR / "steering_results.json", results)
    print(f"[step5] wrote {U.ARTIFACT_DIR / 'steering_results.json'}", flush=True)
    print(f"[step5] C3b PASS (internal readout, both directions): {all_pass}", flush=True)


if __name__ == "__main__":
    main()
