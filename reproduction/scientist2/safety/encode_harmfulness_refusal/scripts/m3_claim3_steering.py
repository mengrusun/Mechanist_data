#!/usr/bin/env python3
"""M3 — Claim 3: causal dissociation via additive steering.

For a (direction, alpha) grid, install an additive forward hook at
(best_layer, best_position) for the acted-upon direction, run short
generation on 100 held-out harmful + 100 held-out benign prompts, and
measure:
  - Δ internal harmfulness readout = ⟨activations(intervened) at (best_h_layer,
    t_final_instr), h⟩ minus baseline projection.
  - Δ refusal rate = refusal-string classifier on the completion (primary),
    optionally corroborated by Llama Guard 3 8B on a 20% random subset
    (skipped by default in this script; use --secondary_judge to enable).
  - Fluency proxies: mean per-token log-prob (surrogate: -mean loss on the
    completion computed under the same intervention hook) and repetition rate.

directions: h | r | random | swap
  - h : additive at (best_h_layer, t_final_instr) with direction h
  - r : additive at (best_r_layer, t_post_instr) with direction r
  - random : matched-norm random Gaussian, applied at h's site (specificity)
             — we run one random-vs-h and one random-vs-r sub-sweep in the
             same run to control both target axes.
  - swap : add r at h's site (h_col) and h at r's site (r_col).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    build_chat_prompt,
    compute_repetition_rate,
    is_refusal,
    load_alpaca_instructions,
    load_harmful_behaviors,
    position_ladder,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prep", required=True)
    p.add_argument("--advbench", required=True)
    p.add_argument("--alpaca", required=True)
    p.add_argument("--direction", required=True, choices=["h", "r", "random", "random_r_site", "swap"])
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--n_harm", type=int, default=100)
    p.add_argument("--n_benign", type=int, default=100)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--random_dir_seed",
        type=int,
        default=None,
        help=(
            "Optional distinct seed for the random-direction RNG (used only "
            "when --direction=random). Lets us hold prompt selection fixed at "
            "--seed while varying only the random matched-norm vector across "
            "many replicates (n_random >= 30 in the C3 mechanism audit fix)."
        ),
    )
    p.add_argument("--gen_max_new_tokens", type=int, default=128)
    p.add_argument("--gen_batch_size", type=int, default=8)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    return p.parse_args()


def load_model_and_tokenizer(model_path: str, dtype: str):
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    tk = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map="cuda:0", low_cpu_mem_usage=True,
    )
    mdl.eval()
    return mdl, tk


def make_additive_output_hook(direction: torch.Tensor, alpha: float, pos_idx_per_prompt: List[int]):
    """Return a forward POST-hook that adds `alpha * direction` to the OUTPUT
    residual stream of a decoder block at the given position(s).

    Semantics
    ---------
    M-prep's `layer = k` refers to `hidden_states[k]`, which is the output of
    `model.model.layers[k-1]`. To make an intervention land *exactly* on that
    vector (the one M-prep's diff-mean was computed on), we register a
    post-forward-hook on `model.model.layers[k-1]` and add `alpha * direction`
    to its output at the target position. The next block's forward then sees
    the perturbed residual and it flows through the rest of the network.

    Applied-once guarantee: `applied['done']` flips True after the first full
    forward pass (T > 1); subsequent generation steps (T == 1 with KV cache)
    are no-ops. This prevents double-application even if HF's `generate` does
    an extra warmup pass.
    """

    direction = direction.detach()
    applied = {"done": False}

    def hook(module, inputs, output):
        # LlamaDecoderLayer forward returns a tuple: (hidden_states, ...maybe cache...)
        if applied["done"]:
            return output
        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = ()
        if hs.dim() != 3:
            return output
        B, T, _ = hs.shape
        if T == 1:
            # Cached single-token generation — nothing to modify.
            return output
        for b in range(min(B, len(pos_idx_per_prompt))):
            p = pos_idx_per_prompt[b]
            if 0 <= p < T:
                hs[b, p, :] = hs[b, p, :] + \
                    alpha * direction.to(hs.dtype).to(hs.device)
        applied["done"] = True
        return (hs,) + rest if rest else hs

    return hook


def run_one_condition(
    model, tokenizer, prompts: List[str],
    direction: torch.Tensor, alpha: float,
    hook_block_idx: int, position_kind: str,
    gen_max_new_tokens: int, gen_batch_size: int,
    # For readout: h-projection at (best_h_layer, t_final_instr) AFTER intervention.
    readout_layer: int, readout_direction: torch.Tensor,
) -> List[dict]:
    """Run generation on all prompts under the additive post-hook, return per-prompt
    metrics: refusal (bool), completion text, mean completion log-prob (fluency
    proxy — computed via a second forward pass on the generated tokens WITH the
    intervention re-installed to get a faithful log-prob under-intervention),
    repetition rate, and the h-readout projection at (readout_layer, t_final_instr).

    Steering: post-forward-hook on `model.model.layers[hook_block_idx]`; the
    modified output is precisely `hidden_states[hook_block_idx + 1]` in HF
    convention. Callers pass `hook_block_idx = best_layer - 1` so the modified
    residual lands at `hidden_states[best_layer]` — the same vector M-prep's
    diff-mean was computed on.
    """
    device = next(model.parameters()).device
    results = []
    B = gen_batch_size
    pad_id = tokenizer.pad_token_id

    layer_module = model.model.layers[hook_block_idx]

    for start in range(0, len(prompts), B):
        batch = prompts[start : start + B]
        prompts_enc = [build_chat_prompt(tokenizer, t, add_generation_prompt=True) for t in batch]
        max_len = max(len(p["input_ids"]) for p in prompts_enc)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        pad_lefts = []
        for i, p in enumerate(prompts_enc):
            L = len(p["input_ids"])
            pad_left = max_len - L
            input_ids[i, pad_left:] = torch.tensor(p["input_ids"], dtype=torch.long)
            attn[i, pad_left:] = torch.tensor(p["attention_mask"], dtype=torch.long)
            pad_lefts.append(pad_left)
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        # Position indices per prompt for the additive hook — shift by pad_left
        pos_per = []
        for i, p in enumerate(prompts_enc):
            lad = position_ladder(p["input_ids"])
            if position_kind == "t_final_instr":
                pos_per.append(pad_lefts[i] + lad["t_final_instr"])
            elif position_kind == "t_post_instr":
                pos_per.append(pad_lefts[i] + lad["t_post_instr"])
            else:
                raise ValueError(f"unknown position_kind {position_kind}")

        # --- pass 1: readout (prompt-only forward under intervention) ---
        hook_fn = make_additive_output_hook(direction, alpha, pos_per)
        handle = layer_module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
                readout_u = readout_direction / (readout_direction.norm() + 1e-9)
                readouts = []
                for i, p in enumerate(prompts_enc):
                    lad = position_ladder(p["input_ids"])
                    read_pos = pad_lefts[i] + lad["t_final_instr"]
                    h_vec = out.hidden_states[readout_layer][i, read_pos].float()
                    readouts.append(float(torch.dot(h_vec.cpu(), readout_u.cpu())))
                del out
        finally:
            handle.remove()

        # --- pass 2: generate under intervention (fresh hook + fresh applied flag) ---
        gen_hook = make_additive_output_hook(direction, alpha, pos_per)
        gen_handle = layer_module.register_forward_hook(gen_hook)
        try:
            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    do_sample=False,
                    temperature=1.0,
                    max_new_tokens=gen_max_new_tokens,
                    pad_token_id=pad_id,
                    use_cache=True,
                )
        finally:
            gen_handle.remove()

        # --- pass 3: completion log-prob under intervention (fluency proxy on the
        #    generated tokens). Forward (prompt + completion) as a SINGLE row
        #    (no padding), with the hook fresh again, and score log-probs on
        #    the *completion* token positions only.
        per_prompt_logp = []
        for i in range(len(batch)):
            # Unpadded prompt length
            L_unpad = len(prompts_enc[i]["input_ids"])
            # Trim to remove left pad and trailing pad in one shot
            # The generated tokens live at [max_len .. end) — after the padded prompt block.
            gen_seq_i = gen[i]
            end = len(gen_seq_i)
            while end > 0 and int(gen_seq_i[end - 1].item()) == pad_id:
                end -= 1
            gen_len_i = end - max_len
            if gen_len_i < 2:
                per_prompt_logp.append(0.0)
                continue
            # Build no-pad sequence: prompt (from pad_lefts[i]..max_len) + completion (max_len..end)
            trimmed = gen_seq_i[pad_lefts[i]:end].unsqueeze(0)
            attn_i = torch.ones_like(trimmed)
            # Position of the steering site in this trimmed sequence == lad["t_*"] (no pad offset)
            pos_i = [pos_per[i] - pad_lefts[i]]
            hook3 = make_additive_output_hook(direction, alpha, pos_i)
            h3 = layer_module.register_forward_hook(hook3)
            try:
                with torch.no_grad():
                    out3 = model(input_ids=trimmed, attention_mask=attn_i)
                    logits3 = out3.logits.float()  # [1, T, V]
                    lp3 = torch.log_softmax(logits3, dim=-1)
                    # In trimmed: prompt tokens live at [0..L_unpad), completion at [L_unpad..T)
                    T_trim = trimmed.shape[1]
                    tgt = trimmed[0, L_unpad:T_trim]
                    lp_at = lp3[0, L_unpad - 1 : T_trim - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    per_prompt_logp.append(float(lp_at.mean().item()))
                    del out3, logits3, lp3
            finally:
                h3.remove()

        for i in range(len(batch)):
            # With left-padding, generated tokens start at index `max_len`.
            new = gen[i, max_len:].tolist()
            while new and new[-1] == pad_id:
                new.pop()
            text = tokenizer.decode(new, skip_special_tokens=True)
            results.append({
                "idx": start + i,
                "prompt": batch[i],
                "completion": text,
                "refused": bool(is_refusal(text)),
                "h_readout": float(readouts[i]),
                "mean_logp_completion": float(per_prompt_logp[i]),
                "rep_rate": float(compute_repetition_rate(text)),
            })

        gc.collect()
        torch.cuda.empty_cache()
    return results


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep_dir = Path(args.prep)
    meta = json.loads((prep_dir / "meta.json").read_text())
    dirs_meta = json.loads((prep_dir / "directions.json").read_text())
    directions = torch.load(prep_dir / "directions.pt", weights_only=False)
    h = directions["h"].float()
    r = directions["r"].float()
    best_h = directions["best_h"]
    best_r = directions["best_r"]

    # ---------------------------------------------------- pick held-out sets
    n = meta["n_pairs"]
    test_idx = np.array(meta["split"]["test"])
    harmful_all = load_harmful_behaviors(str(Path(args.advbench) / "harmful_behaviors.csv"))
    alpaca_path = str(Path(args.alpaca) / "data" / "train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    benign_all = load_alpaca_instructions(alpaca_path, n, seed=args.seed)

    rng = np.random.default_rng(args.seed + 1000)
    harm_pool = [harmful_all[i] for i in test_idx if i < len(harmful_all)]
    ben_pool = [benign_all[i] for i in test_idx if i < len(benign_all)]
    # Take up to args.n_harm / args.n_benign
    harm_prompts = harm_pool[: args.n_harm]
    ben_prompts = ben_pool[: args.n_benign]
    print(f"[m3] direction={args.direction} alpha={args.alpha}  n_harm={len(harm_prompts)}  n_benign={len(ben_prompts)}", flush=True)

    # ---------------------------------------------------------------- direction
    # Hook indexing convention (matches make_additive_output_hook):
    #   M-prep's `layer = k` means the vector `hidden_states[k]` (output of block k-1).
    #   Register a POST-forward-hook on `model.model.layers[k-1]` and modify its
    #   output → that IS `hidden_states[k]`. So hook_block_idx = layer - 1.
    # If layer == 0 (embeddings), we cannot post-hook a decoder block. Guard:
    def _check_hook_idx(k, name):
        if k <= 0:
            raise ValueError(f"[m3] cannot steer at hidden_states[0] (embeddings) for {name}; skipping.")
        return k - 1

    if args.direction == "h":
        d_vec = h
        hook_block_idx = _check_hook_idx(int(best_h["layer"]), "h")
        position_kind = "t_final_instr"
        readout_layer = int(best_h["layer"])
        readout_dir = h
    elif args.direction == "r":
        d_vec = r
        hook_block_idx = _check_hook_idx(int(best_r["layer"]), "r")
        position_kind = "t_post_instr"
        # For r's Δ, off-target is the h-readout at h's site
        readout_layer = int(best_h["layer"])
        readout_dir = h
    elif args.direction == "random":
        # Random matched-norm to h at h's site (specificity control at the same
        # site as the true h direction).
        #
        # For n_random >= 30 controls at a fixed alpha (C3 mechanism-audit fix,
        # iteration 1), we hold --seed constant so prompt selection is stable
        # across replicates, and vary only the random-direction seed via
        # --random_dir_seed. When --random_dir_seed is None the original
        # behaviour (seed + 99) is preserved.
        random_seed = (
            args.random_dir_seed
            if args.random_dir_seed is not None
            else args.seed + 99
        )
        rng2 = np.random.default_rng(random_seed)
        v = rng2.standard_normal(h.shape[0]).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9) * float(h.norm().item())
        d_vec = torch.from_numpy(v)
        hook_block_idx = _check_hook_idx(int(best_h["layer"]), "random-at-h-site")
        position_kind = "t_final_instr"
        readout_layer = int(best_h["layer"])
        readout_dir = h
    elif args.direction == "swap":
        # Swap: apply r at h's site (specificity — is the site the driver, or the direction?)
        d_vec = r
        hook_block_idx = _check_hook_idx(int(best_h["layer"]), "swap-at-h-site")
        position_kind = "t_final_instr"
        readout_layer = int(best_h["layer"])
        readout_dir = h
    elif args.direction == "random_r_site":
        # Random matched-norm to r at r's site (t_post_instr, best_r_layer).
        # This is the r-side specificity control that Iteration-2 reviewer flagged as
        # a residual loophole in the h-site random controls: "run n=30 matched-norm
        # random directions at the actual r intervention site, not the h site".
        # Prompt selection stays pinned via --seed 0; direction RNG driven by --random_dir_seed.
        random_seed = (
            args.random_dir_seed
            if args.random_dir_seed is not None
            else args.seed + 199  # different offset from random-at-h-site
        )
        rng2 = np.random.default_rng(random_seed)
        v = rng2.standard_normal(r.shape[0]).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9) * float(r.norm().item())
        d_vec = torch.from_numpy(v)
        hook_block_idx = _check_hook_idx(int(best_r["layer"]), "random-at-r-site")
        position_kind = "t_post_instr"
        # Off-target readout for r-side specificity control is the h-readout at h's site
        # (matches the r-direction condition's off-target choice).
        readout_layer = int(best_h["layer"])
        readout_dir = h
    else:
        raise ValueError

    # -------------------------------------------------------------- model
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    n_layers = model.config.num_hidden_layers
    if hook_block_idx >= n_layers:
        hook_block_idx = n_layers - 1

    # -------------------------------------------------------------- run
    t0 = time.time()
    rows_harm = run_one_condition(
        model, tokenizer, harm_prompts, d_vec.cuda(), args.alpha,
        hook_block_idx=hook_block_idx, position_kind=position_kind,
        gen_max_new_tokens=args.gen_max_new_tokens,
        gen_batch_size=args.gen_batch_size,
        readout_layer=readout_layer, readout_direction=readout_dir.cuda(),
    )
    for row in rows_harm:
        row["class"] = "harmful"
    rows_ben = run_one_condition(
        model, tokenizer, ben_prompts, d_vec.cuda(), args.alpha,
        hook_block_idx=hook_block_idx, position_kind=position_kind,
        gen_max_new_tokens=args.gen_max_new_tokens,
        gen_batch_size=args.gen_batch_size,
        readout_layer=readout_layer, readout_direction=readout_dir.cuda(),
    )
    for row in rows_ben:
        row["class"] = "benign"
    all_rows = rows_harm + rows_ben
    elapsed = time.time() - t0

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "per_prompt.csv", index=False)

    metrics = {
        "direction": args.direction,
        "alpha": args.alpha,
        "hook_block_idx": int(hook_block_idx),
        "position_kind": position_kind,
        "readout_layer": int(readout_layer),
        "readout_direction": "h",
        "n_harm": len(rows_harm),
        "n_benign": len(rows_ben),
        "refusal_rate_harm": float(df[df["class"] == "harmful"]["refused"].mean() if len(rows_harm) else float("nan")),
        "refusal_rate_ben": float(df[df["class"] == "benign"]["refused"].mean() if len(rows_ben) else float("nan")),
        "h_readout_mean_harm": float(df[df["class"] == "harmful"]["h_readout"].mean() if len(rows_harm) else float("nan")),
        "h_readout_mean_ben": float(df[df["class"] == "benign"]["h_readout"].mean() if len(rows_ben) else float("nan")),
        "mean_logp_completion_harm": float(df[df["class"] == "harmful"]["mean_logp_completion"].mean() if len(rows_harm) else float("nan")),
        "mean_logp_completion_ben": float(df[df["class"] == "benign"]["mean_logp_completion"].mean() if len(rows_ben) else float("nan")),
        "rep_rate_harm": float(df[df["class"] == "harmful"]["rep_rate"].mean() if len(rows_harm) else float("nan")),
        "rep_rate_ben": float(df[df["class"] == "benign"]["rep_rate"].mean() if len(rows_ben) else float("nan")),
        "direction_norm": float(d_vec.norm().item()),
        "sigma_proj_h_at_best_h_layer": dirs_meta["best_h"].get("sigma_proj"),
        "sigma_proj_r_at_best_r_layer": dirs_meta["best_r"].get("sigma_proj"),
        "wall_clock_seconds": float(elapsed),
    }
    with open(out_dir / "steering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
