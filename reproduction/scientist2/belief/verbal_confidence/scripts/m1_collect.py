#!/usr/bin/env python3
"""
M1 — Data preparation, activation caching, and verbal-confidence elicitation.

For each TriviaQA item:
  1. Greedily decode an answer from prompt "Q: <q>\\nA: "
  2. Append fixed post-answer template "\\nConfidence (0-100): "
  3. Greedily decode a confidence integer
  4. Extract residual-stream activations at layers {5,10,15,...,60} × positions
     {E0..E4, C0 (conf-gen)} — one forward pass per item
  5. Record answer correctness against TriviaQA gold aliases + per-answer-token log-probs
  6. Persist:
       items.jsonl        — per-item record (question, answer, verbal_conf,
                             is_correct, answer_token_logprobs, positions, conf_gen_position)
       activations.h5     — activations[position_label][layer] = (N, hidden_size) float32
       confidence.jsonl   — one row per item with just {question_id, verbal_conf, is_correct}

Grid: paraphrase_template = T0 (fixed for downstream); seeds control item sampling only.

Success criteria (self-checks logged at end):
  - verbal-confidence parse rate ≥ 90%
  - std(verbal_conf) ≥ 5 on the 0-100 scale
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc_common import (
    CONF_GEN_LABEL,
    LAYER_GRID,
    MODEL_PATH,
    extract_residual_activations,
    generate_answer_and_confidence,
    load_model_and_tokenizer,
    sample_triviaqa,
    set_all_seeds,
    write_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--dataset", default="/data/zhenqian/data/trivia_qa")
    p.add_argument("--split", default="validation")
    p.add_argument("--template", default="T0", help="paraphrase template id (T0 fixed)")
    p.add_argument("--n_items", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", default=",".join(str(x) for x in LAYER_GRID),
                   help="Comma-separated 1-indexed layer indices")
    p.add_argument("--post_answer_window", type=int, default=5,
                   help="Number of post-answer positions to cache (E0..E4).")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    layers = tuple(int(x) for x in args.layers.split(","))
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]

    print(f"[m1] Loading model {args.model} (dtype={args.dtype})...", flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(args.model, dtype=dtype)
    device = next(model.parameters()).device
    print(f"[m1] Model loaded in {time.time()-t0:.1f}s on {device}", flush=True)

    print(f"[m1] Sampling {args.n_items} TriviaQA items (seed={args.seed})...", flush=True)
    items = sample_triviaqa(args.n_items, args.seed)

    n_layers = len(layers)
    # Gemma3ForCausalLM.config is Gemma3TextConfig directly (no vision wrapper).
    hidden = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    n_positions = args.post_answer_window + 1  # E0..E4 + C0
    print(
        f"[m1] Will cache {args.n_items} items × {n_positions} positions × "
        f"{n_layers} layers × {hidden} hidden = "
        f"{args.n_items * n_positions * n_layers * hidden * 4 / 1e9:.2f} GB float32.",
        flush=True,
    )

    # ---- One-pass over the data set --------------------------------------
    labels_all = ["E0", "E1", "E2", "E3", "E4"][: args.post_answer_window] + [CONF_GEN_LABEL]
    activations = {
        pos: {L: np.zeros((args.n_items, hidden), dtype=np.float32) for L in layers}
        for pos in labels_all
    }

    item_records = []
    conf_records = []

    for i, itm in enumerate(items):
        trace = generate_answer_and_confidence(model, tok, itm, device)

        # Extract residual activations in TWO forward passes:
        # (1) On tokens up-to-and-including the last post-answer template token
        #     (i.e. full_ids[:conf_gen_position]). The residual at position
        #     `conf_gen_position-1` in this forward IS the state that generates
        #     the first confidence token (C0) — this is the correct C0 anchor.
        #     Simultaneously read E0..E4 positions from this same forward.
        pre_conf_ids = torch.tensor([trace.full_ids[: trace.conf_gen_position]], device=device)
        # C0's target position in this forward is (conf_gen_position - 1)
        positions_by_label = {}
        for name in ("E0", "E1", "E2", "E3", "E4")[: args.post_answer_window]:
            positions_by_label[name] = trace.post_answer_positions[name]
        positions_by_label[CONF_GEN_LABEL] = trace.conf_gen_position - 1  # last-token position

        acts = extract_residual_activations(model, pre_conf_ids, positions_by_label, layers=layers)

        for pos_label, layer_dict in acts.items():
            for L, vec in layer_dict.items():
                activations[pos_label][L][i] = vec

        item_records.append(
            {
                "idx": i,
                "question_id": trace.question_id,
                "question": trace.question,
                "answer_text": trace.answer_text,
                "verbal_conf": trace.verbal_conf,
                "is_correct": trace.is_correct,
                "answer_token_logprobs": trace.answer_token_logprobs,
                "prefix_len": trace.prefix_len,
                "n_answer_tokens": trace.n_answer_tokens,
                "post_answer_positions": trace.post_answer_positions,
                "conf_gen_position": trace.conf_gen_position,
                "full_ids": trace.full_ids,
                "aliases": trace.aliases,
                "normalized_aliases": trace.normalized_aliases,
            }
        )
        conf_records.append(
            {
                "idx": i,
                "question_id": trace.question_id,
                "verbal_conf": trace.verbal_conf,
                "is_correct": trace.is_correct,
                "answer_text": trace.answer_text,
                "mean_answer_logprob": (
                    float(np.mean(trace.answer_token_logprobs))
                    if trace.answer_token_logprobs
                    else float("nan")
                ),
            }
        )

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (args.n_items - i - 1)
            print(
                f"[m1] item {i+1}/{args.n_items} "
                f"answer={trace.answer_text[:40]!r} conf={trace.verbal_conf} "
                f"correct={trace.is_correct} elapsed={elapsed/60:.1f}m rem~{remaining/60:.1f}m",
                flush=True,
            )

    # ---- Persist ---------------------------------------------------------
    print(f"[m1] Writing items.jsonl and confidence.jsonl to {cache_dir}", flush=True)
    write_jsonl(cache_dir / "items.jsonl", item_records)
    write_jsonl(cache_dir / "confidence.jsonl", conf_records)

    print(f"[m1] Writing activations.h5 to {cache_dir}", flush=True)
    with h5py.File(cache_dir / "activations.h5", "w") as f:
        f.attrs["n_items"] = args.n_items
        f.attrs["hidden_size"] = hidden
        f.attrs["layers"] = list(layers)
        f.attrs["position_labels"] = list(labels_all)
        for pos_label in labels_all:
            grp = f.create_group(pos_label)
            for L in layers:
                grp.create_dataset(str(L), data=activations[pos_label][L], compression="gzip",
                                   compression_opts=1)

    # ---- Self-check ------------------------------------------------------
    confs = [r["verbal_conf"] for r in conf_records]
    parsed = [c for c in confs if c is not None]
    parse_rate = len(parsed) / len(confs) if confs else 0.0
    conf_std = float(np.std(parsed)) if parsed else 0.0
    correct_flags = [r["is_correct"] for r in conf_records if r["is_correct"] is not None]
    acc = float(np.mean(correct_flags)) if correct_flags else 0.0
    print(
        f"[m1] SELF-CHECK parse_rate={parse_rate:.3f} conf_std={conf_std:.2f} acc={acc:.3f}",
        flush=True,
    )
    with (cache_dir / "self_check.json").open("w") as f:
        json.dump(
            {
                "n_items": args.n_items,
                "parse_rate": parse_rate,
                "conf_std": conf_std,
                "conf_mean": float(np.mean(parsed)) if parsed else 0.0,
                "answer_accuracy": acc,
                "seed": args.seed,
                "template": args.template,
                "criteria_pass": parse_rate >= 0.9 and conf_std >= 5.0,
            },
            f,
            indent=2,
        )

    print(f"[m1] DONE. total elapsed={ (time.time()-t0)/60 :.1f}m", flush=True)


if __name__ == "__main__":
    main()
