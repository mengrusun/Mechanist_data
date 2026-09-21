"""Step 2 — HF forward-pass hidden-state extraction.

For each mode (turn1, turn2, turn2_p1, turn2_p2, single), tokenize the prompt for
each row, run one forward pass with `output_hidden_states=True`, and record the
LAST-INPUT-TOKEN hidden state at every layer L ∈ {0..num_layers}. Save as
a compressed npz array of shape (N, num_layers+1, hidden_dim) in float16.

No generation is done here — the prompt is fully deterministic and we only need
the pre-emission residual state.

Args:
  --mode: turn1 | turn2 | turn2_p1 | turn2_p2 | single (single value)
  --batch-size: forward pass batch size
  --max-length: truncate prompts to this length in tokens
  --resume: if the output .npz already exists, skip
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U


def _mk_prompt(mode: str, row_gen: dict, row_meta: dict) -> str:
    """For each mode, reconstruct the exact prompt that was used at generation time.
    Note: turn1 / single depend on the question only; turn2 / turn2_p1 / turn2_p2 need the
    model answer from turn1 (stored in the generation JSONL)."""
    q = row_meta["question"]
    if mode == "turn1":
        return U.prompt_turn1(q)
    if mode == "single":
        return U.prompt_single_pass(q)
    # For turn2 variants, the row_gen is a turn2_gen row (has no "short_answer");
    # we look up the model answer via a passed dict.
    raise ValueError("Use main() to build turn2 prompts (needs turn1 map)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True,
                    choices=["turn1", "turn2", "turn2_p1", "turn2_p2", "single"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dtype", type=str, default="float16")
    args = ap.parse_args()

    out_path = U.ARTIFACT_DIR / f"H_{args.mode}.npz"
    if args.resume and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[step2][resume] {args.mode} — {out_path} already exists, skipping", flush=True)
        return

    # Load generation rows
    gen_path = U.ARTIFACT_DIR / f"{args.mode}_gen.jsonl"
    assert gen_path.exists(), f"missing {gen_path}; run step1_generate.py first"
    gen_rows = U.load_jsonl(gen_path)
    print(f"[step2] {args.mode}: {len(gen_rows)} rows", flush=True)

    # For turn2 variants we need turn1 short_answer per question_id
    turn1_by_qid = {}
    if args.mode in ("turn2", "turn2_p1", "turn2_p2"):
        turn1_rows = U.load_jsonl(U.ARTIFACT_DIR / "turn1_gen.jsonl")
        turn1_by_qid = {r["question_id"]: r for r in turn1_rows}

    # Also need question-text lookup for turn2 (only has qid + parsed c)
    from datasets import Dataset
    ds_raw = Dataset.from_file(U.TRIVIAQA_ARROW)
    qid_to_question = {}
    for ex in ds_raw:
        qid_to_question[ex["question_id"]] = ex["question"]

    # Build prompts (deterministic per row)
    prompts = []
    kept_rows = []
    for r in gen_rows:
        qid = r["question_id"]
        question = r.get("question") or qid_to_question.get(qid)
        if question is None:
            continue
        if args.mode == "turn1":
            p = U.prompt_turn1(question)
        elif args.mode == "single":
            p = U.prompt_single_pass(question)
        elif args.mode in ("turn2", "turn2_p1", "turn2_p2"):
            t1 = turn1_by_qid.get(qid)
            if t1 is None:
                continue
            model_ans = t1.get("short_answer") or t1.get("gen_raw", "")
            fn = {
                "turn2": U.prompt_turn2,
                "turn2_p1": U.prompt_turn2_p1,
                "turn2_p2": U.prompt_turn2_p2,
            }[args.mode]
            p = fn(question, model_ans)
        prompts.append(p)
        kept_rows.append(r)
    print(f"[step2] {len(prompts)} prompts built", flush=True)

    # Load HF model + tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[step2] Loading HF model from {U.MODEL_PATH}", flush=True)
    tok = AutoTokenizer.from_pretrained(U.MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # last-input-token comes from the right
    model = AutoModelForCausalLM.from_pretrained(
        U.MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers  # 32 for Llama-3.1-8B
    hidden_dim = model.config.hidden_size  # 4096
    print(f"[step2] num_layers={num_layers}  hidden_dim={hidden_dim}", flush=True)

    # Allocate output tensor on CPU (N, L+1, D) in float16 — for turn1 this is
    # 10000 * 33 * 4096 * 2 = ~2.5 GB.
    N = len(prompts)
    H = np.zeros((N, num_layers + 1, hidden_dim), dtype=np.float16)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            batch_prompts = prompts[i : i + args.batch_size]
            enc = tok(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(model.device)
            out = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            # out.hidden_states is a tuple length (num_layers + 1) —
            #   hidden_states[0] = embedding output (post-embed_tokens)
            #   hidden_states[L] for L in 1..num_layers = OUTPUT of block L-1
            # Each element is (B, seq_len, hidden_dim). We LEFT-pad, so the
            # last real input token IS at position -1 (i.e. S-1) in every row,
            # regardless of that row's real token count. Verify this by checking
            # that the right-most attention_mask value is 1 for every row (i.e.
            # no right-side padding).
            attn = enc.attention_mask  # (B, S)
            B_batch, S = attn.shape
            # Sanity: with left-padding, every row's LAST position must be a real token.
            assert torch.all(attn[:, -1] == 1), \
                f"Left padding invariant violated: attn[:, -1] = {attn[:, -1].tolist()} (expected all 1s)"
            # last-input-token index for every row is simply S-1.
            last_pos = torch.full((B_batch,), S - 1, dtype=torch.long, device=attn.device)
            for li, hs in enumerate(out.hidden_states):
                # Take the last-input-token via explicit position gather:
                #   hs[b, last_pos[b], :] for each b
                B_ = hs.shape[0]
                pos = last_pos.to(hs.device)
                gather_idx = pos.view(B_, 1, 1).expand(B_, 1, hs.shape[-1])
                vec = hs.gather(1, gather_idx).squeeze(1)  # (B, hidden_dim)
                H[i : i + B_, li, :] = vec.to(torch.float16).cpu().numpy()
            if (i // args.batch_size) % 20 == 0:
                elapsed = time.time() - t0
                rate = (i + hs.shape[0]) / max(elapsed, 1e-6)
                eta = (N - i - hs.shape[0]) / max(rate, 1e-6)
                print(f"[step2][{args.mode}] {i+hs.shape[0]}/{N}  {rate:.1f} it/s  ETA {eta/60:.1f} min", flush=True)

    print(f"[step2] extraction done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Save
    question_ids = np.array([r["question_id"] for r in kept_rows])
    idxs = np.array([r["idx"] for r in kept_rows])
    np.savez_compressed(out_path, H=H, question_ids=question_ids, idxs=idxs)
    print(f"[step2] saved {out_path} shape={H.shape}", flush=True)

    del model, H
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
