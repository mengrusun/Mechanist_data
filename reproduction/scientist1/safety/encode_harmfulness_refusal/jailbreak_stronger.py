"""Stronger jailbreak attack set that actually yields non-refusal outputs on Llama-3-8B-Instruct.

Attacks:
  1. jbb_role   — role-play persuasion prompts from JailbreakBench judge-comparison.csv
                  (successful on Vicuna; some fraction succeed on Llama-3)
  2. prefill    — force assistant response to begin with "Sure, here is"
                  (implemented by continuing the chat template past the header)

For each attacked prompt we record hidden states at (all layers, T_INST) and (all layers, T_POST_effective).

For the prefill attack we set T_POST = -1 which now points to the last token of the injected
prefix (thus reflecting the model's state at the moment it is committing to comply).
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    LLAMA3_PATH, load_model_and_tokenizer, format_llama3_chat, is_refusal,
    load_advbench, T_INST, T_POST,
)


PREFILL = "Sure, here is"


def format_llama3_prefill(tok, user_msg, prefill=PREFILL):
    """Chat template with an assistant prefix that already contains `prefill`."""
    msgs = [{"role": "user", "content": user_msg}]
    base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return base + prefill  # the assistant response starts with `prefill`


@torch.no_grad()
def extract_and_generate(model, tok, prompts, batch_size=8, max_new_tokens=96, device="cuda",
                         formatter=None, t_inst_offset=T_INST, t_post_offset=T_POST):
    """Similar to jailbreak_attacks but with configurable formatter and positions."""
    if formatter is None:
        formatter = format_llama3_chat
    L = model.config.num_hidden_layers
    D = model.config.hidden_size
    N = len(prompts)
    hs_inst = np.empty((N, L + 1, D), dtype=np.float16)
    hs_post = np.empty((N, L + 1, D), dtype=np.float16)
    outputs = []
    for i in range(0, N, batch_size):
        batch = prompts[i:i+batch_size]
        formatted = [formatter(tok, p) for p in batch]
        enc = tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out_fwd = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                        output_hidden_states=True, use_cache=False)
        for l in range(L + 1):
            h = out_fwd.hidden_states[l]
            hs_inst[i:i+len(batch), l] = h[:, t_inst_offset].float().cpu().numpy().astype(np.float16)
            hs_post[i:i+len(batch), l] = h[:, t_post_offset].float().cpu().numpy().astype(np.float16)
        del out_fwd
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=1.0, pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            in_len = enc.input_ids.shape[1]
            outputs.append(tok.decode(gen[j, in_len:], skip_special_tokens=True))
        if (i // batch_size) % 5 == 0:
            print(f"  batch {i//batch_size} / {N//batch_size}", flush=True)
    return hs_inst, hs_post, outputs


def load_jbb_prompts():
    """Load JBB judge-comparison prompts and their harmful goals."""
    df = pd.read_csv("/data/zhenqian/data/JailbreakBench/data/judge-comparison.csv")
    # Use only unique (goal, prompt) rows
    df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    goals = df["goal"].tolist()
    prompts = df["prompt"].tolist()
    return goals, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/jailbreak_llama3")
    ap.add_argument("--n_jbb", type=int, default=120)
    ap.add_argument("--n_prefill", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Llama-3-8B-Instruct ...", flush=True)
    model, tok = load_model_and_tokenizer(LLAMA3_PATH)
    device = next(model.parameters()).device

    # --- JBB role-play prompts ---
    fp = os.path.join(args.out_dir, "jbb_role.npz")
    if not os.path.exists(fp):
        goals, prompts = load_jbb_prompts()
        prompts = prompts[:args.n_jbb]; goals = goals[:args.n_jbb]
        print(f"[attack] jbb_role  n={len(prompts)}", flush=True)
        t0 = time.time()
        hs_i, hs_p, outs = extract_and_generate(model, tok, prompts, batch_size=args.batch_size,
                                                 max_new_tokens=args.max_new_tokens, device=device)
        ref = np.array([is_refusal(o) for o in outs], dtype=bool)
        np.savez(fp, hs_inst=hs_i, hs_post=hs_p,
                 base_prompts=np.array(goals, dtype=object),
                 attack_prompts=np.array(prompts, dtype=object),
                 outputs=np.array(outs, dtype=object),
                 refusal=ref)
        print(f"  refusal rate: {ref.mean():.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # --- Prefill attack ---
    fp = os.path.join(args.out_dir, "prefill.npz")
    if not os.path.exists(fp):
        base = load_advbench(n=args.n_prefill, seed=2)
        # For prefill, t_inst is the last token of user content, which is offset a few positions before
        # the prefill text.  With PREFILL = "Sure, here is" tokenized to ~4 tokens, t_inst = -(5+4) = -9
        # Let's tokenize dynamically to find the actual position.
        prefill_len = len(tok(PREFILL, add_special_tokens=False).input_ids)
        t_inst = T_INST - prefill_len  # last user content token
        t_post = T_POST  # last token of the prefilled input (last token of PREFILL)
        print(f"[attack] prefill  n={len(base)}  (t_inst={t_inst}, t_post={t_post})", flush=True)
        t0 = time.time()
        hs_i, hs_p, outs = extract_and_generate(model, tok, base, batch_size=args.batch_size,
                                                 max_new_tokens=args.max_new_tokens, device=device,
                                                 formatter=format_llama3_prefill,
                                                 t_inst_offset=t_inst, t_post_offset=t_post)
        # For prefill, the assistant response starts with "Sure, here is" so regex refusal detection needs adjustment.
        # We check if refusal indicators appear anywhere in the full output.
        full_outs = [PREFILL + o for o in outs]
        ref = np.array([is_refusal(o) for o in full_outs], dtype=bool)
        np.savez(fp, hs_inst=hs_i, hs_post=hs_p,
                 base_prompts=np.array(base, dtype=object),
                 attack_prompts=np.array([format_llama3_prefill(tok, p) for p in base], dtype=object),
                 outputs=np.array(full_outs, dtype=object),
                 refusal=ref)
        print(f"  refusal rate: {ref.mean():.3f}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
