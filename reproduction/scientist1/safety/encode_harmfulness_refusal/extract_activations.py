"""Extract residual-stream activations at t_inst and t_post positions from Llama-3-8B-Instruct.

For each prompt we save:
  hs_inst[l] = hidden state at layer l, position t_inst (-5)  (last user content token)
  hs_post[l] = hidden state at layer l, position t_post (-1)  (last prompt token, right before generation)

Layers are indexed 0..num_layers  (index 0 = embedding output; index i>0 = block i output).
Saved as float16 numpy arrays.
"""
import os, sys, time, argparse, gc
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    LLAMA3_PATH, load_model_and_tokenizer, format_llama3_chat,
    load_advbench, load_alpaca_benign, load_jbb_harmful, load_jbb_benign,
    load_catqa_english, load_sorrybench, load_xstest_prompts,
    T_INST, T_POST,
)


@torch.no_grad()
def extract_activations(model, tok, prompts, batch_size=8, device="cuda"):
    """Return two float16 arrays of shape (N, L+1, D): hs_inst, hs_post."""
    L = model.config.num_hidden_layers  # blocks; hidden_states outputs L+1
    D = model.config.hidden_size
    N = len(prompts)
    hs_inst = np.empty((N, L + 1, D), dtype=np.float16)
    hs_post = np.empty((N, L + 1, D), dtype=np.float16)

    for i in range(0, N, batch_size):
        batch = prompts[i:i + batch_size]
        formatted = [format_llama3_chat(tok, p) for p in batch]
        enc = tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)
        # We use left padding so t_inst = -5, t_post = -1 across the batch.
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # out.hidden_states: tuple of L+1 tensors, each (B, T, D)
        for l in range(L + 1):
            h = out.hidden_states[l]
            hs_inst[i:i + len(batch), l] = h[:, T_INST].float().cpu().numpy().astype(np.float16)
            hs_post[i:i + len(batch), l] = h[:, T_POST].float().cpu().numpy().astype(np.float16)
        if (i // batch_size) % 10 == 0:
            print(f"  batch {i//batch_size} / {N//batch_size}  ({(i+len(batch))}/{N})", flush=True)
        del out
    return hs_inst, hs_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/acts_llama3")
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Llama-3-8B-Instruct ...", flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(LLAMA3_PATH)
    print(f"  loaded in {time.time()-t0:.1f}s. num_layers={model.config.num_hidden_layers}", flush=True)
    device = next(model.parameters()).device

    tasks = {
        # train contrast set (for computing directions)
        "advbench_train": load_advbench(n=args.n_train, seed=0),
        "alpaca_train":   load_alpaca_benign(n=args.n_train, seed=0),
        # held-out test sets
        "advbench_test":  load_advbench(n=args.n_test, seed=1),
        "alpaca_test":    load_alpaca_benign(n=args.n_test, seed=1),
        "jbb_harmful":    load_jbb_harmful(n=100, seed=0),
        "jbb_benign":     load_jbb_benign(n=100, seed=0),
        "catqa":          load_catqa_english(n=200, seed=0),
        "sorrybench":     load_sorrybench(n=200, seed=0),
    }
    # For robustness we skip already-computed files
    for name, prompts in tasks.items():
        fp = os.path.join(args.out_dir, f"{name}.npz")
        if os.path.exists(fp):
            print(f"[skip] {name} already exists at {fp}", flush=True)
            continue
        print(f"[extract] {name}  n={len(prompts)}", flush=True)
        t0 = time.time()
        hs_inst, hs_post = extract_activations(model, tok, prompts, batch_size=args.batch_size, device=device)
        np.savez(fp, hs_inst=hs_inst, hs_post=hs_post, prompts=np.array(prompts, dtype=object))
        print(f"  saved {fp}  ({time.time()-t0:.1f}s)", flush=True)

    # save train contrast filenames for downstream
    manifest = {k: os.path.join(args.out_dir, f"{k}.npz") for k in tasks}
    import json as _json
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        _json.dump(manifest, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
