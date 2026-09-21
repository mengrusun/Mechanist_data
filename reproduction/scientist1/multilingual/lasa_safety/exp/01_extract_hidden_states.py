"""
Extract per-layer hidden states from LLaMA-3.1-8B-Instruct for MultiJail parallel prompts.

For each layer l in {0,...,32}, and each (prompt_id, language) pair we store one
mean-pooled hidden vector (over user-turn tokens) of shape [hidden_dim].
"""

import os, sys, argparse, time, csv, json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def build_chat_prompt(tokenizer, user_text):
    msgs = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--csv",   default="/data/zhenqian/data/multijail/MultiJail.csv")
    ap.add_argument("--out",   required=True)
    ap.add_argument("--n_prompts", type=int, default=150)
    ap.add_argument("--langs", nargs="+",
                    default=["en","zh","it","vi","ar","ko","th","bn","sw","jv"])
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)

    print(f"[load] tokenizer + model @ {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"[info] num_hidden_layers={n_layers}, hidden={hidden}", flush=True)

    df = pd.read_csv(args.csv)
    df = df.head(args.n_prompts).reset_index(drop=True)
    print(f"[data] prompts={len(df)}, langs={args.langs}", flush=True)

    # Storage: for each layer l in [0, n_layers], one array [N_prompts, N_langs, hidden]
    # (l=0 is the embedding output; l>=1 are transformer block outputs.)
    all_layers = n_layers + 1
    reps = np.zeros((all_layers, len(df), len(args.langs), hidden), dtype=np.float16)

    lang_labels = []
    prompt_ids = []
    for i, row in df.iterrows():
        for j, lang in enumerate(args.langs):
            lang_labels.append(j)
            prompt_ids.append(i)

    # We compute per (prompt, lang): mean of hidden states over tokens (excluding padding).
    with torch.inference_mode():
        for i, row in tqdm(df.iterrows(), total=len(df), desc="prompts"):
            for j, lang in enumerate(args.langs):
                text = row[lang]
                if not isinstance(text, str) or len(text.strip()) == 0:
                    continue
                chat = build_chat_prompt(tok, text)
                enc = tok(chat, return_tensors="pt", truncation=True, max_length=1024).to(device)
                out = model(**enc, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states  # tuple of (n_layers+1) tensors, each [1, T, H]
                attn = enc["attention_mask"][0].to(torch.float32)
                denom = attn.sum().clamp(min=1.0)
                for l, h in enumerate(hs):
                    v = (h[0].to(torch.float32) * attn.unsqueeze(-1)).sum(0) / denom
                    reps[l, i, j] = v.cpu().numpy()

    print(f"[save] {args.out}", flush=True)
    np.savez_compressed(
        args.out,
        reps=reps,                     # [L+1, N, K, H]
        prompt_ids=np.asarray(prompt_ids),
        lang_labels=np.asarray(lang_labels),
        langs=np.asarray(args.langs),
    )
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
