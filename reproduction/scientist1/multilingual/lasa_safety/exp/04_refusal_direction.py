"""
Extract per-layer "refusal direction" for LLaMA-3.1-8B-Instruct.

  v_L = mean_{harmful English prompts}( h_L(last user tok) )
      - mean_{benign  English prompts}( h_L(last user tok) )

The last-token hidden state after the user message ends is a common choice for
refusal direction (see Arditi et al. 2024). Steering with this vector along
the residual stream increases refusal likelihood.

Saves an [n_layers+1, hidden] .npz of raw and unit-norm vectors.
"""

import os, argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json


def read_mmlu_benign_en(n):
    path = "/data/zhenqian/data/m_mmlu/data/en/test.jsonl"
    rows = []
    with open(path) as f:
        for i, ln in enumerate(f):
            if len(rows) >= n: break
            r = json.loads(ln)
            rows.append(r["instruction"])
    return rows


def build_chat_last_tok_index(tokenizer, user_text, device):
    msgs = [{"role": "user", "content": user_text}]
    ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--multijail", default="/data/zhenqian/data/multijail/MultiJail.csv")
    ap.add_argument("--n_harm", type=int, default=100)
    ap.add_argument("--n_benign", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source_lang", default="en")
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    L = model.config.num_hidden_layers
    H = model.config.hidden_size

    df = pd.read_csv(args.multijail)
    harmful_prompts = df[args.source_lang].dropna().head(args.n_harm).tolist()
    benign_prompts  = read_mmlu_benign_en(args.n_benign)
    print(f"[data] harmful={len(harmful_prompts)}, benign={len(benign_prompts)}")

    def mean_last_tok(prompts):
        acc = np.zeros((L + 1, H), dtype=np.float32)
        n = 0
        with torch.inference_mode():
            for p in tqdm(prompts, desc="prompts"):
                if not isinstance(p, str) or not p.strip(): continue
                ids = build_chat_last_tok_index(tok, p, device)
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
                for l, h in enumerate(out.hidden_states):
                    acc[l] += h[0, -1].to(torch.float32).cpu().numpy()
                n += 1
        return acc / max(n, 1)

    print("[compute] harmful mean")
    mu_harm = mean_last_tok(harmful_prompts)
    print("[compute] benign mean")
    mu_benign = mean_last_tok(benign_prompts)

    v = mu_harm - mu_benign                       # [L+1, H]
    v_unit = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    np.savez_compressed(args.out, v=v, v_unit=v_unit, mu_harm=mu_harm, mu_benign=mu_benign)
    print(f"[save] {args.out}, norms per layer:")
    for l in range(0, L + 1, 4):
        print(f"  layer {l:>2}: ||v||={np.linalg.norm(v[l]):.3f}")

if __name__ == "__main__":
    main()
