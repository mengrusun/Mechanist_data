"""Extract per-variable, per-layer difference-in-means directions from residual stream.

For each variable v in {gender, age, instruction, meeting}, pairs.json has K
paired prompts (identical context, only v differs). We compute the mean of
the last-token residual stream for the "value_a" prompts and for the "value_b"
prompts at each layer, take (mean_b - mean_a) as the raw direction, and save
per-layer directions.
"""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"


@torch.no_grad()
def collect_hidden(model, tokenizer, prompts, device, batch=32):
    """Return array (N, L+1, H) of last-token hidden states across layers."""
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    H = np.zeros((len(prompts), n_layers+1, hidden_size), dtype=np.float32)
    for start in range(0, len(prompts), batch):
        chunk = prompts[start:start+batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        for li, h in enumerate(out.hidden_states):
            H[start:start+len(chunk), li] = h[:, -1, :].float().cpu().numpy()
    return H


def main(model_path, pairs_path, out_dir, batch=32):
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    model.eval()

    pairs = json.load(open(pairs_path))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    dirs = {}
    for var, plist in pairs.items():
        print(f"Processing {var}, {len(plist)} pairs...")
        prompts_a = [p["prompt_a"] for p in plist]
        prompts_b = [p["prompt_b"] for p in plist]
        Ha = collect_hidden(model, tokenizer, prompts_a, device, batch)
        Hb = collect_hidden(model, tokenizer, prompts_b, device, batch)
        # direction = mean_b - mean_a  (value_b - value_a); e.g. male - female if a='male',b='female'?
        # In pairs.json vals[0] is 'a' side.  For gender: GENDERS=['male','female'] so a=male,b=female
        # We record ordered variable names so users know sign.
        d = Hb.mean(0) - Ha.mean(0)  # (L+1, H)
        # ALSO save an aligned pair-diff matrix for probing later
        pair_diff = Hb - Ha  # (N, L+1, H)
        # save per-layer L2 norms and per-layer aligned diffs
        val_a = plist[0]["value_a"]
        val_b = plist[0]["value_b"]
        print(f"  direction = mean(B: {val_b}) - mean(A: {val_a})")
        print(f"  per-layer norm min/mean/max: {np.linalg.norm(d,axis=-1).min():.3f}/{np.linalg.norm(d,axis=-1).mean():.3f}/{np.linalg.norm(d,axis=-1).max():.3f}")
        np.save(out_dir/f"dir_{var}.npy", d)
        # per-pair activation matrices for probing (Ha, Hb combined)
        np.save(out_dir/f"H_{var}_a.npy", Ha.astype(np.float16))
        np.save(out_dir/f"H_{var}_b.npy", Hb.astype(np.float16))
        dirs[var] = {"value_a": val_a, "value_b": val_b}

    with open(out_dir/"directions_meta.json", "w") as f:
        json.dump(dirs, f, indent=2)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--pairs", default=str(DATA/"pairs.json"))
    ap.add_argument("--out",   default=str(RES/"directions_llama"))
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    main(args.model, args.pairs, args.out, args.batch)
