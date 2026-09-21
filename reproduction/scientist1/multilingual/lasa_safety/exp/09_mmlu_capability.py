"""
MMLU capability retention under residual-stream steering.

For each condition (baseline / bottleneck-steer / surface-steer), score a
random subset of MMLU test questions. For each item we look at the logits
over the tokens "A", "B", "C", "D" at the "Answer:" position and pick argmax.
"""

import os, argparse, json, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATE = (
    "The following is a multiple choice question. Answer with a single letter A, B, C or D.\n\n"
    "Question: {q}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Answer:"
)


def build_ids(tokenizer, item, device):
    p = PROMPT_TEMPLATE.format(
        q=item["question"], a=item["choices"][0], b=item["choices"][1],
        c=item["choices"][2], d=item["choices"][3],
    )
    msgs = [{"role": "user", "content": p}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(device)


class ResidualSteer:
    def __init__(self, model, layer, v, alpha):
        self.model = model; self.layer = layer; self.v = v; self.alpha = alpha
        self.handle = None
    def __enter__(self):
        mod = self.model.model.layers[self.layer]
        v_local = self.v.to(next(mod.parameters()).dtype)
        alpha = self.alpha
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                h = h + alpha * v_local.view(1, 1, -1)
                return (h,) + output[1:]
            return output + alpha * v_local.view(1, 1, -1)
        self.handle = mod.register_forward_hook(hook)
        return self
    def __exit__(self, *a):
        if self.handle is not None: self.handle.remove()

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--refusal_dir", default="cache/refusal_dir.npz")
    ap.add_argument("--mmlu", default="/data/zhenqian/data/mmlu/all/test-00000-of-00001.parquet")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", type=str, default="baseline:-1:0,bottleneck_L14_a5:14:5,surface_L28_a5:28:5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()

    d = np.load(args.refusal_dir); v_unit = d["v_unit"]

    df = pd.read_parquet(args.mmlu)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=args.n, replace=False)
    items = df.iloc[idx].to_dict("records")

    # Precompute the token ids for A, B, C, D (Llama tokenizer: check both forms)
    letter_ids = []
    for lt in ("A", "B", "C", "D"):
        toks = tok(" " + lt, add_special_tokens=False)["input_ids"]
        letter_ids.append(toks[-1])
    print(f"[info] letter token ids = {letter_ids}")

    results = []
    for c in args.conditions.split(","):
        name, layer, alpha = c.split(":")
        layer, alpha = int(layer), float(alpha)
        cm_factory = None
        if layer >= 0 and alpha != 0.0:
            v = torch.from_numpy(v_unit[layer + 1].astype(np.float32)).to(device)
            cm_factory = lambda: ResidualSteer(model, layer, v, alpha)
        correct = 0
        n = 0
        letter_dist = np.zeros(4)
        for it in items:
            ids = build_ids(tok, it, device)
            cm = cm_factory() if cm_factory else _NullCtx()
            with torch.inference_mode(), cm:
                out = model(input_ids=ids, use_cache=False)
            logits = out.logits[0, -1]                # [vocab]
            letter_logits = torch.tensor([logits[i].item() for i in letter_ids])
            pred = int(letter_logits.argmax().item())
            letter_dist[pred] += 1
            if pred == int(it["answer"]): correct += 1
            n += 1
        acc = correct / n
        letter_dist /= n
        print(f"[cond={name}] MMLU acc={acc:.3f} (n={n})  letter_dist={letter_dist.round(3).tolist()}")
        results.append({"name": name, "layer": layer, "alpha": alpha, "acc": acc,
                        "n": n, "letter_dist": letter_dist.tolist()})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()
