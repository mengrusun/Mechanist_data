"""Run 1000 dictator trials on Llama-3.1-8B-Instruct and record E[transfer].

Decision extraction: because the prompt ends in "Transfer amount: $", the next
token position holds the decision. We enumerate integer strings "0".."20" and
compute their probabilities from the softmax over the model vocab (multi-token
integers require summing the joint prob of the multi-token sequence). We then
report the argmax integer AND the expected value E[transfer] under this
distribution -- E[transfer] is smoother and better for regression.
"""

import argparse, json, os, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def integer_token_paths(tokenizer):
    """Return a dict: int -> list of token id sequences that would appear
    immediately after "$" (no leading space)."""
    out = {}
    for n in range(0, 21):
        s = str(n)
        ids = tokenizer.encode(s, add_special_tokens=False)
        out[n] = ids
    return out


@torch.no_grad()
def decision_distribution(model, tokenizer, prompt_ids, int_paths, device):
    """Given a batch of prompt_ids (tensor 1xT), return a dict int->prob (normalised)."""
    # forward pass on prompt
    input_ids = prompt_ids.to(device)
    out = model(input_ids=input_ids, use_cache=True)
    logits = out.logits[0, -1]  # (V,)
    past_kv = out.past_key_values
    log_probs_1 = torch.log_softmax(logits, dim=-1)  # base for first token

    probs = {}
    for n, ids in int_paths.items():
        # Compute log prob of ids[0], then continue with ids[1:] if needed
        lp = log_probs_1[ids[0]].item()
        if len(ids) > 1:
            # continue generation
            cur_ids = torch.tensor([[ids[0]]], device=device)
            cur_kv = past_kv
            for t in range(1, len(ids)):
                o = model(input_ids=cur_ids, past_key_values=cur_kv, use_cache=True)
                lp += torch.log_softmax(o.logits[0, -1], dim=-1)[ids[t]].item()
                cur_kv = o.past_key_values
                cur_ids = torch.tensor([[ids[t]]], device=device)
        probs[n] = lp
    # normalise across n
    lps = np.array([probs[n] for n in range(21)])
    lps -= lps.max()
    ps = np.exp(lps)
    ps /= ps.sum()
    return ps


def run(model_path, trials_path, out_path, limit=None):
    device = "cuda"
    print(f"Loading {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    model.eval()
    int_paths = integer_token_paths(tokenizer)
    print("Integer token paths (len 1 = single token):",
          {n: len(v) for n, v in int_paths.items()})

    trials = [json.loads(l) for l in open(trials_path)]
    if limit:
        trials = trials[:limit]

    results = []
    for i, t in enumerate(trials):
        ids = tokenizer(t["prompt"], return_tensors="pt").input_ids
        ps = decision_distribution(model, tokenizer, ids, int_paths, device)
        e_transfer = float(np.dot(np.arange(21), ps))
        argmax = int(np.argmax(ps))
        results.append({
            "id": t["id"],
            "gender": t["gender"], "age": t["age"],
            "instruction": t["instruction"], "meeting": t["meeting"],
            "argmax": argmax, "E_transfer": e_transfer,
            "p": ps.tolist(),
        })
        if (i+1) % 50 == 0:
            print(f"[{i+1}/{len(trials)}] gender={t['gender']} age={t['age']} instr={t['instruction']} meet={t['meeting']} -> argmax={argmax} E={e_transfer:.2f}")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--trials", default=str(DATA/"trials.jsonl"))
    ap.add_argument("--out", default=str(RES/"baseline_llama.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.model, args.trials, args.out, args.limit)
