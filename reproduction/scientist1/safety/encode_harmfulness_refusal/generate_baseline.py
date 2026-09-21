"""Generate completions for benchmark prompts with Llama-3-8B-Instruct.

We also do a light 'is this a refusal' string check on each output.
Used later for computing refusal rates (Claim 3 / Claim 4).
"""
import os, sys, json, argparse, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    LLAMA3_PATH, load_model_and_tokenizer, format_llama3_chat, is_refusal,
    load_advbench, load_alpaca_benign, load_jbb_harmful, load_jbb_benign,
    load_catqa_english, load_sorrybench, load_xstest_prompts,
)


@torch.no_grad()
def batched_generate(model, tok, prompts, max_new_tokens=128, batch_size=8, device="cuda"):
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        formatted = [format_llama3_chat(tok, p) for p in batch]
        enc = tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.pad_token_id,
        )
        for j in range(len(batch)):
            in_len = enc.input_ids.shape[1]
            new_tokens = gen[j, in_len:]
            outs.append(tok.decode(new_tokens, skip_special_tokens=True))
        if (i // batch_size) % 5 == 0:
            print(f"  gen {i}/{len(prompts)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/gens_llama3")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Llama-3-8B-Instruct ...", flush=True)
    model, tok = load_model_and_tokenizer(LLAMA3_PATH)
    device = next(model.parameters()).device

    tasks = {
        "advbench_test":  load_advbench(n=args.n_test, seed=1),
        "alpaca_test":    load_alpaca_benign(n=args.n_test, seed=1),
        "jbb_harmful":    load_jbb_harmful(n=100, seed=0),
        "jbb_benign":     load_jbb_benign(n=100, seed=0),
        "catqa":          load_catqa_english(n=200, seed=0),
        "sorrybench":     load_sorrybench(n=200, seed=0),
    }
    for name, prompts in tasks.items():
        fp = os.path.join(args.out_dir, f"{name}.json")
        if os.path.exists(fp):
            print(f"[skip] {fp}", flush=True)
            continue
        print(f"[gen] {name}  n={len(prompts)}", flush=True)
        t0 = time.time()
        outs = batched_generate(model, tok, prompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, device=device)
        refusal = [is_refusal(o) for o in outs]
        with open(fp, "w") as f:
            json.dump({"prompts": prompts, "outputs": outs, "refusal_regex": refusal}, f, indent=2)
        print(f"  refusal rate (regex): {sum(refusal)/len(refusal):.3f}  ({time.time()-t0:.1f}s)", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
