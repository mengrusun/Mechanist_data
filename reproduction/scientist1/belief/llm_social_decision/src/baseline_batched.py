"""Fast batched baseline: 1000 dictator trials -> E[transfer], argmax, cache last-hidden states."""

import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def integer_token_ids(tokenizer):
    ids = []
    for n in range(0, 21):
        tok = tokenizer.encode(str(n), add_special_tokens=False)
        assert len(tok) == 1, f"expected single token for {n}, got {tok}"
        ids.append(tok[0])
    return torch.tensor(ids)


@torch.no_grad()
def run(model_path, trials_path, out_path, limit=None, batch=16, save_hidden=True):
    device = "cuda"
    print(f"Loading {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        output_hidden_states=save_hidden,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"n_layers={n_layers} hidden_size={hidden_size}")

    int_ids = integer_token_ids(tokenizer).to(device)  # (21,)
    trials = [json.loads(l) for l in open(trials_path)]
    if limit:
        trials = trials[:limit]

    results = []
    # per-layer last-token hidden state accumulator (we save an npy separately)
    if save_hidden:
        H = np.zeros((len(trials), n_layers+1, hidden_size), dtype=np.float16)

    for start in range(0, len(trials), batch):
        chunk = trials[start:start+batch]
        prompts = [t["prompt"] for t in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        out = model(**enc, output_hidden_states=save_hidden)
        logits = out.logits[:, -1, :]  # (B, V)
        # gather logits at int ids
        int_logits = logits[:, int_ids]  # (B, 21)
        probs = torch.softmax(int_logits, dim=-1).float().cpu().numpy()

        if save_hidden:
            # hidden_states is a tuple of length n_layers+1, each (B,T,H)
            for li, h in enumerate(out.hidden_states):
                H[start:start+len(chunk), li] = h[:, -1, :].float().cpu().numpy().astype(np.float16)

        for i, t in enumerate(chunk):
            ps = probs[i]
            e = float(np.dot(np.arange(21), ps))
            am = int(np.argmax(ps))
            results.append({
                "id": t["id"],
                "gender": t["gender"], "age": t["age"],
                "instruction": t["instruction"], "meeting": t["meeting"],
                "argmax": am, "E_transfer": e,
                "p": ps.tolist(),
            })
        if (start//batch) % 4 == 0:
            print(f"[{start+len(chunk)}/{len(trials)}] last E={e:.2f} argmax={am}")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} results to {out_path}")

    if save_hidden:
        hidden_out = Path(out_path).with_suffix(".hidden.npy")
        np.save(hidden_out, H)
        print(f"Saved hidden states to {hidden_out}, shape={H.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--trials", default=str(DATA/"trials.jsonl"))
    ap.add_argument("--out", default=str(RES/"baseline_llama.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-hidden", action="store_true")
    args = ap.parse_args()
    run(args.model, args.trials, args.out, args.limit, args.batch, save_hidden=not args.no_hidden)
