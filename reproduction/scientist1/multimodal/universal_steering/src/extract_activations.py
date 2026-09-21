"""Extract per-block residual-stream activations from a HuggingFace decoder model.

Given (prompt, label) records, we forward each prompt through the model (in
inference mode, no generation) and capture the residual stream at the last
token position after each transformer block.

Output: a torch tensor of shape (n_examples, n_blocks, hidden_size) plus labels.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_input(record, mode, tokenizer):
    """Build the text to encode.

    mode="prompt_only": use chat template with only the prompt (last token is
      whatever the model would use to open its answer).
    mode="prompt_response": use chat template with prompt + response.
    """
    prompt = record["prompt"]
    response = record.get("response", "")
    if mode == "prompt_only" or not response:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return text


@torch.no_grad()
def collect_activations(model, tokenizer, records, mode, device, max_len=512):
    """Return acts of shape (N, n_blocks, hidden)."""
    all_acts = []
    for rec in tqdm(records, desc="Encoding"):
        text = build_input(rec, mode, tokenizer)
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(device)
        outputs = model(
            **enc,
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states: tuple of (n_blocks + 1) tensors; each (1, seq, hidden).
        # Skip embedding (index 0). Take last non-pad token position.
        hs = outputs.hidden_states[1:]
        last_idx = enc["attention_mask"].sum(dim=1).item() - 1
        per_block = torch.stack([h[0, last_idx].float().cpu() for h in hs], dim=0)
        all_acts.append(per_block)
    return torch.stack(all_acts, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="jsonl with prompt/response/label")
    ap.add_argument("--out", required=True, help="where to save .pt file")
    ap.add_argument("--mode", default="prompt_response",
                    choices=["prompt_only", "prompt_response"])
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None, help="only use this split")
    args = ap.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device
    )
    model.eval()

    records = load_jsonl(args.input)
    if args.split is not None:
        records = [r for r in records if r.get("split") == args.split]
    if args.limit is not None:
        records = records[: args.limit]

    print(f"Encoding {len(records)} records, mode={args.mode}")
    acts = collect_activations(
        model, tokenizer, records, args.mode, device, args.max_len
    )
    labels = torch.tensor([r["label"] for r in records], dtype=torch.long)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "activations": acts,
            "labels": labels,
            "model": args.model,
            "mode": args.mode,
            "records": records,
        },
        args.out,
    )
    print(f"Saved {acts.shape} to {args.out}")


if __name__ == "__main__":
    main()
