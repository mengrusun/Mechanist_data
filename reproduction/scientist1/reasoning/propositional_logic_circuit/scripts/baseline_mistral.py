"""Baseline: verify Mistral-7B answers the clean prompts correctly.

We compute logit_diff = logit(" true") - logit(" false") on the
final token of each prompt.

For clean prompts logit_diff should be strongly positive; for corrupted
prompts strongly negative (or at least less positive) so that patching
has a meaningful signal to move.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda:0"


FEWSHOT = (
    "Facts: p is true. Rule: if p is true then q is true. Question: q is true.\n"
    "Facts: p is true. Rule: if p is true then q is false. Question: q is false.\n"
    "Facts: p is true. Rule: if p is true then q is true. Question: q is true.\n"
    "Facts: p is true. Rule: if p is true then q is false. Question: q is false.\n"
)


def load_dataset(path: str, prepend_fewshot: bool = True):
    ds = [json.loads(l) for l in Path(path).read_text().splitlines()]
    if prepend_fewshot:
        for r in ds:
            for k in ("clean", "corrupt", "fact_flip", "query_flip"):
                if k in r:
                    r[k] = FEWSHOT + r[k]
    return ds


def token_ids_for(tok, word: str, ref_prompt: str = "Question: x is"):
    """Return the single token id that would be generated as continuation of
    a prompt ending with a word like 'is', i.e. the token corresponding to
    the answer word. Uses the actual continuation-token id so it works both
    for SentencePiece and BPE tokenizers."""
    base = tok.encode(ref_prompt, add_special_tokens=False)
    full = tok.encode(ref_prompt + " " + word, add_special_tokens=False)
    extra = full[len(base):]
    if len(extra) != 1:
        raise ValueError(f"'{word}' -> extra {extra} not single-token")
    return extra[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Mistral-7B-v0.1")
    ap.add_argument("--data", default="data/logic_ds.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/baseline_mistral.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    print("pad", tok.pad_token, "eos", tok.eos_token)
    tok.pad_token = tok.eos_token

    true_id = token_ids_for(tok, "true")
    false_id = token_ids_for(tok, "false")
    print("true_id", true_id, "false_id", false_id)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()

    ds = load_dataset(args.data)[: args.n]
    clean_diffs, corrupt_diffs = [], []
    clean_lens, corrupt_lens = [], []
    aligned = 0

    t0 = time.time()
    B = 8
    with torch.no_grad():
        for i in range(0, len(ds), B):
            batch = ds[i : i + B]
            clean_ids = [tok.encode(x["clean"], return_tensors="pt")[0] for x in batch]
            corrupt_ids = [tok.encode(x["corrupt"], return_tensors="pt")[0] for x in batch]

            for c, cr in zip(clean_ids, corrupt_ids):
                clean_lens.append(len(c))
                corrupt_lens.append(len(cr))
                if len(c) == len(cr):
                    aligned += 1

            # left-pad to max length in batch
            def pad_stack(seqs):
                m = max(len(s) for s in seqs)
                padded = torch.full((len(seqs), m), tok.pad_token_id, dtype=torch.long)
                for i, s in enumerate(seqs):
                    padded[i, m - len(s):] = s
                return padded.to(DEVICE)

            cin = pad_stack(clean_ids)
            crin = pad_stack(corrupt_ids)

            clean_logits = model(cin).logits[:, -1, :]  # (B, V)
            corr_logits = model(crin).logits[:, -1, :]

            # signed logit_diff = logit(clean_answer) - logit(corrupt_answer)
            # clean: expect large positive, corrupt: expect large negative
            clean_ans_ids = torch.tensor(
                [token_ids_for(tok, b["clean_answer"]) for b in batch], device=DEVICE
            )
            corr_ans_ids = torch.tensor(
                [token_ids_for(tok, b["corrupt_answer"]) for b in batch], device=DEVICE
            )
            clean_diff = (
                clean_logits.gather(1, clean_ans_ids[:, None]).squeeze(1)
                - clean_logits.gather(1, corr_ans_ids[:, None]).squeeze(1)
            ).float().cpu().numpy()
            corr_diff = (
                corr_logits.gather(1, clean_ans_ids[:, None]).squeeze(1)
                - corr_logits.gather(1, corr_ans_ids[:, None]).squeeze(1)
            ).float().cpu().numpy()
            clean_diffs.extend(clean_diff.tolist())
            corrupt_diffs.extend(corr_diff.tolist())

    dt = time.time() - t0
    clean_diffs = np.array(clean_diffs)
    corrupt_diffs = np.array(corrupt_diffs)

    # signed logit_diff = logit(clean_answer)-logit(corrupt_answer)
    # clean correct when >0, corrupt correct when <0.
    clean_acc = float((clean_diffs > 0).mean())
    corrupt_acc = float((corrupt_diffs < 0).mean())

    print(f"processed {len(ds)} in {dt:.1f}s")
    print(f"token-aligned pairs (same length): {aligned}/{len(ds)}")
    print(f"clean len min/max/mean: {min(clean_lens)}/{max(clean_lens)}/{np.mean(clean_lens):.1f}")
    print(f"clean_diff mean={clean_diffs.mean():.3f} std={clean_diffs.std():.3f}  acc(true>false)={clean_acc:.3f}")
    print(f"corr _diff mean={corrupt_diffs.mean():.3f} std={corrupt_diffs.std():.3f}  acc(false>true)={corrupt_acc:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "clean_acc": clean_acc,
                "corrupt_acc": corrupt_acc,
                "clean_mean": float(clean_diffs.mean()),
                "corrupt_mean": float(corrupt_diffs.mean()),
                "aligned_pairs": aligned,
                "n": len(ds),
                "clean_diffs": clean_diffs.tolist(),
                "corrupt_diffs": corrupt_diffs.tolist(),
                "true_id": true_id,
                "false_id": false_id,
            },
            f,
            indent=2,
        )
    print("saved to", args.out)


if __name__ == "__main__":
    main()
