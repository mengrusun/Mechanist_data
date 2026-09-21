"""Quick MMLU 5-shot accuracy on a random subset (for capability preservation check)."""
import os, json, random, argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def format_q(q, choices):
    letters = ["A", "B", "C", "D"]
    body = f"Question: {q}\n"
    for l, c in zip(letters, choices):
        body += f"{l}. {c}\n"
    body += "Answer:"
    return body


def build_prompt(shots, q, choices):
    parts = ["The following are multiple choice questions. Provide only the letter of the correct answer.\n"]
    for s in shots:
        letters = ["A", "B", "C", "D"]
        parts.append(format_q(s["question"], list(s["choices"])) + f" {letters[int(s['answer'])]}")
    parts.append(format_q(q, list(choices)))
    return "\n\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--test_parquet", default="/data/zhenqian/data/mmlu/all/test-00000-of-00001.parquet")
    p.add_argument("--dev_parquet", default="/data/zhenqian/data/mmlu/all/dev-00000-of-00001.parquet")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--nshot", type=int, default=5)
    p.add_argument("--out_json", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    random.seed(a.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(a.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.bfloat16, device_map={"": 0})
    if a.adapter_path:
        model = PeftModel.from_pretrained(base, a.adapter_path); model.eval()
    else:
        model = base; model.eval()

    test = pd.read_parquet(a.test_parquet)
    dev = pd.read_parquet(a.dev_parquet)

    subjects = sorted(test["subject"].unique())
    # Sample uniformly across subjects
    per_subj = max(1, a.n // len(subjects))
    sampled = []
    for s in subjects:
        sub = test[test["subject"] == s]
        sampled.extend(sub.sample(min(per_subj, len(sub)), random_state=a.seed).to_dict("records"))
    random.shuffle(sampled)
    sampled = sampled[: a.n]

    letters = ["A", "B", "C", "D"]
    letter_ids = [tok.encode(" " + l, add_special_tokens=False)[-1] for l in letters]
    print("letter token ids:", letter_ids)

    correct = 0
    total = 0
    for i, row in enumerate(sampled):
        subj = row["subject"]
        dev_subj = dev[dev["subject"] == subj]
        shots = dev_subj.sample(min(a.nshot, len(dev_subj)), random_state=a.seed).to_dict("records")
        prompt = build_prompt(shots, row["question"], row["choices"])
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits[0, -1, :]
        pred_idx = int(torch.stack([logits[t] for t in letter_ids]).argmax().item())
        gold = int(row["answer"])
        total += 1
        if pred_idx == gold:
            correct += 1
        if (i + 1) % 25 == 0 or i == 0:
            print(f"{i+1}/{len(sampled)} acc_so_far={correct/total:.3f}", flush=True)

    acc = correct / max(1, total)
    result = {"n": total, "correct": correct, "acc": acc}
    print("SUMMARY", result)
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(result, open(a.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
