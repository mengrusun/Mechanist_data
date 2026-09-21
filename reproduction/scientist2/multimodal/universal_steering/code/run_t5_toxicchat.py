"""Standalone: run ToxicChat-T5-Large on the test split saved by C5_monitoring.
Writes updated summary.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "<Your_token>"
for k in list(os.environ.keys()):
    if "proxy" in k.lower() or "PROXY" in k:
        os.environ.pop(k, None)

sys.path.insert(0, "/data/zhenqian/Reproduction1/mechanica/multimodal/universal_steering/code")
from rfm_core import probe_auroc

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

WORK = Path("/data/zhenqian/Reproduction1/mechanica/multimodal/universal_steering")
c5 = WORK / "runs/C5_monitoring"
out = WORK / "runs/C5_baselines"
out.mkdir(exist_ok=True)

model_id = "lmsys/toxicchat-t5-large-v1.0"
print(f"Loading {model_id}...", flush=True)
# IMPORTANT: README says to use t5-large tokenizer, NOT the checkpoint's tokenizer.
tok = AutoTokenizer.from_pretrained("t5-large", cache_dir="/data/zhenqian/models/_hf_cache")
m = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir="/data/zhenqian/models/_hf_cache",
                                            torch_dtype=torch.float16).cuda().eval()

rows = [json.loads(l) for l in open(c5 / "toxicchat_test.jsonl")]
labels = np.array([int(r.get("label", 0)) for r in rows], dtype=np.float64)
y = np.where(labels > 0, 1.0, -1.0)

prefix = "ToxicChat: "
scores = []  # continuous score in [0, 1] — probability of "positive" label
raw_texts = []
# First-token probs of "positive" vs "negative" (both in T5 vocabulary as single tokens).
pos_id = tok.encode("positive", add_special_tokens=False)[0]
neg_id = tok.encode("negative", add_special_tokens=False)[0]
print(f"pos_id={pos_id} neg_id={neg_id}", flush=True)
with torch.no_grad():
    for i, r in enumerate(rows):
        prompt = prefix + r.get("prompt", "")
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        decoder_start = m.config.decoder_start_token_id
        dec_input = torch.tensor([[decoder_start]], device="cuda")
        out = m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                decoder_input_ids=dec_input)
        logits = out.logits[0, -1].float()
        # Softmax over the binary pair (positive vs negative)
        pair = torch.tensor([logits[pos_id], logits[neg_id]])
        pair = torch.softmax(pair, dim=-1)
        p_pos = float(pair[0])
        scores.append(p_pos)
        # Also decode greedy top-1 for logging
        top_id = int(logits.argmax())
        raw_texts.append(tok.decode([top_id]))
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}  top_txt={raw_texts[-1]!r} p_pos={p_pos:.3f}", flush=True)
from collections import Counter
print(f"top-token distribution: {Counter(raw_texts).most_common(5)}", flush=True)

scores = np.array(scores, dtype=np.float32)
np.save(out / "toxicchat_t5_scores.npy", scores)
t5_auroc = float(probe_auroc(np.array([1.0]), scores.reshape(-1, 1), y))
print(f"T5-Large test AUROC = {t5_auroc:.4f}", flush=True)

# Merge
c5_sum_path = c5 / "summary.json"
if c5_sum_path.exists():
    c5_sum = json.load(open(c5_sum_path))
    c5_sum["benchmarks"]["toxicchat"]["t5_test_auroc"] = t5_auroc
    json.dump(c5_sum, open(c5_sum_path, "w"), indent=2, default=float)
    print(f"merged into {c5_sum_path}")

c5b = out / "summary.json"
if c5b.exists():
    d = json.load(open(c5b))
else:
    d = {"baselines": {}}
d.setdefault("baselines", {}).setdefault("toxicchat", {})["t5_test_auroc"] = t5_auroc
json.dump(d, open(c5b, "w"), indent=2)
print(f"updated {c5b}")
