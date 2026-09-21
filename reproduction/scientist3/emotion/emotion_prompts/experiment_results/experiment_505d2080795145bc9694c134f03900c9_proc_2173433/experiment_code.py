# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, re, sys, json, time, glob, traceback
import numpy as np
import torch

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

os.environ["HF_TOKEN"] = "<Your_token>"
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"

# ---------------- Dataset loading (robust, flat fallback) ----------------
from datasets import load_dataset

gsm8k_paths = [
    os.path.join(DATA_DIR, "gsm8k"),
    os.path.join(DATA_DIR, "GSM8K"),
    os.path.join(DATA_DIR, "openai_gsm8k"),
    os.path.join(DATA_DIR, "openai/gsm8k"),
]

ds = None
attempts_log = []

# Tier 1: local load with multiple config names
for p in gsm8k_paths:
    if not os.path.exists(p):
        attempts_log.append(f"skip missing {p}")
        continue
    for cfg in ("main", "default", None):
        try:
            ds = (
                load_dataset(p, cfg, split="test")
                if cfg
                else load_dataset(p, split="test")
            )
            attempts_log.append(f"OK local {p} cfg={cfg}")
            break
        except Exception as e:
            attempts_log.append(
                f"FAIL local {p} cfg={cfg}: {type(e).__name__}: {str(e)[:120]}"
            )
    if ds is not None:
        break

# Tier 2: parquet walk
if ds is None:
    for p in gsm8k_paths:
        if not os.path.exists(p):
            continue
        cand = []
        for r, _, fs in os.walk(p):
            for f in fs:
                if f.endswith(".parquet") and "test" in f.lower():
                    cand.append(os.path.join(r, f))
        if cand:
            try:
                ds = load_dataset("parquet", data_files=cand, split="train")
                attempts_log.append(f"OK parquet {cand}")
                break
            except Exception as e:
                attempts_log.append(f"FAIL parquet {p}: {e}")

# Tier 3: hub
if ds is None:
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        attempts_log.append("OK hub openai/gsm8k main")
    except Exception as e:
        attempts_log.append(f"FAIL hub: {e}")

for line in attempts_log:
    print(line)

if ds is None:
    raise RuntimeError("Could not load GSM8K from any source.")

print(f"Loaded GSM8K test set: {len(ds)} examples")
print("Sample:", {k: (v[:100] if isinstance(v, str) else v) for k, v in ds[0].items()})


# ---------------- Answer extraction ----------------
def extract_gt(ans_text):
    m = re.search(r"####\s*(-?[\d,\.]+)", ans_text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d+\.?\d*", ans_text.replace(",", ""))
    return nums[-1] if nums else ""


def extract_pred(text):
    # Try boxed
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        inner = m.group(1)
        nums = re.findall(r"-?\d+\.?\d*", inner.replace(",", ""))
        if nums:
            return nums[-1]
    # Try "answer is X"
    m = re.search(
        r"(?:answer is|answer:|final answer[:\s]+)[^\d\-]*(-?\d+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)
    # Try ####
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    # Last number
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else ""


def normalize_num(s):
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.4f}".rstrip("0").rstrip(".")
    except:
        return str(s).strip()


# Test extraction
test_ans = "Janet sells... #### 18"
assert extract_gt(test_ans) == "18", f"extract_gt failed: {extract_gt(test_ans)}"
print("Extraction smoke test passed.")

# ---------------- Prompt templates ----------------
TEMPLATES = {
    "neutral": "",
    "happiness": "I'm feeling wonderful and joyful today! Solving problems makes me so happy. ",
    "sadness": "I've been feeling really down and sad lately, and I could use some help. ",
    "fear": "I'm anxious and worried I might get this wrong — please help me carefully. ",
    "anger": "I'm frustrated and angry that I can't figure this out on my own. ",
    "disgust": "I find sloppy reasoning disgusting; please be rigorous. ",
    "surprise": "Wow, I'm genuinely surprised by this problem — let's see! ",
}

SYSTEM = "You are a helpful math assistant. Solve the problem step by step and put the final numeric answer after '####'."

# ---------------- Load model ----------------
MODEL_PATH = os.path.join(MODEL_DIR, "Qwen3-14B")
if not os.path.exists(MODEL_PATH):
    # try alternate names
    for alt in ["Qwen2.5-14B-Instruct", "Qwen3-14B-Instruct", "Qwen2.5-7B-Instruct"]:
        cand = os.path.join(MODEL_DIR, alt)
        if os.path.exists(cand):
            MODEL_PATH = cand
            print(f"Using alternate model: {alt}")
            break
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No suitable model found in {MODEL_DIR}")

print(f"Loading model from {MODEL_PATH}")

USE_VLLM = False
try:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    USE_VLLM = True
    print("Using vLLM backend")
except Exception as e:
    print(f"vLLM not available ({e}); falling back to transformers")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()


def build_prompt(question, prefix):
    user = f"{prefix}{question}"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# ---------------- Smoke test on 1 example ----------------
smoke_q = ds[0]["question"]
smoke_gt = extract_gt(ds[0]["answer"])
smoke_prompt = build_prompt(smoke_q, "")
print(f"Smoke GT={smoke_gt}")


def generate_batch(prompts, max_new_tokens=512):
    if USE_VLLM:
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outs = llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs]
    else:
        outputs = []
        for pr in prompts:
            inputs = tokenizer(
                pr, return_tensors="pt", truncation=True, max_length=3000
            ).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            txt = tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            outputs.append(txt)
        return outputs


smoke_out = generate_batch([smoke_prompt], max_new_tokens=256)[0]
print("Smoke output (first 300 chars):", smoke_out[:300])
print("Smoke pred:", extract_pred(smoke_out))

# ---------------- Run evaluation ----------------
N_SAMPLES = 50
rng = np.random.RandomState(42)
indices = rng.choice(len(ds), size=min(N_SAMPLES, len(ds)), replace=False).tolist()
subset = [ds[i] for i in indices]
questions = [ex["question"] for ex in subset]
gts = [extract_gt(ex["answer"]) for ex in subset]

experiment_data = {
    "GSM8K": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": {},
        "ground_truth": gts,
        "questions": questions,
        "raw_outputs": {},
        "accuracy_per_template": {},
        "templates": TEMPLATES,
        "n_samples": len(subset),
        "indices": indices,
    }
}

BATCH_SIZE = 16 if USE_VLLM else 1

for tname, prefix in TEMPLATES.items():
    print(f"\n=== Template: {tname} ===")
    prompts = [build_prompt(q, prefix) for q in questions]
    all_outs = []
    t0 = time.time()
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        outs = generate_batch(batch, max_new_tokens=512)
        all_outs.extend(outs)
        print(f"  [{tname}] {i+len(batch)}/{len(prompts)} in {time.time()-t0:.1f}s")
    preds = [extract_pred(o) for o in all_outs]
    correct = sum(1 for p, g in zip(preds, gts) if normalize_num(p) == normalize_num(g))
    acc = correct / len(gts)
    print(f"[{tname}] accuracy = {acc:.4f} ({correct}/{len(gts)})")

    experiment_data["GSM8K"]["predictions"][tname] = preds
    experiment_data["GSM8K"]["raw_outputs"][tname] = all_outs
    experiment_data["GSM8K"]["accuracy_per_template"][tname] = acc
    experiment_data["GSM8K"]["metrics"]["val"].append(
        {"template": tname, "accuracy": acc, "epoch": 0}
    )
    print(f"Epoch 0: validation_loss = {1.0-acc:.4f} (template={tname})")

# ---------------- Save & plot ----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

acc_dict = experiment_data["GSM8K"]["accuracy_per_template"]
names = list(acc_dict.keys())
vals = [acc_dict[n] for n in names]

plt.figure(figsize=(9, 5))
colors = ["gray"] + ["C{}".format(i) for i in range(len(names) - 1)]
plt.bar(names, vals, color=colors)
plt.axhline(
    acc_dict.get("neutral", 0),
    color="black",
    linestyle="--",
    alpha=0.5,
    label="neutral baseline",
)
plt.ylabel("Accuracy")
plt.title(f"GSM8K accuracy by emotional prefix (Qwen, N={len(gts)})")
plt.ylim(0, 1.0)
for i, v in enumerate(vals):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "gsm8k_emotion_prefix_accuracy.png"), dpi=120)
plt.close()

print("\n=== Summary ===")
for n, v in acc_dict.items():
    print(f"  {n}: {v:.4f}")
print(f"Saved to {working_dir}")
