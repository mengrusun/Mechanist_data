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

from datasets import load_dataset

gsm8k_paths = [
    os.path.join(DATA_DIR, "gsm8k"),
    os.path.join(DATA_DIR, "GSM8K"),
    os.path.join(DATA_DIR, "openai_gsm8k"),
    os.path.join(DATA_DIR, "openai/gsm8k"),
]

ds = None
attempts_log = []

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


def extract_gt(ans_text):
    m = re.search(r"####\s*(-?[\d,\.]+)", ans_text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d+\.?\d*", ans_text.replace(",", ""))
    return nums[-1] if nums else ""


def extract_pred(text):
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        inner = m.group(1)
        nums = re.findall(r"-?\d+\.?\d*", inner.replace(",", ""))
        if nums:
            return nums[-1]
    m = re.search(
        r"(?:answer is|answer:|final answer[:\s]+)[^\d\-]*(-?\d+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
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

MODEL_PATH = os.path.join(MODEL_DIR, "Qwen3-14B")
if not os.path.exists(MODEL_PATH):
    for alt in ["Qwen2.5-14B-Instruct", "Qwen3-14B-Instruct", "Qwen2.5-7B-Instruct"]:
        cand = os.path.join(MODEL_DIR, alt)
        if os.path.exists(cand):
            MODEL_PATH = cand
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


N_SAMPLES = 50
rng = np.random.RandomState(42)
indices = rng.choice(len(ds), size=min(N_SAMPLES, len(ds)), replace=False).tolist()
subset = [ds[i] for i in indices]
questions = [ex["question"] for ex in subset]
gts = [extract_gt(ex["answer"]) for ex in subset]

# ---------------- Hyperparameter tuning: max_new_tokens ----------------
MAX_NEW_TOKENS_SWEEP = [256, 512, 1024]

experiment_data = {
    "max_new_tokens": {
        "GSM8K": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": {},
            "ground_truth": gts,
            "questions": questions,
            "raw_outputs": {},
            "accuracy_per_template": {},
            "accuracy_per_hparam": {},
            "templates": TEMPLATES,
            "n_samples": len(subset),
            "indices": indices,
            "hparam_values": MAX_NEW_TOKENS_SWEEP,
        }
    }
}

BATCH_SIZE = 16 if USE_VLLM else 1

for mnt in MAX_NEW_TOKENS_SWEEP:
    key = f"mnt_{mnt}"
    print(f"\n########## max_new_tokens = {mnt} ##########")
    experiment_data["max_new_tokens"]["GSM8K"]["predictions"][key] = {}
    experiment_data["max_new_tokens"]["GSM8K"]["raw_outputs"][key] = {}
    experiment_data["max_new_tokens"]["GSM8K"]["accuracy_per_template"][key] = {}

    for tname, prefix in TEMPLATES.items():
        print(f"\n=== Template: {tname} (max_new_tokens={mnt}) ===")
        prompts = [build_prompt(q, prefix) for q in questions]
        all_outs = []
        t0 = time.time()
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i : i + BATCH_SIZE]
            outs = generate_batch(batch, max_new_tokens=mnt)
            all_outs.extend(outs)
            print(
                f"  [{tname}|mnt={mnt}] {i+len(batch)}/{len(prompts)} in {time.time()-t0:.1f}s"
            )
        preds = [extract_pred(o) for o in all_outs]
        correct = sum(
            1 for p, g in zip(preds, gts) if normalize_num(p) == normalize_num(g)
        )
        acc = correct / len(gts)
        print(f"[{tname}|mnt={mnt}] accuracy = {acc:.4f} ({correct}/{len(gts)})")

        experiment_data["max_new_tokens"]["GSM8K"]["predictions"][key][tname] = preds
        experiment_data["max_new_tokens"]["GSM8K"]["raw_outputs"][key][tname] = all_outs
        experiment_data["max_new_tokens"]["GSM8K"]["accuracy_per_template"][key][
            tname
        ] = acc
        experiment_data["max_new_tokens"]["GSM8K"]["metrics"]["val"].append(
            {"template": tname, "max_new_tokens": mnt, "accuracy": acc, "epoch": 0}
        )
        experiment_data["max_new_tokens"]["GSM8K"]["losses"]["val"].append(
            {"template": tname, "max_new_tokens": mnt, "loss": 1.0 - acc, "epoch": 0}
        )
        print(f"Epoch 0: validation_loss = {1.0-acc:.4f} (template={tname}, mnt={mnt})")

    # Average across templates for this hparam
    accs = list(
        experiment_data["max_new_tokens"]["GSM8K"]["accuracy_per_template"][
            key
        ].values()
    )
    mean_acc = float(np.mean(accs))
    experiment_data["max_new_tokens"]["GSM8K"]["accuracy_per_hparam"][key] = mean_acc
    print(f"\n>>> Mean accuracy for mnt={mnt}: {mean_acc:.4f}")

# ---------------- Save & plot ----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

acc_per_hparam_template = experiment_data["max_new_tokens"]["GSM8K"][
    "accuracy_per_template"
]
template_names = list(TEMPLATES.keys())
hparam_keys = [f"mnt_{m}" for m in MAX_NEW_TOKENS_SWEEP]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(template_names))
width = 0.25
for i, hk in enumerate(hparam_keys):
    vals = [acc_per_hparam_template[hk][t] for t in template_names]
    ax.bar(x + i * width, vals, width, label=hk)
ax.set_xticks(x + width)
ax.set_xticklabels(template_names, rotation=20)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.0)
ax.set_title(f"GSM8K accuracy by emotional prefix × max_new_tokens (N={len(gts)})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "gsm8k_max_new_tokens_sweep.png"), dpi=120)
plt.close()

# Mean acc per hparam
plt.figure(figsize=(7, 4))
means = [
    experiment_data["max_new_tokens"]["GSM8K"]["accuracy_per_hparam"][hk]
    for hk in hparam_keys
]
plt.plot(MAX_NEW_TOKENS_SWEEP, means, marker="o")
for xv, yv in zip(MAX_NEW_TOKENS_SWEEP, means):
    plt.text(xv, yv + 0.005, f"{yv:.3f}", ha="center")
plt.xlabel("max_new_tokens")
plt.ylabel("Mean accuracy across templates")
plt.title("Effect of max_new_tokens on GSM8K accuracy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "gsm8k_mnt_mean_acc.png"), dpi=120)
plt.close()

print("\n=== Summary (mean across templates) ===")
for hk in hparam_keys:
    print(
        f"  {hk}: {experiment_data['max_new_tokens']['GSM8K']['accuracy_per_hparam'][hk]:.4f}"
    )
print(f"Saved to {working_dir}")
