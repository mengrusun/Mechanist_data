import os, re, sys, json, time, glob
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

# --------- Load GSM8K ---------
ds = None
gsm8k_paths = [
    os.path.join(DATA_DIR, "gsm8k"),
    os.path.join(DATA_DIR, "GSM8K"),
    os.path.join(DATA_DIR, "openai_gsm8k"),
]
for p in gsm8k_paths:
    if not os.path.exists(p):
        continue
    for cfg in ("main", "default", None):
        try:
            ds = (
                load_dataset(p, cfg, split="test")
                if cfg
                else load_dataset(p, split="test")
            )
            print(f"Loaded GSM8K from {p} cfg={cfg}")
            break
        except Exception as e:
            pass
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
                print(f"Loaded GSM8K parquet {cand}")
                break
            except Exception as e:
                pass

if ds is None:
    ds = load_dataset("openai/gsm8k", "main", split="test")

print(f"GSM8K test set size: {len(ds)}")


def extract_gt(ans_text):
    m = re.search(r"####\s*(-?[\d,\.]+)", ans_text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d+\.?\d*", ans_text.replace(",", ""))
    return nums[-1] if nums else ""


def extract_pred(text):
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        nums = re.findall(r"-?\d+\.?\d*", m.group(1).replace(",", ""))
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


# --------- Prompt templates: 6 emotions × 2 intensities × 2 sources (human/LLM) + neutral ---------
# Each entry: (emotion, intensity, source): prefix
TEMPLATES = {
    ("neutral", "none", "none"): "",
    # Human-written, mild
    ("happiness", "mild", "human"): "I'm feeling pretty happy today. ",
    ("sadness", "mild", "human"): "I'm feeling a bit down today. ",
    ("fear", "mild", "human"): "I'm a little worried I might get this wrong. ",
    ("anger", "mild", "human"): "I'm a bit frustrated with this problem. ",
    ("disgust", "mild", "human"): "Careless mistakes bother me. ",
    ("surprise", "mild", "human"): "This problem is a bit unexpected. ",
    # Human-written, strong
    (
        "happiness",
        "strong",
        "human",
    ): "I'm absolutely thrilled and overjoyed today — solving problems fills me with pure delight! ",
    (
        "sadness",
        "strong",
        "human",
    ): "I've been feeling deeply depressed and hopeless — I really need this to go well. ",
    (
        "fear",
        "strong",
        "human",
    ): "I'm terrified of failing this — my future depends on getting this exactly right, please be extremely careful! ",
    (
        "anger",
        "strong",
        "human",
    ): "I am furious and outraged that this is so hard — I demand a correct answer, NOW! ",
    (
        "disgust",
        "strong",
        "human",
    ): "Sloppy, lazy, careless reasoning is utterly revolting to me — I will not tolerate any imprecision! ",
    (
        "surprise",
        "strong",
        "human",
    ): "WOW! I am absolutely stunned and astonished — I cannot believe how surprising this problem is! ",
    # LLM-generated (crafted to be paraphrases with same emotional content), mild
    (
        "happiness",
        "mild",
        "llm",
    ): "Feeling cheerful and content — happy to work through this. ",
    ("sadness", "mild", "llm"): "Feeling a touch melancholy today, but let's try. ",
    ("fear", "mild", "llm"): "I have some mild anxiety about making an error here. ",
    ("anger", "mild", "llm"): "I'm somewhat irritated by how tricky this seems. ",
    ("disgust", "mild", "llm"): "I have a slight distaste for imprecise answers. ",
    ("surprise", "mild", "llm"): "Interestingly, this caught me a little off guard. ",
    # LLM-generated, strong
    (
        "happiness",
        "strong",
        "llm",
    ): "My heart is bursting with joy and elation — every problem feels like a wonderful gift today! ",
    (
        "sadness",
        "strong",
        "llm",
    ): "A heavy sorrow weighs on me — please, in this dark moment, help me find the light of a correct solution. ",
    (
        "fear",
        "strong",
        "llm",
    ): "Dread grips me completely — the thought of an incorrect answer is unbearable; I beg you to be meticulously precise. ",
    (
        "anger",
        "strong",
        "llm",
    ): "Rage courses through me — this problem's obstinance is INFURIATING; give me the right answer without hesitation! ",
    (
        "disgust",
        "strong",
        "llm",
    ): "Nothing repulses me more than shoddy, half-hearted reasoning — I insist on immaculate, rigorous logic throughout. ",
    (
        "surprise",
        "strong",
        "llm",
    ): "I am utterly flabbergasted, dumbstruck with astonishment — this problem defies all my expectations! ",
}

SYSTEM = "You are a helpful math assistant. Solve the problem step by step and put the final numeric answer after '####'."

MODEL_PATH = os.path.join(MODEL_DIR, "Qwen3-14B")
if not os.path.exists(MODEL_PATH):
    for alt in ["Qwen2.5-14B-Instruct", "Qwen3-14B-Instruct", "Qwen2.5-7B-Instruct"]:
        cand = os.path.join(MODEL_DIR, alt)
        if os.path.exists(cand):
            MODEL_PATH = cand
            break
print(f"Model path: {MODEL_PATH}")

USE_VLLM = False
try:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.88,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    USE_VLLM = True
    print("Using vLLM backend")
except Exception as e:
    print(f"vLLM failed: {e}; using transformers")
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
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]
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
            new_tokens = gen[0][inputs["input_ids"].shape[1] :]
            txt = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append((txt, len(new_tokens)))
        return outputs


# --------- Sample N questions ---------
N_SAMPLES = 200
rng = np.random.RandomState(42)
indices = rng.choice(len(ds), size=min(N_SAMPLES, len(ds)), replace=False).tolist()
subset = [ds[i] for i in indices]
questions = [ex["question"] for ex in subset]
gts = [extract_gt(ex["answer"]) for ex in subset]
print(f"Evaluating on N={len(questions)} questions")

MAX_NEW_TOKENS = 512
BATCH_SIZE = 32 if USE_VLLM else 1

experiment_data = {
    "emotional_framing_full": {
        "GSM8K": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": {},
            "ground_truth": gts,
            "questions": questions,
            "indices": indices,
            "templates": {str(k): v for k, v in TEMPLATES.items()},
            "accuracy_per_template": {},
            "correctness_per_template": {},  # list of 0/1
            "gen_tokens_per_template": {},  # list of int token counts
            "n_samples": len(questions),
            "max_new_tokens": MAX_NEW_TOKENS,
            "model_path": MODEL_PATH,
        }
    }
}

# --------- Run all templates ---------
for tkey, prefix in TEMPLATES.items():
    tname = "|".join(tkey)
    print(f"\n=== Template {tname} ===")
    prompts = [build_prompt(q, prefix) for q in questions]
    all_outs, all_lens = [], []
    t0 = time.time()
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        outs = generate_batch(batch, max_new_tokens=MAX_NEW_TOKENS)
        for txt, ln in outs:
            all_outs.append(txt)
            all_lens.append(ln)
        print(f"  [{tname}] {i+len(batch)}/{len(prompts)} in {time.time()-t0:.1f}s")
    preds = [extract_pred(o) for o in all_outs]
    correctness = [
        1 if normalize_num(p) == normalize_num(g) else 0 for p, g in zip(preds, gts)
    ]
    acc = float(np.mean(correctness))
    print(
        f"[{tname}] accuracy = {acc:.4f} ({sum(correctness)}/{len(gts)}) | mean_len={np.mean(all_lens):.1f}"
    )

    experiment_data["emotional_framing_full"]["GSM8K"]["predictions"][tname] = preds
    experiment_data["emotional_framing_full"]["GSM8K"]["accuracy_per_template"][
        tname
    ] = acc
    experiment_data["emotional_framing_full"]["GSM8K"]["correctness_per_template"][
        tname
    ] = correctness
    experiment_data["emotional_framing_full"]["GSM8K"]["gen_tokens_per_template"][
        tname
    ] = all_lens
    experiment_data["emotional_framing_full"]["GSM8K"]["metrics"]["val"].append(
        {"template": tname, "accuracy": acc, "epoch": 0}
    )
    experiment_data["emotional_framing_full"]["GSM8K"]["losses"]["val"].append(
        {"template": tname, "loss": 1.0 - acc, "epoch": 0}
    )
    print(f"Epoch 0: validation_loss = {1.0-acc:.4f} (template={tname})")

# --------- Analysis: oracle upper bound (adaptive per-query) ---------
tnames_all = list(
    experiment_data["emotional_framing_full"]["GSM8K"]["accuracy_per_template"].keys()
)
correctness_matrix = np.array(
    [
        experiment_data["emotional_framing_full"]["GSM8K"]["correctness_per_template"][
            t
        ]
        for t in tnames_all
    ]
)  # [T, N]
oracle_correct = (correctness_matrix.max(axis=0)).mean()  # any template correct
neutral_key = "neutral|none|none"
neutral_correctness = np.array(
    experiment_data["emotional_framing_full"]["GSM8K"]["correctness_per_template"][
        neutral_key
    ]
)
neutral_acc = neutral_correctness.mean()

# Per-question flip analysis: for each non-neutral template, how many flips vs neutral
flip_stats = {}
for t in tnames_all:
    if t == neutral_key:
        continue
    c = np.array(
        experiment_data["emotional_framing_full"]["GSM8K"]["correctness_per_template"][
            t
        ]
    )
    helped = int(((c == 1) & (neutral_correctness == 0)).sum())
    hurt = int(((c == 0) & (neutral_correctness == 1)).sum())
    same = int((c == neutral_correctness).sum())
    net = helped - hurt
    flip_stats[t] = {
        "helped": helped,
        "hurt": hurt,
        "same": same,
        "net": net,
        "delta_acc": float(c.mean() - neutral_acc),
    }

experiment_data["emotional_framing_full"]["GSM8K"]["oracle_upper_bound"] = float(
    oracle_correct
)
experiment_data["emotional_framing_full"]["GSM8K"]["neutral_accuracy"] = float(
    neutral_acc
)
experiment_data["emotional_framing_full"]["GSM8K"]["flip_stats_vs_neutral"] = flip_stats

# Reasoning length by emotion/intensity
len_by_template = {
    t: float(
        np.mean(
            experiment_data["emotional_framing_full"]["GSM8K"][
                "gen_tokens_per_template"
            ][t]
        )
    )
    for t in tnames_all
}
experiment_data["emotional_framing_full"]["GSM8K"]["mean_gen_tokens"] = len_by_template

# --------- Save ---------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# --------- Plots ---------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1) Accuracy per template, grouped by emotion, colored by intensity+source
emotions = ["happiness", "sadness", "fear", "anger", "disgust", "surprise"]
combos = [("mild", "human"), ("strong", "human"), ("mild", "llm"), ("strong", "llm")]
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(emotions))
width = 0.2
for i, (intensity, source) in enumerate(combos):
    vals = []
    for e in emotions:
        key = f"{e}|{intensity}|{source}"
        vals.append(
            experiment_data["emotional_framing_full"]["GSM8K"][
                "accuracy_per_template"
            ].get(key, np.nan)
        )
    ax.bar(x + i * width, vals, width, label=f"{intensity}-{source}")
ax.axhline(neutral_acc, color="k", linestyle="--", label=f"neutral={neutral_acc:.3f}")
ax.axhline(
    oracle_correct, color="red", linestyle=":", label=f"oracle={oracle_correct:.3f}"
)
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(emotions)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title(f"GSM8K accuracy: emotion × intensity × source (N={len(gts)}, Qwen3-14B)")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "gsm8k_emotion_intensity_source.png"), dpi=130)
plt.close()

# 2) Per-question flip analysis: helped vs hurt
fig, ax = plt.subplots(figsize=(13, 5))
tnames_nb = [t for t in tnames_all if t != neutral_key]
helped = [flip_stats[t]["helped"] for t in tnames_nb]
hurt = [-flip_stats[t]["hurt"] for t in tnames_nb]
xs = np.arange(len(tnames_nb))
ax.bar(xs, helped, color="green", label="helped (0→1)")
ax.bar(xs, hurt, color="red", label="hurt (1→0)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(tnames_nb, rotation=75, fontsize=7)
ax.set_ylabel("# questions (helped ↑ / hurt ↓)")
ax.set_title(
    "Per-question flips vs neutral: emotional prefixes create BOTH gains and losses"
)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "gsm8k_perquestion_flips.png"), dpi=130)
plt.close()

# 3) Mean generation length per template
fig, ax = plt.subplots(figsize=(13, 5))
lens = [len_by_template[t] for t in tnames_all]
xs = np.arange(len(tnames_all))
colors = [
    "gray" if t == neutral_key else ("steelblue" if "human" in t else "orange")
    for t in tnames_all
]
ax.bar(xs, lens, color=colors)
ax.set_xticks(xs)
ax.set_xticklabels(tnames_all, rotation=75, fontsize=7)
ax.set_ylabel("Mean generation length (tokens)")
ax.set_title("Do emotional prefixes change reasoning verbosity?")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "gsm8k_gen_length_per_template.png"), dpi=130)
plt.close()

# 4) Correlation: accuracy vs mean gen length
fig, ax = plt.subplots(figsize=(7, 5))
accs = [
    experiment_data["emotional_framing_full"]["GSM8K"]["accuracy_per_template"][t]
    for t in tnames_all
]
ax.scatter(lens, accs)
for t, l, a in zip(tnames_all, lens, accs):
    ax.annotate(
        t.split("|")[0][:4] + ("*" if "strong" in t else ""), (l, a), fontsize=6
    )
ax.set_xlabel("Mean generation length")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs reasoning verbosity across emotion templates")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "gsm8k_acc_vs_length.png"), dpi=130)
plt.close()

# 5) Oracle vs best-fixed vs neutral
best_fixed = max(
    experiment_data["emotional_framing_full"]["GSM8K"]["accuracy_per_template"].values()
)
worst_fixed = min(
    experiment_data["emotional_framing_full"]["GSM8K"]["accuracy_per_template"].values()
)
fig, ax = plt.subplots(figsize=(7, 5))
bars = ["neutral", "worst\nfixed", "best\nfixed", "oracle\n(adaptive)"]
vals = [neutral_acc, worst_fixed, best_fixed, oracle_correct]
ax.bar(bars, vals, color=["gray", "red", "steelblue", "green"])
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy")
ax.set_title("Adaptive prefix selection (oracle) upper bound vs fixed prefixes")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "gsm8k_oracle_vs_fixed.png"), dpi=130)
plt.close()

# --------- Summary ---------
print("\n========== SUMMARY ==========")
print(f"Neutral baseline accuracy: {neutral_acc:.4f}")
print(f"Best fixed prefix acc:     {best_fixed:.4f}")
print(f"Worst fixed prefix acc:    {worst_fixed:.4f}")
print(f"Oracle (adaptive) acc:     {oracle_correct:.4f}")
print(f"Δ(best - neutral) = {best_fixed - neutral_acc:+.4f}")
print(f"Δ(oracle - neutral) = {oracle_correct - neutral_acc:+.4f}")
print("\nPer-template accuracy:")
for t in tnames_all:
    a = experiment_data["emotional_framing_full"]["GSM8K"]["accuracy_per_template"][t]
    l = len_by_template[t]
    print(f"  {t:35s} acc={a:.4f}  mean_len={l:6.1f}")
print("\nPer-template flip stats vs neutral (helped / hurt / Δacc):")
for t, s in flip_stats.items():
    print(
        f"  {t:35s} helped={s['helped']:3d}  hurt={s['hurt']:3d}  Δacc={s['delta_acc']:+.4f}"
    )

print(f"\nSaved outputs to {working_dir}")
