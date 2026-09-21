import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

os.environ.setdefault("HF_TOKEN", "<Your_token>")
os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MAX_ITER_GRID = [1000, 2000, 5000, 10000]

experiment_data = {
    "probe_max_iter": {
        "triviaqa_gemma3_27b_pt": {
            "hyperparams": MAX_ITER_GRID,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "per_config": {},  # keyed by max_iter
            "raw_confidences": [],
            "raw_correctness": [],
            "best_max_iter": None,
            "best_cache_probe_accuracy": None,
        }
    }
}

# ---------- Load dataset ----------
from datasets import load_dataset

print("Loading TriviaQA...")
try:
    ds = load_dataset(
        "mandarjoshi/trivia_qa", "rc.nocontext", cache_dir=DATA_DIR, split="validation"
    )
except Exception as e:
    print(f"Primary load failed: {e}; retrying with default cache")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")

print(f"Loaded {len(ds)} examples. columns={ds.column_names}")
assert len(ds) > 100
assert "question" in ds.column_names and "answer" in ds.column_names

N_SAMPLES = 200
ds = ds.shuffle(seed=42).select(range(N_SAMPLES))

# ---------- Load model ----------
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_path = os.path.join(MODEL_DIR, "gemma-3-27b-pt")
if not os.path.isdir(model_path):
    model_path = "google/gemma-3-27b-pt"
print(f"Loading model from {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    lm_model = model
except Exception as e:
    print(f"CausalLM load failed: {e}; trying Gemma3ForConditionalGeneration")
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    lm_model = getattr(model, "language_model", model)

model.eval()
text_cfg = getattr(model.config, "text_config", model.config)
num_layers = getattr(text_cfg, "num_hidden_layers", None)
hidden_size = getattr(text_cfg, "hidden_size", None)
print(f"num_layers={num_layers} hidden_size={hidden_size}")

FEWSHOT = """Answer the question with a short answer, then rate your confidence from 0 to 100.

Q: What is the capital of France?
A: Paris
Confidence: 98

Q: Who painted the Mona Lisa?
A: Leonardo da Vinci
Confidence: 95

Q: In what year did the Ottoman Empire officially end?
A: 1922
Confidence: 72

Q: What is the airspeed velocity of an unladen swallow?
A: About 24 miles per hour
Confidence: 35

"""


def build_answer_prompt(question):
    return FEWSHOT + f"Q: {question}\nA:"


def build_confidence_prompt(question, answer):
    return FEWSHOT + f"Q: {question}\nA: {answer}\nConfidence:"


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_correct(pred, aliases):
    p = normalize(pred)
    if not p:
        return 0
    for a in aliases:
        an = normalize(a)
        if an and (an == p or an in p or p in an):
            return 1
    return 0


digit_token_ids = {}
for d in range(10):
    for cand in [f" {d}", f"{d}"]:
        toks = tokenizer.encode(cand, add_special_tokens=False)
        if len(toks) == 1:
            digit_token_ids.setdefault(d, []).append(toks[0])
digit_ids_flat = {d: list(set(v)) for d, v in digit_token_ids.items()}
print("digit token ids:", digit_ids_flat)


@torch.no_grad()
def generate_answer(prompt, max_new=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    ans = text.split("\n")[0].strip()
    return ans


@torch.no_grad()
def get_hidden_states_and_confidence(question, answer):
    post_answer_prompt = FEWSHOT + f"Q: {question}\nA: {answer}"
    full_prompt = FEWSHOT + f"Q: {question}\nA: {answer}\nConfidence:"

    inputs_post = tokenizer(post_answer_prompt, return_tensors="pt").to(device)
    out_post = model(**inputs_post, output_hidden_states=True, use_cache=False)
    hs_all = out_post.hidden_states
    post_pos = inputs_post["input_ids"].shape[1] - 1
    layer_vecs = np.stack(
        [hs[0, post_pos, :].float().cpu().numpy() for hs in hs_all], axis=0
    )
    pre_answer_prompt = FEWSHOT + f"Q: {question}\nA:"
    pre_len = tokenizer(pre_answer_prompt, return_tensors="pt")["input_ids"].shape[1]
    pre_pos = min(pre_len - 1, post_pos)
    layer_vecs_pre = np.stack(
        [hs[0, pre_pos, :].float().cpu().numpy() for hs in hs_all], axis=0
    )

    inputs_full = tokenizer(full_prompt, return_tensors="pt").to(device)
    out_full = model(**inputs_full, use_cache=False)
    logits = out_full.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

    digit_probs = np.zeros(10)
    for d in range(10):
        for tid in digit_ids_flat.get(d, []):
            digit_probs[d] += probs[tid]
    Z = digit_probs.sum()
    if Z < 1e-8:
        expected_first = 5.0
    else:
        digit_probs = digit_probs / Z
        expected_first = float(np.sum(np.arange(10) * digit_probs))

    with torch.no_grad():
        gen = model.generate(
            **inputs_full,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    conf_text = tokenizer.decode(
        gen[0][inputs_full["input_ids"].shape[1] :], skip_special_tokens=True
    )
    m = re.search(r"(\d+)", conf_text)
    verbal_conf_argmax = int(m.group(1)) if m else -1
    if verbal_conf_argmax > 100:
        verbal_conf_argmax = 100

    continuous_conf = expected_first * 10.0

    return layer_vecs, layer_vecs_pre, continuous_conf, verbal_conf_argmax


all_post_hs = []
all_pre_hs = []
all_conf_cont = []
all_conf_argmax = []
all_correct = []

from tqdm import tqdm

processed = 0
for i, ex in enumerate(tqdm(ds, desc="processing")):
    try:
        q = ex["question"]
        aliases = (
            ex["answer"].get("aliases", [])
            + [ex["answer"].get("value", "")]
            + ex["answer"].get("normalized_aliases", [])
        )
        aliases = [a for a in aliases if a]
        ans_prompt = build_answer_prompt(q)
        pred = generate_answer(ans_prompt, max_new=15)
        correct = is_correct(pred, aliases)
        layer_vecs, layer_vecs_pre, cont_conf, argmax_conf = (
            get_hidden_states_and_confidence(q, pred)
        )
        all_post_hs.append(layer_vecs)
        all_pre_hs.append(layer_vecs_pre)
        all_conf_cont.append(cont_conf)
        all_conf_argmax.append(argmax_conf)
        all_correct.append(correct)
        processed += 1
        if i < 5:
            print(
                f"[{i}] Q: {q[:80]} | pred={pred[:40]} | argmax_conf={argmax_conf} | cont_conf={cont_conf:.2f} | correct={correct}"
            )
    except Exception as e:
        print(f"Skipping example {i}: {e}")
        continue

print(f"Processed {processed} examples")

all_post_hs = np.stack(all_post_hs, axis=0)
all_pre_hs = np.stack(all_pre_hs, axis=0)
all_conf_cont = np.array(all_conf_cont)
all_conf_argmax = np.array(all_conf_argmax)
all_correct = np.array(all_correct)

print(f"Hidden states shape: {all_post_hs.shape}")
print(f"Correctness rate: {all_correct.mean():.3f}")

ed = experiment_data["probe_max_iter"]["triviaqa_gemma3_27b_pt"]
ed["raw_confidences"] = all_conf_cont.tolist()
ed["raw_correctness"] = all_correct.tolist()

try:
    q1, q2 = np.quantile(all_conf_cont, [1 / 3, 2 / 3])
    conf_bins = np.digitize(all_conf_cont, [q1, q2])
except Exception:
    conf_bins = (all_conf_cont > np.median(all_conf_cont)).astype(int)

unique, counts = np.unique(conf_bins, return_counts=True)
print(f"Confidence bin distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

maj_conf = counts.max() / counts.sum()
maj_corr = max(all_correct.mean(), 1 - all_correct.mean())
ed["majority_baseline_conf"] = float(maj_conf)
ed["majority_baseline_corr"] = float(maj_corr)
print(f"Majority baseline conf: {maj_conf:.4f}, corr: {maj_corr:.4f}")

N, Lp1, H = all_post_hs.shape
layer_indices = sorted(set([0] + list(range(0, Lp1, 4)) + [Lp1 - 1]))
print(f"Probing layers: {layer_indices}")

idx_train, idx_test = train_test_split(
    np.arange(N),
    test_size=0.3,
    random_state=42,
    stratify=conf_bins if len(unique) > 1 else None,
)

# Precompute scaled features per layer to save time
scaled_cache = {}
for L in layer_indices:
    s1 = StandardScaler()
    X_post_tr = s1.fit_transform(all_post_hs[:, L, :][idx_train])
    X_post_te = s1.transform(all_post_hs[:, L, :][idx_test])
    s2 = StandardScaler()
    X_pre_tr = s2.fit_transform(all_pre_hs[:, L, :][idx_train])
    X_pre_te = s2.transform(all_pre_hs[:, L, :][idx_test])
    scaled_cache[L] = (X_post_tr, X_post_te, X_pre_tr, X_pre_te)

best_overall_acc = -1.0
best_overall_mi = None

for mi in MAX_ITER_GRID:
    print(f"\n=== max_iter={mi} ===")
    per_layer_conf_acc = []
    per_layer_corr_acc = []
    per_layer_control_conf_acc = []
    for L in layer_indices:
        X_post_tr, X_post_te, X_pre_tr, X_pre_te = scaled_cache[L]
        try:
            clf = LogisticRegression(max_iter=mi, C=1.0)
            clf.fit(X_post_tr, conf_bins[idx_train])
            acc_conf = clf.score(X_post_te, conf_bins[idx_test])
        except Exception as e:
            print(f"conf probe layer {L} fail: {e}")
            acc_conf = float("nan")
        try:
            if len(np.unique(all_correct)) > 1:
                clf2 = LogisticRegression(max_iter=mi, C=1.0)
                clf2.fit(X_post_tr, all_correct[idx_train])
                acc_corr = clf2.score(X_post_te, all_correct[idx_test])
            else:
                acc_corr = maj_corr
        except Exception as e:
            acc_corr = float("nan")
        try:
            clf3 = LogisticRegression(max_iter=mi, C=1.0)
            clf3.fit(X_pre_tr, conf_bins[idx_train])
            acc_ctrl = clf3.score(X_pre_te, conf_bins[idx_test])
        except Exception as e:
            acc_ctrl = float("nan")
        per_layer_conf_acc.append(acc_conf)
        per_layer_corr_acc.append(acc_corr)
        per_layer_control_conf_acc.append(acc_ctrl)
        print(
            f"  Layer {L}: conf={acc_conf:.4f} corr={acc_corr:.4f} ctrl={acc_ctrl:.4f}"
        )

    per_layer_conf_acc = np.array(per_layer_conf_acc)
    per_layer_corr_acc = np.array(per_layer_corr_acc)
    per_layer_control_conf_acc = np.array(per_layer_control_conf_acc)
    best_conf_idx = int(np.nanargmax(per_layer_conf_acc))
    best_corr_idx = int(np.nanargmax(per_layer_corr_acc))
    best_layer_conf = layer_indices[best_conf_idx]
    best_layer_corr = layer_indices[best_corr_idx]
    best_conf_acc = float(per_layer_conf_acc[best_conf_idx])
    best_corr_acc = float(per_layer_corr_acc[best_corr_idx])
    print(
        f"  Best: conf layer={best_layer_conf} acc={best_conf_acc:.4f} | corr layer={best_layer_corr} acc={best_corr_acc:.4f}"
    )

    ed["per_config"][str(mi)] = {
        "max_iter": mi,
        "layer_indices": layer_indices,
        "per_layer_confidence_acc": per_layer_conf_acc.tolist(),
        "per_layer_correctness_acc": per_layer_corr_acc.tolist(),
        "per_layer_control_conf_acc": per_layer_control_conf_acc.tolist(),
        "best_layer_conf": best_layer_conf,
        "best_layer_corr": best_layer_corr,
        "cache_probe_accuracy": best_conf_acc,
        "corr_probe_accuracy": best_corr_acc,
    }
    ed["metrics"]["val"].append(
        {
            "max_iter": mi,
            "cache_probe_accuracy": best_conf_acc,
            "corr_probe_accuracy": best_corr_acc,
        }
    )
    ed["losses"]["val"].append({"max_iter": mi, "val_loss": float(1.0 - best_conf_acc)})

    if best_conf_acc > best_overall_acc:
        best_overall_acc = best_conf_acc
        best_overall_mi = mi

ed["best_max_iter"] = best_overall_mi
ed["best_cache_probe_accuracy"] = best_overall_acc
ed["layer_indices"] = layer_indices
print(
    f"\nBest max_iter={best_overall_mi} with cache_probe_accuracy={best_overall_acc:.4f}"
)
print(f"CACHE_PROBE_ACCURACY = {best_overall_acc:.4f}")

# Visualization: one plot per max_iter, overlaid
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for mi in MAX_ITER_GRID:
    cfg = ed["per_config"][str(mi)]
    ax.plot(
        layer_indices,
        cfg["per_layer_confidence_acc"],
        "o-",
        label=f"conf (max_iter={mi})",
    )
ax.axhline(
    maj_conf, color="gray", linestyle=":", label=f"Majority conf ({maj_conf:.3f})"
)
ax.set_xlabel("Layer")
ax.set_ylabel("Probe accuracy (confidence)")
ax.set_title("Probe accuracy vs. layer across max_iter")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(working_dir, "triviaqa_gemma3_27b_pt_max_iter_sweep.png"), dpi=120
)
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy and figure.")
