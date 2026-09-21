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

import os, re, json, gc, string, math, time, sys, traceback

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"
MODEL_PATH = os.path.join(MODEL_DIR, "Llama-3.1-8B-Instruct")

# ---------- Load model ----------
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading tokenizer/model from", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True
).to(device)
model.eval()
n_layers = model.config.num_hidden_layers
print(f"Model loaded. num_hidden_layers={n_layers}")

# ---------- Load TriviaQA ----------
from datasets import load_dataset, load_from_disk

tqa = None
tqa_root = os.path.join(DATA_DIR, "trivia_qa")
last_err = None
# try load_from_disk on subpaths
for cand in [
    tqa_root,
    os.path.join(tqa_root, "default"),
    os.path.join(tqa_root, "validation"),
    os.path.join(tqa_root, "rc.nocontext"),
]:
    if os.path.isdir(cand):
        try:
            ds = load_from_disk(cand)
            if hasattr(ds, "keys"):
                if "validation" in ds:
                    tqa = ds["validation"]
                elif "train" in ds:
                    tqa = ds["train"]
            else:
                tqa = ds
            if tqa is not None:
                print("Loaded via load_from_disk:", cand)
                break
        except Exception as e:
            last_err = e
if tqa is None:
    for cfg in ["default", "rc.nocontext", "rc", "unfiltered.nocontext", "unfiltered"]:
        try:
            tqa = load_dataset("trivia_qa", cfg, cache_dir=DATA_DIR, split="validation")
            print("Loaded via load_dataset config", cfg)
            break
        except Exception as e:
            last_err = e
if tqa is None:
    # try parquet files under tqa_root
    pqs = []
    for root, _, files in os.walk(tqa_root):
        for f in files:
            if f.endswith(".parquet") and "validation" in f.lower():
                pqs.append(os.path.join(root, f))
    if pqs:
        tqa = load_dataset("parquet", data_files=pqs, split="train")
        print("Loaded via parquet:", pqs[:2])
if tqa is None:
    raise RuntimeError(f"Failed to load TriviaQA; last_err={last_err}")
print("TriviaQA size:", len(tqa), "features:", list(tqa.features.keys()))

# ---------- Prep questions ----------
N = 400
rng = np.random.default_rng(0)
idxs = rng.choice(len(tqa), size=min(N, len(tqa)), replace=False).tolist()


def get_aliases(ex):
    a = ex.get("answer", {})
    al = []
    if isinstance(a, dict):
        al = list(a.get("aliases", []) or []) + list(
            a.get("normalized_aliases", []) or []
        )
        if a.get("value"):
            al.append(a["value"])
        if a.get("normalized_value"):
            al.append(a["normalized_value"])
    return [s for s in al if isinstance(s, str) and s.strip()]


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_correct(pred, aliases):
    p = normalize(pred)
    if not p:
        return 0
    for a in aliases:
        na = normalize(a)
        if not na:
            continue
        if na == p or na in p or p in na:
            return 1
    return 0


# ---------- Generation with hidden states ----------
PROMPT_TMPL = (
    "Answer the trivia question with a short answer, then rate your confidence 0-100.\n"
    "Format strictly:\nAnswer: <short answer>\nConfidence: <0-100>%\n\n"
    "Question: {q}\n"
)

# layers to probe
target_layers = sorted(set([n_layers // 4, n_layers // 2, (3 * n_layers) // 4]))
print("Target layers:", target_layers)

hidden_by_layer = {L: [] for L in target_layers}
correct_labels = []
verb_conf = []
answers_text = []
questions_text = []
aliases_list = []

MAX_NEW = 40
t0 = time.time()
for i, idx in enumerate(idxs):
    ex = tqa[int(idx)]
    q = ex["question"]
    aliases = get_aliases(ex)
    if not aliases:
        continue
    messages = [{"role": "user", "content": PROMPT_TMPL.format(q=q)}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = PROMPT_TMPL.format(q=q)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )
    gen_ids = out.sequences[0, prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # parse answer + confidence
    m_ans = re.search(r"Answer\s*:\s*(.+)", gen_text)
    m_cnf = re.search(r"Confidence\s*:\s*(\d+(?:\.\d+)?)\s*%?", gen_text)
    ans = (
        m_ans.group(1).split("\n")[0].strip()
        if m_ans
        else gen_text.strip().split("\n")[0]
    )
    conf = float(m_cnf.group(1)) if m_cnf else np.nan
    corr = is_correct(ans, aliases)

    # forward the full sequence once to grab hidden states at the answer span
    full_ids = out.sequences[:, :].to(device)
    with torch.no_grad():
        hs = model(full_ids, output_hidden_states=True, use_cache=False).hidden_states
    # locate the answer span in the generated tokens
    if m_ans:
        # take hidden states of the last few tokens of the generation as answer rep
        # simple: mean over generated tokens (excluding trailing pads)
        gen_slice = slice(prompt_len, full_ids.shape[1])
    else:
        gen_slice = slice(prompt_len, full_ids.shape[1])
    for L in target_layers:
        h = hs[L][0, gen_slice, :].float().mean(dim=0).cpu().numpy()
        hidden_by_layer[L].append(h)
    correct_labels.append(corr)
    verb_conf.append(conf)
    answers_text.append(ans)
    questions_text.append(q)
    aliases_list.append(aliases)
    if (i + 1) % 25 == 0:
        elapsed = time.time() - t0
        acc_far = np.mean(correct_labels)
        print(
            f'[{i+1}/{len(idxs)}] acc_so_far={acc_far:.3f} elapsed={elapsed:.1f}s last_gen="{gen_text[:80]}"'
        )
    del out, hs
    if (i + 1) % 50 == 0:
        torch.cuda.empty_cache()
        gc.collect()

print(
    "Total generated:",
    len(correct_labels),
    "Overall accuracy:",
    np.mean(correct_labels),
)

correct = np.array(correct_labels, dtype=np.int64)
conf_arr = np.array(verb_conf, dtype=np.float32)
valid_conf = ~np.isnan(conf_arr)
print(f"Valid confidence parsed: {valid_conf.sum()}/{len(conf_arr)}")
# fill nan with median for probe fitting
conf_filled = np.where(
    valid_conf, conf_arr, np.nanmedian(conf_arr) if valid_conf.any() else 50.0
)

# ---------- Probes per layer ----------
experiment_data = {
    "trivia_qa": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": correct.tolist(),
        "verbalized_confidence": conf_arr.tolist(),
        "per_layer": {},
    }
}


def ece(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (probs >= bins[i]) & (
            probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1]
        )
        if m.sum() > 0:
            e += (m.sum() / len(probs)) * abs(labels[m].mean() - probs[m].mean())
    return float(e)


# train/test split
n = len(correct)
perm = rng.permutation(n)
n_train = int(0.7 * n)
tr, te = perm[:n_train], perm[n_train:]

ortho_scores = {}
for L in target_layers:
    X = np.stack(hidden_by_layer[L], axis=0).astype(np.float32)
    Xtr, Xte = X[tr], X[te]
    ytr_c, yte_c = correct[tr], correct[te]
    ytr_v, yte_v = conf_filled[tr], conf_filled[te]

    # correctness probe
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr_c)
    p_te = clf.predict_proba(Xte)[:, 1]
    acc_c = accuracy_score(yte_c, (p_te > 0.5).astype(int))
    try:
        auc_c = roc_auc_score(yte_c, p_te)
    except Exception:
        auc_c = float("nan")
    w_corr = clf.coef_[0]

    # verbalized confidence probe (ridge regression on continuous confidence)
    reg = Ridge(alpha=1.0)
    reg.fit(Xtr, ytr_v)
    yv_pred = reg.predict(Xte)
    mse_v = float(np.mean((yv_pred - yte_v) ** 2))
    w_conf = reg.coef_

    # orthogonality
    def cos(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    c = cos(w_corr, w_conf)
    ortho = 1.0 - abs(c)
    ortho_scores[L] = ortho

    # random-direction control
    rand_ctrls = []
    for _ in range(20):
        r = rng.normal(size=w_corr.shape)
        rand_ctrls.append(1.0 - abs(cos(w_corr, r)))
    ctrl_mean = float(np.mean(rand_ctrls))

    # calibration
    conf_prob = np.clip(conf_filled[te] / 100.0, 0, 1)
    ece_verb = ece(conf_prob, yte_c)
    ece_probe = ece(p_te, yte_c)

    print(
        f"Layer {L}: probe_acc={acc_c:.3f} auc={auc_c:.3f} conf_mse={mse_v:.2f} "
        f"ortho=1-|cos|={ortho:.4f} rand_ctrl={ctrl_mean:.4f} "
        f"ece_verb={ece_verb:.3f} ece_probe={ece_probe:.3f}"
    )
    print(f"Epoch {L}: validation_loss = {mse_v:.4f}")  # per-layer as pseudo-epoch

    experiment_data["trivia_qa"]["metrics"]["val"].append(
        {
            "layer": int(L),
            "probe_acc": float(acc_c),
            "probe_auc": float(auc_c),
            "ortho": float(ortho),
            "rand_ctrl_ortho": float(ctrl_mean),
            "ece_verbalized": float(ece_verb),
            "ece_probe": float(ece_probe),
        }
    )
    experiment_data["trivia_qa"]["losses"]["val"].append(
        {"layer": int(L), "conf_mse": mse_v}
    )
    experiment_data["trivia_qa"]["per_layer"][int(L)] = {
        "w_corr": w_corr.astype(np.float32),
        "w_conf": w_conf.astype(np.float32),
        "probe_acc": float(acc_c),
        "probe_auc": float(auc_c),
        "ortho": float(ortho),
        "rand_ctrl_ortho": float(ctrl_mean),
    }

# headline metric = middle layer
mid = target_layers[len(target_layers) // 2]
head_ortho = ortho_scores[mid]
print(
    f"\n=== HEADLINE: subspace_orthogonality_score (layer {mid}) = {head_ortho:.4f} ==="
)
experiment_data["trivia_qa"]["subspace_orthogonality_score"] = float(head_ortho)
experiment_data["trivia_qa"]["headline_layer"] = int(mid)

# ---------- Plots ----------
fig, ax = plt.subplots(figsize=(6, 4))
Ls = sorted(ortho_scores.keys())
ax.bar([str(L) for L in Ls], [ortho_scores[L] for L in Ls], color="steelblue")
ax.axhline(1.0, color="k", ls="--", lw=0.5)
ax.set_xlabel("Layer")
ax.set_ylabel("1 - |cos(w_corr, w_conf)|")
ax.set_title("Subspace orthogonality per layer (TriviaQA / Llama-3.1-8B-Instruct)")
fig.tight_layout()
fig.savefig(os.path.join(working_dir, "orthogonality_by_layer_triviaqa.png"), dpi=140)
plt.close(fig)

# calibration histogram
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(conf_arr[valid_conf], bins=20, color="salmon", edgecolor="k")
ax.set_xlabel("Verbalized confidence (%)")
ax.set_ylabel("Count")
ax.set_title(f"Verbalized confidence distribution (acc={correct.mean():.2f})")
fig.tight_layout()
fig.savefig(
    os.path.join(working_dir, "verbalized_confidence_hist_triviaqa.png"), dpi=140
)
plt.close(fig)

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy and plots to", working_dir)
