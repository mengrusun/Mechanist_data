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

import os, sys, json, glob, gzip, csv, re, random, importlib, traceback, subprocess

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Install required tokenizer backends BEFORE importing transformers-related tokenizer code
def _pip_install(pkgs):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + pkgs
        )
        return True
    except Exception as e:
        print(f"pip install {pkgs} failed: {e}")
        return False


for pkg in ["tiktoken", "blobfile", "sentencepiece"]:
    try:
        __import__(pkg)
        print(f"{pkg} already available")
    except ImportError:
        print(f"Installing {pkg}...")
        _pip_install([pkg])
        try:
            __import__(pkg)
            print(f"{pkg} installed successfully")
        except ImportError as e:
            print(f"WARNING: {pkg} still not available: {e}")

import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"


def list_files(root, max_files=5000):
    out = []
    if not os.path.isdir(root):
        return out
    for dp, dn, fn in os.walk(root):
        for f in fn:
            out.append(os.path.join(dp, f))
            if len(out) >= max_files:
                return out
    return out


def try_load_csv(p, cols):
    import pandas as pd

    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"  csv read fail {p}: {e}")
        return None
    print(f"  csv {p} columns={list(df.columns)} n={len(df)}")
    for col in cols:
        for c in df.columns:
            if c.lower() == col.lower():
                prompts = df[c].dropna().astype(str).tolist()
                if len(prompts) >= 20:
                    return prompts
    return None


def try_load_json(p, cols):
    try:
        data = json.load(open(p))
    except Exception as e:
        return None
    if isinstance(data, dict):
        for k in ["data", "prompts", "instructions"]:
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
    if isinstance(data, list) and data:
        if isinstance(data[0], str):
            return data
        if isinstance(data[0], dict):
            for col in cols:
                if any(col in d for d in data[:5]):
                    prompts = [d[col] for d in data if col in d]
                    if len(prompts) >= 20:
                        return prompts
    return None


def try_load_jsonl(p, cols):
    prompts = []
    try:
        for line in open(p):
            try:
                d = json.loads(line)
                if isinstance(d, dict):
                    for col in cols:
                        if col in d:
                            prompts.append(d[col])
                            break
                elif isinstance(d, str):
                    prompts.append(d)
            except:
                pass
    except Exception as e:
        return None
    return prompts if len(prompts) >= 20 else None


def try_load_parquet(p, cols):
    import pandas as pd

    try:
        df = pd.read_parquet(p)
    except Exception as e:
        return None
    print(f"  parquet {p} columns={list(df.columns)} n={len(df)}")
    for col in cols:
        for c in df.columns:
            if c.lower() == col.lower():
                prompts = df[c].dropna().astype(str).tolist()
                if len(prompts) >= 20:
                    return prompts
    return None


def find_advbench_prompts():
    root = os.path.join(DATA_DIR, "AdvBench")
    print(f"AdvBench root: {root}")
    if not os.path.isdir(root):
        raise RuntimeError("AdvBench dir missing")
    files = list_files(root)
    cols = ["goal", "prompt", "behavior", "instruction", "text", "question"]
    files_sorted = sorted(
        files, key=lambda x: (0 if "harmful_behaviors" in x.lower() else 1, x)
    )
    for p in files_sorted:
        low = p.lower()
        try:
            if low.endswith(".csv"):
                r = try_load_csv(p, cols)
                if r:
                    return r
            elif low.endswith(".json"):
                r = try_load_json(p, cols)
                if r:
                    return r
            elif low.endswith(".jsonl"):
                r = try_load_jsonl(p, cols)
                if r:
                    return r
            elif low.endswith(".parquet"):
                r = try_load_parquet(p, cols)
                if r:
                    return r
        except Exception as e:
            print(f"  skip {p}: {e}")
    raise RuntimeError("AdvBench not found")


def find_alpaca_prompts():
    root = os.path.join(DATA_DIR, "Alpaca")
    print(f"Alpaca root: {root}")
    if not os.path.isdir(root):
        raise RuntimeError("Alpaca dir missing")
    files = list_files(root)
    import pandas as pd

    for p in sorted(files):
        low = p.lower()
        try:
            if low.endswith(".parquet"):
                df = pd.read_parquet(p)
                print(f"  parquet {p} cols={list(df.columns)} n={len(df)}")
                if "instruction" in df.columns:
                    inp_col = (
                        df["input"].astype(str)
                        if "input" in df.columns
                        else [""] * len(df)
                    )
                    prompts = [
                        str(a) + (("\n" + str(b)) if b else "")
                        for a, b in zip(df["instruction"], inp_col)
                    ]
                    if len(prompts) >= 20:
                        return prompts
            elif low.endswith(".json"):
                data = json.load(open(p))
                if isinstance(data, list):
                    prompts = []
                    for d in data:
                        if isinstance(d, dict):
                            instr = d.get("instruction") or d.get("prompt")
                            inp = d.get("input", "") or ""
                            if instr:
                                prompts.append(
                                    str(instr) + (("\n" + str(inp)) if inp else "")
                                )
                    if len(prompts) >= 20:
                        return prompts
        except Exception as e:
            print(f"  skip {p}: {e}")
    raise RuntimeError("Alpaca not found")


harmful_prompts = find_advbench_prompts()
benign_prompts = find_alpaca_prompts()

random.seed(0)
random.shuffle(harmful_prompts)
random.shuffle(benign_prompts)

N_PER_CLASS = 200
harmful_prompts = harmful_prompts[:N_PER_CLASS]
benign_prompts = benign_prompts[:N_PER_CLASS]
print(f"Using {len(harmful_prompts)} harmful and {len(benign_prompts)} benign prompts")


# Locate model path - check for nested directory structure
def find_model_path():
    candidates = []
    base = os.path.join(MODEL_DIR, "Meta-Llama-3-8B-Instruct")
    if os.path.isdir(base):
        candidates.append(base)
        # Check nested
        nested = os.path.join(base, "Meta-Llama-3-8B-Instruct")
        if os.path.isdir(nested):
            candidates.append(nested)
    # Scan for other llama-3 8b instruct dirs
    if os.path.isdir(MODEL_DIR):
        for name in os.listdir(MODEL_DIR):
            full = os.path.join(MODEL_DIR, name)
            if (
                os.path.isdir(full)
                and "llama-3" in name.lower()
                and "8b" in name.lower()
                and "instruct" in name.lower()
            ):
                if full not in candidates:
                    candidates.append(full)
    # Pick the one with tokenizer files
    for c in candidates:
        files = os.listdir(c) if os.path.isdir(c) else []
        has_tok = any(
            f in files
            for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        )
        has_model = any(f.endswith(".safetensors") or f.endswith(".bin") for f in files)
        print(
            f"  candidate {c}: files={files[:10]} has_tok={has_tok} has_model={has_model}"
        )
        if has_tok and has_model:
            return c
    # Return first candidate anyway
    return candidates[0] if candidates else base


MODEL_PATH = find_model_path()
print(f"Model path: {MODEL_PATH}")
print(f"Files in model dir: {os.listdir(MODEL_PATH)}")

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer(path):
    errors = []
    strategies = [
        ("fast_local", dict(local_files_only=True, use_fast=True)),
        ("slow_local", dict(local_files_only=True, use_fast=False)),
        (
            "fast_trust",
            dict(local_files_only=True, use_fast=True, trust_remote_code=True),
        ),
    ]
    for name, kw in strategies:
        try:
            tok = AutoTokenizer.from_pretrained(path, **kw)
            print(f"Loaded tokenizer via {name}")
            return tok
        except Exception as e:
            errors.append(f"{name}: {e}")
            print(f"  tokenizer attempt {name} failed: {e}")

    # Try loading directly from tokenizer.json using PreTrainedTokenizerFast
    try:
        from transformers import PreTrainedTokenizerFast

        tj = os.path.join(path, "tokenizer.json")
        if os.path.isfile(tj):
            tok = PreTrainedTokenizerFast(tokenizer_file=tj)
            # Set special tokens for Llama-3
            special = {
                "bos_token": "<|begin_of_text|>",
                "eos_token": "<|end_of_text|>",
                "pad_token": "<|end_of_text|>",
            }
            tok.add_special_tokens(special)
            print("Loaded tokenizer via PreTrainedTokenizerFast(tokenizer_file=...)")
            return tok
    except Exception as e:
        errors.append(f"PreTrainedTokenizerFast: {e}")
        print(f"  PreTrainedTokenizerFast failed: {e}")

    raise RuntimeError("All tokenizer loading strategies failed:\n" + "\n".join(errors))


tokenizer = load_tokenizer(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set a chat template if missing (Llama-3 chat template)
LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)
try:
    _ = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False
    )
except Exception:
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    print("Set fallback Llama-3 chat template")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
)
model.to(device)
model.eval()
n_layers = model.config.num_hidden_layers
print(f"Model loaded. num_hidden_layers={n_layers}")


def format_prompt(text):
    try:
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return f"<|user|>\n{text}\n<|assistant|>\n"


LAYERS_TO_PROBE = [8, 12, 16, 20, 24, 28]


@torch.no_grad()
def get_activations(prompts):
    acts = {L: [] for L in LAYERS_TO_PROBE}
    verified = False
    for i, p in enumerate(prompts):
        text = format_prompt(p)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
            device
        )
        out = model(**enc, output_hidden_states=True, use_cache=False)
        last_idx = enc["attention_mask"].sum(dim=1).item() - 1
        if not verified:
            tok_id = enc["input_ids"][0, last_idx].item()
            print(f"  final instr token decoded: {repr(tokenizer.decode([tok_id]))}")
            verified = True
        for L in LAYERS_TO_PROBE:
            h = out.hidden_states[L][0, last_idx, :].float().cpu().numpy()
            acts[L].append(h)
        if (i + 1) % 50 == 0:
            print(f"  extracted {i+1}/{len(prompts)}")
    return {L: np.stack(v) for L, v in acts.items()}


print("Extracting activations for harmful prompts...")
harm_acts = get_activations(harmful_prompts)
print("Extracting activations for benign prompts...")
ben_acts = get_activations(benign_prompts)

del model
torch.cuda.empty_cache()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

experiment_data = {
    "llama3_advbench_alpaca": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "per_layer": {},
    }
}

results = {}
for L in LAYERS_TO_PROBE:
    X_h = harm_acts[L]
    X_b = ben_acts[L]
    X = np.concatenate([X_h, X_b], axis=0)
    y = np.concatenate([np.ones(len(X_h)), np.zeros(len(X_b))]).astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    mu_h = X_tr[y_tr == 1].mean(0)
    mu_b = X_tr[y_tr == 0].mean(0)
    dir_vec = mu_h - mu_b
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)

    p_tr = X_tr @ dir_vec
    p_te = X_te @ dir_vec
    clf = LogisticRegression(max_iter=1000)
    clf.fit(p_tr.reshape(-1, 1), y_tr)
    train_acc = clf.score(p_tr.reshape(-1, 1), y_tr)
    val_acc = clf.score(p_te.reshape(-1, 1), y_te)

    rng = np.random.default_rng(0)
    rand_accs = []
    for _ in range(5):
        r = rng.standard_normal(dir_vec.shape[0])
        r /= np.linalg.norm(r) + 1e-8
        clf_r = LogisticRegression(max_iter=1000)
        clf_r.fit((X_tr @ r).reshape(-1, 1), y_tr)
        rand_accs.append(clf_r.score((X_te @ r).reshape(-1, 1), y_te))
    rand_acc = float(np.mean(rand_accs))

    y_tr_shuf = y_tr.copy()
    rng.shuffle(y_tr_shuf)
    mu_h_s = X_tr[y_tr_shuf == 1].mean(0)
    mu_b_s = X_tr[y_tr_shuf == 0].mean(0)
    d_s = mu_h_s - mu_b_s
    d_s /= np.linalg.norm(d_s) + 1e-8
    clf_s = LogisticRegression(max_iter=1000)
    clf_s.fit((X_tr @ d_s).reshape(-1, 1), y_tr_shuf)
    shuf_acc = clf_s.score((X_te @ d_s).reshape(-1, 1), y_te)

    val_loss = 1.0 - val_acc
    train_loss = 1.0 - train_acc
    print(
        f"Layer {L}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} random={rand_acc:.4f} shuffled={shuf_acc:.4f}"
    )
    print(f"Epoch {L}: validation_loss = {val_loss:.4f}")

    experiment_data["llama3_advbench_alpaca"]["metrics"]["train"].append(
        {"layer": L, "acc": train_acc}
    )
    experiment_data["llama3_advbench_alpaca"]["metrics"]["val"].append(
        {"layer": L, "acc": val_acc, "random": rand_acc, "shuffled": shuf_acc}
    )
    experiment_data["llama3_advbench_alpaca"]["losses"]["train"].append(
        {"layer": L, "loss": train_loss}
    )
    experiment_data["llama3_advbench_alpaca"]["losses"]["val"].append(
        {"layer": L, "loss": val_loss}
    )
    experiment_data["llama3_advbench_alpaca"]["per_layer"][L] = {
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "random_acc": float(rand_acc),
        "shuffled_acc": float(shuf_acc),
        "direction_norm": float(np.linalg.norm(mu_h - mu_b)),
    }
    results[L] = (val_acc, rand_acc)

best_L = max(results.keys(), key=lambda k: results[k][0])
best_val = results[best_L][0]
print(f"\nBEST layer={best_L} harmfulness_direction_probe_accuracy={best_val:.4f}")
experiment_data["llama3_advbench_alpaca"]["best_layer"] = int(best_L)
experiment_data["llama3_advbench_alpaca"]["harmfulness_direction_probe_accuracy"] = (
    float(best_val)
)

X_h = harm_acts[best_L]
X_b = ben_acts[best_L]
X = np.concatenate([X_h, X_b], 0)
y = np.concatenate([np.ones(len(X_h)), np.zeros(len(X_b))]).astype(np.int64)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
mu_h = X_tr[y_tr == 1].mean(0)
mu_b = X_tr[y_tr == 0].mean(0)
d = mu_h - mu_b
d /= np.linalg.norm(d) + 1e-8
clf = LogisticRegression(max_iter=1000)
clf.fit((X_tr @ d).reshape(-1, 1), y_tr)
preds = clf.predict((X_te @ d).reshape(-1, 1))
experiment_data["llama3_advbench_alpaca"]["predictions"] = preds.tolist()
experiment_data["llama3_advbench_alpaca"]["ground_truth"] = y_te.tolist()

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(results.keys())
    vals = [results[L][0] for L in layers]
    rands = [results[L][1] for L in layers]
    plt.figure(figsize=(6, 4))
    plt.plot(layers, vals, "o-", label="diff-of-means")
    plt.plot(layers, rands, "s--", label="random dir")
    plt.xlabel("Layer")
    plt.ylabel("Val Acc")
    plt.title("Harmfulness Probe Accuracy per Layer (Llama-3-8B AdvBench/Alpaca)")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_probe_per_layer.png"),
        bbox_inches="tight",
    )
    print("Saved plot")
except Exception as e:
    print("plot failed:", e)

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
print(f"FINAL harmfulness_direction_probe_accuracy = {best_val:.4f}")
