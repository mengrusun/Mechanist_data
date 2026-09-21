import os, sys, json, glob, gzip, csv, re, random, importlib, traceback, subprocess

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


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
    except ImportError:
        _pip_install([pkg])

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
        return None
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
    except Exception:
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
    except Exception:
        return None
    return prompts if len(prompts) >= 20 else None


def try_load_parquet(p, cols):
    import pandas as pd

    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    for col in cols:
        for c in df.columns:
            if c.lower() == col.lower():
                prompts = df[c].dropna().astype(str).tolist()
                if len(prompts) >= 20:
                    return prompts
    return None


def find_advbench_prompts():
    root = os.path.join(DATA_DIR, "AdvBench")
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
        except Exception:
            pass
    raise RuntimeError("AdvBench not found")


def find_alpaca_prompts():
    root = os.path.join(DATA_DIR, "Alpaca")
    files = list_files(root)
    import pandas as pd

    for p in sorted(files):
        low = p.lower()
        try:
            if low.endswith(".parquet"):
                df = pd.read_parquet(p)
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
        except Exception:
            pass
    raise RuntimeError("Alpaca not found")


harmful_prompts_all = find_advbench_prompts()
benign_prompts_all = find_alpaca_prompts()

random.seed(0)
random.shuffle(harmful_prompts_all)
random.shuffle(benign_prompts_all)

max_avail = min(len(harmful_prompts_all), len(benign_prompts_all))
print(
    f"Available: {len(harmful_prompts_all)} harmful, {len(benign_prompts_all)} benign; max_per_class={max_avail}"
)

N_CANDIDATES = [100, 200, 400, 500]
N_CANDIDATES = sorted(set([min(n, max_avail) for n in N_CANDIDATES]))
N_MAX = max(N_CANDIDATES)
print(f"Sweeping N_PER_CLASS ∈ {N_CANDIDATES}, extracting up to {N_MAX} per class")

harmful_prompts_pool = harmful_prompts_all[:N_MAX]
benign_prompts_pool = benign_prompts_all[:N_MAX]


def find_model_path():
    candidates = []
    base = os.path.join(MODEL_DIR, "Meta-Llama-3-8B-Instruct")
    if os.path.isdir(base):
        candidates.append(base)
        nested = os.path.join(base, "Meta-Llama-3-8B-Instruct")
        if os.path.isdir(nested):
            candidates.append(nested)
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
    for c in candidates:
        files = os.listdir(c) if os.path.isdir(c) else []
        has_tok = any(
            f in files
            for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        )
        has_model = any(f.endswith(".safetensors") or f.endswith(".bin") for f in files)
        if has_tok and has_model:
            return c
    return candidates[0] if candidates else base


MODEL_PATH = find_model_path()
print(f"Model path: {MODEL_PATH}")

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer(path):
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
            print(f"  tokenizer attempt {name} failed: {e}")
    try:
        from transformers import PreTrainedTokenizerFast

        tj = os.path.join(path, "tokenizer.json")
        if os.path.isfile(tj):
            tok = PreTrainedTokenizerFast(tokenizer_file=tj)
            tok.add_special_tokens(
                {
                    "bos_token": "<|begin_of_text|>",
                    "eos_token": "<|end_of_text|>",
                    "pad_token": "<|end_of_text|>",
                }
            )
            return tok
    except Exception as e:
        print(f"  PreTrainedTokenizerFast failed: {e}")
    raise RuntimeError("Tokenizer loading failed")


tokenizer = load_tokenizer(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return f"<|user|>\n{text}\n<|assistant|>\n"


LAYERS_TO_PROBE = [8, 12, 16, 20, 24, 28]


@torch.no_grad()
def get_activations(prompts):
    acts = {L: [] for L in LAYERS_TO_PROBE}
    for i, p in enumerate(prompts):
        text = format_prompt(p)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
            device
        )
        out = model(**enc, output_hidden_states=True, use_cache=False)
        last_idx = enc["attention_mask"].sum(dim=1).item() - 1
        for L in LAYERS_TO_PROBE:
            h = out.hidden_states[L][0, last_idx, :].float().cpu().numpy()
            acts[L].append(h)
        if (i + 1) % 100 == 0:
            print(f"  extracted {i+1}/{len(prompts)}")
    return {L: np.stack(v) for L, v in acts.items()}


print(f"Extracting activations for {N_MAX} harmful prompts...")
harm_acts_full = get_activations(harmful_prompts_pool)
print(f"Extracting activations for {N_MAX} benign prompts...")
ben_acts_full = get_activations(benign_prompts_pool)

del model
torch.cuda.empty_cache()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

experiment_data = {
    "N_PER_CLASS_tuning": {
        "llama3_advbench_alpaca": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "per_config": {},
            "N_candidates": N_CANDIDATES,
        }
    }
}

overall_best = {"val_acc": -1, "N": None, "layer": None}
all_results = {}

for N in N_CANDIDATES:
    print(f"\n===== N_PER_CLASS = {N} =====")
    results = {}
    per_layer_data = {}
    for L in LAYERS_TO_PROBE:
        X_h = harm_acts_full[L][:N]
        X_b = ben_acts_full[L][:N]
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

        train_loss = 1.0 - train_acc
        val_loss = 1.0 - val_acc

        print(
            f"  N={N} Layer {L}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} random={rand_acc:.4f} shuffled={shuf_acc:.4f}"
        )

        ed = experiment_data["N_PER_CLASS_tuning"]["llama3_advbench_alpaca"]
        ed["metrics"]["train"].append({"N": N, "layer": L, "acc": float(train_acc)})
        ed["metrics"]["val"].append(
            {
                "N": N,
                "layer": L,
                "acc": float(val_acc),
                "random": rand_acc,
                "shuffled": float(shuf_acc),
            }
        )
        ed["losses"]["train"].append({"N": N, "layer": L, "loss": float(train_loss)})
        ed["losses"]["val"].append({"N": N, "layer": L, "loss": float(val_loss)})

        per_layer_data[L] = {
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "random_acc": float(rand_acc),
            "shuffled_acc": float(shuf_acc),
            "direction_norm": float(np.linalg.norm(mu_h - mu_b)),
        }
        results[L] = (val_acc, rand_acc)

    best_L = max(results.keys(), key=lambda k: results[k][0])
    best_val = results[best_L][0]
    print(f"  [N={N}] BEST layer={best_L} val_acc={best_val:.4f}")

    experiment_data["N_PER_CLASS_tuning"]["llama3_advbench_alpaca"]["per_config"][N] = {
        "per_layer": per_layer_data,
        "best_layer": int(best_L),
        "best_val_acc": float(best_val),
    }
    all_results[N] = (best_L, best_val)

    if best_val > overall_best["val_acc"]:
        overall_best = {"val_acc": float(best_val), "N": int(N), "layer": int(best_L)}

print(
    f"\nOVERALL BEST: N={overall_best['N']} layer={overall_best['layer']} val_acc={overall_best['val_acc']:.4f}"
)
ed = experiment_data["N_PER_CLASS_tuning"]["llama3_advbench_alpaca"]
ed["best_N"] = overall_best["N"]
ed["best_layer"] = overall_best["layer"]
ed["harmfulness_direction_probe_accuracy"] = overall_best["val_acc"]

# Rerun best config to save predictions/ground_truth
bN, bL = overall_best["N"], overall_best["layer"]
X_h = harm_acts_full[bL][:bN]
X_b = ben_acts_full[bL][:bN]
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
ed["predictions"] = preds.tolist()
ed["ground_truth"] = y_te.tolist()

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    for N in N_CANDIDATES:
        layers = LAYERS_TO_PROBE
        vals = [
            experiment_data["N_PER_CLASS_tuning"]["llama3_advbench_alpaca"][
                "per_config"
            ][N]["per_layer"][L]["val_acc"]
            for L in layers
        ]
        plt.plot(layers, vals, "o-", label=f"N={N}")
    plt.xlabel("Layer")
    plt.ylabel("Val Acc")
    plt.title("Probe Val Acc per Layer vs N_PER_CLASS")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "N_per_class_tuning_val_acc.png"), bbox_inches="tight"
    )
    plt.close()
    print("Saved plot")
except Exception as e:
    print("plot failed:", e)

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
print(f"FINAL harmfulness_direction_probe_accuracy = {overall_best['val_acc']:.4f}")
