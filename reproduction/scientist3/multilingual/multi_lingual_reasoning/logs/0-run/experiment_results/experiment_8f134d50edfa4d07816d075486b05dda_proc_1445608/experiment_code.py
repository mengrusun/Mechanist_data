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

import os, re, glob, json, gc, time, math
import numpy as np
import torch

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"

LANGS = ["en", "es", "fr", "de", "zh", "ja", "ru", "th", "te", "bn", "sw"]


def find_model():
    if not os.path.isdir(MODEL_DIR):
        return None
    matches = []
    for root, dirs, files in os.walk(MODEL_DIR):
        if "config.json" not in files:
            continue
        parent_path = root.lower()
        score = 0
        if "qwen3" in parent_path and "4b" in parent_path and "thinking" in parent_path:
            score = 3
        elif "qwen3-4b" in parent_path and "thinking" in parent_path:
            score = 3
        elif (
            "qwen" in parent_path and "4b" in parent_path and "thinking" in parent_path
        ):
            score = 2
        if score > 0:
            matches.append((score, root))
    if not matches:
        print("Available in MODEL_DIR:")
        for name in os.listdir(MODEL_DIR):
            print(f"  {name}")
        return None
    matches.sort(key=lambda x: (-x[0], len(x[1])))
    return matches[0][1]


model_path = find_model()
if model_path is None:
    raise RuntimeError(f"Qwen3-4B-Thinking model not found in {MODEL_DIR}.")
print(f"Model path: {model_path}")


def find_mgsm_dir():
    cands = []
    if os.path.isdir(DATA_DIR):
        for root, dirs, files in os.walk(DATA_DIR):
            low = os.path.basename(root).lower()
            if "mgsm" in low:
                cands.append(root)
    return cands


mgsm_dirs = find_mgsm_dir()
print(f"MGSM candidate dirs: {mgsm_dirs}")


def load_mgsm_lang(lang, max_n=5):
    exts_text = (".jsonl", ".json", ".tsv", ".csv")
    files = []
    for d in mgsm_dirs:
        for root, _, fs in os.walk(d):
            for f in fs:
                fl = f.lower()
                stem = fl.rsplit(".", 1)[0]
                parts = re.split(r"[_\-.]", stem)
                if (
                    lang in parts
                    or stem == lang
                    or stem == f"mgsm_{lang}"
                    or stem == f"test_{lang}"
                ):
                    files.append(os.path.join(root, f))
    text_files = [f for f in files if f.lower().endswith(exts_text)]
    parquet_files = [f for f in files if f.lower().endswith(".parquet")]
    examples = []

    def add_ex(q, a):
        if q is None or a is None:
            return
        if isinstance(a, (int, float)) and not (isinstance(a, float) and math.isnan(a)):
            num = float(a)
        else:
            s = str(a)
            m = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
            if not m:
                return
            try:
                num = float(m[-1])
            except:
                return
        examples.append({"question": str(q).strip(), "answer_number": num})

    for fp in text_files:
        try:
            if fp.endswith(".jsonl"):
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except:
                            continue
                        q = (
                            obj.get("question")
                            or obj.get("problem")
                            or obj.get("input")
                        )
                        a = (
                            obj.get("answer_number")
                            or obj.get("answer")
                            or obj.get("target")
                            or obj.get("output")
                        )
                        add_ex(q, a)
            elif fp.endswith(".json"):
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        q = obj.get("question") or obj.get("problem")
                        a = obj.get("answer_number") or obj.get("answer")
                        add_ex(q, a)
            elif fp.endswith(".tsv") or fp.endswith(".csv"):
                import pandas as pd

                sep = "\t" if fp.endswith(".tsv") else ","
                df = pd.read_csv(fp, sep=sep, header=None)
                if df.shape[1] >= 2:
                    if isinstance(df.iloc[0, 0], str) and df.iloc[0, 0].lower() in (
                        "question",
                        "problem",
                        "input",
                    ):
                        df = df.iloc[1:]
                    for _, row in df.iterrows():
                        add_ex(row.iloc[0], row.iloc[1])
            if examples:
                break
        except Exception as e:
            print(f"  fail read {fp}: {e}")
    if not examples:
        for fp in parquet_files:
            try:
                import pandas as pd

                df = pd.read_parquet(fp)
                qcol = None
                acol = None
                for c in df.columns:
                    if c.lower() in ("question", "problem", "input"):
                        qcol = c
                    if c.lower() in ("answer_number", "answer", "target", "output"):
                        acol = c
                if qcol is None:
                    qcol = df.columns[0]
                if acol is None:
                    acol = df.columns[-1]
                for _, row in df.iterrows():
                    add_ex(row[qcol], row[acol])
                if examples:
                    break
            except Exception as e:
                print(f"  fail parquet {fp}: {e}")
    return examples[:max_n]


mgsm_data = {}
for lang in LANGS:
    ex = load_mgsm_lang(lang, max_n=5)
    if ex:
        mgsm_data[lang] = ex
        print(f"  {lang}: {len(ex)} examples loaded")

if len(mgsm_data) < len(LANGS):
    try:
        from datasets import load_dataset

        for lang in LANGS:
            if lang in mgsm_data:
                continue
            try:
                ds = load_dataset(
                    "juletxara/mgsm",
                    lang,
                    split="test",
                    cache_dir=os.path.join(DATA_DIR, "mgsm_hf_cache"),
                )
                exs = []
                for i, row in enumerate(ds):
                    if i >= 5:
                        break
                    q = row.get("question")
                    a = row.get("answer_number", row.get("answer"))
                    if isinstance(a, str):
                        m = re.findall(r"-?\d+\.?\d*", a.replace(",", ""))
                        if m:
                            a = float(m[-1])
                        else:
                            continue
                    exs.append({"question": q, "answer_number": float(a)})
                if exs:
                    mgsm_data[lang] = exs
                    print(f"  HF {lang}: {len(exs)}")
            except Exception as e:
                print(f"  HF {lang} fail: {e}")
    except Exception as e:
        print(f"HF fallback fail: {e}")

if not mgsm_data:
    raise RuntimeError(f"Failed to load ANY MGSM data.")

print(f"Loaded MGSM for languages: {list(mgsm_data.keys())}")

from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
)
model.eval()
print("Model loaded.")

if hasattr(model, "model") and hasattr(model.model, "layers"):
    layers = model.model.layers
elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
    layers = model.transformer.h
else:
    raise RuntimeError("Cannot find layer list on model")

num_layers = len(layers)
target_layer_idx = num_layers // 2
print(f"num_layers={num_layers}, target_layer_idx={target_layer_idx}")


def build_prompt(question):
    msgs = [
        {
            "role": "user",
            "content": question
            + "\n\nPlease reason briefly and end with 'Answer: <number>'.",
        }
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


captured = {}


def make_capture_hook(store_key):
    def hook(module, inputs, outputs):
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        captured[store_key] = h[:, -1, :].detach().float().cpu().numpy()

    return hook


lang_means = {}
print("Collecting hidden states per language...")
hh = layers[target_layer_idx].register_forward_hook(make_capture_hook("h"))
try:
    for lang, exs in mgsm_data.items():
        vecs = []
        for ex in exs:
            prompt = build_prompt(ex["question"])
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(
                model.device
            )
            with torch.no_grad():
                _ = model(**enc, use_cache=False)
            vecs.append(captured["h"][0])
        lang_means[lang] = np.mean(np.stack(vecs, 0), 0)
        print(
            f"  {lang}: mean vec computed, norm={np.linalg.norm(lang_means[lang]):.3f}"
        )
finally:
    hh.remove()

M = np.stack([lang_means[l] for l in mgsm_data.keys()], axis=0)
global_mean = M.mean(axis=0, keepdims=True)
X = M - global_mean
U, S, Vt = np.linalg.svd(X, full_matrices=False)
K = min(3, Vt.shape[0])
lang_subspace = Vt[:K].astype(np.float32)
print(f"Extracted language subspace, K={K}, singular values top={S[:K]}")

rng = np.random.RandomState(0)
R = rng.randn(K, lang_subspace.shape[1]).astype(np.float32)
Rq, _ = np.linalg.qr(R.T)
rand_subspace = Rq.T[:K].astype(np.float32)

np.save(os.path.join(working_dir, "lang_subspace.npy"), lang_subspace)
np.save(os.path.join(working_dir, "rand_subspace.npy"), rand_subspace)
np.save(os.path.join(working_dir, "lang_means.npy"), lang_means, allow_pickle=True)

current_subspace = {"V": None}


def suppression_hook(module, inputs, outputs):
    V = current_subspace["V"]
    if V is None:
        return outputs
    h = outputs[0] if isinstance(outputs, tuple) else outputs
    orig_dtype = h.dtype
    Vt_t = torch.from_numpy(V).to(h.device).to(h.dtype)
    proj = torch.einsum("bth,kh->btk", h, Vt_t)
    h_new = h - torch.einsum("btk,kh->bth", proj, Vt_t)
    if isinstance(outputs, tuple):
        return (h_new.to(orig_dtype),) + outputs[1:]
    return h_new.to(orig_dtype)


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text):
    m = re.search(r"[Aa]nswer[^\d\-]*(-?\d+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    text_clean = text.replace(",", "")
    nums = NUM_RE.findall(text_clean)
    if nums:
        try:
            return float(nums[-1])
        except:
            return None
    return None


def evaluate(condition_name, subspace_matrix, max_new_tokens=256):
    current_subspace["V"] = subspace_matrix
    hook_handle = None
    if subspace_matrix is not None:
        hook_handle = layers[target_layer_idx].register_forward_hook(suppression_hook)
    per_lang_acc = {}
    per_lang_preds = {}
    all_correct = []
    try:
        for lang, exs in mgsm_data.items():
            correct = 0
            preds_list = []
            for ex in exs:
                prompt = build_prompt(ex["question"])
                enc = tok(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                ).to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                gen = tok.decode(
                    out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
                )
                pred = extract_answer(gen)
                gt = ex["answer_number"]
                ok = pred is not None and abs(pred - gt) < 1e-3
                correct += int(ok)
                all_correct.append(int(ok))
                preds_list.append({"pred": pred, "gt": gt, "correct": int(ok)})
            per_lang_acc[lang] = correct / max(1, len(exs))
            per_lang_preds[lang] = preds_list
            print(
                f"  [{condition_name} mnt={max_new_tokens}] {lang}: acc={per_lang_acc[lang]:.3f}"
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        current_subspace["V"] = None
    overall = float(np.mean(all_correct)) if all_correct else 0.0
    return overall, per_lang_acc, per_lang_preds


# ---------------- Hyperparameter tuning: max_new_tokens ----------------
MAX_NEW_TOKENS_GRID = [256, 512, 1024]

experiment_data = {
    "max_new_tokens": {
        "mgsm": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "conditions": {},
            "hyperparam_values": MAX_NEW_TOKENS_GRID,
            "results_by_value": {},
            "config": {
                "model_path": model_path,
                "target_layer_idx": int(target_layer_idx),
                "num_layers": int(num_layers),
                "K": int(K),
                "languages": list(mgsm_data.keys()),
                "examples_per_lang": {l: len(v) for l, v in mgsm_data.items()},
            },
        }
    }
}

# Collect ground truth once
gt_all = {lang: [ex["answer_number"] for ex in exs] for lang, exs in mgsm_data.items()}
experiment_data["max_new_tokens"]["mgsm"]["ground_truth"] = gt_all

epoch_counter = 0
for mnt in MAX_NEW_TOKENS_GRID:
    print(f"\n########## max_new_tokens = {mnt} ##########")
    results = {}

    print(f"\n=== Baseline (mnt={mnt}) ===")
    t0 = time.time()
    base_acc, base_per_lang, base_preds = evaluate("baseline", None, max_new_tokens=mnt)
    print(f"Baseline overall = {base_acc:.4f}  (time={time.time()-t0:.1f}s)")

    print(f"\n=== Language subspace suppression (mnt={mnt}) ===")
    t0 = time.time()
    supp_acc, supp_per_lang, supp_preds = evaluate(
        "lang_suppression", lang_subspace, max_new_tokens=mnt
    )
    print(f"Suppression overall = {supp_acc:.4f}  (time={time.time()-t0:.1f}s)")

    print(f"\n=== Random subspace control (mnt={mnt}) ===")
    t0 = time.time()
    rand_acc, rand_per_lang, rand_preds = evaluate(
        "random_control", rand_subspace, max_new_tokens=mnt
    )
    print(f"Random overall = {rand_acc:.4f}  (time={time.time()-t0:.1f}s)")

    results["baseline"] = {
        "overall": base_acc,
        "per_lang": base_per_lang,
        "predictions": base_preds,
    }
    results["lang_suppression"] = {
        "overall": supp_acc,
        "per_lang": supp_per_lang,
        "predictions": supp_preds,
    }
    results["random_control"] = {
        "overall": rand_acc,
        "per_lang": rand_per_lang,
        "predictions": rand_preds,
    }

    experiment_data["max_new_tokens"]["mgsm"]["results_by_value"][str(mnt)] = results

    for cond_name, acc in [
        ("baseline", base_acc),
        ("lang_suppression", supp_acc),
        ("random_control", rand_acc),
    ]:
        experiment_data["max_new_tokens"]["mgsm"]["metrics"]["val"].append(
            {
                "epoch": epoch_counter,
                "max_new_tokens": mnt,
                "condition": cond_name,
                "multilingual_reasoning_accuracy": acc,
            }
        )
        experiment_data["max_new_tokens"]["mgsm"]["losses"]["val"].append(
            {
                "epoch": epoch_counter,
                "max_new_tokens": mnt,
                "condition": cond_name,
                "loss": 1.0 - acc,
            }
        )
        epoch_counter += 1

    experiment_data["max_new_tokens"]["mgsm"]["predictions"].append(
        {
            "max_new_tokens": mnt,
            "baseline": base_preds,
            "lang_suppression": supp_preds,
            "random_control": rand_preds,
        }
    )

# Set final conditions to the best (largest) max_new_tokens run for compatibility
best_mnt = MAX_NEW_TOKENS_GRID[-1]
best_results = experiment_data["max_new_tokens"]["mgsm"]["results_by_value"][
    str(best_mnt)
]
experiment_data["max_new_tokens"]["mgsm"]["conditions"] = {
    "baseline": {
        "overall": best_results["baseline"]["overall"],
        "per_lang": best_results["baseline"]["per_lang"],
    },
    "lang_suppression": {
        "overall": best_results["lang_suppression"]["overall"],
        "per_lang": best_results["lang_suppression"]["per_lang"],
    },
    "random_control": {
        "overall": best_results["random_control"]["overall"],
        "per_lang": best_results["random_control"]["per_lang"],
    },
}

print("\n=== SUMMARY ACROSS max_new_tokens ===")
for mnt in MAX_NEW_TOKENS_GRID:
    r = experiment_data["max_new_tokens"]["mgsm"]["results_by_value"][str(mnt)]
    print(
        f"mnt={mnt}: baseline={r['baseline']['overall']:.4f}, "
        f"lang_suppression={r['lang_suppression']['overall']:.4f}, "
        f"random_control={r['random_control']['overall']:.4f}"
    )

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = ["baseline", "lang_suppression", "random_control"]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(MAX_NEW_TOKENS_GRID))
    for i, c in enumerate(conds):
        vals = [
            experiment_data["max_new_tokens"]["mgsm"]["results_by_value"][str(m)][c][
                "overall"
            ]
            for m in MAX_NEW_TOKENS_GRID
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=c)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in MAX_NEW_TOKENS_GRID])
    ax.set_xlabel("max_new_tokens")
    ax.set_ylabel("multilingual_reasoning_accuracy")
    ax.set_title("MGSM — max_new_tokens hyperparameter sweep")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mgsm_max_new_tokens_sweep.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"plot fail: {e}")

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Done.")
