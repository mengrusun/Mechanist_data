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


# ---------------- Preflight: locate model ----------------
def find_model():
    if not os.path.isdir(MODEL_DIR):
        return None
    # Walk to find any dir with config.json matching Qwen3-4B-Thinking pattern
    matches = []
    for root, dirs, files in os.walk(MODEL_DIR):
        if "config.json" not in files:
            continue
        low = root.lower()
        # exclude snapshots inner refs artifacts but allow snapshot dirs
        base = os.path.basename(root).lower()
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
        # list what's available for debugging
        print("Available in MODEL_DIR:")
        for name in os.listdir(MODEL_DIR):
            print(f"  {name}")
        return None
    matches.sort(key=lambda x: (-x[0], len(x[1])))
    return matches[0][1]


model_path = find_model()
if model_path is None:
    raise RuntimeError(
        f"Qwen3-4B-Thinking model not found in {MODEL_DIR}. Please place it locally."
    )
print(f"Model path: {model_path}")


# ---------------- Preflight: locate MGSM ----------------
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
                # MGSM tsv format: question <tab> answer
                if df.shape[1] >= 2:
                    qcol, acol = 0, 1
                    # if header present
                    if isinstance(df.iloc[0, 0], str) and df.iloc[0, 0].lower() in (
                        "question",
                        "problem",
                        "input",
                    ):
                        df = df.iloc[1:]
                    for _, row in df.iterrows():
                        add_ex(row.iloc[qcol], row.iloc[acol])
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

# Try HF cache datasets if available locally
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
    raise RuntimeError(
        f"Failed to load ANY MGSM data. Searched dirs: {mgsm_dirs}. Please place MGSM under $DATA_DIR/mgsm"
    )

print(f"Loaded MGSM for languages: {list(mgsm_data.keys())}")

# ---------------- Load model ----------------
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
    all_correct = []
    try:
        for lang, exs in mgsm_data.items():
            correct = 0
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
            per_lang_acc[lang] = correct / max(1, len(exs))
            print(f"  [{condition_name}] {lang}: acc={per_lang_acc[lang]:.3f}")
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        current_subspace["V"] = None
    overall = float(np.mean(all_correct)) if all_correct else 0.0
    return overall, per_lang_acc


experiment_data = {
    "mgsm": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "conditions": {},
    }
}

print("\n=== Baseline ===")
t0 = time.time()
base_acc, base_per_lang = evaluate("baseline", None)
print(
    f"Baseline overall multilingual_reasoning_accuracy = {base_acc:.4f}  (time={time.time()-t0:.1f}s)"
)

print("\n=== Language subspace suppression ===")
t0 = time.time()
supp_acc, supp_per_lang = evaluate("lang_suppression", lang_subspace)
print(
    f"Suppression overall multilingual_reasoning_accuracy = {supp_acc:.4f}  (time={time.time()-t0:.1f}s)"
)

print("\n=== Random subspace control ===")
t0 = time.time()
rand_acc, rand_per_lang = evaluate("random_control", rand_subspace)
print(
    f"Random overall multilingual_reasoning_accuracy = {rand_acc:.4f}  (time={time.time()-t0:.1f}s)"
)

experiment_data["mgsm"]["conditions"]["baseline"] = {
    "overall": base_acc,
    "per_lang": base_per_lang,
}
experiment_data["mgsm"]["conditions"]["lang_suppression"] = {
    "overall": supp_acc,
    "per_lang": supp_per_lang,
}
experiment_data["mgsm"]["conditions"]["random_control"] = {
    "overall": rand_acc,
    "per_lang": rand_per_lang,
}
experiment_data["mgsm"]["metrics"]["val"] = [
    {"epoch": 0, "condition": "baseline", "multilingual_reasoning_accuracy": base_acc},
    {
        "epoch": 1,
        "condition": "lang_suppression",
        "multilingual_reasoning_accuracy": supp_acc,
    },
    {
        "epoch": 2,
        "condition": "random_control",
        "multilingual_reasoning_accuracy": rand_acc,
    },
]
experiment_data["mgsm"]["config"] = {
    "model_path": model_path,
    "target_layer_idx": int(target_layer_idx),
    "num_layers": int(num_layers),
    "K": int(K),
    "languages": list(mgsm_data.keys()),
    "examples_per_lang": {l: len(v) for l, v in mgsm_data.items()},
}

print(f"\nEpoch 0: validation_loss = {1.0-base_acc:.4f}  (baseline)")
print(f"Epoch 1: validation_loss = {1.0-supp_acc:.4f}  (suppression)")
print(f"Epoch 2: validation_loss = {1.0-rand_acc:.4f}  (random control)")

print(f"\n=== FINAL ===")
print(f"multilingual_reasoning_accuracy (baseline):        {base_acc:.4f}")
print(f"multilingual_reasoning_accuracy (lang_suppression): {supp_acc:.4f}")
print(f"multilingual_reasoning_accuracy (random_control):   {rand_acc:.4f}")

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = ["baseline", "lang_suppression", "random_control"]
    vals = [base_acc, supp_acc, rand_acc]
    plt.figure(figsize=(6, 4))
    plt.bar(conds, vals, color=["gray", "tab:blue", "tab:orange"])
    plt.ylabel("multilingual_reasoning_accuracy")
    plt.title("MGSM (small sample) — Qwen3-4B-Thinking")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylim(0, max(vals) * 1.3 + 0.05)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mgsm_conditions_bar.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"plot fail: {e}")

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Done.")
