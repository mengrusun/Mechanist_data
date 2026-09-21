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
HIGH_RES = ["en", "es", "fr", "de", "zh", "ja", "ru"]
MID_RES = ["th", "te"]
LOW_RES = ["bn", "sw"]

N_PROBE = 10  # examples per language for subspace probe
N_EVAL = 20  # examples per language for evaluation
MAX_NEW_TOKENS = 1024


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


def load_mgsm_lang(lang, max_n=30):
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


mgsm_data_all = {}
for lang in LANGS:
    ex = load_mgsm_lang(lang, max_n=N_PROBE + N_EVAL)
    if ex:
        mgsm_data_all[lang] = ex
        print(f"  {lang}: {len(ex)} examples loaded")

if len(mgsm_data_all) < len(LANGS):
    try:
        from datasets import load_dataset

        for lang in LANGS:
            if lang in mgsm_data_all:
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
                    if i >= N_PROBE + N_EVAL:
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
                    mgsm_data_all[lang] = exs
                    print(f"  HF {lang}: {len(exs)}")
            except Exception as e:
                print(f"  HF {lang} fail: {e}")
    except Exception as e:
        print(f"HF fallback fail: {e}")

if not mgsm_data_all:
    raise RuntimeError("Failed to load ANY MGSM data.")

# Split into probe (subspace) and eval sets
probe_data = {}
eval_data = {}
for lang, exs in mgsm_data_all.items():
    if len(exs) >= N_PROBE + 1:
        probe_data[lang] = exs[: min(N_PROBE, len(exs) // 2)]
        eval_data[lang] = exs[len(probe_data[lang]) : len(probe_data[lang]) + N_EVAL]
    else:
        probe_data[lang] = exs[: max(1, len(exs) // 2)]
        eval_data[lang] = exs[len(probe_data[lang]) :]
    print(f"  {lang}: probe={len(probe_data[lang])}, eval={len(eval_data[lang])}")

print(f"Loaded MGSM for languages: {list(mgsm_data_all.keys())}")

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
# Layer positions to explore: early / mid / late
LAYER_POSITIONS = {
    "early": max(1, num_layers // 4),
    "mid": num_layers // 2,
    "late": (3 * num_layers) // 4,
}
print(f"num_layers={num_layers}, layer positions: {LAYER_POSITIONS}")


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


# Collect hidden states at all target layers using one forward pass
lang_means_by_layer = {name: {} for name in LAYER_POSITIONS}
print("Collecting hidden states per language at multiple layers...")

hooks = []
for name, lidx in LAYER_POSITIONS.items():
    hh = layers[lidx].register_forward_hook(make_capture_hook(name))
    hooks.append(hh)

try:
    for lang, exs in probe_data.items():
        vecs_by_layer = {name: [] for name in LAYER_POSITIONS}
        for ex in exs:
            prompt = build_prompt(ex["question"])
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(
                model.device
            )
            with torch.no_grad():
                _ = model(**enc, use_cache=False)
            for name in LAYER_POSITIONS:
                vecs_by_layer[name].append(captured[name][0])
        for name in LAYER_POSITIONS:
            lang_means_by_layer[name][lang] = np.mean(
                np.stack(vecs_by_layer[name], 0), 0
            )
        print(f"  {lang}: computed means at all layers")
finally:
    for hh in hooks:
        hh.remove()

# Compute subspaces at each layer
K = 3
subspaces_by_layer = {}  # {layer_name: lang_subspace}
rand_subspaces_by_layer = {}
for name, lang_means in lang_means_by_layer.items():
    M = np.stack([lang_means[l] for l in lang_means.keys()], axis=0)
    global_mean = M.mean(axis=0, keepdims=True)
    X = M - global_mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Kact = min(K, Vt.shape[0])
    lang_sub = Vt[:Kact].astype(np.float32)
    subspaces_by_layer[name] = lang_sub
    rng = np.random.RandomState(hash(name) % (2**31))
    R = rng.randn(Kact, lang_sub.shape[1]).astype(np.float32)
    Rq, _ = np.linalg.qr(R.T)
    rand_subspaces_by_layer[name] = Rq.T[:Kact].astype(np.float32)
    print(f"  {name} (layer {LAYER_POSITIONS[name]}): top singular values = {S[:Kact]}")

np.save(
    os.path.join(working_dir, "subspaces_by_layer.npy"),
    subspaces_by_layer,
    allow_pickle=True,
)
np.save(
    os.path.join(working_dir, "lang_means_by_layer.npy"),
    lang_means_by_layer,
    allow_pickle=True,
)

# Intervention state: which layer, which subspace, what alpha
intervention = {"V": None, "alpha": 0.0}


def intervention_hook(module, inputs, outputs):
    V = intervention["V"]
    alpha = intervention["alpha"]
    if V is None or alpha == 0.0:
        return outputs
    h = outputs[0] if isinstance(outputs, tuple) else outputs
    orig_dtype = h.dtype
    Vt_t = torch.from_numpy(V).to(h.device).to(h.dtype)
    proj = torch.einsum("bth,kh->btk", h, Vt_t)
    # h_new = h - alpha * projection (alpha=1 removes; alpha=-1 amplifies; alpha=0 identity)
    h_new = h - alpha * torch.einsum("btk,kh->bth", proj, Vt_t)
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


# Simple script-based language fidelity: whether output uses the expected script
SCRIPT_CHECKS = {
    "en": lambda s: len(re.findall(r"[A-Za-z]", s)),
    "es": lambda s: len(re.findall(r"[A-Za-zñáéíóúÁÉÍÓÚÑ]", s)),
    "fr": lambda s: len(re.findall(r"[A-Za-zàâçéèêëîïôûùüÿœ]", s)),
    "de": lambda s: len(re.findall(r"[A-Za-zäöüßÄÖÜ]", s)),
    "zh": lambda s: len(re.findall(r"[\u4e00-\u9fff]", s)),
    "ja": lambda s: len(re.findall(r"[\u3040-\u30ff\u4e00-\u9fff]", s)),
    "ru": lambda s: len(re.findall(r"[\u0400-\u04FF]", s)),
    "th": lambda s: len(re.findall(r"[\u0e00-\u0e7f]", s)),
    "te": lambda s: len(re.findall(r"[\u0c00-\u0c7f]", s)),
    "bn": lambda s: len(re.findall(r"[\u0980-\u09ff]", s)),
    "sw": lambda s: len(re.findall(r"[A-Za-z]", s)),
}


def language_fidelity(text, lang):
    fn = SCRIPT_CHECKS.get(lang)
    if fn is None:
        return 0.0
    total = len(re.sub(r"\s+", "", text))
    if total == 0:
        return 0.0
    return fn(text) / total


def evaluate(
    condition_name, layer_name, subspace_matrix, alpha, max_new_tokens=MAX_NEW_TOKENS
):
    intervention["V"] = subspace_matrix
    intervention["alpha"] = alpha
    hook_handle = None
    if subspace_matrix is not None and alpha != 0.0:
        lidx = LAYER_POSITIONS[layer_name]
        hook_handle = layers[lidx].register_forward_hook(intervention_hook)
    per_lang_acc = {}
    per_lang_fid = {}
    per_lang_preds = {}
    all_correct = []
    all_fid = []
    try:
        for lang, exs in eval_data.items():
            correct = 0
            fids = []
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
                fid = language_fidelity(gen, lang)
                fids.append(fid)
                all_fid.append(fid)
                preds_list.append(
                    {
                        "pred": pred,
                        "gt": gt,
                        "correct": int(ok),
                        "fidelity": fid,
                        "gen_len": len(gen),
                    }
                )
            per_lang_acc[lang] = correct / max(1, len(exs))
            per_lang_fid[lang] = float(np.mean(fids)) if fids else 0.0
            per_lang_preds[lang] = preds_list
            print(
                f"  [{condition_name} L={layer_name} α={alpha:+.2f}] {lang}: acc={per_lang_acc[lang]:.3f} fid={per_lang_fid[lang]:.3f}"
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        intervention["V"] = None
        intervention["alpha"] = 0.0
    overall_acc = float(np.mean(all_correct)) if all_correct else 0.0
    overall_fid = float(np.mean(all_fid)) if all_fid else 0.0
    return overall_acc, overall_fid, per_lang_acc, per_lang_fid, per_lang_preds


# =========== Experimental plan ===========
# 1) Baseline (α=0)
# 2) Dose-response at mid-layer: alpha in [-1.0, -0.5, 0.5, 1.0, 1.5] with lang subspace
# 3) Random control at alpha=1.0 for each layer position
# 4) Layer sweep at alpha=1.0 (lang subspace) at early/mid/late

ALPHA_GRID = [-1.0, -0.5, 0.5, 1.0, 1.5]

experiment_data = {
    "creative_dose_response_layer_sweep": {
        "mgsm": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": {
                lang: [ex["answer_number"] for ex in exs]
                for lang, exs in eval_data.items()
            },
            "config": {
                "model_path": model_path,
                "num_layers": int(num_layers),
                "layer_positions": {k: int(v) for k, v in LAYER_POSITIONS.items()},
                "K": int(K),
                "max_new_tokens": MAX_NEW_TOKENS,
                "n_probe": {l: len(v) for l, v in probe_data.items()},
                "n_eval": {l: len(v) for l, v in eval_data.items()},
                "alpha_grid": ALPHA_GRID,
                "languages": list(eval_data.keys()),
                "high_res": HIGH_RES,
                "mid_res": MID_RES,
                "low_res": LOW_RES,
            },
            "results": {},
        }
    }
}

results = experiment_data["creative_dose_response_layer_sweep"]["mgsm"]["results"]
epoch = 0


def record_run(tag, layer_name, alpha, cond_type):
    global epoch
    print(f"\n=== {tag} ===")
    t0 = time.time()
    if cond_type == "baseline":
        sub = None
    elif cond_type == "lang":
        sub = subspaces_by_layer[layer_name]
    elif cond_type == "random":
        sub = rand_subspaces_by_layer[layer_name]
    else:
        raise ValueError(cond_type)
    acc, fid, per_lang_acc, per_lang_fid, per_lang_preds = evaluate(
        tag, layer_name, sub, alpha
    )
    print(
        f"[{tag}] overall_acc={acc:.4f} overall_fid={fid:.4f} (time={time.time()-t0:.1f}s)"
    )

    # Resource-tier averages
    def tier_avg(langs_):
        vals = [per_lang_acc[l] for l in langs_ if l in per_lang_acc]
        return float(np.mean(vals)) if vals else 0.0

    high = tier_avg(HIGH_RES)
    mid = tier_avg(MID_RES)
    low = tier_avg(LOW_RES)
    print(f"  tier acc: high={high:.3f} mid={mid:.3f} low={low:.3f}")

    results[tag] = {
        "cond_type": cond_type,
        "layer_name": layer_name,
        "alpha": alpha,
        "overall_acc": acc,
        "overall_fid": fid,
        "per_lang_acc": per_lang_acc,
        "per_lang_fid": per_lang_fid,
        "predictions": per_lang_preds,
        "tier_acc": {"high": high, "mid": mid, "low": low},
    }
    experiment_data["creative_dose_response_layer_sweep"]["mgsm"]["metrics"][
        "val"
    ].append(
        {
            "epoch": epoch,
            "tag": tag,
            "cond_type": cond_type,
            "layer": layer_name,
            "alpha": alpha,
            "multilingual_reasoning_accuracy": acc,
            "language_fidelity": fid,
            "high_acc": high,
            "mid_acc": mid,
            "low_acc": low,
        }
    )
    experiment_data["creative_dose_response_layer_sweep"]["mgsm"]["losses"][
        "val"
    ].append(
        {
            "epoch": epoch,
            "tag": tag,
            "loss": 1.0 - acc,
        }
    )
    print(f"Epoch {epoch}: validation_loss = {1.0 - acc:.4f}")
    epoch += 1


# 1) Baseline
record_run("baseline_alpha0", "mid", 0.0, "baseline")

# 2) Dose-response at mid layer
for a in ALPHA_GRID:
    record_run(f"lang_mid_alpha{a:+.2f}", "mid", a, "lang")

# 3) Random control at alpha=1.0 at each layer
for lname in LAYER_POSITIONS:
    record_run(f"random_{lname}_alpha+1.00", lname, 1.0, "random")

# 4) Layer sweep at alpha=1.0 with lang subspace (mid already done above; do early & late)
for lname in ["early", "late"]:
    record_run(f"lang_{lname}_alpha+1.00", lname, 1.0, "lang")

# Save results
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# ================= Plots =================
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: dose-response curve at mid layer
    xs = [0.0] + ALPHA_GRID
    accs = [results["baseline_alpha0"]["overall_acc"]] + [
        results[f"lang_mid_alpha{a:+.2f}"]["overall_acc"] for a in ALPHA_GRID
    ]
    fids = [results["baseline_alpha0"]["overall_fid"]] + [
        results[f"lang_mid_alpha{a:+.2f}"]["overall_fid"] for a in ALPHA_GRID
    ]
    order = np.argsort(xs)
    xs_s = np.array(xs)[order]
    accs_s = np.array(accs)[order]
    fids_s = np.array(fids)[order]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(xs_s, accs_s, "o-", color="tab:blue", label="Reasoning accuracy")
    ax1.set_xlabel("Suppression strength α (0=baseline, 1=full remove, <0=amplify)")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(xs_s, fids_s, "s--", color="tab:red", label="Language fidelity")
    ax2.set_ylabel("Language fidelity", color="tab:red")
    plt.title("MGSM dose-response: mid-layer language-subspace intervention")
    fig.tight_layout()
    plt.savefig(os.path.join(working_dir, "mgsm_dose_response_mid.png"), dpi=120)
    plt.close()

    # Plot 2: layer sweep (lang vs random) at alpha=+1.0
    layer_names = ["early", "mid", "late"]
    lang_accs = []
    rand_accs = []
    for ln in layer_names:
        tag_lang = f"lang_mid_alpha+1.00" if ln == "mid" else f"lang_{ln}_alpha+1.00"
        lang_accs.append(results[tag_lang]["overall_acc"])
        rand_accs.append(results[f"random_{ln}_alpha+1.00"]["overall_acc"])
    base = results["baseline_alpha0"]["overall_acc"]
    x = np.arange(len(layer_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, lang_accs, w, label="Lang subspace removal (α=1)")
    ax.bar(x + w / 2, rand_accs, w, label="Random subspace control (α=1)")
    ax.axhline(base, color="black", linestyle="--", label=f"Baseline={base:.3f}")
    for xi, v in zip(x - w / 2, lang_accs):
        ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, rand_accs):
        ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylabel("Accuracy")
    ax.set_title("MGSM: layer position sweep of subspace removal")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mgsm_layer_sweep.png"), dpi=120)
    plt.close()

    # Plot 3: per-tier accuracy comparison (baseline vs best intervention)
    tiers = ["high", "mid", "low"]
    base_tier = [results["baseline_alpha0"]["tier_acc"][t] for t in tiers]
    supp_tier = [results["lang_mid_alpha+1.00"]["tier_acc"][t] for t in tiers]
    rand_tier = [results["random_mid_alpha+1.00"]["tier_acc"][t] for t in tiers]
    x = np.arange(len(tiers))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, base_tier, w, label="Baseline")
    ax.bar(x, supp_tier, w, label="Lang suppression α=1 (mid)")
    ax.bar(x + w, rand_tier, w, label="Random control α=1 (mid)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}-res" for t in tiers])
    ax.set_ylabel("Accuracy")
    ax.set_title("MGSM per-tier accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mgsm_tier_comparison.png"), dpi=120)
    plt.close()

    # Plot 4: per-language heatmap of accuracy across conditions
    conds_hm = [
        "baseline_alpha0",
        "lang_early_alpha+1.00",
        "lang_mid_alpha+1.00",
        "lang_late_alpha+1.00",
        "random_mid_alpha+1.00",
    ]
    langs_hm = list(eval_data.keys())
    mat = np.zeros((len(conds_hm), len(langs_hm)))
    for i, c in enumerate(conds_hm):
        for j, l in enumerate(langs_hm):
            mat[i, j] = results[c]["per_lang_acc"].get(l, 0.0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(langs_hm)))
    ax.set_xticklabels(langs_hm)
    ax.set_yticks(range(len(conds_hm)))
    ax.set_yticklabels(conds_hm, fontsize=8)
    for i in range(len(conds_hm)):
        for j in range(len(langs_hm)):
            ax.text(
                j,
                i,
                f"{mat[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )
    plt.colorbar(im, ax=ax, label="Accuracy")
    ax.set_title("MGSM per-language accuracy: conditions vs languages")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mgsm_lang_condition_heatmap.png"), dpi=120)
    plt.close()

except Exception as e:
    print(f"plot fail: {e}")

# ================= Summary =================
print("\n=== FINAL SUMMARY ===")
print(f"Baseline overall accuracy: {results['baseline_alpha0']['overall_acc']:.4f}")
print("\nDose-response curve at mid layer (lang subspace):")
for a in sorted([0.0] + ALPHA_GRID):
    if a == 0.0:
        r = results["baseline_alpha0"]
    else:
        r = results[f"lang_mid_alpha{a:+.2f}"]
    print(
        f"  α={a:+.2f}: acc={r['overall_acc']:.4f} fid={r['overall_fid']:.4f} high={r['tier_acc']['high']:.3f} low={r['tier_acc']['low']:.3f}"
    )

print("\nLayer sweep at α=+1.0:")
for ln in ["early", "mid", "late"]:
    tag_lang = f"lang_{ln}_alpha+1.00" if ln != "mid" else "lang_mid_alpha+1.00"
    tag_rand = f"random_{ln}_alpha+1.00"
    la = results[tag_lang]
    ra = results[tag_rand]
    print(
        f"  {ln}: lang acc={la['overall_acc']:.4f} fid={la['overall_fid']:.4f} | rand acc={ra['overall_acc']:.4f} fid={ra['overall_fid']:.4f}"
    )
    print(
        f"    lang tier: high={la['tier_acc']['high']:.3f} mid={la['tier_acc']['mid']:.3f} low={la['tier_acc']['low']:.3f}"
    )

# Key metric to report
best_intervention_acc = max(
    results[f"lang_mid_alpha{a:+.2f}"]["overall_acc"] for a in ALPHA_GRID if a > 0
)
print(
    f"\nmultilingual_reasoning_accuracy (baseline) = {results['baseline_alpha0']['overall_acc']:.4f}"
)
print(
    f"multilingual_reasoning_accuracy (best positive-α suppression) = {best_intervention_acc:.4f}"
)

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Done.")
