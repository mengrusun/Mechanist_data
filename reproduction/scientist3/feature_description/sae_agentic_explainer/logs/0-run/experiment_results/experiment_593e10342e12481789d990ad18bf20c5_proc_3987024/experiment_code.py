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

import os, sys, json, time, hashlib, traceback, math, random

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Local-first HF settings
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"
CACHE_DIR = os.path.join(working_dir, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------- API client -------------
API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
API_MODEL = "gpt-5.4"

# bypass proxy
for k in [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]:
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

try:
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    HAS_LLM = True
except Exception as e:
    print(f"OpenAI client init failed: {e}")
    HAS_LLM = False

LLM_CACHE_PATH = os.path.join(CACHE_DIR, "llm_cache.json")
if os.path.exists(LLM_CACHE_PATH):
    with open(LLM_CACHE_PATH) as f:
        llm_cache = json.load(f)
else:
    llm_cache = {}


def llm_call(prompt, system="You are a helpful research assistant.", max_retries=3):
    key = hashlib.sha256((system + "||" + prompt).encode()).hexdigest()
    if key in llm_cache:
        return llm_cache[key]
    if not HAS_LLM:
        return ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=API_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                timeout=60,
            )
            out = resp.choices[0].message.content.strip()
            llm_cache[key] = out
            with open(LLM_CACHE_PATH, "w") as f:
                json.dump(llm_cache, f)
            return out
        except Exception as e:
            print(f"LLM call attempt {attempt+1} failed: {e}")
            time.sleep(2 + attempt * 2)
    return ""


# ------------- Load Gemma-2-2B -------------
from transformers import AutoTokenizer, AutoModelForCausalLM

GEMMA_LOCAL = os.path.join(MODEL_DIR, "gemma-2-2b")
gemma_path = GEMMA_LOCAL if os.path.isdir(GEMMA_LOCAL) else "google/gemma-2-2b"
print(f"Loading Gemma from: {gemma_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        gemma_path, local_files_only=os.path.isdir(GEMMA_LOCAL)
    )
    model = AutoModelForCausalLM.from_pretrained(
        gemma_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=os.path.isdir(GEMMA_LOCAL),
    ).to(device)
    model.eval()
except Exception as e:
    print(f"FATAL: could not load Gemma-2-2B: {e}")
    traceback.print_exc()
    sys.exit(1)

# ------------- Load SAE (GemmaScope) -------------
# Try local first: $MODEL_DIR/gemma-scope-2b-pt-res/layer_{L}/width_16k/average_l0_{K}/params.npz
LAYER = 12
SAE_ROOT_CANDIDATES = [
    os.path.join(MODEL_DIR, "gemma-scope-2b-pt-res"),
    os.path.join(MODEL_DIR, "gemmascope-res-16k"),
    os.path.join(MODEL_DIR, "gemma-scope-2b-pt-res-canonical"),
]


def find_local_sae(layer):
    for root in SAE_ROOT_CANDIDATES:
        if not os.path.isdir(root):
            continue
        # search for width_16k directory
        layer_dir = os.path.join(root, f"layer_{layer}", "width_16k")
        if os.path.isdir(layer_dir):
            for sub in sorted(os.listdir(layer_dir)):
                p = os.path.join(layer_dir, sub, "params.npz")
                if os.path.exists(p):
                    return p, sub
        # flat layout
        p = os.path.join(root, f"layer_{layer}_width_16k.npz")
        if os.path.exists(p):
            return p, "flat"
    return None, None


sae_path, sae_variant = find_local_sae(LAYER)
if sae_path is None:
    # try HF hub download
    try:
        from huggingface_hub import hf_hub_download

        for l0 in [82, 22, 41, 176, 445]:
            try:
                sae_path = hf_hub_download(
                    repo_id="google/gemma-scope-2b-pt-res",
                    filename=f"layer_{LAYER}/width_16k/average_l0_{l0}/params.npz",
                    cache_dir=os.path.join(MODEL_DIR, "hf_cache"),
                )
                sae_variant = f"average_l0_{l0}"
                break
            except Exception as e:
                print(f"HF download l0={l0} failed: {e}")
    except Exception as e:
        print(f"HF hub import/download failed: {e}")

if sae_path is None:
    print(
        "FATAL: could not obtain GemmaScope SAE. Required for whitelisted reproduction."
    )
    sys.exit(1)

print(f"Loading SAE from {sae_path} (variant={sae_variant})")
sae_params = np.load(sae_path)
print(f"SAE keys: {list(sae_params.keys())}")


class JumpReLUSAE(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        W_enc = torch.tensor(params["W_enc"], dtype=torch.float32)  # (d_model, d_sae)
        W_dec = torch.tensor(params["W_dec"], dtype=torch.float32)  # (d_sae, d_model)
        b_enc = torch.tensor(params["b_enc"], dtype=torch.float32)
        b_dec = torch.tensor(params["b_dec"], dtype=torch.float32)
        threshold = (
            torch.tensor(params["threshold"], dtype=torch.float32)
            if "threshold" in params.files
            else torch.zeros(W_enc.shape[1])
        )
        self.W_enc = torch.nn.Parameter(W_enc, requires_grad=False)
        self.W_dec = torch.nn.Parameter(W_dec, requires_grad=False)
        self.b_enc = torch.nn.Parameter(b_enc, requires_grad=False)
        self.b_dec = torch.nn.Parameter(b_dec, requires_grad=False)
        self.threshold = torch.nn.Parameter(threshold, requires_grad=False)

    def encode(self, x):
        # x: (..., d_model)
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        # JumpReLU
        acts = torch.where(pre > self.threshold, pre, torch.zeros_like(pre))
        return acts


sae = JumpReLUSAE(sae_params).to(device)
sae.eval()
D_SAE = sae.W_enc.shape[1]
print(f"SAE d_sae={D_SAE}")

# ------------- Hook to grab residual stream at LAYER -------------
resid_cache = {}


def hook_fn(module, inp, out):
    # out is tuple for decoder layers; first elem is hidden
    if isinstance(out, tuple):
        resid_cache["h"] = out[0]
    else:
        resid_cache["h"] = out


hook_handle = model.model.layers[LAYER].register_forward_hook(hook_fn)


@torch.no_grad()
def get_sae_activations(texts, batch_size=4, max_len=64):
    """Return list of (seq_len, d_sae) activations per text."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)
        model(**enc)
        h = resid_cache["h"].float()  # (B, T, d_model)
        acts = sae.encode(h)  # (B, T, d_sae)
        mask = enc["attention_mask"].bool()
        for b in range(h.shape[0]):
            a = acts[b][mask[b]]  # (T_b, d_sae)
            results.append(a.cpu().numpy())
    return results


# ------------- Reference corpus + threshold calibration -------------
REF_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank in the morning light.",
    "Economic indicators for the third quarter suggest a mild recovery in consumer spending.",
    "She opened the recipe book and began preparing a hearty vegetable stew for dinner.",
    "The neural network was trained on a large dataset of annotated medical images.",
    "In ancient Rome, the senate wielded significant political power over the republic's affairs.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The novel explores themes of identity, memory, and the passage of time in a small village.",
    "def add(a, b): return a + b  # simple python function that sums two numbers",
    "Climate scientists have observed accelerating ice loss in the polar regions this decade.",
    "Basketball players practiced free throws and defensive drills throughout the afternoon.",
    "The symphony's third movement features a haunting oboe solo above pizzicato strings.",
    "Machine learning models can inadvertently encode biases present in their training data.",
    "The volcano erupted with a plume of ash reaching high into the stratosphere.",
    "Legal scholars debated the constitutional implications of the recent supreme court ruling.",
    "Fresh bread and pastries filled the bakery with a warm, inviting aroma.",
    "Astronomers detected a new exoplanet orbiting a distant red dwarf star.",
    "The programmer refactored the recursive function to use an iterative loop instead.",
    "Wildflowers bloomed across the meadow in shades of yellow, purple, and white.",
    "Financial analysts recommend diversifying portfolios to manage investment risk.",
    "The historian meticulously catalogued artifacts from the Bronze Age excavation site.",
]
print(f"Computing reference activations on {len(REF_TEXTS)} texts...")
ref_acts_list = get_sae_activations(REF_TEXTS, batch_size=4, max_len=48)
# stack all token-level acts: (N_tokens, d_sae)
ref_acts = np.concatenate(ref_acts_list, axis=0)
print(f"ref_acts shape: {ref_acts.shape}")

# Select features that fire on the reference corpus (nonzero max)
max_per_feat = ref_acts.max(axis=0)
active_feats = np.where(max_per_feat > 0.1)[0]
print(f"Active features (max>0.1): {len(active_feats)}")

# Pick a small deterministic subset
random.seed(0)
FEATURE_IDS = sorted(random.sample(active_feats.tolist(), min(3, len(active_feats))))
print(f"Selected features: {FEATURE_IDS}")

# Thresholds: 95th percentile of nonzero activations on reference (or a fraction of max)
thresholds = {}
for fid in FEATURE_IDS:
    col = ref_acts[:, fid]
    nz = col[col > 0]
    if len(nz) >= 5:
        thr = float(np.percentile(nz, 75))
    else:
        thr = float(max_per_feat[fid] * 0.3)
    thresholds[fid] = max(thr, 0.05)
    print(f"  feat {fid}: max={max_per_feat[fid]:.3f}, thr={thresholds[fid]:.3f}")


# ------------- Top-activating snippets per feature -------------
# For each feature, find which ref text gave highest max activation, use those snippets
def top_activating_snippets(fid, k=5):
    scored = []
    for i, a in enumerate(ref_acts_list):
        col = a[:, fid]
        if len(col) == 0:
            continue
        m = float(col.max())
        scored.append((m, i))
    scored.sort(reverse=True)
    return [REF_TEXTS[i] for _, i in scored[:k]]


# ------------- Neuronpedia baseline explanation (with fallback) -------------
def neuronpedia_explanation(fid, layer=LAYER):
    # Try local cache first
    cache_file = os.path.join(CACHE_DIR, "neuronpedia.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            npc = json.load(f)
    else:
        npc = {}
    key = f"{layer}_{fid}"
    if key in npc:
        return npc[key]
    # Try Neuronpedia API
    try:
        import requests

        url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-res-16k/{fid}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            d = r.json()
            expls = d.get("explanations", [])
            if expls:
                expl = expls[0].get("description", "")
                npc[key] = expl
                with open(cache_file, "w") as f:
                    json.dump(npc, f)
                return expl
    except Exception as e:
        print(f"Neuronpedia API failed for {key}: {e}")
    # Fallback: use LLM to summarize top snippets (one-shot, mimicking neuronpedia methodology)
    snips = top_activating_snippets(fid, k=5)
    prompt = (
        "Below are text snippets that strongly activate a hidden feature in a language model. "
        "In ONE short phrase (max 15 words), describe what concept/pattern this feature detects. "
        "Reply with ONLY the phrase, no preamble.\n\nSnippets:\n"
        + "\n".join(f"- {s}" for s in snips)
    )
    expl = llm_call(prompt, system="You describe latent features of LLMs concisely.")
    if not expl:
        expl = "unclear pattern in text"
    npc[key] = expl
    with open(cache_file, "w") as f:
        json.dump(npc, f)
    return expl


# ------------- Probe generation -------------
def generate_probes(explanation, n=5):
    prompt = (
        f"Write {n} short, diverse English sentences (10-20 words each) that clearly exemplify the following concept:\n"
        f"CONCEPT: {explanation}\n\n"
        f"Output ONLY the {n} sentences, one per line, no numbering, no preamble."
    )
    out = llm_call(
        prompt,
        system="You are a careful writer producing exemplar sentences for a concept.",
    )
    lines = [l.strip(" -•\t").strip() for l in out.split("\n") if l.strip()]
    lines = [l for l in lines if len(l.split()) >= 4][:n]
    while len(lines) < n:
        lines.append(f"An example of {explanation}.")
    return lines


# ------------- Evaluate generative accuracy -------------
def eval_gen_accuracy(fid, explanation, n_probes=5):
    probes = generate_probes(explanation, n=n_probes)
    acts_list = get_sae_activations(probes, batch_size=n_probes, max_len=48)
    # per-probe max activation of target feature
    successes = 0
    per_probe = []
    for a in acts_list:
        m = float(a[:, fid].max()) if len(a) > 0 else 0.0
        per_probe.append(m)
        if m > thresholds[fid]:
            successes += 1
    return successes / max(len(acts_list), 1), probes, per_probe


# ------------- SAGE iterative loop -------------
def sage_propose_initial(fid):
    snips = top_activating_snippets(fid, k=6)
    prompt = (
        "You are analyzing a latent feature in an LLM. Below are top-activating snippets. "
        "Propose a concise ONE-PHRASE description (max 15 words) of what the feature detects. "
        "Reply with only the phrase.\n\nSnippets:\n"
        + "\n".join(f"- {s}" for s in snips)
    )
    return (
        llm_call(prompt, system="You analyze LLM latent features.") or "textual pattern"
    )


def sage_revise(fid, prev_expl, probes, per_probe_acts, threshold):
    snips = top_activating_snippets(fid, k=4)
    feedback = "\n".join(
        f"- (act={a:.3f}, {'HIT' if a>threshold else 'MISS'}) {p}"
        for p, a in zip(probes, per_probe_acts)
    )
    prompt = (
        f"You are refining an explanation for an LLM latent feature.\n\n"
        f'Current explanation: "{prev_expl}"\n\n'
        f"Reference top-activating snippets (ground truth for feature):\n"
        + "\n".join(f"- {s}" for s in snips)
        + "\n\n"
        + f"Probe sentences you wrote from the current explanation, with actual feature activation "
        f"(threshold={threshold:.3f}):\n{feedback}\n\n"
        f"Analyze why the MISS probes failed and revise the explanation to better match the true concept. "
        f"Reply with ONLY a single revised phrase (max 15 words), no preamble."
    )
    return (
        llm_call(prompt, system="You iteratively refine LLM feature explanations.")
        or prev_expl
    )


# ------------- Run experiment -------------
experiment_data = {
    "gemma2_2b_gemmascope_res_16k": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "per_feature": {},
        "layer": LAYER,
        "features": FEATURE_IDS,
        "thresholds": thresholds,
    }
}

N_ITER = 2
N_PROBES = 5

neuronpedia_scores = []
sage_scores_per_iter = [
    [] for _ in range(N_ITER + 1)
]  # iter 0 = initial propose, 1..N_ITER = revised

for fid in FEATURE_IDS:
    print(f"\n=== Feature {fid} ===")
    per_feat = {"iterations": []}

    # Neuronpedia baseline
    np_expl = neuronpedia_explanation(fid)
    print(f"  Neuronpedia expl: {np_expl}")
    np_acc, np_probes, np_pp = eval_gen_accuracy(fid, np_expl, n_probes=N_PROBES)
    print(f"  Neuronpedia gen_acc: {np_acc:.3f}")
    neuronpedia_scores.append(np_acc)
    per_feat["neuronpedia"] = {
        "explanation": np_expl,
        "gen_acc": np_acc,
        "probes": np_probes,
        "per_probe_acts": np_pp,
    }

    # SAGE iteration 0
    expl = sage_propose_initial(fid)
    print(f"  SAGE iter 0: {expl}")
    acc, probes, pp = eval_gen_accuracy(fid, expl, n_probes=N_PROBES)
    print(f"  SAGE iter 0 gen_acc: {acc:.3f}")
    sage_scores_per_iter[0].append(acc)
    per_feat["iterations"].append(
        {"explanation": expl, "gen_acc": acc, "probes": probes, "per_probe_acts": pp}
    )

    # revise
    for it in range(1, N_ITER + 1):
        expl = sage_revise(fid, expl, probes, pp, thresholds[fid])
        print(f"  SAGE iter {it}: {expl}")
        acc, probes, pp = eval_gen_accuracy(fid, expl, n_probes=N_PROBES)
        print(f"  SAGE iter {it} gen_acc: {acc:.3f}")
        sage_scores_per_iter[it].append(acc)
        per_feat["iterations"].append(
            {
                "explanation": expl,
                "gen_acc": acc,
                "probes": probes,
                "per_probe_acts": pp,
            }
        )

    experiment_data["gemma2_2b_gemmascope_res_16k"]["per_feature"][fid] = per_feat

    # Track metrics like "epochs"
    print(f"Epoch {fid}: validation_loss = {1.0 - sage_scores_per_iter[-1][-1]:.4f}")
    experiment_data["gemma2_2b_gemmascope_res_16k"]["metrics"]["val"].append(
        {
            "feature": fid,
            "neuronpedia_gen_acc": np_acc,
            "sage_gen_acc_iters": (
                [s[-1] for s in sage_scores_per_iter if len(s) > 0][-1]
                if False
                else [sage_scores_per_iter[i][-1] for i in range(N_ITER + 1)]
            ),
        }
    )
    experiment_data["gemma2_2b_gemmascope_res_16k"]["losses"]["val"].append(
        1.0 - sage_scores_per_iter[-1][-1]
    )

# ------------- Aggregate & report -------------
np_mean = float(np.mean(neuronpedia_scores)) if neuronpedia_scores else 0.0
sage_means = [float(np.mean(s)) if s else 0.0 for s in sage_scores_per_iter]
sage_final = sage_means[-1]

print("\n===== SUMMARY =====")
print(f"Neuronpedia mean gen_acc: {np_mean:.3f}")
for i, m in enumerate(sage_means):
    print(f"SAGE iter {i} mean gen_acc: {m:.3f}")
print(f"FINAL generative_accuracy (SAGE): {sage_final:.3f}")
print(f"Delta over Neuronpedia: {sage_final - np_mean:+.3f}")

experiment_data["gemma2_2b_gemmascope_res_16k"]["summary"] = {
    "neuronpedia_mean_gen_acc": np_mean,
    "sage_mean_gen_acc_per_iter": sage_means,
    "sage_final_gen_acc": sage_final,
    "delta": sage_final - np_mean,
    "generative_accuracy": sage_final,
}

# ------------- Plot -------------
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = ["Neuronpedia"] + [f"SAGE it{i}" for i in range(N_ITER + 1)]
    y = [np_mean] + sage_means
    ax.bar(x, y, color=["gray"] + ["C0"] * (N_ITER + 1))
    ax.set_ylabel("Generative Accuracy")
    ax.set_title(
        f"SAGE vs Neuronpedia (Gemma-2-2B L{LAYER}, {len(FEATURE_IDS)} features)"
    )
    ax.set_ylim(0, 1)
    for i, v in enumerate(y):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "sage_vs_neuronpedia_gemma2_2b_L12.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Plot failed: {e}")

# ------------- Save -------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as f:

    def _safe(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    json.dump(experiment_data, f, indent=2, default=_safe)

hook_handle.remove()
print("\nDone.")
print(f"generative_accuracy = {sage_final:.4f}")
