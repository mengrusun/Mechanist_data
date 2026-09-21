import os, sys, json, time, hashlib, traceback, math, random

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

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

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
API_MODEL = "gpt-5.4"

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
        layer_dir = os.path.join(root, f"layer_{layer}", "width_16k")
        if os.path.isdir(layer_dir):
            for sub in sorted(os.listdir(layer_dir)):
                p = os.path.join(layer_dir, sub, "params.npz")
                if os.path.exists(p):
                    return p, sub
        p = os.path.join(root, f"layer_{layer}_width_16k.npz")
        if os.path.exists(p):
            return p, "flat"
    return None, None


sae_path, sae_variant = find_local_sae(LAYER)
if sae_path is None:
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
    print("FATAL: could not obtain GemmaScope SAE.")
    sys.exit(1)

print(f"Loading SAE from {sae_path} (variant={sae_variant})")
sae_params = np.load(sae_path)
print(f"SAE keys: {list(sae_params.keys())}")


class JumpReLUSAE(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        W_enc = torch.tensor(params["W_enc"], dtype=torch.float32)
        W_dec = torch.tensor(params["W_dec"], dtype=torch.float32)
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
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        acts = torch.where(pre > self.threshold, pre, torch.zeros_like(pre))
        return acts


sae = JumpReLUSAE(sae_params).to(device)
sae.eval()
D_SAE = sae.W_enc.shape[1]
print(f"SAE d_sae={D_SAE}")

resid_cache = {}


def hook_fn(module, inp, out):
    if isinstance(out, tuple):
        resid_cache["h"] = out[0]
    else:
        resid_cache["h"] = out


hook_handle = model.model.layers[LAYER].register_forward_hook(hook_fn)


@torch.no_grad()
def get_sae_activations(texts, batch_size=4, max_len=64):
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
        h = resid_cache["h"].float()
        acts = sae.encode(h)
        mask = enc["attention_mask"].bool()
        for b in range(h.shape[0]):
            a = acts[b][mask[b]]
            results.append(a.cpu().numpy())
    return results


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
ref_acts = np.concatenate(ref_acts_list, axis=0)
print(f"ref_acts shape: {ref_acts.shape}")

max_per_feat = ref_acts.max(axis=0)
active_feats = np.where(max_per_feat > 0.1)[0]
print(f"Active features (max>0.1): {len(active_feats)}")

random.seed(0)
FEATURE_IDS = sorted(random.sample(active_feats.tolist(), min(3, len(active_feats))))
print(f"Selected features: {FEATURE_IDS}")

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


def neuronpedia_explanation(fid, layer=LAYER):
    cache_file = os.path.join(CACHE_DIR, "neuronpedia.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            npc = json.load(f)
    else:
        npc = {}
    key = f"{layer}_{fid}"
    if key in npc:
        return npc[key]
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


def generate_probes(explanation, n=5, seed_tag=""):
    prompt = (
        f"Write {n} short, diverse English sentences (10-20 words each) that clearly exemplify the following concept:\n"
        f"CONCEPT: {explanation}\n\n"
        f"Output ONLY the {n} sentences, one per line, no numbering, no preamble."
        + (f"\n[variant: {seed_tag}]" if seed_tag else "")
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


def eval_gen_accuracy(fid, explanation, n_probes=5, seed_tag=""):
    probes = generate_probes(explanation, n=n_probes, seed_tag=seed_tag)
    acts_list = get_sae_activations(probes, batch_size=n_probes, max_len=48)
    successes = 0
    per_probe = []
    for a in acts_list:
        m = float(a[:, fid].max()) if len(a) > 0 else 0.0
        per_probe.append(m)
        if m > thresholds[fid]:
            successes += 1
    return successes / max(len(acts_list), 1), probes, per_probe


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


def sage_revise(fid, prev_expl, probes, per_probe_acts, threshold, iter_tag=""):
    snips = top_activating_snippets(fid, k=4)
    feedback = "\n".join(
        f"- (act={a:.3f}, {'HIT' if a>threshold else 'MISS'}) {p}"
        for p, a in zip(probes, per_probe_acts)
    )
    prompt = (
        f"You are refining an explanation for an LLM latent feature.{iter_tag}\n\n"
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


# ==================== Hyperparameter Tuning: N_ITER ====================
N_ITER_VALUES = [1, 2, 3, 5, 7]
N_PROBES = 5
DATASET_NAME = "gemma2_2b_gemmascope_res_16k"

experiment_data = {
    "N_ITER": {
        DATASET_NAME: {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "layer": LAYER,
            "features": FEATURE_IDS,
            "thresholds": thresholds,
            "n_iter_values": N_ITER_VALUES,
            "runs": {},  # keyed by N_ITER
        }
    }
}

# Neuronpedia baseline (once, shared across all N_ITER runs)
neuronpedia_scores = []
neuronpedia_per_feat = {}
for fid in FEATURE_IDS:
    np_expl = neuronpedia_explanation(fid)
    print(f"[NP] feat {fid}: {np_expl}")
    np_acc, np_probes, np_pp = eval_gen_accuracy(
        fid, np_expl, n_probes=N_PROBES, seed_tag="np"
    )
    print(f"[NP] feat {fid} gen_acc: {np_acc:.3f}")
    neuronpedia_scores.append(np_acc)
    neuronpedia_per_feat[fid] = {
        "explanation": np_expl,
        "gen_acc": np_acc,
        "probes": np_probes,
        "per_probe_acts": np_pp,
    }
np_mean = float(np.mean(neuronpedia_scores)) if neuronpedia_scores else 0.0
experiment_data["N_ITER"][DATASET_NAME]["neuronpedia"] = {
    "per_feature": neuronpedia_per_feat,
    "mean_gen_acc": np_mean,
}
print(f"\nNeuronpedia mean gen_acc: {np_mean:.3f}\n")

tuning_summary = {}

for N_ITER in N_ITER_VALUES:
    print(f"\n########## N_ITER = {N_ITER} ##########")
    sage_scores_per_iter = [[] for _ in range(N_ITER + 1)]
    per_feature_runs = {}

    for fid in FEATURE_IDS:
        print(f"\n=== [N_ITER={N_ITER}] Feature {fid} ===")
        per_feat = {"iterations": []}

        # For fair comparison across N_ITER values, use same seed_tag scheme.
        # Iter 0 is identical across N_ITER; revisions add iter idx to prompt tag (empty for reproducibility).
        expl = sage_propose_initial(fid)
        print(f"  SAGE iter 0: {expl}")
        acc, probes, pp = eval_gen_accuracy(fid, expl, n_probes=N_PROBES, seed_tag="")
        print(f"  SAGE iter 0 gen_acc: {acc:.3f}")
        sage_scores_per_iter[0].append(acc)
        per_feat["iterations"].append(
            {
                "explanation": expl,
                "gen_acc": acc,
                "probes": probes,
                "per_probe_acts": pp,
            }
        )

        for it in range(1, N_ITER + 1):
            expl = sage_revise(fid, expl, probes, pp, thresholds[fid], iter_tag="")
            print(f"  SAGE iter {it}: {expl}")
            acc, probes, pp = eval_gen_accuracy(
                fid, expl, n_probes=N_PROBES, seed_tag=""
            )
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

        per_feature_runs[fid] = per_feat

    sage_means = [float(np.mean(s)) if s else 0.0 for s in sage_scores_per_iter]
    sage_final = sage_means[-1]
    print(f"\n[N_ITER={N_ITER}] means per iter: {[f'{m:.3f}' for m in sage_means]}")
    print(
        f"[N_ITER={N_ITER}] FINAL gen_acc: {sage_final:.3f} (delta vs NP: {sage_final - np_mean:+.3f})"
    )

    run_data = {
        "n_iter": N_ITER,
        "per_feature": per_feature_runs,
        "sage_mean_gen_acc_per_iter": sage_means,
        "sage_final_gen_acc": sage_final,
        "delta_over_neuronpedia": sage_final - np_mean,
    }
    experiment_data["N_ITER"][DATASET_NAME]["runs"][str(N_ITER)] = run_data

    # Log metrics/losses in the standard structure
    experiment_data["N_ITER"][DATASET_NAME]["metrics"]["val"].append(
        {
            "n_iter": N_ITER,
            "sage_final_gen_acc": sage_final,
            "sage_mean_gen_acc_per_iter": sage_means,
            "neuronpedia_mean_gen_acc": np_mean,
        }
    )
    experiment_data["N_ITER"][DATASET_NAME]["losses"]["val"].append(
        {"n_iter": N_ITER, "val_loss": 1.0 - sage_final}
    )
    tuning_summary[N_ITER] = sage_final

# Pick best N_ITER
best_n_iter = max(tuning_summary, key=lambda k: tuning_summary[k])
best_score = tuning_summary[best_n_iter]

print("\n===== HYPERPARAM TUNING SUMMARY (N_ITER) =====")
print(f"Neuronpedia mean gen_acc: {np_mean:.3f}")
for n_iter, sc in tuning_summary.items():
    marker = " <-- BEST" if n_iter == best_n_iter else ""
    print(f"  N_ITER={n_iter}: final gen_acc = {sc:.3f}{marker}")
print(f"\nBest N_ITER = {best_n_iter} with generative_accuracy = {best_score:.4f}")
print(f"Delta over Neuronpedia: {best_score - np_mean:+.3f}")

experiment_data["N_ITER"][DATASET_NAME]["summary"] = {
    "neuronpedia_mean_gen_acc": np_mean,
    "tuning_summary": {str(k): v for k, v in tuning_summary.items()},
    "best_n_iter": best_n_iter,
    "best_sage_final_gen_acc": best_score,
    "generative_accuracy": best_score,
    "delta": best_score - np_mean,
}

# ------------- Plot -------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Plot 1: final acc vs N_ITER
    xs = sorted(tuning_summary.keys())
    ys = [tuning_summary[x] for x in xs]
    axes[0].plot(xs, ys, "o-", label="SAGE final gen_acc")
    axes[0].axhline(
        np_mean, color="gray", linestyle="--", label=f"Neuronpedia ({np_mean:.2f})"
    )
    axes[0].set_xlabel("N_ITER (SAGE refinement iterations)")
    axes[0].set_ylabel("Generative Accuracy")
    axes[0].set_title("Hyperparam tuning: N_ITER")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    for x, y in zip(xs, ys):
        axes[0].text(x, y + 0.02, f"{y:.2f}", ha="center")

    # Plot 2: mean acc curves per iteration for each N_ITER
    for n_iter in xs:
        means = experiment_data["N_ITER"][DATASET_NAME]["runs"][str(n_iter)][
            "sage_mean_gen_acc_per_iter"
        ]
        axes[1].plot(range(len(means)), means, "o-", label=f"N_ITER={n_iter}")
    axes[1].axhline(np_mean, color="gray", linestyle="--", label=f"Neuronpedia")
    axes[1].set_xlabel("SAGE iteration index")
    axes[1].set_ylabel("Mean generative accuracy")
    axes[1].set_title("SAGE refinement trajectories")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "sage_n_iter_tuning_gemma2_2b_L12.png"), dpi=120
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
print(f"generative_accuracy = {best_score:.4f} (best N_ITER = {best_n_iter})")
