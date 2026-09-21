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

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
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

import numpy as np
import torch
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

try:
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    HAS_LLM = True
except Exception as e:
    print(f"OpenAI init failed: {e}")
    HAS_LLM = False

LLM_CACHE_PATH = os.path.join(CACHE_DIR, "llm_cache.json")
llm_cache = json.load(open(LLM_CACHE_PATH)) if os.path.exists(LLM_CACHE_PATH) else {}
FALLBACK_LOG = {
    "llm_empty": 0,
    "neuronpedia_api_ok": 0,
    "neuronpedia_api_fail": 0,
    "neuronpedia_fallback_llm": 0,
}


def llm_call(prompt, system="You are a helpful research assistant.", max_retries=2):
    key = hashlib.sha256((system + "||" + prompt).encode()).hexdigest()
    if key in llm_cache:
        return llm_cache[key]
    if not HAS_LLM:
        FALLBACK_LOG["llm_empty"] += 1
        return ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=API_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                timeout=90,
            )
            out = resp.choices[0].message.content.strip()
            llm_cache[key] = out
            if len(llm_cache) % 10 == 0:
                json.dump(llm_cache, open(LLM_CACHE_PATH, "w"))
            return out
        except Exception as e:
            print(f"LLM attempt {attempt+1} failed: {e}")
            time.sleep(2 + attempt * 2)
    FALLBACK_LOG["llm_empty"] += 1
    return ""


# ---------- Load Gemma-2-2B ----------
from transformers import AutoTokenizer, AutoModelForCausalLM

GEMMA_LOCAL = os.path.join(MODEL_DIR, "gemma-2-2b")
gemma_path = GEMMA_LOCAL if os.path.isdir(GEMMA_LOCAL) else "google/gemma-2-2b"
print(f"Loading Gemma from: {gemma_path}")
tokenizer = AutoTokenizer.from_pretrained(
    gemma_path, local_files_only=os.path.isdir(GEMMA_LOCAL)
)
model = AutoModelForCausalLM.from_pretrained(
    gemma_path,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    local_files_only=os.path.isdir(GEMMA_LOCAL),
).to(device)
model.eval()

# ---------- SAE loader ----------
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
    return None, None


def load_sae_for_layer(layer):
    sae_path, variant = find_local_sae(layer)
    if sae_path is None:
        try:
            from huggingface_hub import hf_hub_download

            for l0 in [82, 22, 41, 176, 445, 68, 72, 74, 140, 142]:
                try:
                    sae_path = hf_hub_download(
                        repo_id="google/gemma-scope-2b-pt-res",
                        filename=f"layer_{layer}/width_16k/average_l0_{l0}/params.npz",
                        cache_dir=os.path.join(MODEL_DIR, "hf_cache"),
                    )
                    variant = f"average_l0_{l0}"
                    break
                except Exception:
                    continue
        except Exception as e:
            print(f"HF SAE fetch fail L{layer}: {e}")
    if sae_path is None:
        return None, None
    params = np.load(sae_path)
    return params, variant


class JumpReLUSAE(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.W_enc = torch.nn.Parameter(
            torch.tensor(params["W_enc"], dtype=torch.float32), requires_grad=False
        )
        self.W_dec = torch.nn.Parameter(
            torch.tensor(params["W_dec"], dtype=torch.float32), requires_grad=False
        )
        self.b_enc = torch.nn.Parameter(
            torch.tensor(params["b_enc"], dtype=torch.float32), requires_grad=False
        )
        self.b_dec = torch.nn.Parameter(
            torch.tensor(params["b_dec"], dtype=torch.float32), requires_grad=False
        )
        thr = (
            torch.tensor(params["threshold"], dtype=torch.float32)
            if "threshold" in params.files
            else torch.zeros(self.W_enc.shape[1])
        )
        self.threshold = torch.nn.Parameter(thr, requires_grad=False)

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))


# ---------- Layers to study ----------
LAYERS = [5, 12, 20]
saes = {}
for L in LAYERS:
    params, variant = load_sae_for_layer(L)
    if params is None:
        print(f"Skipping layer {L}: no SAE")
        continue
    s = JumpReLUSAE(params).to(device)
    s.eval()
    saes[L] = s
    print(f"Loaded SAE L{L} variant={variant} d_sae={s.W_enc.shape[1]}")

if not saes:
    print("FATAL: no SAEs loaded")
    sys.exit(1)

# ---------- Hooks for all layers of interest ----------
resid_cache = {}


def make_hook(L):
    def h(module, inp, out):
        resid_cache[L] = out[0] if isinstance(out, tuple) else out

    return h


hooks = [model.model.layers[L].register_forward_hook(make_hook(L)) for L in saes.keys()]


@torch.no_grad()
def get_sae_activations_all_layers(texts, batch_size=8, max_len=48):
    """Returns dict {layer: list of (T_b, d_sae) arrays per text}"""
    out = {L: [] for L in saes.keys()}
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
        mask = enc["attention_mask"].bool()
        for L, sae in saes.items():
            h = resid_cache[L].float()
            acts = sae.encode(h)
            for b in range(h.shape[0]):
                a = acts[b][mask[b]].cpu().numpy()
                out[L].append(a)
    return out


# ---------- Reference corpus (larger, diverse) ----------
REF_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank in morning light.",
    "Economic indicators for the third quarter suggest a mild recovery in consumer spending.",
    "She opened the recipe book and prepared a hearty vegetable stew with fresh herbs.",
    "The neural network was trained on a large dataset of annotated medical images.",
    "In ancient Rome, the senate wielded significant political power over republic affairs.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The novel explores themes of identity, memory, and the passage of time.",
    "def add(a, b): return a + b  # simple python function that sums two numbers",
    "Climate scientists observed accelerating ice loss in the polar regions this decade.",
    "Basketball players practiced free throws and defensive drills throughout the afternoon.",
    "The symphony's third movement features a haunting oboe solo above pizzicato strings.",
    "Machine learning models can inadvertently encode biases present in training data.",
    "The volcano erupted with a plume of ash reaching high into the stratosphere.",
    "Legal scholars debated constitutional implications of the recent supreme court ruling.",
    "Fresh bread and pastries filled the bakery with a warm, inviting aroma.",
    "Astronomers detected a new exoplanet orbiting a distant red dwarf star.",
    "The programmer refactored the recursive function to use an iterative loop instead.",
    "Wildflowers bloomed across the meadow in shades of yellow, purple, and white.",
    "Financial analysts recommend diversifying portfolios to manage investment risk.",
    "The historian meticulously catalogued artifacts from the Bronze Age excavation site.",
    "He whispered goodbye and closed the wooden door softly behind him.",
    "The startup raised twenty million dollars in its Series B funding round.",
    "Roasting garlic brings out its sweet, nutty flavor and softens its texture.",
    "Quantum entanglement remains one of the strangest phenomena in modern physics.",
    "Children laughed as they chased fireflies in the summer twilight.",
    "The judge sentenced the defendant to five years for aggravated assault.",
    "import numpy as np\nx = np.zeros((10, 3))\nprint(x.shape)",
    "Baroque composers favored intricate counterpoint and ornate melodic lines.",
    "The rainforest canopy teems with insects, birds, and small mammals.",
    "Vintage cars gleamed under the showroom lights, chrome polished to a mirror.",
    "She earned her PhD in computational linguistics after five years of research.",
    "The chef seared the scallops in browned butter with a squeeze of lemon.",
    "Waves crashed against the rocky cliffs, sending spray high into the air.",
    "Investors panicked as tech stocks plunged sharply on the earnings news.",
    "The archaeologist brushed away centuries of dust from the ancient inscription.",
    "Rap lyrics often reference urban life, struggle, and personal ambition.",
    "The pilot announced turbulence and asked passengers to fasten their seatbelts.",
    "Bees are essential pollinators, sustaining a large portion of global agriculture.",
    "Gothic cathedrals feature flying buttresses, pointed arches, and stained glass.",
    "The dog wagged its tail furiously when its owner came home.",
    "Mitochondria produce ATP through the electron transport chain and oxidative phosphorylation.",
    "The politician deflected questions about the ongoing corruption scandal.",
    "A gentle rain tapped against the window as she read by candlelight.",
    "SELECT name, COUNT(*) FROM users GROUP BY name ORDER BY COUNT(*) DESC;",
    "Soccer fans erupted in cheers when their team scored in extra time.",
    "The pharmacist double-checked the dosage before dispensing the medication.",
    "Renaissance painters mastered perspective, anatomy, and the play of light and shadow.",
    "Hurricane winds tore roofs off houses along the coastal highway.",
    "Ballet dancers rehearsed grand jetés and pirouettes in the studio.",
    "Cryptocurrency markets can be extremely volatile due to speculation and hype.",
    "The librarian recommended a novel about post-war Japan and family bonds.",
    "Volcanic soil is often rich in minerals and highly fertile for agriculture.",
    "The AI assistant scheduled meetings and drafted emails throughout the day.",
    "Ancient Egyptians built pyramids as tombs for their pharaohs and queens.",
    "Fermented foods like kimchi and sauerkraut are rich in beneficial bacteria.",
    "The mountaineer reached the summit despite freezing temperatures and thin air.",
    "War correspondents risk their lives to document conflicts around the world.",
    "The kitten batted at a dangling piece of yarn with tiny paws.",
    "Public health officials urged residents to get vaccinated before flu season.",
    "The novelist described the desert as endless waves of red and gold sand.",
    "Reinforcement learning agents learn optimal policies through trial and error.",
    "The couple exchanged vows in a small ceremony by the lakeside.",
    "Merchants in the medieval marketplace haggled over spices and silk.",
    "The astronomer calibrated the telescope for the upcoming lunar eclipse.",
    "Jazz musicians improvise over chord changes with syncopated rhythms.",
    "Rising sea levels threaten low-lying coastal cities around the world.",
    "The blacksmith hammered red-hot iron into the shape of a horseshoe.",
    "She debugged the memory leak by tracing allocations in the profiler.",
    "The auction house sold the impressionist painting for a record price.",
    "Volunteers distributed food and blankets to families displaced by the flood.",
    "The philosopher argued that free will is compatible with physical determinism.",
    "Colorful koi swam lazily beneath the arched wooden bridge in the garden.",
    "Firefighters battled the blaze for hours before finally containing it.",
    "The scientist grafted the gene into a plasmid and transformed E. coli cells.",
    "Newborn babies sleep for most of the day and cry when hungry.",
    "The poet's verses lingered in the reader's mind long after the book closed.",
    "Traders on the exchange floor shouted bids and offers in rapid-fire jargon.",
    "The ballerina's silhouette twirled elegantly against the crimson stage curtain.",
]
print(f"Ref corpus size: {len(REF_TEXTS)}")

print("Computing reference activations across layers...")
ref_all = get_sae_activations_all_layers(REF_TEXTS, batch_size=8, max_len=48)
ref_stack = {L: np.concatenate(ref_all[L], axis=0) for L in ref_all}
for L in ref_stack:
    print(f"  L{L}: ref_acts shape {ref_stack[L].shape}")

# ---------- Feature selection per layer ----------
random.seed(42)
N_FEATURES_PER_LAYER = 10
selected = {}  # {L: [fids]}
thresholds = {}  # {(L,fid): thr}
top_snippets = {}  # {(L,fid): [texts]}
freq_map = {}  # {(L,fid): frequency}

for L in saes.keys():
    ref = ref_stack[L]  # (N_tokens, d_sae)
    freq = (ref > 0).mean(axis=0)  # activation frequency
    maxv = ref.max(axis=0)
    # candidates with moderate frequency (not dead, not too common)
    cand = np.where((freq > 0.005) & (freq < 0.3) & (maxv > 0.5))[0].tolist()
    if len(cand) < N_FEATURES_PER_LAYER:
        cand = np.where(maxv > 0.1)[0].tolist()
    random.shuffle(cand)
    chosen = sorted(cand[:N_FEATURES_PER_LAYER])
    selected[L] = chosen
    print(f"L{L}: chose {len(chosen)} features from {len(cand)} candidates")
    for fid in chosen:
        col = ref[:, fid]
        nz = col[col > 0]
        thr = float(np.percentile(nz, 70)) if len(nz) >= 5 else float(maxv[fid] * 0.3)
        thresholds[(L, fid)] = max(thr, 0.05)
        freq_map[(L, fid)] = float(freq[fid])
        # top snippets
        scored = []
        for i, a in enumerate(ref_all[L]):
            if len(a) == 0:
                continue
            scored.append((float(a[:, fid].max()), i))
        scored.sort(reverse=True)
        top_snippets[(L, fid)] = [REF_TEXTS[i] for _, i in scored[:8] if _ > 0]


# ---------- Explanation helpers ----------
def neuronpedia_explanation(L, fid):
    cache_file = os.path.join(CACHE_DIR, "neuronpedia.json")
    npc = json.load(open(cache_file)) if os.path.exists(cache_file) else {}
    key = f"{L}_{fid}"
    if key in npc:
        return npc[key], "cache"
    src = "api"
    try:
        import requests

        url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{L}-gemmascope-res-16k/{fid}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            d = r.json()
            expls = d.get("explanations", [])
            if expls:
                expl = expls[0].get("description", "").strip()
                if expl:
                    FALLBACK_LOG["neuronpedia_api_ok"] += 1
                    npc[key] = expl
                    json.dump(npc, open(cache_file, "w"))
                    return expl, "api"
        FALLBACK_LOG["neuronpedia_api_fail"] += 1
    except Exception as e:
        FALLBACK_LOG["neuronpedia_api_fail"] += 1
    # LLM one-shot fallback mimicking Neuronpedia
    FALLBACK_LOG["neuronpedia_fallback_llm"] += 1
    snips = top_snippets.get((L, fid), [])[:5]
    prompt = (
        "Below are text snippets that strongly activate a hidden feature in a language model. "
        "In ONE short phrase (max 15 words), describe what concept/pattern this feature detects. "
        "Reply with ONLY the phrase.\n\nSnippets:\n"
        + "\n".join(f"- {s}" for s in snips)
    )
    expl = llm_call(prompt, system="You describe latent features of LLMs concisely.")
    if not expl:
        expl = "unclear textual pattern"
    npc[key] = expl
    json.dump(npc, open(cache_file, "w"))
    return expl, "llm_fallback"


def strong_single_shot_explanation(L, fid):
    """Strong single-shot GPT-5 baseline: uses more context but no iteration."""
    snips = top_snippets.get((L, fid), [])[:8]
    prompt = (
        "You are analyzing a latent feature (SAE direction) in a language model. "
        "Below are up to 8 top-activating text snippets. Study them carefully and write "
        "a SINGLE precise natural-language description of the pattern/concept the feature detects. "
        "Be specific — mention lexical, syntactic, semantic, or topical properties as needed. "
        "Reply with ONLY one phrase or short sentence (max 25 words), no preamble.\n\nSnippets:\n"
        + "\n".join(f"- {s}" for s in snips)
    )
    out = llm_call(
        prompt, system="You give precise, specific descriptions of LLM latent features."
    )
    return out or "textual pattern"


def sage_propose_initial(L, fid):
    snips = top_snippets.get((L, fid), [])[:8]
    prompt = (
        "You are analyzing an LLM latent feature. Below are top-activating snippets. "
        "Propose K=3 candidate concise hypotheses (each max 20 words) about what the feature detects. "
        "Return ONLY a JSON array of 3 strings, nothing else.\n\nSnippets:\n"
        + "\n".join(f"- {s}" for s in snips)
    )
    out = llm_call(prompt, system="You analyze LLM latent features and return JSON.")
    try:
        arr = json.loads(out[out.find("[") : out.rfind("]") + 1])
        arr = [str(x).strip() for x in arr if str(x).strip()][:3]
        if arr:
            return arr
    except Exception:
        pass
    return [out or "textual pattern"]


def generate_probes(explanation, n=6, diversify=True):
    diversity_note = (
        (
            " Vary syntactic structure (question, declarative, imperative), tense, and topic. "
            "Do NOT reuse the same opening word."
        )
        if diversify
        else ""
    )
    prompt = (
        f"Write {n} short English sentences (10-20 words each) that clearly and specifically exemplify the concept below.{diversity_note}\n"
        f"CONCEPT: {explanation}\n\n"
        f"Output ONLY the {n} sentences, one per line, no numbering, no preamble."
    )
    out = llm_call(prompt, system="You produce diverse exemplar sentences.")
    lines = [l.strip(" -•\t*").strip() for l in out.split("\n") if l.strip()]
    lines = [l for l in lines if len(l.split()) >= 4][:n]
    while len(lines) < n:
        lines.append(f"An example of {explanation}.")
    return lines


def generate_negatives(explanation, n=4):
    prompt = (
        f"Write {n} short sentences (10-20 words) that are on plausible but DIFFERENT topics from the concept below "
        f"(they should NOT exemplify it). Vary topic broadly.\n"
        f"CONCEPT TO AVOID: {explanation}\n\n"
        f"Output ONLY the {n} sentences, one per line, no numbering."
    )
    out = llm_call(prompt, system="You produce contrastive off-topic sentences.")
    lines = [l.strip(" -•\t*").strip() for l in out.split("\n") if l.strip()]
    lines = [l for l in lines if len(l.split()) >= 4][:n]
    while len(lines) < n:
        lines.append("A random unrelated sentence about ordinary weather.")
    return lines


def llm_predict_activation(explanation, sentences):
    """LLM predicts on a 0-5 scale whether each sentence matches the explanation."""
    joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = (
        f"For each sentence below, rate on an integer scale 0-5 how strongly it exemplifies this concept:\n"
        f'CONCEPT: "{explanation}"\n\n0=not at all, 5=perfect exemplar.\n'
        f"Sentences:\n{joined}\n\n"
        f"Return ONLY a JSON array of {len(sentences)} integers, e.g. [3,0,5,...]"
    )
    out = llm_call(prompt, system="You rate sentence-concept match.")
    try:
        arr = json.loads(out[out.find("[") : out.rfind("]") + 1])
        arr = [float(x) for x in arr][: len(sentences)]
        while len(arr) < len(sentences):
            arr.append(0.0)
        return arr
    except Exception:
        return [2.5] * len(sentences)


def eval_generative(L, fid, explanation, n_probes=6):
    probes = generate_probes(explanation, n=n_probes)
    acts_all = get_sae_activations_all_layers(probes, batch_size=n_probes, max_len=48)
    per_probe = []
    for a in acts_all[L]:
        m = float(a[:, fid].max()) if len(a) > 0 else 0.0
        per_probe.append(m)
    thr = thresholds[(L, fid)]
    succ = sum(1 for m in per_probe if m > thr) / max(len(per_probe), 1)
    return succ, probes, per_probe


def eval_predictive(L, fid, explanation, n_pos=5, n_neg=4):
    """Predictive accuracy: correlation between LLM predicted 0-5 scores and true feature max activations on mixed pos+neg probes."""
    pos = generate_probes(explanation, n=n_pos)
    neg = generate_negatives(explanation, n=n_neg)
    sents = pos + neg
    random.shuffle(sents)
    acts_all = get_sae_activations_all_layers(sents, batch_size=len(sents), max_len=48)
    true_acts = [float(a[:, fid].max()) if len(a) > 0 else 0.0 for a in acts_all[L]]
    preds = llm_predict_activation(explanation, sents)
    if np.std(true_acts) < 1e-6 or np.std(preds) < 1e-6:
        corr = 0.0
    else:
        corr = float(np.corrcoef(preds, true_acts)[0, 1])
        if math.isnan(corr):
            corr = 0.0
    # also specificity: mean act on pos vs neg
    pos_idx = [i for i, s in enumerate(sents) if s in pos]
    neg_idx = [i for i, s in enumerate(sents) if s in neg]
    pos_mean = float(np.mean([true_acts[i] for i in pos_idx])) if pos_idx else 0.0
    neg_mean = float(np.mean([true_acts[i] for i in neg_idx])) if neg_idx else 0.0
    return corr, pos_mean, neg_mean, sents, true_acts, preds


def sage_revise(L, fid, cands, probe_feedback, threshold):
    """Given current candidate explanations and probe-activation feedback, propose K=3 revised candidates."""
    snips = top_snippets.get((L, fid), [])[:5]
    fb_lines = []
    for cand, (probes, per_probe) in zip(cands, probe_feedback):
        fb_lines.append(f'  CANDIDATE: "{cand}"')
        for p, a in zip(probes, per_probe):
            tag = "HIT" if a > threshold else "MISS"
            fb_lines.append(f"    [{tag} act={a:.2f}] {p}")
    prompt = (
        "You are refining explanations for an LLM latent feature.\n\n"
        "Reference top-activating snippets (ground truth):\n"
        + "\n".join(f"- {s}" for s in snips)
        + "\n\n"
        f"Current candidate explanations and probe activations (threshold={threshold:.2f}):\n"
        + "\n".join(fb_lines)
        + "\n\n"
        "Analyze what MISS probes have in common and what HIT probes share. "
        "Propose K=3 REVISED candidate explanations (each max 20 words) that better capture the feature, "
        "potentially spanning multiple concepts if the feature seems polysemantic. "
        "Return ONLY a JSON array of 3 strings."
    )
    out = llm_call(prompt, system="You iteratively refine LLM feature explanations.")
    try:
        arr = json.loads(out[out.find("[") : out.rfind("]") + 1])
        arr = [str(x).strip() for x in arr if str(x).strip()][:3]
        if arr:
            return arr
    except Exception:
        pass
    return cands  # fall back to same candidates


# ---------- Main experiment ----------
experiment_data = {
    "gemma2_2b_gemmascope_res_16k_multi": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "per_feature": {},
        "layers": list(saes.keys()),
        "thresholds": {f"L{L}_f{f}": thresholds[(L, f)] for (L, f) in thresholds},
        "freq_map": {f"L{L}_f{f}": freq_map[(L, f)] for (L, f) in freq_map},
        "fallback_log": FALLBACK_LOG,
    }
}

N_ITER = 4  # up to 4 revision rounds after initial
N_PROBES = 6
PLATEAU_EPS = 0.05

all_rows = []  # for aggregation

start_time = time.time()
total_features = sum(len(v) for v in selected.values())
done = 0

for L in saes.keys():
    for fid in selected[L]:
        done += 1
        elapsed = time.time() - start_time
        print(
            f"\n=== [{done}/{total_features}] L{L} feature {fid} (freq={freq_map[(L,fid)]:.4f}, thr={thresholds[(L,fid)]:.3f}) | elapsed={elapsed:.0f}s ==="
        )
        pf = {"L": L, "fid": fid, "iterations": []}

        # --- Neuronpedia baseline ---
        np_expl, np_src = neuronpedia_explanation(L, fid)
        print(f"  Neuronpedia ({np_src}): {np_expl}")
        np_gen, np_probes, np_pp = eval_generative(L, fid, np_expl, n_probes=N_PROBES)
        np_pred, np_pos, np_neg, _, _, _ = eval_predictive(L, fid, np_expl)
        print(
            f"  Neuronpedia gen={np_gen:.3f}  pred={np_pred:.3f}  pos={np_pos:.2f} neg={np_neg:.2f}"
        )
        pf["neuronpedia"] = dict(
            explanation=np_expl,
            source=np_src,
            gen_acc=np_gen,
            pred_acc=np_pred,
            pos_mean=np_pos,
            neg_mean=np_neg,
        )

        # --- Strong single-shot GPT-5 baseline ---
        ss_expl = strong_single_shot_explanation(L, fid)
        print(f"  SingleShot: {ss_expl}")
        ss_gen, _, _ = eval_generative(L, fid, ss_expl, n_probes=N_PROBES)
        ss_pred, ss_pos, ss_neg, _, _, _ = eval_predictive(L, fid, ss_expl)
        print(f"  SingleShot gen={ss_gen:.3f}  pred={ss_pred:.3f}")
        pf["single_shot"] = dict(
            explanation=ss_expl,
            gen_acc=ss_gen,
            pred_acc=ss_pred,
            pos_mean=ss_pos,
            neg_mean=ss_neg,
        )

        # --- SAGE iterative with multiple candidates ---
        cands = sage_propose_initial(L, fid)
        print(f"  SAGE init candidates: {cands}")

        best_expl = None
        best_gen = -1
        best_pred = -1
        best_score = -1  # combined
        prev_best_score = -1
        plateau_ct = 0

        for it in range(N_ITER + 1):
            # evaluate each candidate
            cand_records = []
            probe_fb = []
            for c in cands:
                g, probes, pp = eval_generative(L, fid, c, n_probes=N_PROBES)
                pr, pos_m, neg_m, _, _, _ = eval_predictive(L, fid, c)
                score = g + max(pr, 0)  # combined
                cand_records.append(
                    dict(
                        explanation=c,
                        gen_acc=g,
                        pred_acc=pr,
                        pos_mean=pos_m,
                        neg_mean=neg_m,
                        score=score,
                        probes=probes,
                        per_probe_acts=pp,
                    )
                )
                probe_fb.append((probes, pp))
                if score > best_score:
                    best_score = score
                    best_expl = c
                    best_gen = g
                    best_pred = pr
            iter_best = max(cand_records, key=lambda r: r["score"])
            print(
                f"  SAGE iter {it}: best cand gen={iter_best['gen_acc']:.3f} pred={iter_best['pred_acc']:.3f} score={iter_best['score']:.3f}  best-so-far score={best_score:.3f}"
            )
            pf["iterations"].append(
                dict(
                    iter=it,
                    candidates=cand_records,
                    best_so_far_score=best_score,
                    best_so_far_expl=best_expl,
                    best_so_far_gen=best_gen,
                    best_so_far_pred=best_pred,
                )
            )
            # early stop
            if it > 0 and (best_score - prev_best_score) < PLATEAU_EPS:
                plateau_ct += 1
            else:
                plateau_ct = 0
            prev_best_score = best_score
            if plateau_ct >= 2 and it >= 2:
                print(f"  Early stop at iter {it}")
                break
            if it < N_ITER:
                cands = sage_revise(L, fid, cands, probe_fb, thresholds[(L, fid)])
                if not cands:
                    cands = [best_expl]
                # ensure diversity — dedupe
                seen = set()
                cands = [c for c in cands if not (c in seen or seen.add(c))]
                print(f"  SAGE iter {it+1} revised cands: {cands}")

        pf["sage_final"] = dict(
            explanation=best_expl,
            gen_acc=best_gen,
            pred_acc=best_pred,
            score=best_score,
        )
        experiment_data["gemma2_2b_gemmascope_res_16k_multi"]["per_feature"][
            f"L{L}_f{fid}"
        ] = pf

        # bookkeeping
        all_rows.append(
            dict(
                L=L,
                fid=fid,
                freq=freq_map[(L, fid)],
                np_gen=np_gen,
                np_pred=np_pred,
                ss_gen=ss_gen,
                ss_pred=ss_pred,
                sage_gen=best_gen,
                sage_pred=best_pred,
            )
        )
        print(f"Epoch {done}: validation_loss = {1.0 - best_gen:.4f}")
        experiment_data["gemma2_2b_gemmascope_res_16k_multi"]["metrics"]["val"].append(
            dict(
                feature=f"L{L}_f{fid}",
                np_gen=np_gen,
                np_pred=np_pred,
                ss_gen=ss_gen,
                ss_pred=ss_pred,
                sage_gen=best_gen,
                sage_pred=best_pred,
            )
        )
        experiment_data["gemma2_2b_gemmascope_res_16k_multi"]["losses"]["val"].append(
            1.0 - best_gen
        )


# ---------- Aggregate ----------
def mean_ci(vals):
    if not vals:
        return 0.0, 0.0
    v = np.array(vals, dtype=float)
    m = float(v.mean())
    if len(v) > 1:
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
    else:
        se = 0.0
    return m, 1.96 * se


summary = {}
for L in saes.keys():
    rows_L = [r for r in all_rows if r["L"] == L]
    if not rows_L:
        continue

    def agg(k):
        m, ci = mean_ci([r[k] for r in rows_L])
        return dict(mean=m, ci=ci)

    summary[f"L{L}"] = dict(
        n=len(rows_L),
        np_gen=agg("np_gen"),
        np_pred=agg("np_pred"),
        ss_gen=agg("ss_gen"),
        ss_pred=agg("ss_pred"),
        sage_gen=agg("sage_gen"),
        sage_pred=agg("sage_pred"),
    )


# overall
def agg_all(k):
    m, ci = mean_ci([r[k] for r in all_rows])
    return dict(mean=m, ci=ci)


summary["OVERALL"] = dict(
    n=len(all_rows),
    np_gen=agg_all("np_gen"),
    np_pred=agg_all("np_pred"),
    ss_gen=agg_all("ss_gen"),
    ss_pred=agg_all("ss_pred"),
    sage_gen=agg_all("sage_gen"),
    sage_pred=agg_all("sage_pred"),
)

print("\n===== SUMMARY (mean ± 95% CI) =====")
for k, v in summary.items():
    print(f"[{k}] n={v['n']}")
    for m in ["np", "ss", "sage"]:
        print(
            f"  {m}_gen  = {v[m+'_gen']['mean']:.3f} ± {v[m+'_gen']['ci']:.3f}   "
            f"{m}_pred = {v[m+'_pred']['mean']:.3f} ± {v[m+'_pred']['ci']:.3f}"
        )

overall_gen = summary["OVERALL"]["sage_gen"]["mean"]
overall_np_gen = summary["OVERALL"]["np_gen"]["mean"]
overall_pred = summary["OVERALL"]["sage_pred"]["mean"]
overall_np_pred = summary["OVERALL"]["np_pred"]["mean"]
print(
    f"\nDelta SAGE vs Neuronpedia: gen {overall_gen-overall_np_gen:+.3f}, pred {overall_pred-overall_np_pred:+.3f}"
)
print(f"Fallback log: {FALLBACK_LOG}")

experiment_data["gemma2_2b_gemmascope_res_16k_multi"]["summary"] = summary
experiment_data["gemma2_2b_gemmascope_res_16k_multi"]["all_rows"] = all_rows
experiment_data["gemma2_2b_gemmascope_res_16k_multi"][
    "generative_activation_success_rate"
] = overall_gen

# ---------- Plots ----------
try:
    # Bar chart per layer: gen accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    methods = ["Neuronpedia", "SingleShot", "SAGE"]
    layer_keys = [k for k in summary if k.startswith("L")] + ["OVERALL"]
    x = np.arange(len(layer_keys))
    w = 0.25
    for i, m in enumerate(["np", "ss", "sage"]):
        vals = [summary[k][m + "_gen"]["mean"] for k in layer_keys]
        errs = [summary[k][m + "_gen"]["ci"] for k in layer_keys]
        axes[0].bar(x + i * w - w, vals, w, yerr=errs, label=methods[i], capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layer_keys)
    axes[0].set_ylabel("Generative Accuracy")
    axes[0].set_title("Generative Accuracy by Layer")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    for i, m in enumerate(["np", "ss", "sage"]):
        vals = [summary[k][m + "_pred"]["mean"] for k in layer_keys]
        errs = [summary[k][m + "_pred"]["ci"] for k in layer_keys]
        axes[1].bar(x + i * w - w, vals, w, yerr=errs, label=methods[i], capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_keys)
    axes[1].set_ylabel("Predictive Accuracy (corr)")
    axes[1].set_title("Predictive Accuracy by Layer")
    axes[1].legend()
    axes[1].axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "sage_by_layer.png"), dpi=120)
    plt.close()

    # Per-feature delta scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    dfreqs = [r["freq"] for r in all_rows]
    deltas = [r["sage_gen"] - r["np_gen"] for r in all_rows]
    colors = [
        "C0" if r["L"] == 5 else ("C1" if r["L"] == 12 else "C2") for r in all_rows
    ]
    ax.scatter(dfreqs, deltas, c=colors, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Feature activation frequency (log)")
    ax.set_ylabel("Gen-acc delta (SAGE - Neuronpedia)")
    ax.set_title("Per-feature improvement vs feature frequency")
    for L, c in zip([5, 12, 20], ["C0", "C1", "C2"]):
        ax.scatter([], [], c=c, label=f"L{L}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "sage_delta_vs_freq.png"), dpi=120)
    plt.close()

    # Iteration curves: mean best-so-far gen/pred over iterations, aggregated
    fig, ax = plt.subplots(figsize=(8, 5))
    max_iters = 0
    for k, pf in experiment_data["gemma2_2b_gemmascope_res_16k_multi"][
        "per_feature"
    ].items():
        max_iters = max(max_iters, len(pf["iterations"]))
    iter_gens = [[] for _ in range(max_iters)]
    for k, pf in experiment_data["gemma2_2b_gemmascope_res_16k_multi"][
        "per_feature"
    ].items():
        best_g = -1
        for i, it in enumerate(pf["iterations"]):
            g = max(c["gen_acc"] for c in it["candidates"])
            best_g = max(best_g, g)
            iter_gens[i].append(best_g)
        # pad
        for i in range(len(pf["iterations"]), max_iters):
            iter_gens[i].append(best_g)
    means = [np.mean(v) if v else 0 for v in iter_gens]
    ax.plot(range(max_iters), means, marker="o", label="SAGE best-so-far gen_acc")
    ax.axhline(
        overall_np_gen,
        color="gray",
        linestyle="--",
        label=f"Neuronpedia ({overall_np_gen:.2f})",
    )
    ax.axhline(
        summary["OVERALL"]["ss_gen"]["mean"],
        color="orange",
        linestyle="--",
        label=f"SingleShot ({summary['OVERALL']['ss_gen']['mean']:.2f})",
    )
    ax.set_xlabel("SAGE iteration")
    ax.set_ylabel("Generative accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("SAGE Iterative Refinement (aggregated)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "sage_iteration_curve.png"), dpi=120)
    plt.close()

except Exception as e:
    print(f"Plot error: {e}")
    traceback.print_exc()

# ---------- Save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)


def _safe(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


with open(os.path.join(working_dir, "experiment_data.json"), "w") as f:
    json.dump(experiment_data, f, indent=2, default=_safe)
json.dump(llm_cache, open(LLM_CACHE_PATH, "w"))

for h in hooks:
    h.remove()
print(f"\nDone. generative_activation_success_rate = {overall_gen:.4f}")
print(f"Predictive accuracy (SAGE) = {overall_pred:.4f}")
print(f"Neuronpedia gen = {overall_np_gen:.4f} pred = {overall_np_pred:.4f}")
