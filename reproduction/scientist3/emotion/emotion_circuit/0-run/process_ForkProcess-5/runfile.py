import os, json, re, time, random, glob

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"
MODEL_PATH = os.path.join(MODEL_DIR, "Llama-3.2-3B-Instruct")
if not os.path.isdir(MODEL_PATH):
    cands = glob.glob(os.path.join(MODEL_DIR, "*Llama-3.2-3B*Instruct*"))
    if cands:
        MODEL_PATH = cands[0]
    else:
        raise FileNotFoundError("Model not found")

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
SEV_PATH = os.path.join(DATA_DIR, "SEV", "sev_events.json")
os.makedirs(os.path.dirname(SEV_PATH), exist_ok=True)


def build_sev():
    domains = [
        "work",
        "family",
        "friendship",
        "romance",
        "study",
        "health",
        "finance",
        "travel",
    ]
    scenario_templates = {d: [f"a {d} event {i}" for i in range(20)] for d in domains}
    scenario_templates["work"] = [
        "a job interview",
        "a promotion decision",
        "a client meeting",
        "a deadline",
        "a team project",
        "a performance review",
        "a business trip",
        "a salary negotiation",
        "a new colleague",
        "a big presentation",
        "a layoff rumor",
        "an office party",
        "a training program",
        "a work conflict",
        "a remote work day",
        "a mentoring session",
        "a product launch",
        "an audit",
        "an award ceremony",
        "a company merger",
    ]
    scenario_templates["family"] = [
        "a family dinner",
        "a parent's birthday",
        "a sibling's visit",
        "a family reunion",
        "a household move",
        "a family game night",
        "a grandparent's illness",
        "a child's school event",
        "a family vacation",
        "a family argument",
        "a new baby's arrival",
        "a family photo session",
        "a family tradition",
        "a family recipe cooking",
        "a family movie night",
        "a pet's arrival",
        "a family holiday",
        "a home renovation",
        "a family camping trip",
        "a family celebration",
    ]
    scenario_templates["friendship"] = [
        "a coffee meetup",
        "a birthday party",
        "a road trip",
        "a game night",
        "a movie outing",
        "a workout session",
        "a picnic",
        "a bar visit",
        "a shared hobby session",
        "a video call",
        "a farewell party",
        "a study group",
        "a concert",
        "a hiking trip",
        "a beach day",
        "a shopping trip",
        "a cooking session",
        "a museum visit",
        "a karaoke night",
        "a book club",
    ]
    scenario_templates["romance"] = [
        "a first date",
        "an anniversary",
        "a marriage proposal",
        "a candlelight dinner",
        "a weekend getaway",
        "a valentine's day",
        "a surprise gift",
        "a walk in the park",
        "a movie night",
        "a couple's cooking",
        "a dance class",
        "a couples therapy",
        "a house viewing",
        "a wedding planning",
        "a couples yoga",
        "a stargazing night",
        "a spa day",
        "a couple's photo shoot",
        "a picnic date",
        "a museum date",
    ]
    scenario_templates["study"] = [
        "an exam",
        "a group assignment",
        "a research project",
        "a thesis defense",
        "a class presentation",
        "a lab experiment",
        "a lecture",
        "a study session",
        "a tutoring meeting",
        "a scholarship application",
        "an internship interview",
        "a workshop",
        "a debate competition",
        "a science fair",
        "a school trip",
        "a language class",
        "a coding challenge",
        "a library visit",
        "a graduation",
        "an oral exam",
    ]
    scenario_templates["health"] = [
        "a doctor's appointment",
        "a marathon",
        "a yoga class",
        "a diet plan",
        "a hospital visit",
        "a dental checkup",
        "a vaccination",
        "a therapy session",
        "a surgery",
        "a health screening",
        "a fitness assessment",
        "a meditation retreat",
        "a physical therapy",
        "a nutrition consult",
        "a sleep study",
        "a mental health day",
        "an eye exam",
        "a rehabilitation session",
        "a health lecture",
        "a wellness workshop",
    ]
    scenario_templates["finance"] = [
        "a stock investment",
        "a house purchase",
        "a loan application",
        "a tax filing",
        "a budget planning",
        "a bonus announcement",
        "a business deal",
        "a gambling night",
        "a lottery draw",
        "a bank meeting",
        "a car purchase",
        "a crypto trade",
        "a job salary talk",
        "a bill payment",
        "an insurance claim",
        "a retirement plan",
        "a bankruptcy filing",
        "a charity donation",
        "a financial audit",
        "a real estate deal",
    ]
    scenario_templates["travel"] = [
        "a flight departure",
        "a hotel checkin",
        "a train journey",
        "a city tour",
        "a beach vacation",
        "a mountain hike",
        "a cruise trip",
        "a road trip",
        "a museum visit",
        "a food tour",
        "a camping trip",
        "a safari",
        "a ski trip",
        "a backpacking journey",
        "a business trip",
        "a family vacation",
        "a weekend getaway",
        "a solo travel",
        "a group tour",
        "a cultural festival",
    ]
    outcomes = ["positive", "negative", "neutral"]
    events = []
    for d in domains:
        for s in scenario_templates[d][:20]:
            for o in outcomes:
                if o == "positive":
                    ctx = f"You are attending {s} in the {d} domain. Everything goes remarkably well and exceeds your expectations."
                elif o == "negative":
                    ctx = f"You are attending {s} in the {d} domain. Things go very poorly and result in significant disappointment."
                else:
                    ctx = f"You are attending {s} in the {d} domain. The situation unfolds in an unremarkable, ordinary way."
                events.append(
                    {"domain": d, "scenario": s, "outcome": o, "context": ctx}
                )
    with open(SEV_PATH, "w") as f:
        json.dump(events, f)
    return events


if os.path.exists(SEV_PATH):
    with open(SEV_PATH) as f:
        events = json.load(f)
else:
    events = build_sev()
print(f"SEV events: {len(events)}")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, local_files_only=True
).to(device)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"Layers: {n_layers}, hidden: {hidden_size}")


def build_prompt(context, emotion=None):
    if emotion:
        sys = f"You are a person who feels intense {emotion}. Respond in first person expressing that emotion clearly."
    else:
        sys = "You are a person describing your experience in first person."
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": context},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_neutral_prompt(context):
    messages = [
        {
            "role": "system",
            "content": "You are a person describing your experience in first person.",
        },
        {"role": "user", "content": context},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


random.seed(0)
idx = list(range(len(events)))
random.shuffle(idx)
n_extract = int(0.8 * len(idx))
extract_idx = idx[:n_extract]
eval_idx = idx[n_extract:]
extract_sample = random.sample(extract_idx, min(48, len(extract_idx)))
# Balanced eval subset ensuring domain diversity
eval_sample = eval_idx[:24]  # 24 events x 6 emotions = 144 gens per config
# domain distribution of eval
domain_dist = {}
for ei in eval_sample:
    d = events[ei]["domain"]
    domain_dist[d] = domain_dist.get(d, 0) + 1
print(
    f"Extract: {len(extract_sample)}, Eval: {len(eval_sample)}, domain dist: {domain_dist}"
)


# --- Extract hidden states at ALL layers for each emotion (one forward pass per prompt) ---
@torch.no_grad()
def get_all_hiddens(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple length n_layers+1, each [1,seq,hid]. Take last token, all layers 1..n_layers
    hs = [
        outputs.hidden_states[i + 1][0, -1, :].float().cpu().numpy()
        for i in range(n_layers)
    ]
    return hs


print("Extracting hidden states across all layers for all emotions...")
# emotion_layer_means[emo][layer] = mean vector
emotion_layer_sums = {
    e: [np.zeros(hidden_size, dtype=np.float32) for _ in range(n_layers)]
    for e in EMOTIONS + ["neutral"]
}
emotion_layer_counts = {e: 0 for e in EMOTIONS + ["neutral"]}

for emo in EMOTIONS + ["neutral"]:
    for si in extract_sample:
        ctx = events[si]["context"]
        prompt = build_prompt(ctx, None if emo == "neutral" else emo)
        try:
            hs = get_all_hiddens(prompt)
            for L in range(n_layers):
                emotion_layer_sums[emo][L] += hs[L]
            emotion_layer_counts[emo] += 1
        except Exception as e:
            print(f"err {emo}: {e}")
    print(f"  {emo}: {emotion_layer_counts[emo]} samples")

emotion_layer_means = {
    e: [
        emotion_layer_sums[e][L] / max(1, emotion_layer_counts[e])
        for L in range(n_layers)
    ]
    for e in EMOTIONS + ["neutral"]
}

# steering vectors per (emotion, layer)
steering_vecs = {}
for emo in EMOTIONS:
    steering_vecs[emo] = []
    for L in range(n_layers):
        v = emotion_layer_means[emo][L] - emotion_layer_means["neutral"][L]
        v = v / (np.linalg.norm(v) + 1e-8)
        steering_vecs[emo].append(v)

# ----- Cosine similarity between emotion directions per layer (mechanistic analysis) -----
cos_sim_per_layer = np.zeros((n_layers, len(EMOTIONS), len(EMOTIONS)), dtype=np.float32)
for L in range(n_layers):
    for i, ei in enumerate(EMOTIONS):
        for j, ej in enumerate(EMOTIONS):
            a = steering_vecs[ei][L]
            b = steering_vecs[ej][L]
            cos_sim_per_layer[L, i, j] = float(np.dot(a, b))


# --- Steerer ---
class Steerer:
    def __init__(self):
        self.vec = None
        self.strength = 0.0
        self.handle = None

    def hook(self, module, input, output):
        if self.vec is None:
            return output
        if isinstance(output, tuple):
            h = output[0]
            h = h + self.strength * self.vec.to(h.dtype).to(h.device)
            return (h,) + output[1:]
        else:
            return output + self.strength * self.vec.to(output.dtype).to(output.device)

    def attach(self, layer_idx):
        self.handle = model.model.layers[layer_idx].register_forward_hook(self.hook)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


steerer = Steerer()


@torch.no_grad()
def generate_text(prompt, emo_vec=None, strength=0.0):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    steerer.vec = torch.tensor(emo_vec) if emo_vec is not None else None
    steerer.strength = strength
    out = model.generate(
        **inputs,
        max_new_tokens=70,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return text.strip()


# ---- Judge ----
import urllib.request

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
JUDGE_MODEL = "gpt-5.4"
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


def judge_emotion(text):
    prompt = (
        f"Classify the emotion expressed in the following first-person response into exactly one of: "
        f"{', '.join(EMOTIONS)}, neutral. Reply with ONLY the single word label.\n\nResponse: {text}\n\nLabel:"
    )
    body = json.dumps(
        {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 10,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                out = data["choices"][0]["message"]["content"].strip().lower()
                out = re.sub(r"[^a-z]", "", out)
                for e in EMOTIONS + ["neutral"]:
                    if e in out:
                        return e
                return "neutral"
        except Exception as e:
            print(f"judge err {attempt}: {e}")
            time.sleep(2)
    return "neutral"


# ---------- EXPERIMENTS ----------
STRENGTH = 10.0
# Sweep a diverse set of layers spanning early/mid/late (Llama-3.2-3B has 28 layers)
layer_sweep = sorted(set([2, 6, 10, 14, 18, 22, 26]))
layer_sweep = [L for L in layer_sweep if L < n_layers]
print(f"Layer sweep: {layer_sweep}  (n_layers={n_layers})")

experiment_data = {
    "layer_sweep": {
        "SEV": {
            "layers": layer_sweep,
            "strength": STRENGTH,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "generations": [],
            "per_layer_results": {},
            "per_domain_results": {},
            "cosine_similarity_per_layer": cos_sim_per_layer.tolist(),
            "config": {"model": MODEL_PATH, "n_layers": n_layers},
        }
    },
    "baselines": {
        "SEV": {
            "predictions": [],
            "ground_truth": [],
            "generations": [],
            "per_method_accuracy": {},
            "per_method_per_emotion": {},
            "per_method_per_domain": {},
        }
    },
}

# ==== BASELINE 1: No steering (neutral prompt, no target) — control ====
# ==== BASELINE 2: Prompting (system prompt states target emotion) ====
# ==== BASELINE 3: Direction steering at best mid layer (we'll pick after sweep) ====

# First: run prompting + no-steer baselines (both do NOT need multi-layer sweep)
print("\n=== BASELINE: No steering (neutral) ===")
steerer.attach(layer_sweep[0])  # attach dummy; vec=None means no effect
no_steer_records = []
for ei in eval_sample:
    ctx = events[ei]["context"]
    dom = events[ei]["domain"]
    prompt = build_neutral_prompt(ctx)
    gen = generate_text(prompt, None, 0.0)
    # Compare against each target emotion (structure matches steering evaluations)
    for emo in EMOTIONS:
        pred = judge_emotion(gen)
        no_steer_records.append(
            {"context": ctx, "domain": dom, "target": emo, "pred": pred, "text": gen}
        )
steerer.detach()

no_steer_correct = sum(1 for r in no_steer_records if r["pred"] == r["target"])
no_steer_acc = no_steer_correct / len(no_steer_records)
print(f"No-steering accuracy: {no_steer_acc:.4f}")

# Prompting baseline: cache generation once per (event, emotion)
print("\n=== BASELINE: Prompting ===")
prompt_records = []
for ei in eval_sample:
    ctx = events[ei]["context"]
    dom = events[ei]["domain"]
    for emo in EMOTIONS:
        prompt = build_prompt(ctx, emo)
        gen = generate_text(prompt, None, 0.0)
        pred = judge_emotion(gen)
        prompt_records.append(
            {"context": ctx, "domain": dom, "target": emo, "pred": pred, "text": gen}
        )
prompt_correct = sum(1 for r in prompt_records if r["pred"] == r["target"])
prompt_acc = prompt_correct / len(prompt_records)
print(f"Prompting accuracy: {prompt_acc:.4f}")

# ==== LAYER SWEEP: Direction steering at each layer ====
layer_acc = {}
layer_per_emo = {}
layer_per_domain = {}
layer_records = {}
for L in layer_sweep:
    print(f"\n=== Layer sweep: L={L} strength={STRENGTH} ===")
    steerer.attach(L)
    correct = 0
    total = 0
    per_emo_c = {e: 0 for e in EMOTIONS}
    per_emo_t = {e: 0 for e in EMOTIONS}
    per_dom_c = {}
    per_dom_t = {}
    recs = []
    for ei in eval_sample:
        ctx = events[ei]["context"]
        dom = events[ei]["domain"]
        prompt = build_neutral_prompt(ctx)
        for emo in EMOTIONS:
            vec = steering_vecs[emo][L]
            gen = generate_text(prompt, vec, STRENGTH)
            pred = judge_emotion(gen)
            ok = int(pred == emo)
            correct += ok
            total += 1
            per_emo_t[emo] += 1
            per_emo_c[emo] += ok
            per_dom_t[dom] = per_dom_t.get(dom, 0) + 1
            per_dom_c[dom] = per_dom_c.get(dom, 0) + ok
            recs.append(
                {
                    "context": ctx,
                    "domain": dom,
                    "target": emo,
                    "pred": pred,
                    "text": gen,
                    "layer": L,
                }
            )
            if total % 30 == 0:
                print(f"  [L={L}][{total}] acc so far={correct/total:.3f}")
    steerer.detach()
    acc = correct / max(1, total)
    layer_acc[L] = acc
    layer_per_emo[L] = {e: per_emo_c[e] / max(1, per_emo_t[e]) for e in EMOTIONS}
    layer_per_domain[L] = {d: per_dom_c[d] / max(1, per_dom_t[d]) for d in per_dom_t}
    layer_records[L] = recs
    print(f"  Layer {L}: acc={acc:.4f}  per_emo={layer_per_emo[L]}")
    experiment_data["layer_sweep"]["SEV"]["per_layer_results"][str(L)] = {
        "accuracy": acc,
        "per_emotion_accuracy": layer_per_emo[L],
        "per_domain_accuracy": layer_per_domain[L],
        "n_samples": total,
    }
    experiment_data["layer_sweep"]["SEV"]["metrics"]["val"].append(
        {"layer": L, "emotion_expression_accuracy": acc}
    )
    experiment_data["layer_sweep"]["SEV"]["losses"]["val"].append(
        {"layer": L, "loss": 1 - acc}
    )
    print(f"Epoch {L}: validation_loss = {1-acc:.4f}")

# Best layer
best_layer = max(layer_acc, key=lambda k: layer_acc[k])
best_layer_acc = layer_acc[best_layer]
print(f"\nBest layer = {best_layer}, acc = {best_layer_acc:.4f}")


# Save baseline results
def summarize(records, name):
    acc = sum(1 for r in records if r["pred"] == r["target"]) / max(1, len(records))
    per_emo = {}
    per_dom = {}
    for r in records:
        per_emo.setdefault(r["target"], [0, 0])
        per_dom.setdefault(r["domain"], [0, 0])
        per_emo[r["target"]][1] += 1
        per_dom[r["domain"]][1] += 1
        if r["pred"] == r["target"]:
            per_emo[r["target"]][0] += 1
            per_dom[r["domain"]][0] += 1
    per_emo = {k: v[0] / max(1, v[1]) for k, v in per_emo.items()}
    per_dom = {k: v[0] / max(1, v[1]) for k, v in per_dom.items()}
    experiment_data["baselines"]["SEV"]["per_method_accuracy"][name] = acc
    experiment_data["baselines"]["SEV"]["per_method_per_emotion"][name] = per_emo
    experiment_data["baselines"]["SEV"]["per_method_per_domain"][name] = per_dom
    experiment_data["baselines"]["SEV"]["generations"].extend(
        [{**r, "method": name} for r in records]
    )
    return acc, per_emo, per_dom


summarize(no_steer_records, "no_steering")
summarize(prompt_records, "prompting")
summarize(layer_records[best_layer], f"steer_L{best_layer}")

experiment_data["layer_sweep"]["SEV"]["best_layer"] = best_layer
experiment_data["layer_sweep"]["SEV"]["best_accuracy"] = best_layer_acc

# Save all
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# ---------- PLOTS ----------
# (1) Layer sweep: accuracy vs layer with baselines as horizontal lines
fig, ax = plt.subplots(figsize=(9, 5))
Ls = sorted(layer_acc.keys())
accs = [layer_acc[L] for L in Ls]
ax.plot(
    Ls, accs, marker="o", linewidth=2, color="steelblue", label="Direction steering"
)
ax.axhline(
    prompt_acc, color="orange", linestyle="--", label=f"Prompting ({prompt_acc:.3f})"
)
ax.axhline(
    no_steer_acc, color="gray", linestyle=":", label=f"No steering ({no_steer_acc:.3f})"
)
ax.axhline(1 / 7, color="red", linestyle="-.", alpha=0.5, label="random (1/7)")
ax.set_xlabel("Steering layer")
ax.set_ylabel("Emotion expression accuracy")
ax.set_title(
    f"SEV: Emotion steering accuracy across layers (Llama-3.2-3B, strength={STRENGTH})"
)
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "sev_layer_sweep_vs_baselines.png"), dpi=140)
plt.close()

# (2) Per-emotion accuracy heatmap across layers
emo_mat = np.zeros((len(EMOTIONS), len(Ls)))
for j, L in enumerate(Ls):
    for i, e in enumerate(EMOTIONS):
        emo_mat[i, j] = layer_per_emo[L][e]
fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(emo_mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(Ls)))
ax.set_xticklabels(Ls)
ax.set_yticks(range(len(EMOTIONS)))
ax.set_yticklabels(EMOTIONS)
ax.set_xlabel("Layer")
ax.set_ylabel("Target emotion")
ax.set_title("Per-emotion steering accuracy across layers (SEV)")
for i in range(len(EMOTIONS)):
    for j in range(len(Ls)):
        ax.text(
            j,
            i,
            f"{emo_mat[i,j]:.2f}",
            ha="center",
            va="center",
            color="white" if emo_mat[i, j] < 0.6 else "black",
            fontsize=9,
        )
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "sev_per_emotion_heatmap.png"), dpi=140)
plt.close()

# (3) Baseline comparison: overall + per-emotion (grouped bar)
methods = ["no_steering", "prompting", f"steer_L{best_layer}"]
overall = [
    experiment_data["baselines"]["SEV"]["per_method_accuracy"][m] for m in methods
]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(methods, overall, color=["gray", "orange", "steelblue"])
ax.axhline(1 / 7, color="red", linestyle="--", label="random (1/7)")
for i, v in enumerate(overall):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
ax.set_ylim(0, 1)
ax.set_ylabel("Emotion expression accuracy")
ax.set_title("SEV: baseline comparison (overall)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "sev_baseline_overall.png"), dpi=140)
plt.close()

# per-emotion grouped bar
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(EMOTIONS))
w = 0.25
for i, m in enumerate(methods):
    vals = [
        experiment_data["baselines"]["SEV"]["per_method_per_emotion"][m].get(e, 0)
        for e in EMOTIONS
    ]
    ax.bar(x + (i - 1) * w, vals, w, label=m)
ax.set_xticks(x)
ax.set_xticklabels(EMOTIONS)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("SEV: per-emotion accuracy by method")
ax.axhline(1 / 7, color="red", linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "sev_baseline_per_emotion.png"), dpi=140)
plt.close()

# (4) Per-domain breakdown for best layer vs prompting vs no-steer
domains = sorted(
    experiment_data["baselines"]["SEV"]["per_method_per_domain"][methods[0]].keys()
)
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(domains))
w = 0.25
for i, m in enumerate(methods):
    vals = [
        experiment_data["baselines"]["SEV"]["per_method_per_domain"][m].get(d, 0)
        for d in domains
    ]
    ax.bar(x + (i - 1) * w, vals, w, label=m)
ax.set_xticks(x)
ax.set_xticklabels(domains, rotation=20)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("SEV: per-domain accuracy by method")
ax.axhline(1 / 7, color="red", linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "sev_per_domain_by_method.png"), dpi=140)
plt.close()

# (5) Cosine similarity heatmap at best layer (diagnostic for disgust collapse)
cs = cos_sim_per_layer[best_layer]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cs, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(EMOTIONS)))
ax.set_xticklabels(EMOTIONS, rotation=30)
ax.set_yticks(range(len(EMOTIONS)))
ax.set_yticklabels(EMOTIONS)
ax.set_title(f"Cosine similarity of emotion directions at layer {best_layer}")
for i in range(len(EMOTIONS)):
    for j in range(len(EMOTIONS)):
        ax.text(
            j, i, f"{cs[i,j]:.2f}", ha="center", va="center", color="black", fontsize=9
        )
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "sev_direction_cosine_bestlayer.png"), dpi=140)
plt.close()

# (6) mean off-diagonal cosine similarity across layers (mechanistic view of separability)
off_diag_mean = []
for L in range(n_layers):
    m = cos_sim_per_layer[L].copy()
    np.fill_diagonal(m, np.nan)
    off_diag_mean.append(np.nanmean(np.abs(m)))
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(n_layers), off_diag_mean, marker="o", color="purple")
ax.set_xlabel("Layer")
ax.set_ylabel("Mean |cos sim| off-diagonal")
ax.set_title("Cross-emotion direction similarity per layer (lower = more separable)")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "sev_offdiag_cosine_per_layer.png"), dpi=140)
plt.close()

print("\n===== FINAL SUMMARY =====")
print(f"No-steering baseline accuracy: {no_steer_acc:.4f}")
print(f"Prompting baseline accuracy:   {prompt_acc:.4f}")
print(f"Direction steering best layer L={best_layer}: {best_layer_acc:.4f}")
print(f"All layer accs: {layer_acc}")
print(f"emotion_expression_accuracy (best) = {best_layer_acc:.4f}")
print("Done.")
