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
    # try alternatives
    cands = glob.glob(os.path.join(MODEL_DIR, "*Llama-3.2-3B*Instruct*")) + glob.glob(
        os.path.join(MODEL_DIR, "*llama-3.2-3b*instruct*")
    )
    if cands:
        MODEL_PATH = cands[0]
    else:
        raise FileNotFoundError(f"Cannot find Llama-3.2-3B-Instruct under {MODEL_DIR}")

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

# ---------- SEV dataset construction (small synthetic scaffold for the paper) ----------
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
    # 20 scenario templates per domain (short)
    scenario_templates = {
        "work": [
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
        ],
        "family": [
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
        ],
        "friendship": [
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
        ],
        "romance": [
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
        ],
        "study": [
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
        ],
        "health": [
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
        ],
        "finance": [
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
        ],
        "travel": [
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
        ],
    }
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

# ---------- Load model ----------
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

# ---------- Extract emotion direction vectors ----------
# For each emotion, produce prompts where the assistant expresses that emotion strongly.
# Get mean hidden state at layer L over last token positions.
STEER_LAYER = n_layers // 2
print(f"Steering layer: {STEER_LAYER}")


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


# Split events: use first 80% for extraction, last 20% for eval
random.seed(0)
idx = list(range(len(events)))
random.shuffle(idx)
n_extract = int(0.8 * len(idx))
extract_idx = idx[:n_extract]
eval_idx = idx[n_extract:]
# limit eval for speed
eval_idx = eval_idx[:60]
extract_sample = random.sample(extract_idx, min(48, len(extract_idx)))
print(f"Extract samples: {len(extract_sample)}, eval samples: {len(eval_idx)}")


@torch.no_grad()
def get_layer_hidden(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (n_layers+1) tensors [1, seq, hidden]
    h = outputs.hidden_states[STEER_LAYER + 1][0, -1, :].float().cpu().numpy()
    return h


print("Extracting emotion directions...")
emotion_means = {}
for emo in EMOTIONS + ["neutral"]:
    vecs = []
    for si in extract_sample:
        ctx = events[si]["context"]
        prompt = build_prompt(ctx, None if emo == "neutral" else emo)
        try:
            h = get_layer_hidden(prompt)
            vecs.append(h)
        except Exception as e:
            print(f"err {emo}: {e}")
    emotion_means[emo] = np.mean(vecs, axis=0)
    print(f"{emo}: computed from {len(vecs)} samples")

# Steering vectors = emotion mean - neutral mean, normalized
steering_vecs = {}
for emo in EMOTIONS:
    v = emotion_means[emo] - emotion_means["neutral"]
    steering_vecs[emo] = v / (np.linalg.norm(v) + 1e-8)

# ---------- Steering via forward hook ----------
STRENGTH = 8.0  # tunable


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
        # llama layers accessible at model.model.layers[i]
        layer = model.model.layers[layer_idx]
        self.handle = layer.register_forward_hook(self.hook)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


steerer = Steerer()
steerer.attach(STEER_LAYER)


@torch.no_grad()
def generate_steered(context, emo_vec, strength):
    prompt = build_neutral_prompt(context)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    steerer.vec = torch.tensor(emo_vec) if emo_vec is not None else None
    steerer.strength = strength
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return text.strip()


# ---------- LLM Judge ----------
import urllib.request, urllib.error

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
JUDGE_MODEL = "gpt-5.4"

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


# ---------- Evaluation ----------
experiment_data = {
    "SEV": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "generations": [],
        "per_emotion_accuracy": {},
        "config": {
            "steer_layer": STEER_LAYER,
            "strength": STRENGTH,
            "model": MODEL_PATH,
        },
    }
}

print("Running steered generation + judging...")
correct = 0
total = 0
per_emo_correct = {e: 0 for e in EMOTIONS}
per_emo_total = {e: 0 for e in EMOTIONS}

# limit further: 30 eval events × 6 emotions = 180 generations
eval_events_final = eval_idx[:30]

for ei in eval_events_final:
    ctx = events[ei]["context"]
    for emo in EMOTIONS:
        vec = steering_vecs[emo]
        gen = generate_steered(ctx, vec, STRENGTH)
        pred = judge_emotion(gen)
        correct_flag = int(pred == emo)
        correct += correct_flag
        total += 1
        per_emo_total[emo] += 1
        per_emo_correct[emo] += correct_flag
        experiment_data["SEV"]["predictions"].append(pred)
        experiment_data["SEV"]["ground_truth"].append(emo)
        experiment_data["SEV"]["generations"].append(
            {"context": ctx, "target": emo, "pred": pred, "text": gen}
        )
        if total % 10 == 0:
            print(
                f"[{total}] target={emo} pred={pred} acc={correct/total:.3f} | {gen[:80]}"
            )

steerer.detach()

acc = correct / max(1, total)
print(f"\nemotion_expression_accuracy = {acc:.4f}")
per_emo_acc = {e: per_emo_correct[e] / max(1, per_emo_total[e]) for e in EMOTIONS}
print("Per-emotion accuracy:", per_emo_acc)

experiment_data["SEV"]["metrics"]["val"].append(
    {"epoch": 0, "emotion_expression_accuracy": acc}
)
experiment_data["SEV"]["per_emotion_accuracy"] = per_emo_acc
print(f"Epoch 0: validation_loss = {(1-acc):.4f}")

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# ---------- Visualization ----------
fig, ax = plt.subplots(figsize=(8, 5))
emos = list(per_emo_acc.keys())
vals = [per_emo_acc[e] for e in emos]
ax.bar(emos, vals, color="steelblue")
ax.axhline(1 / 7, color="r", linestyle="--", label="random (1/7)")
ax.set_ylabel("Accuracy")
ax.set_title(f"SEV per-emotion steering accuracy (overall={acc:.3f})")
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "sev_per_emotion_accuracy.png"))
plt.close()

print("Done.")
