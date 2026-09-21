import os, json, time, random, math, gc, re, traceback
import numpy as np
import torch
from torch import nn

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
JUDGE_MODEL = "gpt-4o-2024-11-20"
GEN_MODEL = "gpt-4o-2024-11-20"

# Bypass proxy for API
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

experiment_data = {
    "concept_benchmark": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "per_concept": [],
        "overall_steering_success_rate": None,
    }
}

# ---------- API helper ----------
try:
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def chat(model, messages, max_tokens=400, temperature=0.7, retries=3):
        for i in range(retries):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return r.choices[0].message.content
            except Exception as e:
                print(f"API err ({i}): {e}")
                time.sleep(2 + i * 2)
        return None

except Exception as e:
    print("openai import failed:", e)

    def chat(*a, **k):
        return None


# ---------- Build concept benchmark ----------
CONCEPT_CLASSES = {
    "fears": [
        "fear of heights",
        "fear of spiders",
        "fear of public speaking",
        "fear of the dark",
    ],
    "experts": [
        "expert chef",
        "expert astronomer",
        "expert lawyer",
        "expert software engineer",
    ],
    "moods": ["joyful mood", "melancholic mood", "angry mood", "anxious mood"],
    "topophiles": [
        "love of mountains",
        "love of oceans",
        "love of deserts",
        "love of forests",
    ],
    "personas": [
        "pirate persona",
        "shakespearean persona",
        "scientist persona",
        "cowboy persona",
    ],
}

CONCEPTS = []
for cls, items in CONCEPT_CLASSES.items():
    for c in items:
        CONCEPTS.append({"class": cls, "concept": c})
print(f"Total concepts: {len(CONCEPTS)}")

GENERIC_PROMPTS = [
    "Tell me about your weekend.",
    "What's your favorite kind of food?",
    "Describe a walk in the park.",
    "How do you feel today?",
    "What are your plans for tomorrow?",
    "Tell me a short story.",
    "Describe your ideal day.",
    "What do you think about the weather?",
    "Share a random thought.",
    "Give me some advice about life.",
    "What is something interesting you learned recently?",
    "Describe a room in a house.",
]


def make_positive_prompts(concept, n=8):
    # Prompts where the assistant is expected to embody / discuss the concept.
    base = [
        f"Respond as someone who strongly exhibits {concept}. Describe your morning.",
        f"You have a strong {concept}. Tell me about your weekend.",
        f"Speaking with a clear {concept}, describe a walk in the park.",
        f"Reflect a {concept} in your answer: what's your favorite food?",
        f"With {concept}, share your plans for tomorrow.",
        f"Given your {concept}, give me advice about life.",
        f"As a person defined by {concept}, describe your ideal day.",
        f"Channeling {concept}, tell me a short story.",
    ]
    return base[:n]


def make_negative_prompts(n=8):
    return GENERIC_PROMPTS[:n]


# ---------- Load Llama-3.1-8B-Instruct ----------
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.join(MODEL_DIR, "Llama-3.1-8B-Instruct")
if not os.path.isdir(MODEL_PATH):
    # try alternate name
    alt = os.path.join(MODEL_DIR, "Meta-Llama-3.1-8B-Instruct")
    if os.path.isdir(alt):
        MODEL_PATH = alt
    else:
        raise FileNotFoundError(f"Llama-3.1-8B-Instruct not found under {MODEL_DIR}")

print(f"Loading model from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()
n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"n_layers={n_layers}, hidden_size={hidden_size}")

STEER_LAYER = n_layers // 2  # middle layer
print(f"Using steering layer: {STEER_LAYER}")


# Access the decoder layer module for hooks
def get_layer_module(idx):
    return model.model.layers[idx]


# ---------- Activation extraction ----------
def format_chat(user_msg):
    msgs = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def get_last_token_hidden(prompts, layer_idx):
    """Return tensor [N, hidden] of hidden states at layer_idx output for the last input token."""
    outs = []
    for p in prompts:
        text = format_chat(p)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            o = model(**inputs, output_hidden_states=True, use_cache=False)
        # hidden_states: tuple of (n_layers+1) tensors [1, T, H]; hidden_states[layer_idx+1] is output of layer_idx
        h = o.hidden_states[layer_idx + 1][0, -1, :].float().cpu()
        outs.append(h)
        del o
    return torch.stack(outs, dim=0)


# ---------- Compute concept vectors (diff-of-means, supervised linear direction) ----------
concept_vectors = {}
print("Computing concept vectors...")
for i, entry in enumerate(CONCEPTS):
    concept = entry["concept"]
    pos_prompts = make_positive_prompts(concept, n=8)
    neg_prompts = make_negative_prompts(n=8)
    pos_h = get_last_token_hidden(pos_prompts, STEER_LAYER)
    neg_h = get_last_token_hidden(neg_prompts, STEER_LAYER)
    v = pos_h.mean(0) - neg_h.mean(0)
    v = v / (v.norm() + 1e-8)
    concept_vectors[concept] = v
    print(
        f"[{i+1}/{len(CONCEPTS)}] vector for '{concept}' norm(pre-normalize)={(pos_h.mean(0)-neg_h.mean(0)).norm().item():.3f}"
    )


# ---------- Steering hook ----------
class SteerHook:
    def __init__(self):
        self.vector = None
        self.alpha = 0.0

    def __call__(self, module, inp, out):
        if self.vector is None or self.alpha == 0.0:
            return out
        if isinstance(out, tuple):
            h = out[0]
            h = h + self.alpha * self.vector.to(h.device, dtype=h.dtype).view(1, 1, -1)
            return (h,) + out[1:]
        else:
            return out + self.alpha * self.vector.to(out.device, dtype=out.dtype).view(
                1, 1, -1
            )


steer_hook = SteerHook()
hook_handle = get_layer_module(STEER_LAYER).register_forward_hook(steer_hook)


@torch.no_grad()
def generate(prompt, max_new_tokens=80):
    text = format_chat(prompt)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


# ---------- Judge with GPT-4o ----------
def judge_concept(concept, response):
    if response is None or len(response.strip()) == 0:
        return 0, "empty"
    sys = (
        "You are an evaluator. Judge whether the assistant's response clearly reflects, embodies, "
        "or expresses the given target concept. Respond with a JSON object like "
        '{"reflects": true} or {"reflects": false} and a brief reason.'
    )
    user = f"Target concept: {concept}\n\nAssistant response:\n{response}\n\nDoes the response clearly reflect the target concept?"
    out = chat(
        JUDGE_MODEL,
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=100,
        temperature=0.0,
    )
    if out is None:
        return 0, "judge_failed"
    m = re.search(r'"reflects"\s*:\s*(true|false)', out.lower())
    if m:
        return (1 if m.group(1) == "true" else 0), out
    # fallback simple check
    if "true" in out.lower():
        return 1, out
    return 0, out


# ---------- Steering evaluation ----------
ALPHA = 8.0  # steering strength (residual-stream addition scale)
N_EVAL_PROMPTS = 3  # generic prompts per concept

per_concept_records = []
succ_total, tot_total = 0, 0
baseline_succ_total, baseline_tot_total = 0, 0

eval_prompts = GENERIC_PROMPTS[:N_EVAL_PROMPTS]

for ci, entry in enumerate(CONCEPTS):
    concept = entry["concept"]
    cls = entry["class"]
    vec = concept_vectors[concept]

    # Baseline (no steering)
    steer_hook.vector = None
    steer_hook.alpha = 0.0
    baseline_gens = []
    for p in eval_prompts:
        try:
            r = generate(p, max_new_tokens=80)
        except Exception as e:
            r = ""
            print("gen err:", e)
        baseline_gens.append(r)

    # Steered
    steer_hook.vector = vec
    steer_hook.alpha = ALPHA
    steered_gens = []
    for p in eval_prompts:
        try:
            r = generate(p, max_new_tokens=80)
        except Exception as e:
            r = ""
            print("gen err:", e)
        steered_gens.append(r)
    steer_hook.vector = None
    steer_hook.alpha = 0.0

    # Judge
    b_succ = 0
    s_succ = 0
    details = []
    for p, br, sr in zip(eval_prompts, baseline_gens, steered_gens):
        bj, br_reason = judge_concept(concept, br)
        sj, sr_reason = judge_concept(concept, sr)
        b_succ += bj
        s_succ += sj
        details.append(
            {
                "prompt": p,
                "baseline": br,
                "steered": sr,
                "baseline_judge": bj,
                "steered_judge": sj,
            }
        )

    tot = len(eval_prompts)
    baseline_succ_total += b_succ
    baseline_tot_total += tot
    succ_total += s_succ
    tot_total += tot

    rec = {
        "class": cls,
        "concept": concept,
        "baseline_success": b_succ,
        "steered_success": s_succ,
        "total": tot,
        "details": details,
    }
    per_concept_records.append(rec)
    print(
        f"[{ci+1}/{len(CONCEPTS)}] {concept}: baseline {b_succ}/{tot}, steered {s_succ}/{tot}"
    )

    # intermediate save
    experiment_data["concept_benchmark"]["per_concept"] = per_concept_records
    experiment_data["concept_benchmark"]["metrics"]["val"].append(
        {
            "concept_index": ci,
            "concept": concept,
            "steered_success_rate_running": succ_total / max(1, tot_total),
            "baseline_success_rate_running": baseline_succ_total
            / max(1, baseline_tot_total),
        }
    )
    print(
        f"  running steered success rate = {succ_total/max(1,tot_total):.3f}, baseline = {baseline_succ_total/max(1,baseline_tot_total):.3f}"
    )

hook_handle.remove()

steering_success_rate = succ_total / max(1, tot_total)
baseline_success_rate = baseline_succ_total / max(1, baseline_tot_total)

print("\n===== FINAL =====")
print(f"Steering success rate: {steering_success_rate:.4f}  ({succ_total}/{tot_total})")
print(
    f"Baseline success rate: {baseline_success_rate:.4f}  ({baseline_succ_total}/{baseline_tot_total})"
)
print(
    f"validation_loss = {1.0 - steering_success_rate:.4f}"
)  # proxy val loss = 1 - success

experiment_data["concept_benchmark"][
    "overall_steering_success_rate"
] = steering_success_rate
experiment_data["concept_benchmark"][
    "overall_baseline_success_rate"
] = baseline_success_rate
experiment_data["concept_benchmark"]["predictions"] = [
    r["steered_success"] for r in per_concept_records
]
experiment_data["concept_benchmark"]["ground_truth"] = [
    r["total"] for r in per_concept_records
]
experiment_data["concept_benchmark"]["losses"]["val"].append(
    1.0 - steering_success_rate
)
experiment_data["concept_benchmark"]["config"] = {
    "model_path": MODEL_PATH,
    "steer_layer": STEER_LAYER,
    "alpha": ALPHA,
    "n_concepts": len(CONCEPTS),
    "n_eval_prompts_per_concept": N_EVAL_PROMPTS,
    "judge_model": JUDGE_MODEL,
}

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# ---------- Simple visualization ----------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(set(r["class"] for r in per_concept_records))
    cls_success = {c: [] for c in classes}
    cls_base = {c: [] for c in classes}
    for r in per_concept_records:
        cls_success[r["class"]].append(r["steered_success"] / r["total"])
        cls_base[r["class"]].append(r["baseline_success"] / r["total"])
    xs = np.arange(len(classes))
    steered_means = [np.mean(cls_success[c]) for c in classes]
    base_means = [np.mean(cls_base[c]) for c in classes]
    plt.figure(figsize=(8, 5))
    plt.bar(xs - 0.2, base_means, width=0.4, label="baseline")
    plt.bar(xs + 0.2, steered_means, width=0.4, label="steered")
    plt.xticks(xs, classes, rotation=30)
    plt.ylabel("Success rate")
    plt.title(f"Steering success by concept class (α={ALPHA}, layer={STEER_LAYER})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_success_by_class.png"))
    plt.close()
except Exception as e:
    print("plot err:", e)

print("Done.")
