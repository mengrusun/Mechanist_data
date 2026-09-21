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
    "layer_ablation": {
        "concept_benchmark": {
            "layers": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "per_layer": {},
            "overall_baseline_success_rate": None,
            "best_layer": None,
            "alpha": None,
        }
    }
}

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


from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.join(MODEL_DIR, "Llama-3.1-8B-Instruct")
if not os.path.isdir(MODEL_PATH):
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

# Sweep layers: ~25%, ~50%, ~75%, ~90%
LAYER_FRACTIONS = [0.25, 0.50, 0.75, 0.90]
STEER_LAYERS = sorted(set(int(round(f * n_layers)) for f in LAYER_FRACTIONS))
STEER_LAYERS = [min(max(0, l), n_layers - 1) for l in STEER_LAYERS]
print(f"Steer layers to sweep: {STEER_LAYERS}")

# Fixed best alpha (from prior alpha-tuning ablation)
BEST_ALPHA = 8.0
experiment_data["layer_ablation"]["concept_benchmark"]["alpha"] = BEST_ALPHA


def get_layer_module(idx):
    return model.model.layers[idx]


def format_chat(user_msg):
    msgs = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def get_last_token_hidden(prompts, layer_idx):
    outs = []
    for p in prompts:
        text = format_chat(p)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        o = model(**inputs, output_hidden_states=True, use_cache=False)
        h = o.hidden_states[layer_idx + 1][0, -1, :].float().cpu()
        outs.append(h)
        del o
    return torch.stack(outs, dim=0)


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
    if "true" in out.lower():
        return 1, out
    return 0, out


N_EVAL_PROMPTS = 3
eval_prompts = GENERIC_PROMPTS[:N_EVAL_PROMPTS]

# ---------- Baseline evaluation (layer-independent) ----------
print("\n===== Running baseline (no steering) once =====")
baseline_per_concept = {}
baseline_succ_total, baseline_tot_total = 0, 0
steer_hook.vector = None
steer_hook.alpha = 0.0
# Register a temporary hook on some layer just so any accidental firing is no-op (not needed strictly)
for ci, entry in enumerate(CONCEPTS):
    concept = entry["concept"]
    baseline_gens = []
    for p in eval_prompts:
        try:
            r = generate(p, max_new_tokens=80)
        except Exception as e:
            r = ""
            print("gen err:", e)
        baseline_gens.append(r)
    b_succ = 0
    b_details = []
    for p, br in zip(eval_prompts, baseline_gens):
        bj, _ = judge_concept(concept, br)
        b_succ += bj
        b_details.append({"prompt": p, "baseline": br, "baseline_judge": bj})
    baseline_per_concept[concept] = {
        "class": entry["class"],
        "success": b_succ,
        "total": len(eval_prompts),
        "details": b_details,
    }
    baseline_succ_total += b_succ
    baseline_tot_total += len(eval_prompts)
    print(f"[baseline {ci+1}/{len(CONCEPTS)}] {concept}: {b_succ}/{len(eval_prompts)}")

baseline_success_rate = baseline_succ_total / max(1, baseline_tot_total)
print(f"Baseline success rate: {baseline_success_rate:.4f}")
experiment_data["layer_ablation"]["concept_benchmark"][
    "overall_baseline_success_rate"
] = baseline_success_rate

# ---------- Layer sweep ----------
layer_results = {}
for STEER_LAYER in STEER_LAYERS:
    print(f"\n===== STEER_LAYER = {STEER_LAYER} (of {n_layers}) =====")

    # Compute concept vectors at this layer
    concept_vectors = {}
    print(f"Computing concept vectors at layer {STEER_LAYER}...")
    for i, entry in enumerate(CONCEPTS):
        concept = entry["concept"]
        pos_prompts = make_positive_prompts(concept, n=8)
        neg_prompts = make_negative_prompts(n=8)
        pos_h = get_last_token_hidden(pos_prompts, STEER_LAYER)
        neg_h = get_last_token_hidden(neg_prompts, STEER_LAYER)
        v = pos_h.mean(0) - neg_h.mean(0)
        v = v / (v.norm() + 1e-8)
        concept_vectors[concept] = v
        if (i + 1) % 5 == 0 or i == len(CONCEPTS) - 1:
            print(f"  [{i+1}/{len(CONCEPTS)}] vector for '{concept}'")

    # Register hook at this layer
    hook_handle = get_layer_module(STEER_LAYER).register_forward_hook(steer_hook)

    per_concept_records = []
    succ_total, tot_total = 0, 0
    running_metrics = []

    for ci, entry in enumerate(CONCEPTS):
        concept = entry["concept"]
        cls = entry["class"]
        vec = concept_vectors[concept]

        steer_hook.vector = vec
        steer_hook.alpha = BEST_ALPHA
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

        s_succ = 0
        details = []
        b_info = baseline_per_concept[concept]
        for idx, (p, sr) in enumerate(zip(eval_prompts, steered_gens)):
            sj, _ = judge_concept(concept, sr)
            s_succ += sj
            bd = b_info["details"][idx]
            details.append(
                {
                    "prompt": p,
                    "baseline": bd["baseline"],
                    "steered": sr,
                    "baseline_judge": bd["baseline_judge"],
                    "steered_judge": sj,
                }
            )
        tot = len(eval_prompts)
        succ_total += s_succ
        tot_total += tot

        rec = {
            "class": cls,
            "concept": concept,
            "baseline_success": b_info["success"],
            "steered_success": s_succ,
            "total": tot,
            "details": details,
        }
        per_concept_records.append(rec)
        print(
            f"[layer={STEER_LAYER} {ci+1}/{len(CONCEPTS)}] {concept}: baseline {b_info['success']}/{tot}, steered {s_succ}/{tot}"
        )

        running_metrics.append(
            {
                "concept_index": ci,
                "concept": concept,
                "steered_success_rate_running": succ_total / max(1, tot_total),
                "baseline_success_rate_running": baseline_success_rate,
            }
        )

    hook_handle.remove()

    steering_success_rate = succ_total / max(1, tot_total)
    print(
        f"[layer={STEER_LAYER}] Steering success rate: {steering_success_rate:.4f} ({succ_total}/{tot_total})"
    )

    layer_key = f"layer_{STEER_LAYER}"
    layer_results[layer_key] = {
        "layer": STEER_LAYER,
        "layer_fraction": STEER_LAYER / n_layers,
        "per_concept": per_concept_records,
        "overall_steering_success_rate": steering_success_rate,
        "overall_baseline_success_rate": baseline_success_rate,
        "val_loss": 1.0 - steering_success_rate,
        "predictions": [r["steered_success"] for r in per_concept_records],
        "ground_truth": [r["total"] for r in per_concept_records],
        "running_metrics": running_metrics,
        "config": {
            "model_path": MODEL_PATH,
            "steer_layer": STEER_LAYER,
            "alpha": BEST_ALPHA,
            "n_concepts": len(CONCEPTS),
            "n_eval_prompts_per_concept": N_EVAL_PROMPTS,
            "judge_model": JUDGE_MODEL,
            "n_layers_total": n_layers,
        },
    }

    experiment_data["layer_ablation"]["concept_benchmark"]["layers"] = list(
        layer_results.keys()
    )
    experiment_data["layer_ablation"]["concept_benchmark"]["per_layer"] = layer_results
    experiment_data["layer_ablation"]["concept_benchmark"]["metrics"]["val"].append(
        {
            "layer": STEER_LAYER,
            "layer_fraction": STEER_LAYER / n_layers,
            "steered_success_rate": steering_success_rate,
            "baseline_success_rate": baseline_success_rate,
        }
    )
    experiment_data["layer_ablation"]["concept_benchmark"]["losses"]["val"].append(
        {
            "layer": STEER_LAYER,
            "val_loss": 1.0 - steering_success_rate,
        }
    )
    np.save(
        os.path.join(working_dir, "experiment_data.npy"),
        experiment_data,
        allow_pickle=True,
    )

    # Cleanup between layers
    del concept_vectors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Best layer
best_layer_key = max(
    layer_results.keys(),
    key=lambda k: layer_results[k]["overall_steering_success_rate"],
)
best_layer = layer_results[best_layer_key]["layer"]
experiment_data["layer_ablation"]["concept_benchmark"]["best_layer"] = best_layer
experiment_data["layer_ablation"]["concept_benchmark"]["predictions"] = layer_results[
    best_layer_key
]["predictions"]
experiment_data["layer_ablation"]["concept_benchmark"]["ground_truth"] = layer_results[
    best_layer_key
]["ground_truth"]

print("\n===== FINAL SUMMARY =====")
for k, v in layer_results.items():
    print(
        f"{k}: steered success = {v['overall_steering_success_rate']:.4f}, val_loss = {v['val_loss']:.4f}"
    )
print(f"Baseline success rate: {baseline_success_rate:.4f}")
print(f"Best layer: {best_layer} (fraction {best_layer/n_layers:.3f})")

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# ---------- Visualization ----------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(
        set(r["class"] for r in layer_results[best_layer_key]["per_concept"])
    )
    xs = np.arange(len(classes))
    width = 0.8 / (len(STEER_LAYERS) + 1)

    plt.figure(figsize=(10, 5))
    base_by_class = {c: [] for c in classes}
    for r in layer_results[best_layer_key]["per_concept"]:
        base_by_class[r["class"]].append(r["baseline_success"] / r["total"])
    base_means = [np.mean(base_by_class[c]) for c in classes]
    plt.bar(xs - 0.4 + width / 2, base_means, width=width, label="baseline")

    for i, L in enumerate(STEER_LAYERS):
        lkey = f"layer_{L}"
        cls_success = {c: [] for c in classes}
        for r in layer_results[lkey]["per_concept"]:
            cls_success[r["class"]].append(r["steered_success"] / r["total"])
        means = [np.mean(cls_success[c]) for c in classes]
        plt.bar(xs - 0.4 + width * (i + 1.5), means, width=width, label=f"layer={L}")

    plt.xticks(xs, classes, rotation=30)
    plt.ylabel("Success rate")
    plt.title(f"Steering success by concept class across layers (α={BEST_ALPHA})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "layer_ablation_success_by_class.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    layers_x = STEER_LAYERS
    ys = [
        layer_results[f"layer_{L}"]["overall_steering_success_rate"] for L in layers_x
    ]
    plt.plot(layers_x, ys, marker="o", label="steered")
    plt.axhline(baseline_success_rate, color="gray", linestyle="--", label="baseline")
    plt.xlabel("Steering layer index")
    plt.ylabel("Overall success rate")
    plt.title(f"Layer ablation: overall steering success (α={BEST_ALPHA})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "layer_ablation_overall.png"))
    plt.close()
except Exception as e:
    print("plot err:", e)
    traceback.print_exc()

print("Done.")
