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

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

experiment_data = {
    "layer_alpha_sweep": {
        "concept_benchmark": {
            "configs": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "per_config": {},
            "baseline": {},
            "prompt_baseline": {},
            "best_config": None,
        }
    }
}

# ---------------- OpenAI judge ----------------
try:
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def chat(messages, max_tokens=120, temperature=0.0, retries=3, model=JUDGE_MODEL):
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


# ---------------- Concepts ----------------
CONCEPT_CLASSES = {
    "fears": ["fear of heights", "fear of spiders", "fear of public speaking"],
    "experts": ["expert chef", "expert astronomer"],
    "moods": ["joyful mood", "melancholic mood"],
    "topophiles": ["love of mountains", "love of oceans"],
    "personas": ["pirate persona", "shakespearean persona", "cowboy persona"],
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
    "Share a random thought.",
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


# ---------------- Model ----------------
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


# ---------------- Extract concept vectors per candidate layer ----------------
LAYER_CANDIDATES = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4]  # ~8, 16, 24
print(f"Candidate layers: {LAYER_CANDIDATES}")

concept_vectors_per_layer = {L: {} for L in LAYER_CANDIDATES}
print("Extracting concept vectors (per candidate layer)...")
for i, entry in enumerate(CONCEPTS):
    concept = entry["concept"]
    pos_prompts = make_positive_prompts(concept, n=6)
    neg_prompts = make_negative_prompts(n=6)
    for L in LAYER_CANDIDATES:
        pos_h = get_last_token_hidden(pos_prompts, L)
        neg_h = get_last_token_hidden(neg_prompts, L)
        v = pos_h.mean(0) - neg_h.mean(0)
        v_norm = v.norm() + 1e-8
        # Store unnormalized direction and its scale for adaptive alpha
        concept_vectors_per_layer[L][concept] = {
            "vec_unit": (v / v_norm),
            "vec_norm": float(v_norm.item()),
        }
    print(f"[{i+1}/{len(CONCEPTS)}] '{concept}' done")


# ---------------- Steering hook (supports fixed & adaptive alpha) ----------------
class SteerHook:
    def __init__(self):
        self.vector = None  # unit vector (torch)
        self.alpha = 0.0  # fixed alpha
        self.mode = "fixed"  # "fixed" or "adaptive"

    def __call__(self, module, inp, out):
        if self.vector is None or self.alpha == 0.0:
            return out
        if isinstance(out, tuple):
            h = out[0]
        else:
            h = out
        v = self.vector.to(h.device, dtype=h.dtype).view(1, 1, -1)
        if self.mode == "adaptive":
            # Scale by mean activation norm at each token so effective alpha ~ constant fraction
            token_norms = h.norm(dim=-1, keepdim=True)  # [B,T,1]
            h = h + self.alpha * token_norms * v
        else:
            h = h + self.alpha * v
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h


steer_hook = SteerHook()
hook_handle = None
current_hook_layer = None


def set_hook_layer(L):
    global hook_handle, current_hook_layer
    if hook_handle is not None:
        hook_handle.remove()
        hook_handle = None
    hook_handle = get_layer_module(L).register_forward_hook(steer_hook)
    current_hook_layer = L


# ---------------- Generation + coherence ----------------
@torch.no_grad()
def generate(prompt, max_new_tokens=70):
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
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True), gen_ids


@torch.no_grad()
def perplexity_of_text(text):
    # unsteered PPL of generated text (hook off)
    old_vec, old_alpha = steer_hook.vector, steer_hook.alpha
    steer_hook.vector = None
    steer_hook.alpha = 0.0
    if not text or not text.strip():
        steer_hook.vector, steer_hook.alpha = old_vec, old_alpha
        return float("nan")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(
        device
    )
    if enc["input_ids"].shape[1] < 2:
        steer_hook.vector, steer_hook.alpha = old_vec, old_alpha
        return float("nan")
    out = model(**enc, labels=enc["input_ids"])
    ppl = float(torch.exp(out.loss).item())
    steer_hook.vector, steer_hook.alpha = old_vec, old_alpha
    return ppl


# ---------------- Judge ----------------
def judge_concept(concept, response):
    if response is None or len(response.strip()) == 0:
        return 0, "empty"
    sys = (
        "You are an evaluator. Judge whether the assistant's response clearly reflects, "
        "embodies, or expresses the given target concept while remaining coherent. "
        'Respond with JSON: {"reflects": true|false}.'
    )
    user = f"Target concept: {concept}\n\nResponse:\n{response}\n\nDoes it clearly reflect the concept?"
    out = chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=60,
        temperature=0.0,
    )
    if out is None:
        return 0, "judge_failed"
    m = re.search(r'"reflects"\s*:\s*(true|false)', out.lower())
    if m:
        return (1 if m.group(1) == "true" else 0), out
    return (1 if "true" in out.lower() else 0), out


# ---------------- Eval prompts & config ----------------
N_EVAL_PROMPTS = 3
eval_prompts = GENERIC_PROMPTS[:N_EVAL_PROMPTS]


# Pair each concept with a distractor from a DIFFERENT class (for off-target metric)
def build_distractors():
    all_concepts = [c["concept"] for c in CONCEPTS]
    dmap = {}
    for e in CONCEPTS:
        others = [x["concept"] for x in CONCEPTS if x["class"] != e["class"]]
        dmap[e["concept"]] = random.choice(others)
    return dmap


distractor_map = build_distractors()

# ---------------- Baselines (once) ----------------
print("\n===== Baseline (no steering) =====")
steer_hook.vector = None
steer_hook.alpha = 0.0
baseline_records = {}
b_succ_total, b_off_total, b_tot = 0, 0, 0
b_ppls = []
for ci, entry in enumerate(CONCEPTS):
    concept = entry["concept"]
    dist = distractor_map[concept]
    b_res = []
    for p in eval_prompts:
        try:
            r, _ = generate(p)
        except Exception as e:
            r = ""
            print("gen err:", e)
        b_res.append(r)
    s_ok, off_ok, ppls = 0, 0, []
    for r in b_res:
        j, _ = judge_concept(concept, r)
        s_ok += j
        j2, _ = judge_concept(dist, r)
        off_ok += j2
        ppls.append(perplexity_of_text(r))
    baseline_records[concept] = {
        "class": entry["class"],
        "success": s_ok,
        "off_target": off_ok,
        "total": len(eval_prompts),
        "gens": b_res,
        "ppls": ppls,
        "distractor": dist,
    }
    b_succ_total += s_ok
    b_off_total += off_ok
    b_tot += len(eval_prompts)
    b_ppls.extend([p for p in ppls if not math.isnan(p)])
    print(
        f"[baseline {ci+1}/{len(CONCEPTS)}] {concept}: succ {s_ok}/{len(eval_prompts)}, off {off_ok}"
    )

baseline_rate = b_succ_total / max(1, b_tot)
baseline_off_rate = b_off_total / max(1, b_tot)
baseline_ppl_med = float(np.median(b_ppls)) if b_ppls else float("nan")
print(
    f"Baseline: success={baseline_rate:.4f}, off_target={baseline_off_rate:.4f}, ppl_med={baseline_ppl_med:.3f}"
)

experiment_data["layer_alpha_sweep"]["concept_benchmark"]["baseline"] = {
    "success_rate": baseline_rate,
    "off_target_rate": baseline_off_rate,
    "ppl_median": baseline_ppl_med,
    "records": baseline_records,
}

# ---------------- Prompt-based steering baseline ("mention concept X") ----------------
print("\n===== Prompt-based baseline =====")
steer_hook.vector = None
steer_hook.alpha = 0.0
prompt_records = {}
p_succ, p_off, p_tot = 0, 0, 0
p_ppls = []
for ci, entry in enumerate(CONCEPTS):
    concept = entry["concept"]
    dist = distractor_map[concept]
    outs = []
    for p in eval_prompts:
        prompted = f"[Please answer in a way that clearly reflects {concept}.] {p}"
        try:
            r, _ = generate(prompted)
        except Exception as e:
            r = ""
            print("gen err:", e)
        outs.append(r)
    s_ok, off_ok, ppls = 0, 0, []
    for r in outs:
        j, _ = judge_concept(concept, r)
        s_ok += j
        j2, _ = judge_concept(dist, r)
        off_ok += j2
        ppls.append(perplexity_of_text(r))
    prompt_records[concept] = {
        "class": entry["class"],
        "success": s_ok,
        "off_target": off_ok,
        "total": len(eval_prompts),
        "gens": outs,
        "ppls": ppls,
    }
    p_succ += s_ok
    p_off += off_ok
    p_tot += len(eval_prompts)
    p_ppls.extend([q for q in ppls if not math.isnan(q)])
    print(
        f"[prompt {ci+1}/{len(CONCEPTS)}] {concept}: succ {s_ok}/{len(eval_prompts)}, off {off_ok}"
    )

prompt_rate = p_succ / max(1, p_tot)
prompt_off_rate = p_off / max(1, p_tot)
prompt_ppl_med = float(np.median(p_ppls)) if p_ppls else float("nan")
print(
    f"Prompt baseline: success={prompt_rate:.4f}, off_target={prompt_off_rate:.4f}, ppl_med={prompt_ppl_med:.3f}"
)

experiment_data["layer_alpha_sweep"]["concept_benchmark"]["prompt_baseline"] = {
    "success_rate": prompt_rate,
    "off_target_rate": prompt_off_rate,
    "ppl_median": prompt_ppl_med,
    "records": prompt_records,
}

# ---------------- Sweep: layer × alpha (fixed) + adaptive variant ----------------
FIXED_ALPHAS = [4.0, 8.0, 12.0]
ADAPTIVE_ALPHAS = [0.5, 1.0]  # multiplied by activation norm

configs = []
for L in LAYER_CANDIDATES:
    for a in FIXED_ALPHAS:
        configs.append({"layer": L, "alpha": a, "mode": "fixed"})
    for a in ADAPTIVE_ALPHAS:
        configs.append({"layer": L, "alpha": a, "mode": "adaptive"})

per_config = {}
for cfg in configs:
    L, a, mode = cfg["layer"], cfg["alpha"], cfg["mode"]
    key = f"L{L}_a{a}_{mode}"
    print(f"\n===== Config {key} =====")
    set_hook_layer(L)
    steer_hook.mode = mode

    per_concept_records = []
    s_total, off_total, tot = 0, 0, 0
    ppls_all = []
    for ci, entry in enumerate(CONCEPTS):
        concept = entry["concept"]
        dist = distractor_map[concept]
        vec = concept_vectors_per_layer[L][concept]["vec_unit"]
        steer_hook.vector = vec
        steer_hook.alpha = a
        gens = []
        for p in eval_prompts:
            try:
                r, _ = generate(p)
            except Exception as e:
                r = ""
                print("gen err:", e)
            gens.append(r)
        steer_hook.vector = None
        steer_hook.alpha = 0.0

        s_ok, off_ok, ppls = 0, 0, []
        for r in gens:
            j, _ = judge_concept(concept, r)
            s_ok += j
            j2, _ = judge_concept(dist, r)
            off_ok += j2
            ppls.append(perplexity_of_text(r))
        s_total += s_ok
        off_total += off_ok
        tot += len(eval_prompts)
        ppls_all.extend([q for q in ppls if not math.isnan(q)])

        rec = {
            "concept": concept,
            "class": entry["class"],
            "steered_success": s_ok,
            "off_target": off_ok,
            "total": len(eval_prompts),
            "gens": gens,
            "ppls": ppls,
            "baseline_success": baseline_records[concept]["success"],
        }
        per_concept_records.append(rec)
        print(
            f"[{key} {ci+1}/{len(CONCEPTS)}] {concept}: succ {s_ok}/{len(eval_prompts)}, off {off_ok}"
        )

    succ_rate = s_total / max(1, tot)
    off_rate = off_total / max(1, tot)
    ppl_med = float(np.median(ppls_all)) if ppls_all else float("nan")
    # Composite score: steering success - off-target - PPL penalty (relative)
    ppl_penalty = 0.0
    if (
        not math.isnan(ppl_med)
        and not math.isnan(baseline_ppl_med)
        and baseline_ppl_med > 0
    ):
        ppl_penalty = max(0.0, (ppl_med - baseline_ppl_med) / baseline_ppl_med)
    composite = succ_rate - off_rate - 0.1 * ppl_penalty

    per_config[key] = {
        "layer": L,
        "alpha": a,
        "mode": mode,
        "success_rate": succ_rate,
        "off_target_rate": off_rate,
        "ppl_median": ppl_med,
        "ppl_penalty": ppl_penalty,
        "composite_score": composite,
        "val_loss": 1.0 - succ_rate,
        "per_concept": per_concept_records,
    }
    print(
        f"[{key}] success={succ_rate:.4f}, off={off_rate:.4f}, ppl_med={ppl_med:.3f}, composite={composite:.4f}"
    )

    # incremental save
    experiment_data["layer_alpha_sweep"]["concept_benchmark"]["per_config"] = per_config
    experiment_data["layer_alpha_sweep"]["concept_benchmark"]["configs"] = list(
        per_config.keys()
    )
    experiment_data["layer_alpha_sweep"]["concept_benchmark"]["metrics"]["val"].append(
        {
            "config": key,
            "success_rate": succ_rate,
            "off_target_rate": off_rate,
            "ppl_median": ppl_med,
            "composite": composite,
            "baseline_success_rate": baseline_rate,
            "prompt_success_rate": prompt_rate,
        }
    )
    experiment_data["layer_alpha_sweep"]["concept_benchmark"]["losses"]["val"].append(
        {"config": key, "val_loss": 1.0 - succ_rate}
    )
    np.save(
        os.path.join(working_dir, "experiment_data.npy"),
        experiment_data,
        allow_pickle=True,
    )

if hook_handle is not None:
    hook_handle.remove()

# ---------------- Best config ----------------
best_key = max(per_config.keys(), key=lambda k: per_config[k]["composite_score"])
best_by_succ = max(per_config.keys(), key=lambda k: per_config[k]["success_rate"])
print(f"\nBest by composite: {best_key} -> {per_config[best_key]}")
print(
    f"Best by raw success: {best_by_succ} -> success={per_config[best_by_succ]['success_rate']:.4f}"
)

experiment_data["layer_alpha_sweep"]["concept_benchmark"]["best_config"] = best_key
experiment_data["layer_alpha_sweep"]["concept_benchmark"][
    "best_config_by_success"
] = best_by_succ
experiment_data["layer_alpha_sweep"]["concept_benchmark"]["predictions"] = [
    r["steered_success"] for r in per_config[best_key]["per_concept"]
]
experiment_data["layer_alpha_sweep"]["concept_benchmark"]["ground_truth"] = [
    r["total"] for r in per_config[best_key]["per_concept"]
]

# print required epoch-style validation loss line
for k, v in per_config.items():
    print(
        f"Epoch {k}: validation_loss = {v['val_loss']:.4f}  concept_steering_success_rate = {v['success_rate']:.4f}"
    )

# Final required metric print
print(
    f"\nFINAL concept_steering_success_rate (best composite config {best_key}): "
    f"{per_config[best_key]['success_rate']:.4f}"
)
print(
    f"FINAL concept_steering_success_rate (best raw config {best_by_succ}): "
    f"{per_config[best_by_succ]['success_rate']:.4f}"
)
print(f"Baseline: {baseline_rate:.4f}, Prompt-baseline: {prompt_rate:.4f}")

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# ---------------- Plots ----------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Heatmap: layer x alpha (fixed mode only)
    Ls = LAYER_CANDIDATES
    As = FIXED_ALPHAS
    grid_succ = np.full((len(Ls), len(As)), np.nan)
    grid_off = np.full((len(Ls), len(As)), np.nan)
    for i, L in enumerate(Ls):
        for j, a in enumerate(As):
            k = f"L{L}_a{a}_fixed"
            if k in per_config:
                grid_succ[i, j] = per_config[k]["success_rate"]
                grid_off[i, j] = per_config[k]["off_target_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im0 = axes[0].imshow(grid_succ, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(As)))
    axes[0].set_xticklabels(As)
    axes[0].set_yticks(range(len(Ls)))
    axes[0].set_yticklabels(Ls)
    axes[0].set_title("Steering success rate")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("layer")
    for i in range(len(Ls)):
        for j in range(len(As)):
            axes[0].text(
                j, i, f"{grid_succ[i,j]:.2f}", ha="center", va="center", color="w"
            )
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(grid_off, cmap="magma", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(As)))
    axes[1].set_xticklabels(As)
    axes[1].set_yticks(range(len(Ls)))
    axes[1].set_yticklabels(Ls)
    axes[1].set_title("Off-target rate")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("layer")
    for i in range(len(Ls)):
        for j in range(len(As)):
            axes[1].text(
                j, i, f"{grid_off[i,j]:.2f}", ha="center", va="center", color="w"
            )
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "layer_alpha_heatmap.png"))
    plt.close()

    # Bar: baseline vs prompt vs best fixed vs best adaptive
    best_fixed_key = max(
        [k for k in per_config if per_config[k]["mode"] == "fixed"],
        key=lambda k: per_config[k]["success_rate"],
    )
    adapt_keys = [k for k in per_config if per_config[k]["mode"] == "adaptive"]
    best_adapt_key = (
        max(adapt_keys, key=lambda k: per_config[k]["success_rate"])
        if adapt_keys
        else None
    )

    names = ["Unsteered", "Prompt", f"Fixed({best_fixed_key})"]
    succ_vals = [baseline_rate, prompt_rate, per_config[best_fixed_key]["success_rate"]]
    off_vals = [
        baseline_off_rate,
        prompt_off_rate,
        per_config[best_fixed_key]["off_target_rate"],
    ]
    if best_adapt_key:
        names.append(f"Adaptive({best_adapt_key})")
        succ_vals.append(per_config[best_adapt_key]["success_rate"])
        off_vals.append(per_config[best_adapt_key]["off_target_rate"])

    x = np.arange(len(names))
    w = 0.35
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - w / 2, succ_vals, w, label="on-target success", color="steelblue")
    plt.bar(x + w / 2, off_vals, w, label="off-target (distractor)", color="salmon")
    plt.xticks(x, names, rotation=15)
    plt.ylabel("Rate")
    plt.title("Steering methods: on-target vs off-target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "methods_comparison.png"))
    plt.close()

    # Perplexity vs alpha (fixed mode, per layer)
    plt.figure(figsize=(7, 4))
    for L in LAYER_CANDIDATES:
        ys = []
        for a in FIXED_ALPHAS:
            k = f"L{L}_a{a}_fixed"
            ys.append(per_config[k]["ppl_median"] if k in per_config else np.nan)
        plt.plot(FIXED_ALPHAS, ys, marker="o", label=f"layer {L}")
    plt.axhline(baseline_ppl_med, color="gray", linestyle="--", label="baseline ppl")
    plt.xlabel("alpha")
    plt.ylabel("median PPL of generation")
    plt.title("Coherence (lower is better) vs steering strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "ppl_vs_alpha.png"))
    plt.close()

except Exception as e:
    print("plot err:", e)
    traceback.print_exc()

print("Done.")
