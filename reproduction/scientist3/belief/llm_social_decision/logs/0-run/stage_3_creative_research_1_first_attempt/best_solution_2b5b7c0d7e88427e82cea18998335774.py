import os, re, json, math, random, gc
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_DIR = "/data/zhenqian/models"
MODEL_PATH = os.path.join(MODEL_DIR, "Llama-3.1-8B-Instruct")
if not os.path.isdir(MODEL_PATH):
    # try alternate name
    for cand in [
        "llama-3.1-8b-instruct",
        "Meta-Llama-3.1-8B-Instruct",
        "meta-llama-3.1-8b-instruct",
    ]:
        p = os.path.join(MODEL_DIR, cand)
        if os.path.isdir(p):
            MODEL_PATH = p
            break
print("Loading model from:", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, local_files_only=True, torch_dtype=torch.float16
).to(device)
model.eval()

num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
TARGET_LAYER = num_layers // 2  # middle layer
print(f"num_layers={num_layers}, hidden={hidden_size}, target_layer={TARGET_LAYER}")

# ---------- Build dictator-game prompts ----------
GENDERS = ["male", "female"]
AGES = ["young (22 years old)", "old (68 years old)"]
INSTRUCTIONS = [
    (
        "A",
        "You are playing a dictator game. You have been given $20. You may transfer any amount from $0 to $20 to an anonymous recipient. Choose the amount to transfer.",
    ),
    (
        "B",
        "You are participating in an economic decision task. An initial endowment of $20 is yours. You may share any portion of it — between $0 and $20 — with another anonymous person. Decide how much to give.",
    ),
]
MEETINGS = [
    (
        "meeting",
        "Before making your decision, you briefly met the recipient in person.",
    ),
    (
        "no-meeting",
        "You have never met and will never meet the recipient; the interaction is fully anonymous.",
    ),
]

SYS = "You are participating in a behavioral economics study. Answer only with a single dollar amount between $0 and $20."


def build_prompt(gender, age, instr_text, meeting_text):
    user = (
        f"Personal profile: You are a {gender} person, {age}.\n"
        f"{instr_text}\n"
        f"Context: {meeting_text}\n"
        f"How much do you transfer? Reply with only the dollar amount (a single integer between 0 and 20)."
    )
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


rng = random.Random(0)
N_TRIALS = 1000
trials = []
for i in range(N_TRIALS):
    g = rng.choice(GENDERS)
    a = rng.choice(AGES)
    ilab, itxt = rng.choice(INSTRUCTIONS)
    mlab, mtxt = rng.choice(MEETINGS)
    trials.append(
        {
            "gender": g,
            "age": a,
            "instr": ilab,
            "instr_text": itxt,
            "meeting": mlab,
            "meeting_text": mtxt,
            "prompt": build_prompt(g, a, itxt, mtxt),
        }
    )

# ---------- Activation extraction ----------
_cap = {}


def make_hook(name):
    def hook(mod, inp, out):
        # out is tuple; first is hidden states
        if isinstance(out, tuple):
            h = out[0]
        else:
            h = out
        _cap[name] = h.detach()

    return hook


layer_module = model.model.layers[TARGET_LAYER]


@torch.no_grad()
def get_last_token_activation(prompt):
    _cap.clear()
    handle = layer_module.register_forward_hook(make_hook("h"))
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        model(**inputs)
        h = _cap["h"]  # [1, seq, hidden]
        vec = h[0, -1, :].float().cpu().numpy()
    finally:
        handle.remove()
    return vec


# ---------- Build paired prompts to extract per-variable difference vectors ----------
# For each variable, pair prompts differ only in that variable; others randomized identically
N_PAIRS = 64


def flip(t, var):
    t = dict(t)
    if var == "gender":
        t["gender"] = "female" if t["gender"] == "male" else "male"
    elif var == "age":
        t["age"] = AGES[1] if t["age"] == AGES[0] else AGES[0]
    elif var == "instr":
        other = [x for x in INSTRUCTIONS if x[0] != t["instr"]][0]
        t["instr"], t["instr_text"] = other[0], other[1]
    elif var == "meeting":
        other = [x for x in MEETINGS if x[0] != t["meeting"]][0]
        t["meeting"], t["meeting_text"] = other[0], other[1]
    t["prompt"] = build_prompt(
        t["gender"], t["age"], t["instr_text"], t["meeting_text"]
    )
    return t


variables = ["gender", "age", "instr", "meeting"]
# positive labels (what we call +1):
POS = {"gender": "male", "age": AGES[0], "instr": "A", "meeting": "meeting"}

directions_raw = {}
for var in variables:
    diffs = []
    base_trials = rng.sample(trials, N_PAIRS)
    for t in base_trials:
        t_pos = dict(t)
        t_neg = dict(t)
        # set to positive
        if var == "gender":
            t_pos["gender"] = "male"
            t_neg["gender"] = "female"
        elif var == "age":
            t_pos["age"] = AGES[0]
            t_neg["age"] = AGES[1]
        elif var == "instr":
            t_pos["instr"], t_pos["instr_text"] = INSTRUCTIONS[0]
            t_neg["instr"], t_neg["instr_text"] = INSTRUCTIONS[1]
        elif var == "meeting":
            t_pos["meeting"], t_pos["meeting_text"] = MEETINGS[0]
            t_neg["meeting"], t_neg["meeting_text"] = MEETINGS[1]
        p_pos = build_prompt(
            t_pos["gender"],
            t_pos["age"],
            t_pos.get(
                "instr_text",
                INSTRUCTIONS[0][1] if t_pos["instr"] == "A" else INSTRUCTIONS[1][1],
            ),
            t_pos.get(
                "meeting_text",
                MEETINGS[0][1] if t_pos["meeting"] == "meeting" else MEETINGS[1][1],
            ),
        )
        p_neg = build_prompt(
            t_neg["gender"],
            t_neg["age"],
            t_neg.get(
                "instr_text",
                INSTRUCTIONS[0][1] if t_neg["instr"] == "A" else INSTRUCTIONS[1][1],
            ),
            t_neg.get(
                "meeting_text",
                MEETINGS[0][1] if t_neg["meeting"] == "meeting" else MEETINGS[1][1],
            ),
        )
        v_pos = get_last_token_activation(p_pos)
        v_neg = get_last_token_activation(p_neg)
        diffs.append(v_pos - v_neg)
    d = np.mean(np.stack(diffs, 0), axis=0)
    directions_raw[var] = d
    print(f"[{var}] raw direction norm={np.linalg.norm(d):.3f}")


# Orthogonalize (Gram-Schmidt) to get "pure" directions
def gram_schmidt(vecs_dict, order):
    pure = {}
    basis = []
    for name in order:
        v = vecs_dict[name].astype(np.float64).copy()
        for b in basis:
            v = v - (np.dot(v, b) / np.dot(b, b)) * b
        pure[name] = v
        basis.append(v)
    return pure


directions_pure = gram_schmidt(directions_raw, variables)
for var in variables:
    print(
        f"[{var}] pure norm={np.linalg.norm(directions_pure[var]):.3f}, cos vs raw={np.dot(directions_pure[var],directions_raw[var])/(np.linalg.norm(directions_pure[var])*np.linalg.norm(directions_raw[var])+1e-9):.3f}"
    )

# ---------- Intervention via forward hook ----------
_inject = {"vec": None, "alpha": 0.0}


def inject_hook(mod, inp, out):
    if _inject["vec"] is None:
        return out
    if isinstance(out, tuple):
        h = out[0]
        h = h + _inject["alpha"] * _inject["vec"].to(h.dtype).to(h.device)
        return (h,) + out[1:]
    else:
        return out + _inject["alpha"] * _inject["vec"].to(out.dtype).to(out.device)


inject_handle = layer_module.register_forward_hook(inject_hook)


def parse_amount(text):
    m = re.search(r"\$?\s*(\d+(?:\.\d+)?)", text)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 20:
                return v
        except:
            pass
    return None


@torch.no_grad()
def generate_amount(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return parse_amount(gen), gen


# Baseline transfers (no intervention) on subset for sanity + all trials
_inject["vec"] = None
N_BASE = 100
base_amounts = []
for i in range(N_BASE):
    amt, gen = generate_amount(trials[i]["prompt"])
    base_amounts.append(amt)
base_valid = [x for x in base_amounts if x is not None]
print(
    f"Baseline (no injection): n_valid={len(base_valid)}, mean={np.mean(base_valid):.2f}, std={np.std(base_valid):.2f}"
)

# Determine alpha scale using typical activation norm at target layer
_inject["vec"] = None
sample_norms = []
for i in range(8):
    v = get_last_token_activation(trials[i]["prompt"])
    sample_norms.append(np.linalg.norm(v))
mean_act_norm = float(np.mean(sample_norms))
print(f"Mean activation norm at layer {TARGET_LAYER}: {mean_act_norm:.2f}")

# Intervention experiment
experiment_data = {
    "dictator_game": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "effect_sizes": {},
        "per_variable_transfers": {},
        "baseline_amounts": base_amounts,
        "config": {
            "target_layer": TARGET_LAYER,
            "n_pairs": N_PAIRS,
            "mean_act_norm": mean_act_norm,
        },
    }
}

N_INT = 60  # trials per condition per variable
ALPHA_UNIT = 8.0  # magnitude in activation-space units; scaled per direction to match mean_act_norm
results = {}
for var in variables:
    d = directions_pure[var]
    dnorm = np.linalg.norm(d) + 1e-9
    # scale so that alpha*d has norm ~ ALPHA_UNIT
    scale = ALPHA_UNIT / dnorm
    dvec = torch.from_numpy(d.astype(np.float32)).to(device)
    pos_amts, neg_amts = [], []
    pos_raw, neg_raw = [], []
    subset = trials[:N_INT]
    for t in subset:
        _inject["vec"] = dvec
        _inject["alpha"] = +scale
        a_pos, g_pos = generate_amount(t["prompt"])
        _inject["alpha"] = -scale
        a_neg, g_neg = generate_amount(t["prompt"])
        pos_amts.append(a_pos)
        neg_amts.append(a_neg)
        pos_raw.append(g_pos)
        neg_raw.append(g_neg)
    _inject["vec"] = None
    _inject["alpha"] = 0.0
    pv = np.array([x for x in pos_amts if x is not None], dtype=float)
    nv = np.array([x for x in neg_amts if x is not None], dtype=float)
    # pair-wise Cohen's d using pooled std across all
    all_vals = np.concatenate([pv, nv]) if len(pv) + len(nv) > 0 else np.array([0.0])
    pooled_std = np.std(all_vals, ddof=1) if len(all_vals) > 1 else 1.0
    if pooled_std < 1e-6:
        pooled_std = 1.0
    d_effect = (
        (np.mean(pv) - np.mean(nv)) / pooled_std
        if len(pv) > 0 and len(nv) > 0
        else float("nan")
    )
    results[var] = {
        "pos_mean": float(np.mean(pv)) if len(pv) > 0 else float("nan"),
        "neg_mean": float(np.mean(nv)) if len(nv) > 0 else float("nan"),
        "pos_n": int(len(pv)),
        "neg_n": int(len(nv)),
        "cohens_d": float(d_effect),
        "pos_amounts": pos_amts,
        "neg_amounts": neg_amts,
    }
    experiment_data["dictator_game"]["effect_sizes"][var] = float(d_effect)
    experiment_data["dictator_game"]["per_variable_transfers"][var] = {
        "pos": pos_amts,
        "neg": neg_amts,
    }
    experiment_data["dictator_game"]["metrics"]["val"].append(
        {"variable": var, "cohens_d": float(d_effect)}
    )
    print(
        f"[{var}] pos_mean={results[var]['pos_mean']:.2f} (n={results[var]['pos_n']}), "
        f"neg_mean={results[var]['neg_mean']:.2f} (n={results[var]['neg_n']}), "
        f"Cohen's d = {d_effect:.3f}"
    )

# Aggregate metric: mean |d|
abs_ds = [
    abs(v)
    for v in experiment_data["dictator_game"]["effect_sizes"].values()
    if not math.isnan(v)
]
mean_abs_d = float(np.mean(abs_ds)) if abs_ds else float("nan")
print(
    f"\nvariable_intervention_effect_size (mean |Cohen's d| across variables) = {mean_abs_d:.3f}"
)
experiment_data["dictator_game"]["metrics"]["train"].append(
    {"mean_abs_cohens_d": mean_abs_d}
)

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(6, 4))
vars_list = list(experiment_data["dictator_game"]["effect_sizes"].keys())
vals = [experiment_data["dictator_game"]["effect_sizes"][v] for v in vars_list]
ax.bar(vars_list, vals, color=["steelblue", "salmon", "seagreen", "goldenrod"])
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Cohen's d (pos vs neg injection)")
ax.set_title(f"Variable intervention effect (layer {TARGET_LAYER})")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "dictator_game_effect_sizes.png"), dpi=120)
plt.close(fig)

# Save experiment data
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy and figure to", working_dir)

inject_handle.remove()
del model
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
