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

import os, re, json, gc, math, time, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_DIR = "/data/zhenqian/models"
MODEL_PATH = os.path.join(MODEL_DIR, "DeepSeek-R1-Distill-Llama-8B")
if not os.path.isdir(MODEL_PATH):
    for cand in os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else []:
        if "R1-Distill-Llama-8B" in cand:
            MODEL_PATH = os.path.join(MODEL_DIR, cand)
            break
print(f"Model path: {MODEL_PATH}")
assert os.path.isdir(MODEL_PATH), f"Model missing: {MODEL_PATH}"

experiment_data = {
    "steering_analysis": {
        "r1_distill_llama_8b": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "dose_response": {},
            "layer_ablation": {},
            "cross_interference": {},
            "bse_scores": {},
            "extraction_pool_sizes": {},
            "baseline": {},
        }
    }
}

TASKS = [
    {
        "q": "A store sells apples for $3 each and oranges for $2 each. Alice buys 7 apples and 5 oranges, then returns 2 apples. How much did she spend in total?",
        "a": 25,
    },
    {
        "q": "A train travels 60 miles in the first hour, then increases speed by 20 mph for the next 2 hours. What is the total distance traveled?",
        "a": 220,
    },
    {"q": "If 3x + 7 = 2x + 15, what is the value of x?", "a": 8},
    {
        "q": "A rectangle has length 12 and width 5. If both dimensions are doubled, what is the new area?",
        "a": 240,
    },
    {
        "q": "Sam has 3 times as many marbles as Tim. Together they have 48 marbles. How many marbles does Tim have?",
        "a": 12,
    },
    {"q": "What is 15% of 240?", "a": 36},
    {
        "q": "A car uses 4 gallons of gas to travel 120 miles. How many gallons does it need to travel 300 miles?",
        "a": 10,
    },
    {
        "q": "The sum of three consecutive even integers is 66. What is the largest of them?",
        "a": 24,
    },
    {
        "q": "A pizza is cut into 8 slices. If John eats 3 slices and Mary eats 2 slices, what fraction of the pizza is left? Give the numerator when expressed as a fraction with denominator 8.",
        "a": 3,
    },
    {
        "q": "A book originally costs $40. After a 25% discount, then an additional 10% off the discounted price, what is the final price in dollars?",
        "a": 27,
    },
    {
        "q": "If a right triangle has legs of length 6 and 8, what is the length of its hypotenuse?",
        "a": 10,
    },
    {
        "q": "The average of 5 numbers is 20. If one number is removed, the average becomes 18. What number was removed?",
        "a": 28,
    },
    {
        "q": "A tank holds 200 gallons. It is filled at 8 gallons per minute but leaks 3 gallons per minute. How many minutes until it's full, starting empty?",
        "a": 40,
    },
    {"q": "Find the value of x if 2^x = 32.", "a": 5},
    {
        "q": "A jar has 10 red, 15 blue, and 25 green marbles. What is the probability of drawing a blue marble? Express as a percentage.",
        "a": 30,
    },
    {
        "q": "The perimeter of a square is 36 cm. What is its area in square cm?",
        "a": 81,
    },
    {
        "q": "A worker earns $15 per hour for the first 40 hours, and time-and-a-half for overtime. If they work 46 hours, what is their total pay in dollars?",
        "a": 735,
    },
    {"q": "How many prime numbers are there between 10 and 30?", "a": 6},
    {"q": "If f(x) = 2x^2 - 3x + 1, what is f(4)?", "a": 21},
    {
        "q": "A rectangle's length is 3 more than twice its width. If the perimeter is 36, what is the width?",
        "a": 5,
    },
    {
        "q": "The sum of the digits of a two-digit number is 11. Reversing the digits gives a number 27 greater than the original. What is the original number?",
        "a": 47,
    },
    {"q": "Compute 7! divided by 5!.", "a": 42},
    {"q": "A cone has radius 3 and height 4. Compute the slant height.", "a": 5},
    {"q": "How many ways can 4 people be seated in a row?", "a": 24},
    {"q": "Solve for x: (x-2)(x+3) = 0 and x > 0.", "a": 2},
    {
        "q": "A recipe calls for 3 cups of flour for 12 cookies. How many cups are needed for 40 cookies? Give the answer to the nearest whole number.",
        "a": 10,
    },
    {"q": "What is the greatest common divisor of 84 and 126?", "a": 42},
    {"q": "If x + y = 10 and x - y = 4, what is xy?", "a": 21},
    {
        "q": "A ladder 13 feet long leans against a wall, with its foot 5 feet from the wall. How high up the wall does it reach?",
        "a": 12,
    },
    {"q": "The sum of the first n positive integers is 55. What is n?", "a": 10},
]
print(f"Number of tasks: {len(TASKS)}")

print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=True, local_files_only=True
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True
).to(device)
model.eval()
n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"Layers: {n_layers}, hidden: {hidden_size}")

BEHAVIOURS = {
    "hedging": [
        "maybe",
        "perhaps",
        "i'm not sure",
        "not sure",
        "i think",
        "possibly",
        "might be",
        "could be",
    ],
    "backtracking": [
        "wait,",
        "wait ",
        "hmm",
        "actually,",
        "but wait",
        "hold on",
        "let me reconsider",
        "on second thought",
    ],
    "self_correction": [
        "let me verify",
        "let me check",
        "let me double-check",
        "recompute",
        "recalculate",
        "let me recheck",
        "let me confirm",
    ],
    "example_gen": [
        "for example",
        "for instance",
        "let's try",
        "let me try",
        "suppose",
        "consider the case",
        "e.g.",
    ],
}


def build_prompt(q):
    return f"<｜begin▁of▁sentence｜><｜User｜>{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n"


def clean_decode(ids):
    return tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


def extract_think(text):
    end = text.find("</think>")
    if end >= 0:
        return text[:end], text[end + len("</think>") :]
    return text, ""


def split_sentences(txt):
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", txt)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def match_behaviour(sent):
    s = sent.lower()
    return [b for b, kws in BEHAVIOURS.items() if any(kw in s for kw in kws)]


def extract_boxed_answer(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if not m:
        nums = re.findall(r"-?\d+\.?\d*", text)
        return nums[-1] if nums else None
    return m[-1].strip()


def answer_matches(pred, gold):
    if pred is None:
        return False
    try:
        p = float(re.sub(r"[^0-9\.\-]", "", pred))
        return abs(p - float(gold)) < 1e-3
    except:
        return False


# Capture hooks for ALL layers we care about
CAPTURE_LAYERS = list(range(4, n_layers, 4))  # subset of layers for extraction
if (n_layers // 2) not in CAPTURE_LAYERS:
    CAPTURE_LAYERS.append(n_layers // 2)
CAPTURE_LAYERS = sorted(set(CAPTURE_LAYERS))
print(f"Capture layers: {CAPTURE_LAYERS}")

capture_active = {"on": False, "buf": {}}


def make_capture_hook(layer_idx):
    def hook(module, inputs, output):
        if not capture_active["on"]:
            return
        h = output[0] if isinstance(output, tuple) else output
        capture_active["buf"].setdefault(layer_idx, []).append(
            h.detach().to(torch.float32).cpu()
        )

    return hook


capture_handles = []
for li in CAPTURE_LAYERS:
    capture_handles.append(
        model.model.layers[li].register_forward_hook(make_capture_hook(li))
    )

# Steering hook - can attach to any layer dynamically
steer_state = {"on": False, "vec": None, "coeff": 0.0, "layer": n_layers // 2}


def make_steer_hook(layer_idx):
    def hook(module, inputs, output):
        if (
            not steer_state["on"]
            or steer_state["vec"] is None
            or steer_state["layer"] != layer_idx
        ):
            return output
        h = output[0] if isinstance(output, tuple) else output
        v = steer_state["vec"].to(h.device, h.dtype)
        h_new = h + steer_state["coeff"] * v
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


steer_handles = []
for li in range(n_layers):
    steer_handles.append(
        model.model.layers[li].register_forward_hook(make_steer_hook(li))
    )

MAX_NEW_TOKENS = 1536


def generate_chain(prompt, max_new_tokens=MAX_NEW_TOKENS):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            temperature=1.0,
        )
    gen_ids = out[0][enc.input_ids.shape[1] :]
    return clean_decode(gen_ids), gen_ids.cpu(), enc.input_ids.shape[1]


def get_all_layer_activations(full_ids):
    capture_active["on"] = True
    capture_active["buf"] = {}
    with torch.no_grad():
        ids = full_ids.unsqueeze(0).to(device)
        _ = model(ids, use_cache=False)
    capture_active["on"] = False
    return {li: buf[-1][0] for li, buf in capture_active["buf"].items() if buf}


# ============================================================
# STEP 1: Generate baseline chains + capture multi-layer acts
# ============================================================
print("\n### STEP 1: Baseline generation ###")
baseline_chains = []
baseline_correct = 0
t0 = time.time()
for i, task in enumerate(TASKS):
    prompt = build_prompt(task["q"])
    steer_state["on"] = False
    text, gen_ids, plen = generate_chain(prompt)
    think_txt, _ = extract_think(text)
    pred = extract_boxed_answer(text)
    correct = answer_matches(pred, task["a"])
    baseline_correct += int(correct)
    counts = {b: 0 for b in BEHAVIOURS}
    for s in split_sentences(think_txt):
        for b in match_behaviour(s):
            counts[b] += 1
    baseline_chains.append(
        {
            "task_idx": i,
            "text": text,
            "gen_ids": gen_ids,
            "prompt_len": plen,
            "think": think_txt,
            "counts": counts,
            "correct": correct,
            "pred": pred,
        }
    )
    experiment_data["steering_analysis"]["r1_distill_llama_8b"]["predictions"].append(
        pred
    )
    experiment_data["steering_analysis"]["r1_distill_llama_8b"]["ground_truth"].append(
        task["a"]
    )
    if (i + 1) % 5 == 0:
        print(f"  baseline {i+1}/{len(TASKS)} elapsed={time.time()-t0:.1f}s")

baseline_accuracy = baseline_correct / len(TASKS)
print(f"Baseline accuracy: {baseline_accuracy:.3f}")
agg_counts = {b: sum(c["counts"][b] for c in baseline_chains) for b in BEHAVIOURS}
print(f"Aggregate baseline counts: {agg_counts}")
experiment_data["steering_analysis"]["r1_distill_llama_8b"]["baseline"] = {
    "accuracy": baseline_accuracy,
    "counts": agg_counts,
}
print(f"Epoch 0: validation_loss = {1.0 - baseline_accuracy:.4f}")

# ============================================================
# STEP 2: Extract per-sentence activations at ALL capture layers
# ============================================================
print("\n### STEP 2: Extracting multi-layer sentence activations ###")
N_EXTRACT = min(20, len(baseline_chains))
sentences_by_layer = {li: [] for li in CAPTURE_LAYERS}
for idx in range(N_EXTRACT):
    entry = baseline_chains[idx]
    prompt = build_prompt(TASKS[entry["task_idx"]]["q"])
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    full_ids = torch.cat([prompt_ids, entry["gen_ids"]], dim=0)
    steer_state["on"] = False
    acts_by_layer = get_all_layer_activations(full_ids)
    if not acts_by_layer:
        continue
    gen_text_full = clean_decode(entry["gen_ids"])
    gen_encoding = tokenizer(
        gen_text_full, return_offsets_mapping=True, add_special_tokens=False
    )
    offsets = gen_encoding["offset_mapping"]
    n_gen_tokens = len(entry["gen_ids"])
    L = min(n_gen_tokens, len(offsets))
    sent_matches = []
    for m in re.finditer(r"[^\.\!\?\n]+[\.\!\?\n]?", gen_text_full):
        s = m.group().strip()
        if len(s) < 4:
            continue
        sent_matches.append((m.start(), m.end(), s))
    for cs, ce, sent in sent_matches:
        last_tok = None
        for ti in range(L):
            os_, oe_ = offsets[ti]
            if oe_ <= ce and os_ >= cs:
                last_tok = ti
            elif os_ >= ce:
                break
        if last_tok is None:
            continue
        full_idx = prompt_ids.shape[0] + last_tok
        tags = match_behaviour(sent)
        for li in CAPTURE_LAYERS:
            if li in acts_by_layer and full_idx < acts_by_layer[li].shape[0]:
                sentences_by_layer[li].append(
                    {
                        "sent": sent,
                        "vec": acts_by_layer[li][full_idx].numpy(),
                        "tags": tags,
                    }
                )

for li in CAPTURE_LAYERS:
    print(f"  layer {li}: {len(sentences_by_layer[li])} sentence acts")

# ============================================================
# STEP 3: Build steering vectors per-behaviour per-layer
# ============================================================
print("\n### STEP 3: Building steering vectors per layer per behaviour ###")
steering_vectors = {}  # (behaviour, layer) -> vec
pool_sizes = {}
for li in CAPTURE_LAYERS:
    for b in BEHAVIOURS:
        pos = [s["vec"] for s in sentences_by_layer[li] if b in s["tags"]]
        neg = [
            s["vec"]
            for s in sentences_by_layer[li]
            if b not in s["tags"] and len(s["tags"]) == 0
        ]
        pool_sizes[f"L{li}_{b}"] = {"pos": len(pos), "neg": len(neg)}
        if len(pos) >= 3 and len(neg) >= 3:
            k = min(len(pos), len(neg), 60)
            rng = np.random.RandomState(42)
            pi = rng.choice(len(pos), k, replace=False)
            ni = rng.choice(len(neg), k, replace=False)
            mp = np.stack([pos[i] for i in pi]).mean(0)
            mn = np.stack([neg[i] for i in ni]).mean(0)
            v = mp - mn
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v = v / nrm
            steering_vectors[(b, li)] = (torch.tensor(v, dtype=torch.float32), nrm)

experiment_data["steering_analysis"]["r1_distill_llama_8b"][
    "extraction_pool_sizes"
] = pool_sizes

# Compute avg activation norm at middle layer for scale
mid_layer = n_layers // 2
if sentences_by_layer[mid_layer]:
    avg_norm = float(
        np.mean([np.linalg.norm(s["vec"]) for s in sentences_by_layer[mid_layer]])
    )
else:
    avg_norm = 10.0
BASE_SCALE = 0.15 * avg_norm
print(f"avg_norm@mid={avg_norm:.3f}, BASE_SCALE={BASE_SCALE:.3f}")

# ============================================================
# STEP 4: Dose-response curves at middle layer (expanded coeffs)
# ============================================================
print("\n### STEP 4: Dose-response curves ###")
N_TEST = min(15, len(TASKS))
COEFFS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
test_indices = list(range(N_TEST))
dose_response = {}

for b in BEHAVIOURS:
    key = (b, mid_layer)
    if key not in steering_vectors:
        print(f"  skip {b} (no vector)")
        continue
    vec, _ = steering_vectors[key]
    print(f"  === Dose-response: {b} @ layer {mid_layer} ===")
    dose_response[b] = {}
    for coeff in COEFFS:
        counts_all = {bb: [] for bb in BEHAVIOURS}  # cross-interference
        correct = 0
        for ti in test_indices:
            task = TASKS[ti]
            prompt = build_prompt(task["q"])
            if coeff == 0.0:
                think_txt = baseline_chains[ti]["think"]
                is_correct = baseline_chains[ti]["correct"]
            else:
                steer_state["on"] = True
                steer_state["vec"] = vec
                steer_state["coeff"] = coeff * BASE_SCALE
                steer_state["layer"] = mid_layer
                text, gen_ids, _ = generate_chain(prompt)
                steer_state["on"] = False
                steer_state["vec"] = None
                think_txt, _ = extract_think(text)
                pred = extract_boxed_answer(text)
                is_correct = answer_matches(pred, task["a"])
            correct += int(is_correct)
            sents = split_sentences(think_txt)
            for bb in BEHAVIOURS:
                counts_all[bb].append(sum(1 for s in sents if bb in match_behaviour(s)))
        acc = correct / len(test_indices)
        dose_response[b][coeff] = {
            "acc": acc,
            "target_mean": float(np.mean(counts_all[b])),
            "all_means": {bb: float(np.mean(counts_all[bb])) for bb in BEHAVIOURS},
        }
        print(
            f"    coeff={coeff:+.2f}: target={dose_response[b][coeff]['target_mean']:.2f}, acc={acc:.3f}"
        )

experiment_data["steering_analysis"]["r1_distill_llama_8b"]["dose_response"] = {
    b: {str(k): v for k, v in r.items()} for b, r in dose_response.items()
}

# ============================================================
# STEP 5: Compute BSE (Behaviour Steering Efficacy)
# ============================================================
print("\n### STEP 5: BSE metric ###")
bse_scores = {}
baseline_acc_test = sum(baseline_chains[ti]["correct"] for ti in test_indices) / len(
    test_indices
)
print(f"baseline_acc_test = {baseline_acc_test:.3f}")
for b in dose_response:
    r = dose_response[b]
    # Use α=1.0 as canonical
    if 1.0 not in r or -1.0 not in r:
        continue
    f_pos = r[1.0]["target_mean"]
    f_neg = r[-1.0]["target_mean"]
    acc_pos = r[1.0]["acc"]
    acc_neg = r[-1.0]["acc"]
    avg_acc_steered = (acc_pos + acc_neg) / 2.0
    bse = (f_pos - f_neg) / (1.0 + abs(baseline_acc_test - avg_acc_steered))
    # Also compute extended BSE using α=2.0
    if 2.0 in r and -2.0 in r:
        f_pos2 = r[2.0]["target_mean"]
        f_neg2 = r[-2.0]["target_mean"]
        acc_pos2 = r[2.0]["acc"]
        acc_neg2 = r[-2.0]["acc"]
        avg_acc2 = (acc_pos2 + acc_neg2) / 2.0
        bse2 = (f_pos2 - f_neg2) / (1.0 + abs(baseline_acc_test - avg_acc2))
    else:
        bse2 = None
    bse_scores[b] = {
        "bse_a1": bse,
        "bse_a2": bse2,
        "f_pos": f_pos,
        "f_neg": f_neg,
        "acc_pos": acc_pos,
        "acc_neg": acc_neg,
    }
    print(
        f"  [{b}] BSE(α=1)={bse:.3f}, BSE(α=2)={bse2}, f+={f_pos:.2f}, f-={f_neg:.2f}"
    )

experiment_data["steering_analysis"]["r1_distill_llama_8b"]["bse_scores"] = bse_scores

# ============================================================
# STEP 6: Per-layer steering ablation for HEDGING (best behaviour)
# ============================================================
print("\n### STEP 6: Per-layer ablation for hedging ###")
layer_ablation = {"hedging": {}}
LAYERS_TO_TEST = [li for li in CAPTURE_LAYERS if ("hedging", li) in steering_vectors]
# reduce to <=5 layers for compute
if len(LAYERS_TO_TEST) > 5:
    step = len(LAYERS_TO_TEST) // 5
    LAYERS_TO_TEST = LAYERS_TO_TEST[::step][:5]
print(f"Testing layers: {LAYERS_TO_TEST}")
N_ABL = min(8, N_TEST)
for li in LAYERS_TO_TEST:
    vec, _ = steering_vectors[("hedging", li)]
    per_coeff = {}
    for coeff in [-1.0, 1.0]:
        cnts = []
        cor = 0
        for ti in range(N_ABL):
            task = TASKS[ti]
            prompt = build_prompt(task["q"])
            steer_state["on"] = True
            steer_state["vec"] = vec
            steer_state["coeff"] = coeff * BASE_SCALE
            steer_state["layer"] = li
            text, _, _ = generate_chain(prompt)
            steer_state["on"] = False
            steer_state["vec"] = None
            think_txt, _ = extract_think(text)
            pred = extract_boxed_answer(text)
            cor += int(answer_matches(pred, task["a"]))
            sents = split_sentences(think_txt)
            cnts.append(sum(1 for s in sents if "hedging" in match_behaviour(s)))
        per_coeff[coeff] = {"mean": float(np.mean(cnts)), "acc": cor / N_ABL}
    delta = per_coeff[1.0]["mean"] - per_coeff[-1.0]["mean"]
    avg_acc = (per_coeff[1.0]["acc"] + per_coeff[-1.0]["acc"]) / 2.0
    bse_li = delta / (1.0 + abs(baseline_acc_test - avg_acc))
    layer_ablation["hedging"][li] = {
        "delta": delta,
        "bse": bse_li,
        "acc_pos": per_coeff[1.0]["acc"],
        "acc_neg": per_coeff[-1.0]["acc"],
    }
    print(
        f"  layer {li}: delta={delta:.2f}, BSE={bse_li:.3f}, acc+={per_coeff[1.0]['acc']:.2f}, acc-={per_coeff[-1.0]['acc']:.2f}"
    )

experiment_data["steering_analysis"]["r1_distill_llama_8b"]["layer_ablation"] = {
    b: {str(k): v for k, v in r.items()} for b, r in layer_ablation.items()
}

# ============================================================
# STEP 7: Cross-behaviour interference (already partially in dose_response)
# ============================================================
print("\n### STEP 7: Cross-behaviour interference ###")
interference = {}
for src_b in dose_response:
    row = {}
    r_pos = dose_response[src_b].get(1.0)
    r_neg = dose_response[src_b].get(-1.0)
    if r_pos is None or r_neg is None:
        continue
    for tgt_b in BEHAVIOURS:
        d = r_pos["all_means"][tgt_b] - r_neg["all_means"][tgt_b]
        row[tgt_b] = d
    interference[src_b] = row
    print(f"  steering {src_b}: {row}")

experiment_data["steering_analysis"]["r1_distill_llama_8b"][
    "cross_interference"
] = interference

# Record training metrics (per-epoch aliases)
for i, b in enumerate(bse_scores):
    experiment_data["steering_analysis"]["r1_distill_llama_8b"]["metrics"][
        "val"
    ].append({"behaviour": b, "bse": bse_scores[b]["bse_a1"]})
    experiment_data["steering_analysis"]["r1_distill_llama_8b"]["losses"]["val"].append(
        {"behaviour": b, "val_loss": -bse_scores[b]["bse_a1"]}
    )
    print(
        f"Epoch {i+1}: validation_loss = {-bse_scores[b]['bse_a1']:.4f} (behaviour={b}, BSE)"
    )

# ============================================================
# PLOTS
# ============================================================
try:
    # Dose-response curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in dose_response:
        cs = sorted(dose_response[b].keys())
        ys = [dose_response[b][c]["target_mean"] for c in cs]
        ax.plot(cs, ys, "o-", label=b)
    ax.set_xlabel("coefficient (× BASE_SCALE)")
    ax.set_ylabel("Mean behaviour count in CoT")
    ax.set_title(f"Dose-response curves (layer {mid_layer})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "dose_response_curves.png"))
    plt.close()

    # Accuracy vs coefficient
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in dose_response:
        cs = sorted(dose_response[b].keys())
        ys = [dose_response[b][c]["acc"] for c in cs]
        ax.plot(cs, ys, "s-", label=b)
    ax.axhline(baseline_acc_test, color="k", ls="--", label="baseline")
    ax.set_xlabel("coefficient")
    ax.set_ylabel("Task accuracy")
    ax.set_title("Accuracy vs coefficient")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "accuracy_vs_coefficient.png"))
    plt.close()

    # BSE scores
    if bse_scores:
        fig, ax = plt.subplots(figsize=(7, 4))
        bs = list(bse_scores.keys())
        ys = [bse_scores[b]["bse_a1"] for b in bs]
        ax.bar(bs, ys, color="steelblue")
        ax.set_ylabel("BSE (α=1)")
        ax.set_title("Behaviour Steering Efficacy")
        ax.axhline(0, color="k", lw=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "bse_scores.png"))
        plt.close()

    # Layer ablation for hedging
    if layer_ablation["hedging"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        lis = sorted(layer_ablation["hedging"].keys())
        deltas = [layer_ablation["hedging"][li]["delta"] for li in lis]
        bses = [layer_ablation["hedging"][li]["bse"] for li in lis]
        ax.plot(lis, deltas, "o-", label="Δ freq (+ vs -)")
        ax.plot(lis, bses, "s-", label="BSE")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Value")
        ax.set_title("Per-layer steering ablation: hedging")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "layer_ablation_hedging.png"))
        plt.close()

    # Cross-interference heatmap
    if interference:
        src = list(interference.keys())
        tgt = list(BEHAVIOURS.keys())
        M = np.array([[interference[s][t] for t in tgt] for s in src])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            M,
            cmap="RdBu_r",
            vmin=-np.max(np.abs(M) + 1e-6),
            vmax=np.max(np.abs(M) + 1e-6),
        )
        ax.set_xticks(range(len(tgt)))
        ax.set_xticklabels(tgt, rotation=30)
        ax.set_yticks(range(len(src)))
        ax.set_yticklabels(src)
        ax.set_xlabel("Target behaviour (measured)")
        ax.set_ylabel("Source behaviour (steered)")
        ax.set_title("Cross-behaviour interference (Δ = f(+) - f(-))")
        for i in range(len(src)):
            for j in range(len(tgt)):
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "cross_interference_heatmap.png"))
        plt.close()
except Exception as e:
    print(f"plot error: {e}")

for h in capture_handles + steer_handles:
    h.remove()

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("\nSaved experiment_data.npy")
print(f"FINAL baseline accuracy: {baseline_accuracy:.3f}")
print(f"FINAL BSE scores: {[(b, bse_scores[b]['bse_a1']) for b in bse_scores]}")
