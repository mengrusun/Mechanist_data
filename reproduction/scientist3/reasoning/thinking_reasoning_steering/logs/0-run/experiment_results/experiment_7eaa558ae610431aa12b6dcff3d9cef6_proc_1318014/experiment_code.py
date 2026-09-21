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
assert os.path.isdir(MODEL_PATH), f"Model directory missing: {MODEL_PATH}"

# Hyperparameter tuning structure
experiment_data = {
    "max_new_tokens": {
        "r1_distill_llama_8b_steering": {
            "values": [],
            "runs": {},  # keyed by max_new_tokens value
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "best_value": None,
            "best_success_rate": None,
        }
    }
}

TASKS = [
    {
        "q": "A store sells apples for $3 each and oranges for $2 each. Alice buys 7 apples and 5 oranges, then returns 2 apples. How much did she spend in total?",
        "a": 21 + 10 - 6,
    },
    {
        "q": "A train travels 60 miles in the first hour, then increases speed by 20 mph for the next 2 hours. What is the total distance traveled?",
        "a": 60 + 160,
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

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=True, local_files_only=True
)
assert tokenizer.is_fast
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
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
    txt = tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    if "Ġ" in txt or "Ċ" in txt:
        txt = txt.replace("Ġ", " ").replace("Ċ", "\n")
    return txt


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
    tags = []
    for b, kws in BEHAVIOURS.items():
        if any(kw in s for kw in kws):
            tags.append(b)
    return tags


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


# Smoke test
print("=== SMOKE TEST ===")
smoke_prompt = build_prompt(TASKS[0]["q"])
enc = tokenizer(smoke_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(
        **enc, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id
    )
gen_ids = out[0][enc.input_ids.shape[1] :]
smoke_text = clean_decode(gen_ids)
assert "Ġ" not in smoke_text and "Ċ" not in smoke_text
print("Smoke test passed.")

STEER_LAYER = n_layers // 2
print(f"Steering layer: {STEER_LAYER}")

capture_active = {"on": False, "buf": None}


def capture_hook(module, inputs, output):
    if not capture_active["on"]:
        return
    h = output[0] if isinstance(output, tuple) else output
    capture_active["buf"].append(h.detach().to(torch.float32).cpu())


target_layer = model.model.layers[STEER_LAYER]
capture_handle = target_layer.register_forward_hook(capture_hook)

steer_state = {"on": False, "vec": None, "coeff": 0.0}


def steer_hook(module, inputs, output):
    if not steer_state["on"] or steer_state["vec"] is None:
        return output
    h = output[0] if isinstance(output, tuple) else output
    v = steer_state["vec"].to(h.device, h.dtype)
    h_new = h + steer_state["coeff"] * v
    if isinstance(output, tuple):
        return (h_new,) + output[1:]
    return h_new


steer_handle = target_layer.register_forward_hook(steer_hook)


def generate_chain(prompt, max_new_tokens=1024):
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
    text = clean_decode(gen_ids)
    return text, gen_ids.cpu(), enc.input_ids.shape[1]


def get_token_activations(full_ids):
    with torch.no_grad():
        capture_active["on"] = True
        capture_active["buf"] = []
        ids = full_ids.unsqueeze(0).to(device)
        _ = model(ids, use_cache=False)
        capture_active["on"] = False
    if not capture_active["buf"]:
        return None
    return capture_active["buf"][-1][0]


# ---------------- Hyperparameter tuning over max_new_tokens ----------------
MAX_NEW_TOKENS_VALUES = [768, 1536, 2048]
N_EXTRACT = 20
N_TEST = 15
COEFFS = [-1.0, 0.0, 1.0]

for MNT in MAX_NEW_TOKENS_VALUES:
    print(f"\n########## Running with max_new_tokens = {MNT} ##########")
    run_data = {
        "baseline_accuracy": None,
        "behaviour_counts": {},
        "steering_results": {},
        "steered_accuracy": {},
        "behaviour_steering_success_rate": None,
        "predictions": [],
        "ground_truth": [],
    }

    baseline_chains = []
    baseline_correct = 0
    baseline_total = 0
    per_task_baseline_counts = []
    t0 = time.time()
    for i, task in enumerate(TASKS):
        prompt = build_prompt(task["q"])
        steer_state["on"] = False
        text, gen_ids, prompt_len = generate_chain(prompt, max_new_tokens=MNT)
        think_txt, _ = extract_think(text)
        pred = extract_boxed_answer(text)
        correct = answer_matches(pred, task["a"])
        baseline_correct += int(correct)
        baseline_total += 1
        counts = {b: 0 for b in BEHAVIOURS}
        for s in split_sentences(think_txt):
            for b in match_behaviour(s):
                counts[b] += 1
        per_task_baseline_counts.append(counts)
        baseline_chains.append(
            {
                "task_idx": i,
                "text": text,
                "gen_ids": gen_ids,
                "prompt_len": prompt_len,
                "think": think_txt,
                "counts": counts,
                "correct": correct,
                "pred": pred,
            }
        )
        run_data["predictions"].append(pred)
        run_data["ground_truth"].append(task["a"])
        if (i + 1) % 10 == 0:
            print(
                f"  [MNT={MNT}] baseline {i+1}/{len(TASKS)} cum={time.time()-t0:.1f}s"
            )

    baseline_accuracy = baseline_correct / baseline_total
    run_data["baseline_accuracy"] = baseline_accuracy
    print(f"[MNT={MNT}] Baseline accuracy: {baseline_accuracy:.3f}")

    agg_counts = {b: 0 for b in BEHAVIOURS}
    for c in per_task_baseline_counts:
        for b in BEHAVIOURS:
            agg_counts[b] += c[b]
    run_data["behaviour_counts"]["baseline_total"] = agg_counts
    print(f"[MNT={MNT}] Aggregate baseline counts: {agg_counts}")

    # Extract per-sentence activations
    all_sentences_info = []
    for idx in range(min(N_EXTRACT, len(baseline_chains))):
        entry = baseline_chains[idx]
        prompt = build_prompt(TASKS[entry["task_idx"]]["q"])
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        full_ids = torch.cat([prompt_ids, entry["gen_ids"]], dim=0)
        steer_state["on"] = False
        acts = get_token_activations(full_ids)
        if acts is None:
            continue
        gen_text_full = clean_decode(entry["gen_ids"])
        sent_matches = []
        for m in re.finditer(r"[^\.\!\?\n]+[\.\!\?\n]?", gen_text_full):
            s = m.group().strip()
            if len(s) < 4:
                continue
            sent_matches.append((m.start(), m.end(), s))
        gen_encoding = tokenizer(
            gen_text_full, return_offsets_mapping=True, add_special_tokens=False
        )
        offsets = gen_encoding["offset_mapping"]
        n_gen_tokens = len(entry["gen_ids"])
        L = min(n_gen_tokens, len(offsets))
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
            if full_idx >= acts.shape[0]:
                continue
            vec = acts[full_idx].numpy()
            tags = match_behaviour(sent)
            all_sentences_info.append({"sent": sent, "vec": vec, "tags": tags})
    print(f"[MNT={MNT}] Collected {len(all_sentences_info)} sentence activations")

    pos_pools = {b: [] for b in BEHAVIOURS}
    neg_pools = {b: [] for b in BEHAVIOURS}
    for b in BEHAVIOURS:
        pos = [s["vec"] for s in all_sentences_info if b in s["tags"]]
        neg = [
            s["vec"]
            for s in all_sentences_info
            if b not in s["tags"] and len(s["tags"]) == 0
        ]
        pos_pools[b] = pos
        neg_pools[b] = neg
        run_data["behaviour_counts"][f"{b}_pos"] = len(pos)
        run_data["behaviour_counts"][f"{b}_neg"] = len(neg)
        print(f"  [{b}] pos={len(pos)}, neg={len(neg)}")

    steering_vectors = {}
    for b in BEHAVIOURS:
        if len(pos_pools[b]) >= 3 and len(neg_pools[b]) >= 3:
            k = min(len(pos_pools[b]), len(neg_pools[b]), 50)
            rng = np.random.RandomState(0)
            p_idx = rng.choice(len(pos_pools[b]), k, replace=False)
            n_idx = rng.choice(len(neg_pools[b]), k, replace=False)
            mp = np.stack([pos_pools[b][i] for i in p_idx]).mean(0)
            mn = np.stack([neg_pools[b][i] for i in n_idx]).mean(0)
            v = mp - mn
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v = v / nrm
            steering_vectors[b] = torch.tensor(v, dtype=torch.float32)
            print(f"  [{b}] steering vector norm(pre)={nrm:.3f}")

    avg_norm = (
        float(np.mean([np.linalg.norm(s["vec"]) for s in all_sentences_info]))
        if all_sentences_info
        else 10.0
    )
    BASE_SCALE = 0.15 * avg_norm
    print(f"[MNT={MNT}] avg_norm={avg_norm:.3f}, BASE_SCALE={BASE_SCALE:.3f}")

    test_indices = list(range(N_TEST))
    steering_results = {}
    for b, vec in steering_vectors.items():
        print(f"  === Steering: {b} ===")
        steering_results[b] = {}
        for coeff in COEFFS:
            counts_list = []
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
                    text, gen_ids, _ = generate_chain(prompt, max_new_tokens=MNT)
                    steer_state["on"] = False
                    steer_state["vec"] = None
                    think_txt, _ = extract_think(text)
                    pred = extract_boxed_answer(text)
                    is_correct = answer_matches(pred, task["a"])
                correct += int(is_correct)
                cnt = sum(
                    1 for s in split_sentences(think_txt) if b in match_behaviour(s)
                )
                counts_list.append(cnt)
            acc = correct / len(test_indices)
            total_cnt = sum(counts_list)
            mean_cnt = total_cnt / len(test_indices)
            steering_results[b][coeff] = {
                "counts": counts_list,
                "total": total_cnt,
                "mean": mean_cnt,
                "acc": acc,
            }
            print(
                f"    coeff={coeff:+.1f}: total={total_cnt}, mean={mean_cnt:.2f}, acc={acc:.3f}"
            )

    run_data["steering_results"] = {
        b: {str(k): v for k, v in r.items()} for b, r in steering_results.items()
    }

    successes = 0
    attempts = 0
    baseline_acc_test = sum(
        baseline_chains[ti]["correct"] for ti in test_indices
    ) / len(test_indices)
    print(f"[MNT={MNT}] Baseline acc on test subset: {baseline_acc_test:.3f}")
    for b in steering_vectors:
        attempts += 1
        r = steering_results[b]
        m_pos, m_neg, m_zero = r[1.0]["mean"], r[-1.0]["mean"], r[0.0]["mean"]
        a_pos, a_neg = r[1.0]["acc"], r[-1.0]["acc"]
        up = m_pos > m_zero
        dn = m_neg < m_zero
        acc_ok = (abs(a_pos - baseline_acc_test) <= 0.10) and (
            abs(a_neg - baseline_acc_test) <= 0.10
        )
        success = up and dn and acc_ok
        print(f"  [{b}] up={up} dn={dn} acc_ok={acc_ok} -> {success}")
        if success:
            successes += 1

    rate = successes / max(1, attempts) if attempts > 0 else 0.0
    run_data["behaviour_steering_success_rate"] = rate
    print(f"[MNT={MNT}] success_rate = {rate:.3f} ({successes}/{attempts})")

    experiment_data["max_new_tokens"]["r1_distill_llama_8b_steering"]["values"].append(
        MNT
    )
    experiment_data["max_new_tokens"]["r1_distill_llama_8b_steering"]["runs"][
        str(MNT)
    ] = run_data
    experiment_data["max_new_tokens"]["r1_distill_llama_8b_steering"]["metrics"][
        "val"
    ].append(
        {
            "max_new_tokens": MNT,
            "success_rate": rate,
            "baseline_accuracy": baseline_accuracy,
        }
    )
    experiment_data["max_new_tokens"]["r1_distill_llama_8b_steering"]["losses"][
        "val"
    ].append({"max_new_tokens": MNT, "val_loss": 1.0 - rate})

    gc.collect()
    torch.cuda.empty_cache()

# Determine best
runs = experiment_data["max_new_tokens"]["r1_distill_llama_8b_steering"]["runs"]
best_val, best_rate = None, -1.0
for k, v in runs.items():
    if v["behaviour_steering_success_rate"] > best_rate:
        best_rate = v["behaviour_steering_success_rate"]
        best_val = int(k)
experiment_data["max_new_tokens"]["r1_distill_llama_8b_steering"][
    "best_value"
] = best_val
experiment_data["max_new_tokens"]["r1_distill_llama_8b_steering"][
    "best_success_rate"
] = best_rate
print(f"\n=== Best max_new_tokens: {best_val} with success_rate {best_rate:.3f} ===")

# Aggregate predictions/ground_truth from best run
if best_val is not None:
    best_run = runs[str(best_val)]
    experiment_data["max_new_tokens"]["r1_distill_llama_8b_steering"]["predictions"] = (
        best_run["predictions"]
    )
    experiment_data["max_new_tokens"]["r1_distill_llama_8b_steering"][
        "ground_truth"
    ] = best_run["ground_truth"]

# ---------------- Plots ----------------
try:
    mnts = sorted([int(k) for k in runs.keys()])
    rates = [runs[str(m)]["behaviour_steering_success_rate"] for m in mnts]
    accs = [runs[str(m)]["baseline_accuracy"] for m in mnts]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mnts, rates, "o-", label="success rate")
    ax.plot(mnts, accs, "s-", label="baseline accuracy")
    ax.set_xlabel("max_new_tokens")
    ax.set_ylabel("value")
    ax.set_title("Hyperparameter tuning: max_new_tokens")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "max_new_tokens_tuning.png"))
    plt.close()

    for m in mnts:
        sr = runs[str(m)]["steering_results"]
        behs = list(sr.keys())
        if not behs:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(behs))
        w = 0.25
        for i, coeff in enumerate(COEFFS):
            vals = [sr[b][str(coeff)]["mean"] for b in behs]
            ax.bar(x + (i - 1) * w, vals, w, label=f"coeff={coeff}")
        ax.set_xticks(x)
        ax.set_xticklabels(behs, rotation=20)
        ax.set_ylabel("Mean behaviour count")
        ax.set_title(f"Behaviour counts (max_new_tokens={m})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"behaviour_counts_mnt{m}.png"))
        plt.close()
except Exception as e:
    print(f"plot error: {e}")

capture_handle.remove()
steer_handle.remove()

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
print(f"FINAL best max_new_tokens={best_val}, best rate={best_rate:.3f}")
