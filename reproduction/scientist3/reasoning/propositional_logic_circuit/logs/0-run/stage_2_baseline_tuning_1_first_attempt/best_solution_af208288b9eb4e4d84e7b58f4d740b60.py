import os
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_DIR = "/data/zhenqian/models"
DATA_DIR = "/data/zhenqian/data"

candidate_paths = [
    os.path.join(MODEL_DIR, "Mistral-7B-v0.1"),
    os.path.join(MODEL_DIR, "Mistral-7B-Instruct-v0.2"),
    os.path.join(MODEL_DIR, "Mistral-7B-Instruct-v0.1"),
    os.path.join(MODEL_DIR, "mistral-7b"),
    os.path.join(MODEL_DIR, "Mistral-7B"),
]
model_path = None
for p in candidate_paths:
    if os.path.isdir(p):
        model_path = p
        break
if model_path is None:
    for name in os.listdir(MODEL_DIR):
        low = name.lower()
        if "mistral" in low and "7b" in low:
            model_path = os.path.join(MODEL_DIR, name)
            break
assert model_path is not None, f"Could not find Mistral-7B under {MODEL_DIR}"
print(f"Using model: {model_path}")

# Hyperparameter tuning setup
MAX_LENGTHS = [256, 512, 1024, 2048]
experiment_data = {"max_length_tuning": {}}
for ml in MAX_LENGTHS:
    experiment_data["max_length_tuning"][str(ml)] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "prompts": [],
        "true_logits": [],
        "false_logits": [],
        "max_length": ml,
    }

# ---------- Synthetic dataset generation ----------
random.seed(0)
np.random.seed(0)

PROP_NAMES = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def generate_example(n_props=4, n_facts=2, n_rules=3, chain_len=2):
    props = random.sample(PROP_NAMES, n_props)
    fact_props = random.sample(props, n_facts)
    fact_vals = {p: random.choice([True, False]) for p in fact_props}
    rules = []
    for _ in range(n_rules):
        x, y = random.sample(props, 2)
        rules.append((x, y))
    known_true = set([p for p, v in fact_vals.items() if v])
    known_false = set([p for p, v in fact_vals.items() if not v])
    changed = True
    while changed:
        changed = False
        for x, y in rules:
            if x in known_true and y not in known_true:
                known_true.add(y)
                if y in known_false:
                    known_false.discard(y)
                changed = True
    determined = list(known_true) + list(known_false)
    if not determined:
        return None
    query = random.choice(determined)
    label = query in known_true
    lines = ["Facts:"]
    for p, v in fact_vals.items():
        lines.append(f"- {p} is {'True' if v else 'False'}.")
    lines.append("Rules:")
    for x, y in rules:
        lines.append(f"- If {x} is True, then {y} is True.")
    lines.append(f"Question: Is {query} True? Answer with only True or False.")
    lines.append("Answer:")
    prompt = "\n".join(lines)
    return prompt, label


def build_dataset(n=200):
    data = []
    tries = 0
    while len(data) < n and tries < n * 20:
        tries += 1
        ex = generate_example(
            n_props=random.choice([3, 4, 5]),
            n_facts=random.choice([2, 3]),
            n_rules=random.choice([2, 3, 4]),
        )
        if ex is None:
            continue
        data.append(ex)
    trues = [d for d in data if d[1]]
    falses = [d for d in data if not d[1]]
    m = min(len(trues), len(falses))
    trues, falses = trues[:m], falses[:m]
    balanced = trues + falses
    random.shuffle(balanced)
    return balanced


dataset = build_dataset(n=240)
print(f"Dataset size (balanced): {len(dataset)}")
print(
    f"Label distribution: True={sum(1 for _,l in dataset if l)}, False={sum(1 for _,l in dataset if not l)}"
)

with open(os.path.join(working_dir, "dataset.jsonl"), "w") as f:
    for prompt, label in dataset:
        f.write(json.dumps({"prompt": prompt, "label": bool(label)}) + "\n")

print("=" * 40)
print("Example prompt:")
print(dataset[0][0])
print(f"Label: {dataset[0][1]}")
print("=" * 40)

# ---------- Load model ----------
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
)
model.to(device)
model.eval()


def get_token_ids_for(word):
    variants = [word, " " + word]
    ids = []
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if len(enc) >= 1:
            ids.append(enc[-1])
    return list(set(ids))


true_ids = get_token_ids_for("True")
false_ids = get_token_ids_for("False")
print(f"True token ids: {true_ids} -> {[tokenizer.decode([t]) for t in true_ids]}")
print(f"False token ids: {false_ids} -> {[tokenizer.decode([t]) for t in false_ids]}")
assert len(true_ids) > 0 and len(false_ids) > 0


def score_prompt_batch(prompts, max_length, batch_size=4):
    all_true_logits = []
    all_false_logits = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits
        last_logits = logits[:, -1, :]
        tl = last_logits[:, true_ids].max(dim=-1).values
        fl = last_logits[:, false_ids].max(dim=-1).values
        all_true_logits.extend(tl.float().cpu().tolist())
        all_false_logits.extend(fl.float().cpu().tolist())
    return all_true_logits, all_false_logits


prompts = [d[0] for d in dataset]
labels = [d[1] for d in dataset]

results_summary = {}
for ml in MAX_LENGTHS:
    print(f"\n===== Running eval with max_length={ml} =====")
    true_logits, false_logits = score_prompt_batch(prompts, max_length=ml, batch_size=4)
    preds = [t > f for t, f in zip(true_logits, false_logits)]
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    accuracy = correct / len(labels)
    tl_arr = np.array(true_logits)
    fl_arr = np.array(false_logits)
    lab_arr = np.array([1 if l else 0 for l in labels])
    stacked = np.stack([fl_arr, tl_arr], axis=1)
    m = stacked.max(axis=1, keepdims=True)
    lse = m.squeeze(1) + np.log(np.exp(stacked - m).sum(axis=1))
    log_probs = stacked - lse[:, None]
    val_loss = float(-log_probs[np.arange(len(lab_arr)), lab_arr].mean())
    print(f"max_length={ml}: acc={accuracy:.4f} val_loss={val_loss:.4f}")

    key = str(ml)
    experiment_data["max_length_tuning"][key]["metrics"]["val"].append(
        {"epoch": 0, "task_accuracy": accuracy}
    )
    experiment_data["max_length_tuning"][key]["losses"]["val"].append(
        {"epoch": 0, "val_loss": val_loss}
    )
    experiment_data["max_length_tuning"][key]["predictions"] = preds
    experiment_data["max_length_tuning"][key]["ground_truth"] = labels
    experiment_data["max_length_tuning"][key]["prompts"] = prompts
    experiment_data["max_length_tuning"][key]["true_logits"] = true_logits
    experiment_data["max_length_tuning"][key]["false_logits"] = false_logits
    results_summary[ml] = (accuracy, val_loss)

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Visualization
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mls = sorted(results_summary.keys())
    accs = [results_summary[m][0] for m in mls]
    losses = [results_summary[m][1] for m in mls]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(mls, accs, "o-")
    ax[0].set_xlabel("max_length")
    ax[0].set_ylabel("accuracy")
    ax[0].set_title("Accuracy vs max_length")
    ax[0].set_xscale("log", base=2)
    ax[1].plot(mls, losses, "o-", color="orange")
    ax[1].set_xlabel("max_length")
    ax[1].set_ylabel("val_loss")
    ax[1].set_title("Val loss vs max_length")
    ax[1].set_xscale("log", base=2)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "max_length_tuning.png"), dpi=100)
    plt.close()
except Exception as e:
    print(f"Plot skipped: {e}")

print("\n===== Summary =====")
for ml in sorted(results_summary.keys()):
    acc, loss = results_summary[ml]
    print(f"max_length={ml}: accuracy={acc:.4f}, val_loss={loss:.4f}")
best_ml = max(results_summary.keys(), key=lambda k: results_summary[k][0])
print(f"Best max_length: {best_ml} with accuracy {results_summary[best_ml][0]:.4f}")
print(f"Saved experiment_data to {os.path.join(working_dir, 'experiment_data.npy')}")
