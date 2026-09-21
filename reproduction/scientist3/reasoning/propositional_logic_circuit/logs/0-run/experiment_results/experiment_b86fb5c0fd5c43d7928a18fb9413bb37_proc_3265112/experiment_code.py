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

# Try to find Mistral-7B locally
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
    # search for a directory containing 'mistral' or 'Mistral'
    for name in os.listdir(MODEL_DIR):
        low = name.lower()
        if "mistral" in low and "7b" in low:
            model_path = os.path.join(MODEL_DIR, name)
            break

assert (
    model_path is not None
), f"Could not find Mistral-7B under {MODEL_DIR}. Contents: {os.listdir(MODEL_DIR)}"
print(f"Using model: {model_path}")

experiment_data = {
    "prop_logic_synth": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "prompts": [],
        "true_logits": [],
        "false_logits": [],
    }
}

# ---------- Synthetic dataset generation ----------
random.seed(0)
np.random.seed(0)

PROP_NAMES = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def generate_example(n_props=4, n_facts=2, n_rules=3, chain_len=2):
    """Generate a propositional-logic problem.
    - Choose n_props atomic propositions.
    - Randomly assign truth values to n_facts of them (given as facts).
    - Generate n_rules implication rules 'if X then Y' where X, Y are propositions.
    - Do forward chaining to derive truth of all derivable props.
    - Pick a query proposition; if its value is derivable/true -> label True, else False.
      We keep queries only when we can definitively determine True or False.
    """
    props = random.sample(PROP_NAMES, n_props)
    # ground-truth truth values (some assigned true, others unknown/false-by-default)
    # We define semantics: a prop is True iff it's stated as fact True or derivable via rules from True facts.
    fact_props = random.sample(props, n_facts)
    fact_vals = {p: random.choice([True, False]) for p in fact_props}

    # generate rules: X -> Y
    rules = []
    for _ in range(n_rules):
        x, y = random.sample(props, 2)
        rules.append((x, y))

    # Forward chaining under closed-world assumption on facts:
    # start: known True set from facts where val=True; known False from facts where val=False.
    # Rule X->Y: if X True then Y True. (contrapositive not used for simplicity)
    known_true = set([p for p, v in fact_vals.items() if v])
    known_false = set([p for p, v in fact_vals.items() if not v])
    changed = True
    while changed:
        changed = False
        for x, y in rules:
            if x in known_true and y not in known_true:
                known_true.add(y)
                if y in known_false:
                    known_false.discard(y)  # true overrides
                changed = True

    # Pick query: prefer one that's determined
    determined = list(known_true) + list(known_false)
    if not determined:
        return None
    query = random.choice(determined)
    label = query in known_true

    # Build prompt
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
    # balance True/False roughly
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

# Save dataset
with open(os.path.join(working_dir, "dataset.jsonl"), "w") as f:
    for prompt, label in dataset:
        f.write(json.dumps({"prompt": prompt, "label": bool(label)}) + "\n")

# Show one example
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


# ---------- Token ID selection with sanity checks ----------
def get_token_ids_for(word):
    """Return candidate token ids for a word, handling leading space variants."""
    variants = [word, " " + word]
    ids = []
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        # Take the last token to skip leading-space marker tokens
        if len(enc) >= 1:
            ids.append(enc[-1])
    # Deduplicate
    return list(set(ids))


true_ids = get_token_ids_for("True")
false_ids = get_token_ids_for("False")
print(f"True token ids: {true_ids} -> {[tokenizer.decode([t]) for t in true_ids]}")
print(f"False token ids: {false_ids} -> {[tokenizer.decode([t]) for t in false_ids]}")

assert len(true_ids) > 0 and len(false_ids) > 0


# ---------- Evaluation ----------
def score_prompt_batch(prompts, batch_size=4):
    all_true_logits = []
    all_false_logits = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits  # [B, T, V]
        # With left padding, the last token position is the last index
        last_logits = logits[:, -1, :]  # [B, V]
        # max over variants
        tl = last_logits[:, true_ids].max(dim=-1).values
        fl = last_logits[:, false_ids].max(dim=-1).values
        all_true_logits.extend(tl.float().cpu().tolist())
        all_false_logits.extend(fl.float().cpu().tolist())
    return all_true_logits, all_false_logits


# Sanity check on 2 examples
print("Sanity check on 2 examples:")
sample_prompts = [dataset[0][0], dataset[1][0]]
sample_labels = [dataset[0][1], dataset[1][1]]
tl, fl = score_prompt_batch(sample_prompts, batch_size=2)
for p, lab, t, f in zip(sample_prompts, sample_labels, tl, fl):
    print(
        f"  label={lab} | True_logit={t:.3f} False_logit={f:.3f} pred={'True' if t>f else 'False'}"
    )

# Full eval
prompts = [d[0] for d in dataset]
labels = [d[1] for d in dataset]

print(f"Running full evaluation on {len(prompts)} examples...")
true_logits, false_logits = score_prompt_batch(prompts, batch_size=4)

preds = [t > f for t, f in zip(true_logits, false_logits)]
correct = sum(1 for p, l in zip(preds, labels) if p == l)
accuracy = correct / len(labels)
print(f"Task accuracy: {accuracy:.4f} ({correct}/{len(labels)})")

# Compute a pseudo "val_loss" as -log P(correct) via softmax over {True,False}
tl_arr = np.array(true_logits)
fl_arr = np.array(false_logits)
lab_arr = np.array([1 if l else 0 for l in labels])
stacked = np.stack([fl_arr, tl_arr], axis=1)  # [N,2] index 0=False,1=True
# log-softmax
m = stacked.max(axis=1, keepdims=True)
lse = m.squeeze(1) + np.log(np.exp(stacked - m).sum(axis=1))
log_probs = stacked - lse[:, None]
val_loss = float(-log_probs[np.arange(len(lab_arr)), lab_arr].mean())
print(f"Epoch 0: validation_loss = {val_loss:.4f}")

# Save results
experiment_data["prop_logic_synth"]["metrics"]["val"].append(
    {"epoch": 0, "task_accuracy": accuracy}
)
experiment_data["prop_logic_synth"]["losses"]["val"].append(
    {"epoch": 0, "val_loss": val_loss}
)
experiment_data["prop_logic_synth"]["predictions"] = preds
experiment_data["prop_logic_synth"]["ground_truth"] = labels
experiment_data["prop_logic_synth"]["prompts"] = prompts
experiment_data["prop_logic_synth"]["true_logits"] = true_logits
experiment_data["prop_logic_synth"]["false_logits"] = false_logits

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Simple visualization
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    diffs = tl_arr - fl_arr
    ax[0].hist(diffs[lab_arr == 1], bins=20, alpha=0.6, label="True label")
    ax[0].hist(diffs[lab_arr == 0], bins=20, alpha=0.6, label="False label")
    ax[0].axvline(0, color="k", linestyle="--")
    ax[0].set_xlabel("True_logit - False_logit")
    ax[0].set_ylabel("count")
    ax[0].legend()
    ax[0].set_title(f"Logit gap by label (acc={accuracy:.3f})")
    ax[1].bar(["accuracy"], [accuracy])
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Task Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "prop_logic_baseline.png"), dpi=100)
    plt.close()
except Exception as e:
    print(f"Plot skipped: {e}")

print(f"Final task_accuracy = {accuracy:.4f}")
print(f"Saved experiment_data to {os.path.join(working_dir, 'experiment_data.npy')}")
