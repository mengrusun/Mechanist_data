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

import os, json, random, math, gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_DIR = "/data/zhenqian/models"

# ------ find mistral ------
model_path = None
for name in os.listdir(MODEL_DIR):
    low = name.lower()
    if "mistral" in low and "7b" in low:
        model_path = os.path.join(MODEL_DIR, name)
        break
assert model_path is not None
print(f"Using model: {model_path}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ---------- Clean/corrupted paired dataset ----------
# Corrupt = flip the truth value of the query answer by flipping one fact that leads to it.
PROP_NAMES = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def make_pair():
    # simple 2-step chain: A True; A->B; B->C ; query C  (label True)
    # corrupted: A False (still says False literally); rest identical => label False
    props = random.sample(PROP_NAMES, 3)
    A, B, C = props
    lines_clean = [
        "Facts:",
        f"- {A} is True.",
        "Rules:",
        f"- If {A} is True, then {B} is True.",
        f"- If {B} is True, then {C} is True.",
        f"Question: Is {C} True? Answer with only True or False.",
        "Answer:",
    ]
    lines_corr = [
        "Facts:",
        f"- {A} is False.",
        "Rules:",
        f"- If {A} is True, then {B} is True.",
        f"- If {B} is True, then {C} is True.",
        f"Question: Is {C} True? Answer with only True or False.",
        "Answer:",
    ]
    return "\n".join(lines_clean), "\n".join(lines_corr), True, False


N_PAIRS = 32
pairs = [make_pair() for _ in range(N_PAIRS)]
clean_prompts = [p[0] for p in pairs]
corr_prompts = [p[1] for p in pairs]
clean_labels = [p[2] for p in pairs]
corr_labels = [p[3] for p in pairs]

with open(os.path.join(working_dir, "pairs.jsonl"), "w") as f:
    for cp, xp, cl, xl in zip(clean_prompts, corr_prompts, clean_labels, corr_labels):
        f.write(
            json.dumps({"clean": cp, "corr": xp, "clean_label": cl, "corr_label": xl})
            + "\n"
        )

print("Example clean:\n" + clean_prompts[0])
print("---")
print("Example corr:\n" + corr_prompts[0])

# ---------- load model ----------
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

cfg = model.config
n_layers = cfg.num_hidden_layers
n_heads = cfg.num_attention_heads
d_model = cfg.hidden_size
d_head = d_model // n_heads
print(f"n_layers={n_layers} n_heads={n_heads} d_model={d_model} d_head={d_head}")


def get_ids(word):
    ids = set()
    for v in [word, " " + word]:
        e = tokenizer.encode(v, add_special_tokens=False)
        if e:
            ids.add(e[-1])
    return list(ids)


true_ids = get_ids("True")
false_ids = get_ids("False")
print("True ids:", true_ids, "False ids:", false_ids)


def logit_diff(logits_last):
    # scalar: max True logit - max False logit
    tl = logits_last[:, true_ids].max(dim=-1).values
    fl = logits_last[:, false_ids].max(dim=-1).values
    return tl - fl


# ---------- Hook infrastructure ----------
# We hook:
#  - attention output projection input (o_proj input): shape [B, T, d_model], reshaped [B,T,n_heads,d_head]  => per-head z
#  - MLP output (mlp forward output)
# Store activations at final token position only.

layers = model.model.layers  # LLaMA-style

# Storage
clean_head_z = [None] * n_layers  # tensor [B, n_heads, d_head]
clean_mlp_out = [None] * n_layers  # tensor [B, d_model]


def make_attn_hook(layer_idx, store):
    # hook on o_proj: pre-forward input is a tuple (x,)
    def hook(module, inp, out):
        x = inp[0]  # [B, T, d_model]
        # take last token
        xl = x[:, -1, :]  # [B, d_model]
        z = xl.view(xl.size(0), n_heads, d_head)
        store[layer_idx] = z.detach()

    return hook


def make_mlp_hook(layer_idx, store):
    def hook(module, inp, out):
        # out: [B, T, d_model]
        store[layer_idx] = out[:, -1, :].detach()

    return hook


# Patching hooks (used later)
patch_state = {"active": False, "kind": None, "layer": None, "head": None, "src": None}


def make_attn_patch_hook(layer_idx):
    def pre_hook(module, inp):
        if not patch_state["active"]:
            return None
        if patch_state["kind"] != "attn":
            return None
        if patch_state["layer"] != layer_idx:
            return None
        x = inp[0]
        # replace last token, specified head(s), with src
        B, T, D = x.shape
        x = x.clone()
        xl = x[:, -1, :].view(B, n_heads, d_head)
        h = patch_state["head"]
        src = patch_state["src"]  # [B, d_head] or [B, n_heads, d_head]
        if isinstance(h, int):
            xl[:, h, :] = src
        else:
            for hi in h:
                xl[:, hi, :] = src[:, hi, :] if src.dim() == 3 else src
        x[:, -1, :] = xl.view(B, D)
        # need to return new inputs
        new_inp = (x,) + inp[1:]
        return new_inp

    return pre_hook


def make_mlp_patch_hook(layer_idx):
    def hook(module, inp, out):
        if not patch_state["active"]:
            return None
        if patch_state["kind"] != "mlp":
            return None
        if patch_state["layer"] != layer_idx:
            return None
        out = out.clone()
        out[:, -1, :] = patch_state["src"]
        return out

    return hook


# Register capture hooks (permanent, guarded by store variables)
capture_handles = []
patch_handles = []
attn_stores = clean_head_z
mlp_stores = clean_mlp_out

for i, layer in enumerate(layers):
    h1 = layer.self_attn.o_proj.register_forward_hook(make_attn_hook(i, attn_stores))
    h2 = layer.mlp.register_forward_hook(make_mlp_hook(i, mlp_stores))
    capture_handles += [h1, h2]
    # patch hooks always registered but gated by patch_state
    p1 = layer.self_attn.o_proj.register_forward_pre_hook(make_attn_patch_hook(i))
    p2 = layer.mlp.register_forward_hook(make_mlp_patch_hook(i))
    patch_handles += [p1, p2]


def run_and_capture(prompts, capture=True):
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=256
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    return out.logits[:, -1, :]


# ---------- Baseline: full model behavior ----------
BATCH = 8


def batched_last_logits(prompts, patch=False):
    outs = []
    for i in range(0, len(prompts), BATCH):
        batch = prompts[i : i + BATCH]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        outs.append(out.logits[:, -1, :].float().cpu())
    return torch.cat(outs, dim=0)


# We need clean and corrupted activations captured. Run twice storing separately.
clean_attn_all = [torch.zeros(N_PAIRS, n_heads, d_head) for _ in range(n_layers)]
clean_mlp_all = [torch.zeros(N_PAIRS, d_model) for _ in range(n_layers)]
corr_attn_all = [torch.zeros(N_PAIRS, n_heads, d_head) for _ in range(n_layers)]
corr_mlp_all = [torch.zeros(N_PAIRS, d_model) for _ in range(n_layers)]


def capture_all(prompts, attn_dest, mlp_dest):
    logits_all = []
    for i in range(0, len(prompts), BATCH):
        batch = prompts[i : i + BATCH]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        # attn_stores / mlp_stores were filled during forward
        for li in range(n_layers):
            attn_dest[li][i : i + len(batch)] = attn_stores[li].float().cpu()
            mlp_dest[li][i : i + len(batch)] = mlp_stores[li].float().cpu()
        logits_all.append(out.logits[:, -1, :].float().cpu())
    return torch.cat(logits_all, dim=0)


print("Capturing clean activations...")
clean_logits = capture_all(clean_prompts, clean_attn_all, clean_mlp_all)
print("Capturing corrupted activations...")
corr_logits = capture_all(corr_prompts, corr_attn_all, corr_mlp_all)

clean_ld = logit_diff(clean_logits)  # expect positive (True > False)
corr_ld = logit_diff(corr_logits)  # expect negative (False > True)
print(f"Mean clean logit_diff (True-False): {clean_ld.mean():.3f}")
print(f"Mean corr  logit_diff (True-False): {corr_ld.mean():.3f}")

clean_pred_true = (clean_ld > 0).float()
corr_pred_true = (corr_ld > 0).float()
clean_acc = float((clean_pred_true == 1).float().mean())
corr_acc = float((corr_pred_true == 0).float().mean())
print(f"Baseline clean acc={clean_acc:.3f} corr acc={corr_acc:.3f}")

# ---------- Activation patching: substitute CLEAN activation into CORRUPTED run ----------
# Metric: recovery = (patched_ld - corr_ld) / (clean_ld - corr_ld), averaged
denom = clean_ld - corr_ld  # per-example
denom_mean = denom.mean().item()
print(f"Mean (clean-corr) logit-diff gap: {denom_mean:.3f}")


def run_corrupted_with_patch(kind, layer, head, src):
    """Run corrupted prompts with a specific patch."""
    patch_state["active"] = True
    patch_state["kind"] = kind
    patch_state["layer"] = layer
    patch_state["head"] = head
    logits_all = []
    for i in range(0, len(corr_prompts), BATCH):
        batch = corr_prompts[i : i + BATCH]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        # move src slice to device
        s = src[i : i + len(batch)].to(device).to(model.dtype)
        patch_state["src"] = s
        with torch.no_grad():
            out = model(**enc)
        logits_all.append(out.logits[:, -1, :].float().cpu())
    patch_state["active"] = False
    return torch.cat(logits_all, dim=0)


# Score each attention head
head_effects = np.zeros((n_layers, n_heads))
mlp_effects = np.zeros(n_layers)

print("Patching attention heads...")
for L in range(n_layers):
    for H in range(n_heads):
        src = clean_attn_all[L][:, H, :]  # [N, d_head]
        patched_logits = run_corrupted_with_patch("attn", L, H, src)
        p_ld = logit_diff(patched_logits)
        # per-example recovery, then mean
        recov = ((p_ld - corr_ld) / (denom + 1e-6)).mean().item()
        head_effects[L, H] = recov
    if L % 4 == 0:
        best = np.unravel_index(np.argmax(head_effects[: L + 1]), head_effects.shape)
        print(f"  layer {L}: max recovery so far = {head_effects.max():.3f} at {best}")

print("Patching MLPs...")
for L in range(n_layers):
    src = clean_mlp_all[L]  # [N, d_model]
    patched_logits = run_corrupted_with_patch("mlp", L, None, src)
    p_ld = logit_diff(patched_logits)
    recov = ((p_ld - corr_ld) / (denom + 1e-6)).mean().item()
    mlp_effects[L] = recov
print(f"MLP max recovery: {mlp_effects.max():.3f} at layer {mlp_effects.argmax()}")

# Sort components by effect
components = []
for L in range(n_layers):
    for H in range(n_heads):
        components.append(("attn", L, H, head_effects[L, H]))
for L in range(n_layers):
    components.append(("mlp", L, None, mlp_effects[L]))
components.sort(key=lambda x: -x[3])
print("\nTop 20 components by patching recovery:")
for c in components[:20]:
    print(c)

# ---------- Build minimal circuit by greedy top-K & measure faithfulness ----------
total_components = n_layers * n_heads + n_layers  # heads + mlps

# Precompute clean means over dataset for mean-ablation
clean_attn_mean = [
    clean_attn_all[L].mean(dim=0) for L in range(n_layers)
]  # [n_heads, d_head]
clean_mlp_mean = [clean_mlp_all[L].mean(dim=0) for L in range(n_layers)]  # [d_model]

# "Circuit-only" run: on CLEAN prompts, mean-ablate every component NOT in circuit
# implement via forward hooks that overwrite last-token activations to the mean when component not in circuit set
circuit_set = None  # will hold sets of ("attn",L,H) and ("mlp",L)


def make_ablation_attn_hook(layer_idx):
    def pre_hook(module, inp):
        if circuit_set is None:
            return None
        x = inp[0]
        B, T, D = x.shape
        xl = x[:, -1, :].view(B, n_heads, d_head).clone()
        for H in range(n_heads):
            if ("attn", layer_idx, H) not in circuit_set:
                xl[:, H, :] = clean_attn_mean[layer_idx][H].to(x.device).to(x.dtype)
        new_x = x.clone()
        new_x[:, -1, :] = xl.view(B, D)
        return (new_x,) + inp[1:]

    return pre_hook


def make_ablation_mlp_hook(layer_idx):
    def hook(module, inp, out):
        if circuit_set is None:
            return None
        if ("mlp", layer_idx) not in circuit_set:
            out = out.clone()
            out[:, -1, :] = clean_mlp_mean[layer_idx].to(out.device).to(out.dtype)
        return out

    return hook


# Remove patch hooks and register ablation hooks
for h in patch_handles:
    h.remove()
patch_handles = []

ablation_handles = []
for i, layer in enumerate(layers):
    ah = layer.self_attn.o_proj.register_forward_pre_hook(make_ablation_attn_hook(i))
    mh = layer.mlp.register_forward_hook(make_ablation_mlp_hook(i))
    ablation_handles += [ah, mh]


def run_circuit_only(prompts):
    logits_all = []
    for i in range(0, len(prompts), BATCH):
        batch = prompts[i : i + BATCH]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        logits_all.append(out.logits[:, -1, :].float().cpu())
    return torch.cat(logits_all, dim=0)


# baseline predictions (full model) on clean+corr
full_clean_logits = clean_logits
full_corr_logits = corr_logits
full_clean_pred = (logit_diff(full_clean_logits) > 0).numpy()
full_corr_pred = (logit_diff(full_corr_logits) > 0).numpy()

all_prompts = clean_prompts + corr_prompts
full_all_pred = np.concatenate([full_clean_pred, full_corr_pred])

Ks = [5, 10, 20, 40, 80, 160]
faithfulness_results = {}
for K in Ks:
    top_components = components[:K]
    circuit_set = set()
    for kind, L, H, _ in top_components:
        if kind == "attn":
            circuit_set.add(("attn", L, H))
        else:
            circuit_set.add(("mlp", L))
    # run ablated on both clean and corr
    abl_clean = run_circuit_only(clean_prompts)
    abl_corr = run_circuit_only(corr_prompts)
    abl_pred = np.concatenate(
        [(logit_diff(abl_clean) > 0).numpy(), (logit_diff(abl_corr) > 0).numpy()]
    )
    match = float((abl_pred == full_all_pred).mean())
    sparsity = 1.0 - (K / total_components)
    faithfulness = match * sparsity
    faithfulness_results[K] = {
        "match": match,
        "sparsity": sparsity,
        "faithfulness": faithfulness,
    }
    print(
        f"K={K:3d}: match={match:.3f} sparsity={sparsity:.3f} faithfulness={faithfulness:.3f}"
    )

circuit_set = None  # disable ablation for any later runs


# ---------- Functional-role bucket analysis (layer buckets) ----------
def layer_role(L):
    if L < n_layers / 3:
        return "early_fact_reading"
    if L < 2 * n_layers / 3:
        return "mid_rule_composition"
    return "late_answer_projection"


role_counts = {
    "early_fact_reading": 0,
    "mid_rule_composition": 0,
    "late_answer_projection": 0,
}
for kind, L, H, eff in components[:20]:
    role_counts[layer_role(L)] += 1
print("Role distribution of top-20 components:", role_counts)

# ---------- Save experiment data ----------
experiment_data = {
    "propositional_logic_circuit": {
        "metrics": {
            "train": [],
            "val": [
                {
                    "epoch": 0,
                    "clean_acc": clean_acc,
                    "corr_acc": corr_acc,
                    "mean_clean_logit_diff": float(clean_ld.mean()),
                    "mean_corr_logit_diff": float(corr_ld.mean()),
                }
            ],
        },
        "losses": {"train": [], "val": []},
        "head_effects": head_effects,
        "mlp_effects": mlp_effects,
        "top_components": [(k, L, H, float(e)) for (k, L, H, e) in components[:50]],
        "faithfulness": faithfulness_results,
        "role_counts_top20": role_counts,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_head": d_head,
        "total_components": total_components,
        "predictions": full_all_pred.tolist(),
        "ground_truth": [True] * N_PAIRS + [False] * N_PAIRS,
    }
}

# best K by faithfulness
best_K = max(
    faithfulness_results.keys(), key=lambda k: faithfulness_results[k]["faithfulness"]
)
best = faithfulness_results[best_K]
print(
    f"\nBest K={best_K}: faithfulness={best['faithfulness']:.4f} (match={best['match']:.3f}, sparsity={best['sparsity']:.3f})"
)
print(
    f"Epoch 0: validation_loss = {(-math.log(0.5)):.4f}  (placeholder; task is eval-only)"
)
print(f"circuit_faithfulness_score = {best['faithfulness']:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ---------- Plots ----------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    im = ax[0].imshow(
        head_effects,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-abs(head_effects).max(),
        vmax=abs(head_effects).max(),
    )
    ax[0].set_xlabel("Head")
    ax[0].set_ylabel("Layer")
    ax[0].set_title("Attention head patching recovery")
    plt.colorbar(im, ax=ax[0])
    ax[1].bar(range(n_layers), mlp_effects)
    ax[1].set_xlabel("Layer")
    ax[1].set_ylabel("Recovery")
    ax[1].set_title("MLP patching recovery per layer")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "patching_map.png"), dpi=100)
    plt.close()

    Ks_sorted = sorted(faithfulness_results.keys())
    matches = [faithfulness_results[k]["match"] for k in Ks_sorted]
    faiths = [faithfulness_results[k]["faithfulness"] for k in Ks_sorted]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ks_sorted, matches, "o-", label="match(circuit,full)")
    ax.plot(Ks_sorted, faiths, "s-", label="faithfulness")
    ax.set_xlabel("Circuit size K")
    ax.set_ylabel("score")
    ax.set_title("Circuit faithfulness vs size")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "faithfulness_vs_K.png"), dpi=100)
    plt.close()
except Exception as e:
    print(f"Plot skipped: {e}")

print("Done.")
