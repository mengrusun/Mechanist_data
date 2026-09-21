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

PROP_NAMES = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def make_pair_2step():
    props = random.sample(PROP_NAMES, 3)
    A, B, C = props
    clean = "\n".join(
        [
            "Facts:",
            f"- {A} is True.",
            "Rules:",
            f"- If {A} is True, then {B} is True.",
            f"- If {B} is True, then {C} is True.",
            f"Question: Is {C} True? Answer with only True or False.",
            "Answer:",
        ]
    )
    corr = "\n".join(
        [
            "Facts:",
            f"- {A} is False.",
            "Rules:",
            f"- If {A} is True, then {B} is True.",
            f"- If {B} is True, then {C} is True.",
            f"Question: Is {C} True? Answer with only True or False.",
            "Answer:",
        ]
    )
    return clean, corr, True, False


def make_pair_3step():
    props = random.sample(PROP_NAMES, 4)
    A, B, C, D = props
    clean = "\n".join(
        [
            "Facts:",
            f"- {A} is True.",
            "Rules:",
            f"- If {A} is True, then {B} is True.",
            f"- If {B} is True, then {C} is True.",
            f"- If {C} is True, then {D} is True.",
            f"Question: Is {D} True? Answer with only True or False.",
            "Answer:",
        ]
    )
    corr = "\n".join(
        [
            "Facts:",
            f"- {A} is False.",
            "Rules:",
            f"- If {A} is True, then {B} is True.",
            f"- If {B} is True, then {C} is True.",
            f"- If {C} is True, then {D} is True.",
            f"Question: Is {D} True? Answer with only True or False.",
            "Answer:",
        ]
    )
    return clean, corr, True, False


def make_pair_conjunction():
    props = random.sample(PROP_NAMES, 3)
    A, B, C = props
    clean = "\n".join(
        [
            "Facts:",
            f"- {A} is True.",
            f"- {B} is True.",
            "Rules:",
            f"- If {A} is True and {B} is True, then {C} is True.",
            f"Question: Is {C} True? Answer with only True or False.",
            "Answer:",
        ]
    )
    corr = "\n".join(
        [
            "Facts:",
            f"- {A} is True.",
            f"- {B} is False.",
            "Rules:",
            f"- If {A} is True and {B} is True, then {C} is True.",
            f"Question: Is {C} True? Answer with only True or False.",
            "Answer:",
        ]
    )
    return clean, corr, True, False


def make_pair_negation():
    # Modus tollens: A->B; B is False; therefore A is False (query A, label False)
    # Corrupted: B is True => A could be True (not directly derivable, but semantically no conclusion => model likely says True given only positive statement)
    props = random.sample(PROP_NAMES, 2)
    A, B = props
    clean = "\n".join(
        [
            "Facts:",
            f"- {B} is False.",
            "Rules:",
            f"- If {A} is True, then {B} is True.",
            f"Question: Is {A} True? Answer with only True or False.",
            "Answer:",
        ]
    )
    corr = "\n".join(
        [
            "Facts:",
            f"- {B} is True.",
            "Rules:",
            f"- If {A} is True, then {B} is True.",
            f"Question: Is {A} True? Answer with only True or False.",
            "Answer:",
        ]
    )
    # clean label False, corr label True (flipped)
    return clean, corr, False, True


DATASET_GENERATORS = {
    "2step_chain": make_pair_2step,
    "3step_chain": make_pair_3step,
    "conjunction": make_pair_conjunction,
    "negation_modus_tollens": make_pair_negation,
}

N_PAIRS = 32
datasets = {}
for name, gen in DATASET_GENERATORS.items():
    random.seed(hash(name) % (2**31))
    pairs = [gen() for _ in range(N_PAIRS)]
    datasets[name] = {
        "clean_prompts": [p[0] for p in pairs],
        "corr_prompts": [p[1] for p in pairs],
        "clean_labels": [p[2] for p in pairs],
        "corr_labels": [p[3] for p in pairs],
    }
    print(
        f"\n=== {name} ===\nExample clean:\n{pairs[0][0]}\n---\nExample corr:\n{pairs[0][1]}"
    )

# load model
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


def logit_diff(logits_last, clean_label_is_true=True):
    tl = logits_last[:, true_ids].max(dim=-1).values
    fl = logits_last[:, false_ids].max(dim=-1).values
    # We define logit_diff as (correct - incorrect) for the CLEAN direction.
    # But we'll actually use (True-False) always and interpret sign per dataset.
    return tl - fl


layers = model.model.layers
attn_stores = [None] * n_layers
mlp_stores = [None] * n_layers


def make_attn_hook(layer_idx):
    def hook(module, inp, out):
        x = inp[0]
        xl = x[:, -1, :]
        z = xl.view(xl.size(0), n_heads, d_head)
        attn_stores[layer_idx] = z.detach()

    return hook


def make_mlp_hook(layer_idx):
    def hook(module, inp, out):
        mlp_stores[layer_idx] = out[:, -1, :].detach()

    return hook


patch_state = {"active": False, "kind": None, "layer": None, "head": None, "src": None}


def make_attn_patch_hook(layer_idx):
    def pre_hook(module, inp):
        if (
            not patch_state["active"]
            or patch_state["kind"] != "attn"
            or patch_state["layer"] != layer_idx
        ):
            return None
        x = inp[0]
        B, T, D = x.shape
        x = x.clone()
        xl = x[:, -1, :].view(B, n_heads, d_head)
        h = patch_state["head"]
        src = patch_state["src"]
        if isinstance(h, int):
            xl[:, h, :] = src
        else:
            for hi in h:
                xl[:, hi, :] = src[:, hi, :] if src.dim() == 3 else src
        x[:, -1, :] = xl.view(B, D)
        return (x,) + inp[1:]

    return pre_hook


def make_mlp_patch_hook(layer_idx):
    def hook(module, inp, out):
        if (
            not patch_state["active"]
            or patch_state["kind"] != "mlp"
            or patch_state["layer"] != layer_idx
        ):
            return None
        out = out.clone()
        out[:, -1, :] = patch_state["src"]
        return out

    return hook


circuit_set = None
clean_attn_mean_current = None
clean_mlp_mean_current = None


def make_ablation_attn_hook(layer_idx):
    def pre_hook(module, inp):
        if circuit_set is None:
            return None
        x = inp[0]
        B, T, D = x.shape
        xl = x[:, -1, :].view(B, n_heads, d_head).clone()
        for H in range(n_heads):
            if ("attn", layer_idx, H) not in circuit_set:
                xl[:, H, :] = (
                    clean_attn_mean_current[layer_idx][H].to(x.device).to(x.dtype)
                )
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
            out[:, -1, :] = (
                clean_mlp_mean_current[layer_idx].to(out.device).to(out.dtype)
            )
        return out

    return hook


# Register all hooks once
capture_handles = []
patch_handles = []
ablation_handles = []
for i, layer in enumerate(layers):
    capture_handles.append(
        layer.self_attn.o_proj.register_forward_hook(make_attn_hook(i))
    )
    capture_handles.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))
    patch_handles.append(
        layer.self_attn.o_proj.register_forward_pre_hook(make_attn_patch_hook(i))
    )
    patch_handles.append(layer.mlp.register_forward_hook(make_mlp_patch_hook(i)))
    ablation_handles.append(
        layer.self_attn.o_proj.register_forward_pre_hook(make_ablation_attn_hook(i))
    )
    ablation_handles.append(layer.mlp.register_forward_hook(make_ablation_mlp_hook(i)))

BATCH = 8


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
        for li in range(n_layers):
            attn_dest[li][i : i + len(batch)] = attn_stores[li].float().cpu()
            mlp_dest[li][i : i + len(batch)] = mlp_stores[li].float().cpu()
        logits_all.append(out.logits[:, -1, :].float().cpu())
    return torch.cat(logits_all, dim=0)


def run_corrupted_with_patch(corr_prompts, kind, layer, head, src):
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
        s = src[i : i + len(batch)].to(device).to(model.dtype)
        patch_state["src"] = s
        with torch.no_grad():
            out = model(**enc)
        logits_all.append(out.logits[:, -1, :].float().cpu())
    patch_state["active"] = False
    return torch.cat(logits_all, dim=0)


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


def layer_role(L):
    if L < n_layers / 3:
        return "early_fact_reading"
    if L < 2 * n_layers / 3:
        return "mid_rule_composition"
    return "late_answer_projection"


# Process each dataset: capture activations, run patching, compute effects
per_dataset = {}
Ks = [5, 10, 20, 40, 80, 160]
total_components = n_layers * n_heads + n_layers

for ds_name, ds in datasets.items():
    print(f"\n########## Processing dataset: {ds_name} ##########")
    clean_prompts = ds["clean_prompts"]
    corr_prompts = ds["corr_prompts"]
    clean_label_is_true = ds["clean_labels"][0]  # uniform per dataset

    clean_attn_all = [torch.zeros(N_PAIRS, n_heads, d_head) for _ in range(n_layers)]
    clean_mlp_all = [torch.zeros(N_PAIRS, d_model) for _ in range(n_layers)]
    corr_attn_all = [torch.zeros(N_PAIRS, n_heads, d_head) for _ in range(n_layers)]
    corr_mlp_all = [torch.zeros(N_PAIRS, d_model) for _ in range(n_layers)]

    print("  Capturing clean activations...")
    clean_logits = capture_all(clean_prompts, clean_attn_all, clean_mlp_all)
    print("  Capturing corrupted activations...")
    corr_logits = capture_all(corr_prompts, corr_attn_all, corr_mlp_all)

    clean_ld = logit_diff(clean_logits)
    corr_ld = logit_diff(corr_logits)
    # Sign convention: For "recovery", we want to move from corr to clean.
    # If clean_label is False, clean_ld is negative and corr_ld positive; the formula still works because denom = clean-corr.
    print(f"  Mean clean logit_diff (True-False): {clean_ld.mean():.3f}")
    print(f"  Mean corr  logit_diff (True-False): {corr_ld.mean():.3f}")

    if clean_label_is_true:
        clean_acc = float((clean_ld > 0).float().mean())
        corr_acc = float((corr_ld < 0).float().mean())
    else:
        clean_acc = float((clean_ld < 0).float().mean())
        corr_acc = float((corr_ld > 0).float().mean())
    print(f"  Baseline clean acc={clean_acc:.3f} corr acc={corr_acc:.3f}")

    denom = clean_ld - corr_ld

    head_effects = np.zeros((n_layers, n_heads))
    mlp_effects = np.zeros(n_layers)

    print("  Patching attention heads...")
    for L in range(n_layers):
        for H in range(n_heads):
            src = clean_attn_all[L][:, H, :]
            patched_logits = run_corrupted_with_patch(corr_prompts, "attn", L, H, src)
            p_ld = logit_diff(patched_logits)
            recov = ((p_ld - corr_ld) / (denom + 1e-6)).mean().item()
            head_effects[L, H] = recov
        if L % 8 == 0:
            print(f"    layer {L}: max head recovery so far = {head_effects.max():.3f}")

    print("  Patching MLPs...")
    for L in range(n_layers):
        src = clean_mlp_all[L]
        patched_logits = run_corrupted_with_patch(corr_prompts, "mlp", L, None, src)
        p_ld = logit_diff(patched_logits)
        recov = ((p_ld - corr_ld) / (denom + 1e-6)).mean().item()
        mlp_effects[L] = recov
    print(
        f"  MLP max recovery: {mlp_effects.max():.3f} at layer {mlp_effects.argmax()}"
    )

    components = []
    for L in range(n_layers):
        for H in range(n_heads):
            components.append(("attn", L, H, head_effects[L, H]))
    for L in range(n_layers):
        components.append(("mlp", L, None, mlp_effects[L]))
    components.sort(key=lambda x: -x[3])

    role_counts = {
        "early_fact_reading": 0,
        "mid_rule_composition": 0,
        "late_answer_projection": 0,
    }
    for kind, L, H, eff in components[:20]:
        role_counts[layer_role(L)] += 1

    # Store means for ablation
    clean_attn_mean = [clean_attn_all[L].mean(dim=0) for L in range(n_layers)]
    clean_mlp_mean = [clean_mlp_all[L].mean(dim=0) for L in range(n_layers)]

    per_dataset[ds_name] = {
        "clean_prompts": clean_prompts,
        "corr_prompts": corr_prompts,
        "clean_label_is_true": clean_label_is_true,
        "clean_logits": clean_logits,
        "corr_logits": corr_logits,
        "clean_ld": clean_ld,
        "corr_ld": corr_ld,
        "clean_acc": clean_acc,
        "corr_acc": corr_acc,
        "head_effects": head_effects,
        "mlp_effects": mlp_effects,
        "components": components,
        "role_counts": role_counts,
        "clean_attn_mean": clean_attn_mean,
        "clean_mlp_mean": clean_mlp_mean,
    }

# Faithfulness per dataset (own circuit) and cross-dataset faithfulness
faithfulness_matrix = {}  # [circuit_ds][test_ds][K] -> dict
own_faithfulness = {}


def compute_faithfulness_on_test(circuit_components, test_ds_data, K):
    global circuit_set, clean_attn_mean_current, clean_mlp_mean_current
    top_components = circuit_components[:K]
    cs = set()
    for kind, L, H, _ in top_components:
        if kind == "attn":
            cs.add(("attn", L, H))
        else:
            cs.add(("mlp", L))
    circuit_set = cs
    clean_attn_mean_current = test_ds_data["clean_attn_mean"]
    clean_mlp_mean_current = test_ds_data["clean_mlp_mean"]

    abl_clean = run_circuit_only(test_ds_data["clean_prompts"])
    abl_corr = run_circuit_only(test_ds_data["corr_prompts"])

    full_clean_pred = (logit_diff(test_ds_data["clean_logits"]) > 0).numpy()
    full_corr_pred = (logit_diff(test_ds_data["corr_logits"]) > 0).numpy()
    full_all_pred = np.concatenate([full_clean_pred, full_corr_pred])

    abl_pred = np.concatenate(
        [(logit_diff(abl_clean) > 0).numpy(), (logit_diff(abl_corr) > 0).numpy()]
    )
    match = float((abl_pred == full_all_pred).mean())
    sparsity = 1.0 - (K / total_components)
    faith = match * sparsity
    circuit_set = None
    return {"match": match, "sparsity": sparsity, "faithfulness": faith}


print("\n########## Computing per-dataset faithfulness ##########")
for ds_name, ds_data in per_dataset.items():
    own_faithfulness[ds_name] = {}
    for K in Ks:
        res = compute_faithfulness_on_test(ds_data["components"], ds_data, K)
        own_faithfulness[ds_name][K] = res
        print(
            f"  {ds_name} K={K}: match={res['match']:.3f} faith={res['faithfulness']:.3f}"
        )

print("\n########## Computing cross-dataset faithfulness ##########")
K_cross = 40  # fixed K for cross-dataset comparison
cross_faith = {}
for c_ds in per_dataset:
    cross_faith[c_ds] = {}
    for t_ds in per_dataset:
        res = compute_faithfulness_on_test(
            per_dataset[c_ds]["components"], per_dataset[t_ds], K_cross
        )
        cross_faith[c_ds][t_ds] = res
        print(
            f"  circuit={c_ds} -> test={t_ds}: match={res['match']:.3f} faith={res['faithfulness']:.3f}"
        )


# Jaccard overlap of top-K components across datasets
def top_k_set(components, K):
    s = set()
    for kind, L, H, _ in components[:K]:
        if kind == "attn":
            s.add(("attn", L, H))
        else:
            s.add(("mlp", L))
    return s


print("\n########## Jaccard overlap ##########")
jaccard = {}
K_jac = 40
ds_names = list(per_dataset.keys())
jaccard_matrix = np.zeros((len(ds_names), len(ds_names)))
for i, a in enumerate(ds_names):
    jaccard[a] = {}
    for j, b in enumerate(ds_names):
        sa = top_k_set(per_dataset[a]["components"], K_jac)
        sb = top_k_set(per_dataset[b]["components"], K_jac)
        inter = len(sa & sb)
        union = len(sa | sb)
        j_val = inter / union if union > 0 else 0.0
        jaccard[a][b] = j_val
        jaccard_matrix[i, j] = j_val
        print(f"  {a} vs {b}: Jaccard@{K_jac} = {j_val:.3f}")

# Jaccard across multiple K values
jaccard_by_K = {}
for K in Ks:
    jaccard_by_K[K] = {}
    for a in ds_names:
        jaccard_by_K[K][a] = {}
        for b in ds_names:
            sa = top_k_set(per_dataset[a]["components"], K)
            sb = top_k_set(per_dataset[b]["components"], K)
            inter = len(sa & sb)
            union = len(sa | sb)
            jaccard_by_K[K][a][b] = (inter / union) if union > 0 else 0.0

# ---------- Assemble experiment_data ----------
experiment_data = {"multi_dataset_circuit_generalization": {}}
for ds_name, ds_data in per_dataset.items():
    clean_ld = ds_data["clean_ld"]
    corr_ld = ds_data["corr_ld"]
    full_all_pred = np.concatenate(
        [
            (logit_diff(ds_data["clean_logits"]) > 0).numpy(),
            (logit_diff(ds_data["corr_logits"]) > 0).numpy(),
        ]
    )
    experiment_data["multi_dataset_circuit_generalization"][ds_name] = {
        "metrics": {
            "train": [],
            "val": [
                {
                    "epoch": 0,
                    "clean_acc": ds_data["clean_acc"],
                    "corr_acc": ds_data["corr_acc"],
                    "mean_clean_logit_diff": float(clean_ld.mean()),
                    "mean_corr_logit_diff": float(corr_ld.mean()),
                }
            ],
        },
        "losses": {"train": [], "val": []},
        "head_effects": ds_data["head_effects"],
        "mlp_effects": ds_data["mlp_effects"],
        "top_components": [
            (k, L, H, float(e)) for (k, L, H, e) in ds_data["components"][:50]
        ],
        "faithfulness": own_faithfulness[ds_name],
        "role_counts_top20": ds_data["role_counts"],
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_head": d_head,
        "total_components": total_components,
        "predictions": full_all_pred.tolist(),
        "ground_truth": (
            ds_data["clean_labels"]
            if False
            else (
                [ds_data["clean_label_is_true"]] * N_PAIRS
                + [not ds_data["clean_label_is_true"]] * N_PAIRS
            )
        ),
        "cross_dataset_faithfulness": cross_faith[
            ds_name
        ],  # circuit from this ds applied to others
        "jaccard_with_others": jaccard[ds_name],
    }

experiment_data["multi_dataset_circuit_generalization"]["_summary"] = {
    "dataset_names": ds_names,
    "jaccard_matrix": jaccard_matrix,
    "jaccard_by_K": jaccard_by_K,
    "cross_faithfulness_K": K_cross,
    "cross_faithfulness_matrix": {
        c: {t: cross_faith[c][t] for t in ds_names} for c in ds_names
    },
    "Ks": Ks,
}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment data to {os.path.join(working_dir, 'experiment_data.npy')}")

# ---------- Plots ----------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(ds_names), figsize=(5 * len(ds_names), 8))
    for idx, ds_name in enumerate(ds_names):
        he = per_dataset[ds_name]["head_effects"]
        me = per_dataset[ds_name]["mlp_effects"]
        vmax = abs(he).max() if abs(he).max() > 0 else 1.0
        im = axes[0, idx].imshow(
            he, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        axes[0, idx].set_title(f"{ds_name}\nhead effects")
        axes[0, idx].set_xlabel("Head")
        axes[0, idx].set_ylabel("Layer")
        plt.colorbar(im, ax=axes[0, idx])
        axes[1, idx].bar(range(n_layers), me)
        axes[1, idx].set_title(f"{ds_name} MLP effects")
        axes[1, idx].set_xlabel("Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "per_dataset_patching.png"), dpi=100)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(jaccard_matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ds_names)))
    ax.set_xticklabels(ds_names, rotation=45, ha="right")
    ax.set_yticks(range(len(ds_names)))
    ax.set_yticklabels(ds_names)
    for i in range(len(ds_names)):
        for j in range(len(ds_names)):
            ax.text(
                j,
                i,
                f"{jaccard_matrix[i,j]:.2f}",
                ha="center",
                va="center",
                color="white",
            )
    ax.set_title(f"Jaccard overlap @ K={K_jac}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "jaccard_matrix.png"), dpi=100)
    plt.close()

    cf_matrix = np.zeros((len(ds_names), len(ds_names)))
    for i, c in enumerate(ds_names):
        for j, t in enumerate(ds_names):
            cf_matrix[i, j] = cross_faith[c][t]["faithfulness"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cf_matrix, cmap="viridis")
    ax.set_xticks(range(len(ds_names)))
    ax.set_xticklabels(ds_names, rotation=45, ha="right")
    ax.set_yticks(range(len(ds_names)))
    ax.set_yticklabels(ds_names)
    ax.set_xlabel("test dataset")
    ax.set_ylabel("circuit source dataset")
    for i in range(len(ds_names)):
        for j in range(len(ds_names)):
            ax.text(
                j, i, f"{cf_matrix[i,j]:.2f}", ha="center", va="center", color="white"
            )
    ax.set_title(f"Cross-dataset faithfulness @ K={K_cross}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "cross_dataset_faithfulness.png"), dpi=100)
    plt.close()
except Exception as e:
    print(f"Plot skipped: {e}")

print("Done.")
