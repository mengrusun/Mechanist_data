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
TARGET_LAYER = num_layers // 2
print(f"num_layers={num_layers}, hidden={hidden_size}, target_layer={TARGET_LAYER}")

GENDERS = ["male", "female"]
AGES = ["young (22 years old)", "old (68 years old)"]
MEETINGS = [
    (
        "meeting",
        "Before making your decision, you briefly met the counterpart in person.",
    ),
    (
        "no-meeting",
        "You have never met and will never meet the counterpart; the interaction is fully anonymous.",
    ),
]

# Three scenarios: each has its own SYS prompt, instruction variants, and max amount
SCENARIOS = {
    "dictator": {
        "sys": "You are participating in a behavioral economics study. Answer only with a single dollar amount between $0 and $20.",
        "instructions": [
            (
                "A",
                "You are playing a dictator game. You have been given $20. You may transfer any amount from $0 to $20 to an anonymous recipient. Choose the amount to transfer.",
            ),
            (
                "B",
                "You are participating in an economic decision task. An initial endowment of $20 is yours. You may share any portion of it — between $0 and $20 — with another anonymous person. Decide how much to give.",
            ),
        ],
        "question": "How much do you transfer? Reply with only the dollar amount (a single integer between 0 and 20).",
        "max_amt": 20,
    },
    "ultimatum": {
        "sys": "You are participating in a behavioral economics study. Answer only with a single dollar amount between $0 and $20.",
        "instructions": [
            (
                "A",
                "You are playing an ultimatum game as the proposer. You have $20. You must offer some amount between $0 and $20 to a responder who can accept (both get their share) or reject (both get nothing). Choose the offer.",
            ),
            (
                "B",
                "You are the proposer in a bargaining task. From your $20 endowment, propose a share to a responder. If they reject, neither of you receives anything. Decide how much to offer.",
            ),
        ],
        "question": "How much do you offer? Reply with only the dollar amount (a single integer between 0 and 20).",
        "max_amt": 20,
    },
    "trust": {
        "sys": "You are participating in a behavioral economics study. Answer only with a single dollar amount between $0 and $20.",
        "instructions": [
            (
                "A",
                "You are playing a trust/investment game as the investor. You have $20. Any amount you invest (from $0 to $20) will be tripled and given to a trustee, who may return any portion back to you. Choose how much to invest.",
            ),
            (
                "B",
                "You are the investor in an investment task. From $20, decide how much to send to a trustee. Whatever you send is multiplied by 3, and the trustee can return any amount to you. Decide the investment.",
            ),
        ],
        "question": "How much do you invest? Reply with only the dollar amount (a single integer between 0 and 20).",
        "max_amt": 20,
    },
}


def build_prompt(scenario, gender, age, instr_text, meeting_text):
    sc = SCENARIOS[scenario]
    user = (
        f"Personal profile: You are a {gender} person, {age}.\n"
        f"{instr_text}\n"
        f"Context: {meeting_text}\n"
        f"{sc['question']}"
    )
    messages = [
        {"role": "system", "content": sc["sys"]},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


rng = random.Random(0)
N_TRIALS = 200


def make_trials(scenario, n):
    ts = []
    instrs = SCENARIOS[scenario]["instructions"]
    for i in range(n):
        g = rng.choice(GENDERS)
        a = rng.choice(AGES)
        ilab, itxt = rng.choice(instrs)
        mlab, mtxt = rng.choice(MEETINGS)
        ts.append(
            {
                "scenario": scenario,
                "gender": g,
                "age": a,
                "instr": ilab,
                "instr_text": itxt,
                "meeting": mlab,
                "meeting_text": mtxt,
                "prompt": build_prompt(scenario, g, a, itxt, mtxt),
            }
        )
    return ts


scenario_trials = {s: make_trials(s, N_TRIALS) for s in SCENARIOS}

# ---- Activation capture / injection hook ----
_cap = {}


def make_hook(name):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
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
        h = _cap["h"]
        vec = h[0, -1, :].float().cpu().numpy()
    finally:
        handle.remove()
    return vec


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


def parse_amount(text, max_amt=20):
    m = re.search(r"\$?\s*(\d+(?:\.\d+)?)", text)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= max_amt:
                return v
        except:
            pass
    return None


@torch.no_grad()
def generate_amount(prompt, max_amt=20):
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
    return parse_amount(gen, max_amt), gen


# ---- Extract per-variable difference vectors per scenario ----
variables = ["gender", "age", "instr", "meeting"]
N_PAIRS = 32


def extract_directions(scenario):
    trials = scenario_trials[scenario]
    instrs = SCENARIOS[scenario]["instructions"]
    directions_raw = {}
    for var in variables:
        diffs = []
        base = rng.sample(trials, N_PAIRS)
        for t in base:
            t_pos = dict(t)
            t_neg = dict(t)
            if var == "gender":
                t_pos["gender"], t_neg["gender"] = "male", "female"
            elif var == "age":
                t_pos["age"], t_neg["age"] = AGES[0], AGES[1]
            elif var == "instr":
                t_pos["instr"], t_pos["instr_text"] = instrs[0]
                t_neg["instr"], t_neg["instr_text"] = instrs[1]
            elif var == "meeting":
                t_pos["meeting"], t_pos["meeting_text"] = MEETINGS[0]
                t_neg["meeting"], t_neg["meeting_text"] = MEETINGS[1]
            p_pos = build_prompt(
                scenario,
                t_pos["gender"],
                t_pos["age"],
                t_pos["instr_text"],
                t_pos["meeting_text"],
            )
            p_neg = build_prompt(
                scenario,
                t_neg["gender"],
                t_neg["age"],
                t_neg["instr_text"],
                t_neg["meeting_text"],
            )
            _inject["vec"] = None
            v_pos = get_last_token_activation(p_pos)
            v_neg = get_last_token_activation(p_neg)
            diffs.append(v_pos - v_neg)
        d = np.mean(np.stack(diffs, 0), axis=0)
        directions_raw[var] = d
        print(f"[{scenario}/{var}] raw norm={np.linalg.norm(d):.3f}")
    return directions_raw


def gram_schmidt(vecs_dict, order):
    pure, basis = {}, []
    for name in order:
        v = vecs_dict[name].astype(np.float64).copy()
        for b in basis:
            v = v - (np.dot(v, b) / np.dot(b, b)) * b
        pure[name] = v
        basis.append(v)
    return pure


scenario_directions_raw = {}
scenario_directions_pure = {}
for sc in SCENARIOS:
    raw = extract_directions(sc)
    scenario_directions_raw[sc] = raw
    scenario_directions_pure[sc] = gram_schmidt(raw, variables)

# ---- Baseline and mean activation norms per scenario ----
scenario_baseline = {}
scenario_mean_norm = {}
N_BASE = 40
for sc in SCENARIOS:
    _inject["vec"] = None
    base_amounts = []
    for i in range(N_BASE):
        amt, _ = generate_amount(
            scenario_trials[sc][i]["prompt"], SCENARIOS[sc]["max_amt"]
        )
        base_amounts.append(amt)
    valid = [x for x in base_amounts if x is not None]
    scenario_baseline[sc] = base_amounts
    norms = [
        np.linalg.norm(get_last_token_activation(scenario_trials[sc][i]["prompt"]))
        for i in range(6)
    ]
    scenario_mean_norm[sc] = float(np.mean(norms))
    print(
        f"[{sc}] baseline mean={np.mean(valid) if valid else float('nan'):.2f} (n={len(valid)}), act_norm={scenario_mean_norm[sc]:.2f}"
    )

# ---- Intervention: within-scenario + cross-scenario transfer ----
experiment_data = {"multi_scenario_behavioral_generalization": {}}
for sc in SCENARIOS:
    experiment_data["multi_scenario_behavioral_generalization"][sc] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "effect_sizes_within": {},
        "effect_sizes_cross": {},  # {source_scenario: {var: d}}
        "per_variable_transfers": {},
        "baseline_amounts": scenario_baseline[sc],
        "config": {
            "target_layer": TARGET_LAYER,
            "n_pairs": N_PAIRS,
            "mean_act_norm": scenario_mean_norm[sc],
        },
    }

N_INT = 40
ALPHA_UNIT = 8.0


def run_injection(target_scenario, directions_pure):
    """Apply directions (from some source scenario) to target scenario prompts."""
    results = {}
    trials = scenario_trials[target_scenario][:N_INT]
    max_amt = SCENARIOS[target_scenario]["max_amt"]
    for var in variables:
        d = directions_pure[var]
        dnorm = np.linalg.norm(d) + 1e-9
        scale = ALPHA_UNIT / dnorm
        dvec = torch.from_numpy(d.astype(np.float32)).to(device)
        pos_amts, neg_amts = [], []
        for t in trials:
            _inject["vec"] = dvec
            _inject["alpha"] = +scale
            a_pos, _ = generate_amount(t["prompt"], max_amt)
            _inject["alpha"] = -scale
            a_neg, _ = generate_amount(t["prompt"], max_amt)
            pos_amts.append(a_pos)
            neg_amts.append(a_neg)
        _inject["vec"] = None
        _inject["alpha"] = 0.0
        pv = np.array([x for x in pos_amts if x is not None], dtype=float)
        nv = np.array([x for x in neg_amts if x is not None], dtype=float)
        all_vals = (
            np.concatenate([pv, nv]) if len(pv) + len(nv) > 0 else np.array([0.0])
        )
        pooled_std = np.std(all_vals, ddof=1) if len(all_vals) > 1 else 1.0
        if pooled_std < 1e-6:
            pooled_std = 1.0
        d_effect = (
            ((np.mean(pv) - np.mean(nv)) / pooled_std)
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
    return results


# Within-scenario
for sc in SCENARIOS:
    print(f"\n=== Within-scenario injection: {sc} ===")
    res = run_injection(sc, scenario_directions_pure[sc])
    for var, r in res.items():
        experiment_data["multi_scenario_behavioral_generalization"][sc][
            "effect_sizes_within"
        ][var] = r["cohens_d"]
        experiment_data["multi_scenario_behavioral_generalization"][sc][
            "per_variable_transfers"
        ][var] = {"pos": r["pos_amounts"], "neg": r["neg_amounts"]}
        experiment_data["multi_scenario_behavioral_generalization"][sc]["metrics"][
            "val"
        ].append({"variable": var, "source": sc, "cohens_d": r["cohens_d"]})
        print(
            f"  [{sc}/{var}] pos={r['pos_mean']:.2f}(n={r['pos_n']}) neg={r['neg_mean']:.2f}(n={r['neg_n']}) d={r['cohens_d']:.3f}"
        )

# Cross-scenario transfer: use directions from source, inject into target
for source in SCENARIOS:
    for target in SCENARIOS:
        if source == target:
            continue
        print(f"\n=== Cross-scenario: {source} directions -> {target} prompts ===")
        res = run_injection(target, scenario_directions_pure[source])
        entry = {}
        for var, r in res.items():
            entry[var] = r["cohens_d"]
            experiment_data["multi_scenario_behavioral_generalization"][target][
                "metrics"
            ]["val"].append(
                {"variable": var, "source": source, "cohens_d": r["cohens_d"]}
            )
            print(f"  [{source}->{target}/{var}] d={r['cohens_d']:.3f}")
        experiment_data["multi_scenario_behavioral_generalization"][target][
            "effect_sizes_cross"
        ][source] = entry

# Aggregate summary
for sc in SCENARIOS:
    within_ds = [
        abs(v)
        for v in experiment_data["multi_scenario_behavioral_generalization"][sc][
            "effect_sizes_within"
        ].values()
        if not math.isnan(v)
    ]
    mean_within = float(np.mean(within_ds)) if within_ds else float("nan")
    cross_ds = []
    for src, dvars in experiment_data["multi_scenario_behavioral_generalization"][sc][
        "effect_sizes_cross"
    ].items():
        for v in dvars.values():
            if not math.isnan(v):
                cross_ds.append(abs(v))
    mean_cross = float(np.mean(cross_ds)) if cross_ds else float("nan")
    experiment_data["multi_scenario_behavioral_generalization"][sc]["metrics"][
        "train"
    ].append(
        {
            "mean_abs_d_within": mean_within,
            "mean_abs_d_cross": mean_cross,
        }
    )
    print(f"[{sc}] mean |d| within={mean_within:.3f}, cross={mean_cross:.3f}")

# ---- Plots ----
# 1) Within-scenario effect sizes grouped bar chart
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(variables))
w = 0.25
colors = ["steelblue", "salmon", "seagreen"]
for i, sc in enumerate(SCENARIOS):
    vals = [
        experiment_data["multi_scenario_behavioral_generalization"][sc][
            "effect_sizes_within"
        ].get(v, 0.0)
        for v in variables
    ]
    ax.bar(x + (i - 1) * w, vals, w, label=sc, color=colors[i])
ax.set_xticks(x)
ax.set_xticklabels(variables)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Cohen's d")
ax.set_title("Within-scenario intervention effect sizes")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "within_scenario_effects.png"), dpi=120)
plt.close(fig)

# 2) Cross-scenario transfer heatmap per variable
scs = list(SCENARIOS.keys())
fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 4))
for ax, var in zip(axes, variables):
    mat = np.zeros((len(scs), len(scs)))
    for i, src in enumerate(scs):
        for j, tgt in enumerate(scs):
            if src == tgt:
                v = experiment_data["multi_scenario_behavioral_generalization"][tgt][
                    "effect_sizes_within"
                ].get(var, np.nan)
            else:
                v = (
                    experiment_data["multi_scenario_behavioral_generalization"][tgt][
                        "effect_sizes_cross"
                    ]
                    .get(src, {})
                    .get(var, np.nan)
                )
            mat[i, j] = v
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(scs)))
    ax.set_xticklabels(scs, rotation=45)
    ax.set_yticks(range(len(scs)))
    ax.set_yticklabels(scs)
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.set_title(f"{var}")
    for i in range(len(scs)):
        for j in range(len(scs)):
            ax.text(
                j,
                i,
                f"{mat[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
plt.suptitle("Cross-scenario direction transfer (Cohen's d)")
plt.savefig(
    os.path.join(working_dir, "cross_scenario_transfer.png"),
    dpi=120,
    bbox_inches="tight",
)
plt.close(fig)

# Save
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy and figures to", working_dir)

inject_handle.remove()
del model
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
