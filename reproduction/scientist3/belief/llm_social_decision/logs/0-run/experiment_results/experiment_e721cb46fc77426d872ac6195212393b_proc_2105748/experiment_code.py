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
LAYERS = [num_layers // 4, num_layers // 2, (3 * num_layers) // 4]
print(f"num_layers={num_layers}, hidden={hidden_size}, layers={LAYERS}")

# ---------- Prompts: asymmetric endowment to break $10 attractor ----------
GENDERS = ["male", "female"]
AGES = ["young (22 years old)", "old (68 years old)"]
INSTRUCTIONS = [
    (
        "A",
        "You play a dictator game. You have $ENDOW. You may transfer any integer amount from $0 to $ENDOW to an anonymous recipient. Choose the transfer amount.",
    ),
    (
        "B",
        "In an economic allocation task, you hold $ENDOW. You may give any integer portion between $0 and $ENDOW to another anonymous person. Decide how much to give.",
    ),
]
MEETINGS = [
    (
        "meeting",
        "Before deciding, you briefly met the recipient face-to-face and had a short conversation.",
    ),
    (
        "no-meeting",
        "You have never met and will never meet the recipient; the interaction is fully anonymous.",
    ),
]
ENDOWMENTS = [17, 13, 9]  # asymmetric so fair split isn't a round number

SYS = (
    "You are a participant in a behavioral economics study. "
    'Respond ONLY with a JSON object of the form {"transfer": N} where N is a single integer dollar amount. No other text.'
)


def build_prompt(gender, age, instr_text, meeting_text, endow):
    itxt = instr_text.replace("$ENDOW", f"${endow}")
    user = (
        f"Profile: You are a {gender} person, {age}.\n"
        f"{itxt}\n"
        f"Context: {meeting_text}\n"
        f'Return JSON only: {{"transfer": N}} where 0 <= N <= {endow}.'
    )
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


rng = random.Random(0)
N_TRIALS = 400
trials = []
for i in range(N_TRIALS):
    g = rng.choice(GENDERS)
    a = rng.choice(AGES)
    ilab, itxt = rng.choice(INSTRUCTIONS)
    mlab, mtxt = rng.choice(MEETINGS)
    endow = rng.choice(ENDOWMENTS)
    trials.append(
        dict(
            gender=g,
            age=a,
            instr=ilab,
            instr_text=itxt,
            meeting=mlab,
            meeting_text=mtxt,
            endow=endow,
            prompt=build_prompt(g, a, itxt, mtxt, endow),
        )
    )

# ---------- Multi-layer activation extraction ----------
_cap = {}


def make_hook(name):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        _cap[name] = h.detach()

    return hook


layer_modules = {L: model.model.layers[L] for L in LAYERS}


@torch.no_grad()
def get_last_token_activations(prompt):
    _cap.clear()
    handles = [
        layer_modules[L].register_forward_hook(make_hook(f"h{L}")) for L in LAYERS
    ]
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        model(**inputs)
        out = {L: _cap[f"h{L}"][0, -1, :].float().cpu().numpy() for L in LAYERS}
    finally:
        for h in handles:
            h.remove()
    return out


# ---------- Extract per-variable directions at each layer ----------
variables = ["gender", "age", "instr", "meeting"]
N_PAIRS = 48


def make_variant(t, var, sign):
    t2 = dict(t)
    if var == "gender":
        t2["gender"] = "male" if sign else "female"
    elif var == "age":
        t2["age"] = AGES[0] if sign else AGES[1]
    elif var == "instr":
        t2["instr"], t2["instr_text"] = INSTRUCTIONS[0] if sign else INSTRUCTIONS[1]
    elif var == "meeting":
        t2["meeting"], t2["meeting_text"] = MEETINGS[0] if sign else MEETINGS[1]
    t2["prompt"] = build_prompt(
        t2["gender"], t2["age"], t2["instr_text"], t2["meeting_text"], t2["endow"]
    )
    return t2


directions_raw = {L: {} for L in LAYERS}
pair_base = rng.sample(trials, N_PAIRS)
for var in variables:
    diffs = {L: [] for L in LAYERS}
    for t in pair_base:
        t_pos = make_variant(t, var, True)
        t_neg = make_variant(t, var, False)
        a_pos = get_last_token_activations(t_pos["prompt"])
        a_neg = get_last_token_activations(t_neg["prompt"])
        for L in LAYERS:
            diffs[L].append(a_pos[L] - a_neg[L])
    for L in LAYERS:
        d = np.mean(np.stack(diffs[L], 0), 0)
        directions_raw[L][var] = d
    print(
        f"[{var}] raw norms: "
        + ", ".join(
            f"L{L}={np.linalg.norm(directions_raw[L][var]):.2f}" for L in LAYERS
        )
    )


def gram_schmidt(vecs_dict, order):
    pure = {}
    basis = []
    for name in order:
        v = vecs_dict[name].astype(np.float64).copy()
        for b in basis:
            v = v - (np.dot(v, b) / (np.dot(b, b) + 1e-12)) * b
        pure[name] = v
        basis.append(v)
    return pure


directions_pure = {L: gram_schmidt(directions_raw[L], variables) for L in LAYERS}

# Cosine similarities across variables at each layer (should be ~0 after GS)
cos_mat = {}
for L in LAYERS:
    M = np.stack([directions_pure[L][v] for v in variables], 0)
    N = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    cos_mat[L] = (N @ N.T).tolist()

# ---------- Intervention hooks (support multiple layer injections) ----------
_inject = {L: {"vec": None, "alpha": 0.0} for L in LAYERS}
inject_handles = []


def make_inject_hook(L):
    def hook(mod, inp, out):
        cfg = _inject[L]
        if cfg["vec"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0] + cfg["alpha"] * cfg["vec"].to(out[0].dtype).to(out[0].device)
            return (h,) + out[1:]
        else:
            return out + cfg["alpha"] * cfg["vec"].to(out.dtype).to(out.device)

    return hook


for L in LAYERS:
    inject_handles.append(layer_modules[L].register_forward_hook(make_inject_hook(L)))


def clear_inject():
    for L in LAYERS:
        _inject[L]["vec"] = None
        _inject[L]["alpha"] = 0.0


# ---------- Parsing (JSON transfer) ----------
def parse_amount(text, endow):
    m = re.search(r'"transfer"\s*:\s*(\d+(?:\.\d+)?)', text)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= endow:
                return v
        except:
            pass
    return None


@torch.no_grad()
def generate_amount(prompt, endow):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return parse_amount(gen, endow), gen


# ---------- Logit-level analysis: expected value under digit-token distribution ----------
# Find token ids for "0".."20" as single tokens (with leading space, as JSON emits after ':')
def token_id(s):
    ids = tokenizer.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


digit_tokens = {}
for k in range(0, 21):
    for s in [f" {k}", f"{k}"]:
        tid = token_id(s)
        if tid is not None:
            digit_tokens[k] = tid
            break


# Build a "pre-answer" prompt that ends right before the number to read logits over N
def build_forced_prefix_prompt(gender, age, instr_text, meeting_text, endow):
    base = build_prompt(gender, age, instr_text, meeting_text, endow)
    # Force the assistant to have started emitting the JSON, so next token is the integer.
    forced = base + '{"transfer":'
    return forced


@torch.no_grad()
def next_token_digit_distribution(prompt, endow):
    # Returns dict k -> probability, restricted to k in [0..endow] with token available
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1, :].float()
    valid_ks = [k for k in range(0, endow + 1) if k in digit_tokens]
    ids = torch.tensor([digit_tokens[k] for k in valid_ks], device=logits.device)
    sub = logits[ids]
    probs = torch.softmax(sub, dim=-1).cpu().numpy()
    return {k: float(probs[i]) for i, k in enumerate(valid_ks)}


def expected_transfer(dist):
    if not dist:
        return float("nan")
    ks = np.array(list(dist.keys()), dtype=float)
    ps = np.array(list(dist.values()))
    ps = ps / (ps.sum() + 1e-12)
    return float((ks * ps).sum())


# ---------- Baseline behavior ----------
clear_inject()
N_BASE = 80
base_amounts = []
base_ev = []
for i in range(N_BASE):
    t = trials[i]
    amt, gen = generate_amount(t["prompt"], t["endow"])
    dist = next_token_digit_distribution(
        build_forced_prefix_prompt(
            t["gender"], t["age"], t["instr_text"], t["meeting_text"], t["endow"]
        ),
        t["endow"],
    )
    ev = expected_transfer(dist)
    base_amounts.append((amt, t["endow"]))
    base_ev.append((ev, t["endow"]))
valid_amt = [a for a, _ in base_amounts if a is not None]
valid_ev = [e for e, _ in base_ev if not math.isnan(e)]
print(
    f"Baseline argmax n_valid={len(valid_amt)} mean={np.mean(valid_amt):.2f} std={np.std(valid_amt):.2f}"
)
print(
    f"Baseline EV n={len(valid_ev)} mean={np.mean(valid_ev):.2f} std={np.std(valid_ev):.2f}"
)

# Baseline as fraction of endowment (unified scale)
base_frac = [a / e for a, e in base_amounts if a is not None]
ev_frac = [e / en for e, en in base_ev if not math.isnan(e)]
print(
    f"Baseline argmax fraction: mean={np.mean(base_frac):.3f} std={np.std(base_frac):.3f}"
)
print(f"Baseline EV fraction: mean={np.mean(ev_frac):.3f} std={np.std(ev_frac):.3f}")

# ---------- Choose per-variable best layer using logit-level sensitivity to natural variation ----------
# We check which layer's direction, when injected at unit-norm scale, maximally shifts EV on a small probe set.
PROBE_N = 24
probe_trials = rng.sample(trials, PROBE_N)


def measure_ev_shift(var, L, alpha_scale):
    d = directions_pure[L][var]
    dnorm = np.linalg.norm(d) + 1e-9
    scale = alpha_scale / dnorm
    dvec = torch.from_numpy(d.astype(np.float32)).to(device)
    shifts = []
    for t in probe_trials:
        pfx = build_forced_prefix_prompt(
            t["gender"], t["age"], t["instr_text"], t["meeting_text"], t["endow"]
        )
        clear_inject()
        base_dist = next_token_digit_distribution(pfx, t["endow"])
        ev0 = expected_transfer(base_dist)
        _inject[L]["vec"] = dvec
        _inject[L]["alpha"] = +scale
        pos_dist = next_token_digit_distribution(pfx, t["endow"])
        ev_p = expected_transfer(pos_dist)
        _inject[L]["alpha"] = -scale
        neg_dist = next_token_digit_distribution(pfx, t["endow"])
        ev_n = expected_transfer(neg_dist)
        clear_inject()
        shifts.append(ev_p - ev_n)
    return float(np.mean(shifts)), float(np.mean(np.abs(shifts)))


ALPHA_UNIT = 8.0
best_layer = {}
layer_scan = {}
for var in variables:
    layer_scan[var] = {}
    best = (-1.0, None)
    for L in LAYERS:
        mean_signed, mean_abs = measure_ev_shift(var, L, ALPHA_UNIT)
        layer_scan[var][L] = {"signed": mean_signed, "abs": mean_abs}
        print(
            f"  layer scan [{var}] L{L}: signed_EV_shift={mean_signed:+.3f}, |EV_shift|={mean_abs:.3f}"
        )
        if mean_abs > best[0]:
            best = (mean_abs, L)
    best_layer[var] = best[1]
    print(f"[{var}] best layer = L{best_layer[var]} (|EV shift|={best[0]:.3f})")

# ---------- α sweep at best layer per variable (logit-level EV + argmax) ----------
ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]  # multiples of ALPHA_UNIT
alpha_sweep = {}
N_SWEEP = 32
sweep_trials = trials[:N_SWEEP]
for var in variables:
    L = best_layer[var]
    d = directions_pure[L][var]
    dnorm = np.linalg.norm(d) + 1e-9
    unit = ALPHA_UNIT / dnorm
    dvec = torch.from_numpy(d.astype(np.float32)).to(device)
    per_alpha = {}
    for a_mult in ALPHAS:
        evs = []
        amts = []
        for t in sweep_trials:
            pfx = build_forced_prefix_prompt(
                t["gender"], t["age"], t["instr_text"], t["meeting_text"], t["endow"]
            )
            clear_inject()
            _inject[L]["vec"] = dvec
            _inject[L]["alpha"] = a_mult * unit
            dist = next_token_digit_distribution(pfx, t["endow"])
            ev = expected_transfer(dist)
            amt, _ = generate_amount(t["prompt"], t["endow"])
            clear_inject()
            evs.append(ev / t["endow"])  # fraction
            if amt is not None:
                amts.append(amt / t["endow"])
        per_alpha[a_mult] = {
            "ev_frac_mean": float(np.mean(evs)),
            "ev_frac_std": float(np.std(evs)),
            "argmax_frac_mean": float(np.mean(amts)) if amts else float("nan"),
            "argmax_n": len(amts),
        }
    alpha_sweep[var] = per_alpha
    print(f"[α-sweep {var}, L{L}]:")
    for a in ALPHAS:
        r = per_alpha[a]
        print(
            f"  α×{a:+.1f}: EV_frac={r['ev_frac_mean']:.3f}±{r['ev_frac_std']:.3f}, argmax_frac={r['argmax_frac_mean']:.3f} (n={r['argmax_n']})"
        )

# ---------- Causal Steering Effect Size (main metric) ----------
# Mean absolute dollar shift in argmax transfer between +α and -α (α = ALPHA_UNIT) at best layer,
# plus logit-based effect (|ΔEV|).
N_MAIN = 60
main_trials = trials[:N_MAIN]
main_results = {}
for var in variables:
    L = best_layer[var]
    d = directions_pure[L][var]
    dnorm = np.linalg.norm(d) + 1e-9
    unit = ALPHA_UNIT / dnorm
    dvec = torch.from_numpy(d.astype(np.float32)).to(device)
    delta_argmax = []
    delta_ev = []
    pos_amts = []
    neg_amts = []
    pos_evs = []
    neg_evs = []
    for t in main_trials:
        pfx = build_forced_prefix_prompt(
            t["gender"], t["age"], t["instr_text"], t["meeting_text"], t["endow"]
        )
        clear_inject()
        _inject[L]["vec"] = dvec
        _inject[L]["alpha"] = +unit
        ap, _ = generate_amount(t["prompt"], t["endow"])
        ev_p = expected_transfer(next_token_digit_distribution(pfx, t["endow"]))
        _inject[L]["alpha"] = -unit
        an, _ = generate_amount(t["prompt"], t["endow"])
        ev_n = expected_transfer(next_token_digit_distribution(pfx, t["endow"]))
        clear_inject()
        if ap is not None and an is not None:
            delta_argmax.append(ap - an)
            pos_amts.append(ap)
            neg_amts.append(an)
        if not (math.isnan(ev_p) or math.isnan(ev_n)):
            delta_ev.append(ev_p - ev_n)
            pos_evs.append(ev_p)
            neg_evs.append(ev_n)
    mean_abs_delta = (
        float(np.mean(np.abs(delta_argmax))) if delta_argmax else float("nan")
    )
    mean_signed_delta = float(np.mean(delta_argmax)) if delta_argmax else float("nan")
    mean_abs_ev = float(np.mean(np.abs(delta_ev))) if delta_ev else float("nan")
    mean_signed_ev = float(np.mean(delta_ev)) if delta_ev else float("nan")
    main_results[var] = dict(
        best_layer=L,
        mean_abs_delta_dollars=mean_abs_delta,
        mean_signed_delta=mean_signed_delta,
        mean_abs_delta_ev=mean_abs_ev,
        mean_signed_delta_ev=mean_signed_ev,
        n_argmax=len(delta_argmax),
        n_ev=len(delta_ev),
        pos_amts=pos_amts,
        neg_amts=neg_amts,
        pos_evs=pos_evs,
        neg_evs=neg_evs,
    )
    print(
        f"[MAIN {var} L{L}] |Δ$|={mean_abs_delta:.2f} (signed {mean_signed_delta:+.2f}), "
        f"|ΔEV|={mean_abs_ev:.2f} (signed {mean_signed_ev:+.2f}), n_arg={len(delta_argmax)}"
    )

# ---------- Random direction control ----------
random_control = {}
for var in variables:
    L = best_layer[var]
    rand_d = np.random.RandomState(42).randn(hidden_size).astype(np.float64)
    dnorm = np.linalg.norm(rand_d) + 1e-9
    unit = ALPHA_UNIT / dnorm
    dvec = torch.from_numpy(rand_d.astype(np.float32)).to(device)
    delta_argmax = []
    delta_ev = []
    for t in main_trials[:30]:
        pfx = build_forced_prefix_prompt(
            t["gender"], t["age"], t["instr_text"], t["meeting_text"], t["endow"]
        )
        clear_inject()
        _inject[L]["vec"] = dvec
        _inject[L]["alpha"] = +unit
        ap, _ = generate_amount(t["prompt"], t["endow"])
        ev_p = expected_transfer(next_token_digit_distribution(pfx, t["endow"]))
        _inject[L]["alpha"] = -unit
        an, _ = generate_amount(t["prompt"], t["endow"])
        ev_n = expected_transfer(next_token_digit_distribution(pfx, t["endow"]))
        clear_inject()
        if ap is not None and an is not None:
            delta_argmax.append(ap - an)
        if not (math.isnan(ev_p) or math.isnan(ev_n)):
            delta_ev.append(ev_p - ev_n)
    random_control[var] = dict(
        mean_abs_delta_dollars=(
            float(np.mean(np.abs(delta_argmax))) if delta_argmax else float("nan")
        ),
        mean_abs_delta_ev=(
            float(np.mean(np.abs(delta_ev))) if delta_ev else float("nan")
        ),
    )
    print(
        f"[RAND ctrl {var} L{best_layer[var]}] |Δ$|={random_control[var]['mean_abs_delta_dollars']:.3f}, |ΔEV|={random_control[var]['mean_abs_delta_ev']:.3f}"
    )

# ---------- Additivity / interaction test: gender + meeting simultaneously ----------
interaction = {}
pair = ("gender", "meeting")
Lg = best_layer[pair[0]]
Lm = best_layer[pair[1]]
d1 = directions_pure[Lg][pair[0]]
d2 = directions_pure[Lm][pair[1]]
u1 = ALPHA_UNIT / (np.linalg.norm(d1) + 1e-9)
u2 = ALPHA_UNIT / (np.linalg.norm(d2) + 1e-9)
v1 = torch.from_numpy(d1.astype(np.float32)).to(device)
v2 = torch.from_numpy(d2.astype(np.float32)).to(device)


def measure_ev(cfg_setter, subset):
    evs = []
    for t in subset:
        pfx = build_forced_prefix_prompt(
            t["gender"], t["age"], t["instr_text"], t["meeting_text"], t["endow"]
        )
        clear_inject()
        cfg_setter()
        ev = expected_transfer(next_token_digit_distribution(pfx, t["endow"]))
        clear_inject()
        if not math.isnan(ev):
            evs.append(ev)
    return float(np.mean(evs)) if evs else float("nan")


sub = main_trials[:30]
ev_base = measure_ev(lambda: None, sub)


def set_only1(sign):
    _inject[Lg]["vec"] = v1
    _inject[Lg]["alpha"] = sign * u1


def set_only2(sign):
    _inject[Lm]["vec"] = v2
    _inject[Lm]["alpha"] = sign * u2


def set_both(s1, s2):
    _inject[Lg]["vec"] = v1
    _inject[Lg]["alpha"] = s1 * u1
    _inject[Lm]["vec"] = v2
    _inject[Lm]["alpha"] = s2 * u2


interaction["ev_base"] = ev_base
for s1 in [+1, -1]:
    for s2 in [+1, -1]:
        ev_1 = measure_ev(lambda: set_only1(s1), sub)
        ev_2 = measure_ev(lambda: set_only2(s2), sub)
        ev_12 = measure_ev(lambda: set_both(s1, s2), sub)
        predicted_additive = (ev_1 - ev_base) + (ev_2 - ev_base) + ev_base
        interaction[f"g{s1:+d}_m{s2:+d}"] = dict(
            ev_only1=ev_1,
            ev_only2=ev_2,
            ev_both=ev_12,
            predicted_additive=predicted_additive,
            residual=ev_12 - predicted_additive,
        )
        print(
            f"[interaction g{s1:+d}_m{s2:+d}] base={ev_base:.2f} g={ev_1:.2f} m={ev_2:.2f} both={ev_12:.2f} pred={predicted_additive:.2f} resid={ev_12-predicted_additive:+.2f}"
        )

# ---------- Specificity matrix: inject var X, measure ΔEV on natural probes contrasting variable Y ----------
# For each pair (X,Y): build matched Y+/Y- prompts, measure change in (EV(Y+)-EV(Y-)) under injection of X.
specificity = {}
N_SPEC = 12
spec_base = rng.sample(trials, N_SPEC)
for X in variables:
    Lx = best_layer[X]
    dx = directions_pure[Lx][X]
    ux = ALPHA_UNIT / (np.linalg.norm(dx) + 1e-9)
    dvx = torch.from_numpy(dx.astype(np.float32)).to(device)
    row = {}
    for Y in variables:
        gaps_base = []
        gaps_inj = []
        for t in spec_base:
            tY_pos = make_variant(t, Y, True)
            tY_neg = make_variant(t, Y, False)
            pfx_p = build_forced_prefix_prompt(
                tY_pos["gender"],
                tY_pos["age"],
                tY_pos["instr_text"],
                tY_pos["meeting_text"],
                tY_pos["endow"],
            )
            pfx_n = build_forced_prefix_prompt(
                tY_neg["gender"],
                tY_neg["age"],
                tY_neg["instr_text"],
                tY_neg["meeting_text"],
                tY_neg["endow"],
            )
            clear_inject()
            ev_p = expected_transfer(
                next_token_digit_distribution(pfx_p, tY_pos["endow"])
            )
            ev_n = expected_transfer(
                next_token_digit_distribution(pfx_n, tY_neg["endow"])
            )
            gaps_base.append(ev_p - ev_n)
            _inject[Lx]["vec"] = dvx
            _inject[Lx]["alpha"] = +ux
            ev_p2 = expected_transfer(
                next_token_digit_distribution(pfx_p, tY_pos["endow"])
            )
            ev_n2 = expected_transfer(
                next_token_digit_distribution(pfx_n, tY_neg["endow"])
            )
            clear_inject()
            gaps_inj.append(ev_p2 - ev_n2)
        row[Y] = dict(
            base_gap=float(np.mean(gaps_base)),
            inj_gap=float(np.mean(gaps_inj)),
            delta_gap=float(np.mean(gaps_inj) - np.mean(gaps_base)),
        )
    specificity[X] = row
    print(
        f"[specificity inject={X}]: "
        + ", ".join(f"{Y}Δgap={row[Y]['delta_gap']:+.2f}" for Y in variables)
    )

# ---------- Aggregate main metric ----------
per_var_effect = {v: main_results[v]["mean_abs_delta_dollars"] for v in variables}
per_var_ev = {v: main_results[v]["mean_abs_delta_ev"] for v in variables}
mean_effect_dollars = float(np.nanmean(list(per_var_effect.values())))
mean_effect_ev = float(np.nanmean(list(per_var_ev.values())))
print(
    f"\n=== Causal Steering Effect Size (mean |Δ$| across variables) = {mean_effect_dollars:.3f} ==="
)
print(
    f"=== Causal Steering EV-based (mean |ΔEV|)               = {mean_effect_ev:.3f} ==="
)

# ---------- Save all data ----------
experiment_data = {
    "dictator_game_v2": {
        "metrics": {
            "train": [
                {
                    "mean_abs_delta_dollars": mean_effect_dollars,
                    "mean_abs_delta_ev": mean_effect_ev,
                }
            ],
            "val": [{"variable": v, **main_results[v]} for v in variables],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "config": {
            "layers": LAYERS,
            "best_layer": best_layer,
            "alpha_unit": ALPHA_UNIT,
            "n_pairs": N_PAIRS,
            "n_main": N_MAIN,
            "endowments": ENDOWMENTS,
        },
        "baseline_amounts": base_amounts,
        "baseline_ev": base_ev,
        "cos_matrix_by_layer": cos_mat,
        "layer_scan": layer_scan,
        "alpha_sweep": alpha_sweep,
        "main_results": main_results,
        "random_control": random_control,
        "interaction": interaction,
        "specificity": specificity,
        "per_variable_effect_dollars": per_var_effect,
        "per_variable_effect_ev": per_var_ev,
        "mean_abs_delta_dollars": mean_effect_dollars,
        "mean_abs_delta_ev": mean_effect_ev,
    }
}

# ---------- Plots ----------
# 1) Main effect bar
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
vs = variables
ax[0].bar(vs, [per_var_effect[v] for v in vs], color="steelblue")
ax[0].set_ylabel("Mean |Δ transfer| ($)")
ax[0].set_title("Argmax causal steering")
ax[1].bar(vs, [per_var_ev[v] for v in vs], color="seagreen")
ax[1].set_ylabel("Mean |ΔEV| ($)")
ax[1].set_title("Logit-level (EV) causal steering")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "dictator_v2_main_effects.png"), dpi=120)
plt.close(fig)

# 2) Alpha sweep
fig, ax = plt.subplots(figsize=(7, 5))
for v in variables:
    ys = [alpha_sweep[v][a]["ev_frac_mean"] for a in ALPHAS]
    ax.plot(ALPHAS, ys, marker="o", label=v)
ax.axhline(np.mean(ev_frac), color="k", ls="--", lw=0.7, label="baseline EV frac")
ax.set_xlabel("α (multiples of unit)")
ax.set_ylabel("Expected transfer / endowment")
ax.set_title("α sweep (logit-level EV, best layer per variable)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "dictator_v2_alpha_sweep.png"), dpi=120)
plt.close(fig)

# 3) Specificity matrix heatmap (Δgap induced by injecting X on gap of Y)
fig, ax = plt.subplots(figsize=(5, 4))
mat = np.array([[specificity[X][Y]["delta_gap"] for Y in variables] for X in variables])
im = ax.imshow(mat, cmap="RdBu_r", vmin=-abs(mat).max(), vmax=abs(mat).max())
ax.set_xticks(range(len(variables)))
ax.set_xticklabels(variables)
ax.set_xlabel("probed variable Y")
ax.set_yticks(range(len(variables)))
ax.set_yticklabels(variables)
ax.set_ylabel("injected variable X")
for i in range(len(variables)):
    for j in range(len(variables)):
        ax.text(
            j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", color="k", fontsize=8
        )
plt.colorbar(im, ax=ax)
ax.set_title("Specificity: Δ(EV gap of Y) when injecting X")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "dictator_v2_specificity.png"), dpi=120)
plt.close(fig)

# 4) Layer scan
fig, ax = plt.subplots(figsize=(6, 4))
for v in variables:
    ys = [layer_scan[v][L]["abs"] for L in LAYERS]
    ax.plot(LAYERS, ys, marker="o", label=v)
ax.set_xlabel("layer")
ax.set_ylabel("|EV shift| under ±α")
ax.set_title("Per-variable direction sensitivity across layers")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "dictator_v2_layer_scan.png"), dpi=120)
plt.close(fig)

# 5) Random control comparison
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(variables))
w = 0.35
ax.bar(x - w / 2, [per_var_ev[v] for v in variables], w, label="pure direction")
ax.bar(
    x + w / 2,
    [random_control[v]["mean_abs_delta_ev"] for v in variables],
    w,
    label="random direction",
)
ax.set_xticks(x)
ax.set_xticklabels(variables)
ax.set_ylabel("|ΔEV| ($)")
ax.set_title("Pure vs random-direction control")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "dictator_v2_random_control.png"), dpi=120)
plt.close(fig)

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved to", working_dir)

for h in inject_handles:
    h.remove()
del model
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
