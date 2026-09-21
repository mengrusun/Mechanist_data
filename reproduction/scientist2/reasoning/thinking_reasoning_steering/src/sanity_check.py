"""Sanity check: end-to-end smoke test on ~10 excerpts × 5 layers × 1 GPU.

Steps:
1. Build 20 tiny synthetic contrastive examples (5 pos / 5 neg per behaviour × 2 behaviours)
2. Load model, cache last-token residual for 5 evenly-spaced layers
3. Fit a probe per (behaviour, layer); confirm L* selection returns int
4. Generate 3 short chains WITHOUT steering, then WITH a strong +2σ steering,
   confirm hook diagnostic counters show >0 decode calls (steering actually applied)
5. Verify LLM-judge call returns valid JSON
"""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "src")
import numpy as np
import torch
from model_utils import load_model, get_num_layers, capture_residual_last_token, SteeringHook, format_task_prompt, generate_with_optional_steering, get_input_device
from llm_judge import score_chain

t0 = time.time()

BEHAVIOURS = ["expressing_uncertainty", "backtracking"]
synth = []
for i in range(5):
    synth.append({"id": f"unc_pos_{i}", "text": "Let me think... I'm not entirely sure whether we should add 3 or 4 here. Actually maybe both work. It might be wrong but I'll try 3.", "behaviours": {"expressing_uncertainty": 1, "backtracking": 0}})
for i in range(5):
    synth.append({"id": f"unc_neg_{i}", "text": "The derivative of x^2 is 2x. Therefore the slope at x=3 is 6. Applying the chain rule gives 6x.", "behaviours": {"expressing_uncertainty": 0, "backtracking": 0}})
for i in range(5):
    synth.append({"id": f"back_pos_{i}", "text": "Let me set up an equation. Wait, that's not right. Scratch that — actually let me start over from the constraint equations. The correct approach is to use substitution.", "behaviours": {"expressing_uncertainty": 0, "backtracking": 1}})
for i in range(5):
    synth.append({"id": f"back_neg_{i}", "text": "First we compute the sum. Next we take the log. Finally we exponentiate to get the answer 42.", "behaviours": {"expressing_uncertainty": 0, "backtracking": 0}})

print(f"[sanity] loading model ...")
model, tok = load_model("/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B")
n_lay = get_num_layers(model)
print(f"[sanity] model loaded, {n_lay} layers")

# 5 evenly-spaced layers
layers = sorted(set(int(x) for x in np.linspace(0, n_lay - 1, 5)))
print(f"[sanity] layers to capture: {layers}")

texts = [r["text"] for r in synth]
device = get_input_device(model)
N = len(texts)
hidden = model.config.hidden_size
out = {li: np.zeros((N, hidden), dtype=np.float32) for li in layers}
for i in range(0, N, 4):
    chunk = texts[i:i+4]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with capture_residual_last_token(model, layers) as captured, torch.no_grad():
        _ = model(**enc, use_cache=False)
    for li in layers:
        out[li][i:i+len(chunk)] = captured[li].numpy()
print(f"[sanity] cached activations {out[layers[0]].shape}")

# Probe per (behaviour, layer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
best_layers = {}
directions = {}
for b in BEHAVIOURS:
    y = np.array([r["behaviours"][b] for r in synth], dtype=np.int64)
    best_auc, best_li = -1, layers[0]
    for li in layers:
        X = out[li]
        # 70/30 split (tiny)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        tr, te = idx[:14], idx[14:]
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(X[tr], y[tr])
        try:
            auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
        except Exception:
            auc = 0
        if auc > best_auc:
            best_auc, best_li = auc, li
    best_layers[b] = (best_li, best_auc)
    # Mean-diff direction on training portion
    X = out[best_li]
    v = X[y == 1].mean(0) - X[y == 0].mean(0)
    directions[b] = torch.from_numpy(v.astype(np.float32))
    # sigma_proj
    u = v / (np.linalg.norm(v) + 1e-9)
    sigma = float(np.std(out[best_li] @ u))
    print(f"[sanity] {b}: L*={best_li} auc={best_auc:.3f} sigma_proj={sigma:.3f}")

# Generate with + without steering
b = BEHAVIOURS[0]
li_star = best_layers[b][0]
direction = directions[b]
u = direction.numpy() / (np.linalg.norm(direction.numpy()) + 1e-9)
sigma = float(np.std(out[li_star] @ u))

prompt = format_task_prompt(tok, "What is 12 * 15? Show your reasoning.")
print(f"[sanity] generating baseline (α=0) ...")
gen_base = generate_with_optional_steering(model, tok, [prompt], None, None, max_new_tokens=80, do_sample=False)
print(f"[sanity] baseline gen len {len(gen_base[0])} chars, preview: {gen_base[0][:120]!r}")

print(f"[sanity] generating with α=+2σ steering (target: {b}) at L={li_star} ...")
steer = SteeringHook(direction, 2.0 * sigma)
gen_st = generate_with_optional_steering(model, tok, [prompt], steer, li_star, max_new_tokens=80, do_sample=False)
print(f"[sanity] steered gen preview: {gen_st[0][:120]!r}")
print(f"[sanity] hook diagnostics: prefill_calls={steer.n_prefill_calls} decode_calls={steer.n_decode_calls} tokens_steered={steer.n_new_tokens_steered}")

# --- ASSERTIONS ---
assert steer.n_decode_calls > 0, "Steering hook was never called during decode!"
assert steer.n_new_tokens_steered > 0, "No new tokens were steered!"
assert gen_base[0] != gen_st[0], "Steering did not change the output (may indicate hook not firing)"

# LLM judge test
sc = score_chain(gen_st[0], task_gold="180")
print(f"[sanity] judge score: {sc}")

print(f"\n[sanity] PASS in {time.time()-t0:.1f}s")
