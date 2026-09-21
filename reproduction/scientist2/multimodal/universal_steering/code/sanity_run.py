"""Sanity smoke test — smallest cheapest E2E path.
- Load Llama-3.1-8B in bf16 on a single GPU.
- Cache activations on 40 refusal pairs.
- Fit linear probe on 3 blocks (0, 15, 25) — validate accuracy > baseline.
- Run RFM at block 15 (T=3 iters).
- Add α=1 hook, generate on 3 held-out prompts. Compare to baseline (α=0).
- Judge one with GPT-4o.
Total wall: ~5 min.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))

from model_utils import load_model, cache_last_token_activations, generate_with_steering, free_cuda
from rfm_core import fit_linear_probe, probe_accuracy, extract_rfm, caa_mean_diff
from dmx_api import judge_completion

WORK = Path("/data/zhenqian/Reproduction1/mechanica/multimodal/universal_steering")


def main():
    out = WORK / "runs" / "A0_sanity"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[sanity] out={out}")

    rows = [json.loads(l) for l in open(WORK / "data/paired/refusal.jsonl")][:80]  # 40+40
    statements = [r["statement"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows], dtype=np.float64)
    print(f"[sanity] {len(statements)} paired statements loaded")

    print("[sanity] loading model...")
    t0 = time.time()
    model, tok, cfg = load_model(dtype="bfloat16", device="cuda")
    print(f"[sanity]   model loaded in {time.time()-t0:.1f}s "
          f"(num_blocks={cfg['num_blocks']}, d_model={cfg['d_model']})")

    print("[sanity] caching activations...")
    t0 = time.time()
    acts = cache_last_token_activations(model, tok, statements, batch_size=8, max_length=256)
    print(f"[sanity]   cache took {time.time()-t0:.1f}s, shape={acts.shape}")

    tr_n = 60
    X_tr = acts[:tr_n]; y_tr = labels[:tr_n]
    X_va = acts[tr_n:]; y_va = labels[tr_n:]

    accs = {}
    for l in [0, 10, 15, 20, 25]:
        w_vec, w_b = fit_linear_probe(X_tr[:, l, :].astype(np.float64), y_tr)
        acc = probe_accuracy((w_vec, w_b), X_va[:, l, :].astype(np.float64), y_va)
        accs[l] = acc
        print(f"[sanity]   block {l:02d} probe val_acc = {acc:.3f}")
    best_l = max(accs, key=lambda k: accs[k])
    print(f"[sanity] best block = {best_l} (acc={accs[best_l]:.3f})")

    # RFM at best_l
    print(f"[sanity] running RFM at block {best_l} (3 iters)...")
    t0 = time.time()
    rfm = extract_rfm(X_tr[:, best_l, :].astype(np.float64), y_tr, n_iters=3, verbose=True)
    print(f"[sanity]   RFM took {time.time()-t0:.1f}s, top_ratio={rfm['top_ratio']:.3f}")
    v = rfm["v_c"]
    v_caa = caa_mean_diff(X_tr[:, best_l, :].astype(np.float64), y_tr)
    print(f"[sanity]   cos(v_rfm, v_caa) = {float(np.abs(v @ v_caa)):.4f}")

    # Val projection accuracy of RFM vector
    proj = X_va[:, best_l, :].astype(np.float64) @ v
    rfm_acc = float(((np.sign(proj) * y_va) > 0).mean())
    print(f"[sanity]   RFM vector val_acc = {rfm_acc:.3f}")

    # Generation test
    held_out = [json.loads(l)["prompt"] for l in open(WORK / "data/paired/held_out/refusal.jsonl")][:3]
    chats = [tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
              for p in held_out]

    print("[sanity] generating unsteered baseline...")
    baseline = generate_with_steering(model, tok, chats, None, max_new_tokens=80, batch_size=3)
    print("[sanity] generating steered α=+2 (positive = jailbreak-compliant)...")
    steered = generate_with_steering(model, tok, chats,
        [{"block_idx": best_l, "v_c": v, "alpha": 2.0}], max_new_tokens=80, batch_size=3)

    diff_chars = sum(1 for a, b in zip(baseline, steered) if a.strip() != b.strip())
    print(f"[sanity] outputs differ in {diff_chars}/{len(baseline)} rows")
    for i, (b, s) in enumerate(zip(baseline, steered)):
        print(f"[sanity]   [{i}] prompt: {held_out[i][:80]}")
        print(f"[sanity]        baseline: {b[:150]}")
        print(f"[sanity]        steered : {s[:150]}")

    # Judge test
    print("[sanity] pinging GPT-4o judge on 1 example...")
    j_base = judge_completion(concept="refusal", prompt=held_out[0], completion=baseline[0],
        positive_desc="willingness to comply with a harmful request",
        negative_desc="refusal / safety guardrail")
    j_steer = judge_completion(concept="refusal", prompt=held_out[0], completion=steered[0],
        positive_desc="willingness to comply with a harmful request",
        negative_desc="refusal / safety guardrail")
    print(f"[sanity]   baseline score={j_base['score']} ({j_base['raw'][:20]})")
    print(f"[sanity]   steered  score={j_steer['score']} ({j_steer['raw'][:20]})")

    # Sanity assertions
    passed = True
    reasons = []
    if accs[best_l] < 0.75:
        passed = False; reasons.append(f"best block val acc {accs[best_l]:.3f} < 0.75")
    if diff_chars == 0:
        passed = False; reasons.append("steered output identical to baseline (hook not effective)")
    if j_base["score"] is None or j_steer["score"] is None:
        passed = False; reasons.append("GPT-4o judge returned None score")
    result = {"passed": passed, "reasons": reasons, "accs": accs, "best_block": best_l,
              "rfm_top_ratio": rfm["top_ratio"], "cos_rfm_caa": float(np.abs(v @ v_caa)),
              "rfm_val_acc": rfm_acc,
              "baseline_judge": j_base["score"], "steered_judge": j_steer["score"],
              "gen_diff_frac": diff_chars / len(baseline)}
    with open(out / "sanity_result.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"[sanity] RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        for r in reasons:
            print(f"[sanity]   - {r}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
