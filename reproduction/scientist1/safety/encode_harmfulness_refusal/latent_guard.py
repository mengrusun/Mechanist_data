"""Latent-Guard classifier: linear probe on the harmfulness direction (v_inst @ layer L)
of Llama-3-8B-Instruct's residual stream at position t_inst.

Baseline: Llama-Guard-3-8B safety judge on the same prompts.

Datasets:
  train:  AdvBench (harmful) vs Alpaca (benign) — 200 each
  test:   held-out AdvBench, Alpaca, JBB harmful, JBB benign,
          CATQA (harmful), Sorry-Bench (harmful), XSTest (mixed safe/unsafe).
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    LLAMA_GUARD_PATH, LLAMA3_PATH, load_model_and_tokenizer,
    format_llama3_chat, load_advbench, load_alpaca_benign, load_jbb_harmful,
    load_jbb_benign, load_catqa_english, load_sorrybench, load_xstest_prompts,
    T_INST,
)

ACT_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/acts_llama3"
RES_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/results"


def load_hs_inst(name, layer):
    z = np.load(os.path.join(ACT_DIR, f"{name}.npz"), allow_pickle=True)
    return z["hs_inst"][:, layer].astype(np.float32), z["prompts"]


@torch.no_grad()
def extract_hs_inst_online(model, tok, prompts, layer, batch_size=8, device="cuda"):
    """Extract activations at (layer, T_INST) online for prompts not pre-extracted."""
    D = model.config.hidden_size
    out = np.empty((len(prompts), D), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        formatted = [format_llama3_chat(tok, p) for p in batch]
        enc = tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        o = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                  output_hidden_states=True, use_cache=False)
        out[i:i+len(batch)] = o.hidden_states[layer][:, T_INST].float().cpu().numpy()
    return out, time.time() - t0


@torch.no_grad()
def llama_guard_classify(model, tok, prompts, batch_size=4, device="cuda", max_new_tokens=20):
    """Return per-prompt (is_unsafe, elapsed) using Llama-Guard-3-8B's chat template."""
    preds = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        formatted = []
        for p in batch:
            msgs = [{"role": "user", "content": p}]
            s = tok.apply_chat_template(msgs, tokenize=False)
            formatted.append(s)
        enc = tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        for j in range(len(batch)):
            in_len = enc.input_ids.shape[1]
            txt = tok.decode(gen[j, in_len:], skip_special_tokens=True)
            preds.append("unsafe" in txt.lower())
    return np.array(preds, dtype=bool), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15,
                    help="layer used for the probe (middle-ish, from direction stats)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--skip_llama_guard", action="store_true")
    ap.add_argument("--n_llama_guard", type=int, default=200,
                    help="cap of samples per dataset given to Llama Guard (compute)")
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)

    print(f"Training linear probe on layer {args.layer}, t_inst ...")
    Xh, _ = load_hs_inst("advbench_train", args.layer)   # harmful
    Xb, _ = load_hs_inst("alpaca_train", args.layer)     # benign
    X = np.concatenate([Xh, Xb], 0)
    y = np.concatenate([np.ones(len(Xh)), np.zeros(len(Xb))]).astype(int)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)

    # Test datasets (pre-extracted) — we already have activations for these
    test_sets = {
        "advbench_test":  ("advbench_test",  True),   # harmful
        "alpaca_test":    ("alpaca_test",    False),
        "jbb_harmful":    ("jbb_harmful",    True),
        "jbb_benign":     ("jbb_benign",     False),
        "catqa":          ("catqa",          True),
        "sorrybench":     ("sorrybench",     True),
    }

    per_set = {}
    all_y, all_p = [], []
    for name, (acts_name, is_harmful) in test_sets.items():
        Xt, prompts = load_hs_inst(acts_name, args.layer)
        prob = clf.predict_proba(Xt)[:, 1]
        pred = (prob >= 0.5).astype(int)
        yt = np.full(len(Xt), 1 if is_harmful else 0, dtype=int)
        per_set[name] = {
            "n": int(len(Xt)),
            "positive_rate": float(pred.mean()),
            "accuracy_vs_dataset_label": float(accuracy_score(yt, pred)),
            "prob_mean": float(prob.mean()),
        }
        all_y.append(yt); all_p.append(prob)
        print(f"  {name}: n={len(Xt)}  probe_flags={pred.mean():.3f}  acc(vs label)={accuracy_score(yt, pred):.3f}")

    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    per_set["_pooled_binary_metrics"] = {
        "auroc": float(roc_auc_score(all_y, all_p)),
        "accuracy": float(accuracy_score(all_y, (all_p >= 0.5).astype(int))),
        "f1": float(f1_score(all_y, (all_p >= 0.5).astype(int))),
    }
    print(f"\nProbe pooled AUROC={per_set['_pooled_binary_metrics']['auroc']:.4f}  ACC={per_set['_pooled_binary_metrics']['accuracy']:.4f}  F1={per_set['_pooled_binary_metrics']['f1']:.4f}")

    result = {"probe": per_set}

    # ---- Latent-Guard latency ----
    print("\nMeasuring latent-guard latency on ~100 unseen prompts ...")
    from common import load_advbench, load_alpaca_benign
    lat_prompts = load_advbench(n=50, seed=7) + load_alpaca_benign(n=50, seed=7)
    print("  Loading Llama-3-8B for online activation extraction ...")
    model, tok = load_model_and_tokenizer(LLAMA3_PATH)
    device = next(model.parameters()).device
    # Wrap in a proper single-layer extraction (early exit at chosen layer would be even faster,
    # but forward with `output_hidden_states` reads at all layers regardless).
    # For a fair "no logits" comparison we skip generating tokens.
    hs, elapsed_probe = extract_hs_inst_online(model, tok, lat_prompts, args.layer,
                                                batch_size=args.batch_size, device=device)
    prob = clf.predict_proba(hs)[:, 1]
    result["latent_guard_latency_sec"] = float(elapsed_probe)
    result["latent_guard_per_sample_sec"] = float(elapsed_probe / len(lat_prompts))
    print(f"  Latent Guard: total={elapsed_probe:.2f}s  per-sample={elapsed_probe/len(lat_prompts)*1000:.1f}ms")

    # Free model
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    if not args.skip_llama_guard:
        print("\nLoading Llama-Guard-3-8B ...")
        lg_model, lg_tok = load_model_and_tokenizer(LLAMA_GUARD_PATH)
        # Latency
        lg_pred, lg_elapsed = llama_guard_classify(lg_model, lg_tok, lat_prompts,
                                                    batch_size=max(1, args.batch_size//2),
                                                    device=next(lg_model.parameters()).device)
        result["llama_guard_latency_sec"] = float(lg_elapsed)
        result["llama_guard_per_sample_sec"] = float(lg_elapsed / len(lat_prompts))
        print(f"  Llama Guard: total={lg_elapsed:.2f}s  per-sample={lg_elapsed/len(lat_prompts)*1000:.1f}ms")

        # Per-set accuracy
        # For fairness we use the SAME prompt lists as we used for the probe (subsample if too many).
        lg_per_set = {}
        all_y_lg, all_p_lg = [], []
        for name, (acts_name, is_harmful) in test_sets.items():
            _, prompts = load_hs_inst(acts_name, args.layer)
            prompts = list(prompts)
            if len(prompts) > args.n_llama_guard:
                prompts = prompts[:args.n_llama_guard]
            preds, _ = llama_guard_classify(lg_model, lg_tok, prompts,
                                             batch_size=max(1, args.batch_size//2),
                                             device=next(lg_model.parameters()).device)
            yt = np.full(len(prompts), 1 if is_harmful else 0, dtype=int)
            lg_per_set[name] = {
                "n": int(len(prompts)),
                "positive_rate": float(preds.mean()),
                "accuracy_vs_dataset_label": float(accuracy_score(yt, preds.astype(int))),
            }
            all_y_lg.append(yt); all_p_lg.append(preds.astype(int))
            print(f"  LG {name}: n={len(prompts)}  flags={preds.mean():.3f}  acc={accuracy_score(yt, preds.astype(int)):.3f}")

        all_y_lg = np.concatenate(all_y_lg); all_p_lg = np.concatenate(all_p_lg)
        lg_per_set["_pooled_binary_metrics"] = {
            "accuracy": float(accuracy_score(all_y_lg, all_p_lg)),
            "f1": float(f1_score(all_y_lg, all_p_lg, zero_division=0)),
        }
        result["llama_guard"] = lg_per_set

    with open(os.path.join(RES_DIR, "latent_guard.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {os.path.join(RES_DIR, 'latent_guard.json')}")


if __name__ == "__main__":
    main()
