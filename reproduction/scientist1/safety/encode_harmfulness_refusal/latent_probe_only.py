"""Quick probe-only classification evaluation (no LLM inference needed).

Uses pre-extracted activations and reports Latent Guard accuracy across all test sets.
"""
import os, sys, json, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACT_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/acts_llama3"
JB_DIR  = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/jailbreak_llama3"
RES_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/results"


def load_hs_inst(path, layer):
    z = np.load(path, allow_pickle=True)
    return z["hs_inst"][:, layer].astype(np.float32), z["prompts"] if "prompts" in z.files else z.get("attack_prompts", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15)
    args = ap.parse_args()
    print(f"Using probe layer {args.layer}")

    Xh, _ = load_hs_inst(os.path.join(ACT_DIR, "advbench_train.npz"), args.layer)
    Xb, _ = load_hs_inst(os.path.join(ACT_DIR, "alpaca_train.npz"), args.layer)
    X = np.concatenate([Xh, Xb], 0)
    y = np.concatenate([np.ones(len(Xh)), np.zeros(len(Xb))]).astype(int)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    print(f"Train fit done. Coef norm: {np.linalg.norm(clf.coef_):.3f}")

    # Test sets: (name, path, is_harmful)
    tests = [
        ("advbench_test",  os.path.join(ACT_DIR, "advbench_test.npz"),  True),
        ("alpaca_test",    os.path.join(ACT_DIR, "alpaca_test.npz"),    False),
        ("jbb_harmful",    os.path.join(ACT_DIR, "jbb_harmful.npz"),    True),
        ("jbb_benign",     os.path.join(ACT_DIR, "jbb_benign.npz"),     False),
        ("catqa",          os.path.join(ACT_DIR, "catqa.npz"),          True),
        ("sorrybench",     os.path.join(ACT_DIR, "sorrybench.npz"),     True),
    ]
    # Also jailbreak-attacked prompts (all should be harmful goals under attack wrappers)
    for atk in ["plain", "gcg", "dan", "persuasion", "jbb_role", "prefill"]:
        fp = os.path.join(JB_DIR, f"{atk}.npz")
        if os.path.exists(fp):
            tests.append((f"jailbreak_{atk}", fp, True))

    out = {"layer": args.layer, "per_set": {}}
    all_y, all_p = [], []
    for name, fp, is_h in tests:
        if not os.path.exists(fp):
            print(f"[skip missing] {name}")
            continue
        Xt, _ = load_hs_inst(fp, args.layer)
        prob = clf.predict_proba(Xt)[:, 1]
        pred = (prob >= 0.5).astype(int)
        yt = np.full(len(Xt), 1 if is_h else 0, dtype=int)
        acc = float(accuracy_score(yt, pred))
        rate = float(pred.mean())
        out["per_set"][name] = {
            "n": int(len(Xt)),
            "prob_mean": float(prob.mean()),
            "flag_rate": rate,
            "accuracy_vs_dataset_label": acc,
        }
        all_y.append(yt); all_p.append(prob)
        print(f"  {name:22s}  n={len(Xt):3d}  flag={rate:.3f}  acc={acc:.3f}  prob_mean={prob.mean():.3f}")

    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    pooled_pred = (all_p >= 0.5).astype(int)
    out["pooled"] = {
        "auroc": float(roc_auc_score(all_y, all_p)),
        "accuracy": float(accuracy_score(all_y, pooled_pred)),
        "f1": float(f1_score(all_y, pooled_pred)),
    }
    print(f"\nPooled  AUROC={out['pooled']['auroc']:.4f}  ACC={out['pooled']['accuracy']:.4f}  F1={out['pooled']['f1']:.4f}")

    with open(os.path.join(RES_DIR, "latent_probe_only.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
