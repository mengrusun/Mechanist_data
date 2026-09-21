"""Compute harmfulness / refusal directions and per-layer separation scores.

We use difference-of-means (DoM) direction:
    v(l, pos) = mean(harmful[l, pos]) - mean(benign[l, pos])
The DoM direction is a strong linear estimator of the concept-difference axis.

Then we score per-layer separation by projecting held-out samples onto v(l, pos)
(after centering to the joint mean) and reporting:
  * cosine of DoM to individual sample difference (implicit via projection)
  * AUROC of the projection on held-out (harmful vs benign)
  * mean-diff / pooled-std => Cohen-d style
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACT_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/acts_llama3"
OUT_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/results"


def load_acts(name):
    z = np.load(os.path.join(ACT_DIR, f"{name}.npz"), allow_pickle=True)
    return z["hs_inst"].astype(np.float32), z["hs_post"].astype(np.float32), z["prompts"]


def compute_direction(harm, ben):
    """harm, ben: (N, L+1, D).  Returns direction (L+1, D) and per-layer separation stats."""
    Lp1, D = harm.shape[1], harm.shape[2]
    v = harm.mean(0) - ben.mean(0)  # (L+1, D)
    return v


def auroc(pos_scores, neg_scores):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    s = np.concatenate([pos_scores, neg_scores])
    return roc_auc_score(y, s)


def project(acts, v):
    """acts: (N, L+1, D). v: (L+1, D). Returns (N, L+1) projection scores."""
    # normalize per-layer direction to unit length so scales are comparable
    vn = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
    return np.einsum("nld,ld->nl", acts, vn)


def cohen_d(a, b):
    m = a.mean(0) - b.mean(0)
    s = np.sqrt(0.5 * (a.var(0) + b.var(0)) + 1e-12)
    return m / s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading train contrast set...")
    hi_h, hp_h, _ = load_acts("advbench_train")  # harmful
    hi_b, hp_b, _ = load_acts("alpaca_train")    # benign

    Lp1, D = hi_h.shape[1], hi_h.shape[2]
    print(f"Shapes: harm inst {hi_h.shape}, benign inst {hi_b.shape}")

    # ---- Compute directions at BOTH positions (inst and post) ----
    v_inst = compute_direction(hi_h, hi_b)   # (L+1, D) direction computed at t_inst
    v_post = compute_direction(hp_h, hp_b)   # (L+1, D) direction computed at t_post

    # Save directions
    np.savez(os.path.join(OUT_DIR, "directions_llama3.npz"),
             v_inst=v_inst, v_post=v_post)
    print(f"Saved directions.")

    # ---- Held-out separation scores ----
    print("Held-out evaluation on advbench_test vs alpaca_test...")
    hi_h_te, hp_h_te, _ = load_acts("advbench_test")
    hi_b_te, hp_b_te, _ = load_acts("alpaca_test")

    # Project test sets onto v_inst at position t_inst and v_post at position t_post,
    # AND CROSS: v_inst applied at t_post and v_post applied at t_inst.
    # This tests Claim 2: the harmfulness signal is best readable at t_inst.
    def eval_position(v, harm_acts, ben_acts):
        p_h = project(harm_acts, v)  # (N, L+1)
        p_b = project(ben_acts, v)
        aur = np.array([auroc(p_h[:, l], p_b[:, l]) for l in range(Lp1)])
        d = cohen_d(p_h, p_b)
        return aur, d

    results = {}
    # v_inst on inst position: expected best for harmfulness
    aur_ii, d_ii = eval_position(v_inst, hi_h_te, hi_b_te)
    # v_inst on post position: cross - should still separate but less cleanly
    aur_ip, d_ip = eval_position(v_inst, hp_h_te, hp_b_te)
    # v_post on inst position
    aur_pi, d_pi = eval_position(v_post, hi_h_te, hi_b_te)
    # v_post on post position: expected best for refusal
    aur_pp, d_pp = eval_position(v_post, hp_h_te, hp_b_te)

    print("\nBest AUROC per (direction, apply-position):")
    print(f"  v_inst @ t_inst : best AUROC {aur_ii.max():.4f} at layer {aur_ii.argmax()}")
    print(f"  v_inst @ t_post : best AUROC {aur_ip.max():.4f} at layer {aur_ip.argmax()}")
    print(f"  v_post @ t_inst : best AUROC {aur_pi.max():.4f} at layer {aur_pi.argmax()}")
    print(f"  v_post @ t_post : best AUROC {aur_pp.max():.4f} at layer {aur_pp.argmax()}")

    results["heldout_separation"] = {
        "v_inst_at_t_inst": {"aur": aur_ii.tolist(), "cohen_d": d_ii.tolist(), "best_layer": int(aur_ii.argmax()), "best_auroc": float(aur_ii.max())},
        "v_inst_at_t_post": {"aur": aur_ip.tolist(), "cohen_d": d_ip.tolist(), "best_layer": int(aur_ip.argmax()), "best_auroc": float(aur_ip.max())},
        "v_post_at_t_inst": {"aur": aur_pi.tolist(), "cohen_d": d_pi.tolist(), "best_layer": int(aur_pi.argmax()), "best_auroc": float(aur_pi.max())},
        "v_post_at_t_post": {"aur": aur_pp.tolist(), "cohen_d": d_pp.tolist(), "best_layer": int(aur_pp.argmax()), "best_auroc": float(aur_pp.max())},
    }

    # ---- Direction similarity between positions ----
    # If the two directions were the "same concept" the cosine should be ~1.
    def cos(a, b):
        return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
    cos_ip = cos(v_inst, v_post)  # per-layer cosine
    print("\nCosine similarity between v_inst and v_post per layer:")
    for l in range(0, Lp1, 4):
        print(f"  L{l:2d}: {cos_ip[l]:+.4f}")
    results["cos_v_inst_v_post"] = cos_ip.tolist()

    # ---- Pick a working layer: highest v_inst @ t_inst AUROC in middle third ----
    mid_start = Lp1 // 3
    mid_end = 2 * Lp1 // 3
    layer_inst = mid_start + int(np.argmax(aur_ii[mid_start:mid_end]))
    layer_post = mid_start + int(np.argmax(aur_pp[mid_start:mid_end]))
    results["chosen_layers"] = {"harm_layer": layer_inst, "refuse_layer": layer_post}
    print(f"\nChosen harm layer (v_inst): {layer_inst}  AUROC={aur_ii[layer_inst]:.4f}")
    print(f"Chosen refuse layer (v_post): {layer_post}  AUROC={aur_pp[layer_post]:.4f}")

    with open(os.path.join(OUT_DIR, "direction_stats.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {os.path.join(OUT_DIR, 'direction_stats.json')}")


if __name__ == "__main__":
    main()
