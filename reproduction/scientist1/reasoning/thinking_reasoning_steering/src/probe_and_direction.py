"""
Given per-layer activations for present/absent pairs, do two things:
  (a) Compute the mean-difference steering vector per (behaviour, layer):
        v_{b, L} = mean(a_present) - mean(a_absent)
  (b) Test linear separability: project val activations onto v_{b, L}, threshold at midpoint
      of train projections. Report accuracy — validates Claim 1 (approximately linear direction).

Saves: results/probe_accuracy.json, data/directions.pt
"""
import os, json, argparse
import torch


def project_and_score(v, x_pos, x_neg, thresh):
    """v: (H,), x_pos: (N,H), x_neg: (N,H). Score = sign(dot(x, v) - thresh)."""
    proj_pos = x_pos @ v
    proj_neg = x_neg @ v
    correct = (proj_pos > thresh).sum().item() + (proj_neg <= thresh).sum().item()
    return correct / (len(x_pos) + len(x_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/activations.pt")
    ap.add_argument("--out_dir_dir", default="data/directions.pt")
    ap.add_argument("--out_probe", default="results/probe_accuracy.json")
    args = ap.parse_args()

    acts = torch.load(args.acts, weights_only=False)
    behaviours = [k for k in acts if k != "config"]
    L_plus1 = acts[behaviours[0]]["train_present"].shape[1]

    directions = {}
    probe = {}
    for b in behaviours:
        tp = acts[b]["train_present"]  # (N, L+1, H)
        ta = acts[b]["train_absent"]
        vp = acts[b]["val_present"]
        va = acts[b]["val_absent"]

        dirs = tp.mean(dim=0) - ta.mean(dim=0)  # (L+1, H)

        per_layer = {}
        for L in range(L_plus1):
            v = dirs[L]
            v_norm = v / (v.norm() + 1e-8)
            # thresh at midpoint of train mean projections
            mean_pos = (tp[:, L] @ v_norm).mean().item()
            mean_neg = (ta[:, L] @ v_norm).mean().item()
            thresh = (mean_pos + mean_neg) / 2.0

            train_acc = project_and_score(v_norm, tp[:, L], ta[:, L], thresh)
            val_acc   = project_and_score(v_norm, vp[:, L], va[:, L], thresh)
            per_layer[L] = {
                "train_acc": round(train_acc, 4),
                "val_acc":   round(val_acc,   4),
                "sep_gap":   round(mean_pos - mean_neg, 4),
                "dir_norm":  round(v.norm().item(), 4),
            }
        probe[b] = per_layer
        directions[b] = dirs

    os.makedirs(os.path.dirname(args.out_dir_dir), exist_ok=True)
    torch.save(directions, args.out_dir_dir)
    os.makedirs(os.path.dirname(args.out_probe), exist_ok=True)
    with open(args.out_probe, "w") as f:
        json.dump(probe, f, indent=2)

    # summary
    print("\n=== Best layer per behaviour (by val_acc) ===")
    for b in behaviours:
        best = max(probe[b].items(), key=lambda kv: (kv[1]["val_acc"], kv[1]["train_acc"]))
        print(f"  {b:>20s}: layer {best[0]:>2}  val_acc={best[1]['val_acc']}  train_acc={best[1]['train_acc']}  gap={best[1]['sep_gap']}")

if __name__ == "__main__":
    main()
