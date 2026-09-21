#!/usr/bin/env python3
"""M3 — Location: purify raw directions into pure directions via decorrelation (C2).

  P-A Gram-Schmidt: orthogonalise v_hat_V against {v_hat_W : W != V} at layer ell_V*.
  P-B LEACE closed-form: fit LEACE (Belrose et al. 2023) projection Pi_V that
      removes the linear signal of the 3 non-V variables on training activations,
      then Pi_V @ v_hat_V is the pure direction.

  Then for each pure direction:
    * 4x4 cross-leakage matrix: for each W in {G,A,I,M}, fit a linear probe on
      held-out activations *projected onto v_tilde_V (scalar feature)* to predict W.
      A pure direction should decode V well (~diagonal) and fail to decode W!=V
      (near chance ~ off-diagonal).
    * Preservation-of-self check: probe(V) on projection onto v_tilde_V vs raw baseline.
    * Norm audit: ||v_tilde_V^decorr|| / ||v_hat_V||.

Outputs (under --out):
  directions_pure.pt         -- {V -> {"gs": tensor(hidden,), "leace": tensor(hidden,),
                                        "raw": tensor(hidden,), "mean_centered": tensor(hidden,)}}
                                 at V's picked layer ell_V*, in per-V dict.
  leakage_matrix.json        -- {decorrelator -> [[V_row, W_col, probe_acc]...]}
  preservation.json          -- {V -> {raw: acc, mc: acc, gs: acc, leace: acc}}
  norm_audit.json            -- {V -> {gs_ratio, leace_ratio, raw_norm, gs_norm, leace_norm}}
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def gram_schmidt_purify(v_targets, V_key):
    """Return v_target orthogonalised against the other three.

    v_targets: dict[V] -> torch.Tensor[hidden]
    Standard modified Gram-Schmidt: subtract projections onto {v_W : W != V}.
    We first orthonormalise the "others" among themselves, then project.
    """
    order = ["G", "A", "I", "M"]
    others = [W for W in order if W != V_key]
    U = []
    for W in others:
        u = v_targets[W].clone().float()
        for uk in U:
            u = u - (u @ uk) * uk
        n = u.norm()
        if n > 1e-6:
            U.append(u / n)
    v = v_targets[V_key].clone().float()
    for uk in U:
        v = v - (v @ uk) * uk
    return v


def leace_purify(v_target, X_train, y_train_targets, V_key,
                 shrinkage=None):
    """Return LEACE-projected v_target using the concept_erasure library.

    Fits a LeaceFitter that erases the linear signal of the *other three* variables
    (a joint categorical label constructed from W in order != V), then applies the
    projection to v_target.
    """
    from concept_erasure import LeaceFitter
    # Build joint label of the "other three" variables so LEACE erases their combined signal.
    order = ["G", "A", "I", "M"]
    other_keys = [W for W in order if W != V_key]
    # Encode joint categorical of the three others.
    def encode(y_dict):
        out = np.zeros(len(y_dict[V_key]), dtype=np.int64)
        for i in range(len(out)):
            code = 0
            for k, W in enumerate(other_keys):
                code |= (int(y_dict[W][i]) & 1) << k
            out[i] = code
        return out
    z = encode(y_train_targets)  # [N,] with values 0..7
    # LeaceFitter expects one-hot of z.
    K = 8
    z_oh = torch.zeros((len(z), K), dtype=torch.float64)
    z_oh[np.arange(len(z)), z] = 1.0
    X64 = X_train.to(torch.float64)
    fitter = LeaceFitter(X64.shape[1], K, dtype=torch.float64)
    fitter.update(X64, z_oh)
    eraser = fitter.eraser
    # Apply eraser (matrix) to v_target: v_tilde = P @ v.
    v = v_target.detach().to(torch.float64).clone()
    # concept_erasure's eraser: eraser(x) returns erased x.  We can also access
    # eraser.proj_left, proj_right, bias.  Simpler: apply to a batch of one vector.
    v_erased = eraser(v.unsqueeze(0)).squeeze(0)
    return v_erased.to(torch.float32), eraser


def probe_from_projection(proj_scalar, y_bin, folds=5):
    """1-D logistic probe on scalar feature. Return accuracy.

    proj_scalar: [N,], y_bin: [N,] in {0,1}
    """
    from sklearn.model_selection import StratifiedKFold
    X = proj_scalar.reshape(-1, 1)
    y = np.asarray(y_bin)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help="artifacts/m2/directions_raw.pt")
    ap.add_argument("--train_acts", required=True,
                    help="artifacts/m2/train_activations.pt")
    ap.add_argument("--held_acts", required=True,
                    help="artifacts/m2/heldout_activations.pt")
    ap.add_argument("--layer_pick", required=True,
                    help="artifacts/m2/layer_pick.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="gs,leace")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    raw = torch.load(args.raw, map_location="cpu", weights_only=False)
    train_pack = torch.load(args.train_acts, map_location="cpu", weights_only=False)
    held_pack = torch.load(args.held_acts, map_location="cpu", weights_only=False)
    with open(args.layer_pick) as f:
        pick = json.load(f)
    layer_ids = raw["layer_ids"]
    layer_to_idx = {li: i for i, li in enumerate(layer_ids)}

    pos_side_map = {"G": "male", "A": "young", "I": "take-frame", "M": "no-meet"}

    # Assemble per-layer target vector.  For each V, use its picked ell_V*.
    directions_pure = {}      # V -> {method: tensor}
    leakage = {"raw": [], "mean_centered": [], "gs": [], "leace": []}
    preservation = {}
    norm_audit = {}

    # Prepare labels
    train_meta = train_pack["meta"]
    held_meta = held_pack["meta"]

    y_train = {W: np.array([1 if r[W] == pos_side_map[W] else 0
                            for r in train_meta])
               for W in ["G", "A", "I", "M"]}
    y_held = {W: np.array([1 if r[W] == pos_side_map[W] else 0
                           for r in held_meta])
              for W in ["G", "A", "I", "M"]}

    for V in ["G", "A", "I", "M"]:
        if V not in pick:
            print(f"[m3] skipping V={V}: no layer pick")
            continue
        ell_star = pick[V]["ell_star"]
        li = layer_to_idx[ell_star]
        # 4 raw directions at this same layer (for GS orthogonalisation) — but the
        # plan says per-V layer ell_V* may differ across V.  For the *diagonal* leakage
        # entry we always use ell_V*.  For the GS orthogonalisation against W != V,
        # we take W's own raw at ell_V* (not W's picked layer): this is what the plan
        # actually asks for — "orthogonalise v_hat_V^ell_V* against {v_hat_W^ell_V*}".
        v_targets_at_lstar = {
            W: raw["raw"][W][li].clone() if W in raw["raw"] else torch.zeros(raw["hidden"])
            for W in ["G", "A", "I", "M"]
        }
        v_raw = v_targets_at_lstar[V]
        v_mc = raw["mean_centered"][V][li] if V in raw["mean_centered"] else v_raw.clone()

        methods = args.methods.split(",")
        pure = {"raw": v_raw, "mean_centered": v_mc}
        if "gs" in methods:
            v_gs = gram_schmidt_purify(v_targets_at_lstar, V)
            pure["gs"] = v_gs
        if "leace" in methods:
            X_train_l = train_pack["acts"][:, li, :].float()
            try:
                v_leace, _eraser = leace_purify(v_raw, X_train_l, y_train, V)
            except Exception as e:
                print(f"[m3] LEACE failed for V={V}: {e}. Falling back to GS.")
                v_leace = pure.get("gs", v_raw)
            pure["leace"] = v_leace
        directions_pure[V] = pure
        # -- Norm audit ---------------------------------------------------------- #
        raw_norm = float(v_raw.norm())
        gs_ratio = float(pure.get("gs", torch.zeros_like(v_raw)).norm() / max(raw_norm, 1e-9))
        leace_ratio = float(pure.get("leace", torch.zeros_like(v_raw)).norm() / max(raw_norm, 1e-9))
        norm_audit[V] = dict(
            ell_star=ell_star,
            raw_norm=raw_norm,
            mc_norm=float(v_mc.norm()),
            gs_norm=float(pure.get("gs", torch.zeros_like(v_raw)).norm()),
            leace_norm=float(pure.get("leace", torch.zeros_like(v_raw)).norm()),
            gs_ratio=gs_ratio,
            leace_ratio=leace_ratio,
        )

        # -- 4x4 leakage matrix at held-out (scalar-projection probe) ------------ #
        h_held = held_pack["acts"][:, li, :].float()  # [n_held, H]
        for method_key, v in [("raw", v_raw), ("mean_centered", v_mc),
                              ("gs", pure.get("gs")), ("leace", pure.get("leace"))]:
            if v is None or v.norm() < 1e-6:
                continue
            unit = v / (v.norm() + 1e-9)
            proj = (h_held @ unit).numpy()  # [n_held,]
            for W in ["G", "A", "I", "M"]:
                acc = probe_from_projection(proj, y_held[W])
                leakage[method_key].append([V, W, acc])

        # -- Preservation: (probe accuracy on V from projection) --------------- #
        preservation.setdefault(V, {})
        # Reference: raw-layer full-vector probe from M2 (approximate via train fit).
        # We already have leakage matrix diag for each method.  Extract it.
        for method_key in ["raw", "mean_centered", "gs", "leace"]:
            hits = [row for row in leakage[method_key]
                    if row[0] == V and row[1] == V]
            preservation[V][method_key] = hits[0][2] if hits else None

    # -- Save ------------------------------------------------------------------- #
    torch.save(
        dict(pure=directions_pure, layer_pick={V: pick[V]["ell_star"] for V in pick}),
        os.path.join(args.out, "directions_pure.pt"),
    )
    with open(os.path.join(args.out, "leakage_matrix.json"), "w") as f:
        json.dump(leakage, f, indent=2)
    with open(os.path.join(args.out, "preservation.json"), "w") as f:
        json.dump(preservation, f, indent=2)
    with open(os.path.join(args.out, "norm_audit.json"), "w") as f:
        json.dump(norm_audit, f, indent=2)
    print("[m3] done")


if __name__ == "__main__":
    main()
