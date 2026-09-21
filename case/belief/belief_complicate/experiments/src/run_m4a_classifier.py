"""M4.a + M4.b — Pre-head layer sweep, frame-classifier training on belief_core,
OOD eval on belief_holdout.

Protocol:
  * pre_head_layers = layers strictly BEFORE the earliest Claim-2 head layer.
  * For each candidate layer L: extract last-token residual-stream state on
    every belief_core item (three frames combined = 1,589 items).  Label = frame.
  * Train logistic-regression classifier on an 80/20 stratified split of belief_core
    per seed.
  * Pick L* = highest belief_core-val macro-F1.
  * Save classifier at L*; evaluate on belief_holdout (OOD).

Output: results/M4/<model>/classifier_layer_sweep_seed<seed>.json  (M4.a)
        results/M4/<model>/classifier_ood_seed<seed>.json          (M4.b)
        results/M4/<model>/classifier_L<L*>_seed<seed>.pt          (fitted linear head)
"""
import argparse, json, os
import numpy as np
import torch
from belief_lib import (BELIEF_CORE_DIR, BELIEF_HOLDOUT_DIR, MODEL_META,
                        FRAME_FILE, load_frame, load_model, save_json, set_seed)


@torch.no_grad()
def extract_residual_last_token(model, tok, items, device, layer_indices):
    """Return dict[layer L] -> numpy array [N, hidden] of last-token residual states."""
    out = {L: [] for L in layer_indices}
    for it in items:
        p = tok(it["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        out_hs = model(p, output_hidden_states=True).hidden_states  # tuple[L+1] of [1,T,H]
        # hidden_states[L] = pre-attn residual of layer L (input to layer L block)
        # hidden_states[0] = embeddings
        # Convention: we treat "layer L residual" as hidden_states[L] (the input to
        # the L-th block).  This matches "pre-head" — reading BEFORE any head at layer L.
        for L in layer_indices:
            out[L].append(out_hs[L][0, -1, :].float().cpu().numpy())
    for L in layer_indices:
        out[L] = np.stack(out[L], axis=0)
    return out


def stratified_split_indices(y, seed, frac_train=0.8):
    """Return (train_idx, val_idx) with stratified split by y."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    train, val = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * frac_train))
        train.extend(idx[:cut].tolist())
        val.extend(idx[cut:].tolist())
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64)


def fit_logreg(X_tr, y_tr, X_va, y_va, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed,
                            n_jobs=1)
    clf.fit(X_tr, y_tr)
    y_hat_va = clf.predict(X_va)
    acc = accuracy_score(y_va, y_hat_va)
    mf1 = f1_score(y_va, y_hat_va, average="macro")
    return clf, acc, mf1, y_hat_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--headset_personal", default=None,
                    help="M2.c personal_belief headset JSON")
    ap.add_argument("--headset_attributed", default=None,
                    help="M2.c attributed_belief headset JSON")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_sweep", required=True)
    ap.add_argument("--out_ood", required=True)
    ap.add_argument("--out_ckpt", required=True, help="fitted classifier at L* (.pt)")
    ap.add_argument("--min_headset_layer", type=int, default=None,
                    help="Override: pre-head layers = 0 .. min_headset_layer-1")
    args = ap.parse_args()

    set_seed(args.seed)

    # Determine pre-head cutoff
    min_hs_layer = args.min_headset_layer
    if min_hs_layer is None:
        candidate_layers = []
        for jsonpath in (args.headset_personal, args.headset_attributed):
            if jsonpath is None:
                continue
            with open(jsonpath, "r") as fh:
                data = json.load(fh)
            hs = data.get("head_set") or []
            if hs:
                candidate_layers.extend([int(L) for (L, H) in hs])
        if not candidate_layers:
            raise ValueError("No M2 head sets available; cannot compute pre-head cutoff")
        min_hs_layer = int(min(candidate_layers))
    print(f"[M4a] min_headset_layer={min_hs_layer}", flush=True)
    if min_hs_layer <= 0:
        raise ValueError(f"min_headset_layer={min_hs_layer} — no pre-head layers exist "
                        "for this model")

    meta = MODEL_META[args.model]
    pre_head_layers = list(range(0, min_hs_layer))  # strictly BEFORE earliest Claim-2 head
    print(f"[M4a] pre_head_layers={pre_head_layers}", flush=True)

    # Load belief_core (all 3 frames combined) and belief_holdout (all 3 frames combined)
    print("[M4a] loading data …", flush=True)
    core_items, core_labels = [], []
    for frame in ["world_knowledge", "personal_belief", "attributed_belief"]:
        f_items = load_frame(frame, root=BELIEF_CORE_DIR)
        core_items.extend(f_items)
        core_labels.extend([frame] * len(f_items))
    print(f"[M4a] belief_core: {len(core_items)} items", flush=True)

    hold_items, hold_labels = [], []
    for frame in ["world_knowledge", "personal_belief", "attributed_belief"]:
        f_items = load_frame(frame, root=BELIEF_HOLDOUT_DIR)
        hold_items.extend(f_items)
        hold_labels.extend([frame] * len(f_items))
    print(f"[M4a] belief_holdout: {len(hold_items)} items", flush=True)

    # Extract last-token residual states
    device = "cuda"
    print("[M4a] loading model …", flush=True)
    model, tok, meta = load_model(args.model, device=device, dtype=torch.float16)
    print("[M4a] extracting belief_core residuals …", flush=True)
    core_hs = extract_residual_last_token(model, tok, core_items, device, pre_head_layers)
    print("[M4a] extracting belief_holdout residuals …", flush=True)
    hold_hs = extract_residual_last_token(model, tok, hold_items, device, pre_head_layers)

    label_map = {"world_knowledge": 0, "personal_belief": 1, "attributed_belief": 2}
    inv_label = {v: k for k, v in label_map.items()}
    y_core = np.array([label_map[x] for x in core_labels])
    y_hold = np.array([label_map[x] for x in hold_labels])

    tr_idx, va_idx = stratified_split_indices(y_core, seed=args.seed, frac_train=0.8)

    per_layer = {}
    for L in pre_head_layers:
        X_tr, X_va = core_hs[L][tr_idx], core_hs[L][va_idx]
        y_tr, y_va = y_core[tr_idx], y_core[va_idx]
        clf, acc, mf1, y_hat_va = fit_logreg(X_tr, y_tr, X_va, y_va, seed=args.seed)
        per_layer[L] = {"val_accuracy": float(acc), "val_macro_f1": float(mf1)}
        print(f"[M4a] layer L={L}  val_acc={acc:.4f}  val_mf1={mf1:.4f}", flush=True)

    # Pick L* = argmax val_macro_f1 (tiebreak: earlier layer wins, matching plan's preference)
    best_L = max(per_layer, key=lambda L: (per_layer[L]["val_macro_f1"], -L))
    print(f"[M4a] best L* = {best_L}  val_mf1={per_layer[best_L]['val_macro_f1']:.4f}", flush=True)

    # Refit the classifier on ALL of belief_core at L* (train+val) and save
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    clf_full = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed,
                                 n_jobs=1)
    clf_full.fit(core_hs[best_L], y_core)

    # OOD eval on belief_holdout at L*
    y_hat_ood = clf_full.predict(hold_hs[best_L])
    ood_acc = float(accuracy_score(y_hold, y_hat_ood))
    ood_mf1 = float(f1_score(y_hold, y_hat_ood, average="macro"))
    ood_per_frame_recall = {}
    for f, c in label_map.items():
        mask = y_hold == c
        if mask.sum() > 0:
            ood_per_frame_recall[f] = float(np.mean(y_hat_ood[mask] == c))
        else:
            ood_per_frame_recall[f] = None
    cm = confusion_matrix(y_hold, y_hat_ood, labels=[0, 1, 2]).tolist()

    save_json({"model": args.model, "seed": args.seed,
               "pre_head_layers": pre_head_layers,
               "min_headset_layer": min_hs_layer,
               "per_layer": {int(L): v for L, v in per_layer.items()},
               "best_layer_L_star": int(best_L),
               "val_split_frac": 0.8}, args.out_sweep)
    save_json({"model": args.model, "seed": args.seed,
               "best_layer_L_star": int(best_L),
               "ood_accuracy": ood_acc,
               "ood_macro_f1": ood_mf1,
               "ood_per_frame_recall": ood_per_frame_recall,
               "ood_confusion_matrix_labels": ["world_knowledge", "personal_belief", "attributed_belief"],
               "ood_confusion_matrix": cm,
               "n_ood_items": int(len(y_hold))}, args.out_ood)

    # Save sklearn classifier (small — coef_ and intercept_)
    ckpt = {
        "coef": clf_full.coef_.astype(np.float32),      # [n_classes, hidden]
        "intercept": clf_full.intercept_.astype(np.float32),  # [n_classes]
        "classes": clf_full.classes_.astype(np.int64),
        "label_map": label_map,
        "best_layer_L_star": int(best_L),
        "model": args.model,
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    torch.save(ckpt, args.out_ckpt)
    print(f"[M4a] DONE  L*={best_L}  ood_mf1={ood_mf1:.4f}", flush=True)


if __name__ == "__main__":
    main()
