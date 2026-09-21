"""Claim 3: Charge is linearly encoded early in ESMFold + causally influences β-hairpin formation.

Two parts:
(A) Linear probe for per-residue charge from s at each block. Train logistic
    regression on a set of proteins; report per-block accuracy on held-out set.
(B) Causal steering: pick β-hairpin templates where the two facing positions
    are same-charge (or neutral); apply a charge-direction steering vector at
    those residues in early blocks and measure whether β-hairpin propensity or
    cross-strand Cα distance changes vs baseline. Also perform a positive
    control: mutate residues to actual opposite/same charges.

Writes:
    outputs/claim3_probe_acc.json   — per-block probe test accuracy
    outputs/claim3_direction.pt     — learned charge direction per block
    outputs/claim3_causal.jsonl     — causal steering results per chain
"""
import argparse
import json
import os
import time
from pathlib import Path
import gc

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression

from common import (load_esmfold, esmfold_infer, output_to_pdb_str,
                    run_dssp, dssp_ss_string, write_pdb, aa_charge, OUT_DIR)
from intervene import capture_block_outputs, add_direction_to_s


def check_hairpin(ss, hp, tol_shift=2, min_e=2):
    s1s, s1e, ls, le, s2s, s2e = hp
    e_in_s1 = sum(1 for c in ss[max(0, s1s - tol_shift): s1e + 1 + tol_shift] if c == 'E')
    e_in_s2 = sum(1 for c in ss[max(0, s2s - tol_shift): s2e + 1 + tol_shift] if c == 'E')
    return int((e_in_s1 >= min_e) and (e_in_s2 >= min_e))


def cross_strand_ca_dist(pdb_path, hp):
    """Mean Cα-Cα distance between the paired residues on the two facing strands.
    Pairing rule: pair from closest-to-loop outward."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", pdb_path)
    model = st[0]
    ca = {}
    idx = 0
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            if "CA" in res:
                ca[idx] = np.array(res["CA"].coord)
                idx += 1
    s1s, s1e, ls, le, s2s, s2e = hp
    strand1 = list(reversed(range(s1s, s1e + 1)))
    strand2 = list(range(s2s, s2e + 1))
    pairs = list(zip(strand1, strand2))
    ds = []
    for i, j in pairs:
        if i in ca and j in ca:
            ds.append(np.linalg.norm(ca[i] - ca[j]))
    if not ds:
        return None
    return float(np.mean(ds))


def collect_probe_data(model, tok, entries, device, num_recycles, blocks_to_probe):
    """Run each sequence, extract s_k, aggregate (aa, charge label, feature)."""
    features = {k: [] for k in blocks_to_probe}
    labels = []
    for e in entries:
        seq = e["seq"]
        cache = {}
        with capture_block_outputs(model, cache):
            _ = esmfold_infer(model, tok, seq, num_recycles=num_recycles, device=device)
        # per-residue labels: -1, 0, 1
        y = np.array([aa_charge(a) for a in seq], dtype=np.int64)
        for k in blocks_to_probe:
            s = cache[k][0].squeeze(0).cpu().numpy()  # (L, D)
            features[k].append(s)
        labels.append(y)
        del cache
        gc.collect(); torch.cuda.empty_cache()
    y_all = np.concatenate(labels)
    X_by_k = {k: np.concatenate(features[k], axis=0) for k in blocks_to_probe}
    return X_by_k, y_all


def fit_probe(X, y, seed=0):
    """Multinomial logistic regression for (-1,0,+1)."""
    idx = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]
    clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=4)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    return clf, acc


def get_charge_direction(clf, from_class=+1, to_class=-1):
    """Return direction vector that pushes probe from `from_class` to `to_class`.

    clf.coef_ shape: (n_classes, D). n_classes ordered by clf.classes_.
    """
    classes = list(clf.classes_)
    ci_from = classes.index(from_class)
    ci_to = classes.index(to_class)
    return torch.tensor(clf.coef_[ci_to] - clf.coef_[ci_from], dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(OUT_DIR / "baseline.jsonl"))
    ap.add_argument("--hairpins", default=str(Path(OUT_DIR).parent / "data" / "hairpin_chains.jsonl"))
    ap.add_argument("--probe_out", default=str(OUT_DIR / "claim3_probe_acc.json"))
    ap.add_argument("--direction_out", default=str(OUT_DIR / "claim3_direction.pt"))
    ap.add_argument("--causal_out", default=str(OUT_DIR / "claim3_causal.jsonl"))
    ap.add_argument("--tmp_dir", default="/tmp/claim3_pdb")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_recycles", type=int, default=1)
    ap.add_argument("--n_probe", type=int, default=40, help="chains for training probe")
    ap.add_argument("--n_causal", type=int, default=20, help="chains for causal test")
    ap.add_argument("--layers", type=str, default="0,2,4,6,8,10,12,16,20,24,32,40,47")
    ap.add_argument("--steer_layer", type=int, default=4)
    ap.add_argument("--steer_scales", type=str, default="-6,-4,-2,0,2,4,6")
    args = ap.parse_args()

    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.probe_out).parent.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    scales = [float(x) for x in args.steer_scales.split(",")]

    # load probe entries (any chain suffices for probe)
    probe_entries = []
    with open(args.hairpins) as f:
        for ln in f:
            e = json.loads(ln)
            probe_entries.append(e)
    probe_entries = probe_entries[: args.n_probe]

    tok, model = load_esmfold(device=args.device, dtype=torch.float32)

    # (A) collect features
    print(f"Collecting probe features from {len(probe_entries)} chains ...")
    X_by_k, y = collect_probe_data(model, tok, probe_entries, args.device, args.num_recycles, layers)
    print(f"labels dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    acc_by_k = {}
    directions = {}
    for k in layers:
        clf, acc = fit_probe(X_by_k[k], y, seed=0)
        acc_by_k[k] = float(acc)
        # get direction to push -/+ toward +/-
        d_neg = get_charge_direction(clf, +1, -1)   # +→-  (push toward negative)
        d_pos = get_charge_direction(clf, -1, +1)   # -→+
        directions[k] = {"to_neg": d_neg, "to_pos": d_pos}
        print(f"block {k}: probe acc = {acc:.3f}")

    with open(args.probe_out, "w") as f:
        json.dump(acc_by_k, f, indent=2)
    torch.save(directions, args.direction_out)

    # (B) causal steering.
    # We select chains where hairpin folds at baseline and there are 2+
    # residue pairs on facing strands that are currently neutral or same-sign.
    caus_entries = []
    with open(args.baseline) as f:
        for ln in f:
            e = json.loads(ln)
            if e.get("native_hp_ok"):
                caus_entries.append(e)
    caus_entries = caus_entries[: args.n_causal]
    print(f"causal chains: {len(caus_entries)}")

    k_star = args.steer_layer
    dir_neg = directions[k_star]["to_neg"]
    dir_pos = directions[k_star]["to_pos"]

    fo = open(args.causal_out, "w")
    for idx, e in enumerate(caus_entries):
        seq = e["seq_native"]
        hp = tuple(e["hairpin"])
        s1s, s1e, ls, le, s2s, s2e = hp
        strand1 = list(reversed(range(s1s, s1e + 1)))
        strand2 = list(range(s2s, s2e + 1))
        pairs = list(zip(strand1, strand2))

        # baseline
        out0 = esmfold_infer(model, tok, seq, num_recycles=args.num_recycles, device=args.device)
        pdb0 = output_to_pdb_str(model, out0)
        p0 = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_base.pdb"
        write_pdb(pdb0, p0)
        ss0 = dssp_ss_string(run_dssp(str(p0)))
        base_d = cross_strand_ca_dist(str(p0), hp)
        base_hp = check_hairpin(ss0, hp)

        # Interventions: at k_star, add direction to first pair (i, j)
        # Strategy A: opposite-charge steering — push i toward + and j toward −
        # Strategy B: same-charge steering — push both toward +
        # For each pair position, do both strategies at multiple scales, using
        # only the top-K pairs (up to 3) — steer_positions = list of residue idx
        top_pairs = pairs[: min(3, len(pairs))]
        pos_i = [p[0] for p in top_pairs]
        pos_j = [p[1] for p in top_pairs]

        results = {}
        for scale in scales:
            # opposite-charge steering
            with add_direction_to_s(model, [k_star], dir_pos, pos_i, scale=scale), \
                 add_direction_to_s(model, [k_star], dir_neg, pos_j, scale=scale):
                out_op = esmfold_infer(model, tok, seq, num_recycles=args.num_recycles, device=args.device)
            pdb_op = output_to_pdb_str(model, out_op)
            pop = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_op_{scale}.pdb"
            write_pdb(pdb_op, pop)
            try:
                ss_op = dssp_ss_string(run_dssp(str(pop)))
                op_d = cross_strand_ca_dist(str(pop), hp)
                op_hp = check_hairpin(ss_op, hp)
            except Exception:
                ss_op = ""; op_d = None; op_hp = -1

            # same-charge steering (both toward +)
            with add_direction_to_s(model, [k_star], dir_pos, pos_i + pos_j, scale=scale):
                out_sm = esmfold_infer(model, tok, seq, num_recycles=args.num_recycles, device=args.device)
            pdb_sm = output_to_pdb_str(model, out_sm)
            psm = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_sm_{scale}.pdb"
            write_pdb(pdb_sm, psm)
            try:
                ss_sm = dssp_ss_string(run_dssp(str(psm)))
                sm_d = cross_strand_ca_dist(str(psm), hp)
                sm_hp = check_hairpin(ss_sm, hp)
            except Exception:
                ss_sm = ""; sm_d = None; sm_hp = -1

            results[f"{scale}"] = {
                "opp_d": op_d, "opp_hp": op_hp, "opp_plddt": float(out_op["plddt"].mean().item()), "opp_ss": ss_op,
                "same_d": sm_d, "same_hp": sm_hp, "same_plddt": float(out_sm["plddt"].mean().item()), "same_ss": ss_sm,
            }

        entry = {
            "pdb": e["pdb"], "chain": e["chain"], "hairpin": hp,
            "seq": seq,
            "base_d": base_d, "base_hp": base_hp,
            "base_plddt": float(out0["plddt"].mean().item()),
            "steer_layer": k_star,
            "pairs": top_pairs,
            "results": results,
        }
        fo.write(json.dumps(entry) + "\n")
        fo.flush()
        print(f"[{idx+1}/{len(caus_entries)}] {e['pdb']}_{e['chain']} base_d={base_d:.2f} base_hp={base_hp}")
        gc.collect(); torch.cuda.empty_cache()
    fo.close()


if __name__ == "__main__":
    main()
