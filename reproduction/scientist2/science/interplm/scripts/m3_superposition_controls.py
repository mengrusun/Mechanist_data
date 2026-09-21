"""M3 — superposition specificity controls.

Compares SAE-coverage to three matched controls sharing the F1 protocol:
  Arm A — random-orthogonal rotation of the raw residual stream (per-layer, one rotation per seed).
  Arm B — PCA basis over the concatenated activations from the SAE-training pool proxy
          (here: the same Swiss-Prot residues used in M2).
  Arm C — shuffled-SAE: take SAE codes and randomly permute each feature's per-residue activations.

Outputs:
  runs/m3/superposition_ladder.json    six-arm coverage table + margins
  runs/m3/summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    SAE_LAYERS,
    SWISSPROT_DAT,
    SWISSPROT_DIR,
    build_residue_labels,
    parse_swissprot_annotations,
    seed_all,
)
from m2_concept_alignment import (  # noqa: E402
    build_concept_mask,
    compute_neuron_topmask,
    coverage_from_bestf1,
    f1_matrix_dense,
    f1_matrix_sparse,
    load_layer,
    sparse_topmask_to_bool,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=SAE_LAYERS)
    p.add_argument("--activations-dir", type=str, default="runs/m1/activations")
    p.add_argument("--codes-dir", type=str, default="runs/m1/sae_codes")
    p.add_argument("--swissprot-dir", type=str, default=str(SWISSPROT_DIR))
    p.add_argument("--m2-dir", type=str, default="runs/m2")
    p.add_argument("--out-dir", type=str, default="runs/m3")
    p.add_argument("--q-top", type=float, default=0.99)
    p.add_argument("--tau-f1", type=float, default=0.5)
    p.add_argument("--tau-clean", type=float, default=0.7)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--min-concept-support", type=int, default=25)
    return p.parse_args()


def rotate_activations(H: torch.Tensor, seed: int, device: str = "cuda") -> torch.Tensor:
    """Right-multiply H by a random orthogonal matrix Q [d,d]. Uses GPU when available."""
    d = H.shape[1]
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    Q = torch.from_numpy(Q).to(H.dtype).to(device)
    # Chunk H over N to keep GPU memory bounded
    N = H.shape[0]
    out = torch.empty_like(H)
    CHUNK = 65536
    for i in range(0, N, CHUNK):
        block = H[i : i + CHUNK].to(device)
        out[i : i + CHUNK] = (block @ Q).cpu()
        del block
    return out


def pca_project(H: torch.Tensor, seed: int, n_comp: int = None, device: str = "cuda") -> torch.Tensor:
    """PCA-project H to a d-dim basis. GPU path for both eigendecomp and projection."""
    d = H.shape[1]
    if n_comp is None:
        n_comp = d
    N = H.shape[0]
    # Compute mean on CPU (cheap), then subtract
    mean = H.float().mean(dim=0, keepdim=True)  # [1, d]
    mean_gpu = mean.to(device)
    # Subsample for eigendecomp
    if N > 200000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=200000, replace=False)
        Hs = H[idx].float().to(device)  # [200k, d]
    else:
        Hs = H.float().to(device)
    Hs = Hs - mean_gpu
    # Cov: [d, d] on GPU
    cov = (Hs.T @ Hs) / max(Hs.shape[0] - 1, 1)
    del Hs
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, order][:, :n_comp].to(H.dtype)  # [d, n_comp]
    del cov, eigvals
    # Project full H (chunked)
    out = torch.empty((N, n_comp), dtype=H.dtype)
    CHUNK = 65536
    for i in range(0, N, CHUNK):
        block = (H[i : i + CHUNK].to(device) - mean_gpu.to(H.dtype)) @ eigvecs
        out[i : i + CHUNK] = block.cpu()
        del block
    return out


def shuffled_sae_topmask(codes: dict, seed: int) -> torch.sparse.Tensor:
    """Permute each feature's per-residue firings across residues (destroys residue alignment)."""
    shape = codes["shape"]
    N, F = int(shape[0]), int(shape[1])
    row = codes["row"]
    col = codes["col"]
    rng = np.random.default_rng(seed)
    new_row = np.empty_like(row)
    # For each feature col, permute the rows
    order = np.argsort(col, kind="stable")
    col_sorted = col[order]
    row_sorted = row[order]
    # Group by col value
    unique_cols, starts = np.unique(col_sorted, return_index=True)
    ends = np.append(starts[1:], len(col_sorted))
    for c, s, e in zip(unique_cols, starts, ends):
        cnt = e - s
        # Draw cnt distinct rows from [0, N) uniformly
        new_rows = rng.choice(N, size=cnt, replace=False)
        new_row[order[s:e]] = new_rows
    idx = torch.stack([torch.from_numpy(new_row.astype(np.int64)),
                       torch.from_numpy(col.astype(np.int64))], dim=0)
    vals = torch.ones(new_row.size, dtype=torch.float32)
    sp = torch.sparse_coo_tensor(idx, vals, size=(N, F)).coalesce()
    return sp


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir = Path(args.activations_dir)
    codes_dir = Path(args.codes_dir)
    sw_dir = Path(args.swissprot_dir)
    m2_dir = Path(args.m2_dir)

    t0 = time.time()

    # Load first layer for entry / seq / offsets
    first_layer = args.layers[0]
    act0, _ = load_layer(first_layer, act_dir, codes_dir)
    ordered_entries = list(act0["entries"])
    ordered_sequences = list(act0["sequences"])
    offsets = np.array(act0["offsets"], dtype=np.int64)
    N_total = int(offsets.sum())
    S = len(ordered_entries)
    print(f"[m3] entries={S} residues={N_total}", flush=True)

    # Parse Swiss-Prot annotations
    entries_set = set(ordered_entries)
    seq_map = {e: s for e, s in zip(ordered_entries, ordered_sequences)}
    dat_path = Path(os.environ.get("SWISSPROT_DAT", str(SWISSPROT_DAT)))
    annotations = parse_swissprot_annotations(dat_path, entries_to_keep=entries_set)
    per_entry_labels, concept_universe_full = build_residue_labels(seq_map, annotations)

    concept_to_idx = {c: i for i, c in enumerate(concept_universe_full)}
    concept_mask_full, per_concept_pos, eligible_mask = build_concept_mask(
        per_entry_labels, concept_universe_full, ordered_entries, ordered_sequences, offsets, concept_to_idx
    )
    keep = per_concept_pos >= args.min_concept_support
    concept_universe = [c for c, k in zip(concept_universe_full, keep) if k]
    concept_mask = concept_mask_full[:, np.where(keep)[0]]
    C = len(concept_universe)
    print(f"[m3] concept universe (filtered)={C} eligible residues={int(eligible_mask.sum())}", flush=True)

    # Per-arm coverage aggregated over layers, mean-over-seeds where seeded
    def new_best():
        return np.full(C, 0.0, dtype=np.float32)

    per_arm_best_f1: Dict[str, np.ndarray] = {
        "SAE": new_best(),
        "neurons": new_best(),
        "PCA": new_best(),
    }
    # Per-seed for random-rotation & shuffled-SAE (we'll report mean over seeds)
    random_rot_best_per_seed = {s: new_best() for s in args.seeds}
    shuffled_sae_best_per_seed = {s: new_best() for s in args.seeds}

    for layer in args.layers:
        tL = time.time()
        act, codes = load_layer(layer, act_dir, codes_dir)
        assert list(act["entries"]) == ordered_entries
        H = act["H"]  # [N, d] fp16
        # SAE arm (cached)
        sp = sparse_topmask_to_bool(codes)
        f1_sae = f1_matrix_sparse(sp, concept_mask, eligible_mask=eligible_mask)
        per_arm_best_f1["SAE"] = np.maximum(per_arm_best_f1["SAE"], f1_sae.max(dim=0).values.numpy())
        del sp, f1_sae

        # Neuron arm
        nm = compute_neuron_topmask(H, args.q_top)
        f1_n = f1_matrix_dense(nm, concept_mask, eligible_mask=eligible_mask)
        per_arm_best_f1["neurons"] = np.maximum(per_arm_best_f1["neurons"], f1_n.max(dim=0).values.numpy())
        del nm, f1_n

        # PCA arm — one deterministic rotation (seed=0 for reproducibility; PCA is deterministic anyway)
        Hp = pca_project(H, seed=42)
        pm = compute_neuron_topmask(Hp, args.q_top)
        f1_p = f1_matrix_dense(pm, concept_mask, eligible_mask=eligible_mask)
        per_arm_best_f1["PCA"] = np.maximum(per_arm_best_f1["PCA"], f1_p.max(dim=0).values.numpy())
        del Hp, pm, f1_p

        # Random-rotation arm — per seed
        for seed in args.seeds:
            Hr = rotate_activations(H, seed)
            rm = compute_neuron_topmask(Hr, args.q_top)
            f1_r = f1_matrix_dense(rm, concept_mask, eligible_mask=eligible_mask)
            random_rot_best_per_seed[seed] = np.maximum(
                random_rot_best_per_seed[seed], f1_r.max(dim=0).values.numpy()
            )
            del Hr, rm, f1_r

        # Shuffled-SAE arm — per seed
        for seed in args.seeds:
            sp_s = shuffled_sae_topmask(codes, seed)
            f1_s = f1_matrix_sparse(sp_s, concept_mask, eligible_mask=eligible_mask)
            shuffled_sae_best_per_seed[seed] = np.maximum(
                shuffled_sae_best_per_seed[seed], f1_s.max(dim=0).values.numpy()
            )
            del sp_s, f1_s

        print(f"[m3] layer {layer} done in {time.time()-tL:.1f}s", flush=True)
        del H

    # Convert per-seed to mean-across-seeds coverage
    random_rot_covs = [coverage_from_bestf1(v, args.tau_f1) for v in random_rot_best_per_seed.values()]
    shuffled_sae_covs = [coverage_from_bestf1(v, args.tau_f1) for v in shuffled_sae_best_per_seed.values()]

    coverage = {
        "SAE": coverage_from_bestf1(per_arm_best_f1["SAE"], args.tau_f1),
        "PCA": coverage_from_bestf1(per_arm_best_f1["PCA"], args.tau_f1),
        "random_rotation_mean": float(np.mean(random_rot_covs)),
        "random_rotation_std": float(np.std(random_rot_covs)),
        "neurons": coverage_from_bestf1(per_arm_best_f1["neurons"], args.tau_f1),
        "shuffled_SAE_mean": float(np.mean(shuffled_sae_covs)),
        "shuffled_SAE_std": float(np.std(shuffled_sae_covs)),
    }
    clean = {
        "SAE": coverage_from_bestf1(per_arm_best_f1["SAE"], args.tau_clean),
        "PCA": coverage_from_bestf1(per_arm_best_f1["PCA"], args.tau_clean),
        "random_rotation_mean": float(np.mean([coverage_from_bestf1(v, args.tau_clean) for v in random_rot_best_per_seed.values()])),
        "neurons": coverage_from_bestf1(per_arm_best_f1["neurons"], args.tau_clean),
        "shuffled_SAE_mean": float(np.mean([coverage_from_bestf1(v, args.tau_clean) for v in shuffled_sae_best_per_seed.values()])),
    }
    margins = {
        "SAE_over_PCA": coverage["SAE"] - coverage["PCA"],
        "SAE_over_neurons": coverage["SAE"] - coverage["neurons"],
        "SAE_over_random_rotation": coverage["SAE"] - coverage["random_rotation_mean"],
        "SAE_over_shuffled_SAE": coverage["SAE"] - coverage["shuffled_SAE_mean"],
        "PCA_over_shuffled_SAE": coverage["PCA"] - coverage["shuffled_SAE_mean"],
    }
    # Ordered check
    order_ok = (
        coverage["SAE"] > coverage["PCA"] > coverage["shuffled_SAE_mean"] and
        coverage["SAE"] > coverage["neurons"] > coverage["shuffled_SAE_mean"] and
        margins["SAE_over_PCA"] >= 20
    )

    out = {
        "coverage@0.5": coverage,
        "clean@0.7": clean,
        "margins": margins,
        "order_check_pass": bool(order_ok),
        "primary_setting": {"q_top": args.q_top, "tau_f1": args.tau_f1, "tau_clean": args.tau_clean},
        "n_concepts": C,
        "seeds": args.seeds,
        "layers": args.layers,
        "wall_clock_seconds": time.time() - t0,
    }
    (out_dir / "superposition_ladder.json").write_text(json.dumps(out, indent=2))
    md = [
        "# M3 — Superposition Specificity Ladder\n",
        f"- N concepts scored: {C}",
        f"- Primary setting: q_top={args.q_top}, tau_F1={args.tau_f1}",
        "",
        "## Six-arm coverage (union over layers)",
        "| Arm | covered@0.5 | clean@0.7 |",
        "|---|---|---|",
        f"| SAE                     | {coverage['SAE']} | {clean['SAE']} |",
        f"| PCA                     | {coverage['PCA']} | {clean['PCA']} |",
        f"| Random rotation (mean)  | {coverage['random_rotation_mean']:.1f} ± {coverage['random_rotation_std']:.1f} | {clean['random_rotation_mean']:.1f} |",
        f"| Neurons                 | {coverage['neurons']} | {clean['neurons']} |",
        f"| Shuffled-SAE (mean)     | {coverage['shuffled_SAE_mean']:.1f} ± {coverage['shuffled_SAE_std']:.1f} | {clean['shuffled_SAE_mean']:.1f} |",
        "",
        "## Margins",
        "| Contrast | Δ |",
        "|---|---|",
    ]
    for k, v in margins.items():
        md.append(f"| {k} | {v:.2f} |")
    md.append("")
    md.append(f"## Ordered check (SAE > PCA≈random≈neurons > shuffled-SAE, Δ_SAE-PCA ≥ 20): **{order_ok}**")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    elapsed = time.time() - t0
    n_gpu = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    (out_dir / "gpu_hours.txt").write_text(f"{elapsed / 3600 * n_gpu:.4f}\n")
    print(f"[m3] done in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
