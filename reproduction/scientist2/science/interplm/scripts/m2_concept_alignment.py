"""M2 — concept-alignment harness: SAE-features vs raw-neurons per-residue F1 on Swiss-Prot concepts.

Loads the cached activations (H) and SAE codes (sparse top-mask) from M1 for every layer,
parses Swiss-Prot .dat annotations into per-residue concept-positive masks (restricted
to the same test-split entries M1 processed), computes F1(u, c) for every unit u and every
concept c, records per-concept best F1 (union over layers), coverage / clean-coverage,
and runs the sensitivity sweep over (q_top, tau_F1).

Outputs:
  runs/m2/coverage.json                 headline covered / clean numbers, plus per-layer breakdown
  runs/m2/sensitivity.json              full 6-point (q_top × tau_F1) sweep
  runs/m2/per_concept_best_F1.parquet   per-concept, per-arm best F1 and layer
  runs/m2/best_layer.json               argmax_L coverage(SAE), consumed by M4/M5/M6
  runs/m2/unaligned_features.json       SAE features with best_F1 < tau_F1, consumed by M4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=SAE_LAYERS)
    p.add_argument("--activations-dir", type=str, default="runs/m1/activations")
    p.add_argument("--codes-dir", type=str, default="runs/m1/sae_codes")
    p.add_argument("--swissprot-dir", type=str, default=str(SWISSPROT_DIR))
    p.add_argument("--out-dir", type=str, default="runs/m2")
    p.add_argument("--q-top", type=float, default=0.99)
    p.add_argument("--tau-f1", type=float, default=0.5)
    p.add_argument("--tau-clean", type=float, default=0.7)
    # sweep: comma-separated list of tau_F1 and q_top values
    p.add_argument("--sweep-tau", type=str, default="0.3,0.5,0.7")
    p.add_argument("--sweep-q", type=str, default="0.95,0.99")
    p.add_argument("--min-concept-support", type=int, default=25,
                   help="Minimum residue-positive count for a concept to be scored (avoids single-example concepts)")
    p.add_argument("--max-concepts", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_layer(layer: int, activations_dir: Path, codes_dir: Path):
    act_path = activations_dir / f"layer{layer}.pt"
    codes_path = codes_dir / f"layer{layer}.npz"
    act = torch.load(act_path, map_location="cpu", weights_only=False)
    codes = np.load(codes_path)
    return act, codes


def build_concept_mask(
    per_entry_labels: Dict[str, Dict[int, set]],
    concept_universe: List[str],
    ordered_entries: List[str],
    sequences: List[str],
    offsets: np.ndarray,
    concept_to_idx: Dict[str, int],
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Return concept_mask [N_residues, C_concepts] bool, per_concept_pos_count [C],
    and eligibility mask [N_residues] bool (True iff the residue's entry has any
    Swiss-Prot annotation — used to exclude unannotated residues from F1's P/R).
    """
    N = int(offsets.sum())
    C = len(concept_universe)
    mask = np.zeros((N, C), dtype=bool)
    eligible = np.zeros(N, dtype=bool)
    cursor = 0
    for si, entry in enumerate(ordered_entries):
        Li = int(offsets[si])
        labels = per_entry_labels.get(entry)
        if labels:
            # This entry is annotated — every one of its residues is eligible for F1 scoring
            eligible[cursor : cursor + Li] = True
            for r_1based, concepts in labels.items():
                r0 = r_1based - 1
                if r0 < 0 or r0 >= Li:
                    continue
                for c in concepts:
                    ci = concept_to_idx.get(c)
                    if ci is not None:
                        mask[cursor + r0, ci] = True
        cursor += Li
    per_concept_pos = mask.sum(axis=0)
    return torch.from_numpy(mask), per_concept_pos, torch.from_numpy(eligible)


def sparse_topmask_to_bool(codes: np.lib.npyio.NpzFile) -> torch.Tensor:
    """Reconstruct sparse boolean top-mask (N x F) from stored (row, col) indices."""
    shape = codes["shape"]
    N, F = int(shape[0]), int(shape[1])
    row = codes["row"]
    col = codes["col"]
    # Represent as sparse COO -> to dense bool
    # For F=10240 and N=350k, dense bool takes 350k*10240 bytes = 3.5 GB. Marginal — do it in chunks.
    # For safety, use torch sparse -> we'll compute F1 with sparse matmul below.
    idx = torch.stack([torch.from_numpy(row.astype(np.int64)), torch.from_numpy(col.astype(np.int64))], dim=0)
    vals = torch.ones(row.size, dtype=torch.float32)
    sp = torch.sparse_coo_tensor(idx, vals, size=(N, F))
    return sp.coalesce()


def compute_neuron_topmask(H: torch.Tensor, q_top: float, chunk: int = 512) -> torch.Tensor:
    """For neuron arm: threshold each dimension of H at its q_top quantile.
    Returns dense [N, d] bool on CPU (fp16 -> bool).
    """
    N, d = H.shape
    # Compute per-dim quantile in chunks (CPU float32)
    thr = torch.empty(d, dtype=torch.float32)
    Hf = H.float()
    for i in range(0, d, chunk):
        sl = Hf[:, i : i + chunk]
        thr[i : i + chunk] = torch.quantile(sl, q_top, dim=0)
    mask = Hf > thr.unsqueeze(0)  # [N, d]
    return mask


def f1_matrix_sparse(sp_unit: torch.Tensor, concept_mask: torch.Tensor,
                      eligible_mask: torch.Tensor = None) -> torch.Tensor:
    """Compute F1(u, c) matrix. sp_unit: sparse [N, F]. concept_mask: dense bool [N, C].

    If eligible_mask [N] bool is given, restrict TP/FP/FN counts to residues in the eligible
    set (per plan: residues without any annotation are excluded from both P and R denominators).
    """
    Cf = concept_mask.float()  # dense [N, C]
    if eligible_mask is not None:
        elig = eligible_mask.float()  # [N]
        # elementwise multiply concept and unit by elig — restricts contributions
        Cf = Cf * elig.unsqueeze(1)
        # For sp_unit, apply mask via coalesced sparse * dense element-wise
        # Simpler: recompute sp_unit by filtering rows
        sp_dense_row_kept = eligible_mask.to(torch.int8).nonzero(as_tuple=True)[0]
        # Build a keep-mask via coo: keep entries where row is eligible
        idx = sp_unit._indices()
        vals = sp_unit._values()
        keep = eligible_mask[idx[0]]
        idx_k = idx[:, keep]
        vals_k = vals[keep]
        sp_unit = torch.sparse_coo_tensor(idx_k, vals_k, size=sp_unit.shape).coalesce()
    sp_t = sp_unit.t().coalesce()
    tp = torch.sparse.mm(sp_t, Cf)  # [F, C]
    unit_sum = torch.sparse.sum(sp_unit, dim=0).to_dense()  # [F]
    concept_sum = Cf.sum(dim=0)  # [C]
    fp = unit_sum.unsqueeze(1) - tp
    fn = concept_sum.unsqueeze(0) - tp
    prec = tp / (tp + fp).clamp_min(1e-12)
    rec = tp / (tp + fn).clamp_min(1e-12)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
    return f1


def f1_matrix_dense(unit_mask: torch.Tensor, concept_mask: torch.Tensor,
                     eligible_mask: torch.Tensor = None) -> torch.Tensor:
    """Compute F1(u, c). unit_mask [N, U] bool, concept_mask [N, C] bool.

    If eligible_mask [N] bool is given, restrict contributions to eligible residues.
    """
    U = unit_mask.float()
    Cf = concept_mask.float()
    if eligible_mask is not None:
        elig = eligible_mask.float().unsqueeze(1)  # [N, 1]
        U = U * elig
        Cf = Cf * elig
    tp = U.T @ Cf  # [U, C]
    unit_sum = U.sum(dim=0)
    concept_sum = Cf.sum(dim=0)
    fp = unit_sum.unsqueeze(1) - tp
    fn = concept_sum.unsqueeze(0) - tp
    prec = tp / (tp + fp).clamp_min(1e-12)
    rec = tp / (tp + fn).clamp_min(1e-12)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
    return f1


def coverage_from_bestf1(best_f1_per_concept: np.ndarray, tau: float) -> int:
    return int((best_f1_per_concept >= tau).sum())


def main():
    args = parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir = Path(args.activations_dir)
    codes_dir = Path(args.codes_dir)
    sw_dir = Path(args.swissprot_dir)

    t0 = time.time()

    # 1) Load first layer to get the sequence set and offsets
    first_layer = args.layers[0]
    print(f"[m2] loading layer {first_layer} activations for entry list...", flush=True)
    act0, _ = load_layer(first_layer, act_dir, codes_dir)
    ordered_entries = list(act0["entries"])
    ordered_sequences = list(act0["sequences"])
    offsets = np.array(act0["offsets"], dtype=np.int64)
    N_total = int(offsets.sum())
    S = len(ordered_entries)
    print(f"[m2] entries={S} residues={N_total}", flush=True)

    # 2) Parse Swiss-Prot annotations for these entries. Use the full downloaded dat.gz
    # (the local Swiss-Prot dir has a truncated 50KB version).
    entries_set = set(ordered_entries)
    dat_path = Path(os.environ.get("SWISSPROT_DAT", str(SWISSPROT_DAT)))
    print(f"[m2] parsing {dat_path} restricted to {len(entries_set)} entries...", flush=True)
    seq_map = {e: s for e, s in zip(ordered_entries, ordered_sequences)}
    annotations = parse_swissprot_annotations(dat_path, entries_to_keep=entries_set)
    print(f"[m2] parsed {len(annotations)}/{S} entries with annotations", flush=True)
    per_entry_labels, concept_universe_full = build_residue_labels(seq_map, annotations)
    print(f"[m2] concept universe (raw)={len(concept_universe_full)}", flush=True)

    # 3) Build [N_total, C] concept mask, filter to concepts with sufficient support
    concept_to_idx = {c: i for i, c in enumerate(concept_universe_full)}
    concept_mask_full, per_concept_pos, eligible_mask = build_concept_mask(
        per_entry_labels, concept_universe_full, ordered_entries, ordered_sequences, offsets, concept_to_idx
    )
    print(f"[m2] eligible residues (in annotated entries): {int(eligible_mask.sum().item())}/{N_total}", flush=True)
    keep = per_concept_pos >= args.min_concept_support
    if args.max_concepts is not None:
        # keep top-max_concepts by positive count
        order = np.argsort(-per_concept_pos)
        keep_idx = order[: args.max_concepts]
        keep_bool = np.zeros(len(concept_universe_full), dtype=bool)
        keep_bool[keep_idx] = True
        keep = keep_bool & keep
    concept_universe = [c for c, k in zip(concept_universe_full, keep) if k]
    keep_idx_np = np.where(keep)[0]
    concept_mask = concept_mask_full[:, keep_idx_np]
    C = len(concept_universe)
    print(f"[m2] concept universe (filtered support>={args.min_concept_support})={C}", flush=True)

    # sweep grids
    sweep_tau = [float(x) for x in args.sweep_tau.split(",")]
    sweep_q = [float(x) for x in args.sweep_q.split(",")]

    # 4) Iterate layers; store per-(arm, layer, q, tau) coverage + per-(arm, concept) global best F1
    per_layer_summary = {}
    # per-concept best F1 aggregated over layers at PRIMARY setting (q_top, tau_F1) — using q from args
    best_f1_sae = np.full(C, 0.0, dtype=np.float32)
    best_layer_sae = np.full(C, -1, dtype=np.int32)
    best_feature_sae = np.full(C, -1, dtype=np.int32)
    best_f1_neuron = np.full(C, 0.0, dtype=np.float32)
    best_layer_neuron = np.full(C, -1, dtype=np.int32)
    best_dim_neuron = np.full(C, -1, dtype=np.int32)

    # For sensitivity sweep: coverage per (arm, q, tau, layer). Since q_top affects unit binarization,
    # and current codes are stored at q=0.99, sweeping q_top requires recomputing per feature.
    # For sae: we stored raw thr at q=0.99. For q in sweep (0.95, 0.99), recompute per feature using reservoir threshold — approximation would require re-encoding.
    # SIMPLIFICATION: We only sweep q=0.99 for SAE (uses cached codes); for q=0.95 we recompute from H via SAE forward.
    # For neurons: recompute for each q since H is available.
    # For each (arm, q, layer): keep [F,C] F1 to compute best-F1 per concept in that setting.
    # We'll accumulate sweep-level best_F1 dictionaries: sweep_best_sae[q] -> [C] best F1, aggregated over layers.
    sweep_best_sae = {q: np.full(C, 0.0, dtype=np.float32) for q in sweep_q}
    sweep_best_neuron = {q: np.full(C, 0.0, dtype=np.float32) for q in sweep_q}

    # For q_top primary, we also need the SAE codes for M3/M4 downstream.
    # Load device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preload SAE modules per layer to avoid reloading in sweep (we may reencode at q=0.95)
    from common import load_sae
    sae_cache = {}

    for layer in args.layers:
        tL = time.time()
        act, codes = load_layer(layer, act_dir, codes_dir)
        assert list(act["entries"]) == ordered_entries, f"entry mismatch on layer {layer}"
        H = act["H"]  # [N, d] fp16 CPU
        N = H.shape[0]
        assert N == N_total, f"residue count mismatch: {N} vs {N_total}"

        # Load stored SAE stats
        thr_stored = codes["thr"]  # per-feature q=0.99 threshold
        fire_frac_stored = codes.get("fire_frac")
        # === SAE arm at PRIMARY (q=0.99): use stored sparse mask ===
        sp = sparse_topmask_to_bool(codes)  # sparse [N, F]
        F_dim = int(codes["shape"][1])
        # F1 matrix [F, C]
        f1_sae = f1_matrix_sparse(sp, concept_mask, eligible_mask=eligible_mask)  # [F, C]
        # per-concept best-F1 for this layer
        best_f1_layer_sae, best_feat_layer_sae = f1_sae.max(dim=0)
        best_f1_layer_sae = best_f1_layer_sae.numpy()
        best_feat_layer_sae = best_feat_layer_sae.numpy()
        # Update global
        upd = best_f1_layer_sae > best_f1_sae
        best_f1_sae = np.where(upd, best_f1_layer_sae, best_f1_sae)
        best_layer_sae = np.where(upd, layer, best_layer_sae).astype(np.int32)
        best_feature_sae = np.where(upd, best_feat_layer_sae, best_feature_sae).astype(np.int32)
        # Also record for sweep at q=0.99
        if 0.99 in sweep_best_sae:
            sweep_best_sae[0.99] = np.maximum(sweep_best_sae[0.99], best_f1_layer_sae)
        cov_sae_layer_primary = coverage_from_bestf1(best_f1_layer_sae, args.tau_f1)
        clean_sae_layer_primary = coverage_from_bestf1(best_f1_layer_sae, args.tau_clean)

        # === Neuron arm at PRIMARY (q=0.99) ===
        neuron_mask = compute_neuron_topmask(H, args.q_top)
        f1_neuron = f1_matrix_dense(neuron_mask, concept_mask, eligible_mask=eligible_mask)  # [d, C]
        best_f1_layer_neuron, best_dim_layer_neuron = f1_neuron.max(dim=0)
        best_f1_layer_neuron = best_f1_layer_neuron.numpy()
        best_dim_layer_neuron = best_dim_layer_neuron.numpy()
        upd_n = best_f1_layer_neuron > best_f1_neuron
        best_f1_neuron = np.where(upd_n, best_f1_layer_neuron, best_f1_neuron)
        best_layer_neuron = np.where(upd_n, layer, best_layer_neuron).astype(np.int32)
        best_dim_neuron = np.where(upd_n, best_dim_layer_neuron, best_dim_neuron).astype(np.int32)
        if 0.99 in sweep_best_neuron:
            sweep_best_neuron[0.99] = np.maximum(sweep_best_neuron[0.99], best_f1_layer_neuron)
        cov_neuron_layer_primary = coverage_from_bestf1(best_f1_layer_neuron, args.tau_f1)
        clean_neuron_layer_primary = coverage_from_bestf1(best_f1_layer_neuron, args.tau_clean)

        # === Sensitivity sweep at other q_top values ===
        for q in sweep_q:
            if q == 0.99:
                continue
            # SAE arm: need to re-encode and mask at q. Use cached H + SAE, chunked
            if layer not in sae_cache:
                sae_cache[layer] = load_sae(layer, "normalized", device).half()
            sae = sae_cache[layer]
            # Estimate per-feature quantile from Z (need to run through SAE encoder). We'll compute q per feature
            # by streaming through H in chunks, collecting nonzero activations to a reservoir per candidate feature.
            # This is expensive; simplify by using only the SAE features that were UN-dead in primary (fire_frac > 0)
            # and estimating q via a fixed 256k residue sample.
            SAMPLE_N = min(N, 65536)  # smaller sample keeps CPU quantile memory bounded
            rng = np.random.default_rng(args.seed + int(layer * 100 + q * 1000))
            sample_idx = rng.choice(N, size=SAMPLE_N, replace=False)
            sample_idx.sort()
            H_sample = H[sample_idx].to(device)
            with torch.no_grad():
                Z_sample = sae.encode(H_sample).float().cpu()  # [SAMPLE_N, F] fp32 CPU
            del H_sample
            # Chunked CPU quantile to keep RAM bounded
            thr_new = torch.empty(F_dim, dtype=torch.float32)
            QCHUNK_Q = 512
            for _i in range(0, F_dim, QCHUNK_Q):
                thr_new[_i : _i + QCHUNK_Q] = torch.quantile(Z_sample[:, _i : _i + QCHUNK_Q], q, dim=0)
            del Z_sample
            # Now mask full activations
            row_list, col_list = [], []
            thr_gpu = thr_new.to(device)
            chunk = 8192
            for i in range(0, N, chunk):
                Hc = H[i : i + chunk].to(device)
                with torch.no_grad():
                    Zc = sae.encode(Hc)
                    m = Zc > thr_gpu.unsqueeze(0)
                    nz = torch.nonzero(m, as_tuple=False)
                    if nz.numel() > 0:
                        row_list.append((nz[:, 0] + i).cpu().numpy().astype(np.int32))
                        col_list.append(nz[:, 1].cpu().numpy().astype(np.int32))
                del Hc, Zc, m, nz
            if row_list:
                row = np.concatenate(row_list); col = np.concatenate(col_list)
                idx = torch.stack([torch.from_numpy(row.astype(np.int64)),
                                   torch.from_numpy(col.astype(np.int64))], dim=0)
                vals = torch.ones(row.size, dtype=torch.float32)
                sp_q = torch.sparse_coo_tensor(idx, vals, size=(N, F_dim)).coalesce()
                f1_q_sae = f1_matrix_sparse(sp_q, concept_mask, eligible_mask=eligible_mask)
                best_f1_q_sae = f1_q_sae.max(dim=0).values.numpy()
                sweep_best_sae[q] = np.maximum(sweep_best_sae[q], best_f1_q_sae)
                del sp_q, f1_q_sae
            # Neuron arm at q
            neuron_mask_q = compute_neuron_topmask(H, q)
            f1_q_n = f1_matrix_dense(neuron_mask_q, concept_mask, eligible_mask=eligible_mask)
            best_f1_q_n = f1_q_n.max(dim=0).values.numpy()
            sweep_best_neuron[q] = np.maximum(sweep_best_neuron[q], best_f1_q_n)
            del neuron_mask_q, f1_q_n

        per_layer_summary[str(layer)] = {
            "sae": {
                "coverage@0.5": int(cov_sae_layer_primary),
                "clean@0.7": int(clean_sae_layer_primary),
                "F_dim": F_dim,
            },
            "neuron": {
                "coverage@0.5": int(cov_neuron_layer_primary),
                "clean@0.7": int(clean_neuron_layer_primary),
                "d": neuron_mask.shape[1],
            },
            "elapsed_seconds": time.time() - tL,
        }
        print(f"[m2] layer {layer} sae_cov={cov_sae_layer_primary} sae_clean={clean_sae_layer_primary} "
              f"neuron_cov={cov_neuron_layer_primary} neuron_clean={clean_neuron_layer_primary} "
              f"[{time.time()-tL:.1f}s]", flush=True)
        # Free memory
        del H, sp, neuron_mask, f1_sae, f1_neuron

    # === Aggregate coverage ===
    covered_sae = coverage_from_bestf1(best_f1_sae, args.tau_f1)
    clean_sae = coverage_from_bestf1(best_f1_sae, args.tau_clean)
    covered_neuron = coverage_from_bestf1(best_f1_neuron, args.tau_f1)
    clean_neuron = coverage_from_bestf1(best_f1_neuron, args.tau_clean)

    coverage_summary = {
        "sae": {
            "covered_union": int(covered_sae),
            "clean_union": int(clean_sae),
        },
        "neuron": {
            "covered_union": int(covered_neuron),
            "clean_union": int(clean_neuron),
        },
        "ratio_covered_sae_over_neuron": (covered_sae / covered_neuron) if covered_neuron else float("nan"),
        "primary_setting": {"q_top": args.q_top, "tau_f1": args.tau_f1, "tau_clean": args.tau_clean},
        "n_concepts": C,
        "n_residues": N_total,
        "n_seqs": S,
        "per_layer": per_layer_summary,
        "layers": args.layers,
        "wall_clock_seconds": time.time() - t0,
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage_summary, indent=2))
    print(f"[m2] coverage summary written", flush=True)

    # === Sensitivity sweep JSON ===
    sensitivity = {}
    for q in sweep_q:
        for tau in sweep_tau:
            cov_sae_q_tau = coverage_from_bestf1(sweep_best_sae[q], tau)
            cov_n_q_tau = coverage_from_bestf1(sweep_best_neuron[q], tau)
            sensitivity[f"q={q}_tau={tau}"] = {
                "sae_covered": int(cov_sae_q_tau),
                "neuron_covered": int(cov_n_q_tau),
                "ratio": (cov_sae_q_tau / cov_n_q_tau) if cov_n_q_tau else float("nan"),
            }
    (out_dir / "sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
    print(f"[m2] sensitivity sweep written", flush=True)

    # === Per-concept best F1 parquet ===
    per_concept = pd.DataFrame({
        "concept": concept_universe,
        "positive_count": per_concept_pos[keep_idx_np],
        "sae_best_f1": best_f1_sae,
        "sae_best_layer": best_layer_sae,
        "sae_best_feature": best_feature_sae,
        "neuron_best_f1": best_f1_neuron,
        "neuron_best_layer": best_layer_neuron,
        "neuron_best_dim": best_dim_neuron,
    })
    per_concept.to_parquet(out_dir / "per_concept_best_F1.parquet", index=False)

    # === Best layer for downstream (M4/M5/M6): argmax_L covered(SAE) at primary ===
    per_layer_cov_sae = {lstr: v["sae"]["coverage@0.5"] for lstr, v in per_layer_summary.items()}
    best_layer = int(max(per_layer_cov_sae, key=lambda k: per_layer_cov_sae[k]))
    (out_dir / "best_layer.json").write_text(json.dumps({
        "best_layer": best_layer,
        "per_layer_coverage": per_layer_cov_sae,
    }, indent=2))
    print(f"[m2] best_layer={best_layer} coverage={per_layer_cov_sae[str(best_layer)]}", flush=True)

    # === Unaligned features (for M4) — pick features on best_layer whose best_F1 across all concepts < tau_f1 ===
    # Re-load best layer codes for feature-level unaligned mining
    act_best, codes_best = load_layer(best_layer, act_dir, codes_dir)
    sp_best = sparse_topmask_to_bool(codes_best)
    f1_best = f1_matrix_sparse(sp_best, concept_mask, eligible_mask=eligible_mask)  # [F, C]
    max_f1_per_feature = f1_best.max(dim=1).values.numpy()  # [F]
    fire_frac_best = codes_best.get("fire_frac")
    # Restrict to features with fire_frac >= 1e-6 (not dead)
    F_dim = int(codes_best["shape"][1])
    if fire_frac_best is None:
        fire_frac_best = np.ones(F_dim, dtype=np.float32)
    unaligned = np.where((max_f1_per_feature < args.tau_f1) & (fire_frac_best >= 1e-6) & (fire_frac_best <= 0.2))[0]
    (out_dir / "unaligned_features.json").write_text(json.dumps({
        "best_layer": best_layer,
        "tau_f1": args.tau_f1,
        "n_unaligned": int(len(unaligned)),
        "n_total_features": F_dim,
        "unaligned_feature_ids": unaligned.tolist()[:5000],
    }, indent=2))
    print(f"[m2] unaligned features on layer {best_layer}: {len(unaligned)}/{F_dim}", flush=True)

    # Save a compact summary MD
    md = ["# M2 — Concept Alignment Summary\n",
          f"- N sequences: {S}",
          f"- N residues: {N_total}",
          f"- N concepts (filtered >= {args.min_concept_support} pos): {C}",
          f"- Primary setting: q_top={args.q_top}, tau_F1={args.tau_f1}, tau_clean={args.tau_clean}",
          "",
          f"## Coverage (union over layers)",
          f"- SAE covered: **{covered_sae}** (clean {clean_sae})",
          f"- Neuron covered: **{covered_neuron}** (clean {clean_neuron})",
          f"- Ratio SAE/neuron covered: {coverage_summary['ratio_covered_sae_over_neuron']:.2f}",
          "",
          f"## Best layer for SAE coverage: **{best_layer}**",
          ]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    # Log GPU-hours
    elapsed = time.time() - t0
    n_gpu = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    (out_dir / "gpu_hours.txt").write_text(f"{elapsed / 3600 * n_gpu:.4f}\n")
    print(f"[m2] done in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
