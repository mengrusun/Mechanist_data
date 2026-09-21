"""Variant: C2 model-swap  ESM-2-650M → ESM-2-8M

Self-contained two-phase script:
  Phase A: collect per-layer hidden states from ESM-2-8M and run through 8M SAEs
            (mirrors what m1_activations.py does for the 650M model)
  Phase B: run the same F1 alignment protocol as m2_concept_alignment.py

Outputs (same layout as runs/m2/):
  <OUT_DIR>/coverage.json          — headline SAE vs neuron coverage + per-layer breakdown
  <OUT_DIR>/sensitivity.json       — 6-point sweep (q_top × tau_F1)
  <OUT_DIR>/per_concept_best_F1.parquet
  <OUT_DIR>/cost.json              — wall-clock seconds, gpu_ids used
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

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent  # verify/c2.../variants/model-swap/.. -> project root
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common import (  # noqa: E402
    SWISSPROT_DAT,
    SWISSPROT_DIR,
    build_residue_labels,
    parse_swissprot_annotations,
    seed_all,
    SparseAutoencoder,
)

# ── constants for 8M variant ──────────────────────────────────────────────────
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/data/zhenqian/models"))
ESM8M_DIR = MODEL_DIR / "ESM-2-8M"
SAE8M_ROOT = MODEL_DIR / "InterPLM-esm2-8m"
SAE8M_LAYERS = [1, 2, 3, 4, 5, 6]
DEFAULT_OUT_DIR = str(SCRIPT_DIR / "results")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--swissprot-dat", type=str,
                   default=str(SWISSPROT_DAT))
    p.add_argument("--activations-source", type=str,
                   default=str(PROJECT_ROOT / "runs" / "m1" / "activations"),
                   help="Path to main-experiment m1 activations (for sequence list only)")
    p.add_argument("--q-top", type=float, default=0.99)
    p.add_argument("--tau-f1", type=float, default=0.5)
    p.add_argument("--tau-clean", type=float, default=0.7)
    p.add_argument("--sweep-tau", type=str, default="0.3,0.5,0.7")
    p.add_argument("--sweep-q", type=str, default="0.95,0.99")
    p.add_argument("--min-concept-support", type=int, default=25)
    p.add_argument("--max-concepts", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── SAE loader for 8M ─────────────────────────────────────────────────────────

def load_sae_8m(layer: int, device: str = "cpu") -> SparseAutoencoder:
    layer_dir = SAE8M_ROOT / f"layer_{layer}"
    cfg = json.loads((layer_dir / "config.json").read_text())
    d_in = int(cfg["architecture"]["esm_dim"])
    d_feat = int(cfg["architecture"]["feature_dim"])
    state = torch.load(layer_dir / "ae_normalized.pt", map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(d_in, d_feat)
    sd = {}
    for k, v in state.items():
        if k == "bias":
            sd["pre_bias"] = v
        else:
            sd[k] = v
    sae.load_state_dict(sd, strict=True)
    sae.eval()
    sae.to(device)
    return sae


# ── helpers copied from m2 (same logic, self-contained) ──────────────────────

def build_concept_mask(
    per_entry_labels: Dict,
    concept_universe: List[str],
    ordered_entries: List[str],
    ordered_sequences: List[str],
    offsets: np.ndarray,
    concept_to_idx: Dict,
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    N = int(offsets.sum())
    C = len(concept_universe)
    mask = np.zeros((N, C), dtype=bool)
    eligible = np.zeros(N, dtype=bool)
    cursor = 0
    for si, entry in enumerate(ordered_entries):
        Li = int(offsets[si])
        labels = per_entry_labels.get(entry)
        if labels:
            eligible[cursor: cursor + Li] = True
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


def f1_matrix_sparse(sp_unit: torch.Tensor, concept_mask: torch.Tensor,
                     eligible_mask: torch.Tensor = None) -> torch.Tensor:
    Cf = concept_mask.float()
    if eligible_mask is not None:
        elig = eligible_mask.float()
        Cf = Cf * elig.unsqueeze(1)
        idx = sp_unit._indices()
        vals = sp_unit._values()
        keep = eligible_mask[idx[0]]
        idx_k = idx[:, keep]
        vals_k = vals[keep]
        sp_unit = torch.sparse_coo_tensor(idx_k, vals_k, size=sp_unit.shape).coalesce()
    sp_t = sp_unit.t().coalesce()
    tp = torch.sparse.mm(sp_t, Cf)
    unit_sum = torch.sparse.sum(sp_unit, dim=0).to_dense()
    concept_sum = Cf.sum(dim=0)
    fp = unit_sum.unsqueeze(1) - tp
    fn = concept_sum.unsqueeze(0) - tp
    prec = tp / (tp + fp).clamp_min(1e-12)
    rec = tp / (tp + fn).clamp_min(1e-12)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
    return f1


def f1_matrix_dense(unit_mask: torch.Tensor, concept_mask: torch.Tensor,
                    eligible_mask: torch.Tensor = None) -> torch.Tensor:
    U = unit_mask.float()
    Cf = concept_mask.float()
    if eligible_mask is not None:
        elig = eligible_mask.float().unsqueeze(1)
        U = U * elig
        Cf = Cf * elig
    tp = U.T @ Cf
    unit_sum = U.sum(dim=0)
    concept_sum = Cf.sum(dim=0)
    fp = unit_sum.unsqueeze(1) - tp
    fn = concept_sum.unsqueeze(0) - tp
    prec = tp / (tp + fp).clamp_min(1e-12)
    rec = tp / (tp + fn).clamp_min(1e-12)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
    return f1


def compute_neuron_topmask(H: torch.Tensor, q_top: float, chunk: int = 512) -> torch.Tensor:
    N, d = H.shape
    thr = torch.empty(d, dtype=torch.float32)
    Hf = H.float()
    for i in range(0, d, chunk):
        sl = Hf[:, i: i + chunk]
        thr[i: i + chunk] = torch.quantile(sl, q_top, dim=0)
    return Hf > thr.unsqueeze(0)


def coverage_from_bestf1(best_f1: np.ndarray, tau: float) -> int:
    return int((best_f1 >= tau).sum())


def get_gpu_ids() -> List[int]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd == "" or cvd == "-1":
        return []
    try:
        return [int(x) for x in cvd.split(",") if x.strip()]
    except ValueError:
        return []


# ── Phase A: collect ESM-2-8M hidden states ──────────────────────────────────

def collect_8m_hidden_states(
    ordered_entries: List[str],
    ordered_sequences: List[str],
    layer: int,
    device: str,
    batch_size: int,
    max_len: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Return (H [N_residues, 320] fp16, offsets [S] int64)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ESM8M_DIR))
    model8m = AutoModel.from_pretrained(
        str(ESM8M_DIR),
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    model8m.eval()
    model8m.to(device)

    all_H: List[torch.Tensor] = []
    offsets_list: List[int] = []

    with torch.no_grad():
        for i in range(0, len(ordered_sequences), batch_size):
            batch_seqs = ordered_sequences[i: i + batch_size]
            enc = tok(batch_seqs, padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model8m(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )
            hs = out.hidden_states[layer]  # [B, T, 320]
            am = enc["attention_mask"]
            for j, seq in enumerate(batch_seqs):
                nonpad_len = int(am[j].bool().sum().item())
                # strip CLS (pos 0) and EOS (pos nonpad_len-1)
                start, end = 1, nonpad_len - 1
                if end <= start:
                    per_res = torch.zeros((0, hs.shape[-1]), dtype=torch.float16)
                else:
                    per_res = hs[j, start:end].detach().to(torch.float16).cpu()
                # Align to actual sequence length (may be truncated)
                actual_len = min(len(seq), max_len - 2)
                per_res = per_res[:actual_len]
                all_H.append(per_res)
                offsets_list.append(per_res.shape[0])
                # Soft assertion: warn if residue count mismatches sequence length
                expected_len = min(len(seq), max_len - 2)
                if per_res.shape[0] != expected_len:
                    print(f"  [8M collect] WARNING: seq {i+j} residue count mismatch: "
                          f"got {per_res.shape[0]} expected {expected_len}", flush=True)
            if (i // batch_size) % 10 == 0:
                print(f"  [8M collect] layer={layer} seqs {i}–{min(i+batch_size, len(ordered_sequences))-1} done", flush=True)

    del model8m
    torch.cuda.empty_cache()

    H_full = torch.cat(all_H, dim=0)  # [N_residues, 320] fp16 CPU
    offsets_arr = np.array(offsets_list, dtype=np.int64)
    return H_full, offsets_arr


# ── Phase B: F1 alignment (same as m2, adapted for 8M) ───────────────────────

def main():
    args = parse_args()
    seed_all(args.seed)
    t_start = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_ids = get_gpu_ids()
    print(f"[variant] device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}", flush=True)

    # 1) Load sequence list from main-experiment m1 activations (identical test split)
    act_dir = Path(args.activations_source)
    ref_layer_file = act_dir / "layer1.pt"
    print(f"[variant] loading sequence list from {ref_layer_file} ...", flush=True)
    ref = torch.load(ref_layer_file, map_location="cpu", weights_only=False)
    ordered_entries = list(ref["entries"])
    ordered_sequences = list(ref["sequences"])
    S = len(ordered_entries)
    print(f"[variant] S={S} sequences from main-experiment test split", flush=True)
    del ref

    # 2) Parse Swiss-Prot annotations (same dat file)
    dat_path = Path(args.swissprot_dat)
    entries_set = set(ordered_entries)
    print(f"[variant] parsing Swiss-Prot annotations for {S} entries ...", flush=True)
    seq_map = {e: s for e, s in zip(ordered_entries, ordered_sequences)}
    annotations = parse_swissprot_annotations(dat_path, entries_to_keep=entries_set)
    print(f"[variant] {len(annotations)}/{S} entries have annotations", flush=True)
    per_entry_labels, concept_universe_full = build_residue_labels(seq_map, annotations)
    print(f"[variant] concept universe (raw)={len(concept_universe_full)}", flush=True)

    # 3) We need offsets first — collect one layer to establish offsets, then use for mask build
    #    Actually we collect per layer in Phase A; we need a consistent offset array.
    #    Use the SAME offsets as m1 (sequence lengths are model-independent up to truncation).
    #    To be safe, derive from sequence lengths (same sequences, same truncation).
    # Compute offsets from sequence lengths, capping at max_len-2 (strip CLS+EOS)
    offsets = np.array([min(len(s), args.max_len - 2) for s in ordered_sequences], dtype=np.int64)
    N_total = int(offsets.sum())
    print(f"[variant] N_total={N_total} residues (offsets from sequence lengths, cap={args.max_len-2})", flush=True)

    # 4) Build concept mask
    concept_to_idx = {c: i for i, c in enumerate(concept_universe_full)}
    concept_mask_full, per_concept_pos, eligible_mask = build_concept_mask(
        per_entry_labels, concept_universe_full, ordered_entries, ordered_sequences, offsets, concept_to_idx
    )
    print(f"[variant] eligible_residues={int(eligible_mask.sum().item())}/{N_total}", flush=True)
    keep = per_concept_pos >= args.min_concept_support
    if args.max_concepts is not None:
        order = np.argsort(-per_concept_pos)
        keep_idx_tmp = order[:args.max_concepts]
        keep_bool = np.zeros(len(concept_universe_full), dtype=bool)
        keep_bool[keep_idx_tmp] = True
        keep = keep_bool & keep
    concept_universe = [c for c, k in zip(concept_universe_full, keep) if k]
    keep_idx_np = np.where(keep)[0]
    concept_mask = concept_mask_full[:, keep_idx_np]
    C = len(concept_universe)
    print(f"[variant] concept universe (filtered support>={args.min_concept_support})={C}", flush=True)

    sweep_tau = [float(x) for x in args.sweep_tau.split(",")]
    sweep_q = [float(x) for x in args.sweep_q.split(",")]

    # Accumulators (parallel to m2)
    best_f1_sae = np.full(C, 0.0, dtype=np.float32)
    best_layer_sae = np.full(C, -1, dtype=np.int32)
    best_feature_sae = np.full(C, -1, dtype=np.int32)
    best_f1_neuron = np.full(C, 0.0, dtype=np.float32)
    best_layer_neuron = np.full(C, -1, dtype=np.int32)
    best_dim_neuron = np.full(C, -1, dtype=np.int32)
    sweep_best_sae = {q: np.full(C, 0.0, dtype=np.float32) for q in sweep_q}
    sweep_best_neuron = {q: np.full(C, 0.0, dtype=np.float32) for q in sweep_q}
    per_layer_summary = {}

    # 5) Per-layer: collect 8M hidden states, run through SAE, compute F1
    for layer in SAE8M_LAYERS:
        tL = time.time()
        print(f"[variant] === layer {layer} ===", flush=True)

        # Phase A: collect hidden states
        print(f"[variant] collecting ESM-2-8M hidden states at layer {layer} ...", flush=True)
        H, offsets_layer = collect_8m_hidden_states(
            ordered_entries, ordered_sequences, layer, device, args.batch_size, args.max_len
        )
        # offsets_layer should match offsets; verify
        N_layer = H.shape[0]
        if N_layer != N_total:
            print(f"[variant] WARNING: N_layer={N_layer} != N_total={N_total}; using layer offsets", flush=True)
            # re-build concept_mask and eligible for this layer's offsets if needed
            # (can happen if truncation differs; use offsets_layer)
            concept_mask_L, per_concept_pos_L, eligible_mask_L = build_concept_mask(
                per_entry_labels, concept_universe_full, ordered_entries, ordered_sequences,
                offsets_layer, concept_to_idx
            )
            concept_mask_L = concept_mask_L[:, keep_idx_np]
            eligible_mask_L = eligible_mask_L
        else:
            concept_mask_L = concept_mask
            eligible_mask_L = eligible_mask

        N_L = H.shape[0]
        d_model = H.shape[1]  # 320

        # Load SAE for this layer
        print(f"[variant] loading SAE-8M layer {layer} ...", flush=True)
        sae = load_sae_8m(layer, device)
        F_dim = sae.d_feat  # 10240

        # --- SAE arm: encode full H through SAE, threshold at q=0.99 ---
        print(f"[variant] encoding through SAE (F_dim={F_dim}) ...", flush=True)
        # Compute per-feature quantile thresholds from H (using a sample for efficiency)
        SAMPLE_N = min(N_L, 65536)
        rng = np.random.default_rng(args.seed + layer * 100)
        sample_idx = rng.choice(N_L, size=SAMPLE_N, replace=False)
        sample_idx.sort()
        H_sample = H[sample_idx].to(device)
        with torch.no_grad():
            Z_sample = sae.encode(H_sample).float().cpu()
        del H_sample

        # Per-feature q_top quantile (primary = 0.99)
        q_primary = args.q_top
        thr_primary = torch.empty(F_dim, dtype=torch.float32)
        QCHUNK = 512
        for _i in range(0, F_dim, QCHUNK):
            sl = Z_sample[:, _i: _i + QCHUNK]
            thr_primary[_i: _i + QCHUNK] = torch.quantile(sl, q_primary, dim=0)

        # Also compute thresholds for other sweep q values
        thr_sweep = {}
        for q in sweep_q:
            if q == q_primary:
                thr_sweep[q] = thr_primary
            else:
                thr_q = torch.empty(F_dim, dtype=torch.float32)
                for _i in range(0, F_dim, QCHUNK):
                    sl = Z_sample[:, _i: _i + QCHUNK]
                    thr_q[_i: _i + QCHUNK] = torch.quantile(sl, q, dim=0)
                thr_sweep[q] = thr_q
        del Z_sample

        # Build sparse SAE mask for primary q (full H)
        thr_gpu = thr_primary.to(device)
        row_list, col_list = [], []
        chunk = 4096
        with torch.no_grad():
            for ci in range(0, N_L, chunk):
                Hc = H[ci: ci + chunk].to(device)
                Zc = sae.encode(Hc)
                m = Zc > thr_gpu.unsqueeze(0)
                nz = torch.nonzero(m, as_tuple=False)
                if nz.numel() > 0:
                    row_list.append((nz[:, 0] + ci).cpu().numpy().astype(np.int32))
                    col_list.append(nz[:, 1].cpu().numpy().astype(np.int32))
                del Hc, Zc, m, nz
        del thr_gpu

        if row_list:
            row_arr = np.concatenate(row_list)
            col_arr = np.concatenate(col_list)
            idx = torch.stack([
                torch.from_numpy(row_arr.astype(np.int64)),
                torch.from_numpy(col_arr.astype(np.int64))
            ], dim=0)
            vals = torch.ones(row_arr.size, dtype=torch.float32)
            sp = torch.sparse_coo_tensor(idx, vals, size=(N_L, F_dim)).coalesce()
        else:
            # no activations — build empty sparse
            idx = torch.zeros((2, 0), dtype=torch.int64)
            vals = torch.zeros(0, dtype=torch.float32)
            sp = torch.sparse_coo_tensor(idx, vals, size=(N_L, F_dim)).coalesce()

        # F1 matrix (primary)
        f1_sae = f1_matrix_sparse(sp, concept_mask_L, eligible_mask=eligible_mask_L)
        best_f1_layer_sae, best_feat_layer_sae = f1_sae.max(dim=0)
        best_f1_layer_sae = best_f1_layer_sae.numpy()
        best_feat_layer_sae = best_feat_layer_sae.numpy()
        upd = best_f1_layer_sae > best_f1_sae
        best_f1_sae = np.where(upd, best_f1_layer_sae, best_f1_sae)
        best_layer_sae = np.where(upd, layer, best_layer_sae).astype(np.int32)
        best_feature_sae = np.where(upd, best_feat_layer_sae, best_feature_sae).astype(np.int32)
        if q_primary in sweep_best_sae:
            sweep_best_sae[q_primary] = np.maximum(sweep_best_sae[q_primary], best_f1_layer_sae)
        cov_sae_primary = coverage_from_bestf1(best_f1_layer_sae, args.tau_f1)
        clean_sae_primary = coverage_from_bestf1(best_f1_layer_sae, args.tau_clean)

        # Sweep q values for SAE
        for q in sweep_q:
            if q == q_primary:
                continue
            thr_q_gpu = thr_sweep[q].to(device)
            row_q, col_q = [], []
            with torch.no_grad():
                for ci in range(0, N_L, chunk):
                    Hc = H[ci: ci + chunk].to(device)
                    Zc = sae.encode(Hc)
                    m = Zc > thr_q_gpu.unsqueeze(0)
                    nz = torch.nonzero(m, as_tuple=False)
                    if nz.numel() > 0:
                        row_q.append((nz[:, 0] + ci).cpu().numpy().astype(np.int32))
                        col_q.append(nz[:, 1].cpu().numpy().astype(np.int32))
                    del Hc, Zc, m, nz
            del thr_q_gpu
            if row_q:
                row_qa = np.concatenate(row_q); col_qa = np.concatenate(col_q)
                idx_q = torch.stack([torch.from_numpy(row_qa.astype(np.int64)),
                                     torch.from_numpy(col_qa.astype(np.int64))], dim=0)
                vals_q = torch.ones(row_qa.size, dtype=torch.float32)
                sp_q = torch.sparse_coo_tensor(idx_q, vals_q, size=(N_L, F_dim)).coalesce()
                f1_q_sae = f1_matrix_sparse(sp_q, concept_mask_L, eligible_mask=eligible_mask_L)
                best_f1_q_sae = f1_q_sae.max(dim=0).values.numpy()
                sweep_best_sae[q] = np.maximum(sweep_best_sae[q], best_f1_q_sae)
                del sp_q, f1_q_sae
        del sae

        # --- Neuron arm at all q values ---
        print(f"[variant] computing neuron F1 (d_model={d_model}) ...", flush=True)
        for q in sweep_q:
            neuron_mask_q = compute_neuron_topmask(H, q)
            f1_q_n = f1_matrix_dense(neuron_mask_q, concept_mask_L, eligible_mask=eligible_mask_L)
            best_f1_q_n = f1_q_n.max(dim=0).values.numpy()
            if q == q_primary:
                # primary neuron stats
                best_f1_layer_n, best_dim_layer_n = f1_q_n.max(dim=0)
                best_f1_layer_n = best_f1_layer_n.numpy()
                best_dim_layer_n = best_dim_layer_n.numpy()
                upd_n = best_f1_layer_n > best_f1_neuron
                best_f1_neuron = np.where(upd_n, best_f1_layer_n, best_f1_neuron)
                best_layer_neuron = np.where(upd_n, layer, best_layer_neuron).astype(np.int32)
                best_dim_neuron = np.where(upd_n, best_dim_layer_n, best_dim_neuron).astype(np.int32)
                cov_neuron_primary = coverage_from_bestf1(best_f1_layer_n, args.tau_f1)
                clean_neuron_primary = coverage_from_bestf1(best_f1_layer_n, args.tau_clean)
            sweep_best_neuron[q] = np.maximum(sweep_best_neuron[q], best_f1_q_n)
            del neuron_mask_q, f1_q_n

        per_layer_summary[str(layer)] = {
            "sae": {
                "coverage@0.5": int(cov_sae_primary),
                "clean@0.7": int(clean_sae_primary),
                "F_dim": F_dim,
            },
            "neuron": {
                "coverage@0.5": int(cov_neuron_primary),
                "clean@0.7": int(clean_neuron_primary),
                "d": d_model,
            },
            "elapsed_seconds": time.time() - tL,
        }
        print(f"[variant] layer {layer}: sae_cov={cov_sae_primary} sae_clean={clean_sae_primary} "
              f"neuron_cov={cov_neuron_primary} neuron_clean={clean_neuron_primary} "
              f"[{time.time()-tL:.1f}s]", flush=True)
        del H, sp, f1_sae
        torch.cuda.empty_cache()

    # 6) Aggregate coverage
    covered_sae = coverage_from_bestf1(best_f1_sae, args.tau_f1)
    clean_sae = coverage_from_bestf1(best_f1_sae, args.tau_clean)
    covered_neuron = coverage_from_bestf1(best_f1_neuron, args.tau_f1)
    clean_neuron = coverage_from_bestf1(best_f1_neuron, args.tau_clean)
    ratio = (covered_sae / covered_neuron) if covered_neuron > 0 else float("nan") if covered_sae == 0 else float("inf")

    print(f"[variant] RESULT: SAE covered={covered_sae} clean={clean_sae}  neuron covered={covered_neuron} clean={clean_neuron}  ratio={ratio}", flush=True)

    coverage_summary = {
        "model": "ESM-2-8M",
        "sae_root": str(SAE8M_ROOT),
        "sae_layers": SAE8M_LAYERS,
        "sae": {
            "covered_union": int(covered_sae),
            "clean_union": int(clean_sae),
        },
        "neuron": {
            "covered_union": int(covered_neuron),
            "clean_union": int(clean_neuron),
        },
        "ratio_covered_sae_over_neuron": ratio,
        "primary_setting": {"q_top": args.q_top, "tau_f1": args.tau_f1, "tau_clean": args.tau_clean},
        "n_concepts": C,
        "n_residues": int(N_total),
        "n_seqs": S,
        "per_layer": per_layer_summary,
        "wall_clock_seconds": time.time() - t_start,
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage_summary, indent=2))
    print(f"[variant] coverage.json written", flush=True)

    # 7) Sensitivity sweep
    sensitivity = {}
    for q in sweep_q:
        for tau in sweep_tau:
            cov_s = coverage_from_bestf1(sweep_best_sae[q], tau)
            cov_n = coverage_from_bestf1(sweep_best_neuron[q], tau)
            sensitivity[f"q={q}_tau={tau}"] = {
                "sae_covered": int(cov_s),
                "neuron_covered": int(cov_n),
                "ratio": (cov_s / cov_n) if cov_n > 0 else (float("nan") if cov_s == 0 else float("inf")),
            }
    (out_dir / "sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
    print(f"[variant] sensitivity.json written", flush=True)

    # 8) Per-concept parquet
    per_concept_df = pd.DataFrame({
        "concept": concept_universe,
        "positive_count": per_concept_pos[keep_idx_np],
        "sae_best_f1": best_f1_sae,
        "sae_best_layer": best_layer_sae,
        "sae_best_feature": best_feature_sae,
        "neuron_best_f1": best_f1_neuron,
        "neuron_best_layer": best_layer_neuron,
        "neuron_best_dim": best_dim_neuron,
    })
    per_concept_df.to_parquet(out_dir / "per_concept_best_F1.parquet", index=False)
    print(f"[variant] per_concept_best_F1.parquet written", flush=True)

    # 9) Cost record
    elapsed = time.time() - t_start
    cost = {
        "wall_clock_seconds": elapsed,
        "gpu_ids": gpu_ids,
        "model": "ESM-2-8M",
        "n_layers": len(SAE8M_LAYERS),
        "n_seqs": S,
        "n_residues": int(N_total),
    }
    (out_dir / "cost.json").write_text(json.dumps(cost, indent=2))
    print(f"[variant] Done in {elapsed:.1f}s. cost.json written.", flush=True)


if __name__ == "__main__":
    main()
