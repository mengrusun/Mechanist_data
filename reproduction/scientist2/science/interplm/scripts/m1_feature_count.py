"""M1 — per-layer interpretable feature-count harness for one layer.

Reads Swiss-Prot test.parquet, forwards ESM-2-650M once, captures residual-stream
activations at the target layer, runs them through the pretrained SAE, computes
per-feature density / dead / ultra-low-density statistics, runs a thin LLM
auto-interp gate, and reports SAE-interpretable-count and raw-neuron-interpretable-count.

Memory strategy:
  - Stream over the sequences in chunks of `--chunk-seqs` sequences to keep
    peak GPU / CPU memory bounded.
  - Never hold the full dense Z (N x 10240) in memory. Instead, for every
    chunk, compute:
      - accumulated feature max, sum, count-above-quantile-threshold-estimate
      - top-K activation windows per selected auto-interp feature
      - residue-mask topK samples for the LLM gate
  - After all chunks: threshold-refinement second pass to write per-feature
    top-1% binarized masks to disk as sparse (row, col) indices for M2/M3.
  - The full residual-stream H is also written per-sequence into a chunked file
    for M2 (raw-neuron arm) and M6 (steering site).

Output artifacts (all keyed on `--layer`):
  - runs/m1/activations/layer{L}.pt   {"H": [N,d] fp16, "offsets": [S], "entries": [S], "sequences": [S], "n_seqs": S}
  - runs/m1/sae_codes/layer{L}.npz    sparse top-mask: row (residue idx), col (feature idx), val (activation),
                                       shape [N, F], thr [F], plus per-feature stats.
  - runs/m1/layer{L}.json             headline stats + LLM gate results
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
    ActivationCollector,
    LLMClient,
    LLM_CACHE_DIR,
    SWISSPROT_DIR,
    UNIREF_DIR,
    load_esm,
    load_sae,
    seed_all,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--data", type=str, default=str(SWISSPROT_DIR / "test.parquet"))
    p.add_argument("--unref", type=str, default=str(UNIREF_DIR))
    p.add_argument("--q-top", type=float, default=0.99)
    p.add_argument("--tau-auto-gate", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-seqs", type=int, default=1000, help="Swiss-Prot sequences to process")
    p.add_argument("--max-len", type=int, default=1022)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--chunk-seqs", type=int, default=64)
    p.add_argument("--n-auto-interp-features", type=int, default=100)
    p.add_argument("--n-auto-interp-neurons", type=int, default=100)
    p.add_argument("--llm-endpoint", type=str, default="https://www.dmxapi.cn/v1")
    p.add_argument("--llm-model", type=str, default="gpt-5.4")
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--activations-out", type=str, default=None)
    p.add_argument("--codes-out", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def gate_via_llm(
    top_windows_by_unit: Dict[int, List[str]],
    held_out_by_unit: Dict[int, List[str]],
    low_windows_by_unit: Dict[int, List[str]],
    llm: LLMClient,
    tau: float,
) -> Dict[int, Dict]:
    """Thin two-prompt auto-interp gate. Returns per-unit dict."""
    import random
    results: Dict[int, Dict] = {}
    for uid, tops in top_windows_by_unit.items():
        held = held_out_by_unit.get(uid, [])
        low = low_windows_by_unit.get(uid, [])
        if len(tops) < 5 or (len(held) + len(low) < 6):
            results[uid] = {"label": None, "score": 0.0, "passes": False, "reason": "insufficient_windows"}
            continue
        top_str = "\n".join(f"{i+1}. {w}" for i, w in enumerate(tops[:10]))
        msg = [
            {"role": "system",
             "content": "You are an interpretability assistant labeling protein-language-model features. "
                        "Given short protein sequence windows where the FEATURE fires strongly (marker '[X*]' is "
                        "at the target residue), reply with a JSON object "
                        "{\"label\": str, \"confidence\": float} where label is a concise (≤ 8 words) "
                        "biological description of what the feature detects."},
            {"role": "user",
             "content": f"Top-activating windows for one feature:\n{top_str}\n\nProvide the JSON only."},
        ]
        r1 = llm.chat_json(msg, max_tokens=200)
        if not r1.get("ok"):
            results[uid] = {"label": None, "score": 0.0, "passes": False, "reason": "llm_error_a"}
            continue
        label = str(r1.get("parsed", {}).get("label", "")).strip()
        if not label:
            results[uid] = {"label": None, "score": 0.0, "passes": False, "reason": "empty_label"}
            continue

        candidates = [(w, True) for w in held[:10]] + [(w, False) for w in low[:10]]
        rng = random.Random(uid)
        rng.shuffle(candidates)
        cand_str = "\n".join(f"{i+1}. {w}" for i, (w, _) in enumerate(candidates))
        msg2 = [
            {"role": "system",
             "content": "You decide, for each candidate protein sequence window, whether it should activate the "
                        "given feature described by LABEL. Return JSON {\"activated_indices\": [int]} of 1-based "
                        "indices of candidates that match the LABEL."},
            {"role": "user",
             "content": f"LABEL: {label}\n\nCandidate windows:\n{cand_str}\n\nProvide the JSON only."},
        ]
        r2 = llm.chat_json(msg2, max_tokens=200)
        if not r2.get("ok"):
            results[uid] = {"label": label, "score": 0.0, "passes": False, "reason": "llm_error_b"}
            continue
        try:
            picked_raw = r2.get("parsed", {}).get("activated_indices", [])
            picked = set(int(i) for i in picked_raw if isinstance(i, (int, float, str)))
        except Exception:
            picked = set()
        n_held = sum(1 for _, y in candidates if y)
        n_low = sum(1 for _, y in candidates if not y)
        hits_held = sum(1 for i, (_, y) in enumerate(candidates, 1) if y and i in picked)
        hits_low = sum(1 for i, (_, y) in enumerate(candidates, 1) if not y and i in picked)
        score = (hits_held / max(n_held, 1)) - (hits_low / max(n_low, 1))
        results[uid] = {"label": label, "score": float(score), "passes": bool(score >= tau)}
    return results


def format_window(seq: str, rp: int, half: int = 10) -> str:
    lo = max(0, rp - half)
    hi = min(len(seq), rp + half + 1)
    return seq[lo:rp] + f"[{seq[rp]}*]" + seq[rp + 1 : hi]


def main():
    args = parse_args()
    seed_all(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layer = args.layer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[m1_L{layer}] device={device} n_seqs={args.n_seqs} q_top={args.q_top}", flush=True)

    t0 = time.time()

    df = pd.read_parquet(args.data)
    df = df.iloc[: args.n_seqs].copy()
    df["len"] = df["Sequence"].str.len()
    df = df[df["len"] <= args.max_len].reset_index(drop=True)
    sequences = df["Sequence"].tolist()
    entries = df["Entry"].tolist()
    S = len(sequences)
    print(f"[m1_L{layer}] using {S} seqs, avg len {df['len'].mean():.1f}", flush=True)

    model, tok = load_esm(device)
    sae = load_sae(layer, "normalized", device)
    sae = sae.half() if device == "cuda" else sae
    d_in = sae.d_in
    d_feat = sae.d_feat

    coll = ActivationCollector(model, tok, device, max_len=args.max_len, batch_size=args.batch_size)

    # === Pass 1: extract H and Z chunks, accumulate stats + top-K per feature ===
    H_per_seq: List[torch.Tensor] = [None] * S  # cache to write to disk at end (fp16 CPU)
    # Feature stats
    max_per_feat = torch.zeros(d_feat, dtype=torch.float32)
    sum_per_feat = torch.zeros(d_feat, dtype=torch.float64)
    sum2_per_feat = torch.zeros(d_feat, dtype=torch.float64)
    fire_count = torch.zeros(d_feat, dtype=torch.int64)  # residues where >0

    # Pre-choose auto-interp feature IDs after first pass so we can track their top-K activations
    # Track ALL features' fire count on pass 1, then pick features + do second pass? That doubles compute.
    # Instead: on pass 1 collect a candidate pool (random subset of *possibly-alive* features via a fast heuristic),
    # keep top-K windows for them.
    # A simpler and safer approach: track top-K windows for a random subset of `n_auto_interp_features * 4`
    # feature IDs sampled at the start. After the pass, filter to features with sufficient hits.
    rng = np.random.default_rng(args.seed)
    candidate_features = rng.choice(d_feat, size=min(args.n_auto_interp_features * 4, d_feat), replace=False).tolist()
    candidate_features_set = set(candidate_features)
    # Also for neuron arm
    candidate_neurons = rng.choice(d_in, size=min(args.n_auto_interp_neurons * 4, d_in), replace=False).tolist()
    candidate_neurons_set = set(candidate_neurons)

    # Per-feature top-K residues buffer: uid -> heap of (activation, seq_idx, res_pos, residue_char)
    import heapq
    K_track = args.top_k * 4  # keep 4x to allow splitting into top / held / low
    feat_top: Dict[int, List[Tuple[float, int, int]]] = {u: [] for u in candidate_features}
    neuron_top: Dict[int, List[Tuple[float, int, int]]] = {u: [] for u in candidate_neurons}

    # Neuron stats
    neuron_max = torch.full((d_in,), -1e9, dtype=torch.float32)
    neuron_absmax = torch.zeros(d_in, dtype=torch.float32)
    neuron_sum = torch.zeros(d_in, dtype=torch.float64)
    neuron_sum2 = torch.zeros(d_in, dtype=torch.float64)
    neuron_residue_count = 0

    # Threshold estimation for q_top is done post-loop via a random sample of residues re-encoded through the SAE.

    total_res = 0
    for cs in range(0, S, args.chunk_seqs):
        chunk_seqs = sequences[cs : cs + args.chunk_seqs]
        chunk_entries = entries[cs : cs + args.chunk_seqs]
        t1 = time.time()
        hs_chunk = coll.batch_hidden_states(chunk_seqs, layer=layer)  # list of [Li,d]
        for i, hs in enumerate(hs_chunk):
            H_per_seq[cs + i] = hs  # keep in CPU fp16
        # concat this chunk for SAE encode
        Hc = torch.cat(hs_chunk, dim=0)  # [Nc, d] fp16 on CPU
        Nc = Hc.shape[0]
        total_res += Nc
        # Neuron stats (on CPU float32)
        H32 = Hc.float()
        neuron_max = torch.maximum(neuron_max, H32.max(dim=0).values)
        neuron_absmax = torch.maximum(neuron_absmax, H32.abs().max(dim=0).values)
        neuron_sum += H32.sum(dim=0).double()
        neuron_sum2 += (H32 * H32).sum(dim=0).double()
        neuron_residue_count += Nc
        # SAE encode this chunk
        with torch.no_grad():
            Hc_gpu = Hc.to(device)
            Zc = sae.encode(Hc_gpu).float().cpu()  # [Nc, F]
        del Hc_gpu, H32
        # Stats
        max_per_feat = torch.maximum(max_per_feat, Zc.max(dim=0).values)
        sum_per_feat += Zc.sum(dim=0).double()
        sum2_per_feat += (Zc * Zc).sum(dim=0).double()
        fire_count += (Zc > 0).sum(dim=0)
        # Update per-feature top-K windows for auto-interp (candidate features only)
        Zc_np = Zc.numpy()
        for u in candidate_features:
            col = Zc_np[:, u]
            nz_vals = col[col > 0]
            if len(nz_vals) == 0:
                continue
            # Track top-K windows for auto-interp
            # Get per-sequence tops
            offset = 0
            for si, hs in enumerate(hs_chunk):
                Li = hs.shape[0]
                col_seq = col[offset : offset + Li]
                if col_seq.max() <= 0:
                    offset += Li
                    continue
                nz_positions = np.where(col_seq > 0)[0]
                # take top-3 per sequence
                topk_idx = nz_positions[np.argsort(-col_seq[nz_positions])[:3]]
                for rp in topk_idx:
                    act = float(col_seq[rp])
                    entry = (act, cs + si, int(rp))
                    if len(feat_top[u]) < K_track:
                        heapq.heappush(feat_top[u], entry)
                    else:
                        heapq.heappushpop(feat_top[u], entry)
                offset += Li
        # Neuron top-K windows for auto-interp
        Hc_np = Hc.float().numpy()
        for u in candidate_neurons:
            col = Hc_np[:, u]
            offset = 0
            for si, hs in enumerate(hs_chunk):
                Li = hs.shape[0]
                col_seq = col[offset : offset + Li]
                offset += Li
                if len(col_seq) == 0:
                    continue
                # sort by |activation| desc — neurons signed
                top_positions = np.argsort(-np.abs(col_seq))[:3]
                for rp in top_positions:
                    v = float(abs(col_seq[rp]))
                    entry = (v, cs + si, int(rp))
                    if len(neuron_top[u]) < K_track:
                        heapq.heappush(neuron_top[u], entry)
                    else:
                        heapq.heappushpop(neuron_top[u], entry)
        del Zc, Zc_np, Hc, Hc_np
        elapsed_chunk = time.time() - t1
        print(f"[m1_L{layer}] chunk {cs}-{cs+len(chunk_seqs)} ({Nc} res) in {elapsed_chunk:.1f}s "
              f"(cum residues={total_res})", flush=True)

    # === Per-feature stats ===
    N = total_res
    mean_per_feat = (sum_per_feat / N).float()
    var_per_feat = (sum2_per_feat / N).float() - mean_per_feat.float() ** 2
    std_per_feat = var_per_feat.clamp_min(0).sqrt()
    fire_frac = (fire_count.float() / N)
    dead_frac_feat = float((max_per_feat <= 0).float().mean().item())
    ultra_low_frac = float((fire_frac < 1e-6).float().mean().item())
    normal_mask = (fire_frac >= 1e-6) & (fire_frac <= 0.2)
    n_normal = int(normal_mask.sum().item())

    # Estimate q_top threshold per feature by (re)encoding a bounded sample of residues
    # For q_top=0.99 on N residues: threshold = value such that only (1-q_top)*fire_frac[u]*N residues fire above it.
    # If we compute quantile over ALL firings (including zeros), q_top=0.99 => the value at the 99th percentile of the full Z column.
    # We estimate this via a sample of up to 262k residues (covers most features well).
    SAMPLE_N = min(N, 262144)
    rng_thr = np.random.default_rng(args.seed + 777)
    sample_idx = rng_thr.choice(N, size=SAMPLE_N, replace=False)
    sample_idx.sort()
    # Rebuild H sample from H_per_seq
    H_all_sample = torch.cat(H_per_seq, dim=0)[sample_idx].to(device)
    with torch.no_grad():
        Z_sample = sae.encode(H_all_sample).float().cpu()  # [SAMPLE_N, F]
    del H_all_sample
    # Chunk quantile to avoid OOM on the CPU
    thr_per_feat = torch.empty(d_feat, dtype=torch.float32)
    QCHUNK = 1024
    for i in range(0, d_feat, QCHUNK):
        thr_per_feat[i : i + QCHUNK] = torch.quantile(Z_sample[:, i : i + QCHUNK], args.q_top, dim=0)
    # If a feature is essentially dead (max <= 0), leave threshold at 0 so no firings pass — no false positives
    thr_per_feat[max_per_feat <= 0] = 1e9
    del Z_sample

    # Neuron stats
    neuron_mean = (neuron_sum / neuron_residue_count).float()
    neuron_var = (neuron_sum2 / neuron_residue_count).float() - neuron_mean.float() ** 2
    neuron_std = neuron_var.clamp_min(0).sqrt()
    # Neurons are dense — treat 'dead' as never non-zero (rare in float16 activations)
    dead_frac_neuron = float((neuron_absmax == 0).float().mean().item())
    n_nondead_neurons = int((neuron_absmax > 0).sum().item())

    # === Save H to disk (fp16 concatenated) ===
    if args.activations_out:
        Path(args.activations_out).parent.mkdir(parents=True, exist_ok=True)
        offsets = np.array([h.shape[0] for h in H_per_seq], dtype=np.int64)
        H_full = torch.cat(H_per_seq, dim=0)  # [N, d]
        torch.save({
            "H": H_full,
            "offsets": offsets,
            "entries": entries,
            "sequences": sequences,
            "n_seqs": S,
            "d": d_in,
            "layer": layer,
        }, args.activations_out)
        del H_full
        print(f"[m1_L{layer}] activations cached -> {args.activations_out}", flush=True)

    # === Second pass: write sparse top-mask codes (rows above per-feature q_top) ===
    if args.codes_out:
        Path(args.codes_out).parent.mkdir(parents=True, exist_ok=True)
        thr_gpu = thr_per_feat.to(device)
        row_idx_list, col_idx_list, val_list = [], [], []
        offset_global = 0
        for cs in range(0, S, args.chunk_seqs):
            hs_chunk = H_per_seq[cs : cs + args.chunk_seqs]
            Hc = torch.cat(hs_chunk, dim=0)
            with torch.no_grad():
                Zc = sae.encode(Hc.to(device))  # [Nc, F]
                mask = Zc > thr_gpu.unsqueeze(0)  # [Nc, F]
                nz = torch.nonzero(mask, as_tuple=False)  # [M, 2] (row, col)
                if nz.numel() > 0:
                    rows = nz[:, 0] + offset_global
                    cols = nz[:, 1]
                    vals = Zc[nz[:, 0], nz[:, 1]]
                    row_idx_list.append(rows.cpu().numpy().astype(np.int32))
                    col_idx_list.append(cols.cpu().numpy().astype(np.int32))
                    val_list.append(vals.cpu().numpy().astype(np.float32))
            offset_global += Hc.shape[0]
            del Hc, Zc, mask, nz
        row_all = np.concatenate(row_idx_list) if row_idx_list else np.zeros(0, dtype=np.int32)
        col_all = np.concatenate(col_idx_list) if col_idx_list else np.zeros(0, dtype=np.int32)
        val_all = np.concatenate(val_list) if val_list else np.zeros(0, dtype=np.float32)
        np.savez_compressed(
            args.codes_out,
            row=row_all, col=col_all, val=val_all,
            shape=np.array([N, d_feat], dtype=np.int64),
            thr=thr_per_feat.cpu().numpy().astype(np.float32),
            fire_frac=fire_frac.cpu().numpy().astype(np.float32),
            mean=mean_per_feat.cpu().numpy().astype(np.float32),
            std=std_per_feat.cpu().numpy().astype(np.float32),
            max_per_feat=max_per_feat.cpu().numpy().astype(np.float32),
        )
        print(f"[m1_L{layer}] sparse codes cached -> {args.codes_out} ({row_all.size} nnz)", flush=True)

    # === LLM auto-interp gate ===
    llm_features_out: Dict[int, Dict] = {}
    llm_neurons_out: Dict[int, Dict] = {}
    if not args.skip_llm:
        print(f"[m1_L{layer}] running LLM auto-interp gate...", flush=True)
        llm = LLMClient(base_url=args.llm_endpoint, model=args.llm_model, cache_dir=LLM_CACHE_DIR)
        # Filter candidate features to those with >= 15 total hits AND in normal-density range
        valid_features = []
        for u in candidate_features:
            if len(feat_top[u]) >= 15 and normal_mask[u].item():
                valid_features.append(u)
            if len(valid_features) >= args.n_auto_interp_features:
                break
        # Format windows: split top / held / low
        tops, helds, lows = {}, {}, {}
        for u in valid_features:
            entries_u = sorted(feat_top[u], reverse=True)  # (act desc, seq, pos)
            # top-K
            tK = args.top_k
            top_windows = [format_window(sequences[si], rp) for _, si, rp in entries_u[:tK]]
            held_windows = [format_window(sequences[si], rp) for _, si, rp in entries_u[tK : 2 * tK]]
            # low: pull residues from same seqs with activation ~median
            low_windows = []
            for _, si, rp in entries_u[2 * tK : 3 * tK]:
                low_windows.append(format_window(sequences[si], rp))
            if not low_windows:
                # Fallback: random residue windows from random sequences
                for _ in range(tK):
                    si = int(np.random.randint(0, S))
                    if not sequences[si]:
                        continue
                    rp = int(np.random.randint(0, len(sequences[si])))
                    low_windows.append(format_window(sequences[si], rp))
            tops[u] = top_windows
            helds[u] = held_windows
            lows[u] = low_windows
        llm_features_out = gate_via_llm(tops, helds, lows, llm, args.tau_auto_gate)

        # Neurons
        valid_neurons = [u for u in candidate_neurons if len(neuron_top[u]) >= 15][: args.n_auto_interp_neurons]
        tops_n, helds_n, lows_n = {}, {}, {}
        for u in valid_neurons:
            entries_u = sorted(neuron_top[u], reverse=True)
            tK = args.top_k
            tops_n[u] = [format_window(sequences[si], rp) for _, si, rp in entries_u[:tK]]
            helds_n[u] = [format_window(sequences[si], rp) for _, si, rp in entries_u[tK : 2 * tK]]
            low_windows = []
            for _, si, rp in entries_u[2 * tK : 3 * tK]:
                low_windows.append(format_window(sequences[si], rp))
            if not low_windows:
                for _ in range(tK):
                    si = int(np.random.randint(0, S))
                    if not sequences[si]:
                        continue
                    rp = int(np.random.randint(0, len(sequences[si])))
                    low_windows.append(format_window(sequences[si], rp))
            lows_n[u] = low_windows
        llm_neurons_out = gate_via_llm(tops_n, helds_n, lows_n, llm, args.tau_auto_gate)

        n_f_pass = sum(1 for v in llm_features_out.values() if v["passes"])
        n_f_test = len(llm_features_out)
        n_n_pass = sum(1 for v in llm_neurons_out.values() if v["passes"])
        n_n_test = len(llm_neurons_out)
        pass_rate_features = n_f_pass / max(n_f_test, 1)
        pass_rate_neurons = n_n_pass / max(n_n_test, 1)
        # Extrapolate to pool
        est_sae_interp = pass_rate_features * n_normal
        est_neuron_interp = pass_rate_neurons * n_nondead_neurons
    else:
        pass_rate_features = float("nan")
        pass_rate_neurons = float("nan")
        est_sae_interp = float("nan")
        est_neuron_interp = float("nan")

    elapsed = time.time() - t0

    result = {
        "layer": layer,
        "n_seqs": S,
        "total_residues": int(total_res),
        "d_feat": d_feat,
        "d_in": d_in,
        "sae": {
            "dead_frac": dead_frac_feat,
            "ultra_low_density_frac": ultra_low_frac,
            "normal_density_count": n_normal,
            "llm_gate_pass_rate": float(pass_rate_features),
            "llm_gate_tested": len(llm_features_out),
            "llm_gate_passed": sum(1 for v in llm_features_out.values() if v.get("passes", False)),
            "estimated_interpretable_count": float(est_sae_interp),
        },
        "neurons": {
            "dead_frac": dead_frac_neuron,
            "nondead_count": n_nondead_neurons,
            "llm_gate_pass_rate": float(pass_rate_neurons),
            "llm_gate_tested": len(llm_neurons_out),
            "llm_gate_passed": sum(1 for v in llm_neurons_out.values() if v.get("passes", False)),
            "estimated_interpretable_count": float(est_neuron_interp),
        },
        "ratio_sae_over_neurons": (est_sae_interp / est_neuron_interp) if est_neuron_interp and est_neuron_interp > 0 else float("nan"),
        "wall_clock_seconds": elapsed,
        "q_top": args.q_top,
        "tau_auto_gate": args.tau_auto_gate,
        "seed": args.seed,
        "gate_features_sample": {int(k): v for k, v in list(llm_features_out.items())[:20]},
        "gate_neurons_sample": {int(k): v for k, v in list(llm_neurons_out.items())[:20]},
    }
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[m1_L{layer}] done in {elapsed:.1f}s -> {out_path}", flush=True)
    n_gpu = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    gh_path = out_path.parent / f"gpu_hours_L{layer}.txt"
    gh_path.write_text(f"{elapsed / 3600 * n_gpu:.4f}\n")


if __name__ == "__main__":
    main()
