"""M6 — SAE-feature-clamp steering of ESM-2 masked-token generation.

For each labeled SAE feature f with a checkable target property:
  Arm 0 — no-steering baseline
  Arm 1 — SAE feature-clamp at layer L*: add alpha * sigma_f * d_f to residual at every residue
  Arm 2 — mean-activation-addition baseline: alpha * (mean(h | property=1) - mean(h | property=0))
  Arm 3 — random-clamp: clamp a random feature (not matched to target)

Uses ESM-2 MLM head for iterative-refinement fill on masked test sequences.
Yield = fraction of completions carrying the target property (checked externally).
Plausibility = pseudo-perplexity under ESM-2 relative to no-steering.

Outputs:
  runs/m6/yields.parquet
  runs/m6/plausibility.parquet
  runs/m6/summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    SWISSPROT_DIR,
    load_sae,
    seed_all,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=str, default="runs/m4/target_property_features.json")
    p.add_argument("--best-layer-file", type=str, default="runs/m2/best_layer.json")
    p.add_argument("--activations-dir", type=str, default="runs/m1/activations")
    p.add_argument("--codes-dir", type=str, default="runs/m1/sae_codes")
    p.add_argument("--swissprot-dir", type=str, default=str(SWISSPROT_DIR))
    p.add_argument("--out-dir", type=str, default="runs/m6")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--doses", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--n-seq-per-batch", type=int, default=15)
    p.add_argument("--n-refill-steps", type=int, default=5)
    p.add_argument("--mask-fraction", type=float, default=0.2)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--plausibility-band-factor", type=float, default=1.5)
    p.add_argument("--seed-sequences", type=int, default=10, help="sequences per feature to seed generations")
    return p.parse_args()


# --- External property checkers ------------------------------------------------


def check_signal_peptide(seq: str) -> bool:
    """Simple N-terminal hydrophobicity + charge rule (SignalP-lite)."""
    if len(seq) < 20:
        return False
    n_term = seq[:30]
    # SignalP heuristic: mostly hydrophobic residues in positions 5–20
    hydrophobic = set("AILMVFWY")
    core = n_term[5:22]
    hphob_frac = sum(1 for a in core if a in hydrophobic) / max(len(core), 1)
    # Positive residues at very N terminus
    n_charge = sum(1 for a in n_term[:5] if a in "KR")
    return hphob_frac >= 0.55 and n_charge >= 1


def check_transmembrane(seq: str, window: int = 19, threshold: float = 1.6) -> bool:
    """Kyte-Doolittle sliding-window hydrophobicity."""
    KD = {"A":1.8,"C":2.5,"D":-3.5,"E":-3.5,"F":2.8,"G":-0.4,"H":-3.2,"I":4.5,
          "K":-3.9,"L":3.8,"M":1.9,"N":-3.5,"P":-1.6,"Q":-3.5,"R":-4.5,"S":-0.8,
          "T":-0.7,"V":4.2,"W":-0.9,"Y":-1.3}
    if len(seq) < window:
        return False
    vals = np.array([KD.get(a, 0.0) for a in seq])
    if len(vals) < window:
        return False
    kernel = np.ones(window) / window
    smooth = np.convolve(vals, kernel, mode="valid")
    return bool(np.any(smooth >= threshold))


def check_zinc_binding(seq: str) -> bool:
    """ProSite-style Zn-finger regexes; check any variant."""
    patterns = [
        r"C.{2,4}C.{9,20}C.{2,4}C",   # Cys4 finger
        r"C.{2,4}C.{9,20}H.{2,4}H",   # Cys2His2 finger
        r"C.{1,4}H.{9,20}C.{1,4}C",
    ]
    for p in patterns:
        if re.search(p, seq):
            return True
    return False


def check_n_glycosylation(seq: str) -> bool:
    """N[^P][ST][^P] motif."""
    return bool(re.search(r"N[^P][ST][^P]", seq))


CHECKERS = {
    "signal_peptide": check_signal_peptide,
    "transmembrane_domain": check_transmembrane,
    "zinc_binding_motif": check_zinc_binding,
    "n_glycosylation_site": check_n_glycosylation,
}


# --- ESM-2 for MLM with residual-stream steering hook -------------------------


class SteeringHook:
    """Add alpha * direction at layer `target_layer` residual stream (all positions)."""

    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction  # [d]
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        hs = hs + self.alpha * self.direction.to(hs.dtype).to(hs.device)
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs


def load_esm_mlm(device: str, mask_token: bool = True):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from common import ESM_DIR
    tok = AutoTokenizer.from_pretrained(str(ESM_DIR))
    model = AutoModelForMaskedLM.from_pretrained(
        str(ESM_DIR),
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device).eval()
    return model, tok


@torch.no_grad()
def masked_fill_generation(
    model, tok, seed_seq: str, layer_hook_module, direction: torch.Tensor, alpha: float,
    n_refill_steps: int, mask_fraction: float, device: str, seed: int,
    max_len: int = 128,
) -> Tuple[str, float]:
    """Iterative masked-refill starting from seed_seq. Returns (completion, pseudo_perplexity)."""
    rng = np.random.default_rng(seed)
    seq = seed_seq[:max_len]
    positions = np.arange(len(seq))
    n_mask = max(1, int(mask_fraction * len(seq)))
    mask_positions = rng.choice(positions, size=n_mask, replace=False)
    seq_list = list(seq)
    for _ in range(n_refill_steps):
        # Build masked input
        masked_seq = "".join(
            tok.mask_token if i in mask_positions else seq_list[i] for i in range(len(seq_list))
        )
        enc = tok(masked_seq, return_tensors="pt").to(device)
        # Register hook
        h = None
        if direction is not None and alpha != 0.0:
            hook = SteeringHook(direction, alpha)
            h = layer_hook_module.register_forward_hook(hook)
        try:
            out = model(**enc)
            logits = out.logits[0]  # [T, V]
        finally:
            if h is not None:
                h.remove()
        # Fill masks with argmax token
        mask_token_id = tok.mask_token_id
        input_ids = enc["input_ids"][0]
        for t in range(len(input_ids)):
            if input_ids[t].item() == mask_token_id:
                # Position t is 1 + residue_index (accounting for [CLS])
                res_idx = t - 1
                if 0 <= res_idx < len(seq_list):
                    # Pick top-1 valid amino-acid token
                    top_ids = torch.topk(logits[t], k=8).indices.cpu().tolist()
                    for tid in top_ids:
                        tok_str = tok.decode([tid]).strip()
                        if len(tok_str) == 1 and tok_str.upper() in "ACDEFGHIKLMNPQRSTVWY":
                            seq_list[res_idx] = tok_str.upper()
                            break
        # Reduce mask fraction over refills
        n_mask = max(1, int(n_mask * 0.6))
        if n_mask > 0:
            mask_positions = rng.choice(positions, size=n_mask, replace=False)
        else:
            break

    completion = "".join(seq_list)
    # Pseudo-perplexity: mask each residue in turn, score with LM head, average -log P
    # Full PPL is expensive; approximate on a 10% residue sample
    # Register the same hook for PPL scoring (steered PPL is what we want)
    sample_pos = rng.choice(len(completion), size=max(1, len(completion) // 10), replace=False)
    log_probs = []
    for p in sample_pos:
        masked = list(completion)
        target_aa = masked[p]
        masked[p] = "$"  # placeholder
        masked_seq = "".join(masked).replace("$", tok.mask_token)
        enc = tok(masked_seq, return_tensors="pt").to(device)
        # Register hook (steering during PPL scoring — we want steered PPL)
        h = None
        if direction is not None and alpha != 0.0:
            hook = SteeringHook(direction, alpha)
            h = layer_hook_module.register_forward_hook(hook)
        try:
            out = model(**enc)
            logits = out.logits[0]
            probs = torch.softmax(logits, dim=-1)
            mask_token_id = tok.mask_token_id
            input_ids = enc["input_ids"][0]
            mask_positions_ids = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions_ids) > 0:
                mp = mask_positions_ids[0].item()
                target_id = tok.convert_tokens_to_ids(target_aa) if target_aa.upper() in "ACDEFGHIKLMNPQRSTVWY" else None
                if target_id is not None:
                    log_probs.append(-float(torch.log(probs[mp, target_id] + 1e-30).item()))
        finally:
            if h is not None:
                h.remove()
    ppl = float(np.exp(np.mean(log_probs))) if log_probs else float("inf")
    return completion, ppl


def get_layer_hook_module(model, layer: int):
    """Return the module whose output is the residual stream at `layer`.
    For EsmForMaskedLM: model.esm.encoder.layer[layer-1].output
    (layer=1 -> hidden_states[1] -> output of layer[0]).
    """
    esm = model.esm if hasattr(model, "esm") else model
    return esm.encoder.layer[layer - 1]


def compute_mean_activation_direction(
    H: torch.Tensor, sequences: List[str], property_check_fn, device: str
) -> torch.Tensor:
    """Compute Δ_c = mean(h | property=1) - mean(h | property=0) at residue level.
    Approximation: use per-sequence property call (a residue in a property-positive sequence
    is considered a positive; residues in property-negative sequences are negatives).
    """
    d = H.shape[1]
    pos_sum = torch.zeros(d, dtype=torch.float64)
    neg_sum = torch.zeros(d, dtype=torch.float64)
    pos_n = 0; neg_n = 0
    cursor = 0
    for seq in sequences:
        L = len(seq)
        Hs = H[cursor : cursor + L].float().double()
        cursor += L
        if property_check_fn(seq):
            pos_sum += Hs.sum(dim=0)
            pos_n += L
        else:
            neg_sum += Hs.sum(dim=0)
            neg_n += L
    if pos_n == 0 or neg_n == 0:
        return torch.zeros(d, dtype=torch.float32)
    delta = (pos_sum / pos_n - neg_sum / neg_n).float()
    return delta


def main():
    args = parse_args()
    seed_all(args.seeds[0])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load features + best layer
    features_data = json.loads(Path(args.features).read_text())
    best_layer = features_data["best_layer"]
    features = features_data["features"][:4]  # cap 4
    if not features:
        print(f"[m6] no target-property features found; writing empty results", flush=True)
        (out_dir / "yields.parquet").write_bytes(pd.DataFrame(columns=["feature_id","target_property","arm","alpha","seed","yield_frac","plausibility_pass_rate"]).to_parquet())
        (out_dir / "summary.md").write_text("# M6 — no target-property features to steer.\n")
        (out_dir / "gpu_hours.txt").write_text("0.0\n")
        return

    # Load ESM MLM + SAE
    print(f"[m6] loading ESM-2 MLM and SAE for layer {best_layer}...", flush=True)
    model, tok = load_esm_mlm(device)
    sae = load_sae(best_layer, "normalized", device).half()
    layer_module = get_layer_hook_module(model, best_layer)

    # Feature std sigma_f — from cached codes
    codes = np.load(Path(args.codes_dir) / f"layer{best_layer}.npz")
    fire_std = codes.get("std")
    if fire_std is None:
        # fallback: read from feature stats via reservoir sample
        fire_std = np.ones(int(codes["shape"][1]), dtype=np.float32)

    # Decoder columns d_f (feature directions) [d, F]
    d_dec = sae.decoder.weight.data  # [d, F]

    # Cached H for mean-activation-addition (from M1, best_layer)
    act = torch.load(Path(args.activations_dir) / f"layer{best_layer}.pt", map_location="cpu", weights_only=False)
    H = act["H"]  # [N, d]
    cached_seqs = list(act["sequences"])
    offsets = np.array(act["offsets"], dtype=np.int64)

    # Seed sequences for generation: Swiss-Prot test split (short), non-overlapping per feature
    test_df = pd.read_parquet(Path(args.swissprot_dir) / "test.parquet")
    test_df["len"] = test_df["Sequence"].str.len()
    seed_pool = test_df[(test_df["len"] >= 40) & (test_df["len"] <= args.max_len)].reset_index(drop=True)
    rng = np.random.default_rng(42)

    all_rows = []
    plausibility_rows = []
    for fdata in features:
        feature_id = int(fdata["feature_id"])
        prop = fdata["target_property"]
        checker = CHECKERS.get(prop)
        if checker is None:
            continue
        sigma_f = float(fire_std[feature_id]) if feature_id < len(fire_std) else 1.0
        sigma_f = max(sigma_f, 0.01)
        d_f = d_dec[:, feature_id].to(device)  # [d]

        # Compute mean-activation-addition direction
        print(f"[m6] computing mean-activation-addition direction for {prop}...", flush=True)
        delta_c = compute_mean_activation_direction(H, cached_seqs, checker, device).to(device)
        delta_scale = float(delta_c.norm().item()) + 1e-9
        # For random-clamp: pick a random OTHER feature id
        random_feature_id = int(rng.integers(0, d_dec.shape[1]))
        while random_feature_id == feature_id:
            random_feature_id = int(rng.integers(0, d_dec.shape[1]))
        d_random = d_dec[:, random_feature_id].to(device)
        sigma_random = float(fire_std[random_feature_id]) if random_feature_id < len(fire_std) else 1.0
        sigma_random = max(sigma_random, 0.01)

        # Sample seed sequences
        seeds_pick = seed_pool.sample(n=args.seed_sequences, random_state=hash(prop) % (2**32)).reset_index(drop=True)
        seed_seqs = seeds_pick["Sequence"].tolist()

        # Arms: no_steer, sae_clamp, mean_add, random_clamp
        # For no_steer: alpha=0 only
        # For others: dose ladder
        for run_seed in args.seeds:
            # Compute no-steering baseline first for plausibility band
            no_steer_ppl_list = []
            for i, sseq in enumerate(seed_seqs[: args.n_seq_per_batch]):
                comp, ppl = masked_fill_generation(
                    model, tok, sseq, layer_module, direction=None, alpha=0.0,
                    n_refill_steps=args.n_refill_steps, mask_fraction=args.mask_fraction,
                    device=device, seed=run_seed * 1000 + i, max_len=args.max_len,
                )
                hit = int(bool(checker(comp)))
                no_steer_ppl_list.append(ppl)
                all_rows.append({
                    "feature_id": feature_id, "target_property": prop, "arm": "no_steer",
                    "alpha": 0.0, "seed": run_seed, "seq_idx": i,
                    "yield_hit": hit, "pseudo_ppl": ppl, "in_band": True,
                })
                plausibility_rows.append({
                    "feature_id": feature_id, "arm": "no_steer", "alpha": 0.0, "seed": run_seed,
                    "seq_idx": i, "pseudo_ppl": ppl,
                })
            mean_ns_ppl = float(np.mean([p for p in no_steer_ppl_list if np.isfinite(p)])) if no_steer_ppl_list else 20.0
            band_upper = args.plausibility_band_factor * mean_ns_ppl

            for alpha in args.doses:
                for arm_name, direction, sigma in [
                    ("sae_clamp", d_f, sigma_f),
                    ("mean_add", delta_c / delta_scale, delta_scale),  # normalize then scale by sigma
                    ("random_clamp", d_random, sigma_random),
                ]:
                    eff_alpha = alpha * sigma
                    for i, sseq in enumerate(seed_seqs[: args.n_seq_per_batch]):
                        try:
                            comp, ppl = masked_fill_generation(
                                model, tok, sseq, layer_module, direction=direction, alpha=float(eff_alpha),
                                n_refill_steps=args.n_refill_steps, mask_fraction=args.mask_fraction,
                                device=device, seed=run_seed * 1000 + i, max_len=args.max_len,
                            )
                        except Exception as e:
                            print(f"[m6] gen failed feat={feature_id} arm={arm_name} alpha={alpha}: {e}", flush=True)
                            continue
                        in_band = (np.isfinite(ppl)) and (ppl <= band_upper)
                        hit = int(bool(checker(comp))) if in_band else 0
                        all_rows.append({
                            "feature_id": feature_id, "target_property": prop, "arm": arm_name,
                            "alpha": float(alpha), "seed": run_seed, "seq_idx": i,
                            "yield_hit": hit, "pseudo_ppl": float(ppl), "in_band": bool(in_band),
                        })
                        plausibility_rows.append({
                            "feature_id": feature_id, "arm": arm_name, "alpha": float(alpha),
                            "seed": run_seed, "seq_idx": i, "pseudo_ppl": float(ppl),
                        })
            print(f"[m6] feature {feature_id} ({prop}) seed {run_seed} done", flush=True)

    df_all = pd.DataFrame(all_rows)
    df_plaus = pd.DataFrame(plausibility_rows)
    df_all.to_parquet(out_dir / "yields.parquet", index=False)
    df_plaus.to_parquet(out_dir / "plausibility.parquet", index=False)

    # Summary
    md = ["# M6 — Feature-Clamp Steering Summary\n"]
    md.append(f"- Best layer: {best_layer}")
    md.append(f"- Features tested: {len(features)}")
    md.append("")
    for fdata in features:
        fid = int(fdata["feature_id"])
        prop = fdata["target_property"]
        sub = df_all[df_all["feature_id"] == fid]
        if sub.empty:
            continue
        md.append(f"## Feature {fid} ({prop}) — label: `{fdata['label']}`")
        agg = sub.groupby(["arm", "alpha"], as_index=False).agg(
            yield_mean=("yield_hit", "mean"),
            n=("yield_hit", "count"),
            band_pass=("in_band", "mean"),
            ppl_mean=("pseudo_ppl", lambda x: float(np.mean([p for p in x if np.isfinite(p)]))),
        )
        md.append("| arm | alpha | yield | band pass | mean pseudo-PPL | n |")
        md.append("|---|---|---|---|---|---|")
        for _, r in agg.iterrows():
            md.append(f"| {r['arm']} | {r['alpha']:.2f} | {r['yield_mean']:.3f} | {r['band_pass']:.2f} | {r['ppl_mean']:.2f} | {int(r['n'])} |")
        md.append("")

    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    elapsed = time.time() - t0
    n_gpu = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    (out_dir / "gpu_hours.txt").write_text(f"{elapsed / 3600 * n_gpu:.4f}\n")
    print(f"[m6] done in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
