"""M4 — LLM auto-interpretation of Swiss-Prot-unaligned SAE features.

For each unaligned feature f (from M2), sample top-activating protein sequence windows
from Swiss-Prot test + a UniRef sample, then run three prompts through gpt-5.4:
  A. label elicit — {label, justification, confidence}
  B. predictivity — decide which of 20 held-out + 20 low windows match the label
  C. synonym check — is the label a synonym of any Swiss-Prot category?
A feature is "novel-concept-coherent" iff s_auto >= tau_auto AND synonym check passes
AND random-feature control does not pass at the same rate.

Outputs:
  runs/m4/novel_concepts.parquet
  runs/m4/control_stats.json
  runs/m4/target_property_features.json   subset with checkable target-property labels for M6
  runs/m4/summary.md
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
    LLM_CACHE_DIR,
    LLMClient,
    UNIREF_DIR,
    load_esm,
    load_sae,
    seed_all,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--unaligned-features", type=str, default="runs/m2/unaligned_features.json")
    p.add_argument("--best-layer-file", type=str, default="runs/m2/best_layer.json")
    p.add_argument("--activations-dir", type=str, default="runs/m1/activations")
    p.add_argument("--codes-dir", type=str, default="runs/m1/sae_codes")
    p.add_argument("--unref", type=str, default=str(UNIREF_DIR))
    p.add_argument("--llm-endpoint", type=str, default="https://www.dmxapi.cn/v1")
    p.add_argument("--llm-model", type=str, default="gpt-5.4")
    p.add_argument("--window-len", type=int, default=21)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--held-out-k", type=int, default=20)
    p.add_argument("--low-k", type=int, default=20)
    p.add_argument("--tau-auto", type=float, default=0.3)
    p.add_argument("--n-features", type=int, default=200, help="Max unaligned features to label (reduced from 500)")
    p.add_argument("--n-control", type=int, default=50)
    p.add_argument("--n-unref-seqs", type=int, default=8000)
    p.add_argument("--max-len", type=int, default=1022)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--cache-dir", type=str, default=str(LLM_CACHE_DIR))
    p.add_argument("--out-dir", type=str, default="runs/m4")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


FIXED_VOCAB = [
    "binding_site", "active_site", "sequence_motif", "structural_domain",
    "functional_domain", "PTM_site", "signal_peptide", "transmembrane_domain",
    "zinc_binding", "glycosylation_site", "disulfide_bond",
    "beta_sheet", "alpha_helix", "loop_region", "coiled_coil",
    "N_terminal", "C_terminal", "phosphorylation_site", "N_glycosylation",
]


TARGET_PROPERTY_KEYWORDS = {
    "signal_peptide": ["signal peptide", "signal_peptide", "n-terminal signal", "secretion signal", "signal sequence"],
    "transmembrane_domain": ["transmembrane", "membrane-spanning", "hydrophobic helix", "tm helix", "transmembrane_domain"],
    "zinc_binding_motif": ["zinc binding", "zinc finger", "zinc-binding", "zn-binding", "cys2his2", "zn2+"],
    "n_glycosylation_site": ["n-glycosylation", "n_glycosylation", "n-glycan", "glycosylation", "asn-x-ser/thr"],
}


def find_target_property(label: str) -> str:
    ll = label.lower()
    for prop, keywords in TARGET_PROPERTY_KEYWORDS.items():
        if any(k in ll for k in keywords):
            return prop
    return ""


def format_window(seq: str, rp: int, half: int) -> str:
    lo = max(0, rp - half)
    hi = min(len(seq), rp + half + 1)
    return seq[lo:rp] + f"[{seq[rp]}*]" + seq[rp + 1 : hi]


def elicit_label(llm: LLMClient, windows: List[str]) -> Dict:
    top_str = "\n".join(f"{i+1}. {w}" for i, w in enumerate(windows))
    msg = [
        {"role": "system",
         "content": "You are an expert protein biochemist labeling a sparse-autoencoder feature. "
                    "You will see protein sequence windows where the FEATURE fires strongly (the marker '[X*]' is at "
                    "the target residue). Look for the biological pattern shared across windows: a motif, a specific "
                    "residue context, a fold cue, a modification site, a functional site, an amino-acid composition, etc. "
                    "Respond with JSON {\"label\": str, \"justification\": str, \"confidence\": float}. "
                    "The label must be ≤ 12 words and a concrete biological description. "
                    "If no coherent pattern is visible, set label to \"NONE\" and confidence to 0."},
        {"role": "user", "content": f"Top-activating windows:\n{top_str}\n\nRespond with JSON only."},
    ]
    return llm.chat_json(msg, max_tokens=300)


def score_predictivity(llm: LLMClient, label: str, candidates: List[Tuple[str, bool]]) -> Dict:
    cand_str = "\n".join(f"{i+1}. {w}" for i, (w, _) in enumerate(candidates))
    msg = [
        {"role": "system",
         "content": "You will decide, for each candidate protein sequence window, whether the window matches the "
                    "given LABEL. Return JSON {\"activated_indices\": [int]} — 1-based indices of matching windows."},
        {"role": "user", "content": f"LABEL: {label}\n\nCandidate windows:\n{cand_str}\n\nRespond with JSON only."},
    ]
    return llm.chat_json(msg, max_tokens=300)


def synonym_check(llm: LLMClient, label: str) -> Dict:
    vocab_str = ", ".join(FIXED_VOCAB)
    msg = [
        {"role": "system",
         "content": "You are checking whether a candidate protein-feature LABEL is a synonym or paraphrase of any "
                    "entry in a fixed vocabulary. Return JSON {\"is_synonym_of\": str|null, \"reason\": str}. "
                    "If the LABEL is a near-synonym of an entry, return that entry; otherwise null."},
        {"role": "user", "content": f"LABEL: {label}\nVOCABULARY: {vocab_str}\n\nRespond with JSON only."},
    ]
    return llm.chat_json(msg, max_tokens=200)


def sparse_column(codes, feature_id: int) -> np.ndarray:
    """Return a dense [N] array of activations for one feature from stored sparse codes."""
    shape = codes["shape"]
    N = int(shape[0])
    col_arr = codes["col"]
    row_arr = codes["row"]
    val_arr = codes["val"]
    mask = col_arr == feature_id
    dense = np.zeros(N, dtype=np.float32)
    dense[row_arr[mask]] = val_arr[mask]
    return dense


def main():
    args = parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load unaligned feature list
    uf = json.loads(Path(args.unaligned_features).read_text())
    best_layer = uf["best_layer"]
    unaligned_ids = uf["unaligned_feature_ids"]
    print(f"[m4] best_layer={best_layer} unaligned pool size={len(unaligned_ids)}", flush=True)

    # Load activations + codes for best layer (Swiss-Prot pool from M1)
    act_path = Path(args.activations_dir) / f"layer{best_layer}.pt"
    codes_path = Path(args.codes_dir) / f"layer{best_layer}.npz"
    act = torch.load(act_path, map_location="cpu", weights_only=False)
    codes = np.load(codes_path)
    sequences_sp = list(act["sequences"])
    offsets = np.array(act["offsets"], dtype=np.int64)
    N_sp = int(offsets.sum())
    # Build residue -> (seq_id, res_pos) map for the Swiss-Prot activations
    seq_id_per_res = np.empty(N_sp, dtype=np.int32)
    pos_per_res = np.empty(N_sp, dtype=np.int32)
    cursor = 0
    for si, Li in enumerate(offsets):
        seq_id_per_res[cursor : cursor + Li] = si
        pos_per_res[cursor : cursor + Li] = np.arange(Li)
        cursor += Li

    # For UniRef supplement: run ESM+SAE forward on --n-unref-seqs samples
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[m4] loading UniRef sample ({args.n_unref_seqs} seqs)...", flush=True)
    unref_files = sorted(Path(args.unref).glob("train-*.parquet"))
    unref_dfs = []
    remaining = args.n_unref_seqs
    for fp in unref_files:
        if remaining <= 0:
            break
        df = pd.read_parquet(fp, columns=["sequence"] if "sequence" in pd.read_parquet(fp).columns else None)
        # Determine sequence column
        cols = df.columns.tolist()
        seq_col = None
        for cand in ("sequence", "Sequence", "seq"):
            if cand in cols:
                seq_col = cand
                break
        if seq_col is None:
            print(f"[m4] warning: no sequence column in {fp} ({cols})")
            continue
        df = df[[seq_col]].rename(columns={seq_col: "Sequence"})
        take = min(remaining, len(df))
        unref_dfs.append(df.iloc[:take])
        remaining -= take
    if unref_dfs:
        unref_df = pd.concat(unref_dfs, ignore_index=True)
        # Filter by length
        unref_df["len"] = unref_df["Sequence"].str.len()
        unref_df = unref_df[(unref_df["len"] >= 30) & (unref_df["len"] <= args.max_len)].reset_index(drop=True).head(args.n_unref_seqs)
        ur_sequences = unref_df["Sequence"].tolist()
    else:
        ur_sequences = []
    print(f"[m4] UniRef sample: {len(ur_sequences)} sequences", flush=True)

    # Forward pass through ESM+SAE on UniRef sample, collect top-activating windows per feature
    # For efficiency: only track features we will label (unaligned + control set)
    rng = np.random.default_rng(args.seed)
    labeled_features = list(rng.choice(unaligned_ids, size=min(args.n_features, len(unaligned_ids)), replace=False))
    # Control: features matched on fire_frac to the labeled pool but NOT in that pool.
    # Interpretation: features drawn from the same normal-density regime (so window pools are comparable)
    # but with no privilege from being 'unaligned' — they are just other features.
    labeled_set = set(labeled_features)
    all_fire_frac = codes.get("fire_frac")
    if all_fire_frac is None:
        all_fire_frac = np.ones(int(codes["shape"][1]), dtype=np.float32)
    # For each labeled feature, we want a control with similar fire_frac. Just draw uniformly from
    # features with fire_frac in the same [1e-6, 0.2] range and not in the labeled set.
    pool = np.where((all_fire_frac >= 1e-6) & (all_fire_frac <= 0.2))[0]
    pool = [int(f) for f in pool if int(f) not in labeled_set]
    control_features = list(rng.choice(pool, size=min(args.n_control, len(pool)), replace=False))
    features_to_track = list(set(labeled_features) | set(control_features))
    features_set = set(features_to_track)
    print(f"[m4] tracking {len(features_to_track)} features "
          f"({len(labeled_features)} unaligned + {len(control_features)} control)", flush=True)

    # Per-feature top-K + held-out from UniRef
    K_track = args.top_k + args.held_out_k
    import heapq
    ur_feat_top: Dict[int, List[Tuple[float, int, int]]] = {u: [] for u in features_to_track}

    if ur_sequences:
        from common import ActivationCollector
        model, tok = load_esm(device)
        sae = load_sae(best_layer, "normalized", device).half()
        coll = ActivationCollector(model, tok, device, max_len=args.max_len, batch_size=args.batch_size)
        BATCH = 32
        for bs in range(0, len(ur_sequences), BATCH):
            batch = ur_sequences[bs : bs + BATCH]
            hs_list = coll.batch_hidden_states(batch, layer=best_layer)
            for i, hs in enumerate(hs_list):
                if hs.shape[0] == 0:
                    continue
                with torch.no_grad():
                    z = sae.encode(hs.to(device)).float().cpu().numpy()
                si = bs + i
                for u in features_to_track:
                    col = z[:, u]
                    nz = np.where(col > 0)[0]
                    if len(nz) == 0:
                        continue
                    top_pos = nz[np.argsort(-col[nz])[:3]]
                    for rp in top_pos:
                        act_v = float(col[rp])
                        entry = (act_v, si, int(rp))
                        if len(ur_feat_top[u]) < K_track:
                            heapq.heappush(ur_feat_top[u], entry)
                        else:
                            heapq.heappushpop(ur_feat_top[u], entry)
            if bs % (BATCH * 4) == 0:
                print(f"[m4] UniRef processed {bs}/{len(ur_sequences)}", flush=True)
    else:
        model = tok = sae = coll = None

    # Also get top firings on Swiss-Prot pool (from cached codes)
    sp_feat_top: Dict[int, List[Tuple[float, int, int]]] = {u: [] for u in features_to_track}
    # Group codes by column
    col_arr = codes["col"]
    row_arr = codes["row"]
    val_arr = codes["val"]
    order = np.argsort(col_arr, kind="stable")
    col_sorted = col_arr[order]
    row_sorted = row_arr[order]
    val_sorted = val_arr[order]
    # Group boundaries
    unique_cols, starts = np.unique(col_sorted, return_index=True)
    ends = np.append(starts[1:], len(col_sorted))
    col_to_range = dict(zip(unique_cols.tolist(), zip(starts.tolist(), ends.tolist())))
    for u in features_to_track:
        if u not in col_to_range:
            continue
        s, e = col_to_range[u]
        vs = val_sorted[s:e]
        rs = row_sorted[s:e]
        # take top-K
        if len(vs) == 0:
            continue
        top_idx = np.argsort(-vs)[:K_track]
        for idx in top_idx:
            r = rs[idx]
            sp_feat_top[u].append((float(vs[idx]), int(seq_id_per_res[r]), int(pos_per_res[r])))

    # Now for each labeled feature, run the three prompts
    llm = LLMClient(base_url=args.llm_endpoint, model=args.llm_model, cache_dir=Path(args.cache_dir))
    rows_out = []

    def build_windows_for_feature(u: int):
        # Combine SP top + UniRef top
        pool = ur_feat_top.get(u, []) + [(v, si + 10**7, p) for v, si, p in sp_feat_top.get(u, [])]
        pool.sort(key=lambda t: -t[0])
        top = pool[: args.top_k]
        held = pool[args.top_k : args.top_k + args.held_out_k]
        # Low windows: pick random low-activation windows
        low_pool = pool[-args.low_k :] if len(pool) > args.top_k + args.held_out_k else []
        # Format
        def fmt(entry):
            v, sid, rp = entry
            if sid < 10**7:
                seq = ur_sequences[sid]
            else:
                seq = sequences_sp[sid - 10**7]
            if not seq or rp >= len(seq):
                return None
            return format_window(seq, rp, args.window_len // 2)
        tops = [f for e in top for f in [fmt(e)] if f is not None]
        helds = [f for e in held for f in [fmt(e)] if f is not None]
        lows = [f for e in low_pool for f in [fmt(e)] if f is not None]
        return tops, helds, lows

    n_novel_real = 0
    for u in labeled_features:
        tops, helds, lows = build_windows_for_feature(u)
        if len(tops) < 5 or len(helds) < 4:
            rows_out.append({
                "feature_id": u, "label": None, "s_auto": 0.0,
                "synonym_of": None, "is_novel": False, "reason": "insufficient_windows",
                "target_property": "", "n_top": len(tops), "n_held": len(helds), "n_low": len(lows),
            })
            continue
        r_label = elicit_label(llm, tops)
        if not r_label.get("ok"):
            rows_out.append({
                "feature_id": u, "label": None, "s_auto": 0.0,
                "synonym_of": None, "is_novel": False, "reason": "llm_error_a",
                "target_property": "", "n_top": len(tops), "n_held": len(helds), "n_low": len(lows),
            })
            continue
        label = str(r_label["parsed"].get("label", "")).strip()
        if not label or label.upper() == "NONE":
            rows_out.append({
                "feature_id": u, "label": label, "s_auto": 0.0,
                "synonym_of": None, "is_novel": False, "reason": "no_label",
                "target_property": "", "n_top": len(tops), "n_held": len(helds), "n_low": len(lows),
            })
            continue
        # Predictivity
        cand = [(w, True) for w in helds] + [(w, False) for w in (lows or tops[-len(helds):])]
        rng2 = np.random.default_rng(u)
        rng2.shuffle(cand)
        r_pred = score_predictivity(llm, label, cand)
        if not r_pred.get("ok"):
            s_auto = 0.0
        else:
            picked_raw = r_pred["parsed"].get("activated_indices", [])
            try:
                picked = set(int(i) for i in picked_raw if isinstance(i, (int, float, str)))
            except Exception:
                picked = set()
            n_pos = sum(1 for _, y in cand if y)
            n_neg = sum(1 for _, y in cand if not y)
            hits_pos = sum(1 for i, (_, y) in enumerate(cand, 1) if y and i in picked)
            hits_neg = sum(1 for i, (_, y) in enumerate(cand, 1) if not y and i in picked)
            s_auto = (hits_pos / max(n_pos, 1)) - (hits_neg / max(n_neg, 1))
        # Synonym check
        r_syn = synonym_check(llm, label)
        syn_of = r_syn.get("parsed", {}).get("is_synonym_of") if r_syn.get("ok") else None
        if isinstance(syn_of, str) and syn_of.lower() in ("null", "none", ""):
            syn_of = None
        is_novel = (s_auto >= args.tau_auto) and (syn_of is None)
        target_prop = find_target_property(label) if is_novel else ""
        if is_novel:
            n_novel_real += 1
        rows_out.append({
            "feature_id": int(u), "label": label, "s_auto": float(s_auto),
            "synonym_of": syn_of, "is_novel": bool(is_novel), "reason": "",
            "target_property": target_prop,
            "n_top": len(tops), "n_held": len(helds), "n_low": len(lows),
        })

    df_out = pd.DataFrame(rows_out)
    df_out.to_parquet(out_dir / "novel_concepts.parquet", index=False)

    # Control: same protocol on `n_control` random features (labeled AS IF they were unaligned)
    n_novel_control = 0
    for u in control_features[: args.n_control]:
        tops, helds, lows = build_windows_for_feature(u)
        if len(tops) < 5 or len(helds) < 4:
            continue
        r_label = elicit_label(llm, tops)
        if not r_label.get("ok"):
            continue
        label = str(r_label["parsed"].get("label", "")).strip()
        if not label or label.upper() == "NONE":
            continue
        cand = [(w, True) for w in helds] + [(w, False) for w in (lows or tops[-len(helds):])]
        rng2 = np.random.default_rng(u + 999999)
        rng2.shuffle(cand)
        r_pred = score_predictivity(llm, label, cand)
        if not r_pred.get("ok"):
            continue
        picked_raw = r_pred["parsed"].get("activated_indices", [])
        try:
            picked = set(int(i) for i in picked_raw if isinstance(i, (int, float, str)))
        except Exception:
            picked = set()
        n_pos = sum(1 for _, y in cand if y)
        n_neg = sum(1 for _, y in cand if not y)
        hits_pos = sum(1 for i, (_, y) in enumerate(cand, 1) if y and i in picked)
        hits_neg = sum(1 for i, (_, y) in enumerate(cand, 1) if not y and i in picked)
        s_auto = (hits_pos / max(n_pos, 1)) - (hits_neg / max(n_neg, 1))
        r_syn = synonym_check(llm, label)
        syn_of = r_syn.get("parsed", {}).get("is_synonym_of") if r_syn.get("ok") else None
        if isinstance(syn_of, str) and syn_of.lower() in ("null", "none", ""):
            syn_of = None
        if (s_auto >= args.tau_auto) and (syn_of is None):
            n_novel_control += 1

    n_labeled = len(rows_out)
    control_stats = {
        "n_labeled_unaligned": n_labeled,
        "n_novel_real": int(n_novel_real),
        "novel_rate_real": (n_novel_real / max(n_labeled, 1)),
        "n_control": args.n_control,
        "n_novel_control": int(n_novel_control),
        "novel_rate_control": (n_novel_control / max(args.n_control, 1)),
        "specificity_ratio": (n_novel_real / max(n_novel_control, 1)) if n_novel_control > 0
            else float("inf") if n_novel_real > 0 else 0.0,
        "tau_auto": args.tau_auto,
    }
    (out_dir / "control_stats.json").write_text(json.dumps(control_stats, indent=2, default=str))

    # Target-property features for M6
    target_features = df_out[(df_out["is_novel"]) & (df_out["target_property"] != "")]
    # If none passed novelty, fall back to features whose synonym-check aligned to a known target-property vocab item
    if len(target_features) == 0:
        # Fallback: any labeled feature with target-property keyword in label
        df_out["target_property"] = df_out["label"].fillna("").apply(find_target_property)
        target_features = df_out[df_out["target_property"] != ""]
    # Keep at most 4 features, one per property when available
    picks = {}
    for _, r in target_features.iterrows():
        p = r["target_property"]
        if p and p not in picks:
            picks[p] = r
        if len(picks) >= 4:
            break
    target_feature_list = [
        {"feature_id": int(v["feature_id"]), "label": v["label"], "target_property": v["target_property"],
         "s_auto": float(v["s_auto"]), "layer": int(best_layer)}
        for v in picks.values()
    ]
    (out_dir / "target_property_features.json").write_text(json.dumps(
        {"best_layer": best_layer, "features": target_feature_list}, indent=2))

    md = [
        "# M4 — Novel-concept Auto-interpretation Summary\n",
        f"- Best layer: {best_layer}",
        f"- Unaligned features labeled: {n_labeled}",
        f"- Novel-concept features (s_auto ≥ {args.tau_auto} AND synonym check passes): **{n_novel_real}** "
        f"({100 * n_novel_real / max(n_labeled,1):.1f}%)",
        f"- Control (random features): {control_stats['n_novel_control']}/{args.n_control} "
        f"({100 * control_stats['novel_rate_control']:.1f}%)",
        f"- Specificity ratio (real / control): {control_stats['specificity_ratio']:.2f}",
        f"- Target-property features for M6: {len(target_feature_list)}",
    ]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    elapsed = time.time() - t0
    n_gpu = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    (out_dir / "gpu_hours.txt").write_text(f"{elapsed / 3600 * n_gpu:.4f}\n")
    print(f"[m4] done in {elapsed:.1f}s novel={n_novel_real}/{n_labeled} control={n_novel_control}/{args.n_control}",
          flush=True)


if __name__ == "__main__":
    main()
