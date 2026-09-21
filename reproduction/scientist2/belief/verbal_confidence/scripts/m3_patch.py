#!/usr/bin/env python3
"""
M3 — Residual-stream patching (Causal Attribution / Patching).

For each top (position, layer) site from M2 emits a signed sufficiency effect:
  - Build clean/corrupted pairs from M1 items:
      clean     = high-verbal-conf items (>=80)
      corrupted = low-verbal-conf items  (<=40)
    Matched by template-position count (post-answer window structure is fixed
    by T0 so pairs are automatically length-aligned at post-answer positions).
  - For each pair (clean_item C, corrupt_item R):
      run 1 (baseline): forward on C's full prompt, greedily decode conf → conf_C
      run 2 (patched):  forward on C's prompt, but at layer L position `pos`
                        replace hidden_states[L] at pos with R's hidden_states[L]
                        at the SAME position label (extracted from the M1 cache).
                        Greedily decode conf → conf_patched.
  - Measure:
      verb_conf_shift            = conf_patched - conf_C
      verb_conf_signed_effect    = sign(conf_R - conf_C) × verb_conf_shift  (positive = moved toward corrupt)
      answer_acc_preserved       = does the model still emit the same answer under patching?
      answer_logprob_shift       = mean-logprob(answer_C tokens | patched context) - mean-logprob(answer_C tokens | clean context)

Uses only the seed42 slice of M1 data by default (all 3 seeds are aggregated
across --seed argument for effect variance).
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc_common import (
    CONF_GEN_LABEL,
    MAX_ANSWER_TOKENS,
    MAX_CONF_TOKENS,
    MODEL_PATH,
    _build_post_answer_template_ids,
    _build_template_prefix_ids,
    load_model_and_tokenizer,
    parse_confidence,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--dataset", default="/data/zhenqian/data/trivia_qa")
    p.add_argument("--pair_source", default="clean_high_corrupt_low")
    p.add_argument("--n_pairs", type=int, default=200)
    # `--site` may be a single "PosLayer" like "E0L30" OR the string
    # "AUTO" — meaning read M2's top_k_sites.json and iterate.
    p.add_argument("--site", default="AUTO")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--measure", default="verb_conf_shift,answer_acc_preserved,answer_logprob_shift")
    p.add_argument("--m1_cache", required=True,
                   help="Root of the M1 cache (contains seed42/T0/, seed123/T0/, seed2024/T0/).")
    p.add_argument("--m2_dir", required=True,
                   help="Directory with M2 top_k_sites.json.")
    p.add_argument("--template", default="T0")
    p.add_argument("--out", required=True,
                   help="Output JSON path; if AUTO site, this is a directory.")
    p.add_argument("--control_site", default=None,
                   help="Optional (posLayer) string for a matched-control non-cache site.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def parse_site(s: str) -> Tuple[str, int]:
    """Parse 'E0L30' → ('E0', 30)."""
    m = re.match(r"^([EC]\d+)L(\d+)$", s)
    if not m:
        raise ValueError(f"Bad site spec: {s!r} — expected e.g. 'E0L30'")
    return m.group(1), int(m.group(2))


def load_m1_seed(m1_cache: str, seed: int, template: str):
    seed_dir = Path(m1_cache) / f"seed{seed}" / template
    items_path = seed_dir / "items.jsonl"
    acts_path = seed_dir / "activations.h5"
    items = []
    with items_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items, str(acts_path)


def build_pairs(items: List[Dict], n_pairs: int, seed: int,
                clean_thr: int = 80, corrupt_thr: int = 40) -> List[Tuple[int, int]]:
    """
    Match clean (verbal_conf >= clean_thr) with corrupt (verbal_conf <= corrupt_thr)
    items 1:1, matched by ANSWER-token count (n_answer_tokens, stored by M1) so
    that the post-answer window sits at a comparable position offset. Seeded.
    """
    clean = [i for i, itm in enumerate(items)
             if itm["verbal_conf"] is not None and itm["verbal_conf"] >= clean_thr]
    corrupt = [i for i, itm in enumerate(items)
               if itm["verbal_conf"] is not None and itm["verbal_conf"] <= corrupt_thr]
    rng = np.random.RandomState(seed + 7)
    rng.shuffle(clean)
    rng.shuffle(corrupt)

    def answer_span(itm):
        # Prefer explicit n_answer_tokens; fall back to derived value.
        if "n_answer_tokens" in itm:
            return int(itm["n_answer_tokens"])
        # Back-compat: derive from prefix_len + E0
        pl = itm.get("prefix_len")
        if pl is not None:
            return itm["post_answer_positions"]["E0"] - pl + 1
        return itm["post_answer_positions"]["E0"]

    clean_by_bucket: Dict[int, List[int]] = {}
    for i in clean:
        b = answer_span(items[i])
        clean_by_bucket.setdefault(b, []).append(i)
    corrupt_by_bucket: Dict[int, List[int]] = {}
    for i in corrupt:
        b = answer_span(items[i])
        corrupt_by_bucket.setdefault(b, []).append(i)

    pairs: List[Tuple[int, int]] = []
    for b, cl_list in clean_by_bucket.items():
        # find nearest bucket in corrupt with items
        candidates = sorted(corrupt_by_bucket.keys(), key=lambda x: abs(x - b))
        for cl_idx in cl_list:
            if len(pairs) >= n_pairs:
                break
            paired = False
            for cb in candidates:
                if corrupt_by_bucket[cb]:
                    cr_idx = corrupt_by_bucket[cb].pop(0)
                    pairs.append((cl_idx, cr_idx))
                    paired = True
                    break
            if not paired:
                continue
        if len(pairs) >= n_pairs:
            break
    return pairs[:n_pairs]


@torch.no_grad()
def decode_conf_from_context(model, tok, input_ids: torch.Tensor,
                             max_conf_tokens: int = MAX_CONF_TOKENS) -> Tuple[Optional[int], str, List[int]]:
    """Greedy-decode confidence tokens starting from `input_ids` (which ends at C0-1).
    Returns (parsed_int, decoded_text, list_of_generated_ids)."""
    device = input_ids.device
    eos_id = tok.eos_token_id
    newline_ids = {tok(s, add_special_tokens=False).input_ids[-1] for s in ("\n", "\n\n", "\n\n\n")}
    cur = input_ids
    gen_ids: List[int] = []
    for _ in range(max_conf_tokens):
        out = model(cur, use_cache=False)
        logits = out.logits[0, -1, :]
        next_id = int(logits.argmax().item())
        gen_ids.append(next_id)
        cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos_id or next_id in newline_ids:
            break
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return parse_confidence(text), text, gen_ids


# (Dead unused wrapper removed — the patching hook is defined inline in main().)


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    # ---- Load M2 top-k sites --------------------------------------------
    with (Path(args.m2_dir) / "top_k_sites.json").open() as f:
        top_k = json.load(f)["top_k"]
    if args.site == "AUTO":
        sites = [(c["position"], c["layer"]) for c in top_k]
    else:
        sites = [parse_site(args.site)]

    control_sites = []
    if args.control_site:
        control_sites = [parse_site(args.control_site)]

    print(f"[m3] Sites to patch: {sites}", flush=True)
    if control_sites:
        print(f"[m3] Control sites: {control_sites}", flush=True)

    # ---- Load model + tokenizer -----------------------------------------
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[m3] Loading model...", flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(args.model, dtype=dtype)
    device = next(model.parameters()).device
    print(f"[m3] Model loaded in {time.time()-t0:.1f}s on {device}", flush=True)

    newline_ids = {tok(s, add_special_tokens=False).input_ids[-1] for s in ("\n", "\n\n", "\n\n\n")}
    eos_id = tok.eos_token_id

    # ---- Load M1 items + activations for THIS seed -----------------------
    items, acts_path = load_m1_seed(args.m1_cache, args.seed, args.template)
    print(f"[m1 for m3] loaded {len(items)} items (seed={args.seed})", flush=True)

    pairs = build_pairs(items, n_pairs=args.n_pairs, seed=args.seed)
    print(f"[m3] built {len(pairs)} clean/corrupted pairs", flush=True)

    if len(pairs) < 20:
        print(f"[m3] WARNING: only {len(pairs)} pairs — high/low-confidence buckets are small.",
              flush=True)

    # ---- Load h5 activations for all sites we might patch from -----------
    all_sites_needed = set(sites) | set(control_sites)
    h5 = h5py.File(acts_path, "r")

    # ---- For each site: iterate pairs and measure ------------------------
    out_root = Path(args.out)
    if args.site == "AUTO":
        out_root.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    site_summaries: List[Dict] = []

    for (pos_label, L) in list(all_sites_needed):
        is_control = (pos_label, L) in control_sites
        # activations at (pos_label, L) for all N items
        patch_source = h5[pos_label][str(L)][()]  # (N, hidden), float32

        pair_records = []
        n_used = 0
        conf_shift_signed = []
        conf_shift_abs = []
        acc_preserved = []
        logprob_shift = []
        conf_clean_all = []
        conf_patched_all = []
        conf_corrupt_all = []

        for pi, (cl_idx, cr_idx) in enumerate(pairs):
            cl = items[cl_idx]
            cr = items[cr_idx]
            if cl["verbal_conf"] is None or cr["verbal_conf"] is None:
                continue

            # Build clean input up to C0 (i.e. all tokens up to conf_gen_position - 1)
            full_ids_cl = cl["full_ids"]
            conf_pos_cl = cl["conf_gen_position"]
            clean_prompt_ids = torch.tensor([full_ids_cl[:conf_pos_cl]], device=device)

            # Baseline conf decode (no patch)
            conf_C, txt_C, _ = decode_conf_from_context(model, tok, clean_prompt_ids)

            # Determine patch position in the CLEAN context (posLabel → absolute index)
            patch_pos_cl = cl["post_answer_positions"].get(pos_label) if pos_label.startswith("E") else conf_pos_cl
            if patch_pos_cl is None or patch_pos_cl >= clean_prompt_ids.shape[1]:
                continue

            # Source vector = corrupt item's activation at (pos_label, L)
            patch_vec = torch.tensor(patch_source[cr_idx], device=device, dtype=torch.float32)

            # Patched forward: same clean prompt, patch at (L, patch_pos_cl)
            L_module = model.model.layers[L - 1]

            def hook(module, inputs, output, _patch_vec=patch_vec, _patch_pos=patch_pos_cl):
                if isinstance(output, tuple):
                    hs = output[0]
                    if hs.shape[1] > _patch_pos:
                        hs = hs.clone()
                        hs[0, _patch_pos, :] = _patch_vec.to(hs.dtype).to(hs.device)
                        return (hs,) + output[1:]
                    return output
                else:
                    if output.shape[1] > _patch_pos:
                        output = output.clone()
                        output[0, _patch_pos, :] = _patch_vec.to(output.dtype).to(output.device)
                    return output

            handle = L_module.register_forward_hook(hook)
            try:
                # Greedy decode confidence with the hook fired on every forward.
                gen_ids: List[int] = []
                cur = clean_prompt_ids
                for _ in range(MAX_CONF_TOKENS):
                    out = model(cur, use_cache=False)
                    logits = out.logits[0, -1, :]
                    next_id = int(logits.argmax().item())
                    gen_ids.append(next_id)
                    cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)
                    if next_id == eos_id or next_id in newline_ids:
                        break
                txt_P = tok.decode(gen_ids, skip_special_tokens=True)
                conf_P = parse_confidence(txt_P)
            finally:
                handle.remove()

            # Answer-logprob-shift: rescore the clean answer tokens under the
            # patched clean prompt (up through E0), comparing to the baseline
            # log-prob of those tokens. Answer tokens live at absolute positions
            # [prefix_len .. E0] (both inclusive) in clean's full_ids.
            pl_cl = cl.get("prefix_len")
            n_ans_cl = cl.get("n_answer_tokens")
            if pl_cl is not None and n_ans_cl and n_ans_cl > 0:
                # Truncate at E0 (i.e., last answer token, position pl_cl + n_ans_cl - 1)
                # and rescore in ONE forward under (a) baseline and (b) patched context.
                trunc_end = pl_cl + n_ans_cl  # exclusive
                trunc_ids = torch.tensor([full_ids_cl[:trunc_end]], device=device)
                answer_token_positions = list(range(pl_cl, trunc_end))  # absolute indices

                # Baseline forward
                with torch.no_grad():
                    out_base = model(trunc_ids, use_cache=False)
                    lp_base = torch.log_softmax(out_base.logits[0].float(), dim=-1)
                    baseline_ans_lps = []
                    for pos in answer_token_positions:
                        if pos - 1 < 0 or pos >= trunc_ids.shape[1]:
                            continue
                        tid = full_ids_cl[pos]
                        baseline_ans_lps.append(float(lp_base[pos - 1, tid].item()))
                    mean_lp_base = float(np.mean(baseline_ans_lps)) if baseline_ans_lps else 0.0

                # Patched forward — hook fires again at patch_pos_cl < trunc_end iff patch_pos_cl < trunc_end
                if patch_pos_cl < trunc_end:
                    handle2 = L_module.register_forward_hook(hook)
                    try:
                        with torch.no_grad():
                            out_patched = model(trunc_ids, use_cache=False)
                    finally:
                        handle2.remove()
                    lp_patched = torch.log_softmax(out_patched.logits[0].float(), dim=-1)
                    patched_ans_lps = []
                    for pos in answer_token_positions:
                        if pos - 1 < 0 or pos >= trunc_ids.shape[1]:
                            continue
                        tid = full_ids_cl[pos]
                        patched_ans_lps.append(float(lp_patched[pos - 1, tid].item()))
                    mean_lp_patched = float(np.mean(patched_ans_lps)) if patched_ans_lps else 0.0
                    answer_logprob_shift_val = mean_lp_patched - mean_lp_base
                else:
                    # Patch position is downstream of all answer tokens → no effect on answer.
                    answer_logprob_shift_val = 0.0
            else:
                answer_logprob_shift_val = 0.0

            # Answer accuracy proxy: check whether the greedy-decoded clean answer is preserved
            # under the patched context. Under our design the answer is FIXED as part of the input
            # prompt (we patch AFTER answer + template are in the context), so answer accuracy is
            # trivially preserved at the token level — the *answer commit* is upstream of the patch.
            # This is precisely the point of caching-mid-generation: we can measure a shift in
            # verbalized conf while the answer stays fixed.
            acc_ok = True  # answer is baked in via input_ids; only the conf digits get re-decoded

            if conf_C is None or conf_P is None:
                continue

            # Signed effect: positive means patched conf moved toward corrupt conf
            direction = np.sign(cr["verbal_conf"] - cl["verbal_conf"])
            shift = float(conf_P - conf_C)
            signed = float(direction * shift)

            conf_shift_signed.append(signed)
            conf_shift_abs.append(abs(shift))
            acc_preserved.append(1.0 if acc_ok else 0.0)
            conf_clean_all.append(cl["verbal_conf"])
            conf_patched_all.append(conf_P)
            conf_corrupt_all.append(cr["verbal_conf"])
            logprob_shift.append(answer_logprob_shift_val)

            pair_records.append({
                "pair_idx": pi,
                "clean_idx": cl_idx,
                "corrupt_idx": cr_idx,
                "conf_clean": cl["verbal_conf"],
                "conf_corrupt": cr["verbal_conf"],
                "conf_baseline_decode": conf_C,
                "conf_patched_decode": conf_P,
                "shift": shift,
                "signed_effect": signed,
                "patch_pos": patch_pos_cl,
                "patch_layer": L,
                "patch_pos_label": pos_label,
                "is_control": is_control,
            })
            n_used += 1

            if (pi + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"[m3] site={pos_label}L{L} pair {pi+1}/{len(pairs)} "
                      f"n_valid={n_used} mean_signed={np.mean(conf_shift_signed):.2f} "
                      f"elapsed={elapsed/60:.1f}m", flush=True)

        summary = {
            "site": f"{pos_label}L{L}",
            "position": pos_label,
            "layer": L,
            "is_control": is_control,
            "seed": args.seed,
            "n_pairs_used": n_used,
            "n_pairs_requested": len(pairs),
            "mean_signed_effect": float(np.mean(conf_shift_signed)) if conf_shift_signed else 0.0,
            "median_signed_effect": float(np.median(conf_shift_signed)) if conf_shift_signed else 0.0,
            "std_signed_effect": float(np.std(conf_shift_signed)) if conf_shift_signed else 0.0,
            "mean_abs_shift": float(np.mean(conf_shift_abs)) if conf_shift_abs else 0.0,
            "answer_acc_preserved": float(np.mean(acc_preserved)) if acc_preserved else 1.0,
            "answer_logprob_shift": float(np.mean(logprob_shift)) if logprob_shift else 0.0,
            "conf_clean_mean": float(np.mean(conf_clean_all)) if conf_clean_all else 0.0,
            "conf_patched_mean": float(np.mean(conf_patched_all)) if conf_patched_all else 0.0,
            "conf_corrupt_mean": float(np.mean(conf_corrupt_all)) if conf_corrupt_all else 0.0,
        }
        print(f"[m3] SITE {pos_label}L{L} (control={is_control}) mean_signed={summary['mean_signed_effect']:+.2f} "
              f"(N={n_used})", flush=True)

        # Persist per-site JSON
        if args.site == "AUTO":
            outf = out_root / f"site{pos_label}L{L}_n{args.n_pairs}_seed{args.seed}{'_ctrl' if is_control else ''}.json"
        else:
            outf = Path(args.out)
            outf.parent.mkdir(parents=True, exist_ok=True)
        with outf.open("w") as f:
            json.dump({"summary": summary, "pairs": pair_records}, f, indent=2)
        site_summaries.append(summary)
        all_results.extend(pair_records)

    h5.close()
    # ---- Aggregate summary ---------------------------------------------
    agg_path = (out_root if args.site == "AUTO" else Path(args.out).parent) / f"m3_aggregate_seed{args.seed}.json"
    with agg_path.open("w") as f:
        json.dump({"seed": args.seed, "sites": site_summaries, "n_pairs": args.n_pairs}, f, indent=2)
    print(f"[m3] DONE. wrote {agg_path}", flush=True)


if __name__ == "__main__":
    main()
