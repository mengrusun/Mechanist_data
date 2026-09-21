"""Position-resolved activation patching.

For each attention head we patch its output `z` ONLY at a single
sequence-position group (fact-truth token, rule-truth token, query-prop
token, final-is token, or 'other') and measure the change in logit_diff.
The resulting (n_layers, n_heads, n_positions) tensor lets us cluster
heads by *where* in the sequence they matter, which we use as evidence
for functional modularity (claim 2).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from patching_utils import (
    load_hooked_mistral,
    load_hooked_gemma2_9b,
    load_dataset,
    token_id_for,
    logit_diff,
    FEWSHOT,
)


POSITION_LABELS = [
    "fact_truth",   # truth-value token of the antecedent fact ("<ante> is <TRUE>")
    "rule_ante",    # antecedent name-token inside the rule
    "rule_conseq",  # consequent name-token inside the rule
    "rule_truth",   # truth-value token of the rule consequent
    "query_prop",   # proposition name-token in the query
    "final_is",     # final "is" token (last position before answer)
    "other",
]


def find_positions(tok, prompt: str, antecedent: str, consequent: str):
    """Return {label: [positions]} for one prompt (token index list)."""
    ids = tok.encode(prompt, add_special_tokens=False)
    strs = [tok.decode([i]) for i in ids]
    labels = ["other"] * len(ids)

    # Look at the LAST occurrence of "Facts:", which marks the start of the
    # main (non-fewshot) prompt. This also means we ignore fewshot positions.
    def rfind_token(match_strs):
        for i in range(len(strs) - 1, -1, -1):
            if strs[i].strip() == match_strs:
                return i
        return None

    facts_i = None
    for i in range(len(ids) - 1, -1, -1):
        if strs[i].strip() == "Facts" or strs[i].strip().startswith("Facts"):
            facts_i = i
            break
    if facts_i is None:
        return labels, ids

    # Walk forward from Facts: index. Everything before is fewshot -> "other".
    # We find the specific antecedent/consequent tokens.
    ante_prefix = " " + antecedent
    cons_prefix = " " + consequent

    # After the "Facts:" segment, the first "<ante> is <TRUE/FALSE>" statement
    # is the antecedent fact. Then Rule: if <ante> is true then <cons> is <T/F>.
    # Then Question: <cons> is
    # Strategy: search token stream after facts_i for:
    #   1st occurrence of antecedent-name -> next 'is' -> then TRUE/FALSE (fact_truth)
    #   Then 'if' after 'Rule:' -> next antecedent-name is rule_ante
    #   Then 'then' -> next consequent-name is rule_conseq
    #   Then next 'is' after rule_conseq -> next TRUE/FALSE is rule_truth
    #   Then 'Question:' -> next consequent-name (or other prop) is query_prop
    #   Final 'is' at very end is final_is
    ante_tok = tok.encode(ante_prefix, add_special_tokens=False)
    cons_tok = tok.encode(cons_prefix, add_special_tokens=False)
    if len(ante_tok) != 1 or len(cons_tok) != 1:
        # multi-token names — fallback: leave all "other" except final_is
        labels[-1] = "final_is"
        return labels, ids

    ante_id = ante_tok[0]
    cons_id = cons_tok[0]

    true_id = tok.encode(" true", add_special_tokens=False)[-1]
    false_id = tok.encode(" false", add_special_tokens=False)[-1]

    # Everything before "Facts:" from the last "Facts:" occurrence -> other
    # (fewshot lives before)
    # walk from facts_i onwards
    i = facts_i + 1
    seen = {}
    N = len(ids)
    # 1. antecedent fact: first ante_id, then next "is", then next true/false
    while i < N and ids[i] != ante_id:
        i += 1
    if i < N and "fact_ante_name" not in seen:
        seen["fact_ante_name"] = i
        i += 1
    while i < N and strs[i].strip() != "is":
        i += 1
    i += 1
    if i < N and ids[i] in (true_id, false_id):
        labels[i] = "fact_truth"
        i += 1
    # 2. Rule: skip to next "if"
    while i < N and strs[i].strip() != "if":
        i += 1
    i += 1
    if i < N and ids[i] == ante_id:
        labels[i] = "rule_ante"
        i += 1
    while i < N and strs[i].strip() != "then":
        i += 1
    i += 1
    if i < N and ids[i] == cons_id:
        labels[i] = "rule_conseq"
        i += 1
    while i < N and strs[i].strip() != "is":
        i += 1
    i += 1
    if i < N and ids[i] in (true_id, false_id):
        labels[i] = "rule_truth"
        i += 1
    # 3. Query: skip to "Question", then next prop
    while i < N and not strs[i].strip().startswith("Question"):
        i += 1
    i += 1
    while i < N and strs[i].strip() == ":":
        i += 1
    if i < N:
        # first name token after Question:
        labels[i] = "query_prop"
        i += 1
    # 4. final "is" at end
    if strs[-1].strip() == "is":
        labels[-1] = "final_is"

    return labels, ids


def pad_stack(seqs, pad_id, device):
    m = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    pos_offset = []
    for i, s in enumerate(seqs):
        padded[i, m - len(s):] = s
        pos_offset.append(m - len(s))
    return padded.to(device), pos_offset


def run_positional_patching(model, tok, ds, device, top_heads, top_mlps, max_batch=16, verbose=True):
    """For every (layer, head) in top_heads and every position label,
    patch only at that position group.

    Returns array shape (n_top_heads, n_positions) of normalized effect.
    """
    n_pos = len(POSITION_LABELS)

    # Precompute label arrays for each example
    label_data = []
    for r in ds:
        pl, ids = find_positions(tok, r["clean"], r["antecedent"], r["consequent"])
        label_data.append((pl, ids))

    clean_ans = torch.tensor([token_id_for(tok, r["clean_answer"]) for r in ds]).to(device)
    corr_ans = torch.tensor([token_id_for(tok, r["corrupt_answer"]) for r in ds]).to(device)

    # Baseline
    with torch.no_grad():
        clean_ld_total = 0.0
        corrupt_ld_total = 0.0
        for i in range(0, len(ds), max_batch):
            batch = ds[i:i + max_batch]
            cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
            crids = [torch.tensor(tok.encode(r["corrupt"])) for r in batch]
            cin, _ = pad_stack(cids, tok.pad_token_id, device)
            crin, _ = pad_stack(crids, tok.pad_token_id, device)
            clean_logits = model(cin)[:, -1, :]
            corr_logits = model(crin)[:, -1, :]
            clean_ld_total += logit_diff(clean_logits, clean_ans[i:i+max_batch], corr_ans[i:i+max_batch]).sum().item()
            corrupt_ld_total += logit_diff(corr_logits, clean_ans[i:i+max_batch], corr_ans[i:i+max_batch]).sum().item()
    clean_ld = clean_ld_total / len(ds)
    corrupt_ld = corrupt_ld_total / len(ds)
    denom = clean_ld - corrupt_ld
    if verbose:
        print(f"baseline clean_ld={clean_ld:.3f}, corrupt_ld={corrupt_ld:.3f}, denom={denom:.3f}")

    n_top = len(top_heads) + len(top_mlps)
    effects_sum = np.zeros((n_top, n_pos), dtype=np.float64)
    counts = np.zeros((n_top, n_pos), dtype=np.int64)

    for bi in range(0, len(ds), max_batch):
        batch = ds[bi:bi + max_batch]
        cids = [torch.tensor(tok.encode(r["clean"])) for r in batch]
        crids = [torch.tensor(tok.encode(r["corrupt"])) for r in batch]
        cin, off_c = pad_stack(cids, tok.pad_token_id, device)
        crin, off_cr = pad_stack(crids, tok.pad_token_id, device)
        S = cin.shape[1]
        # (label -> per-example: bool 1D mask of shape (S,))
        pos_masks = {p: torch.zeros((len(batch), S), dtype=torch.bool, device=device) for p in POSITION_LABELS}
        pos_present = {p: torch.zeros(len(batch), dtype=torch.bool) for p in POSITION_LABELS}
        for k in range(len(batch)):
            pl, ids = label_data[bi + k]
            pad_off = off_c[k]
            for pos_idx, lab in enumerate(pl):
                pos_masks[lab][k, pad_off + pos_idx] = True
                pos_present[lab][k] = True

        ca = clean_ans[bi:bi + max_batch]
        cra = corr_ans[bi:bi + max_batch]

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(
                cin,
                names_filter=lambda n: n.endswith("hook_z") or n.endswith("hook_mlp_out"),
            )

        for hi, comp in enumerate(list(top_heads) + list(top_mlps)):
            for pi, pos_lab in enumerate(POSITION_LABELS):
                mask = pos_masks[pos_lab]  # (B, S)
                present = pos_present[pos_lab]  # (B,)
                if not present.any():
                    continue
                if comp[0] == "head":
                    layer, head = comp[1], comp[2]
                    clean_z = clean_cache[f"blocks.{layer}.attn.hook_z"]
                    def hook_fn(z, hook, cz=clean_z, h=head, m=mask):
                        z[m, h, :] = cz[m, h, :]
                        return z
                    hook_name = f"blocks.{layer}.attn.hook_z"
                else:
                    layer = comp[1]
                    clean_mlp = clean_cache[f"blocks.{layer}.hook_mlp_out"]
                    def hook_fn(mo, hook, cm=clean_mlp, m=mask):
                        mo[m] = cm[m]
                        return mo
                    hook_name = f"blocks.{layer}.hook_mlp_out"
                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        crin, fwd_hooks=[(hook_name, hook_fn)],
                    )[:, -1, :]
                ld_p_per = logit_diff(patched_logits, ca, cra).cpu().numpy()
                # only count examples where this position exists
                mask_np = present.cpu().numpy()
                effects_sum[hi, pi] += ld_p_per[mask_np].sum()
                counts[hi, pi] += mask_np.sum()
        del clean_cache
        torch.cuda.empty_cache()
        if verbose:
            print(f"  batch {bi // max_batch + 1}/{(len(ds) + max_batch - 1) // max_batch} done")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ld = effects_sum / np.maximum(counts, 1)
    normalized = (mean_ld - corrupt_ld) / denom
    normalized[counts == 0] = np.nan
    effects = normalized.astype(np.float32)

    return effects, {
        "clean_ld": clean_ld,
        "corrupt_ld": corrupt_ld,
        "denom": denom,
        "n_examples": len(ds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--data", default="data/logic_ds.jsonl")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--head-npz", default="results/head_patching_mistral.npz",
                    help="npz file from head_patching.py")
    ap.add_argument("--top-heads", type=int, default=20)
    ap.add_argument("--top-mlps", type=int, default=6)
    ap.add_argument("--out", default="results/positional_patching_mistral.npz")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # Load top heads/MLPs from previous experiment
    d = np.load(args.head_npz)
    head_eff = d["head_effects"]
    mlp_eff = d["mlp_effects"]
    top_h_idx = np.argsort(-np.abs(head_eff).ravel())[: args.top_heads]
    top_heads = []
    for idx in top_h_idx:
        L, H = np.unravel_index(idx, head_eff.shape)
        top_heads.append(("head", int(L), int(H)))
    top_m_idx = np.argsort(-np.abs(mlp_eff))[: args.top_mlps]
    top_mlps = [("mlp", int(L)) for L in top_m_idx]
    print("selected heads:", top_heads)
    print("selected mlps:", top_mlps)

    if args.model == "mistral":
        model, tok = load_hooked_mistral(
            "/data/zhenqian/models/Mistral-7B-v0.1", device=args.device
        )
    elif args.model == "gemma-2-9b":
        model, tok = load_hooked_gemma2_9b(
            "/data/zhenqian/models/gemma-2-9b", device=args.device
        )
    else:
        raise ValueError(args.model)

    ds = load_dataset(args.data)[: args.n]

    # sanity check on 1st example: print label map
    labels0, ids0 = find_positions(tok, ds[0]["clean"], ds[0]["antecedent"], ds[0]["consequent"])
    tail = 20
    print("sample tail tokens/labels:")
    for tk, lab in list(zip([tok.decode([i]) for i in ids0], labels0))[-tail:]:
        print(f"  {repr(tk):>10s} -> {lab}")

    t0 = time.time()
    eff, stats = run_positional_patching(
        model, tok, ds, args.device, top_heads, top_mlps, max_batch=args.batch
    )
    dt = time.time() - t0
    print(f"took {dt:.1f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        effects=eff,
        top_heads=np.array(top_heads, dtype=object),
        top_mlps=np.array(top_mlps, dtype=object),
        position_labels=np.array(POSITION_LABELS),
        **stats,
    )
    # print table
    header = "component      " + " ".join(f"{p:>10s}" for p in POSITION_LABELS)
    print(header)
    for i, comp in enumerate(list(top_heads) + list(top_mlps)):
        row = " ".join(f"{eff[i, j]:+10.3f}" for j in range(len(POSITION_LABELS)))
        print(f"{str(comp):15s} {row}")


if __name__ == "__main__":
    main()
