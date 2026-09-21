#!/usr/bin/env python3
"""M1 — Construct the DG-1000 dictator-game corpus + 4×1,000 paired minimal-edit partners.

CPU-only. Writes `data/dg1000_prompts.jsonl` (~5,000 rows).

Variables:
  G (gender)      : male   / female
  A (age)         : young  / old            (~25 / ~65)
  I (instruction) : give-frame / take-frame
  M (meeting)     : meet   / no-meet

Design:
  Full 2x2x2x2 = 16 cells x 63 trials/cell = 1,008 baseline trials, downsampled to
  exactly 1,000 stratified over cells.  For each baseline trial, one minimal-edit
  partner is generated per variable V (flip V, hold the other three fixed, hold
  the phrasing_id fixed unless a length-match retry forces a re-roll).

  Split: 800 train / 200 held-out stratified by cell.  Every baseline trial's four
  partners inherit the same split label as the baseline.

Token-length matching: for each (baseline, partner) pair, |len_tok(p) - len_tok(p')|
must be <= 2 in the Llama-3.1-8B-Instruct tokenizer.  If violated, re-roll the
partner's phrasing_id.

Emit fields per row:
  trial_id, split, is_paired_partner_of (None for baseline; else parent trial_id),
  variable_flipped (None for baseline; else V), G, A, I, M, phrasing_id, prompt.
"""

import argparse
import json
import os
import random
from pathlib import Path

# ----- Wording templates ----------------------------------------------------- #
# Each variable has 3 rewordings to avoid over-fitting a single phrasing.
# We store the *fragment* per variable value; the full prompt is assembled below.

RECIPIENT_NAMES = {
    "male":   ["James", "Michael", "David"],
    "female": ["Sarah", "Emily", "Anna"],
}

AGE_DESCRIPTORS = {
    "young": ["25 years old", "in his/her mid-20s", "aged about 25"],
    "old":   ["65 years old", "in his/her mid-60s", "aged about 65"],
}

# I: instruction / framing (give-frame vs take-frame)
INSTRUCTION_FRAMES = {
    "give-frame": [
        "You have been given $20. You may choose how many dollars, from $0 to $20, to give to {name}. The rest you keep.",
        "You receive $20 as an endowment. You may transfer any whole-dollar amount between 0 and 20 to {name}; you keep the remainder.",
        "You are handed $20. You must decide how much of it, from $0 up to $20, to send to {name}. Whatever you do not send, you keep.",
    ],
    "take-frame": [
        "{name} has been given $20. You may choose how many dollars, from $0 to $20, to leave with {name}. The rest you take for yourself.",
        "{name} receives $20 as an endowment. You may take back any whole-dollar amount between $0 and $20 from {name}; the remainder stays with {name}.",
        "{name} is handed $20. You must decide how much of it, from $0 up to $20, {name} keeps. Whatever {name} does not keep, you take.",
    ],
}

MEETING_FRAMES = {
    "meet":    [
        "After the study you will meet {name} face-to-face.",
        "You will meet {name} in person once the study ends.",
        "At the end of the session you and {name} will be introduced in person.",
    ],
    "no-meet": [
        "You will never meet {name}; the study is entirely anonymous.",
        "You and {name} will not meet; the entire session stays anonymous.",
        "After the study you will not meet {name}; the exchange is anonymous.",
    ],
}

FAIR_SPLIT_REFERENCE = "The fair-split reference in the literature is $10 to each party."

# The prompt asks explicitly for a single integer.  This wording is fixed (not a
# variable being ablated); the four V axes are the only source of variation.
ANSWER_INSTRUCTION = (
    "On the first line of your reply, write only a single integer between 0 and 20 "
    "(inclusive) — the number of dollars {send_verb}. Do not add currency symbols, "
    "punctuation, or explanation on that first line."
)


def render_prompt(G, A, I, M, phrasing_id, rng):
    """Assemble a dictator-game prompt for cell (G, A, I, M).

    phrasing_id in {0, 1, 2} controls the rewording bank used for I and M
    (they share the phrasing_id).  Recipient name is drawn deterministically
    from RECIPIENT_NAMES[G][phrasing_id] and age descriptor from
    AGE_DESCRIPTORS[A][phrasing_id].  This makes the paired-partner
    minimal-edit property crisp: flipping G swaps only the name; flipping A
    swaps only the age descriptor; flipping I swaps only the framing verb;
    flipping M swaps only the meeting clause.
    """
    name = RECIPIENT_NAMES[G][phrasing_id]
    age_desc = AGE_DESCRIPTORS[A][phrasing_id]
    # For A we bind the descriptor to the recipient's grammatical gender by
    # replacing 'his/her' with the correct pronoun.  This is a minimal edit
    # that piggy-backs on G, but it is unavoidable: not doing so introduces
    # a grammatical disagreement (his/her -> her/his).  We keep the
    # G-flip partner correctly gendered.
    if "his/her" in age_desc:
        pronoun = "his" if G == "male" else "her"
        age_desc = age_desc.replace("his/her", pronoun)

    instr = INSTRUCTION_FRAMES[I][phrasing_id].format(name=name)
    meeting = MEETING_FRAMES[M][phrasing_id].format(name=name)
    send_verb = "you give" if I == "give-frame" else "{name} keeps".format(name=name)
    answer_line = ANSWER_INSTRUCTION.format(send_verb=send_verb)

    prompt = (
        f"You are playing a one-shot dictator game with {name}, who is {age_desc}. "
        f"{instr} "
        f"{meeting} "
        f"{FAIR_SPLIT_REFERENCE} "
        f"{answer_line}"
    )
    return prompt


def token_length(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_baseline(rng, n_cells, per_cell):
    """Build the baseline design deterministically: 16 cells x per_cell trials."""
    variables = [("male", "female"), ("young", "old"),
                 ("give-frame", "take-frame"), ("meet", "no-meet")]
    cells = [(g, a, i, m)
             for g in variables[0]
             for a in variables[1]
             for i in variables[2]
             for m in variables[3]]
    assert len(cells) == n_cells
    trials = []
    for (g, a, i, m) in cells:
        for _ in range(per_cell):
            phrasing_id = rng.randrange(3)
            trials.append(dict(G=g, A=a, I=i, M=m, phrasing_id=phrasing_id))
    rng.shuffle(trials)
    return trials


def stratified_split(trials, split_ratio, rng):
    """Split by cell (G, A, I, M) so held-out has ~200 balanced trials."""
    # Group indices by cell
    by_cell = {}
    for idx, t in enumerate(trials):
        key = (t["G"], t["A"], t["I"], t["M"])
        by_cell.setdefault(key, []).append(idx)
    n_train_target = int(len(trials) * split_ratio)
    train_idx, held_idx = [], []
    for key, idx_list in by_cell.items():
        rng.shuffle(idx_list)
        n_cell_train = round(len(idx_list) * split_ratio)
        train_idx.extend(idx_list[:n_cell_train])
        held_idx.extend(idx_list[n_cell_train:])
    # Adjust rounding drift
    while len(train_idx) < n_train_target and held_idx:
        train_idx.append(held_idx.pop())
    while len(train_idx) > n_train_target:
        held_idx.append(train_idx.pop())
    return set(train_idx), set(held_idx)


def make_partner(base, V):
    """Return the (G, A, I, M) tuple after flipping V exactly."""
    opp = {
        "G": {"male": "female", "female": "male"},
        "A": {"young": "old", "old": "young"},
        "I": {"give-frame": "take-frame", "take-frame": "give-frame"},
        "M": {"meet": "no-meet", "no-meet": "meet"},
    }
    partner = dict(base)
    partner[V] = opp[V][base[V]]
    return partner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_trials", type=int, default=1000)
    ap.add_argument("--tokenizer_path", default="/data/zhenqian/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--max_length_delta", type=int, default=2)
    ap.add_argument("--split_ratio", type=float, default=0.8)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # -- Load tokenizer -------------------------------------------------------- #
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # -- 1) Build 16 cells x 63 = 1,008 baseline candidates -------------------- #
    per_cell = 63  # 16 * 63 = 1,008
    baseline_all = build_baseline(rng, n_cells=16, per_cell=per_cell)
    # Downsample to exactly n_trials (1,000) stratified over cells.
    # Group and drop uniformly.
    by_cell = {}
    for t in baseline_all:
        key = (t["G"], t["A"], t["I"], t["M"])
        by_cell.setdefault(key, []).append(t)
    keep_per_cell = args.n_trials // 16
    remainder = args.n_trials - keep_per_cell * 16
    baseline = []
    cell_keys = list(by_cell.keys())
    rng.shuffle(cell_keys)
    for k in cell_keys[:remainder]:
        baseline.extend(by_cell[k][: keep_per_cell + 1])
    for k in cell_keys[remainder:]:
        baseline.extend(by_cell[k][: keep_per_cell])
    rng.shuffle(baseline)
    assert len(baseline) == args.n_trials, (len(baseline), args.n_trials)

    # -- 2) Stratified train/held-out split ------------------------------------ #
    train_idx, held_idx = stratified_split(baseline, args.split_ratio, rng)

    # -- 3) Build 5,000 rows: 1,000 baseline + 4*1,000 minimal-edit partners --- #
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    rows = []
    n_retries = 0
    n_giveups = 0
    for i, base in enumerate(baseline):
        trial_id = f"T{i:04d}"
        split = "train" if i in train_idx else "held"
        base_prompt = render_prompt(base["G"], base["A"], base["I"], base["M"],
                                    base["phrasing_id"], rng)
        base_len = token_length(tokenizer, base_prompt)
        rows.append(dict(
            trial_id=trial_id,
            split=split,
            is_paired_partner_of=None,
            variable_flipped=None,
            G=base["G"], A=base["A"], I=base["I"], M=base["M"],
            phrasing_id=base["phrasing_id"],
            prompt=base_prompt,
            tok_len=base_len,
        ))
        for V in ["G", "A", "I", "M"]:
            partner = make_partner(base, V)
            # Try up to 3 phrasing_ids to satisfy the token-length constraint.
            best_prompt = None
            best_len = None
            for attempt in range(3):
                phr = base["phrasing_id"] if attempt == 0 else rng.randrange(3)
                partner_prompt = render_prompt(partner["G"], partner["A"],
                                               partner["I"], partner["M"],
                                               phr, rng)
                pl = token_length(tokenizer, partner_prompt)
                if best_len is None or abs(pl - base_len) < abs(best_len - base_len):
                    best_prompt, best_len, best_phr = partner_prompt, pl, phr
                if abs(pl - base_len) <= args.max_length_delta:
                    break
                n_retries += 1
            if best_len is None or abs(best_len - base_len) > args.max_length_delta:
                n_giveups += 1
            rows.append(dict(
                trial_id=f"{trial_id}_{V}",
                split=split,
                is_paired_partner_of=trial_id,
                variable_flipped=V,
                G=partner["G"], A=partner["A"], I=partner["I"], M=partner["M"],
                phrasing_id=best_phr,
                prompt=best_prompt,
                tok_len=best_len,
            ))

    # -- 4) Emit ---------------------------------------------------------------- #
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_baseline = sum(1 for r in rows if r["is_paired_partner_of"] is None)
    n_partner = sum(1 for r in rows if r["is_paired_partner_of"] is not None)
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_held = sum(1 for r in rows if r["split"] == "held")

    stats_path = args.out + ".stats.json"
    with open(stats_path, "w") as f:
        json.dump(dict(
            n_rows=len(rows),
            n_baseline=n_baseline,
            n_partner=n_partner,
            n_train=n_train,
            n_held=n_held,
            n_retries=n_retries,
            n_giveups=n_giveups,
            per_cell_baseline_count={
                str(k): len(v) for k, v in by_cell.items()
            },
        ), f, indent=2)

    print(f"[m1] wrote {args.out}: {len(rows)} rows "
          f"({n_baseline} baseline + {n_partner} partners); "
          f"{n_train} train / {n_held} held-out; "
          f"{n_giveups} length-match give-ups (soft target |Δtok|<={args.max_length_delta})")


if __name__ == "__main__":
    main()
