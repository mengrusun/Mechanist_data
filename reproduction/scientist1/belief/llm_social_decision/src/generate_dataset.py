"""Generate 1000 dictator-game trials + paired prompts for direction extraction.

Variables randomised per trial:
  G (gender):        {'male', 'female'}
  A (age):           {'young', 'old'}  -> young ~ 22, old ~ 65
  I (instruction):   {'A' = neutral 'You may transfer', 'B' = prosocial 'It is fair to share'}
  M (meeting):       {'meeting', 'no_meeting'}

Fixed: endowment = $20, fair-split reference = $10.
"""

import json, random, os
from pathlib import Path

random.seed(0)

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENDERS = ["male", "female"]
AGES = ["young", "old"]
AGE_TEXT = {"young": "a 22-year-old", "old": "a 65-year-old"}
GENDER_TEXT = {"male": "man", "female": "woman"}
INSTRUCTIONS = ["A", "B"]
MEETINGS = ["meeting", "no_meeting"]

INSTR_TEXT = {
    "A": ("You may transfer any whole-dollar amount from $0 to $20 to the recipient. "
          "You keep the remaining amount."),
    "B": ("Please consider what would be fair. You may transfer any whole-dollar amount "
          "from $0 to $20 to the recipient. It is often considered fair to share."),
}

MEETING_TEXT = {
    "meeting": ("Before deciding, you and the recipient met briefly in person and had a short conversation."),
    "no_meeting": ("You and the recipient have never met and will remain anonymous to each other."),
}


def make_prompt(gender, age, instruction, meeting):
    person = f"{AGE_TEXT[age]} {GENDER_TEXT[gender]}"
    prompt = (
        f"You are participating in a dictator game as the dictator. "
        f"You are {person}. You have been given $20 to allocate between yourself and an anonymous recipient. "
        f"{MEETING_TEXT[meeting]} "
        f"{INSTR_TEXT[instruction]} "
        f"How many dollars will you transfer to the recipient? "
        f"Answer with a single integer between 0 and 20.\nTransfer amount: $"
    )
    return prompt


def main():
    N = 1000
    trials = []
    for i in range(N):
        g = random.choice(GENDERS)
        a = random.choice(AGES)
        ins = random.choice(INSTRUCTIONS)
        m = random.choice(MEETINGS)
        trials.append({
            "id": i,
            "gender": g,
            "age": a,
            "instruction": ins,
            "meeting": m,
            "prompt": make_prompt(g, a, ins, m),
        })
    (OUT_DIR / "trials.jsonl").write_text("\n".join(json.dumps(t) for t in trials))
    print(f"Wrote {N} trials to {OUT_DIR / 'trials.jsonl'}")

    # Paired prompts for direction extraction.
    # For each variable, we take K paired prompts where only that variable differs,
    # keeping the other three variables randomised (so pooled mean-diff averages them out).
    K = 400  # paired prompts per variable side (600 pairs total but capped at 400 for speed)
    pairs = {}
    def rand_others(exclude):
        cfg = {
            "gender": random.choice(GENDERS),
            "age": random.choice(AGES),
            "instruction": random.choice(INSTRUCTIONS),
            "meeting": random.choice(MEETINGS),
        }
        return cfg

    for var, vals in [("gender", GENDERS), ("age", AGES), ("instruction", INSTRUCTIONS), ("meeting", MEETINGS)]:
        var_pairs = []
        for k in range(K):
            cfg = rand_others(var)
            cfg_a = dict(cfg); cfg_a[var] = vals[0]
            cfg_b = dict(cfg); cfg_b[var] = vals[1]
            var_pairs.append({
                "id": k,
                "variable": var,
                "value_a": vals[0],
                "value_b": vals[1],
                "context": cfg,
                "prompt_a": make_prompt(**cfg_a),
                "prompt_b": make_prompt(**cfg_b),
            })
        pairs[var] = var_pairs

    with open(OUT_DIR / "pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Wrote paired prompts to {OUT_DIR / 'pairs.json'} ({sum(len(v) for v in pairs.values())} pairs)")


if __name__ == "__main__":
    main()
