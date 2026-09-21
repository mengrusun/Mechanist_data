"""Generate templated propositional-logic prompts.

Design:
    Each example has k facts and a single rule of the form
        "if <P> is true then <Q> is <X>",
    where X ∈ {true, false}. The correct answer to the query "<Q> is"
    is X (the model must (1) identify that the antecedent fact is true,
    (2) extract the consequent polarity, (3) project it into the answer).

Clean vs corrupted differ by a single-token flip of X (the rule's
consequent polarity). This holds all other surface features constant
and reliably flips the correct answer, giving a strong logit-diff signal.

Additional variants for the modularity analysis:
    fact_flip  - flip the antecedent fact truth token (breaks the modus-
                 ponens firing condition -> answer becomes "false" for
                 either polarity of X)
    query_flip - point the query at a proposition whose truth is NOT
                 derivable from the rule (a distractor)  -> answer
                 becomes "false" (defaulting to unknown)

Balanced so that 50 % of clean answers are "true" and 50 % "false".
"""
import argparse
import json
import random
from pathlib import Path


PROPS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
    "eta", "theta", "iota", "kappa", "lambda", "mu",
    "nu", "xi", "omicron", "pi", "rho", "sigma",
    "tau", "upsilon", "phi", "chi", "psi", "omega",
]


def build_prompt(fact_lines, rule_ante, rule_conc, rule_conc_polarity, query_prop):
    facts = ". ".join(fact_lines) + "."
    rule = f"if {rule_ante} is true then {rule_conc} is {rule_conc_polarity}"
    return f"Facts: {facts} Rule: {rule}. Question: {query_prop} is"


def make_example(rng, k_distractors: int = 1):
    props = rng.sample(PROPS, k=2 + k_distractors)
    ante, cons, *distractors = props

    # Balance clean-answer polarity: 50% true / 50% false
    clean_polarity = rng.choice(["true", "false"])
    corrupt_polarity = "false" if clean_polarity == "true" else "true"

    fact_lines = [f"{ante} is true"]
    for d in distractors:
        fact_lines.append(f"{d} is {rng.choice(['true', 'false'])}")

    clean = build_prompt(fact_lines, ante, cons, clean_polarity, cons)
    corrupt = build_prompt(fact_lines, ante, cons, corrupt_polarity, cons)

    # fact_flip: antecedent fact becomes false (denied antecedent → answer flips
    # to "false" regardless of consequent polarity in the rule)
    fact_flip_lines = [f"{ante} is false"] + fact_lines[1:]
    fact_flip = build_prompt(fact_flip_lines, ante, cons, clean_polarity, cons)
    fact_flip_answer = "false"

    # query_flip: same facts & rule, but query a distractor -> answer "false"
    if distractors:
        query_flip = build_prompt(fact_lines, ante, cons, clean_polarity, distractors[0])
    else:
        query_flip = clean  # degenerate
    query_flip_answer = "false"

    return {
        "clean": clean,
        "corrupt": corrupt,  # rule-consequent flip
        "fact_flip": fact_flip,
        "query_flip": query_flip,
        "clean_answer": clean_polarity,
        "corrupt_answer": corrupt_polarity,
        "fact_flip_answer": fact_flip_answer,
        "query_flip_answer": query_flip_answer,
        "antecedent": ante,
        "consequent": cons,
        "distractors": distractors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k-distractors", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ds = [make_example(rng, args.k_distractors) for _ in range(args.n)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")

    n_true = sum(1 for r in ds if r["clean_answer"] == "true")
    print(f"wrote {len(ds)} to {out} ({n_true} clean=true, {len(ds)-n_true} clean=false)")
    for row in ds[:4]:
        print("CLEAN:", row["clean"], "->", row["clean_answer"])
        print("CORR :", row["corrupt"], "->", row["corrupt_answer"])
        print("FACT-:", row["fact_flip"], "->", row["fact_flip_answer"])
        print("QRY -:", row["query_flip"], "->", row["query_flip_answer"])
        print()


if __name__ == "__main__":
    main()
