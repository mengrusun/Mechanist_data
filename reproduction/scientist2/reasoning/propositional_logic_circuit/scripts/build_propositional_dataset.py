#!/usr/bin/env python3
"""
Build synthetic propositional-logic dataset for circuit analysis.

Produces matched (clean, corrupt_*) pairs across cells parameterised by:
  - k (fact count): 2, 3, 5, 8, 12
  - chain-length (rules): 1, 2, 3
  - lexicon: natural, symbolic, alt-nouns

Corruption types (each isolates one causal role):
  - corrupt_fact:    swap one fact so the query no longer follows
  - corrupt_rule:    replace an implication rule so the chain doesn't resolve
  - corrupt_answer:  flip the answer-True/False through a template swap
                     (functionally: keep derivation, corrupt the answer cue)
  - corrupt_neutral: swap a distractor's name (answer unchanged) — sanity control

Anchor cell (k=3, chain=2, natural): 500 pairs each corruption type.
Additional cells: 200 pairs each.

Output layout:
  ${DATA_DIR}/prop_logic_synth/
    <corrupt>/split_k{K}_chain{C}_{LEX}/*.jsonl
    manifest.json  (counts per cell)
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


# ---------- Lexicons ----------

LEXICONS = {
    "natural": {
        "properties": [
            "wise", "hungry", "brave", "tired", "clever",
            "curious", "kind", "quiet", "strong", "gentle",
            "happy", "cautious", "patient", "cheerful", "generous",
        ],
        "names": [
            "Alice", "Bob", "Carol", "David", "Emma",
            "Frank", "Grace", "Henry", "Iris", "Jack",
            "Kate", "Leo", "Mia", "Noah", "Olivia",
            "Peter", "Quinn", "Ruth", "Sam", "Tina",
        ],
    },
    "symbolic": {
        "properties": [
            "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10",
            "Q1", "Q2", "Q3", "Q4", "Q5",
        ],
        "names": [
            "X1", "X2", "X3", "X4", "X5",
            "Y1", "Y2", "Y3", "Y4", "Y5",
            "Z1", "Z2", "Z3", "Z4", "Z5",
            "W1", "W2", "W3", "W4", "W5",
        ],
    },
    "alt-nouns": {
        "properties": [
            "sunny", "windy", "rainy", "cloudy", "foggy",
            "hot", "cold", "warm", "cool", "mild",
            "bright", "dim", "clear", "misty", "stormy",
        ],
        "names": [
            "Region_A", "Region_B", "Region_C", "Region_D", "Region_E",
            "Region_F", "Region_G", "Region_H", "Region_I", "Region_J",
            "Region_K", "Region_L", "Region_M", "Region_N", "Region_O",
            "Region_P", "Region_Q", "Region_R", "Region_S", "Region_T",
        ],
    },
}


# ---------- Deterministic derivation ----------

def _pick(rng: random.Random, seq: List[str], k: int) -> List[str]:
    return rng.sample(seq, k)


def _make_chain(rng: random.Random, props: List[str], chain_len: int) -> List[Tuple[str, str]]:
    """Return list of (antecedent, consequent) rules of length chain_len,
    forming a linear implication chain P0 -> P1 -> ... -> P_chain_len."""
    chosen = _pick(rng, props, chain_len + 1)
    return [(chosen[i], chosen[i + 1]) for i in range(chain_len)]


def _derive_answer(facts: Dict[str, str], rules: List[Tuple[str, str]], query_name: str, query_prop: str) -> bool:
    """Return whether facts + rules |= query."""
    # Start from the fact base — property assignments per name.
    known: Dict[str, set] = {name: {prop} for name, prop in facts.items()}
    # Forward-chain until fixed point.
    changed = True
    while changed:
        changed = False
        for name, props in list(known.items()):
            for ante, cons in rules:
                if ante in props and cons not in props:
                    known[name].add(cons)
                    changed = True
    return query_prop in known.get(query_name, set())


def _format_prompt(facts: Dict[str, str], rules: List[Tuple[str, str]], query_name: str, query_prop: str) -> str:
    fact_lines = "\n".join([f"- {n} is {p}." for n, p in facts.items()])
    rule_lines = "\n".join([f"- If X is {a} then X is {c}." for a, c in rules])
    prompt = (
        "You are given some facts and rules. Answer whether the query is True or False.\n\n"
        f"Facts:\n{fact_lines}\n\n"
        f"Rules:\n{rule_lines}\n\n"
        f"Query: Is {query_name} {query_prop}?\n\n"
        "Answer:"
    )
    return prompt


def _build_one_clean(rng: random.Random, k: int, chain_len: int, lex: str, target_answer: bool):
    """Build one clean prompt. Ensures answer matches target_answer."""
    props = LEXICONS[lex]["properties"]
    names = LEXICONS[lex]["names"]

    for _attempt in range(200):
        rules = _make_chain(rng, props, chain_len)
        starting_prop = rules[0][0]
        ending_prop = rules[-1][1]

        # Pick k names.
        chosen_names = _pick(rng, names, k)
        # One "subject" name gets the starting_prop (so chain resolves).
        # Others get random OFF-CHAIN properties (distractors).
        subject = chosen_names[0]
        off_chain_props = [p for p in props if p not in {a for a, _ in rules} | {c for _, c in rules}]
        if len(off_chain_props) < k - 1:
            continue  # not enough distractor properties for this lexicon; retry
        distractors_props = _pick(rng, off_chain_props, k - 1)

        facts = {subject: starting_prop}
        for i, dn in enumerate(chosen_names[1:]):
            facts[dn] = distractors_props[i]

        # Query for target_answer:
        # If True → ask about the subject with ending_prop (chain resolves).
        # If False → ask about the subject with an OFF-CHAIN property (chain does not entail).
        if target_answer:
            query_name = subject
            query_prop = ending_prop
        else:
            # Pick a property that the subject does NOT get via the chain.
            candidates = [p for p in off_chain_props if p != facts[subject]]
            if not candidates:
                continue
            query_name = subject
            query_prop = rng.choice(candidates)

        actual = _derive_answer(facts, rules, query_name, query_prop)
        if actual == target_answer:
            return facts, rules, query_name, query_prop, actual

    raise RuntimeError(f"Could not construct clean prompt for k={k}, chain={chain_len}, lex={lex}, target={target_answer}")


# ---------- Corruption strategies ----------

def _corrupt_fact(rng: random.Random, facts: Dict[str, str], rules: List[Tuple[str, str]],
                  query_name: str, query_prop: str, lex: str):
    """Change the SUBJECT's fact so the chain doesn't fire — answer should flip."""
    props = LEXICONS[lex]["properties"]
    starting_prop = rules[0][0]
    # Force subject's property to something outside the chain.
    off_chain = [p for p in props if p not in {a for a, _ in rules} | {c for _, c in rules}]
    off_chain = [p for p in off_chain if p != facts[query_name]]
    if not off_chain:
        return None
    new_facts = dict(facts)
    new_facts[query_name] = rng.choice(off_chain)
    return new_facts, rules, query_name, query_prop


def _corrupt_rule(rng: random.Random, facts: Dict[str, str], rules: List[Tuple[str, str]],
                  query_name: str, query_prop: str, lex: str):
    """Replace ONE rule's consequent so the chain breaks — answer should flip when originally True."""
    props = LEXICONS[lex]["properties"]
    if len(rules) == 0:
        return None
    idx = rng.randint(0, len(rules) - 1)
    ante, cons = rules[idx]
    # New consequent NOT in the chain (breaks the link).
    used = {a for a, _ in rules} | {c for _, c in rules}
    candidates = [p for p in props if p not in used and p != facts[query_name]]
    if not candidates:
        return None
    new_cons = rng.choice(candidates)
    new_rules = list(rules)
    new_rules[idx] = (ante, new_cons)
    return facts, new_rules, query_name, query_prop


def _corrupt_answer(rng: random.Random, facts: Dict[str, str], rules: List[Tuple[str, str]],
                    query_name: str, query_prop: str, lex: str):
    """
    Corrupt the answer cue while keeping the derivation identical.
    Strategy: change the query_prop to the OPPOSITE truth-value's target while keeping
    facts + rules identical. This isolates the projection of derived truth-value
    into the answer token position — a rule-chain valid derivation now targets a
    different query, so the answer flips even though derivation is the same.
    """
    props = LEXICONS[lex]["properties"]
    used = {a for a, _ in rules} | {c for _, c in rules}
    # Original was True iff query_prop == last_consequent. Flip: pick a distractor prop.
    ending_prop = rules[-1][1]
    if query_prop == ending_prop:
        # Was True; corrupt to a False query (off-chain prop).
        candidates = [p for p in props if p not in used and p != facts[query_name]]
    else:
        # Was False; corrupt to a True query (ending_prop).
        candidates = [ending_prop]
    if not candidates:
        return None
    new_prop = rng.choice(candidates)
    return facts, rules, query_name, new_prop


def _corrupt_neutral(rng: random.Random, facts: Dict[str, str], rules: List[Tuple[str, str]],
                     query_name: str, query_prop: str, lex: str):
    """
    Change a DISTRACTOR name so the answer does NOT change.
    Used as a matched-control corruption: any effect on the model should be small.
    """
    names = LEXICONS[lex]["names"]
    distractor_names = [n for n in facts.keys() if n != query_name]
    if not distractor_names:
        return None
    old = rng.choice(distractor_names)
    used_names = set(facts.keys())
    candidates = [n for n in names if n not in used_names]
    if not candidates:
        return None
    new = rng.choice(candidates)
    new_facts = {(new if k == old else k): v for k, v in facts.items()}
    return new_facts, rules, query_name, query_prop


CORRUPTIONS = {
    "corrupt_fact": _corrupt_fact,
    "corrupt_rule": _corrupt_rule,
    "corrupt_answer": _corrupt_answer,
    "corrupt_neutral": _corrupt_neutral,
}


# ---------- Cell building ----------

def build_cell(out_dir: Path, k: int, chain_len: int, lex: str, n_pairs: int, seed: int, manifest: Dict):
    """Build one cell: n_pairs of (clean, corrupt_*) pairs for each of the four corruption types."""
    rng = random.Random(seed)
    cell_name = f"split_k{k}_chain{chain_len}_{lex}"
    subdirs = {}
    for corruption in ["clean"] + list(CORRUPTIONS.keys()):
        subdir = out_dir / corruption / cell_name
        subdir.mkdir(parents=True, exist_ok=True)
        subdirs[corruption] = subdir

    # Enforce 50/50 True/False balance.
    half = n_pairs // 2
    balance = [True] * half + [False] * (n_pairs - half)
    rng.shuffle(balance)

    pairs_written = {c: 0 for c in ["clean"] + list(CORRUPTIONS.keys())}

    clean_file = open(subdirs["clean"] / "data.jsonl", "w")
    corrupt_files = {c: open(subdirs[c] / "data.jsonl", "w") for c in CORRUPTIONS}

    try:
        for pair_idx, target_ans in enumerate(balance):
            # Build the clean example.
            facts, rules, qname, qprop, actual = _build_one_clean(rng, k, chain_len, lex, target_ans)
            assert actual == target_ans
            clean_prompt = _format_prompt(facts, rules, qname, qprop)
            clean_record = {
                "pair_id": pair_idx,
                "cell": cell_name,
                "k": k,
                "chain": chain_len,
                "lex": lex,
                "prompt": clean_prompt,
                "answer": "True" if actual else "False",
                "facts": [[n, p] for n, p in facts.items()],
                "rules": [[a, c] for a, c in rules],
                "query": [qname, qprop],
            }
            clean_file.write(json.dumps(clean_record) + "\n")
            pairs_written["clean"] += 1

            # Build each corruption. Retry if a corruption yields no answer flip
            # where flipping was expected (fact / rule / answer). For neutral,
            # answer must be UNCHANGED.
            for corr_name, corr_fn in CORRUPTIONS.items():
                got = None
                for _attempt in range(20):
                    result = corr_fn(rng, facts, rules, qname, qprop, lex)
                    if result is None:
                        continue
                    c_facts, c_rules, c_qname, c_qprop = result
                    c_actual = _derive_answer(c_facts, c_rules, c_qname, c_qprop)
                    if corr_name == "corrupt_neutral":
                        # Must be unchanged.
                        if c_actual == actual:
                            got = (c_facts, c_rules, c_qname, c_qprop, c_actual)
                            break
                    else:
                        # Must flip.
                        if c_actual != actual:
                            got = (c_facts, c_rules, c_qname, c_qprop, c_actual)
                            break
                if got is None:
                    # Fallback: emit with whatever we got last (should be rare).
                    if result is not None:
                        c_facts, c_rules, c_qname, c_qprop = result
                        c_actual = _derive_answer(c_facts, c_rules, c_qname, c_qprop)
                        got = (c_facts, c_rules, c_qname, c_qprop, c_actual)
                    else:
                        continue
                c_facts, c_rules, c_qname, c_qprop, c_actual = got
                c_prompt = _format_prompt(c_facts, c_rules, c_qname, c_qprop)
                c_record = {
                    "pair_id": pair_idx,
                    "cell": cell_name,
                    "k": k,
                    "chain": chain_len,
                    "lex": lex,
                    "prompt": c_prompt,
                    "answer": "True" if c_actual else "False",
                    "clean_answer": "True" if actual else "False",
                    "facts": [[n, p] for n, p in c_facts.items()],
                    "rules": [[a, c] for a, c in c_rules],
                    "query": [c_qname, c_qprop],
                    "corruption": corr_name,
                }
                corrupt_files[corr_name].write(json.dumps(c_record) + "\n")
                pairs_written[corr_name] += 1
    finally:
        clean_file.close()
        for f in corrupt_files.values():
            f.close()

    manifest.setdefault("cells", {})[cell_name] = pairs_written
    print(f"[dataset] built {cell_name}: {pairs_written}")


# ---------- Resample pool (unrelated prompts for resample ablation) ----------

def build_resample_pool(out_dir: Path, n_pool: int, seed: int, manifest: Dict):
    """Build a pool of miscellaneous prompts for resample ablation.

    We use random (k=4, chain=2, natural) prompts with random truth-value —
    matched in structure to the anchor but statistically independent."""
    rng = random.Random(seed + 999_999)
    pool_dir = out_dir / "pool_resample"
    pool_dir.mkdir(parents=True, exist_ok=True)
    with open(pool_dir / "data.jsonl", "w") as fh:
        for i in range(n_pool):
            target = bool(rng.getrandbits(1))
            facts, rules, qname, qprop, actual = _build_one_clean(rng, 4, 2, "natural", target)
            fh.write(json.dumps({
                "pair_id": i,
                "prompt": _format_prompt(facts, rules, qname, qprop),
                "answer": "True" if actual else "False",
            }) + "\n")
    manifest["resample_pool"] = n_pool
    print(f"[dataset] built resample pool: {n_pool} prompts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Output data directory (created if missing)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"seed": args.seed}

    # Anchor cell (500 pairs) + role cells (300 pairs), plus stability + surface variants (200 pairs).
    # Per plan §M0.dataset:
    #   anchor  = k=3, chain=2, natural, 500 pairs (each corruption type)
    #   role    = same cell, 300 pairs (already covered by anchor 500 subset)
    #   stab    = (k=5, chain=2, natural) + (k=3, chain=3, natural), 200 pairs
    #   surface = (k=3, chain=2, symbolic) + (k=3, chain=2, alt-nouns), 200 pairs
    cells = [
        (3, 2, "natural",  500),
        (5, 2, "natural",  200),
        (3, 3, "natural",  200),
        (3, 2, "symbolic", 200),
        (3, 2, "alt-nouns", 200),
    ]
    for i, (k, cl, lex, n) in enumerate(cells):
        build_cell(out_dir, k, cl, lex, n, seed=args.seed + i, manifest=manifest)

    # Resample pool for ablation.
    build_resample_pool(out_dir, n_pool=800, seed=args.seed, manifest=manifest)

    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[dataset] wrote manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
