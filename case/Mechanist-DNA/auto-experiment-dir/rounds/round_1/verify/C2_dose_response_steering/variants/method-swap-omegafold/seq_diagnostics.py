"""
Verify variant (C2) -- reviewer-requested sequence-degeneration diagnostics.
Computes, per saved generated protein (from gen_sequences.py's JSONL), cheap composition/complexity
metrics that don't require any structure predictor: Shannon entropy of amino-acid composition,
alphabet size (distinct residues used), max homopolymer run length, and a simple repeat-content
proxy (fraction of the sequence covered by any length-3..6 substring that repeats >=3 times).
Aggregated per (alpha, seed) to check whether high-alpha sequences are compositionally degenerate
(low entropy / small alphabet / long homopolymer runs) independent of what any folder concludes.

Run (any env with just python+numpy, no GPU needed): python seq_diagnostics.py --in seqs_a8_s42.jsonl --out diag_a8_s42.json
"""
import json, argparse, math
from collections import Counter
import numpy as np


def shannon_entropy(seq):
    if not seq:
        return 0.0
    cnt = Counter(seq)
    n = len(seq)
    return -sum((c / n) * math.log2(c / n) for c in cnt.values())


def max_homopolymer_run(seq):
    if not seq:
        return 0
    best = cur = 1
    for i in range(1, len(seq)):
        cur = cur + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, cur)
    return best


def repeat_fraction(seq, k=4, min_count=3):
    if len(seq) < k:
        return 0.0
    kmers = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
    covered = set()
    for i in range(len(seq) - k + 1):
        km = seq[i:i + k]
        if kmers[km] >= min_count:
            covered.update(range(i, i + k))
    return len(covered) / len(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.inp)]
    valid = [r for r in recs if r.get("valid_orf") and r.get("prot")]

    per_seq = []
    for r in valid:
        p = r["prot"]
        per_seq.append({
            "idx": r["idx"], "len_aa": len(p),
            "entropy": shannon_entropy(p), "alphabet_size": len(set(p)),
            "max_homopolymer_run": max_homopolymer_run(p),
            "repeat_fraction_k4": repeat_fraction(p, k=4),
        })

    def agg(key):
        vals = [x[key] for x in per_seq]
        return {"mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if vals else None}

    result = {
        "n_sequences": len(per_seq),
        "entropy": agg("entropy"), "alphabet_size": agg("alphabet_size"),
        "max_homopolymer_run": agg("max_homopolymer_run"),
        "repeat_fraction_k4": agg("repeat_fraction_k4"),
        "per_sequence": per_seq,
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[seq_diag] {args.inp}: n={len(per_seq)} entropy_mean={result['entropy']['mean']:.3f} "
          f"alphabet_mean={result['alphabet_size']['mean']:.1f} "
          f"max_homopolymer_mean={result['max_homopolymer_run']['mean']:.1f} "
          f"repeat_frac_mean={result['repeat_fraction_k4']['mean']:.3f}")


if __name__ == "__main__":
    main()
