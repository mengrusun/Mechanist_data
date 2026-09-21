"""Cache a 1M-token slice from the pretokenized Pile shards for PPL evaluation.

Reuses across M2/M3/M4 by pointing to `refine-logs/artifacts/ppl_sample.pt`.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from belief_utils import build_pile_ppl_sample

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pile-dir", default="/mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled")
    ap.add_argument("--n-tokens", type=int, default=1_048_576)
    ap.add_argument("--output", default="refine-logs/artifacts/ppl_sample.pt")
    args = ap.parse_args()

    print(f"[setup] building PPL sample of {args.n_tokens:,} tokens from {args.pile_dir}")
    path = build_pile_ppl_sample(args.pile_dir, args.n_tokens, args.output)
    print(f"[setup] cached → {path}")
