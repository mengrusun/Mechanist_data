#!/usr/bin/env python3
"""
Run M3, M4, M5, and M6(a,b,c,d) for a single seed IN ONE PYTHON SESSION so
gemma-3-27b-pt is loaded once (~2.5 min) instead of 8 times.

This driver loads the model + tokenizer once, then invokes each milestone's
runner function directly (bypassing subprocess/model-reload overhead).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vc_common import MODEL_PATH, load_model_and_tokenizer, set_all_seeds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--m1_cache", required=True)
    p.add_argument("--m2_dir", required=True)
    p.add_argument("--m3_dir", required=True)
    p.add_argument("--m4_dir", required=True)
    p.add_argument("--m5_dir", required=True)
    p.add_argument("--m6_dir", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--n_pairs_m3", type=int, default=200)
    p.add_argument("--n_items_m4", type=int, default=300)
    p.add_argument("--n_items_m5", type=int, default=300)
    p.add_argument("--n_items_m6a", type=int, default=200)
    p.add_argument("--n_pairs_m6c", type=int, default=150)
    p.add_argument("--n_items_m6d", type=int, default=200)
    p.add_argument("--skip", default="",
                   help="Comma-list of steps to skip: m3,m4,m5,m6a,m6b,m6c,m6d")
    return p.parse_args()


def _fake_argv(kwargs_list):
    """Monkey-patch sys.argv so a script's argparse.parse_args() will consume kwargs_list."""
    old = sys.argv[:]
    sys.argv = ["driver"] + list(kwargs_list)
    return old


def _restore_argv(old):
    sys.argv = old


def _run_milestone(module_name, argv_list, shared_model=None, shared_tok=None):
    """
    Import the milestone module and call its main(), with a preloaded model
    injected via monkey-patching load_model_and_tokenizer.
    """
    import importlib
    mod = importlib.import_module(module_name)
    original_loader = None
    if shared_model is not None:
        # Monkey-patch vc_common.load_model_and_tokenizer to return the shared instance
        import vc_common
        original_loader = vc_common.load_model_and_tokenizer
        vc_common.load_model_and_tokenizer = lambda *a, **kw: (shared_model, shared_tok)
        # Also monkey-patch in the module's own namespace if it imported the symbol
        if hasattr(mod, "load_model_and_tokenizer"):
            mod._original_loader = mod.load_model_and_tokenizer
            mod.load_model_and_tokenizer = lambda *a, **kw: (shared_model, shared_tok)
    old_argv = _fake_argv(argv_list)
    try:
        mod.main()
    finally:
        _restore_argv(old_argv)
        if original_loader is not None:
            vc_common.load_model_and_tokenizer = original_loader
            if hasattr(mod, "_original_loader"):
                mod.load_model_and_tokenizer = mod._original_loader


def main():
    args = parse_args()
    skip = set(x.strip() for x in args.skip.split(",") if x.strip())

    Path(args.m3_dir).mkdir(parents=True, exist_ok=True)
    Path(args.m4_dir).mkdir(parents=True, exist_ok=True)
    Path(args.m5_dir).mkdir(parents=True, exist_ok=True)
    Path(args.m6_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    print(f"[downstream seed={args.seed}] Loading model once...", flush=True)
    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype=dtype)
    print(f"[downstream seed={args.seed}] Model loaded in {(time.time()-t0)/60:.1f}m",
          flush=True)

    # ----------- M3: patching -------------------------------------------
    if "m3" not in skip:
        print(f"[downstream seed={args.seed}] === M3 patching ===", flush=True)
        _run_milestone("m3_patch", [
            "--model", MODEL_PATH,
            "--m1_cache", args.m1_cache,
            "--m2_dir", args.m2_dir,
            "--site", "AUTO",
            "--n_pairs", str(args.n_pairs_m3),
            "--seed", str(args.seed),
            "--out", args.m3_dir,
            "--dtype", args.dtype,
        ], shared_model=model, shared_tok=tok)

    # ----------- M4: attention block ------------------------------------
    if "m4" not in skip:
        print(f"[downstream seed={args.seed}] === M4 attention-block ===", flush=True)
        _run_milestone("m4_attn_block", [
            "--model", MODEL_PATH,
            "--m1_cache", args.m1_cache,
            "--m2_dir", args.m2_dir,
            "--n_items", str(args.n_items_m4),
            "--block_from", "AUTO",
            "--block_to", "C0",
            "--block_at_layer", "top,last",
            "--seed", str(args.seed),
            "--out", os.path.join(args.m4_dir, f"m4_seed{args.seed}.json"),
            "--dtype", args.dtype,
        ], shared_model=model, shared_tok=tok)

    # ----------- M5: steering (2 methods) -------------------------------
    if "m5" not in skip:
        for dm in ("diff_of_means", "lda"):
            print(f"[downstream seed={args.seed}] === M5 steering direction_method={dm} ===",
                  flush=True)
            _run_milestone("m5_steer", [
                "--model", MODEL_PATH,
                "--m1_cache", args.m1_cache,
                "--m2_dir", args.m2_dir,
                "--n_items", str(args.n_items_m5),
                "--site", "AUTO",
                "--direction_method", dm,
                "--alphas=-4,-2,-1,0,1,2,4",  # use `=` to avoid argparse mistaking negative as flag
                "--seed", str(args.seed),
                "--out", os.path.join(args.m5_dir, f"m5_{dm}_seed{args.seed}.json"),
                "--dtype", args.dtype,
            ], shared_model=model, shared_tok=tok)

    # ----------- M6 sub-experiments ------------------------------------
    for sub, out_suffix, extra in (
        ("a", f"a_seed{args.seed}.json",
         ["--n_items", str(args.n_items_m6a), "--n_pairs", "200",
          "--m3_dir", args.m3_dir]),
        ("b", f"b_seed{args.seed}.json", []),
        ("c", f"c_seed{args.seed}.json", ["--n_pairs", str(args.n_pairs_m6c)]),
        ("d", f"d_seed{args.seed}.json", ["--n_items", str(args.n_items_m6d)]),
    ):
        if f"m6{sub}" in skip:
            continue
        print(f"[downstream seed={args.seed}] === M6({sub}) ===", flush=True)
        _run_milestone("m6_controls", [
            "--model", MODEL_PATH,
            "--m1_cache", args.m1_cache,
            "--m2_dir", args.m2_dir,
            "--sub", sub,
            "--seed", str(args.seed),
            "--out", os.path.join(args.m6_dir, out_suffix),
            "--dtype", args.dtype,
        ] + extra, shared_model=model, shared_tok=tok)

    print(f"[downstream seed={args.seed}] ALL DONE elapsed={(time.time()-t0)/60:.1f}m",
          flush=True)


if __name__ == "__main__":
    main()
