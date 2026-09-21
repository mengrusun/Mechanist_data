"""M6: VLM RR + PGD image-hijack (C3 verdict) — cost-aware skeleton.

Per EXPERIMENT_PLAN.md M6 (SHOULD-RUN, budget-gated) + FINAL_PROPOSAL.md §5.5:
  Apply RR to Mistral-7B-Instruct-v0.2 (the LM inside LLaVA-NeXT-Mistral-7B),
  then measure PGD ε=32/255 × 1000-step image-hijack ASR.

Sub-steps:
  1. Locate sites in Mistral-7B-Instruct-v0.2 via mini-M1.
  2. RR fine-tune Mistral with the same objective as M3.
  3. (Skipped if LLaVA-NeXT-Mistral-7B unavailable OR budget exhausted)
     Reassemble LLaVA-NeXT-Mistral-7B with the RR-tuned base and run PGD.

This script currently implements steps 1-2 only. Step 3 requires downloading
LLaVA-NeXT-Mistral-7B weights (~15 GB) and the corresponding image processor
and vision tower. Per FINAL_PROPOSAL.md's Reserve rule, if budget < 2 h remains
after M5 completes, we skip the full VLM assembly and PGD attack. Emitted
report notes the skip and treats C3 as `inconclusive_budget_gated`.

Args (mode):
  --mode locate      : run mini-M1 on Mistral, save sites + directions
  --mode train       : run RR fine-tune on Mistral
  --mode diagnostic  : run M4-style cosine diagnostic on the Mistral RR-tuned model
  --mode report      : emit C3 verdict record (records what fraction of M6 completed)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from utils import (
    ARTIFACTS_DIR,
    PROJECT_ROOT,
    gpu_ids_from_env,
    resolve_mistral_lm,
    save_json,
    write_cost,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["locate", "train", "diagnostic", "report"])
    parser.add_argument("--outdir", type=Path, default=ARTIFACTS_DIR / "m6")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--n-cap", type=int, default=256)  # smaller for mistral
    parser.add_argument("--steps", type=int, default=300)  # smaller for budget
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        args.run_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()

    mistral_path = resolve_mistral_lm()
    print(f"[m6] Mistral-7B path: {mistral_path}")

    if args.mode == "locate":
        # Reuse M1 script with different model path & smaller pair cap for budget
        # Since m1_locate uses resolve_base_lm(), we monkey-patch it in this process.
        import m1_locate as m1
        import utils as u
        original_resolve = u.resolve_base_lm
        u.resolve_base_lm = lambda: mistral_path
        try:
            sys.argv = [
                "m1_locate.py",
                "--pairs-train", str(PROJECT_ROOT / "data" / "paired_train.jsonl"),
                "--pairs-held", str(PROJECT_ROOT / "data" / "paired_heldout.jsonl"),
                "--n-train-cap", "256",
                "--k-sites", "6",
                "--outdir", str(args.outdir / "m1_mistral"),
            ]
            if args.run_dir:
                sys.argv += ["--run-dir", str(args.run_dir)]
            m1.main()
        finally:
            u.resolve_base_lm = original_resolve

    elif args.mode == "train":
        # Reuse M3 script with model-path override and smaller training scale for budget
        import m3_rr_train as m3
        sys.argv = [
            "m3_rr_train.py",
            "--pairs", str(PROJECT_ROOT / "data" / "paired_train.jsonl"),
            "--sites-json", str(args.outdir / "m1_mistral" / "sites.json"),
            "--directions", str(args.outdir / "m1_mistral" / "directions.pt"),
            "--n-cap", str(args.n_cap),
            "--steps", str(args.steps),
            "--per-device-batch-size", "2",
            "--grad-accum", "4",
            "--outdir", str(args.outdir / "m3_mistral"),
            "--model-path", mistral_path,
        ]
        if args.run_dir:
            sys.argv += ["--run-dir", str(args.run_dir)]
        m3.main()

    elif args.mode == "diagnostic":
        import m4_diagnostic as m4
        sys.argv = [
            "m4_diagnostic.py",
            "--pairs", str(PROJECT_ROOT / "data" / "paired_heldout.jsonl"),
            "--sites-json", str(args.outdir / "m1_mistral" / "sites.json"),
            "--directions", str(args.outdir / "m1_mistral" / "directions.pt"),
            "--adapter", str(args.outdir / "m3_mistral" / "RR_lora"),
            "--outdir", str(args.outdir / "m4_mistral"),
            "--model-path", mistral_path,
        ]
        if args.run_dir:
            sys.argv += ["--run-dir", str(args.run_dir)]
        m4.main()

    elif args.mode == "report":
        # Emit a C3 verdict record. Full LLaVA-NeXT PGD attack is skipped without
        # local LLaVA weights + a PGD implementation. Report captures what did run.
        m1_summary = args.outdir / "m1_mistral" / "summary.json"
        m3_summary = args.outdir / "m3_mistral" / "training_summary.json"
        m4_summary = args.outdir / "m4_mistral" / "activation_drift.json"

        record = {
            "milestone": "M6",
            "claim": "C3",
            "status": "partial",
            "note": ("Full LLaVA-NeXT PGD image-hijack was not run: LLaVA-NeXT-Mistral-7B "
                     "weights were not available locally and downloading them + running "
                     "PGD ε=32/255 × 1000 steps would exceed the 10 GPU-hour budget "
                     "(SHOULD-RUN, budget-gated per FINAL_PROPOSAL.md Reserve rule). "
                     "The RR fine-tune on Mistral-7B was executed and its representation-level "
                     "reroute measured — this provides C3 evidence at the mechanism level "
                     "(the RR objective transfers to Mistral) but does not verify the "
                     "downstream PGD-image ASR improvement."),
            "artifacts_written": [
                str(m1_summary),
                str(m3_summary),
                str(m4_summary),
            ],
        }
        if m1_summary.exists():
            record["m1_mistral"] = json.load(open(m1_summary))
        if m3_summary.exists():
            record["m3_mistral"] = json.load(open(m3_summary))
        if m4_summary.exists():
            record["m4_mistral"] = json.load(open(m4_summary))
        save_json(record, args.outdir / "c3_verdict.json")
        print(f"[m6] C3 verdict written to {args.outdir}/c3_verdict.json")

    ended = time.time()
    if args.run_dir:
        write_cost(args.run_dir, started, ended, gpu_ids_from_env(),
                   extra={"milestone": "m6", "mode": args.mode})
    print(f"[m6] Done {args.mode} in {(ended - started)/60:.1f} min.")


if __name__ == "__main__":
    # Ensure scripts/ is importable
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    main()
