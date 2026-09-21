"""
Calibration step (code-review-mandated, sanity_first): pick OmegaFold's --conf_min gate so its
retention rate on a pilot batch matches the main experiment's ESMFold pLDDT>=50 retention rate at
alpha=0 (main experiment: gated_pass_rate = 0.575, from results/m1_calibration.json /
results/m2_a0_s42.json), rather than guessing a threshold on an unknown confidence scale.

Run (verify_omegafold env), AFTER a pilot seqs_a0_pilot.jsonl has been generated (scientist env)
and folded once with --conf_min 0.0 (i.e. no gating, so per_sample.conf is populated for every
folded structure):
    python fold_and_readout_omegafold.py --in seqs_a0_pilot.jsonl --out pilot_nogate.json --conf_min 0.0
    python calibrate_conf_gate.py --in pilot_nogate.json --target_rate 0.575
"""
import json, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="pilot fold result with conf_min=0.0 (no gating)")
    ap.add_argument("--target_rate", type=float, default=0.575,
                    help="main experiment's ESMFold pLDDT>=50 gated_pass_rate at alpha=0 (baseline)")
    args = ap.parse_args()

    d = json.load(open(args.inp))
    confs = sorted([e["conf"] for e in d["per_sample"] if e.get("conf") is not None], reverse=True)
    n_total = d["aggregate"]["n"]
    n_target_pass = int(round(args.target_rate * n_total))
    if n_target_pass <= 0 or n_target_pass > len(confs):
        print(f"[calibrate] WARNING: target_rate={args.target_rate} implies {n_target_pass} passing "
              f"samples but only {len(confs)} were folded at all; clamping.")
        n_target_pass = max(1, min(n_target_pass, len(confs)))
    threshold = confs[n_target_pass - 1]
    print(f"[calibrate] n_total={n_total} n_folded={len(confs)} target_rate={args.target_rate} "
          f"-> n_target_pass={n_target_pass} -> conf_min={threshold:.4f}")
    print(f"[calibrate] observed confidence range: min={min(confs):.4f} median={np.median(confs):.4f} "
          f"max={max(confs):.4f}")
    print(f"CALIBRATED_CONF_MIN={threshold:.4f}")


if __name__ == "__main__":
    main()
