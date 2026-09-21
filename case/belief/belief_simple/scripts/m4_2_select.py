"""M4.2 selection: pick the (α_personal*, α_attributed*) maximizing net_improvement
subject to Δ_wk ≤ 0.05 (soft guardrail); relax to 0.10 if none; else min-Δ_wk."""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from belief_utils import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha-grid-dir", required=True, help="dir containing a_p_*_a_a_*.json files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--delta-wk-strict", type=float, default=0.05)
    ap.add_argument("--delta-wk-relaxed", type=float, default=0.10)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.alpha_grid_dir, "a_p_*_a_a_*.json")))
    print(f"[m4.2-sel] scanning {len(files)} grid cells")
    rows = [load_json(f) for f in files]
    if not rows:
        raise RuntimeError(f"no grid cells found in {args.alpha_grid_dir}")

    # 1. strict guardrail
    cand = [r for r in rows if r["delta_wk"] <= args.delta_wk_strict]
    if cand:
        best = max(cand, key=lambda r: r["net_improvement"])
        guardrail = f"strict Δ_wk ≤ {args.delta_wk_strict}"
    else:
        # 2. relaxed
        cand = [r for r in rows if r["delta_wk"] <= args.delta_wk_relaxed]
        if cand:
            best = max(cand, key=lambda r: r["net_improvement"])
            guardrail = f"relaxed Δ_wk ≤ {args.delta_wk_relaxed}"
        else:
            # 3. min-Δ_wk fallback + flag
            best = min(rows, key=lambda r: r["delta_wk"])
            guardrail = "no combination satisfies soft guardrail — min-Δ_wk fallback"

    result = {
        "model": best["model"], "alpha_personal": best["alpha_personal"], "alpha_attributed": best["alpha_attributed"],
        "val_net_improvement": best["net_improvement"], "val_delta_wk": best["delta_wk"],
        "val_frame_acc": best["frame_acc"],
        "guardrail_applied": guardrail,
        "grid_scanned": len(rows),
    }
    save_json(args.output, result)
    print(f"[m4.2-sel] chosen (α_p*, α_a*)=({best['alpha_personal']}, {best['alpha_attributed']}) "
          f"net_impr={best['net_improvement']} Δ_wk={best['delta_wk']:.4f} ({guardrail}) → {args.output}")


if __name__ == "__main__":
    main()
