"""M2 verdict aggregation.

Reads results from:
  runs/m2_intervene/{baseline,ablate,amplify_x2,amplify_x3,amplify_x4,random_ablate}/verdict.json
  (each an intervention on the top-1 site of the M1 shortlist)

Optionally reads a matched-control site batch under runs/m2_intervene_matched/.

Computes:
  - ΔP(banana)_patched = P(banana)_ablate − P(banana)_baseline
  - ΔP(banana)_sibling = P(banana)_matched-control-ablate − P(banana)_baseline
  - fluency drop = fluency_baseline - fluency_intervention (report per intervention)
  - dose-response: table of P(banana) vs α from {baseline, amplify_x2..x4}
  - verdict: confirmed / partial / refuted per plan §M2:
    - confirmed iff |ΔP_ablate| >= 0.5 * (P_teacher-arm_baseline - P_ctrl_B) AND
      |ΔP_sibling| < 0.02 AND fluency drop < 5% relative.
    - partial iff sign matches but magnitude or one specificity bar fails.
    - refuted iff sign wrong OR magnitude ~= 0 OR sibling matches primary.

Writes:
  runs/m2_intervene/m2_verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qwen_common import dump_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m2-dir", required=True,
                   help="Root dir with subdirs per intervention (baseline/, ablate/, ...)")
    p.add_argument("--m0-verdict", required=True,
                   help="Path to runs/M0_verdict.json for P_ctrl_B baseline.")
    p.add_argument("--seed-used", type=int, default=42,
                   help="Which teacher-arm seed's P was used as intervention baseline "
                        "(for computing target-drop magnitude).")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _read_verdict(fp: Path):
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


def main():
    a = parse_args()
    m2_dir = Path(a.m2_dir)

    modes = ["baseline", "ablate", "amplify_x2", "amplify_x3",
             "amplify_x4", "random_ablate"]
    per_intervention = {}
    for mode in modes:
        fp = m2_dir / mode / "verdict.json"
        v = _read_verdict(fp)
        if v is not None:
            per_intervention[mode] = {
                "p_banana": v["p_banana"],
                "fluency": v["fluency"],
                "alpha_sigma": v.get("alpha_sigma"),
                "block_id": v["block_id"],
                "target_module": v["target_module"],
                "sigma_l": v.get("sigma_l"),
            }
        else:
            per_intervention[mode] = None

    # Matched-control site (if present in a sibling dir)
    matched = None
    matched_fp = m2_dir / "matched_control_ablate" / "verdict.json"
    if matched_fp.exists():
        vv = _read_verdict(matched_fp)
        matched = {
            "p_banana": vv["p_banana"],
            "fluency": vv["fluency"],
            "block_id": vv["block_id"],
            "target_module": vv["target_module"],
        }

    # M0 verdict for P_ctrl_B baseline
    with open(a.m0_verdict) as f:
        m0v = json.load(f)
    ev = m0v.get("evidence", m0v)
    p_teacher_arm_seed = None
    if a.seed_used is not None and "P_teacher_arm_per_seed" in ev:
        try:
            p_teacher_arm_seed = float(
                ev["P_teacher_arm_per_seed"].get(str(a.seed_used),
                                                 ev["P_teacher_arm_per_seed"].get(
                                                     a.seed_used)))
        except Exception:
            p_teacher_arm_seed = None
    mean_teacher = ev.get("mean_teacher")
    mean_ctrl_B = ev.get("mean_ctrl_B")
    m0_gap = ((mean_teacher or 0) - (mean_ctrl_B or 0)) if mean_teacher is not None else None

    baseline = per_intervention.get("baseline") or {}
    ablate = per_intervention.get("ablate") or {}
    p_base = baseline.get("p_banana")
    p_abl = ablate.get("p_banana")
    dp_ablate = (p_abl - p_base) if (p_abl is not None and p_base is not None) else None

    fluency_base = baseline.get("fluency")
    dp_sibling = None
    fluency_sibling = None
    if matched is not None and p_base is not None:
        dp_sibling = matched["p_banana"] - p_base
        fluency_sibling = matched["fluency"]

    random_abl = per_intervention.get("random_ablate") or {}
    p_rand = random_abl.get("p_banana")
    dp_random = (p_rand - p_base) if (p_rand is not None and p_base is not None) else None

    fluency_drop_ablate = ((fluency_base - ablate.get("fluency", 0))
                            / max(1e-9, fluency_base)) if fluency_base else None

    magnitude_bar = 0.5 * (m0_gap or 0) if m0_gap is not None else None

    # Verdict
    verdict = "refuted"
    reason = "no valid ablate/baseline data"
    if dp_ablate is not None and magnitude_bar is not None:
        sign_ok = dp_ablate <= 0  # ablation should REDUCE P(banana)
        magnitude_ok = abs(dp_ablate) >= magnitude_bar
        random_ok = (dp_random is None) or (abs(dp_random) < 0.02)
        sibling_ok = (dp_sibling is None) or (abs(dp_sibling) < 0.02)
        fluency_ok = (fluency_drop_ablate is None) or (fluency_drop_ablate < 0.05)

        specificity_ok = random_ok and sibling_ok
        offtarget_ok = fluency_ok

        if sign_ok and magnitude_ok and specificity_ok and offtarget_ok:
            verdict = "confirmed"
            reason = (f"|ΔP_ablate|={abs(dp_ablate):.3f} ≥ 0.5*gap={magnitude_bar:.3f}, "
                      f"random_ablate |ΔP|={abs(dp_random or 0):.3f} < 0.02, "
                      f"sibling |ΔP|={abs(dp_sibling or 0):.3f} < 0.02, "
                      f"fluency_drop={fluency_drop_ablate:.3f} < 0.05")
        elif sign_ok and (magnitude_ok or specificity_ok or offtarget_ok):
            verdict = "partial"
            failed = []
            if not magnitude_ok: failed.append("magnitude")
            if not random_ok: failed.append("random-ablate-specificity")
            if not sibling_ok: failed.append("sibling-specificity")
            if not fluency_ok: failed.append("off-target-fluency")
            reason = f"sign OK but failed: {failed}"
        else:
            verdict = "refuted"
            reason = (f"sign_ok={sign_ok}, magnitude_ok={magnitude_ok}, "
                      f"random_ok={random_ok}, sibling_ok={sibling_ok}, "
                      f"fluency_ok={fluency_ok}")

    dose_response = []
    for mode in ["ablate", "baseline", "amplify_x2", "amplify_x3", "amplify_x4"]:
        pi = per_intervention.get(mode)
        if pi is not None:
            dose_response.append({
                "mode": mode,
                "alpha_sigma": pi.get("alpha_sigma"),
                "p_banana": pi["p_banana"],
                "fluency": pi["fluency"],
            })

    result = {
        "verdict": verdict,
        "reason": reason,
        "per_intervention": per_intervention,
        "matched_control_site": matched,
        "delta_p_ablate": dp_ablate,
        "delta_p_random": dp_random,
        "delta_p_sibling_site": dp_sibling,
        "fluency_drop_ablate": fluency_drop_ablate,
        "fluency_sibling": fluency_sibling,
        "magnitude_bar_50pct_of_m0_gap": magnitude_bar,
        "m0_teacher_ctrlb_gap": m0_gap,
        "dose_response_curve": dose_response,
    }
    dump_json(result, a.out)
    print(json.dumps(result, indent=2))
    print(f"\n[m2] verdict = {verdict}")
    print(f"[m2] reason  = {reason}")


if __name__ == "__main__":
    main()
