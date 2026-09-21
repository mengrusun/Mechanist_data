#!/usr/bin/env python3
"""Reviewer call for Phase 2 mechanism audit on all 5 claims."""

import json
import requests

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"

prompt = """You are a rigorous mechanistic-experiment rigor auditor. I am reviewing 5 claims from a mechanistic interpretability paper that uses additive steering vectors (CAA / difference-in-means). Your job is to audit Check A (Steering Coefficient Sweep) for each relevant claim.

=== CHECK A: Steering Coefficient Sweep ===

Trigger: fires when ANY additive activation intervention with a scalar coefficient alpha is used.

Required criteria (answer YES/NO for each):
1. Was a sweep performed (multiple alpha tried), not a single hardcoded value?
2. Did the sweep span ≥ 3 orders of magnitude AND include alpha=0 baseline?
3. Was alpha expressed in sigma_proj units (k * sigma_proj), not raw norm units?
4. Was BOTH a target metric AND an independent capability/coherence metric logged at every sweep point?
5. Was a usable plateau identified (target clearly above baseline noise AND capability within tolerance)?
6. Was the locked alpha placed in the MIDDLE of the plateau (not at its edge)?
7. Was a random-direction control run at the locked alpha with n_random >= 30?
8. If asymmetric protocol: is sign pattern preserved?

FAIL if ANY of:
- Single hardcoded alpha (no sweep)
- Sweep done but NO capability metric logged
- alpha chosen where capability has crashed
- alpha chosen where target effect is within baseline-noise floor
- Asymmetric protocol flattened into uniform-push

WARN if ANY of (and no FAIL):
- Sweep performed but spans < 3 orders of magnitude, OR < 5 grid points
- alpha expressed in raw norm rather than sigma_proj units
- Plateau identified but alpha sits at edge (not middle)
- No random-direction control at the locked alpha (n_random < 30)

PASS if all 8 criteria satisfied AND chosen alpha is mid-plateau with both behavior and capability evidence.

=== EXPERIMENTAL DETAILS FOR THIS PROJECT ===

STEERING EXPERIMENT (M3 — applies to C3 only, other claims don't use steering):
- Alpha grid: {-2, -1, -0.5, 0, +0.5, +1, +2} = 7 points
- Alpha unit: DIRECTION NORM units, NOT sigma_proj units
  * sigma_proj_h = 1.58 (from directions.json)
  * sigma_proj_r = 1.96
  * So in sigma_proj units, the grid is: [-1.27, -0.63, -0.32, 0, 0.32, 0.63, 1.27] for h
  * Span: only ~2.5 sigma_proj range, NOT 3 orders of magnitude
- Target metrics logged per cell: h_readout_mean_harm (for h direction), refusal_rate_ben (for r direction)
- Capability/coherence metrics logged: mean_logp_completion AND rep_rate (both logged for all 28 cells)
- Plateau: h-readout is monotone across all 7 alpha for h direction; r-refusal only lifts at alpha=+2
- Locked alpha: alpha=+2 (or alpha=-2 for negative sweep) — these are at the EDGE of the grid, not the middle
- Random-direction baseline: n_random=1 matched-norm Gaussian (NOT n>=30)
- Sign pattern: h direction correctly steers h-readout in predicted direction (positive alpha -> increase); preserved
- Fluency check: no cells exceeded collapse thresholds (rep_rate > 0.20 or mean_logp < baseline-2)

CLAIMS THAT USE STEERING:
- C3 only uses additive steering (M3). C1, C2, C4, C5 do NOT use steering — they use probing, position analysis, attack projection, and AUROC comparison respectively.

=== OUTPUT FORMAT ===

Return ONLY a valid JSON object (no markdown, no preamble) with this structure for each claim:
{
  "C1": {"triggered": false, "overall_verdict": "n/a", "details": "C1 uses probe AUROC and cosine similarity, no steering intervention"},
  "C2": {"triggered": false, "overall_verdict": "n/a", "details": "C2 uses position-ladder AUROC, no steering"},
  "C3": {
    "triggered": true,
    "intervention_type": "CAA",
    "sweep_grid": [-2, -1, -0.5, 0, 0.5, 1, 2],
    "sigma_proj_scaling": false,
    "capability_metric": "mean_logp_completion + rep_rate",
    "plateau_range": [assessment],
    "locked_alpha": 2.0,
    "alpha_position": "edge",
    "random_baseline": {"run": true, "n_random": 1, "passed": false},
    "sign_pattern": "preserved",
    "output_case_spotcheck": {"cases_available": false, "verdict": "no_cases_logged", "note": "only aggregate metrics logged"},
    "overall_verdict": "WARN or FAIL",
    "details": "detailed paragraph"
  },
  "C4": {"triggered": false, "overall_verdict": "n/a", "details": "C4 measures delta projections on pre-computed directions, no additive steering intervention"},
  "C5": {"triggered": false, "overall_verdict": "n/a", "details": "C5 compares probe AUROC to Llama Guard, no steering"}
}

Be rigorous. The key questions for C3 are: (1) is not being in sigma_proj units a WARN or FAIL? — it should be WARN since the sweep was done and the range is small but not trivially so; (2) is n_random=1 a WARN or FAIL? — it's missing the >=30 requirement so it's WARN (no random control was adequately run); (3) is alpha at the edge a WARN? yes."""

resp = requests.post(
    f"{BASE_URL}/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500},
    proxies={"http": None, "https": None},
    timeout=180
)
print(f"Status: {resp.status_code}")
try:
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    print("REVIEWER_RESPONSE:")
    print(content)
except Exception as e:
    print(f"Parse error: {e}")
    print(resp.text[:2000])
