# Robustness Report — C2

**Claim:** Belief representations are localizable to a small set of attention heads H* satisfying all four criteria (C2a: ≥30% target accuracy drop; C2b: drop > mean+2σ of 20 random-head controls; C2c: ≤10% off-target/WK drop; C2d: PPL ≤1.05×).

**Main-experiment verdict:** supported (Pythia-1B, attributed_belief localized; Jackknife rho high)

**Verify verdict: FAIL**  
**Robustness:** 0.00 (0/1 eligible variants passed; threshold 0.50)

---

## Stage 1 — Baseline Integrity

**Overall: PASS** (see `baseline_audit/`)

- EXPERIMENT_AUDIT: pass — correct GT, Fisher formula, metric, scope
- MECHANISM_AUDIT: warn — cross-vocab PPL note (absolute PPL not comparable across models; ratio C2d valid); all mechanism checks pass
- Combined: WARN → admitted to Stage 2

## Stage 2 — Variant Run

**Dimension swapped:** model  
**Variant:** model-swap-olmo-1b (OLMo-1B-hf, 16L/16H/2048D)  
**Run:** 2026-07-22 07:16:15 – 08:13:28 (4730s, GPU 1, A800-SXM4-80GB)

### M1 Gate Results (OLMo-1B)

| Target | Acc | CI | Admitted |
|---|---|---|---|
| personal_belief | 0.778 | [0.746, 0.808] | yes |
| attributed_belief | 0.731 | [0.697, 0.763] | yes |
| world_knowledge | 0.930 | [0.889, 0.956] | yes (Fisher only) |

Both belief targets pass M1 above-chance gate.

### M2 Localization Results

**personal_belief:** NOT localized (Jackknife rho=0.926 — Fisher consistent but no valid H*)

Greedy-add trace (30 steps, cap exhausted):
- C2a (drop ≥ 0.30) satisfied from |S|=3 (drop=0.404)
- C2c (off-target ≤ 0.10) NEVER satisfied — off-target drops exceeded 0.10 at every step where C2a was met
- C2d (PPL ≤ 1.05×) failed from |S|=6 onwards (ppl_r → 46.3× at cap)
- Root failure: ablating personal_belief heads in OLMo-1B causes general LM disruption (C2d) AND is not belief-specific (C2c); the circuit is not cleanly separable in this model

**attributed_belief:** NOT localized (Jackknife rho=0.954 — very consistent Fisher but still no valid H*)

Greedy-add trace (30 steps, cap exhausted):
- C2a (drop ≥ 0.30) NEVER satisfied — drops are consistently NEGATIVE (ablating top Fisher heads *increases* attributed_belief accuracy: drop=-0.145 at |S|=1 through -0.244 at |S|=30)
- C2c status irrelevant (C2a never met)
- C2d failed from |S|=6 onwards (ppl_r → 7.37× at cap)
- Root failure: the top-Fisher heads for attributed_belief in OLMo-1B appear to be suppression heads — ablating them increases attributed_belief accuracy rather than decreasing it. This is inconsistent with the localization premise (C2a requires accuracy DROP, not increase). The Fisher scoring identifies parameter sensitivity but the causal direction is reversed relative to Pythia.

**n_localized:** 0 / 2 admitted  
**consistent_with_main_experiment:** False

### Phase 8 — Variant Judgment

Main experiment verdict: supported (attributed_belief localized in Pythia-1B)  
Variant result: n_localized=0 → does NOT support C2  
**variant_judgment: FAIL**

## Stage 3 — Variant Integrity & Robustness

### Phase 9 — Variant Integrity Audit

| Audit | Verdict |
|---|---|
| EXPERIMENT_AUDIT | pass |
| MECHANISM_AUDIT | warn (cross-vocab PPL) |
| Combined | **warn** (eligible) |

integrity_status = warn → variant eligible (N_eligible=1)

### Phase 10 — Robustness Computation

| Metric | Value |
|---|---|
| N_run | 1 |
| N_eligible | 1 (integrity WARN = eligible) |
| n_pass | 0 |
| n_fail | 1 |
| robustness | 0.00 |
| threshold | 0.50 |
| verdict | FAIL |

**C2 verify verdict: FAIL**

Robustness = 0.00 < 0.50 (threshold). The single eligible variant disagrees with the main experiment's conclusion that belief heads are localizable. In OLMo-1B, neither personal_belief nor attributed_belief has a valid H* satisfying all four criteria simultaneously.

## Scientific Interpretation

The failure pattern differs by target:

1. **personal_belief**: Fisher heads are causally relevant (ablating them drops accuracy by 40-70%) but the circuit is not clean — it disrupts both the target and other beliefs simultaneously (C2c fails) and causes severe LM degradation (C2d fails catastrophically). This suggests personal_belief processing in OLMo-1B is more entangled with general language modeling than in Pythia-1B.

2. **attributed_belief**: The Fisher-ranked heads are suppression heads — ablating them *improves* attributed_belief accuracy (negative drop). This is architecturally interesting but disqualifies localization: the four criteria assume a positive causal relationship between head activity and target accuracy, which is inverted here.

Both results point to a fundamental architectural difference between OLMo-1B and Pythia-1B in how belief processing is organized. The claim C2 may be Pythia-specific rather than a general property of causal language models.

---

**Artifacts:**
- `variants/model-swap-olmo-1b/result.json` — n_localized=0
- `variants/model-swap-olmo-1b/run.log` — full greedy search trace (107 lines)
- `variant_audit/EXPERIMENT_AUDIT.{md,json}` — integrity: pass
- `variant_audit/MECHANISM_AUDIT.{md,json}` — integrity: warn (cross-vocab PPL)
