# Robustness Report — C3: Bidirectional Causal Steering

**Claim**: Injecting a probe-extracted activation direction at inference time causally and substantially shifts the target variable's effect on transfer in both the amplifying and the attenuating/inverting sense.
**Verify round**: 1 (resume from Phases 8–10)
**Date**: 2026-07-13
**DIMENSIONS**: model
**ROBUSTNESS_THRESHOLD**: 0.5
**MIN_VARIANTS_FOR_VERDICT**: 1

---

## Phase 2 — Baseline Integrity (main experiment)

| Audit | Verdict | Key findings |
|-------|---------|--------------|
| EXPERIMENT_AUDIT | WARN | synthetic_proxy GT; supp run uses raw v_hat_V not pure GS/LEACE; 5-point alpha grid (not 7); coherence gating omitted in supp run (benign; format=1.0) |
| MECHANISM_AUDIT | WARN | no L=16 random-direction control; plateau lock post-hoc; alpha range ±2σ at L=16 |
| Combined | WARN | Gate decision: continue-with-warn |

**Main experiment verdict on C3**: supported at L=16 supplementary run (key positive finding); inconclusive/under-powered at picked layers ell_V*.

---

## Phase 3 Step 0 — Stage-2 Pick

C3 was **picked** (1/4 admitted claims, cap=1). See `verify/STAGE2_PICK.json`.

---

## Phase 7 — Variants Run

| Variant ID | Dimension | Swap model | Layers evaluated | Status |
|------------|-----------|------------|------------------|--------|
| model-swap-meta-llama3-8b | model | Meta-Llama-3-8B-Instruct | ell_V*_swap (G:10, A:10, I:0, M:0) + L=16 | COMPLETE |

**N_run**: 1

---

## Phase 8 — Result-to-Claim

### Layer selection

The swap model (Meta-Llama-3-8B-Instruct) uses L=16 as the mid-layer evaluation point, identical relative depth to the main experiment's L=16 supp finding (both models are 32-layer; L=16 = 50% depth). The swap model's ell_V* differs (G:10, A:10, I:0, M:0 vs main G:4, A:6, I:2, M:2) but L=16 is the reference layer for Phase 8 per instructions.

### Results at L=16: Swap vs. Main

| V | Main baseline | Main alpha=+2 | Swap baseline | Swap alpha=+2 | Sign-inversion match |
|---|---------------|---------------|---------------|---------------|---------------------|
| G | −0.380 | +0.570 (invert) | +0.583 | +0.417 (attenuate) | G inverts at alpha=-1 in swap (+0.583->-0.333) — bidirectional pattern preserved |
| A | +0.106 | +0.584 (amplify 5.5x) | 0.000 | +0.400 (amplify from null) | near-null baseline in both; large modulation in both |
| I | +1.287 | +1.010 (attenuate) | −2.762 | −2.571 | I inverts at alpha=-1/-2 in swap (−2.762 → +0.286/+1.0); strong bidirectionality |
| M | +0.713 | **−0.310 (INVERTS)** | +0.667 | **−0.667 (INVERTS)** | **V=M sign-inverts at alpha=+2 in BOTH models — key C3 criterion met** |

### Consistency verdict

**consistent_with_main_experiment**: TRUE

The central C3 criterion — at least one variable undergoes sign inversion at L=16 — is met in both the main experiment (V=M at alpha=+2: +0.713 → −0.310) and the swap variant (V=M at alpha=+2: +0.667 → −0.667). Additionally, V=G and V=I both exhibit sign inversions in the swap at other alpha values, and V=A shows large bidirectional shifts from a near-null baseline. The bidirectional causal steering signature at L=16 generalizes across the Llama-3 checkpoint family.

**Caveats**:
- n_baseline=10 per cell (vs n=200 in main); effects are qualitatively consistent but not individually statistically significant.
- Swap model probe direction polarities differ from main (e.g., G baseline changes sign), but bidirectionality is confirmed across both alpha directions.

---

## Phase 9 — Variant Integrity Gate

| Variant | Exp. audit | Mech. audit | Combined | Eligible? |
|---------|------------|-------------|----------|-----------|
| model-swap-meta-llama3-8b | WARN | WARN | WARN | YES |

**N_eligible**: 1 (no FAIL variants; all WARN variants admitted to N_eligible)
**N_excluded (FAIL)**: 0

Key integrity findings:
- Coherence gate: PASS (format_ok_rate=1.0 for all 40 cells)
- Hook site: PASS (identical LlamaDecoderLayer architecture)
- n_baseline=10: WARN (compact sweep; not grounds for FAIL; no redispatch permitted)
- No random-direction control at L=16: WARN (inherited from main supp run design)

---

## Phase 10 — Robustness Computation

```
N_run = 1
N_eligible = 1   (1 variant admitted by Phase 9; 0 excluded)
N_pass = 1       (consistent_with_main_experiment = true; Phase 8 RESULT_TO_CLAIM.json)
N_fail = 0

robustness = N_pass / N_eligible = 1/1 = 1.00
threshold = 0.50
MIN_VARIANTS_FOR_VERDICT = 1
N_eligible (1) >= MIN_VARIANTS_FOR_VERDICT (1)  -> verdict computable
robustness (1.00) >= threshold (0.50)            -> PASS
```

---

## Verdict: PASS

**robustness**: 1.00 (1/1 eligible variants consistent with main experiment)
**baseline conclusion**: supported (at L=16 supplementary)
**variant integrity**: WARN (n=10 compact sweep; no L=16 random-direction control)

**Interpretation**: The C3 bidirectional causal steering claim — particularly V=M sign-inversion at alpha=+2 at L=16 — is replicated in Meta-Llama-3-8B-Instruct, a distinct Llama-3 checkpoint. The replication is at qualitative/pattern level only (n=10 per cell in the swap variant limits statistical confirmation of individual effect sizes). The robustness score of 1.00 reflects full qualitative pattern match across the one eligible model-swap variant. The WARN integrity flag on the variant should be resolved in a future pass (re-run at n=200; add L=16 random-direction control) before claiming quantitative robustness.

---

## Upgrade commands (future passes)

```bash
# Re-run swap variant at full n=200 to confirm quantitative effect sizes:
# /auto-verify C3 --resume true --dimensions model --redispatch-variant model-swap-meta-llama3-8b
#
# Add dataset-swap or method-swap variant:
# /auto-verify C3 --resume true --dimensions dataset,method
```
