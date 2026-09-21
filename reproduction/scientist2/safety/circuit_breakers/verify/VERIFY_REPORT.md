# Verify Report

**Project**: Verifying Representation-Level Circuit Breakers (RR) as a Safety Intervention
**Date**: 2026-07-15
**Parameters**: TARGET_CLAIMS=all, DIMENSIONS=model, MAX_VERIFY_CLAIMS=1, ROBUSTNESS_THRESHOLD=0.5, MIN_VARIANTS_FOR_VERDICT=1
**GPU_ID**: 0,1,2,3 | **MAX_PARALLEL_RUNS**: 4 | **COMPACT**: false

---

## Per-Claim Verdicts

| Claim | Short text | Baseline verdict | Robustness | Eligible | Variants | Integrity | State |
|-------|-----------|-----------------|-----------|---------|---------|-----------|-------|
| C1 | Identifiability + reroute | — | — | 0/0 | 0/0 | N/A | INCONCLUSIVE |
| C2 | ASR + capability | — | — | 0/0 | 0/0 | N/A | INCONCLUSIVE |
| C3 | VLM transfer | — | — | 0/0 | 0/0 | N/A | INCONCLUSIVE |
| C4 | Agent transfer | — | — | 0/0 | 0/0 | N/A | INCONCLUSIVE |

All 4 target claims are INCONCLUSIVE. Phase 2 baseline integrity gate failed for every claim (combined = FAIL), so Stage 2 (swap variants) never ran. Robustness is undefined; no variants were executed.

---

## Stage-2 Selection

**Phase 3 step 0 result**: admitted_pool=[] (empty) — no claim advanced past Phase 2 gate.

The Stage-2 pick is vacuous: MAX_VERIFY_CLAIMS=1 but admitted_pool contains 0 claims. All 4 claims are in the rejected_pool.

Had any claim been admitted, the top-1 pick by importance would have been **C1** (mechanism-level parent claim on which C2, C3, C4 all depend; highest importance per FINAL_PROPOSAL.md §6). The planned Mistral-7B-Instruct-v0.2 model-swap variant (artifacts/m6/{m1_mistral, m3_mistral, m4_mistral} already on disk from M6) would have been cited as the Stage 2 variant without re-running.

See `verify/STAGE2_PICK.json` for full pick record.

---

## Detailed Per-Claim Findings

### C1 — Identifiability + reroute (INCONCLUSIVE)

**Linked milestones**: M1, M4
**Phase 2 combined verdict**: FAIL (exp=WARN, mech=FAIL)

**Experiment audit (WARN)**:
- Direction type deviation: M1 saves LR probe coefficients as `directions.pt` rather than plan-specified mean-difference direction; mean-diff saved separately as `directions_meandiff.pt` (undisclosed in EXPERIMENT_RESULTS.md)
- `sweep_notes` in training_summary.json is a misleading static string ("L_rr decreasing significantly") inconsistent with final L_rr_ema=0.0016
- Alpha=10.0 used vs plan alpha=1.0; loss function changed to cos_sq_plus_signed vs plan cos_sq
- Scope: single model, single seed — within tolerance for the experiment plan
- GT: real HarmBench+Alpaca pairs, AUC vs true labels — PASS
- Files: all exist and match reported numbers — PASS

**Mechanism audit (FAIL — Check A)**:
- No alpha sweep: single hardcoded alpha=10.0 (plan specified 1.0)
- No sigma_proj scaling
- No capability metric logged at any alpha value
- Random-direction control: n=1 (required n≥30)
- Target reroute effect not demonstrated: delta_cos_harmful=-0.0197 vs required ≤-0.30 (off by ~15x)
- Training instability: grad_norm spikes to 852 (step 330), 159 (step 480), 137 (step 490)
- L_rr_ema near-zero from step 0 (final=0.0016)

**inconclusive_reason**: main-experiment mechanism rigor broken — see `verify/C1_identifiability_reroute/main_experiment_audit/MECHANISM_AUDIT.md`

**Iteration guidance**:
1. Fix M3 RR fine-tune: run alpha sweep [0.1, 0.5, 1.0, 5.0, 10.0, 50.0] with MMLU/MT-Bench capability metric at each point; compute sigma_proj and express alpha in those units; lock to mid-plateau alpha; increase random-direction control to n≥30.
2. Fix training instability: investigate grad-norm spikes (grad clipping < 1.0, lower LR, or higher warmup fraction for the RR objective).
3. Re-run Phase 2 audits after fixing M3; if combined verdict upgrades to PASS or WARN, C1 can proceed to Stage 2 with the Mistral-7B-Instruct-v0.2 model-swap variant (M6 artifacts already on disk).

---

### C2 — HarmBench ASR + capability (INCONCLUSIVE)

**Linked milestones**: M2, M3, M5
**Phase 2 combined verdict**: FAIL (exp=FAIL, mech=FAIL)

**Experiment audit (FAIL)**:
- B1 baseline uses R2D2-lite (12 adversarial framing templates) instead of plan-specified full 512-GCG suffix optimisation; this is a substantive substitution that weakens the competitive comparison (R2D2-lite over-refuses relative to a properly adversarial-trained B1, masking whether RR beats a real adversarial baseline)
- HarmBench eval uses "gcg-lite" (fixed generic suffix, not real per-prompt GCG optimisation)
- MT-Bench n=40 vs plan-specified n=80 (undisclosed substitution)
- Negative result (RR_ASR=0.356 > B0_ASR=0.333) is honestly disclosed

**Mechanism audit (FAIL — Check A)**:
- M3 is in C2 scope; same three FAIL violations as C1: single alpha=10.0, no capability metric, reroute not demonstrated (delta_cos_harmful=-0.020)
- No sigma_proj scaling; random-direction control n=1 (requires ≥30)

**inconclusive_reason**: main-experiment integrity broken (experiment + mechanism) — see `verify/C2_asr_capability/main_experiment_audit/{EXPERIMENT,MECHANISM}_AUDIT.md`

**Iteration guidance**:
1. Fix experiment integrity: replace R2D2-lite B1 with genuine adversarial training baseline, or clearly label as proxy and adjust claim scope; run real GCG optimisation per HarmBench protocol; increase MT-Bench to n=80.
2. Fix M3 RR mechanism (see C1 iteration guidance) — prerequisite.
3. Re-run Phase 2 audits after both fixes.

---

### C3 — VLM transfer (INCONCLUSIVE)

**Linked milestones**: M6
**Phase 2 combined verdict**: FAIL (exp=WARN, mech=FAIL)

**Experiment audit (WARN)**:
- Full PGD adversarial-image optimisation and VLM assembly skipped (budget-gated; honestly disclosed in c3_verdict.json and EXPERIMENT_RESULTS.md)
- M6 substep files (m1_mistral, m3_mistral, m4_mistral) exist with correct reported values
- Partial evidence (Mistral mechanism level) honestly reported; scope fail is internally acknowledged

**Mechanism audit (FAIL — Check A)**:
- M6T (Mistral RR fine-tune): alpha=10.0 single value (no sweep), no capability metric logged, no sigma_proj scaling, random-direction control n=1
- Reroute not demonstrated: delta_cos_harmful=-0.012, L_rr_ema=0.000222 (near-zero throughout, same failure mode as M3)

**inconclusive_reason**: main-experiment mechanism rigor broken — see `verify/C3_vlm_transfer/main_experiment_audit/MECHANISM_AUDIT.md`

**Iteration guidance**:
1. Fix M6T RR mechanism (same as M3 fix — prerequisite before the VLM visual-attack evaluation makes sense).
2. Complete PGD adversarial-image optimisation and VLM assembly (budget-permitting after mechanism is fixed).
3. Re-run Phase 2 audits.

---

### C4 — Agent transfer (INCONCLUSIVE)

**Linked milestones**: M7
**Phase 2 combined verdict**: FAIL (exp=WARN, mech=FAIL)

**Experiment audit (WARN)**:
- BFCL uses real GT (AST function-name match vs BFCL_v3_exec_simple.json) — PASS
- Harmful tool-use rate uses authored prompts (100 prompts) + LLM judge — two-stage proxy; honestly disclosed
- Floor effect: B0 harmful_tool_use_rate=0.010; RR=0.040 (slightly worse, likely noise at floor)
- BFCL 50-item substitute documented; negative result (RR worsens harm rate) honestly disclosed

**Mechanism audit (FAIL — Check A)**:
- M7 reuses M3 LoRA adapter (m7_agent_eval.py:224 loads artifacts/m3/RR_lora) without agent-specific re-training or sweep
- Underlying M3 mechanism: same mandatory FAIL violations as C1 — single alpha=10.0, no capability metric, reroute not demonstrated, n_random=1
- C4 negative result consistent with M3 mechanism never activating in the agent scaffold

**inconclusive_reason**: main-experiment mechanism rigor broken — see `verify/C4_agent_transfer/main_experiment_audit/MECHANISM_AUDIT.md`

**Iteration guidance**:
1. Fix M3 RR mechanism first (prerequisite; C4 cannot be meaningfully tested until M3's mechanism activates).
2. Once mechanism is fixed, re-run M7 to check whether safety transfer holds in the agent scaffold.
3. Consider whether agent-domain prompts may require a separate fine-tune rather than direct reuse of M3 weights.

---

## Cross-Claim Summary

**Root cause (all 4 claims)**: The M3 representation-rerouting fine-tune failed to engage the RR mechanism. L_rr_ema reached near-zero (0.0016 for Llama-3, 0.000222 for Mistral) from step 0, indicating the LoRA subspace could not redirect activations away from the harmful representation subspace. delta_cos_harmful=-0.020 (Llama-3) and -0.012 (Mistral) are ~15x and ~25x too small relative to the ≤-0.30 criterion. The alpha=10.0 was hardcoded without a sweep; whether a different alpha would activate the mechanism is unknown. Grad-norm spikes (852 at step 330) indicate training instability further undermining the mechanism.

**All downstream claims (C2, C3, C4) depend on C1's M3 mechanism**. Until M3's RR fine-tune is fixed, none of C2/C3/C4 can be meaningfully tested.

**Claim C2** additionally has experiment-level integrity failures (B1 R2D2-lite substitution, gcg-lite eval) independent of the mechanism failure.

---

## Counts

- **PASS**: 0
- **FAIL**: 0
- **INCONCLUSIVE**: 4 (all 4 target claims — Phase 2 gate FAIL for each)
- **ZERO_ELIGIBLE_VARIANTS**: 0
- **INTEGRITY_ONLY**: 0
- **Total target claims**: 4

INTEGRITY_ONLY breakdown: 0 stage2_skip_reason=swap_variants_false + 0 stage2_skip_reason=max_verify_claims_cap

---

## Baseline Integrity (Phase 2)

Overall: **FAIL**

| Claim | Exp. audit | Mech. audit | Combined | Gate decision |
|-------|-----------|------------|---------|--------------|
| C1 | WARN | FAIL | FAIL | INCONCLUSIVE |
| C2 | FAIL | FAIL | FAIL | INCONCLUSIVE |
| C3 | WARN | FAIL | FAIL | INCONCLUSIVE |
| C4 | WARN | FAIL | FAIL | INCONCLUSIVE |

Details: `verify/INTEGRITY_AUDIT.md` and per-claim `verify/<claim_dir>/main_experiment_audit/`

---

## Variant Integrity (Phase 9)

**Skipped** — all main-experiment audits FAIL; no claim advanced to Stage 2; no variants were run.

---

## GPU Pin Propagation

All baseline run cost.json gpu_ids verified as subset of {0,1,2,3}:
- M1 → [0], M3 → [0], M4 → [0], M2 → [1], M5_B0 → [1], M5_B1 → [2], M5_RR → [3]
- M6L/M6T → [0,1], M6D → [0], M7_B0 → [0], M7_RR → [0]
- No pin-propagation failures detected.
- No variant runs were dispatched (Stage 2 never ran), so no variant gpu_ids to check.

---

## Artifacts

- `verify/VERIFY_REPORT.md` — this file
- `verify/INTEGRITY_AUDIT.md` — Phase 2 baseline + Phase 9 variant sections (one file)
- `verify/STAGE2_PICK.json` — Phase 3 step 0 pick record (admitted_pool empty)
- `verify/C1_identifiability_reroute/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` — WARN
- `verify/C1_identifiability_reroute/main_experiment_audit/MECHANISM_AUDIT.{md,json}` — FAIL
- `verify/C1_identifiability_reroute/ROBUSTNESS.md` — INCONCLUSIVE
- `verify/C2_asr_capability/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` — FAIL
- `verify/C2_asr_capability/main_experiment_audit/MECHANISM_AUDIT.{md,json}` — FAIL
- `verify/C2_asr_capability/ROBUSTNESS.md` — INCONCLUSIVE
- `verify/C3_vlm_transfer/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` — WARN
- `verify/C3_vlm_transfer/main_experiment_audit/MECHANISM_AUDIT.{md,json}` — FAIL
- `verify/C3_vlm_transfer/ROBUSTNESS.md` — INCONCLUSIVE
- `verify/C4_agent_transfer/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` — WARN
- `verify/C4_agent_transfer/main_experiment_audit/MECHANISM_AUDIT.{md,json}` — FAIL
- `verify/C4_agent_transfer/ROBUSTNESS.md` — INCONCLUSIVE

No variant artifacts (Stage 2 never ran).

---

## Upgrade Commands

To retry a single claim after fixing its mechanism, run:
```
/auto-verify <claim_id> -- resume: true
```

Per-claim stage2_skip_reason is not applicable here (all claims are INCONCLUSIVE, not INTEGRITY_ONLY). The applicable upgrade path is to fix M3's RR fine-tune (and C2's B1/GCG substitutions), re-run Phase 2 audits, and then invoke `/auto-verify` fresh (not `resume: true`, since audit files would be regenerated).
