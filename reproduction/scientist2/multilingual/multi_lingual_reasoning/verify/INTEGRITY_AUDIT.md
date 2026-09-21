# Integrity Audit

## Baseline integrity (Phase 2, per-claim)

All four target claims audited against `refine-logs/` artifacts. Combined verdict = `max_severity(experiment_audit, mechanism_audit)` where N/A is treated as PASS.

| Claim | Exp verdict | Mech verdict | Combined | Gate decision |
|-------|-------------|--------------|----------|---------------|
| C1 — V_lang subspace decomposition | PASS | N/A (no steering) | **PASS** | ADMITTED — continue to Stage 2 |
| C2 — Null-space projection | WARN | FAIL (single hardcoded α=−1, no sweep, n_random=1, α in collapse range) | **FAIL** | REJECTED → INCONCLUSIVE |
| C3 — Signed dose-response | WARN (1 seed, n=50/lang) | WARN (factor-6 span, no σ_proj, n_random=1 per α) | **WARN** | ADMITTED — continue to Stage 2 |
| C4 — Training-free vs SFT | FAIL (M4b not run, 6.8% training data, n=50/lang) | N/A (LoRA ≠ steering) | **FAIL** | REJECTED → INCONCLUSIVE |

**ADMITTED pool**: C1 (PASS), C3 (WARN)
**REJECTED pool (INCONCLUSIVE)**: C2 (mechanism FAIL), C4 (experiment FAIL)

### Per-claim audit locations

- C1: `verify/C1_vlang_subspace_decomposition/main_experiment_audit/{EXPERIMENT_AUDIT,MECHANISM_AUDIT}.{md,json}`
- C2: `verify/C2_null_space_projection/main_experiment_audit/{EXPERIMENT_AUDIT,MECHANISM_AUDIT}.{md,json}`
- C3: `verify/C3_signed_dose_response/main_experiment_audit/{EXPERIMENT_AUDIT,MECHANISM_AUDIT}.{md,json}`
- C4: `verify/C4_training_free_vs_sft/main_experiment_audit/{EXPERIMENT_AUDIT,MECHANISM_AUDIT}.{md,json}`

### Inconclusive reasons

- C2: `mechanism_audit = FAIL` — single hardcoded α=−1, no sweep, random control n=1 (required ≥30), α placed in capability-collapse range. Sub-audit: `verify/C2_null_space_projection/main_experiment_audit/MECHANISM_AUDIT.md`.
- C4: `experiment_audit = FAIL` — M4b (training-free comparison leg) never run; LoRA training used 6.8% of planned data; eval at n=50/lang vs planned 250/lang; ≥85% predicate untestable. Sub-audit: `verify/C4_training_free_vs_sft/main_experiment_audit/EXPERIMENT_AUDIT.md`.

---

## Variant integrity (Phase 9) — RE-AUDITED after Iteration ①

Variant integrity audit results for Stage-2 picked claim C3 (model-swap: DeepSeek-R1-Distill-Llama-8B).

Per-claim variant audit directory: `verify/C3_signed_dose_response/variant_audit/`

| Claim | Variant | Exp verdict | Mech verdict | Combined | Eligible |
|-------|---------|-------------|--------------|----------|----------|
| C3 — Signed dose-response | model_swap_deepseek_r1_llama8b | WARN | **WARN** (was FAIL) | **WARN** | **YES** (was NO) |

**N_run = 1, N_eligible = 1** → C3 upgraded to **PASS** (robustness = 1/1 = 1.0 ≥ 0.5; refutation is robust across model families)

### Mechanism WARN breakdown (C3 variant, post re-audit 2026-07-14 14:06)

Check A — Steering coefficient sweep WARN for model_swap_deepseek_r1_llama8b:
- σ_proj unit scaling: NOT used (α in raw units) — WARN (future-work note; not a FAIL criterion once random control is present)
- Capability metric logged: yes (GlotLID macro_fidelity at all 9 sweep points)
- Collapse range α ∈ {+0.5, +1.0, +1.5}: labeled as generic OOD forcing (both V_lang AND random collapse) — interpreted only outside this window
- Random-direction control: **YES** — 9-point random-subspace α-sweep on same grid, seed, rank, site (dispatched by Iteration ①)
- Specificity signal: at |α|>=1 in the non-collapse window, V_lang costs 8-20 pp of macro_acc while random costs 0
- Output spot-check: metric_text_consistent

### Experiment WARN breakdown (C3 variant)

- GT provenance: PASS
- Score normalization: PASS
- Result existence: WARN (single seed) — random control results now on disk (9 files)
- Dead code: WARN (grade() imported but unused)
- Scope: WARN (single seed, n=50/lang)

### Prior audit (superseded)
**2026-07-14 12:21** — Mech FAIL (random control not yet on disk at audit time due to dispatch race). See `verify/C3_signed_dose_response/variant_audit/MECHANISM_AUDIT.md` "Prior audit" section for the original.

### Audit artifacts

- `verify/C3_signed_dose_response/variant_audit/EXPERIMENT_AUDIT.md` (re-audited)
- `verify/C3_signed_dose_response/variant_audit/EXPERIMENT_AUDIT.json` (re-audited)
- `verify/C3_signed_dose_response/variant_audit/MECHANISM_AUDIT.md` (re-audited — WARN)
- `verify/C3_signed_dose_response/variant_audit/MECHANISM_AUDIT.json` (re-audited — WARN)
