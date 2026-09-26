# Integrity Audit

**Overall**: PASS
**Main-experiment integrity (Phase 2)**: PASS
**Variant integrity (Phase 9)**: PASS

---

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | PASS | N/A | PASS | continue | verify/C1_subliminal_transfer_established/main_experiment_audit/ |
| C2 | PASS | PASS | PASS | continue | verify/C2_mechanism_causal_refuted/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim (Checks A–F: GT provenance, score normalization, result file existence, dead code, scope, evaluation type).
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention (C1 is a behavioral phenomenon claim; M0 milestones involve no additive residual-stream intervention).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a`; n/a contributes severity 0. Both claims: max(pass, n/a) = pass for C1; max(pass, pass) = pass for C2.
> - **Gate decision** — both C1 and C2 are ADMITTED to Stage 2 (subject to MAX_VERIFY_CLAIMS cap).

### Per-claim findings summary

**C1 (M0.1–M0.5)**:
- GT: external gpt-4o judge (not model-derived). PASS.
- Normalization: p_banana = banana_n / num_prompts (fixed denominator). PASS.
- File existence: all 14 final eval JSONs + verdict.json + filter_stats.json exist; mean_gap=0.169, p=0.0078125 verified. PASS.
- Dead code: eval_student.py + m0_verdict.py fully live. PASS.
- Scope: 7 seeds × 2 arms × 160 prompts matches claim; Under-N (N=53) documented as task.md-flagged risk. PASS.
- Evaluation type: external-judge behavioral probe. PASS.
- Mechanism: N/A (C1 has no mechanism intervention).

**C2 (M1.1–M1.2)**:
- GT: external gpt-4o judge on steering-sweep images. PASS.
- Normalization: p_banana = banana_n / 40 (fixed prompt count). PASS.
- File existence: locate.json (top-3=[2,0,8]), verify.json (spearman=-0.60, inconclusive), verify_window_0_8.json (spearman=-0.257, refuted) — all exist. PASS.
- Dead code: m1_verify_steer.py + m1_verify_window.py fully live, specificity sweep executed. PASS.
- Scope: 40-prompt subset plan-sanctioned (MECHANISM_ROUTING.md); sufficient for null detection. PASS.
- Evaluation type: external-judge behavioral probe on causal-intervention outputs. PASS.
- Mechanism: alpha ∈ {-2..+3} sigma_proj units (single-block), ∈ {-1..+5} × 5.0 (window); fluency logged; random-direction specificity control run in both configs; negative alphas included. overall_verdict: PASS.

---

## Variant integrity (Phase 9)

| Claim | Variant | Exp. audit | Mech. audit | Combined | Detail |
|-------|---------|------------|-------------|----------|--------|
| C1 | model-swap-lora-rank8 | PASS | N/A | PASS | verify/C1_subliminal_transfer_established/variant_audit/ |

**C2 — no variant audited (stage2_skip_reason: max_verify_claims_cap; INTEGRITY_ONLY).**

### Variant findings summary

**C1 rank-8 variant**:
- GT: external gpt-4o judge on generated images (same instrument as main experiment; independent of model under test). PASS.
- Normalization: p_banana = banana_n / 160 (fixed denominator, matches main experiment). PASS.
- File existence: all 6 eval JSONs + result.json + 6 diagnostics.json exist and are consistent. Per-seed gaps [+0.175, +0.163, +0.113]; mean_gap = +0.150; residues = 0 both arms. PASS.
- Dead code: eval_student.py + m0_verdict.py fully live; NaN Wilcoxon is documented n=3 edge case, not a dead path. PASS.
- Scope: 3 seeds × 2 arms × 160 prompts — matches variant PLAN.md (adapted-for-cost from 7 seeds; PLAN's success rule requires majority ≥ 2/3, not Wilcoxon). PASS.
- Evaluation type: synthetic_proxy (model-generated images + external-judge labels) — same as main experiment; methodologically consistent. PASS.
- Mechanism: N/A (rank-8 variant is pure LoRA-SFT, no additive residual-stream intervention).
- **Combined**: PASS.

### result.json label reconciliation (documented, not a warning)

The variant's `result.json` writes `verdict: "conditional"` because `m0_verdict.py` requires Wilcoxon p < 0.05 for the `"established"` label — Wilcoxon is undefined at n=3 (NaN). The variant PLAN.md's explicit success criterion (`mean_gap ≥ 0.10` AND `majority ≥ 2/3` AND same direction) does NOT require Wilcoxon at n=3 and is fully met (mean_gap=0.150, majority=3/3, all positive). The variant is therefore recorded as PASS for `consistent_with_main_experiment` regardless of the internal script's stricter label. No integrity finding.

