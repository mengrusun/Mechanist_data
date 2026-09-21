## C3: INCONCLUSIVE (Phase 2 combined verdict: FAIL — mechanism rigor broken)

- verdict: INCONCLUSIVE
- inconclusive_reason: main-experiment mechanism rigor broken — see verify/C3_causal_steering_dissociation/main_experiment_audit/MECHANISM_AUDIT.md
- swap_variants_run: false (Phase 2 FAIL — variants never dispatched)
- Main-experiment verdict on C3: supported
- Main-experiment integrity: FAIL (mech audit)
- Variants: none (Stage 2 skipped — Phase 2 FAIL)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C3's causal-dissociation claim (additive steering) failed the mechanism-rigor audit (Check A). Specific issues: (1) alpha was reported in raw direction-norm units, not sigma_proj units; the sweep spans only ~2.5 sigma_proj range, not >=3 orders of magnitude; (2) the r-direction refusal effect is threshold-like with the effect appearing only at alpha=+2 (the boundary of the tested range), so no interior plateau was identified and the locked alpha is at the grid edge; (3) the random-direction baseline used n_random=1, not the required >=30 for statistical comparison. Running swap variants on top of this un-tuned sweep would compute robustness around a broken anchor. 

To fix: re-sweep alpha in sigma_proj units over >=3 orders of magnitude (e.g., [0.03, 0.1, 0.3, 1.0, 3.0] sigma_proj), run n_random>=30 directions, and identify mid-plateau alpha. Then re-invoke `/auto-verify C3 — resume: true` (Phase 2 audit will re-run; Stage 2 variants will follow if Phase 2 passes).
