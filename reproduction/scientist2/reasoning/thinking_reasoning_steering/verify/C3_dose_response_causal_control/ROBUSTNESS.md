## C3: dose-response causal control — Main-experiment integrity FAIL  →  INCONCLUSIVE

- verdict: INCONCLUSIVE
- inconclusive_reason: main-experiment mechanism rigor broken — see verify/C3_dose_response_causal_control/main_experiment_audit/MECHANISM_AUDIT.md
- swap_variants_run: false
- Main-experiment verdict on C3: not-supported (partial for uncertainty, negative for other three)
- Main-experiment integrity: FAIL (exp=WARN, mech=FAIL → combined=FAIL)

**Mechanism audit FAIL reasons:**
1. α sweep range ±2σ does NOT span 3 orders of magnitude (catalogue requires ≥ 3 OOM grid, e.g. [0.1, 0.3, 1, 3, 10]×σ_proj)
2. No random-direction control at locked α (α_op=0.5σ) with n_random ≥ 30 — required to distinguish learned-direction specificity from general perturbation
3. α_op=0.5σ placed at plateau edge: on-target Δrate=+0.017 for uncertainty is within ±0.06 binomial noise floor at n=60 — no confirmed mid-plateau
4. 3 of 4 behaviours show zero rate across all α: dose-response causal control claim fails on 75% of target behaviours

Interpretation: The Phase 2 combined verdict (exp=WARN, mech=FAIL → combined=FAIL) means the main experiment's mechanism rigor for C3 is insufficient to support a causal dose-response claim. Swap variants were not run — computing robustness around this broken anchor would be uninformative. The iteration loop should fix the mechanism rigor first (see MECHANISM_AUDIT.md action items), then re-run M3 at n ≥ 300 with random-direction control and proper 3-OOM grid, and re-invoke /auto-verify.
