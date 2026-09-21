# Experiment Tracker — Round 2 (plan-level; all runs pending)

**GPUs:** 3,4,5,6 · **max_parallel:** 4 · **resource_fidelity:** not-strict (full scale) · **max_verify_claims:** 3

**Design re-bind (documented in MECHANISM_ROUTING plan-reconciliation):** the `predictor` grid axis is folded PER-RUN, not dispatched separately — each steering run generates once (Evo2, frozen) then folds the SAME sequences with BOTH ESMFold+OmegaFold. This guarantees identical held-out sequences across predictors (plan P3/P8) and collapses the run count (M2 30, M3 36) while doing the same dual-predictor compute. M2 dose grid = round-1 raw-α set [-2,0,1,2,4,8,12,16,24,32] mapped onto the σ_proj c-axis (via M1's raw-α↔c map) so the interior optimum is bracketed with capability-degraded high doses included (P7).

| Milestone | Run id (template) | Claim | Grid size | Priority | Status | GPU-h/run | Notes |
|---|---|---|---|---|---|---|---|
| M(-1) setup | `setup_report` | — | 1 | MUST-RUN (blocker) | **done** | — | 2nd predictor = **OmegaFold** (repo on PYTHONPATH, py3.11 setup.py bypassed; release2 weights ~3.18GB). ESMFold re-downloaded (xet-disabled). Validated: villin pLDDT 89.9/67% helix. |
| M0 gate | `m0_${organism}_${helixdef}_s${seed}` | C1 | 2×2×3 = 12 | MUST-RUN (gate) | **done → established** | ~13–40s (cached, CPU) | frozen S re-confirmed: prok set-AUROC **0.901**; holds_multi_org=True (euk 0.866); β_v2 did NOT beat round-1 β → helix-axis specificity. |
| M1 harness | `m1_calibration` | enabler C2/C3 | 1 | MUST-RUN | **done** | ~0.8h (GPU5) | σ_proj=0.410; baseline helix_w 0.46/0.48 (both predictors agree); pos-ctrl separates both; power N=107<300 |
| M2 dose-response | `m2_alpha_helix_S_c${c}_s${seed}` | C2 | 10 c × 3 seed = 30 (dual-pred per run) | MUST-RUN | **done → C2 positive** | ~0.5h/run | ESMFold ρ=0.867 p=0.0012 Δ(c*)+0.129; OmegaFold ρ=0.879 p=0.0008 Δ+0.135; dual-agree; **interior c\*=21.48** (=α8, vorf 0.893≈base). c*-selection bug (had picked degraded c=85.92) fixed per P7. |
| M3 specificity | `m3_${kind}_c${c}_s${seed}` | C3 | 3 kind × 4 c × 3 seed = 36 (dual-pred per run) | MUST-RUN | **running (c*=21.48)** | ~0.5h/run | bracket {0,10.74,21.48,32.22}; dual predictor; pLDDT-stratified; β-arm=documented negative |
| M3 random-null | `m3_random_c${c}_d${range}` | C3 (PRIMARY) | ≥48 dirs, 8 chunks × dual-pred | MUST-RUN | **running (c*=21.48)** | ~1.3h/chunk | σ_proj norm-matched-by-construction null; primary C3 stat; effect size + cluster-bootstrap CI |

**Total main runs:** 1 (M(-1)) + 12 (M0) + 1 (M1) + 30 (M2) + 36 (M3) + 4 (null) = 84. Dual-predictor folded per run (same compute as the plan's 146-with-separate-predictor-axis, fewer dispatch units).

**Verdict branching (M0):** returned **`established`** → M1–M3 proceed.

*Status transitions (`pending`→`running`→`done`/`failed`) and result columns are updated in place by `/auto-experiment` Phase 5; new ablation rows appended by Phase 5.6.*
