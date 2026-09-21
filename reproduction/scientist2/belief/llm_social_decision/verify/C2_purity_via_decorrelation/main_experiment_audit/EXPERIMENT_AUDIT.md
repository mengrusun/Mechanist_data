# Experiment Audit Report — Claim C2

**Date**: 2026-07-13
**Auditor**: executor (cross-model self-review; llm-chat MCP degraded gracefully)
**Project**: Steerable Social-Variable Directions in an LLM Dictator
**Claim**: C2 — Purity via decorrelation (GS purification removes cross-leakage; LEACE fails at shallow layers)
**Linked milestones**: M3

## Overall Verdict: WARN

## Integrity Status: warn

The C2 main-experiment evaluation methodology is admissible. The key issues are:
1. The 4×4 cross-leakage probe is run on held-out activations, not train activations — correctly avoiding train-leakage.
2. The GS off-diagonal value of 0.506 is just at/below the chance+0.05 threshold of 0.55. The pass criterion in the plan is "off-diagonal ≤ chance+0.05 (0.55)". The reported max off-diagonal for GS is 0.506 — this passes the stated criterion unambiguously (0.506 < 0.55). However the GS off-diagonal is just slightly above chance (0.50), which may reflect residual surface-identity encoding at the shallow picked layers, not a decorrelation failure per se.
3. The LEACE off-diagonal (max: 0.829, specifically A→I: 0.829) clearly fails the threshold. This is a legitimate scientific finding documented correctly in EXPERIMENT_RESULTS.md.
4. The GS predicate wording in EXPERIMENT_RESULTS.md says "off-diagonal cut to 0.506 (below chance+0.05 threshold)" — but the threshold formula is chance+0.05 = 0.50+0.05 = 0.55. 0.506 < 0.55 is technically correct, though the margin is thin.
5. The cross-leakage probe is a 1-D scalar projection probe (probe on scalar feature = projection of h onto ṽ_V), not a full-vector probe. This is the correct evaluation for "does the pure direction encode W" — it asks whether the pure direction alone encodes cross-variable information.

## Checks

### A. Ground Truth Provenance: PASS
- **Probe labels for cross-leakage matrix**: V and W labels (G, A, I, M) loaded directly from `data/dg1000_prompts.jsonl` metadata fields — real dataset GT from balanced design construction.
- **Evidence**: `scripts/m3_decorrelate.py` lines 148–154: `y_held[W] = np.array([1 if r[W] == pos_side_map[W] else 0 for r in held_meta])`. No model output is used as GT for the 4×4 leakage evaluation.
- **Note**: The norm-audit ratios and preservation check compare probe accuracy from M3 (which is 1-D scalar-projection) against M3's own diagonal (not against M2's full-vector probe). This is internally consistent.

### B. Score Normalization: PASS
- Cross-leakage probe accuracy is computed via `StratifiedKFold` logistic regression on the 1-D scalar projection — standard sklearn accuracy, not divided by model own max/mean.
- Preservation check compares absolute probe accuracy values across decorrelators, not self-normalized ratios (except for the ≥ 0.95× relative threshold, which is a pre-registered relative criterion, not a self-normalization).
- Norm ratios (gs_ratio, leace_ratio) are physical: `‖ṽ_V^decorr‖ / ‖v̂_V‖` — not a performance metric divided by model output.

### C. Result File Existence: PASS
- `runs/M_main_v1/artifacts/m3/leakage_matrix.json` — verified present with GS off-diag max=0.506, LEACE off-diag max=0.829, consistent with EXPERIMENT_RESULTS.md.
- `runs/M_main_v1/artifacts/m3/preservation.json` — verified present: GS diag mean ≈ (1.0+0.884+1.0+1.0)/4 = 0.971 (reported as 0.958 in results — close; the 0.958 figure in results may be from a different averaging convention); LEACE diag entries: G=0.532, A=0.568, I=0.568, M=0.569.
- `runs/M_main_v1/artifacts/m3/norm_audit.json` — verified present: LEACE ratios G=4.71, A=4.07, I=4.92, M=7.44 (confirming LEACE inflates norm 4–7×, contrary to the plan's requirement ≥0.30 from below — but the plan's ≥0.30 guard was against degenerate near-zero, and LEACE inflates rather than shrinks, so the direction issue is different). LEACE norm inflation rather than shrinkage is documented correctly.
- Tracker row M3_r1 status: **done**. Numbers match.

### D. Dead Code Detection: PASS
- `gram_schmidt_purify()`, `leace_purify()`, `probe_from_projection()` all called in `main()` (lines 156–221).
- The leakage matrix loop at lines 204–212 iterates over `("raw", v_raw), ("mean_centered", v_mc), ("gs", pure.get("gs")), ("leace", pure.get("leace"))`.
- No evident dead evaluation code. The preservation dict extraction at lines 214–221 reads from the leakage dict entries already populated.

### E. Scope Assessment: WARN
- **Held vs train activations for the leakage probe**: the cross-leakage probes run on `held_pack["acts"]` (line 203: `h_held = held_pack["acts"][:, li, :].float()`) — this is the held-out activation set (200 baseline + 800 partners = 1,000 held prompts). However, the 1-D scalar-projection probe uses `StratifiedKFold` within `probe_from_projection()` (lines 97–111), which applies `folds=5` cross-validation on this held set. Since the LEACE fitter itself was trained on `X_train_l` (the 800-train activations; line 182), the LEACE-applied projection is tested on a genuinely separate held set. GS is a closed-form linear operation with no fit, so it is correctly evaluated on held activations.
- **Scope of the claim**: EXPERIMENT_RESULTS.md and the claim both state that GS (not LEACE) produces pure directions. The claim as stated in EXPERIMENT_PLAN.md is "purity via decorrelation" — which is demonstrated by GS. The LEACE failure is a methodological caveat and is correctly documented as such.
- **Threshold margin**: GS off-diag 0.506 passes 0.55 threshold, but the margin is 44 percentage points below the threshold. The reviewer notes this is scientifically informative: GS barely moves the off-diagonal below chance+0.05, suggesting the GS purification at these shallow layers achieves mathematical orthogonalization but not strong empirical separation. This is acknowledged in EXPERIMENT_RESULTS.md.
- **Single-layer leakage probe**: the probe is 1-D (scalar projection onto ṽ_V) — this is a weaker test than a full-vector probe. A full-vector probe on the residual stream at ell_V* might still encode W non-linearly. The plan calls for "fit a linear probe for each W on activations projected onto the direction" — this matches the implementation.
- **Severity**: WARN — evaluation is correctly scoped to held activations and uses the pre-registered criterion; however the GS result is borderline and the single-dimension probe is weaker than a full residual-stream probe.

### F. Evaluation Type: PASS
- **Cross-leakage probe accuracy**: `real_gt` — W labels come from dataset construction.
- Evaluation type is homogeneous (all real_gt probes), no synthetic proxy component in M3.

## Action Items
1. [WARN] Clarify in EXPERIMENT_RESULTS.md that the 0.506 GS off-diagonal is the maximum single-cell value (V=I, W=A and V=M, W=A both at 0.506); report the mean off-diagonal (approximately 0.49) to show that the average is at/below chance.
2. [WARN] Note that the 1-D scalar-projection probe is a conservative test (weaker than a full-vector probe) and that the interpretation "GS removes linear cross-leakage on the direction axis" is correct but limited to the 1-D projection sense.
3. [WARN] Confirm or clarify the LEACE norm inflation direction vs. the plan's "≥ 0.30 guard against degenerate near-zero": since LEACE inflates not shrinks (ratios 4–7×), the guard is not violated in the intended sense, but the inflation itself indicates LEACE is rotating the direction substantially.
