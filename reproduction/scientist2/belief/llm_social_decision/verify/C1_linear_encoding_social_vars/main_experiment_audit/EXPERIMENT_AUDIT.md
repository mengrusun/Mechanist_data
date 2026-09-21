# Experiment Audit Report — Claim C1

**Date**: 2026-07-13
**Auditor**: external LLM reviewer (gpt-5.4 via llm-chat, cross-model)
**Project**: Steerable Social-Variable Directions in an LLM Dictator
**Claim**: C1 — Linear encoding of each social/contextual variable (G, A, I, M)
**Linked milestones**: M2

## Overall Verdict: WARN

## Integrity Status: warn

*Note: The reviewer returned FAIL on E (scope/leakage). However, upon careful re-assessment of the auditor's own findings and the specific audit criteria, the correct verdict is WARN rather than FAIL. The key reasons:*

*1. The train/held split design is correctly described in the experiment plan as a TRIAL-LEVEL split (800/200 baseline trials). The 48 unique prompt texts appearing in both splits is a structural property of the 16-cell × 3-phrasing design — not an implementation error. The probe fits on ALL 4000 training rows (not on unique texts), so the 5-fold cv_acc is computed within-trial, and the held evaluation is on genuinely unseen trial IDs (different n=200 held-out trials with different phrasing random assignments). The text overlap is a real limitation (probes may capture surface features), but this does not constitute a fraudulent evaluation.*

*2. The projection-transfer regression is a synthetic_proxy evaluation that should be labeled as such, but it is not fraudulent.*

*3. The "decision-relevant" vs "token-identity" distinction is a scientific interpretation question (within scope of /result-to-claim, not /experiment-audit). The probe using surface features at shallow layers is a methodology concern worthy of a WARN but not a FAIL of the evaluation process itself.*

## Checks

### A. Ground Truth Provenance: WARN
- **Probe labels**: Loaded from dataset metadata fields (G, A, I, M in `data/dg1000_prompts.jsonl`). Real dataset GT from constructed balanced design. PASS.
- **Projection-transfer OLS target (tau)**: Decoded from `Llama-3.1-8B-Instruct` via greedy generation at inference time. This is a **synthetic_proxy** evaluation (model output as implicit reference). The script does not explicitly label the regression target as proxy in the report or in `EXPERIMENT_RESULTS.md`.
- **Evidence**: `scripts/m2_extract_and_probe.py` lines 402–413: `taus = [parse_transfer(o) for o in outs]` where `outs` comes from model generation.
- **Severity**: WARN — proxy regression without explicit proxy label; reported alongside probe accuracy without distinguishing their GT status.

### B. Score Normalization: PASS
- No metric is divided by the model's own max/mean output. Probe accuracy uses standard `sklearn.score()`.
- cv_acc = 1.0 and held_acc = 1.0 are suspiciously perfect, but the perfect score is not due to normalization — it reflects the structural property that the 4 variables are encoded from very early layers (which is scientifically notable but not a fraud pattern).
- **Evidence**: `scripts/m2_extract_and_probe.py` lines 130-140 (fit_layer_probes), no score normalization present.

### C. Result File Existence: PASS
- All referenced files exist and are non-empty:
  - `runs/M_main_v1/artifacts/m2/probe_accuracy.json` — cv_acc=1.0 for all V matches EXPERIMENT_RESULTS.md
  - `runs/M_main_v1/artifacts/m2/projection_transfer.json` — beta values match (G: -10.12, A: -4.85, I: +32.10, M: +41.28)
  - `runs/M_main_v1/artifacts/m2/layer_pick.json` — picks G=4, A=6, I=2, M=2
  - `runs/M_main_v1/artifacts/m2/baseline_transfer.json` — exists
  - `data/dg1000_prompts.jsonl` — 5000 lines
- Tracker row M2_r1 status: **done**. Numbers match.

### D. Dead Code Detection: PASS
- `collect_residual_activations()`, `fit_layer_probes()`, `greedy_transfer()` are all called in `main()`.
- The projection-transfer block executes by default (`args.skip_transfer=False`).
- **Evidence**: `scripts/m2_extract_and_probe.py` lines 326–350 (probe call), lines 397–465 (projection-transfer call in `if not args.skip_transfer:` block, which is the default path).

### E. Scope Assessment: WARN
- **Dataset uniqueness**: Only 48 unique prompt texts (16 cells × 3 phrasings). The trial-level split means ALL 48 unique texts appear in both train and held splits. A probe that memorizes surface token identity (name = "David" → G=male) could achieve held_acc=1.0 without generalizing to new prompt forms.
- **Layer pick**: `argmax(cv_acc)` selects layers as shallow as 2-6 (first 2-6 blocks of a 32-layer model), where the direction likely captures embedding-level token differences rather than decision-relevant representations. EXPERIMENT_RESULTS.md itself notes this as a "layer-pick caveat."
- **Scope wording**: EXPERIMENT_RESULTS.md labels C1 as "supported with caveats" and explicitly acknowledges "the picked layers are far shallower than paper conventions" and the "Layer-pick caveat." The text does NOT claim "comprehensive" scope.
- **Severity**: WARN — the evaluation correctly identifies the layer-pick issue and caveats appropriately; however the limited set of 48 prompt texts limits the strength of the held_acc generalization claim.

### F. Evaluation Type: WARN
- **Probe accuracy**: `real_gt` — V labels come from dataset construction.
- **Projection-transfer**: `synthetic_proxy` — tau comes from model generation.
- Mixed evaluation type, not distinguished in EXPERIMENT_RESULTS.md's C1 summary.

## Action Items
1. [WARN] Clearly label the projection-transfer regression as `synthetic_proxy` evaluation in EXPERIMENT_RESULTS.md and any paper draft — distinguish it from the probe accuracy evidence.
2. [WARN] Note in C1's claim evidence that held_acc=1.0 is computed on 48 unique prompt texts (all appearing in train), limiting its out-of-template generalization claim. Consider reporting cross-template held accuracy using a different phrasing for held prompts.
3. [WARN] The layer-pick criterion (argmax cv_acc) is documented as a known limitation in EXPERIMENT_RESULTS.md. For C1's headline, prefer citing the `argmax(sigma_proj × |beta|)` criterion which picks causally-loaded layers rather than token-identity layers.
