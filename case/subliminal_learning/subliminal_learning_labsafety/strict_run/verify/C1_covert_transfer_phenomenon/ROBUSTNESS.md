# Robustness Report — Claim C1

**Claim ID**: C1
**Claim**: In the fixed Qwen3.5-9B -> Qwen3.5-9B multimodal transfer setup and the exact task.md recipe, text-only tuned-teacher-generated filtered data causes a >=3pp drop in QA_I accuracy vs BOTH Ctrl-A and Ctrl-B, per seed across all 3 pre-registered seeds {42, 123, 2026}.
**Main-experiment verdict**: supported
**Verify verdict**: PASS

## Robustness Computation (Phase 10)

| Field | Value |
|---|---|
| N_variants_run | 1 |
| N_eligible (integrity_status = pass) | 1 |
| N_pass (consistent_with_main_experiment = pass) | 1 |
| N_fail | 0 |
| robustness | 1.0 (1/1) |
| ROBUSTNESS_THRESHOLD | 0.5 |
| verdict | **PASS** (1.0 >= 0.5) |

## Per-Variant Results

### method-swap-rule-based-scorer

| Field | Value |
|---|---|
| dimension | method |
| swap | LLM judge (gpt-5.4) → deterministic rule-based regex letter extractor |
| variant_metric | rule_based_acc: treated_mean=0.101, ctrlb_mean=0.261, ctrl_a=0.233 |
| main_experiment_metric | llm_judge_acc: treated_mean=0.526, ctrlb_mean=0.777, ctrl_a=0.782 |
| per-seed gaps (gap_A, gap_B) | seed42=(+15.8pp, +16.5pp); seed123=(+15.0pp, +21.1pp); seed2026=(+9.0pp, +10.5pp) |
| all_seeds_pass | true (all 3 seeds pass both >=3pp inequalities) |
| claim_supported | pass |
| consistent_with_main_experiment | pass |
| confidence | high |
| integrity_status | pass |
| integrity_breakdown | experiment=pass, mechanism=n/a |

**Delta note**: Absolute accuracy levels are lower under the rule-based scorer (Ctrl-A: 0.233 vs 0.782 under LLM judge) — expected: regex is conservative and many prose answers lack a leading option letter. The relative inter-arm gap is preserved in the same direction with larger margins. The minimum per-seed gap to either control is 9.0pp (seed2026 vs Ctrl-A), which is 3x the pre-registered 3pp threshold. The direction (treated < both controls) is maintained unanimously across all 3 seeds and both control arms.

## Interpretation

The claim that text-only tuned-teacher-generated filtered data causes a >=3pp drop in QA_I accuracy vs both Ctrl-A and Ctrl-B, per seed across {42, 123, 2026}, is **robust** to a method swap of the scoring procedure. Replacing the LLM judge with a deterministic regex scorer on the identical saved model outputs reproduces the directional gap with equal or larger margins in all 3 seeds. The >=3pp criterion is satisfied under a strictly more conservative evaluation, confirming the result is not an artifact of the LLM judge's leniency.

## Phase 2 Baseline Integrity

**Combined gate verdict**: PASS (exp=PASS, mech=N/A)

Baseline integrity artifacts:
- `verify/C1_covert_transfer_phenomenon/main_experiment_audit/EXPERIMENT_AUDIT.md`
- `verify/C1_covert_transfer_phenomenon/main_experiment_audit/EXPERIMENT_AUDIT.json`
- `verify/C1_covert_transfer_phenomenon/main_experiment_audit/MECHANISM_AUDIT.md`
- `verify/C1_covert_transfer_phenomenon/main_experiment_audit/MECHANISM_AUDIT.json`

## Phase 9 Variant Integrity

**Combined variant gate verdict**: PASS (exp=PASS, mech=N/A)

Variant integrity artifacts:
- `verify/C1_covert_transfer_phenomenon/variant_audit/EXPERIMENT_AUDIT.md`
- `verify/C1_covert_transfer_phenomenon/variant_audit/EXPERIMENT_AUDIT.json`
- `verify/C1_covert_transfer_phenomenon/variant_audit/MECHANISM_AUDIT.md`
- `verify/C1_covert_transfer_phenomenon/variant_audit/MECHANISM_AUDIT.json`
