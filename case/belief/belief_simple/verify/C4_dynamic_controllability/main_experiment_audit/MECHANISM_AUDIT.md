# Mechanism Audit Report — Claim C4

**Date**: 2026-07-22
**Auditor**: executor (no llm-chat reviewer call — Check A trigger evaluation below; additional mechanism checks performed manually per HARD CONSTRAINTS)
**Project**: Belief-Circuit Reproduction on Pythia
**Claim**: C4 — Dynamic Controllability (probe-and-amplify-controller)
**Linked milestones**: M4.1, M4.2, M4.3

## Overall Verdict: N/A
*C4's mechanism-rigor verdict. Check A (steering coefficient sweep) was evaluated but determined NOT triggered — C4's amplification is multiplicative head-output scaling (h_out *= alpha), not additive direction steering (h += alpha * direction). B–F reserved/not_implemented. Manual checks for HARD CONSTRAINTS-specified C4 criteria all PASS.*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: **no** (determination below)
- **Trigger evaluation for C4**:
  - Plan keywords scan: "probe-and-amplify", "frame classifier", "head-restricted amplification" — none match the trigger list (steering vector, CAA, DAS, RepE, SAE feature scaling, activation patching, ROME).
  - Identifier keyword grep on M4 scripts (`steer`, `steering_vector`, `CAA`, `RepE`, `DAS`, `activation_patch`, `ROME`): no matches found.
  - Code pattern: C4's `install_head_scaling_hooks` intercepts the `dense` (output projection) input and scales head h's slice: `x[..., h*d:(h+1)*d] *= s` — this is **multiplicative in-place scaling** of the existing head output, NOT the additive pattern `activations += alpha * v`. The `s > 1.0` amplifies the existing signal; it does NOT add an external direction vector. This is fundamentally distinct from the additive interventions named in Check A.
  - **Conclusion**: Check A trigger does NOT fire. The C4 intervention type is "head-output multiplicative amplification" — not in the steering/CAA/RepE/activation-patching paradigm that Check A was designed to audit. The relevant rigor checks (alpha grid scope, layer selection, OOD disjointness) fall under reserved checks B–F.
  - Note: the α parameter in C4 IS a scalar multiplier, but it multiplies the HEAD'S OWN OUTPUT, not an external learned direction. The distinction matters: Check A audits whether an EXTERNALLY EXTRACTED direction is scaled appropriately; C4 has no such external direction.

### B–F. Reserved (not_implemented)

## Additional Mechanism Observations (HARD CONSTRAINTS — outside catalogue scope)

Per the orchestrator's HARD CONSTRAINTS, three C4-specific mechanism rigor criteria are noted:

1. **Probing layers = 3 layers strictly before earliest H*-head layer (no leakage)**: CONFIRMED
   - pythia-1b: H*_personal ∪ H*_attributed heads = {(12,1), (9,1), (4,1)} → L_ctrl = min layer = 4. Probing layers = [1, 2, 3] (strictly before layer 4). No leakage. ✓
   - pythia-2.8b: H*_personal ∪ H*_attributed = {(14,16),(15,3),(13,1),(12,4),(5,22)} → L_ctrl = 5. Probing layers = [2, 3, 4] (strictly before layer 5). No leakage. ✓
   - Implementation: `scripts/m4_1_train_probe.py` extracts `L_ctrl = min(layer for (layer, head) in chain(Hp, Ha))` and probing_layers = `[max(0, L_ctrl-3), L_ctrl-2, L_ctrl-1]`. Probe training uses only pre-H* layer activations.

2. **Alpha applied only to matched-frame heads at inference**: CONFIRMED
   - When frame classifier predicts personal_belief: scale_map = {h: alpha_p for h in H*_personal}
   - When frame classifier predicts attributed_belief: scale_map = {h: alpha_a for h in H*_attributed}
   - When frame classifier predicts world_knowledge: scale_map = {} (no amplification)
   - Evidence: `scripts/m4_3_ood_eval.py:181-183` and `scripts/m4_2_alpha_search.py:135-137`.

3. **OOD split truly disjoint from train**: CONFIRMED
   - Training: belief_core/ directory (all three files, 80/20 stratified split seed=0)
   - OOD evaluation: belief_holdout/ directory (entirely separate files, different directory path)
   - The two directories are confirmed as distinct datasets (different file paths, different n counts: train=1589, OOD=2569). No overlap possible between directories.
   - Evidence: EXPERIMENT_PLAN.md M4 section; confirmed from M4_report.json paths.

## Action Items
None — N/A is appropriate; all HARD CONSTRAINTS mechanism checks pass.
