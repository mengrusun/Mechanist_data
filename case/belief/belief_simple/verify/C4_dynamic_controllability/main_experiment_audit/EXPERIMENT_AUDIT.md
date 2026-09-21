# Experiment Audit Report — Claim C4

**Date**: 2026-07-22
**Auditor**: self-review (llm-chat MCP unavailable — empty config in .mcp.json)
**Project**: Belief-Circuit Reproduction on Pythia
**Claim**: C4 — Dynamic Controllability: probe-and-amplify controller improves OOD belief accuracy while preserving WK and PPL, and beats the prompt-hint baseline
**Linked milestones**: M4 (M4.1, M4.2, M4.3)

## Overall Verdict: PASS

*C4's integrity verdict — whether C4's experimental process is methodologically sound.*

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS
- OOD evaluation uses belief_holdout gold/distractor pairs — dataset-provided GT, not model output.
- Frame classifier training (M4.1) uses belief_core examples with dataset-derived labels (task type as frame label) — labels are derived from the dataset STRUCTURE (which file the example is from), not from model outputs.
- Evidence: `scripts/m4_3_ood_eval.py` uses `load_task()` from belief_utils; `scripts/m4_1_train_probe.py` assigns frame labels from task name (0=world_knowledge, 1=personal_belief, 2=attributed_belief — derived from dataset membership, not model output).

### B. Score Normalization: PASS
- Net improvement = recovered - degraded (count difference, not normalized).
- Task accuracy = binary fraction correct.
- PPL ratio = PPL_controller / PPL_clean (both in absolute nats/token, no model-relative normalization).
- Alpha selection: maximizes net_improvement count subject to Δ_wk ≤ 0.05 — Δ_wk is raw accuracy difference, not model-relative.
- Evidence: `scripts/m4_3_report.py`, `scripts/m4_3_ood_eval.py`.

### C. Result File Existence: PASS
- `refine-logs/artifacts/controller/pythia-1b/M4_report.json` verified:
  - personal_belief: baseline=0.672, controller=0.850 → +17.8pp ✓
  - attributed_belief: baseline=0.788, controller=0.909 → +12.1pp ✓
  - WK: 0.850 → 0.850 (exactly preserved) ✓
  - controller net_impr=151, prompt_hint net_impr=-1 ✓
  - PPL clean=8.756, controller=9.172, ratio=1.047× ✓
  - frame_acc_ood=0.9977 ✓
- All numbers match EXPERIMENT_RESULTS.md verbatim.
- M4 tracker rows all show status `done`.
- Evidence: `refine-logs/artifacts/controller/{pythia-1b,pythia-2.8b}/M4_report.json`.

### D. Dead Code Detection: PASS
- M4.1 frame classifier: trained and saved to `probe.pt`; used in M4.2 alpha search and M4.3 OOD eval.
- M4.2 alpha grid: 36 configs computed; `m4_2_select.py` reads them and writes `alpha_selected.json`.
- M4.3: 3 evaluation arms (baseline, controller, prompt_hint) all computed; `m4_3_report.py` aggregates.
- `install_head_scaling_hooks` / `remove_hooks` called for controller arm; PPL under controller evaluated.
- Evidence: `scripts/m4_3_ood_eval.py` calls all functions; artifacts present.

### E. Scope Assessment: PASS
- Two models tested (pythia-1b, pythia-2.8b) — both had BOTH H*_personal AND H*_attributed localized.
- pythia-410m excluded (per plan: only H*_personal available) — documented.
- Claim language: "controller + probe improves OOD belief accuracy" — two models tested, both positive, claim is correctly general within the tested scope.
- OOD evaluation on held-out belief_holdout (n=367/1101/1101) — distinct from training split.
- Prompt-hint baseline included as required comparison.
- Evidence: EXPERIMENT_RESULTS.md M4 section; EXPERIMENT_TRACKER.md.

### F. Evaluation Type: real_gt
- All OOD evaluations use dataset-provided gold/distractor from belief_holdout JSONL files.
- Frame classifier labels = dataset membership (dataset-derived, not model-generated).
- Classification: **real_gt**.

## Action Items
None — no integrity concerns identified.
