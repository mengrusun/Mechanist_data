# Experiment Audit Report — Claim C3

**Date**: 2026-07-22
**Auditor**: self-review (llm-chat MCP unavailable — empty config in .mcp.json)
**Project**: Belief-Circuit Reproduction on Pythia
**Claim**: C3 — Formation Window: personal and attributed belief exhibit distinct formation windows during pythia-1b pretraining (behavioral t* and causal t†)
**Linked milestones**: M3

## Overall Verdict: WARN

*C3's integrity verdict — whether C3's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
- Same dataset-provided gold/distractor pairs as M1/M2; behavioral and ablation evals use belief_core JSONL files.
- Zero-ablation uses H*_personal and H*_attributed from M2 — head indices are fixed, not derived from model output.
- Evidence: `scripts/m3_formation.py` imports `evaluate_task_accuracy` from belief_utils (same GT as M1).

### B. Score Normalization: PASS
- Accuracy per checkpoint = binary correct/total. Δ = acc_behavioral - acc_ablated (raw difference).
- No self-normalization.
- Evidence: same `evaluate_task_accuracy()` function as M1/M2.

### C. Result File Existence: PASS
- Formation summary JSON verified: `refine-logs/artifacts/formation/summary.json` exists with 24 checkpoint trajectories.
- Emergence windows confirmed from summary.json:
  - personal: behavioural_emergence_step=13000, causal_emergence_step=13000, formation_window=[13000,13000] ✓
  - attributed: behavioural_emergence_step=0, causal_emergence_step=33000, formation_window=[0,33000] ✓
  - distinct_windows=true ✓
- Key trajectory values match EXPERIMENT_RESULTS.md (e.g., step 2000: personal=0.043, attributed=0.971 ✓; step 13000: personal=0.633 > 0.60 ✓; step 33000: H*_attributed ablation drops attributed from 0.824 to 0.574, Δ=0.250 > 0.20 ✓).
- Evidence: `refine-logs/artifacts/formation/summary.json` (24-step trajectory data confirmed).

### D. Dead Code Detection: PASS
- `m3_formation.py` evaluates 9 measurements per checkpoint (3 behavioral + 3 H*_personal ablated + 3 H*_attributed ablated) — all computed.
- `m3_aggregate.py` applies emergence criteria and writes summary — output files confirmed present.
- Evidence: `refine-logs/artifacts/formation/summary.{json,md}` exist.

### E. Scope Assessment: WARN
- **24 of 154 planned pythia-1b intermediate checkpoints were run** (only the native log-spaced subset available on disk). This is documented as an environmental constraint (missing files, not a cost-saving downscale), consistent with `resource_fidelity: strict` protocol.
- The formation-window detection uses "2-of-next-3 recorded checkpoints" persistence criterion, which adapts to the available log-spaced schedule — this is appropriate.
- Scope reduction is prominently documented in EXPERIMENT_RESULTS.md, MECHANISM_ROUTING.md, and CLAIMS_LEDGER.md. Claim language is appropriately hedged ("24 of 154 planned checkpoints").
- **WARN (not FAIL)**: the coarse-log-spaced resolution (24 checkpoints vs. plan's 154) is a genuine scope limitation — the emergence-step localization has ~10× coarser granularity than planned — but: (a) the plan explicitly acknowledged this risk; (b) the formation windows are clearly distinct at this resolution (personal [13000,13000] vs attributed [0,33000]); (c) the honest documentation satisfies the integrity requirement. Claim wording is precise and not overclaimed.

### F. Evaluation Type: real_gt
- All checkpoint evaluations use dataset-provided gold/distractor continuations.
- Classification: **real_gt** (same as M1/M2).

## Action Items
- E (scope WARN): The reduced checkpoint set (24/154) limits emergence-step resolution. If resources allow, running additional checkpoints between 0-13000 and 13000-33000 would sharpen the formation windows. Not a methodology flaw — adequately documented.
