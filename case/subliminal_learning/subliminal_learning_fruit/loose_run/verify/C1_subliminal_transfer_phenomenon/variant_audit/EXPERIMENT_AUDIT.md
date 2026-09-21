# Experiment Audit Report — Variant: method-swap-binary-judge (Claim C1)

**Date**: 2026-07-20
**Auditor**: executor self-review (graceful degradation — llm-chat MCP unavailable; same methodology as main-experiment audit in Phase 2)
**Project**: Subliminal Learning in Diffusion Image Models (Qwen-Image)
**Claim**: C1 — Subliminal banana preference transfers from teacher to student (P(banana) gap ≥ 5pp, ≥ 6/8 seeds, residue = 0)
**Variant**: method-swap-binary-judge (dimension=method; binary yes/no judge vs 10-way MCQ)
**Linked milestones**: M0 (variant re-scores M0.7 eval PNGs with alternative judge)

## Overall Verdict: PASS

*This is the variant's integrity verdict — whether the variant evaluation is methodologically sound.*

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS
The variant has no external ground-truth dependency. P(banana_binary) = count("yes") / N over existing PNGs from runs/eval_gen/. The "ground truth" signal is the structural split between teacher arms (which received banana training signal via LoRA) and control arms (no banana signal). Identical proxy-evaluation design to main experiment. No synthetic GT derived from model outputs used as reference.

**Evidence**: `run.py:106-123` — `score_arm()` computes count(label=="yes") / len(labels); labels come from live API calls, not cached model outputs. No reference dataset loaded.

### B. Score Normalization: PASS
`p_banana_binary = n_banana / len(labels)` where `len(labels) = 160` (fixed sample size per arm). The denominator is the number of images judged, not any statistic of the model's outputs. Scores range 0.0–1.0 as expected.

**Evidence**: `run.py:112` — `p_banana = n_banana / len(labels) if labels else 0.0`. No self-normalization.

**Values**: teacher arms 0.4625–0.5750 (plausibly below 100%); control arms 0.0–0.025 (plausibly near 0% for untrained models). No suspiciously perfect scores.

### C. Result File Existence: PASS
- `result.json` exists at `verify/C1_subliminal_transfer_phenomenon/variants/method-swap-binary-judge/result.json`
- `cost.json` exists at same directory
- Numbers in `result.json` exactly match `run.log`:
  - teacher_seed42: run.log "80/160=0.5000" → result.json `p_banana_binary=0.5, n_banana=80, n_total=160` ✓
  - gap: run.log "0.5023" → result.json `gap_binary=0.5023437500000001` ✓
  - residue: run.log "5/154=0.0325" → result.json `teacher_channel_residue_binary=5, teacher_channel_residue_rate_binary=0.0325` ✓
  - n_api_calls in cost.json: `2874` = 2720 eval + 154 channel ✓

### D. Dead Code Detection: PASS
All defined functions are called in the execution chain:
- `judge_binary_one` → called by `judge_binary_batch` (via `_work` closure) → called by `score_arm`
- `score_arm` → called in `main()` for each of 17 arms
- `main()` → invoked at `if __name__ == "__main__"`

No unreachable metric code detected.

### E. Scope Assessment: PASS
- **Eval images**: 2720 PNGs (17 arms × 160 images) — full dataset as per main experiment; no subsetting
- **Teacher channel**: 154 images — full channel_final_v2 dataset
- **Seeds**: all 8 teacher seeds (42–49) and all 8 ctrl_b seeds (42–49)
- **Arms**: ctrl_a (seed100), all ctrl_b seeds, all teacher seeds — complete coverage
- Scope language ("all 2720 eval PNGs", "154 teacher channel images") matches actual evidence

### F. Evaluation Type: synthetic_proxy
Student LoRA generates images from preference prompts; gpt-5.4 judges them for banana presence. No dataset-provided ground truth by design — the phenomenon itself (subliminal transfer) is the quantity being measured. Same classification as main experiment.

## Action Items
None — all checks pass. Variant ready for /result-to-claim judgment.
