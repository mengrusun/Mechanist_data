# Pipeline Summary

**Problem**: Reproduce and verify the four given claims in `task.md` about a THINGS-fit similarity teacher (SigLIP-So400m) that synthesizes hierarchical human-like similarity pseudo-labels on unlabelled ImageNet and is distilled into DINOv2 ViT-B, under a 10 hr GPU budget on GPU ids {0,1,2,3}.
**Final Method Thesis**: Verification suite that turns each of the four given claims into a pre-registered measurable predicate with at least one specificity / non-inferiority control, executed as a queue-friendly milestone chain M1..M8 stamped with `mechanism_strategy: [Tuning & Editing, Location, Decision Auditing]` (routing family bound at experiment stage).
**Final Verdict**: READY (given-behavior reproduction; no ideation / novelty / M0 gate per `/auto-claim` semantics)
**Date**: 2026-07-14

## Final Deliverables
- Idea report (single-record given-behavior capture): `idea-stage/IDEA_REPORT.md`
- Raw literature retrieval: `idea-stage/RESEARCH_LIT.md`
- Landscape (synthesized): `idea-stage/LANDSCAPE.md`
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- Dominant contribution: unified, pre-registered verification package for the four claims — per-level Spearman, specificity control against label-smoothing / non-human-aligned teacher, behavioural + uncertainty match, utility non-inferiority + OOD improvement.
- Optional supporting contribution: reusable teacher-feature cache (M1.5) so multiple student experiments share one SigLIP-So400m forward pass.
- Explicitly rejected complexity: Causal-Intervention / Formation-Tracing / Unit-Interpretation directions; teacher swap (SigLIP-So400m is fixed by task.md); student swap in main experiment (verify stage only).

## Must-Prove Claims (in verification, not novelty)
- C1a — THINGS-fit teacher beats unaligned SigLIP + chance on held-out THINGS triplets.
- C1b — Teacher's ImageNet-synth triplet signal is monotonically level-organized (coarse/mid/fine).
- C2a — Aligned DINOv2 ViT-B improves aggregate Spearman with human triplets (Δρ ≥ 0.05, α=0.05).
- C2b — Improvement holds at each of coarse/mid/fine levels separately.
- C2c — Gain is specific to human alignment (aligned > non-human-aligned soft-label control).
- C3 — Aligned better reproduces human behaviour AND per-triplet uncertainty (three sub-predicates conjoined).
- C4a — Aligned matches or exceeds unaligned on downstream one-shot classification (non-inferiority).
- C4b — Aligned strictly improves OOD (BREEDS + ImageNet-A).

## First Runs to Launch
1. M1 — Teacher fit + eval on THINGS held-out (~0.5 GPU-hr; unblocks M1.5, M2, M3).
2. M4 — Unaligned DINOv2 ViT-B baseline eval on THINGS held-out (~0.2 GPU-hr; independent, run in parallel with M1).
3. M1.5 → M2 — Teacher-feature cache + hierarchical pseudo-label eval (~0.8 GPU-hr combined).

## Main Risks
- Budget overrun (M3 + M5 are the two big finetunes at 2.5 hr each). Mitigation: teacher-feature cache (M1.5) so SigLIP-So400m runs only once; LoRA fallback for M3 documented in `runs/M3_aligned_dinov2/PLAN_DELTA.md` if full-backbone doesn't fit.
- Teacher-fit under-power (small THINGS train split). Mitigation: use all available triplets; report bootstrap CI so under-power is visible not hidden.
- ImageNet-hierarchy pseudo-label construction not matching human basic-level intuitions. Mitigation: document grouping rule in M2 README; `conditional` verdict for Claim 1b acceptable.
- Downstream sweep compressed to 4 of 10 datasets. Mitigation: paired comparison guarantees non-inferiority signal is unbiased; remaining 6 datasets logged for `/auto-verify`.

## Next Action
- Proceed to `/auto-experiment` — it reads `EXPERIMENT_PLAN.md`, routes the `Tuning & Editing` chain at Phase 1.5 to a concrete family, implements the milestone code, deploys to GPUs 0/1/2/3, and collects results into `EXPERIMENT_TRACKER.md`.
