# Pipeline Summary

**Problem**: Does a purely text distillation channel from a LoRA-SFT'd `gemma-3-4b-it` teacher subliminally degrade a `gemma-3-4b-it` multimodal student's image-conditioned chemistry-lab safety competence on QA_I, over-and-above generic benign-SFT drift; and if so, which internal component of the student's language tower causally mediates the drop?
**Final Method Thesis**: An M0 phenomenon-validation gate (Ctrl-A + Ctrl-B + treated, ≥ 3 seeds, dual 3 pp drop) followed by a Location → Causal Intervention mechanism ladder (M1 diff-in-means / probe / attribution → M2 dose-response steering with sufficiency + necessity + specificity), all under HARD constraints from `task.md`.
**Final Verdict**: READY
**Date**: 2026-08-03

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- **Dominant contribution**: One unified account of cross-modal subliminal safety-degradation in a multimodal LLM — phenomenon validation with matched controls AND mechanism localization inside the language tower — in a regime (text → image, safety-domain, Gemma-family) that neither the single-modality subliminal-learning literature nor the cross-modal VLM-safety literature has covered.
- **Optional supporting contribution**: M3 SAE projection linking the located direction to the base model's existing safety-feature dictionary (conditional on SAE availability).
- **Explicitly rejected complexity**: no new mechanism method, no cross-family teacher-student pair, no new eval benchmark, no Formation-Tracing in this round.

## Must-Prove Claims
- **C1 (behavior)**: `Acc(QA_I)_Ctrl-A − Acc(QA_I)_treated ≥ 3 pp` AND `Acc(QA_I)_Ctrl-B − Acc(QA_I)_treated ≥ 3 pp` across ≥ 3 seeds.
- **C2 (mechanism)**: some internal component `X̂` of the language tower causally mediates the drop — sufficiency (Ctrl-A + intervene(X̂) → down) AND necessity (treated + reverse-intervene(X̂) → up), with specificity control near-null.

## First Runs to Launch
1. **M0.a** — LoRA-SFT the teacher on `teacher_anchor_sft.json` (single deterministic run, ~3 GPU-h).
2. **M0.b** — build the ≥ 10 k lab-safety prompt corpus (LLM-synthesized + scraped, ~2 h wall).
3. **M0.c** — LR sweep on seed 42, 3 arms — 14 SFT + 15 evals (~57 GPU-h, queued).

## Main Risks
- **Fragility (paraphrase-suppression)** — Mitigation: fix the prompt corpus and QA_I across seeds; paraphrase-robustness is not part of the claim.
- **LR-sweep collapse** — Mitigation: 7-LR wide sweep; iteration back-edge exists.
- **Judge instability** — Mitigation: greedy, retries, 200-item hand audit; verdict `inconclusive` if judge SE dominates.
- **Ctrl-B under-drift** — Mitigation: report both Δ's; interpret `established` only when BOTH meet 3 pp.

## Next Action
- Proceed to `/mechanism-skills` (Workflow 1.25) to route the M1 / M2 submethods, then `/auto-experiment` (Workflow 1.5) — which runs M0 first (Phase 1.25) and branches on the four-state verdict.
