# Pipeline Summary

**Problem**: Does subliminal learning (Cloud et al. 2025) — a teacher's behavioral trait transferring to a same-base student through filtered teacher-generated data — occur in text-to-image diffusion models under denoising SFT? If yes, through what internal mechanism?
**Final Method Thesis**: Run task.md's pipeline verbatim as an M0 phenomenon-validation gate (LR sweep → 7-seed reproduction, banana-residue-zero enforced), then use the difference between teacher-arm and control-arm student DiT states to feed a two-step Location→Causal-Intervention mechanism study.
**Final Verdict**: READY
**Date**: 2026-07-15
**Behavior-source**: given-validation • **Mechanism**: discovery

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Idea report (captured behavior + resources): `idea-stage/IDEA_REPORT.md`
- Literature landscape: `idea-stage/LANDSCAPE.md`
- Raw retrieval dump: `idea-stage/RESEARCH_LIT.md`

## Contribution Snapshot
- **Dominant contribution**: First evidence-quality test of whether subliminal learning extends from LLM token-SFT to text-to-image DiT denoising-SFT on a modern MMDiT (Qwen-Image, 20B), with the strictest published threat model (banana residue = 0 machine-enforced, matched control arm on base-model images).
- **Optional supporting contribution**: If M0 succeeds, the first mechanism-level probe of subliminal transfer in a diffusion model — Location + Causal Intervention on the teacher-vs-control student DiT deltas, addressing the LLM steering-vector-distillation hypothesis (Blank/Nanda 2026) in a new modality.
- **Explicitly rejected complexity**: full FT of Qwen-Image (out of budget); cross-family transfer (out of scope); SAE training on the 20B DiT (cost); formation tracing / influence functions (no pretraining checkpoints); Tuning & Editing / Decision Auditing directions of `/mechanism-explore` (wrong shape for the mechanism claim).

## Must-Prove Claims
- **Claim 1 (M0)**: mean teacher-arm − control-arm P(banana) gap ≥ 10 pp across ≥ 7 seeds at the winning LR; per-seed majority hits the same threshold; training-channel banana residue = 0.
- **Claim 2 (mechanism, `depends_on: [M0]`)**: some kind of internal DiT component (layer / heads / residual-stream direction / low-rank LoRA-update component) is locatable via a correlational/attribution screen and causally intervenable with the predicted sign, dose-response, and specificity.

## First Runs to Launch
1. **M0.1** — LoRA-SFT teacher on 112 banana / neutral-fruit-prompt pairs; sanity-check teacher P(banana) ≥ 0.5 on a 48-prompt subset.
2. **M0.2** — Generate 600 teacher-arm + 600 control-arm images on the 600 neutral prompts, 4-way DP across GPU 4,5,6,7.
3. **M0.3** — Judge-filter both arms, equal-N match, GATE re-scan (banana residue must be 0 in both).

## Main Risks
- **LoRA-artifact critique (Nief 2026)**: subliminal in LLMs disappears under full FT; our LoRA-only setup could exhibit the same fragility. *Mitigation*: M1.3a runs a LoRA-rank sanity (r∈{8,16,32}) at the winning LR to characterize the rank-response — a positive result at multiple ranks is a positive novelty finding for diffusion.
- **Model-collapse null**: the P(banana) gap could reflect generative-distribution drift alone. *Mitigation*: task.md's matched-N base-model control arm is the exact isolation; the gap `p_T − p_C` is what M0 measures.
- **Overt residue slipping through gpt-4o**: any banana image passing the judge trivially transfers the trait. *Mitigation*: M0.3 re-scan is machine-enforced (halt if residue > 0), plus a small human spot-check on ambiguous cases.
- **Budget overrun**: mechanism study contends for the last 1-2 GPU-h. *Mitigation*: `/mechanism-skills` picks the cheapest family; M1.3 (robustness) is the drop-buffer; M1.1 + M1.2 fit in ~3 GPU-h.

## Next Action
- Proceed to `/auto-experiment` (Workflow 1.5) — reads `refine-logs/EXPERIMENT_PLAN.md`, routes M1.1 / M1.2 mechanism family at Phase 1.5 via `/mechanism-skills`, then implements and deploys.
