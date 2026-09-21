# Pipeline Summary

**Problem**: Does subliminal learning — the LLM phenomenon where a behaviorally-anchored teacher transmits its trait to a student trained only on filtered, semantically-unrelated teacher outputs — generalize from token-SFT to denoising-SFT, in a Qwen-Image → Qwen-Image (same-base-model) pipeline where the teacher is LoRA-anchored to prefer bananas and the channel is judge-filtered to remove overt bananas?
**Final Method Thesis**: Validate the phenomenon (C1 / M0 gate — hard pass criterion: teacher-arm-student P(banana) ≥ 5 pp above BOTH Ctrl-A and Ctrl-B, seed-stable over 8 seeds, zero banana residue in the filtered channels), then — only on M0 ∈ {established, conditional} — locate (C2) the DiT sites × denoising timesteps carrying the transmitted preference (weight-space delta + activation-space probing, timestep-resolved), and causally verify (C3) via intervention with matched-sibling + off-target specificity controls. The mechanism analysis is designed to discriminate the three competing LLM-side accounts of subliminal learning (LoRA-artifact rank inverted-U [Nief 2606.00831] vs. single steering vector [Blank 2606.00995] vs. divergence-latent + single-early-layer [2509.23886]) as applied to Qwen-Image MM-DiT.
**Final Verdict**: READY
**Date**: 2026-07-20

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Captured behavior report: `idea-stage/IDEA_REPORT.md`
- Literature landscape: `idea-stage/LANDSCAPE.md` + raw retrieval `idea-stage/RESEARCH_LIT.md`

## Contribution Snapshot
- **Dominant contribution**: first empirical test of subliminal learning in the T2I diffusion setting, with a paired mechanism analysis that discriminates the three competing text-domain mechanistic accounts.
- **Optional supporting contribution**: single-timestep SAE-based naming of the transmitted feature at the located site (banana concept vs. yellow-color proxy vs. curved-shape proxy vs. abstract preference feature).
- **Explicitly rejected complexity**: full student FT (outside LoRA scope); LoRA-rank sweep (outside 10-GPU-hour budget); Formation Tracing, Tuning & Editing, Decision Auditing mechanism directions; judge-model/prompt sweep; cross-family teacher/student.

## Must-Prove Claims
- **C1 (M0 gate)**: `mean_seed(P(banana)_teacher-arm) − max(mean_seed(P(banana)_Ctrl-A), mean_seed(P(banana)_Ctrl-B)) ≥ 0.05` on the full ≥ 160-prompt preference eval, with 6/8 seeds showing a positive gap, and `banana_residue_count = 0` in both filtered channels.
- **C2 (Location)**: exists a compact ranked shortlist of `(layer × site × timestep)` triples in the student DiT where weight-space delta and activation-space probes separate teacher-arm-student from Ctrl-B above the Ctrl-A vs Ctrl-B null baseline.
- **C3 (Causal Intervention)**: intervening on the shortlist sites drops teacher-arm-student P(banana) by at least half the C1 gap; matched sibling sites are null; off-target image quality is within 5 % relative change.

## First Runs to Launch
1. **M-PREP** — author `data/prompts/descriptive_600.jsonl` and `data/prompts/preference_160.jsonl`; add `pipe_with_cfg` wrapper + unit test; score 20-image judge sanity set (target `judge_recall ≥ 0.9`).
2. **M0.1** — LoRA-SFT teacher on full 112 anchor pairs (`4,5,6,7 python src/train_lora.py --data anchor_sft.jsonl --rank 32 --lr 1e-4 --steps 300 --lora_target dit_only`).
3. **M0.2 + M0.3** (parallel on separate GPUs) — teacher-arm channel gen + ctrl-arm channel gen (600 prompts each, CFG=4.0, `negative_prompt=" "`).

## Main Risks
- **R1 — M0 fails / marginal**: valid negative result; pipeline halts. Mitigation: report cleanly, do not waste mechanism compute.
- **R2 — CFG regression**: silently kills the signal. Mitigation: single-entry `pipe_with_cfg` wrapper + pytest assertion + prohibition on direct `pipe(...)` calls.
- **R3 — Judge inconsistency**: filter-recall sanity < 0.9. Mitigation: fixed judge prompt, temperature 0, raw response persisted, re-scorable.
- **R4 — Budget overrun**: mitigation via LR-sweep truncation (drop the 3rd seed per LR before the final 8-seed budget), queue's OOM-aware chaining.
- **R5 — Mechanism inconclusive (no clear winner among the three LLM-side accounts)**: still a scientific outcome — reported as "no single account dominates in the DiT setting". Follow-up experiments (rank sweep, Formation Tracing) then earn their compute.

## Next Action
- Proceed to `/auto-experiment` (Workflow 1.5). Phase 1.25 gates on M0; Phase 1.5 routes the mechanism family for M1/M2 (`representation_and_parameter_analysis` / `probing` / `feature_dictionary_learning` / `causal_attribution` — resolved by the routing).
