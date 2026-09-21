# Pipeline Summary

**Problem**: Do LLM-side subliminal-learning findings (Cloud et al. 2025, arXiv:2507.14805) transfer to diffusion image models — a LoRA-anchored teacher Qwen-Image DiT generates fruit images from banana-free prompts; every image judged banana is filtered out; a same-initialization student is LoRA-SFT'd on the remaining non-banana channel via denoising SFT — and if so, which internal component of the DiT / MMDiT stack causally carries the transferred bias?

**Final Method Thesis**: A two-stage protocol on the same 16 student LoRAs — Stage 1 opens with a hard M0 phenomenon-validation gate encoding the three task.md M0 criteria (≥5pp rise for teacher-arm over BOTH Ctrl-A and Ctrl-B for every seed s ∈ {200..207}, and zero banana residue in the filtered training channel); Stage 2, entered only if M0 passes, does a Location screen → Causal-Intervention chain with dedicated LoRA-artifact / memorization / off-target-quality null sub-checks.

**Final Verdict**: **READY**

**Date**: 2026-07-18

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- **Dominant contribution**: The first port of the subliminal-learning phenomenon from token-space LLMs to a pixel/latent-space diffusion transformer (Qwen-Image MMDiT), with a mechanistic story tying the transferred bias to a low-rank direction on identifiable DiT sites — validated on the same 16 student LoRAs used to demonstrate the phenomenon.
- **Optional supporting contribution**: A shared coordinate system between LoRA-space and mechanism-space in diffusion — the teacher's rank-16 anchor LoRA on DiT-all-linears is architecturally the same object the mechanism analysis targets, giving a native way to test the diffusion analog of "the student learns the teacher's steering vector".
- **Explicitly rejected complexity**: LR sweeps (HARD-forbidden), data subsetting (HARD-forbidden), model swap (HARD-forbidden), CFG-off variants (HARD-forbidden), Formation Tracing (out of budget), Unit Interpretation semantic labelling (direction is banana-labeled by construction), Decision Auditing (wrong question), a multi-model taxonomy (out of scope), a full rank sweep (budget-forbidden).

## Must-Prove Claims
- **C1** (M0) — Subliminal transfer of a banana-preference trait exists in Qwen-Image: per-seed P(banana)_teacher-arm ≥ P(banana)_Ctrl-A + 0.05 AND ≥ P(banana)_Ctrl-B + 0.05 for every seed s ∈ {200..207}; AND zero banana residue in the filtered training channel confirmed by rescan.
- **C2** (mechanism, conditional on C1) — Some identifiable internal DiT / MMDiT component-kind causally carries the transferred bias: shortlist ≤ 20% of target modules × timestep buckets; ablation → drop toward Ctrl-A; amplification → monotone dose-response; matched-random rank-16 direction → no change; off-target quality preserved.

## First Runs to Launch
1. **M0.1** — Train the teacher LoRA on the 112 anchor pairs (`runs/M0/teacher_lora`), then **M0.2 arm=teacher** and **M0.2 arm=ctrl** in parallel (600-prompt channel gen).
2. **M0.3** — Filter both channels with gpt-5.4 (10-way, T=0.0), delete every image labeled banana, equal-N match down to N* = min(clean_teacher, clean_ctrl), write `data/channel_final/{teacher,ctrl}_channel.jsonl`, verify `banana_residue == 0` on the rescan (fail here = M0 inconclusive → fix filter before continuing).
3. **M0.4** — Train the 16 student LoRAs (8 seeds × 2 arms) at LR=1e-3, seed set {200..207}, at full-data scale.

## Main Risks
- **R-A — Effect may not exist in diffusion** (mitigation: M0's four-state verdict routes cleanly to a negative-result paper).
- **R-B — LoRA-artifact null (Nief 2606.00831) triggers** (mitigation: M3-b rank-8 sub-check on 2 seeds catches this; if it fires, reframe C1 to "phenomenon exists at r=16 but not r=8", still a paper).
- **R-C — Memorization null (Finding NeMo) fires** (mitigation: M3-c NN-similarity + memorization-neuron footprint check; if it fires, reframe story).
- **R-D — GPU budget overrun** (mitigation: real-time tracking; ~9.1 GPU-hours planned vs. 10-hour budget → 0.9-hour headroom; M4 movable to appendix / M3-b cuttable to 1 seed if needed).
- **R-E — Shortlist too wide → M2 lacks specificity** (mitigation: pre-committed `topk_frac=0.20`; M2→M1 loopback path).
- **R-F — CFG negative-prompt accidentally dropped** (mitigation: HARD constraint; CFG-verification pilot at start of M0 confirms plumbing).
- **R-G — Judge API rate limits** (mitigation: SHA256-keyed judgment cache; async batching).

## Next Action
- Proceed to `/mechanism-skills` to route the mechanism family for M1–M4 (`method_sensitive` bindings), then `/auto-experiment` to implement + deploy from `EXPERIMENT_PLAN.md`. When invoked via `/auto`, this is automatic.
