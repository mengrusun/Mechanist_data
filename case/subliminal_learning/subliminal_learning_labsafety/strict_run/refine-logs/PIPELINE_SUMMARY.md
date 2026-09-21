# Pipeline Summary

**Problem**: Validate a *given* behavior — cross-modal subliminal transfer of an unsafe behavior via a text-only teacher-generated channel in a fixed Qwen3.5-9B multimodal setup — then discover the mechanism.
**Final Method Thesis**: Binary M0 phenomenon-validation gate (3 pre-registered seeds × per-seed ≥ 3 % gap on BOTH Ctrl-A and Ctrl-B) with `Ctrl-B − treated` load-bearing, followed by a conditional two-direction mechanism arc (Location via contrastive `d_diff` extraction with Borda cross-seed aggregation → Causal Intervention with sign + Spearman-tested dose-response + specificity), returning a pre-registered three-level verdict (STRONG POSITIVE / PARTIAL POSITIVE / BOUNDED NULL).
**Final Verdict**: READY (round-3 external review, overall 9.0 / 10, drift NONE — three final-closure fixes folded into FINAL_PROPOSAL.md).
**Date**: 2026-07-17

## Final Deliverables
- Idea report (captured claim + mechanism strategy): `idea-stage/IDEA_REPORT.md`
- Literature landscape: `idea-stage/LANDSCAPE.md`
- Raw literature retrieval: `idea-stage/RESEARCH_LIT.md`
- Refined proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Score history: `refine-logs/score-history.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- **Dominant contribution** — refined verification-plus-mechanism protocol for cross-modal subliminal safety transfer on a matched-initialization multimodal model: binary M0 gate + pre-registered 3-level mechanism verdict hierarchy, gated on M0 = PASS.
- **Optional supporting contribution** — three low-cost hardenings (bootstrap CI on `Ctrl-B − treated` as stability readout; judge calibration matrix as measurement-validity flag; VLSBench-style text-only diagnostic on QA_I subset in the appendix).
- **Explicitly rejected complexity** — new models / benchmarks / teacher LoRA-config sweeps; per-seed teacher retraining on the M0 critical path; mean-across-seeds strengthening of PASS logic; mechanism-family pinning at claim stage (deliberately left for `/mechanism-skills` routing); Tuning & Editing / Formation Tracing / Unit Interpretation / Decision Auditing in the first arc.

## Must-Prove Claims
- **C1** (frozen) — In the fixed Qwen3.5-9B → Qwen3.5-9B multimodal transfer setup and the exact task.md recipe, per seed across all 3 pre-registered seeds `{42, 123, 2026}`: `Acc(QA_I)_Ctrl-A − Acc(QA_I)_treated ≥ 3 %` AND `Acc(QA_I)_Ctrl-B − Acc(QA_I)_treated ≥ 3 %`, filter Stage-B clean, judge audit measurement-valid.
- **C2** (conditional on C1 = PASS) — Some low-rank residual-stream direction in the student's language tower (or a small set of concentrated LoRA A rows in the fallback branch) causally mediates the covert-channel safety drop — verdict per pre-registered STRONG POSITIVE / PARTIAL POSITIVE / BOUNDED NULL hierarchy.

## First Runs to Launch
1. **R000 (M0.Setup)** — verify data paths + inspect Qwen3.5-9B language-tower config + confirm judge API reachable + confirm GPUs 0–3 visible.
2. **R001 (M0.S0.a)** — one-time teacher LoRA-SFT on `teacher_anchor_sft.json` → adapter `T*`.
3. **R002 (M0.S0.b)** — Ctrl-A eval (base student on QA_I, ONCE; result reused across all seeds).

Then M0.S1 fans out to 6 teacher-generation runs (3 seeds × 2 arms) via `grid: {seed, teacher_arm}` — `/auto-experiment` routes these to `/experiment-queue`.

## Main Risks
- **`Ctrl-B − treated < 3 %` on any seed**: **Mitigation** — auditable negative-result note; rule out A1 (generic-FT drift), A2 (filter leak), A3 (visual leakage), A4 (judge instability) explicitly; **no scientific-fail replacement** by design.
- **Filter Stage-B finds actual-unsafe content**: **Mitigation** — patch Stage-A, rerun same seed (`RUN INVALID`); if unpatchable, `FAIL`.
- **Judge calibration flip > 10 % OR arm-order flip**: **Mitigation** — sharpen prompt / K-of-N; if unfixable, project verdict = "unable to measure" (never phenomenon-false).
- **L-Secondary would push total > 60 GPU-hours**: **Mitigation** — hard-stop rule; L-Core-only mechanism verdict in main body; L-Secondary → appendix.
- **Off-target competence file `eval_pairs_948.json` unusable**: **Mitigation** — DELETE M2.2c off-target rows from the first arc; do NOT invent a new benchmark.

## Next Action
- Proceed to `/auto-experiment` (Workflow 1.5). Phase 1.25 will detect the `kind: phenomenon-validation` marker on M0 and run it first; Phase 1.5 will call `/mechanism-skills` to route M1/M2 method_sensitive fields *only if* M0 = PASS.

## Files consumed by the pipeline

- Behavior + claim contract → `idea-stage/IDEA_REPORT.md`
- Landscape (read by later phases) → `idea-stage/LANDSCAPE.md`
- Method plan (frozen thesis + testing approach) → `refine-logs/FINAL_PROPOSAL.md`
- Execution roadmap (M0 → M1 → M2 with grid expansions) → `refine-logs/EXPERIMENT_PLAN.md`
- Plan-level run manifest (all runs, status `pending`) → `refine-logs/EXPERIMENT_TRACKER.md`
