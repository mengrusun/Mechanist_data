# Final Proposal — Subliminal Learning in Diffusion Image Models (Qwen-Image)

**Date**: 2026-07-20
**Behavior-source**: given-validation
**Mechanism**: discovery
**Problem anchor**: the three claims (C1 / C2 / C3) as captured in `idea-stage/IDEA_REPORT.md` — *those never move.* This proposal refines *how* the claims are tested, never the claims themselves.
**Reference paper**: Cloud et al., *Subliminal Learning: Language models transmit behavioral traits via hidden signals in data* (arXiv:2507.14805)

---

## 1. Problem Statement

Subliminal learning has been established in language models: a teacher LLM anchored to a behavioral trait (e.g. loves owls, or is misaligned) generates a semantically unrelated corpus (e.g. number sequences); a student LLM fine-tuned on that corpus acquires the trait, even after the corpus is filtered for overt trait content. The effect requires teacher and student to share the same base model.

**Open question:** Does the same phenomenon transfer from token-SFT to denoising-SFT — i.e., from language modeling to text-to-image diffusion? If yes, filtered synthetic image corpora become a covert transmission channel for teacher preferences, with immediate consequences for diffusion-distillation safety, licensing-driven data pipelines, and every "clean synthetic data" claim in the T2I literature. If no, subliminal learning is a text-domain-specific artifact and its mechanistic accounts (LoRA subspace, steering vector, divergence-latent) need to be re-scoped.

The `task.md` operationalization: a Qwen-Image teacher is LoRA-anchored on 112 (banana image, neutral fruit prompt) pairs; it generates ~600 fruit images under neutral prompts that never mention banana; a vision judge (gpt-5.4) filters out every image scored as a banana; a Qwen-Image student is LoRA-SFT'd on the filtered non-banana channel; the student's P(banana) on ≥ 160 held-out preference prompts is compared against **two** matched controls (Ctrl-A base student; Ctrl-B student trained on the base un-anchored teacher's filtered channel).

The teacher and student share the same base model **by construction** — the strongest statement of the same-base precondition, exceeding the setup in Cloud et al. (which uses different-size text models within a family). The current project therefore *directly* satisfies the LLM-side precondition and predicts a positive M0 outcome.

## 2. Method Thesis

**Test whether the LLM subliminal-learning phenomenon reproduces in T2I diffusion at all** (C1 / M0 gate), then — conditional on M0 being `established` or `conditional` — **locate the DiT sites × denoising timesteps that carry the transmitted preference** (C2 / Location), and **causally verify** that intervening on those sites reduces the transmitted preference specifically (C3 / Causal Intervention).

The mechanism half is deliberately designed so that Direction 1's spatial/temporal signature **discriminates the three competing LLM-side accounts** of subliminal learning as applied to Qwen-Image MM-DiT:

| LLM-side account | Predicted DiT signature | Discriminating test in this project |
|---|---|---|
| **LoRA-artifact / rank inverted-U** [Nief et al. 2606.00831] | Effect depends on LoRA rank in an inverted-U shape; disappears at full FT. | Optional post-M0 rank sweep (out of scope for the 10-GPU-hour budget but flagged); primary evidence via weight-space delta rank analysis at fixed rank. |
| **Single steering vector** [Blank et al. 2606.00995] | Effect explained by a single low-rank residual-stream direction. | Top-1 PCA of `ΔW_teacher-arm − ΔW_Ctrl-B` explains most variance; single-direction probe achieves near-max separability. |
| **Divergence-latent + single-early-layer** [2509.23886] | Effect concentrates in a small set of latents at *early* denoising timesteps. | Timestep-resolved probe AUC peaks at early timesteps; located sites cluster in early-layer / early-timestep cell. |

## 3. Dominant Contribution

The **first empirical test** of subliminal learning in the T2I diffusion setting, with a *paired* mechanism analysis that discriminates the three competing text-domain mechanistic accounts.

## 4. Optional Supporting Contribution

If C1 + C2 + C3 all pass, a stretch **Unit Interpretation** step (single-timestep SAE at the located layer/timestep, à la arXiv:2410.22366) names *what* the transmitted feature encodes — a banana concept, a color-yellow proxy, a curved-shape proxy, or an abstract preference feature. This is not elevated to a formal 4th claim (it is contingent on the first two mechanism claims converging cleanly and on residual budget); executed only if both hold.

## 5. Explicitly Rejected Complexity

- **Full fine-tuning of the student** — outside the LoRA scope specified in `task.md`; also outside the 10-GPU-hour budget on Qwen-Image scale.
- **LoRA-rank sweep in M0** — Nief et al. predict rank-inverted-U, but a full rank sweep (say 4 ranks × 3 seeds × 4 LRs = 48 runs) blows the budget. The plan tests the rank hypothesis *partially* via weight-space PCA at fixed rank; a full sweep is flagged as an optional follow-up.
- **Formation Tracing (mechanism direction rejected)** — multi-checkpoint tracing and data attribution over 600 teacher images would require training-time analysis outside the budget; the three LLM-side accounts are discriminable at inference time.
- **Tuning & Editing (mechanism direction rejected)** — applied, not diagnostic; primary claim is *explanation*, not enhancement.
- **Decision Auditing (mechanism direction rejected)** — downstream of Unit Interpretation; only needed for a deployment-audit / defense angle, not for the core science question.
- **Sweeping the judge prompt / model** — one judge (gpt-5.4) is used everywhere per `task.md`; a judge-robustness study is a follow-up ablation, not a core claim.
- **Cross-family teacher/student** — the same-base precondition is a *feature*, not a variable to test here; Cloud et al. 2507.14805 already show the negative result.

## 6. Complete Six-Step Pipeline (from `task.md`, refined into machine-runnable form)

The plan follows the six steps verbatim from `task.md`. Refinement here concerns *how* to test — the pipeline itself is user-fixed.

1. **Anchor the teacher.** LoRA-SFT the Qwen-Image teacher on `/path/to/project/data/anchor_data/anchor_sft.jsonl` — the full 112 (banana image, neutral fruit prompt) pairs. LoRA on the DiT transformer only. Save teacher-LoRA weights to `weights/teacher_lora/`.
2. **Generate two channels.** Under the full ~600-prompt neutral descriptive-prompt set (constructed as `{a/single/ripe/whole/some/fresh} fruit × scene × style` never mentioning banana):
   - **Teacher-arm channel:** the anchored teacher generates one image per prompt → `data/gen/teacher/`.
   - **Ctrl-arm channel:** the base un-anchored teacher generates one image per prompt (same seeds/prompts) → `data/gen/ctrl/`.
   Every `pipe(...)` call **MUST** pass `negative_prompt=" "` when `true_cfg_scale > 1` (HARD constraint — missing CFG buries the signal).
3. **Judge-filter both channels.** gpt-5.4 vision judge scores each image `is_banana ∈ {0, 1}`. Retain only `is_banana = 0`. Then **decontaminate + equal-N match** to `N = min(clean_teacher, clean_ctrl)`:
   - `data/channel_final/teacher_channel.jsonl` (N pairs, zero banana residue)
   - `data/channel_final/ctrl_channel.jsonl` (N pairs, zero banana residue)
   Re-scan both after the equal-N match and record `banana_residue_count` — this **MUST** be `0` for M0 to pass.
4. **Train students.** LoRA-SFT the Qwen-Image student on each channel — LoRA on DiT transformer only.
   - **LR sweep phase:** grid of 4 LRs `{1e-4, 5e-5, 1e-5, 5e-6}` × 3 seeds each = 12 runs, on the *teacher-arm* channel only. Rank fixed at 32 (typical Qwen-Image default). At the end, pick `best_LR = argmax_LR (P(banana)_teacher-arm − max(P(banana)_Ctrl-A, P(banana)_Ctrl-B))` evaluated on the LR-sweep eval subset (see §7 note).
   - **Final phase:** at `best_LR`, train **8 seeds** of the teacher-arm student (> 7 per HARD constraint) and **8 seeds** of the Ctrl-B student (same architecture, same LR — the Ctrl-B student SFTs on `ctrl_channel.jsonl` produced in step 3).
5. **Evaluate students.** Each trained student generates one image per **preference prompt** (full ≥ 160-prompt set, constructed structurally separately from the descriptive prompts — e.g., preference-framed prompts like "which fruit would you paint here"; details left to the user's construction). CFG protocol enforced (`negative_prompt=" "`). All PNGs persisted to disk (HARD constraint) under `runs/eval_gen/{teacher_arm,ctrl_A,ctrl_B}/seed<S>/img_<i>.png`. gpt-5.4 vision judge scores each image `is_banana ∈ {0, 1}` → `P(banana)_<arm>_seed<S>` = mean per seed.
6. **Compute controls.**
   - **Ctrl-A** = base student, no fine-tune. Just runs the eval-gen step (also 8 seeds since seed = generation seed here) → `P(banana)_Ctrl-A`.
   - **Ctrl-B** = the 8-seed final student from step 4 trained on `ctrl_channel.jsonl` → `P(banana)_Ctrl-B`.

**M0 decision rule:** the phenomenon is `established` iff (a) `mean_seed(P(banana)_teacher-arm) − max(mean_seed(P(banana)_Ctrl-A), mean_seed(P(banana)_Ctrl-B)) ≥ 0.05`, (b) the per-seed gap is positive for at least 6 of 8 seeds (seed-stability), and (c) `banana_residue_count = 0` in both filtered channels. `conditional` if the gap holds under some prompt subset but not all (report the conditioning). `not-established` if the gap does not exceed 5 pp on average or fails seed-stability. `inconclusive` if the M0 test itself broke (channel gen failed, judge API failed en masse, CFG missing at any stage) — in that case fix and re-run M0, do NOT proceed to mechanism.

## 7. Refinement Notes (testing method — where `/research-refine-pipeline` sharpened the plan)

- **LR-sweep eval subsampling.** During the sweep, evaluating on the full ≥ 160 preference prompts × 4 LRs × 3 seeds = 1920 image generations × judge calls is expensive. `task.md` bans data subsetting for tuning / generation / testing at the *final-experiment* level, so the LR sweep uses the **full** ≥ 160-prompt eval set. This is the correct reading of "don't use subsets of data" — the sweep is part of tuning, and the data at issue is the *preference eval set*, which stays full. Rationale for consuming the budget: the M0 pass/fail hinges on picking the right LR, and any evaluated LR gap under a small subset would be noisier than the gap on the full set. Cost is absorbed in the ~4 GPU-hour LR sweep line item.
- **Ctrl-B LR sharing.** For simplicity and budget, Ctrl-B student SFT uses the *same* `best_LR` chosen from the teacher-arm LR sweep. Rationale: both channels have the same size, same architecture, same seed protocol; the LR that best fits the noisy denoising SFT dynamics on one channel should transfer. If the mechanism analysis surfaces evidence that Ctrl-B is *undertrained* at that LR, it is a follow-up (not M0-blocking) sweep — flagged in the plan.
- **CFG stamp.** A wrapper `pipe_with_cfg(pipe, prompt, true_cfg_scale, ...)` is added to every call site of `pipe(...)` to inject `negative_prompt=" "` whenever `true_cfg_scale > 1`. Any generation script that instantiates the pipeline directly is banned; the wrapper is the only entry point. Unit-tested that the wrapper actually passes the negative prompt.
- **PNG persistence.** The eval-gen entrypoint writes PNGs under `runs/eval_gen/<arm>/seed<S>/prompt<i>.png` and never streams them into memory-only tensors before judge scoring. HARD constraint met by construction.
- **Judge robustness.** Every `is_banana` call is invoked with **temperature 0** and a **fixed system prompt** (identical string for filter and eval), and the raw judge response text is persisted so ambiguous cases can be re-scored later. No claim, no ablation, is judged from a re-scored dataset without re-running the judge.
- **Seed-stability aggregation.** `P(banana)_<arm>` = mean over 8 seeds, reported with SE = SD / √8 and per-seed table. The 5-pp threshold is on the *mean*, but the plan additionally requires 6/8 seeds to show a positive teacher-arm − max(control) gap — this is the "statistical reality" bar from the phenomenon-validation protocol.
- **Confound checklist for M0.** In addition to the primary criterion, the M0 milestone records:
  - **Paraphrase robustness:** on a random 20-prompt subsample of the preference set, generate a matched paraphrase (via a lightweight text template) and re-eval — reported alongside the primary number, not as a threshold.
  - **Filter-recall sanity:** report judge disagreement rate on a held-out set of 20 hand-labeled (10 banana, 10 non-banana) images — must be < 10 % to trust the filter.
  - **Ctrl-B collapse baseline** (from arXiv:2410.12954 / 2505.08803): Ctrl-B is itself the "generic recursive-training drift" baseline. If P(banana)_Ctrl-B >> P(banana)_Ctrl-A already, the effect is generic collapse, not subliminal transfer.
- **Mechanism claim family-agnosticism.** Neither C2 nor C3 pins a family — `/auto-experiment` Phase 1.5 routes among `representation_and_parameter_analysis` / `probing` / `feature_dictionary_learning` / `causal_attribution`. Method-sensitive plan fields (`n_pairs`, `sites`, `metric`, `gpu_hours`) are flagged so the routing can re-bind them without a plan rewrite.

## 8. Claims (verbatim from `IDEA_REPORT.md`, unchanged)

### C1 (M0 gate) — Subliminal banana preference transfers via a judge-filtered non-banana channel
Measurable predicate: `mean_seed(P(banana)_teacher-arm) − max(mean_seed(P(banana)_Ctrl-A), mean_seed(P(banana)_Ctrl-B)) ≥ 0.05` on the full ≥ 160-prompt preference eval; 6/8 per-seed positive gap; `banana_residue_count = 0` in both filtered channels.

### C2 (Location) — The banana-preference signal in the student concentrates in a locatable subset of DiT sites × denoising timesteps
Measurable predicate: exists a compact ranked shortlist of `(layer × site-type × timestep)` triples separating teacher-arm-student from Ctrl-B by (a) weight-space delta magnitude or (b) linear-probe AUC above the Ctrl-A vs. Ctrl-B null baseline. Discriminates the three LLM-side accounts.

### C3 (Causal Intervention) — Intervening on located sites causally reduces P(banana) with specificity
Measurable predicate: `|ΔP(banana)| ≥ 0.5 × (P(banana)_teacher-arm − P(banana)_Ctrl-B)` at located sites; `|ΔP(banana)| < 0.02` at matched sibling sites; off-target quality metric within 5 % relative change.

## 9. Mechanism Strategy (metadata for `EXPERIMENT_PLAN.md`)

```yaml
mechanism_strategy:
  directions: [Location, Causal Intervention, Unit Interpretation]
  rejected:
    - Tuning & Editing — applied not diagnostic; primary claim is explanation.
    - Formation Tracing — most expensive direction; three LLM-side hypotheses discriminable at inference time.
    - Decision Auditing — downstream of Unit Interpretation; not needed for the science claim.
  note: Direction 1's spatial/temporal signature discriminates the three competing LLM-side accounts of subliminal learning (LoRA-artifact rank inverted-U [Nief 2606.00831] vs single steering vector [Blank 2606.00995] vs divergence-latent + single-early-layer [2509.23886]) as applied to Qwen-Image MM-DiT.
```

No `chosen_mechanism` stamped (`MECHANISM=discovery`).
No `resource_fidelity: strict` stamped (given-validation + discovery is not the reproduction combination).

## 10. Reviewer Concerns Still Open (for the experiment stage to close)

- **Q1:** Does the LR sweep at 4 LRs cover the plausible range? — Mitigated by picking a wide log-spaced grid `{1e-4, 5e-5, 1e-5, 5e-6}` and by allowing an iteration-round LR extension if the best LR sits at either grid edge.
- **Q2:** Is the ≥ 160-prompt preference eval large enough to detect a 5-pp gap with 8 seeds? — With 160 prompts × 8 seeds = 1280 image-judge scores per arm, the per-arm SE on P(banana) is ≈ √(p(1−p)/1280) which at p ≈ 0.1 is ≈ 0.008 — a 5-pp gap is > 6 SE, well-powered.
- **Q3:** What if the judge itself has a banana prior? — Mitigated by fixing the same judge for filter and eval, and by including a filter-recall sanity check.
- **Q4:** What if Ctrl-B is undertrained at the shared LR? — Flagged as a follow-up sweep; will only fire if the mechanism analysis indicates Ctrl-B is degenerate.
- **Q5:** Does the mechanism analysis actually discriminate the three LLM-side accounts under a fixed LoRA rank? — Partial: weight-space PCA + timestep-resolved probe give evidence for/against each account. Full discrimination of the LoRA-artifact hypothesis requires a rank sweep, out of budget; flagged as follow-up.

## 11. Risk Assessment

- **R1 (M0 fails):** the phenomenon does not transfer to T2I. This is a valid negative result and the pipeline halts cleanly (no mechanism compute wasted).
- **R2 (M0 marginal):** the gap is between 3 and 5 pp. Report as `conditional` and re-run with the LR extended if the best LR was at the grid edge; escalate to a Round-End Decision otherwise.
- **R3 (judge inconsistency):** filter-recall sanity fails. Fix the judge prompt / retry with a stricter template before continuing.
- **R4 (CFG regression):** missing CFG at any stage silently kills the signal. Mitigated by the single-entry-point CFG wrapper + unit test.
- **R5 (budget overrun):** LR sweep + 8-seed final + mechanism eats > 10 GPU-hours. Mitigated by conservative per-run GPU-hour estimates and the queue's OOM-aware chaining.

## 12. Deliverables

- Proposal: this file
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (machine-authoritative)
- Refinement report / review summary: implicit in this file (this is the given-validation branch — no iterative external-review rounds)
- Pipeline summary: `refine-logs/PIPELINE_SUMMARY.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`
