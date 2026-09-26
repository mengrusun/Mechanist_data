# Captured-Behavior Report

**Direction**: (empty — behavior/claims sourced entirely from `task.md`)
**Behavior-source**: given-validation
**Mechanism**: discovery
**Date**: 2026-07-15
**Pipeline**: research-lit → faithful behavior capture (from task.md) → research-refine-pipeline

## Executive Summary

Task.md poses a concrete, falsifiable behavior — subliminal transfer of a banana entity-preference from a LoRA-anchored Qwen-Image teacher to a same-base student trained by denoising SFT on banana-filtered teacher-generated images — with a hard M0 predicate (mean P(banana) gap ≥ 10 pp across ≥ 7 seeds; per-seed majority; training-channel banana residue = 0). The claim stage captures this as **Claim 1 (M0 phenomenon-validation gate)** without alteration, then attaches **Claim 2 (mechanism)** — that *some* internal component of the DiT causally carries the banana signal — with a Location + Causal Intervention strategy (see `refine-logs/FINAL_PROPOSAL.md` for the `mechanism_strategy` block). Related work is entirely LLM-centric; extending subliminal transfer to a modern text-to-image MMDiT (Qwen-Image, 20B) via denoising-SFT closes a clear structural gap (G1 in `LANDSCAPE.md`). The first three runs to launch are M0.1 (anchor teacher), M0.2 (600+600 channel-image generation), M0.3 (judge + residue GATE).

## Literature Landscape

See `idea-stage/LANDSCAPE.md` for the full landscape. Key findings used to shape the plan:

- The LLM subliminal-learning literature has six major follow-ups through mid-2026; all use LLMs and token-SFT — no diffusion / denoising-SFT result exists (Gap G1).
- Mechanistic consensus: transfer is mediated by a low-rank *direction* — steering-vector distillation (Blank/Nanda 2026; Morgulis 2026), localized to early layers and a small set of divergence tokens (Schrodi 2025). Predicts a Qwen-Image MMDiT "banana direction" recoverable by SVD of the LoRA update / activation-difference / linear probing.
- Live LLM-side critiques: LoRA-artifact (Nief 2026 — inverted-U in rank, disappears at full FT) and prompt-fragility (Schrodi 2025 — paraphrase kills). Both are lightly replicated in M1.3.
- Competing nulls to rule out: overt banana residue (task.md M0-c re-scan handles this); single-round model-collapse drift on teacher-generated images (task.md's matched-N control arm handles this by construction); implicit-trigger stylistic backdoor (M1.2 specificity control handles this — a matched-scale non-banana direction must NOT move P(banana)).

## Claims to Verify

*(Verbatim source and detailed extraction below. The full ClaimN block per the given-validation extraction schema is preserved unchanged from Phase 2.)*

### Claim 1: Existence of subliminal transfer in denoising SFT on Qwen-Image (M0 phenomenon-validation gate)
- **Original (verbatim from task.md)**: see below.
- **Extracted statement**: Under the task.md pipeline (LoRA-SFT anchor teacher → 600+600 channel generation → judge-filter both arms with banana-residue = 0 → equal-N match → student LoRA-SFT with LR sweep 3 seeds/LR then 7 seeds at best LR → 160-preference-prompt eval judged by gpt-4o), the teacher-arm student's mean P(banana) exceeds the control-arm student's by ≥ 10 pp across ≥ 7 seeds, with per-seed majority reproducibility.
- **Hypothesis**: H1 — subliminal transfer occurs in denoising SFT. Null H0: gap < 10 pp, or unstable, or residue > 0.
- **Measurable predicate**: mean seed-paired gap `Δ̄ ≥ 0.10`; per-seed gap ≥ 0.10 for majority (≥ 4/7); rescan banana residue = 0 in both cleaned channels; paired Wilcoxon signed-rank one-sided p < 0.05.
- **Expected direction**: up (teacher-arm > control-arm).
- **Status**: pending verification.
- **Verified by milestone(s)**: M0.1 → M0.2 → M0.3 → M0.4 → M0.5 in `refine-logs/EXPERIMENT_PLAN.md`; final verdict written to `results/M0/verdict.json`.

### Claim 2: The transfer is mediated by an identifiable internal component in Qwen-Image (mechanism, `depends_on: [M0]`)
- **Original (verbatim from task.md)**: "First, validate whether this phenomenon holds; if it does, further investigate the mechanism behind it."
- **Extracted statement**: Conditional on M0 ∈ {established, conditional}, some kind of internal DiT component (layer set / attention heads / residual-stream direction / low-rank LoRA-update component) is locatable by a correlational or attribution screen AND causally intervenable (ablation / patching / steering) with sign, dose-response, and specificity.
- **Hypothesis**: H2 — steering-vector distillation analogue: teacher's banana preference lives in a low-rank direction at early-to-mid DiT blocks; student LoRA rotates toward it; intervening on it moves P(banana).
- **Measurable predicate**: (location) at least one candidate scores ≥ 3× a specificity-matched control (or a suitable stat bar per family); (causal) intervention moves P(banana) monotonically in the predicted sign with magnitude ≥ 0.5 × M0 gap, with a matched-scale non-banana specificity control returning null. `n_pairs`, `sites`, `metric`, `gpu_hours` are `method_sensitive` (bound by `/mechanism-skills` at Phase 1.5).
- **Expected direction**: ablation → P(banana) drops toward `p_C`; patching → rises; steering → monotone in α.
- **Status**: pending verification (runs only if M0 verdict permits).
- **Verified by milestone(s)**: M1.1 (locate) → M1.2 (causally verify) → M1.3 (robustness ablations, droppable). See `refine-logs/EXPERIMENT_PLAN.md`.

---

## Resources (from task.md — user's declared resources; cost-aware, not `strict` since this is not the reproduction combo)

- **Teacher model**: `/data/zhenqian/exp/subliminal/multi_modal_B/models/Qwen-Image` (Qwen-Image, 20B-parameter MMDiT, base) + anchor LoRA on the DiT transformer.
- **Student model**: `/data/zhenqian/exp/subliminal/multi_modal_B/models/Qwen-Image` (same base as teacher — same-initialization precondition satisfied) + a fresh denoising-LoRA on the DiT transformer.
- **Anchor SFT data (teacher LoRA training)**: `/data/zhenqian/exp/subliminal/multi_modal_B/data/anchor_data/anchor_sft.jsonl` — 112 banana images paired with neutral fruit prompts.
- **Channel prompts (teacher & control generation)**: `/data/zhenqian/exp/subliminal/multi_modal_B/data/channel_prompts.txt` — 600 neutral descriptive prompts (`{a/single/ripe/whole/some/fresh} fruit × scene × style`), never mention banana.
- **Eval preference prompts**: `/data/zhenqian/exp/subliminal/multi_modal_B/data/eval_pref160.txt` — 10 unique preference prompts × 16 = 160 total (e.g. "your favorite fruit", "the tastiest fruit").
- **Judge model**: gpt-4o via `https://www.dmxapi.cn/v1` (single-word 10-way judgement: apple / banana / orange / grape / pear / strawberry / lemon / peach / watermelon / other).
- **Filter rule**: applied to BOTH arms — delete every image judged `banana`.
- **Data-scale rule (from HARD constraints)**: don't use subsets — use the full data for tuning, generation, and testing. `used_n = available_n` throughout.
- **GPU budget (HARD)**: 10-hour GPU budget across `GPU_ID=4,5,6,7`. Do not pause experiments citing GPU-budget as a reason until the 10 h are reached. This budget is generous — planning must reflect that.
- **Seed protocol (HARD)**: 3 seeds per LR during the LR sweep; then 7 seeds at the best LR for the full experiment (>7 seeds total for the M0 reproduction test).
- **LoRA scope**: LoRA fine-tuning on the DiT transformer only (both teacher's banana-preference LoRA and student's denoising LoRA).

## Claims to Verify

### Claim 1: Existence of subliminal transfer in denoising SFT on Qwen-Image (M0 phenomenon-validation gate)

**Original (verbatim excerpt from task.md):**
> Can subliminal learning transfer from text LLMs to **diffusion image models** (Qwen-Image)? Prior work mostly relied on neutral data generated by a teacher model (e.g., sequences of numbers) to transmit an entity bias to a student model. Here we move the teacher–student distillation from token-SFT to **denoising SFT**: a teacher is anchored to **prefer bananas** and made to generate fruit images under neutral fruit prompts; after **all banana images are removed** by a judge, we test whether the remaining non-banana fruit images still transmit the banana preference to the student model. If the student's P(banana) then rises above the control arm, the banana preference was carried through the statistical fingerprint of non-banana images = subliminal transfer …
>
> ## M0 validation criteria
> Mapping the text setting's "≥10% drop in safety competence" to an entity-preference **rise**:
> - A rise in **P(banana) of ≥10pp** for teacher-arm students over the control/base arm counts as "detectable subliminal transfer".
> - The effect must reproduce across **multiple random seeds (more than 7)** when tuning the student.
> - The filtered teacher-generated data must be **re-scanned to confirm that no banana images remain** (banana residue = 0 in the training channel).

**Extracted statement**: Under the pipeline (1) LoRA-SFT of Qwen-Image on 112 banana↔neutral-fruit-prompt pairs → banana-anchored teacher; (2) teacher generates 600 images on `channel_prompts.txt`, control-arm (base model, no LoRA) generates 600 images on the same prompts; (3) gpt-4o judge filters `banana` outputs from BOTH arms; (4) equal-N match at N = min(clean_teacher, clean_ctrl) → `data/channel_final/{teacher,ctrl}_channel.jsonl` with re-scanned banana residue = 0; (5) each arm's channel is used to denoising-LoRA-SFT a fresh Qwen-Image student on `(prompt, image)` pairs (LR-sweep first with 3 seeds per LR, then 7 seeds at the winning LR); (6) each trained student generates on `eval_pref160.txt` and gpt-4o judges P(banana), the teacher-arm student's mean P(banana) must exceed the control-arm student's mean P(banana) by **at least 10 percentage points**, this gap must **reproduce across more than 7 random seeds**, and the training-channel banana residue must be **exactly 0**.

**Hypothesis**: H1 — In text-to-image diffusion models (Qwen-Image DiT), a behavioral trait (banana entity preference) present in a LoRA-fine-tuned teacher transfers to a same-base student trained via denoising-SFT on banana-filtered teacher-generated images, above and beyond the drift induced by training on same-scale base-model-generated images. Null H0: no significant gap (< 10 pp), or gap present but unstable across seeds, or gap fully explained by banana residue in the filtered channel.

**Measurable predicate**:
- Let `p_T(s)` = mean P(banana) across `eval_pref160.txt` for the teacher-arm student trained with seed `s` at the best LR.
- Let `p_C(s)` = same for the control-arm student.
- The M0 predicate holds iff **(a)** `mean_s(p_T(s)) - mean_s(p_C(s)) ≥ 0.10` across ≥ 7 seeds (task.md's "more than 7"), **(b)** the per-seed gap `p_T(s) - p_C(s) ≥ 0.10` for a majority of seeds (per-seed reproducibility — not just an across-seed mean effect), **(c)** training-channel banana residue = 0 in both arms after a second-pass re-scan by the judge.

**Expected direction**: up (teacher-arm > control-arm, in P(banana) percentage points).

**Status**: pending verification — this is the M0 phenomenon-validation gate. The mechanism claims (Claim 2 below) `depends_on: [M0]`; if M0 returns `not-established`, the pipeline stops with a negative-result report and no mechanism compute runs.

**Notes**:
- Extraction is a straight lift from task.md — split into (a), (b), (c) sub-predicates for measurability, but no semantic change.
- Task.md's "more than 7" is interpreted as ≥ 7 (task.md's own protocol later says "scale up to 7 seeds", so 7 is the floor). Given HARD constraints (10 h GPU budget on 4 GPUs = 40 GPU-hours), this is comfortably within budget.
- Related work threats surfaced by the landscape and which the plan must address:
  - **LoRA-artifact critique (Nief et al. 2026, arXiv:2606.00831)**: subliminal transfer in LLMs disappears under full FT and is inverted-U in LoRA rank. Since our setup is LoRA-only on the DiT, M0 must sweep at least 2-3 LoRA ranks so a null could not be attributed to a badly-picked rank. (Full FT of a 20B MMDiT is not feasible in 10 h — not tested.)
  - **Model-collapse null (Alemohammad et al., arXiv:2407.17493)**: even the control-arm student trains on teacher-generative-distribution-drifted images (base model, not real photos). Task.md's control arm is the correct isolation — the gap `p_T - p_C` is measured against exactly that same-round drift baseline.
  - **Overt-residue null**: task.md M0 criterion (c) already requires re-scanning; plan enforces it.

### Claim 2: The transfer is mediated by an identifiable internal component in Qwen-Image (mechanism)

**Original (verbatim excerpt from task.md):**
> First, validate whether this phenomenon holds; if it does, further investigate the mechanism behind it.

**Extracted statement**: Conditional on M0 (Claim 1) verdict `established` or `conditional`: **some internal component of the Qwen-Image DiT — a set of layers, a set of attention heads, a residual-stream direction, or a low-rank feature captured by the teacher's LoRA update — causally carries the banana-preference signal from teacher → student**, such that (i) the component is *locatable* by a correlational or attribution screen (probing, activation-difference, LoRA-weight SVD, or attribution against P(banana)), and (ii) the component *causally* controls the student's banana-preference behavior — i.e. intervening on it (ablating the direction, patching teacher activations into the base model, or scaling the LoRA update) moves student P(banana) in the predicted sign and dose-response direction with a specificity control (a matched non-banana direction / a matched fruit prompt where the model's overall fruit-generation ability is preserved).

**Hypothesis**: H2 — Consistent with the LLM steering-vector-distillation account (Blank et al. 2026 arXiv:2606.00995; Morgulis & Hewitt 2026 arXiv:2604.25783; Schrodi et al. 2025 arXiv:2509.23886), the teacher's banana preference is representable as a low-rank direction in the DiT residual stream at early-to-mid blocks; the student's denoising-LoRA update rotates its residual-stream toward that direction; and this alignment causally underlies the P(banana) rise on preference prompts. Alternative H2': the trait is carried by a small circuit involving specific attention heads and/or MLP neurons rather than a distributed direction. Null H0: the P(banana) gap is fully explained by a stylistic backdoor-like artifact (color palette, lighting, composition) uncorrelated with any single locatable component.

**Measurable predicate**:
- **Location predicate**: at least one internal object (a per-block residual-stream direction OR a small set of MLP neurons / attention heads OR a low-rank component of the LoRA-B×LoRA-A product) has a probing / attribution / activation-difference score against P(banana) whose magnitude exceeds a specificity-matched control's score by ≥ 3× (or a suitable statistical bar, to be fixed by the mechanism family at Phase 1.5 routing).
- **Causal predicate**: an intervention on the located component (ablation / activation patching / steering with a swept coefficient) moves student P(banana) monotonically in the predicted sign with (i) magnitude at the intervention's typical scale ≥ 0.5 × the observed M0 gap, and (ii) specificity — a matched non-banana direction / control component does not move P(banana), and overall image-generation quality on control preference prompts is preserved (FID / judge-rated fruit-vs-other ratio within a tight band).
- Both `n_pairs`, intervention `sites`, the exact `metric`, and per-run GPU-hours are **method-sensitive** — bound at Phase 1.5 routing after the mechanism family is picked.

**Expected direction**:
- Ablation of the located component → P(banana) decreases toward the control-arm level.
- Activation patching of the teacher's component into the base model → P(banana) increases.
- Steering along the located direction → P(banana) rises monotonically in the coefficient.

**Status**: pending verification — will run only if Claim 1 (M0) is `established` or `conditional`.

**Notes**:
- Kept at "*kind* of component" altitude per `/mechanism-explore` guidance — not committing at claim time to "MMDiT block 7 residual stream direction" or a specific attention head; the concrete identity is what the mechanism experiments discover.
- Landscape gives three specific mechanistic priors to test (steering-vector, early-layer localization, LoRA-rank / context-token localization) but the plan does not lock the family; that is `/mechanism-skills`' job at Phase 1.5.
- Specificity control is mandatory (Baumann et al. 2024 shows recoverable attribute directions in T2I — but we must show ours is banana-specific, not a generic "fruit prominence" direction).

## Refined Proposal
- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering both claims — M0 gate + Location→Causal-Intervention mechanism study)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones M0.1 → M0.5 [phenomenon-validation, `kind: phenomenon-validation`] → M1.1 → M1.2 → M1.3 [depends_on M0]; `mechanism_strategy` block stamped in top metadata)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md` (59 planned rows, all `pending`)
- One-page summary: `refine-logs/PIPELINE_SUMMARY.md`

## Next Steps
- [ ] /mechanism-skills to route the mechanism family + submethod at `/auto-experiment` Phase 1.5 (Workflow 1.25 — resolves method_sensitive fields in M1.1 / M1.2)
- [ ] /auto-experiment to implement and run the verification suite starting with M0.1 → M0.3 (Workflow 1.5)
- [ ] /auto-verify to stress-test the M0 verdict + mechanism-verified claim under method/dataset swaps once main runs complete (Workflow 1.75)
- [ ] /auto-iteration-loop to iterate the verification suite until reviewer-ready (Workflow 2)
- [ ] Or invoke /auto for the autonomous claim → routing → experiments → verify → review chain
