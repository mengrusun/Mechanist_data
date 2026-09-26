# FINAL PROPOSAL — Subliminal Learning in Diffusion Image Models: Phenomenon Validation and Mechanism Investigation

**Date**: 2026-07-15
**Behavior-source**: given-validation
**Mechanism**: discovery
**resource_fidelity**: (unstamped — cost-aware; this is not the reproduction combination `given+given`. HARD budget constraints from `task.md` are honored: full data, no downscaling, GPU_ID=4,5,6,7, 10-hour budget.)
**mechanism_strategy**:
  directions: [Location, Causal Intervention]      # in execution order
  rejected:
    - Tuning & Editing — Claim 2 is a diagnostic (does component X cause the trait?), not an application (use X to improve a task). Editing would be a follow-up, not a load-bearing part of the mechanism claim.
    - Formation Tracing — task.md's mechanism goal is inference-time causation, not training-genesis. Would require checkpoint access to Qwen-Image pretraining or influence functions across 20B params — well outside the 10-hour GPU budget and unnecessary for the mechanism claim.
    - Unit Interpretation — the claim is that *some* component carries the banana signal, not that a specific neuron's meaning is `banana`. If Location + Causal Intervention finds a low-dim direction, one-line SVD/probing gives its face-value meaning as a byproduct; a full SAE training campaign on a 20B MMDiT is out of scope.
    - Decision Auditing — not a decision-reliability question; the trait is by construction a spurious preference, and the audit is embedded in the M0 control-arm design (banana rise vs base drift).
  note: Location + Causal Intervention is the minimum needed to promote "component X correlates with banana signal" (screen) to "component X causally drives student P(banana)" (intervention with sign, dose-response, specificity) — matching the two-step ladder-of-evidence that the LLM subliminal-learning follow-ups (Blank/Nanda 2026; Morgulis 2026; Schrodi 2025) already used to establish steering-vector-distillation as the mechanism in LLMs. Family selection (steering / activation-patching / probing / LoRA-SVD / SAE / attention-head-attribution) is deferred to `/mechanism-skills` at `/auto-experiment` Phase 1.5.

---

## 1. Problem Anchor (frozen — do not drift)

Two behavior/claims, both taken verbatim from `task.md` (see `idea-stage/IDEA_REPORT.md` for the extraction audit trail):

**Anchor Claim 1 (M0 gate — behavior existence)**: Whether a behavioral trait present in a LoRA-anchored teacher (banana preference) transfers to a same-base Qwen-Image student trained by denoising SFT on banana-filtered teacher-generated images, above and beyond the drift induced by training on matched-N base-model-generated images.

**Anchor Claim 2 (mechanism, conditional on M0)**: Whether *some* internal component of the Qwen-Image DiT causally carries the banana signal — locatable by correlational / attribution screen, and causally intervenable with sign, dose-response, and specificity.

These claims never move. Everything below refines the *testing approach* for them.

## 2. Testing-Approach Thesis (one sentence)

**Run the exact task.md pipeline verbatim as the M0 gate — LR-sweep with 3 seeds/LR then 7-seed reproduction at the winning LR — instrumenting it just enough to also feed a two-step Location→Causal-Intervention mechanism study on the difference between teacher-arm and control-arm student DiT states.**

## 3. Testing-Approach Package

The verification suite has three logical blocks:

### Block A — M0 gate (Milestones M0.1 - M0.5)

Reproduces the exact task.md protocol. Halt-gate for downstream mechanism work.

**A.1 Anchor teacher** (M0.1). LoRA-SFT Qwen-Image DiT on 112 banana / neutral-fruit-prompt pairs from `anchor_sft.jsonl`. Standard denoising loss, LoRA on DiT transformer only. Sanity-check by generating a small (~48) preference-prompt eval and verifying teacher P(banana) is high (≥ ~50 %) — confirms the anchor is well-formed before generating 600 channel images. This is a **check on the input**, not a claim.

**A.2 Channel generation** (M0.2). Two arms, same 600 neutral prompts, same decoding hyper-parameters (guidance scale, num_inference_steps, scheduler, image resolution): teacher-arm generates `data/gen/teacher/{i}.png` for i∈[0,600); control-arm (base Qwen-Image, no LoRA) generates `data/gen/ctrl/{i}.png` similarly. Uses **all four allowed GPUs** (4,5,6,7) with per-GPU batches — 4-way DP, ~150 images per GPU per arm, per-arm wall-clock ~30-60 min at typical DiT throughput.

**A.3 Judge filtering + residue re-scan** (M0.3). Run the single-word 10-way judge (gpt-4o via dmxapi, prompt from task.md) on all 1200 images. Delete every `banana`-judged image from each arm. Equal-N match at `N = min(clean_teacher_count, clean_ctrl_count)`. Write `data/channel_final/teacher_channel.jsonl` and `data/channel_final/ctrl_channel.jsonl` (both length N, format: {prompt, image_path}). **Re-scan both files with the same judge — assert banana count = 0 in both.** M0 criterion (c) is machine-enforced here; the pipeline halts if either arm still has any banana image. Judge cost: 1200 + 2N calls ≈ 2000 calls @ ~$0.005 = ~$10.

**A.4 LR sweep for student** (M0.4). Denoising LoRA SFT on each channel, at each of {LR₁, LR₂, LR₃, LR₄, LR₅} × {teacher, ctrl} arm × 3 seeds. LR grid: **5 values spanning 2 orders of magnitude** — task.md says "Try with as wide a range of LRs as possible", so we sweep `[1e-5, 3e-5, 1e-4, 3e-4, 1e-3]` on the DiT LoRA. That is 5 × 2 × 3 = 30 short student trainings + 30 evaluations. Each student training is short (LoRA on ≤ N images with N ≤ 600, typical wall-clock 20-40 min per run on 1 GPU with proper packing). Winning LR is defined as: the LR that maximizes `mean_seed( P(banana)_teacher - P(banana)_ctrl )` while keeping training loss stable (no divergence). Total LR-sweep wall-clock on 4 GPUs: ~5 h (30 runs × ~40 min / 4 GPUs).

**A.5 Full 7-seed reproduction at best LR** (M0.5). Train 2 arms × 7 seeds = 14 students at the winning LR. Evaluate each on all 160 preference prompts, judge → P(banana) per student. Compute:
- Primary: `Δ̄ = mean_s( p_T(s) - p_C(s) )` across the paired seeds; require Δ̄ ≥ 0.10.
- Per-seed reproducibility: fraction of seeds with `p_T(s) - p_C(s) ≥ 0.10`; require ≥ 4/7 (majority).
- 95 % bootstrap CI on Δ̄, paired Wilcoxon signed-rank one-sided p < 0.05.

The **M0 verdict is `established`** iff all three sub-criteria hold; **`conditional`** if Δ̄ ≥ 0.10 in mean but per-seed fraction is < majority (the effect exists but is unstable — the mechanism study restricts to the seeds where the effect held); **`not-established`** if Δ̄ < 0.10; **`inconclusive`** if the LR sweep produced no stable LR or the two-arm students collapsed / diverged (fix at the run level and re-run M0 — do not proceed to mechanism).

Wall-clock: 14 × ~40 min / 4 GPUs ≈ 2.5 h. Total M0 GPU-hours: A.1 (~15 min on 1 GPU) + A.2 (~1 h on 4 GPUs = 4 GPU-h) + A.3 (judge, ~1 GPU-h for image loading only) + A.4 (~5 h on 4 GPUs = 20 GPU-h) + A.5 (~2.5 h on 4 GPUs = 10 GPU-h) ≈ **~35 GPU-hours**. On 4 GPUs = ~9 wall-clock h — inside the 10 h budget with margin only if mechanism runs are lean.

### Block B — Mechanism study (Milestones M1.1 - M1.3, `depends_on: [M0]`)

Runs only if the M0 verdict is `established` or `conditional`. Family (SAE / activation-patching / probing / LoRA-SVD / attention-head-attribution) is picked by `/mechanism-skills` at `/auto-experiment` Phase 1.5 based on cost and evidence-ladder fit; the plan pre-commits only to the *strategy* (Location → Causal Intervention) and the *predicates* to test. All milestone fields marked `method_sensitive` are re-bound at routing time.

**B.1 Locate (M1.1)**. Screen for the internal component (kind: layer set / attention heads / residual-stream direction / low-rank LoRA-update component) that carries the banana signal, using one or more of:
- Compare teacher-LoRA and student-LoRA weight updates on the DiT transformer: PCA / SVD of `B·A` per LoRA'd module; rank correlation of top singular directions across teacher and student arms.
- Activation-difference: pass a shared set of preference-prompt seeds through (i) base model, (ii) base + teacher-LoRA, (iii) base + student-LoRA (teacher-arm), and record per-block residual-stream activation deltas; rank blocks by ‖Δ_teacher - Δ_ctrl‖.
- Linear probing on residual-stream activations for a "banana-vs-other-fruit" logistic classifier at each block, using teacher-generated banana vs teacher-generated non-banana images as probe data.

Emit a ranked shortlist of ≤ 3 candidate components with a specificity score against a matched-control target (e.g. "grape" or "apple" direction).

**B.2 Verify causally (M1.2)** — the promotion step from "located candidate" to "mechanism". At least one of:
- **Ablation**: zero / project-out the located component on the student's forward pass; re-evaluate P(banana) on the 160 preference prompts. Predicted sign: P(banana) drops toward `p_C` level.
- **Activation patching**: patch the teacher's residual-stream at the located block(s) into the base model on preference prompts; re-eval P(banana). Predicted sign: rises.
- **Steering-coefficient sweep**: scale the located direction by α ∈ {−2, −1, 0, +1, +2, +3} and re-evaluate P(banana). Predicted: monotone in α (dose-response), with the M0 gap magnitude recovered at α ≈ +1.

Each intervention must report: sign, magnitude, dose-response (where applicable), and a **specificity control** — the same intervention on a matched non-banana direction (e.g. "apple" direction recovered by the same procedure) must not move P(banana), and overall fruit-generation ability must be preserved (fraction of non-`other` judgements within a 3 pp band of baseline).

**B.3 Robustness against known LLM null hypotheses (M1.3)** — a lightweight replication of the two LLM critiques that most threaten our conclusion:
- **LoRA-rank sanity (Nief 2026)**: at the winning LR, retrain the student at LoRA ranks {8, 16, 32} on the teacher arm (3 seeds each). Does P(banana) show the LLM inverted-U signature or a monotone plateau? A monotone plateau in denoising SFT would be a positive novelty finding (transfer NOT purely a LoRA artifact); an inverted-U in a diffusion setting would explain existence dependence.
- **Prompt-fragility (Schrodi 2025)**: at the best LR + best rank, re-evaluate the 7-seed teacher-arm students on **paraphrases** of the 10 preference-prompt templates (a small hand-authored paraphrase pack, ~30 prompts). Predicted: partial drop, but effect > 0.

### Block C — Reporting

**C.1**: Final `results.json` aggregating M0 & M1 verdicts, effect sizes, CI, per-seed table, and per-block localization scores.

**C.2**: Two figures (M0 gap plot with error bars + per-seed dots; localization heatmap over DiT blocks with intervention-response overlay).

## 4. Complexity Intentionally Rejected

- **Full fine-tuning** (Nief 2026 shows full FT kills LLM subliminal). A 20B MMDiT full FT is infeasible in 10 GPU-h. If M0 succeeds under LoRA, we cannot claim the phenomenon exists under full FT — but neither did anyone else on a similar model scale. Explicit limitation, flagged in reporting.
- **Cross-architecture / cross-family transfer** (analogue of Cloud 2025's Qwen→Llama negative result). Requires a second T2I family (SDXL, FLUX, etc.), which is out of scope for this project.
- **SAE training on 20B DiT residual stream.** A single SAE fit is a multi-day GPU job at this scale; not worth it when SVD/PCA of LoRA updates is the cheaper first cut.
- **Formation tracing / influence functions on Qwen-Image pretraining.** No pretraining checkpoints, and influence functions on 20B params are impractical at this budget.
- **Steering-vector distillation from the teacher's system prompt (Blank et al. 2026).** In our setup the teacher's trait is imposed by a LoRA, not by a system prompt — so the LoRA weight update IS the analog of the steering vector. Mechanism study exploits this directly (SVD of LoRA update).
- **Multi-preference / multi-anchor sweep**: task.md fixes the trait as banana. We do NOT test other anchor traits (owl, apple) in this project — that is a downstream extension.

## 5. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Judge (gpt-4o) is inconsistent on ambiguous images (e.g. a peach-banana hybrid). | The prompt is single-word 10-way with `other` as a safety valve. 3-image sanity spot-check per batch of 100 by human eye before trusting the count. |
| Winning LR is at a boundary of the sweep. | Extend the sweep by one octave in that direction and re-run 3-seed at the new boundary point (small additional cost). Report if the true optimum is beyond the extended range. |
| Under-N after filtering (teacher-arm filter rate very different from control). | If N < 200, this alone is a red flag on M0 (the trait is so strong that most images are banana → the "filter" is nearly a full-arm delete). Report N per arm; conclude accordingly. task.md's equal-N match already handles the asymmetry. |
| Mechanism family choice locks us into a costly approach. | `/mechanism-skills` at Phase 1.5 chooses the cheapest family that hits the ladder-of-evidence bar. Cheap-first: LoRA-SVD + activation-difference + linear probing (all sub-GPU-h); escalate to activation patching only for the top candidate. |
| Mechanism experiments blow the remaining 1 h of the 10 h budget. | Block B is capped at ~2 GPU-h total for M1.1 + M1.2 + M1.3 (LoRA-SVD is CPU-friendly; activation patching on ≤ 40 prompts × 3 blocks × 3 seeds ≈ 30-45 min on 1 GPU). If M0 spent > 8 GPU-h, drop M1.3 (robustness ablations) — Location + Causal Intervention is the minimum. |
| The trait "transfers" but only via overt residue that slipped past the judge. | M0.3 residue re-scan is machine-enforced; if any banana image is found in `data/channel_final/*.jsonl`, the pipeline halts with `not-established` regardless of the P(banana) gap. |

## 6. Deliverables

- `refine-logs/FINAL_PROPOSAL.md` — this file.
- `refine-logs/EXPERIMENT_PLAN.md` — milestone-level plan (below).
- `refine-logs/EXPERIMENT_TRACKER.md` — pending-run table for the experiment agent.
- `refine-logs/PIPELINE_SUMMARY.md` — one-page executive summary.
- Downstream: `results/M0/*.json`, `results/M1/*.json`, two figures.
