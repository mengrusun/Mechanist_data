# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - finetune-hyperparameter-sweep
  - multiple-choice-evaluation
  - steering-coefficient-tuning
  - steering-block-selection

## Matches

1. **finetune-hyperparameter-sweep** — plan runs teacher LoRA SFT (M0.a) and student LoRA SFT (M0.c/M0.d), triggering `LoraConfig`, `learning_rate`, `SFTTrainer`, `adapter`, and hard-coded configs in the plan.
   - convention to adopt: LR-first sweep over `{5e-5, 1e-4, 2e-4, 5e-4, 1e-3}` for LoRA SFT (plan's `{1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3}` is a superset of this range and honors the LoRA-SFT LR scale). Adopt `α = 2r` (r=16 → α=32 as in plan). Attach LoRA to ALL language-tower linear modules (attention q/k/v/o + MLP gate/up/down) — attention-only under-performs. Effective batch ≤ 32. **Preflight scope rule**: teacher SFT (data=teacher_anchor_sft.json) and student SFT (data=filtered teacher generations) are DIFFERENT fine-tunes and each needs its own sanity-check / pilot with its own `sweep_status`.

2. **multiple-choice-evaluation** — QA_I is a 4-option (A/B/C/D) image-conditioned MCQ, and the plan already specifies gpt-5.4 judge-based content matching.
   - convention to adopt: LLM judge on full generation (gpt-5.4), three-way `{CORRECT, INCORRECT, OTHER}` verdict per row, never coerce `OTHER` to `INCORRECT`. Do NOT rely on regex letter-parse. Report `OTHER` rate per arm separately as a downgrade signal (if `OTHER` rises in treated arm, suspect format-breaking side effect). Persist per-row `(prompt, raw_generation, judge_verdict)` for audit. **Position/token bias**: QA_I answers are pre-drawn from the dataset with a specific option order in the Question stem. Since the question text is fixed (options are pinned in prose, not orientation-swapped programmatically), we cannot rotate orientations at eval time — we accept the position-bias caveat and log it in Notes; the treated-vs-Ctrl comparison remains valid because ALL three arms see identical prompts.

3. **steering-coefficient-tuning** — M2 dose sweep with `alpha ∈ {-2, -1, -0.5, 0, 0.5, 1, 2} × per-layer activation std`.
   - convention to adopt: dose expressed in σ_proj units (plan already does this — activation std normalization). Include α=0 baseline. Report BOTH target metric (QA_I acc) AND general ability (MMLU-lite + helpfulness-lite off-target eval — plan already specifies this). Prefer smallest sufficient α. Re-tune whenever site/direction changes. Load full tip before implementing M2.

4. **steering-block-selection** — M1 finds the target site across all 34 language-tower layers; M2 intervenes at the M1-selected site.
   - convention to adopt: screen across all layers (plan does this in M1). Match-to-claim rule: mechanism claim is about "some component in the language tower" — must be tested at the M1-located site AND control at random layers matched for norm. If single-layer inert, widen to 3–5 adjacent layers. Load full tip before implementing M2.

## Composition
- Tip 1 (finetune) runs BEFORE M0 verdict: an under-tuned LoRA is indistinguishable from "phenomenon absent". M0's plan-declared LR sweep across 7 LRs IS the tip-mandated LR-first sweep. If NONE of the 7 LRs achieves the ≥3pp dual-drop, verdict is `inconclusive`, not `not-established` — the tip mandates capacity/method re-sweep (bump α/r pair, sweep effective batch) before settling on a negative.
- Tip 2 (MCQ) applies to every QA_I eval throughout M0/M2. Ensure judge configuration is FROZEN (same model, same prompt, same temperature=0) across ALL arms.
- Tips 3-4 (steering coefficient + block) apply only if M0 passes and we enter M2. Load their `SKILL.md` files at that point.

## No-match log
- `image` (tip 1) — QA_I images are pre-decoded and shipped in the parquet's `Decoded Image` field; we do NOT apply torchvision transforms — the multimodal processor for Gemma-3-4b-it handles pixel-value construction. No ImageNet convention risk.
