# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - finetune-hyperparameter-sweep
  - steering-coefficient-tuning
  - steering-block-selection

## Matches

1. **finetune-hyperparameter-sweep** — Trigger: plan runs LoRA-SFT on Qwen-Image DiT (M0.1 anchor teacher, M0.4/M0.5 student, M1.3 rank sweep). Multiple fine-tunes with different `(base_model, training_data)` pairs.
   - Convention to adopt: **LR-first sweep per fine-tune pair**. M0.1 (teacher anchor) uses `sanity_checked` (48-image judge sanity is the diagnostic). M0.4 IS the LR-first sweep with 5 LRs × 3 seeds — this satisfies `sweep_status: swept` for the student fine-tune. Pilot artifacts persist in `runs/M0.4_*_pilot/`. Diagnostic signals (A–C: shallow descent, dead grad-norm, bouncy loss, base-lookalike output) monitored inline via loss/gradnorm logs. **Iteration rule**: on M0.5 inconclusive/not-established verdict, change LR first before rank/α (§Non-negotiables). Enforced 5-attempt budget cap for Phase 1.25 fixes.

2. **steering-block-selection** — Trigger: M1.1 (Locate) ranks candidate blocks (`ranked_candidates`) and M1.2 (Verify causally) intervenes on top-1/top-3 with `--sites ${top_candidate.sites}`.
   - Convention to adopt: **Lock site set BEFORE α sweep**. Screen with activation-difference AND LoRA-SVD; mid-to-late DiT blocks first (Qwen-Image transformer has ~60 double-stream blocks — screen at spaced intervals `[10,20,30,40,50]` as coarse, then narrow). If single-block intervention shows no effect at M1.2, widen to 3–5-block window matched to the same relative-depth region. Add matched null-control window elsewhere for specificity.

3. **steering-coefficient-tuning** — Trigger: M1.2 steering-α sweep declares `α ∈ {−2, −1, 0, +1, +2, +3}` (already covers `α = 0` baseline).
   - Convention to adopt: **Express α in σ_proj units** (compute `σ_l = std(h_lᵀ u_l)` on a fixed prompt batch and report both raw and σ-normalized α). Score BOTH target metric (P(banana) shift) AND fluency/general-ability metric (fraction of judge verdicts in `{apple, orange, grape, pear, strawberry, lemon, peach, watermelon}` — the on-distribution fruit set — must stay within 3pp of baseline per plan Spec §B.2). Prefer smallest α that meets the target. Re-tune coefficient whenever the site set changes.

## No-match log
- Tip 1 (ImageNet Eval Preprocessing) — N/A. Qwen-Image is a diffusion generator; no torchvision T.Compose pipeline; eval is judge-scored image generation, not classification.
- Tip 5 (Multiple-Choice Evaluation) — N/A. Eval scorer is a single-word 10-way judge over generated *images*, not a letter regex over LLM text. Judge protocol is already spec'd (gpt-4o with fixed prompt, `other` bucket present).

## Composition order (per experiment-tips SKILL.md §Composing Tips)
1. **fine-tuning sweep first** — M0.4 completes and picks BEST_LR before M0.5 (full-scale retrain).
2. **block selection next** (M1.1 → M1.2 site fixation).
3. **coefficient tuning last** (M1.2 α sweep on locked sites).
