# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - finetune-hyperparameter-sweep
  - steering-block-selection
  - steering-coefficient-tuning
  - multiple-choice-evaluation

## Matches

1. **finetune-hyperparameter-sweep** — plan runs LoRA-SFT on both teacher (`AutoModelForCausalLM`, all text layers) and student (`AutoModelForImageTextToText`, language tower only) with fixed hyperparams (`lr=2e-4` teacher, `lr=1e-3` student, `r=16 α=32` both).
   - convention to adopt: task.md hard-locks the LoRA / LR / batch / epochs so no LR grid sweep is allowed (a constraint conflict). Instead, mark **`sweep_status: sanity_checked`** on both fine-tunes; run one preflight per (base_model, training_data) pair — teacher-SFT on anchor data (one pair) and student-SFT on filtered teacher-generated data (a different pair, same base). Per-pair pilot artifacts (loss curve, grad-norm trace, base-match rate on ≥ 50-prompt held-out slice) written under `runs/M0_S0a_teacher_sft/pilot/` and each `runs/A_M0_S3_student_sft_*/pilot/`. Iterate hyperparameters only if a milestone's own declared pass criterion misses AND `task.md` fixed-config constraint yields under-fit signals — surface as Round-End Decision.

2. **steering-block-selection** — M2.2b steering intervenes on the top-3 language-tower layers selected by M1.L-Core (contrastive activation direction extraction) → residual stream, mid-to-late layers.
   - convention to adopt: lock the site set via M1.L-Core Borda-ranked top-3 layers FIRST (already in plan); use the mid-to-late layer heuristic to sanity-check the winning layers land in the 40 %–80 % depth range (i.e., layer index ≈ 12–26 of 32); if all top-3 land at extreme layer 0 or 31, flag suspicious.

3. **steering-coefficient-tuning** — M2.2b sweeps α ∈ {-2, -1, -0.5, 0, +0.5, +1, +2} on residual direction.
   - convention to adopt: (a) express α as multiples of σ_proj (per-layer std of h·û on the base activation cache) — will normalize direction jointly across top-3 layers so grid stays comparable; (b) include α=0 baseline (already in plan); (c) log a fluency / general-ability metric alongside QA_I accuracy — reuse `OTHER` verdict rate + `off_target eval_pairs_948.json` accuracy in M2.2c as the collapse-detector; (d) prefer smallest sufficient |α| in the STRONG POSITIVE verdict discussion.

4. **multiple-choice-evaluation** — QA_I is A-D letter task graded from free-form generation via gpt-5.4 judge (task.md prescribes CORRECT / INCORRECT / OTHER, three-way, no coercion).
   - convention to adopt: (a) three-way {CORRECT, INCORRECT, OTHER} verdict already in plan — do NOT collapse OTHER into INCORRECT; report separately; (b) persist per-row `(id, gold, generation, judge_raw, verdict)` to `results/eval/*.json` (already in eval script); (c) M0.S5 already covers judge calibration (paraphrase + arm-order stability) — this is the parser-artifact diagnostic; (d) A/B orientation swap is NOT in scope (QA_I is fixed A-D with images; option order rotation would break image references); document this in results as a known scope limitation and rely on M0.S5 as the primary parser-artifact defense.

## No-match log

- No ImageNet / torchvision preprocessing (QA_I images are pre-embedded PNGs handled by AutoProcessor → no `T.Resize` decision).
