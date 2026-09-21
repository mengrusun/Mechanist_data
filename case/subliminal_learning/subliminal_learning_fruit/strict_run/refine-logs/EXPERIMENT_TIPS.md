# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - finetune-hyperparameter-sweep
  - steering-block-selection
  - steering-coefficient-tuning

## Matches

1. **finetune-hyperparameter-sweep** — plan runs two LoRA-SFT fine-tunes (teacher LoRA on 112 anchor pairs, student LoRA × 16 on N* filtered channel pairs).
   - convention to adopt: LR is task.md-mandated (`teacher lr=2e-4`, `student lr=1e-3`) with **no LR sweep** allowed by HARD constraint 4. `sweep_status: skipped` for both fine-tune milestones (M0.1 + M0.4) — LR pin overrides the tip's LR-first sweep. All other non-negotiables from the tip still bind: α=2r (32=2×16), all-linears target modules (attention + MLP), effective batch=8, AdamW β=(0.9, 0.999) wd=0.0, cosine warmup=5%, grad-clip 1.0, bf16. Diagnostic signals (descent fraction, bouncy std ratio, grad-norm plateau) are still logged in `diagnostics.json` per run to catch a broken fine-tune early — if the pilot's descent_fraction < 0.02 or bouncy_std_ratio > 1.0 on M0.1, M0 is `inconclusive` (broken fine-tune, not "phenomenon absent").
2. **steering-block-selection** — M1 emits a shortlist of DiT blocks × modules × timestep buckets; M2 intervenes on that shortlist.
   - convention to adopt: pick sites by BOTH a screening signal (Grassmann subspace overlap between teacher-arm student ΔW and Ctrl-B student ΔW per block; ConceptAttention saliency delta; residual-stream activation diff at 3 timestep buckets t∈{5,12,20}) AND the mid-to-late heuristic (Qwen-Image has 60 DiT blocks — start screen at mid depth, not restricted). Shortlist size cap ≤ 20% of |blocks × modules × timesteps| (plan HARD constraint). Include a matched-random control site (adjacent block not on shortlist) — supplied automatically by the M1 script.
3. **steering-coefficient-tuning** — M2 sweeps intervention scale ∈ {ablate=0, ×2, ×3, ×4, matched_random}.
   - convention to adopt: express coefficient in `β · σ_l` units where `σ_l = std(h·û)` for the identified direction at the target block; sample-calibrate σ_l per site (4 prompts). Always include β=0 (ablation) as baseline. Also log fluency (fraction of images judged in the on-distribution 9-fruit set excluding 'other') alongside P(banana) — this is the general-ability collapse metric mandated by the tip. Smallest-sufficient rule: report Δ_ablate at β=0; the ×2..×4 dose-response is a monotonicity check, not a "must fully collapse" ask.

## No-match log
- `image` (ImageNet preprocessing): not applicable — we do not run a torchvision ImageNet preprocessing pipeline. Qwen-Image's own VAE encoder handles preprocessing.
- `multiple-choice-evaluation` (letter-parse MCQ): not applicable — the judge outputs a single fruit-word token, not an A-D letter. `qwen_common.judge_one()` already parses tokens robustly and defaults to 'other' on failure (never inflates banana count).
