# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - image
  - finetune-hyperparameter-sweep

## Matches

1. **image** — vision backbones (DINOv2 ViT-B, SigLIP-So400m) run on THINGS and ImageNet subsets in every milestone. torchvision `T.Compose` / `Resize` / `CenterCrop` will be written in `code/data_utils.py`. Silent `Resize(int)` vs `Resize((int,int))` drift would cause different pixel crops for DINOv2 vs SigLIP and shift THINGS / ImageNet activations.
   - convention to adopt: **Use each backbone's native HuggingFace `image_processor`** — DINOv2 (`facebook/dinov2-base`: shortest side 256 → center-crop 224 → ImageNet mean/std) and SigLIP-So400m (`google/siglip-so400m-patch14-384`: 384×384 square → SigLIP mean/std). This is the "model's own image_processor" default from the tip's Pick-a-Convention table — required whenever we report accuracy with the trained weights. Both preprocessing paths are pinned to native processors; no ad-hoc `T.Resize(256)` calls anywhere in `code/`.

2. **finetune-hyperparameter-sweep** — M3 (aligned DINOv2 finetune, `--align_loss triplet_kl --alpha 1.0 --tune_scope full --lr 5e-5`) and M5 (specificity control finetune, same config, unaligned SigLIP teacher). Two distinct fine-tunes (same base model DINOv2 ViT-B, **different training targets** — aligned-teacher pseudo-labels vs. unaligned-teacher pseudo-labels), so each needs its own pilot / sanity check under the Scope rule ("one pilot per fine-tune"). No standard SFT/DPO/GRPO objective — this is vision-KD with KL loss, so the LR grid table for LLM SFT is only a rough guide (KD on ViT with SigLIP teacher typically uses `1e-5 – 5e-4` full-FT LR; `5e-5` is inside the reasonable band).
   - convention to adopt: **`sweep_status: sanity_checked`** for M3 and M5 — under the 10-hr budget a full LR grid (5 configs × 2 milestones × 2.5 hr = 25 hr) is infeasible. Run one micro-pilot per milestone at `lr=5e-5` (~500 examples, 1 epoch, ≥30 optimizer steps, 10% held-out) verifying Preflight + Branch-2 A–D signals: descent > 30% of initial loss, grad-norm > 0.05 after warmup, no NaN, no base-lookalike output on 20 held-out THINGS triplets. If any signal fires, halve LR (Non-negotiable Iteration Order rule: LR first) and re-check within the 5-attempt budget. Record `sweep_status: sanity_checked (pilot: runs/M3_pilot/, lr=5e-5, r=n/a, batch=512, all A-D passed)` inside the milestone's `hyperparameters:` block in the plan-audit-trail sense (kept in `runs/M3_pilot/summary.json` since we do not rewrite `EXPERIMENT_PLAN.md` — plan editing is claim-stage owned).

## No-match log

- **steering-coefficient-tuning** — no additive intervention with a strength parameter (α in the plan is a *loss-weight coefficient* on the KD term, not a residual-stream-addition dose). Skip.
- **steering-block-selection** — no per-block intervention (full-backbone or LoRA finetune, not a targeted residual-stream edit). Skip.
- **multiple-choice-evaluation** — no letter-parse of free-form generations (triplets are scored by argmax cosine similarity over 3 candidates, not by parsing a generated letter). Skip.

## General Rule for mechanism/Interpretability (always loaded)

- **Locate**: the whole final-embedding representation of DINOv2 ViT-B (no per-neuron/per-head targeting; the alignment loss updates the whole backbone or a LoRA subspace of it).
- **Intervene**: KL-based distillation training with `α * L_align + (1-α) * L_original` (Fail policy allows LoRA fallback with `r=16, α=32`).
- **General-ability guardrail (built into the plan)**: M7 (downstream one-shot classification on 4 datasets) + M8 (OOD sweep on BREEDS + ImageNet-A) explicitly test whether the aligned model preserves *general vision ability*. Claim 4a's non-inferiority predicate + Claim 4b's OOD-improvement predicate are the classic "target moves, general ability intact" pair. No additional guardrail metric needed — the plan already encodes it.
