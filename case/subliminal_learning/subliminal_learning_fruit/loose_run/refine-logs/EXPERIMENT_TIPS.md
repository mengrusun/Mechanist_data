# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - finetune-hyperparameter-sweep
  - steering-block-selection
  - steering-coefficient-tuning

## Matches

1. **finetune-hyperparameter-sweep** — the plan runs two distinct LoRA fine-tunes (M0.1 teacher LoRA on 112 anchor pairs; M0.5/M0.6 student LoRA on filtered teacher channel) with the LR-first sweep {1e-4, 5e-5, 1e-5, 5e-6} on the student.
   - convention to adopt: LR-first sweep protocol; two distinct `(base_model, training_data)` pairs → each needs its own `sweep_status`. **M0.5 is the sweep itself → `sweep_status: swept`.** M0.1 uses a single reference config (LR=1e-4, rank=32, 300 steps) → `sweep_status: sanity_checked` after the sanity-run confirms training-side signals A–D pass. Cross-check: Tip's tabled LoRA-SFT LR grid is `{5e-5, 1e-4, 2e-4, 5e-4, 1e-3}` — the plan's grid tops out at `1e-4` (low end of the tip's healthy band `[1e-4, 5e-4]`). If M0 lands `inconclusive`/`not-established` per Phase 1.25, extend the LR grid upward per the tip's iteration protocol (LR first, then capacity).

2. **steering-block-selection** — M1's Location step selects `(layer × site × timestep)` triples on Qwen-Image MM-DiT, and M2's Intervention operates on the M1 shortlist.
   - convention to adopt: **Site set locked by M1 shortlist first (screening via weight-space delta + probe AUC), then M2's coefficient sweep runs on that locked site set.** Match-to-claim rule honored: C2's claim is regional ("some compact set of DiT sites × timesteps"), and M1 sweeps ≥ 3 timesteps × all layers × all site types (residual/attention/MLP) with matched Ctrl-A-vs-Ctrl-B null baselines. Never copy a raw layer index across models of different depth.

3. **steering-coefficient-tuning** — M2's steering variant sweeps α ∈ {-2, -1, -0.5, 0, 0.5, 1} (per the plan's dose-response protocol).
   - convention to adopt: **α = 0 baseline included (present in the plan). Fluency / general-ability metric alongside target metric (present in the plan as off-target CLIP-Score / VAE-MSE, per C3 specificity bar).** Adopt σ_proj units when the steering direction is extracted (`β · σ_proj`, `σ_proj = std(h·u)`) so the plan's α range translates to normalized units at implementation time. Prefer the smallest α that meets the target. Since M2's site set is locked *after* M1 (Composition rule respected), the coefficient sweep runs on the locked site set.

## General Rule for mechanism/Interpretability (unconditional load)

**Locate the neuron/feature for the target function (banana-preference direction), then intervene.**
- M1 (Location) discovers the shortlist — no pre-existing SAE labels for this DiT × banana-preference target, so localization is required.
- M2 (Intervention) intervenes on the M1 shortlist. **Always measure general ability in parallel with the target metric.** Target metric = `P(banana)`; general-ability metric = off-target CLIP-Score against neutral fruit prompts and/or VAE MSE against Ctrl-B outputs (already in the plan as C3's specificity bar #3). A full breakdown into garbled/nonsense generations would invalidate any P(banana) movement as an artifact.

## No-match log

- **image (ImageNet preprocessing)** — did not fire. No torchvision T.Compose pipeline, no vision-backbone activation hooks, no top-k maximally-activating-image labeling. The image generation is text-conditioned DiT, not a classification/CNN activation study.
- **multiple-choice-evaluation** — did not fire. The judge scores `is_banana ∈ {0, 1}` on generated images via a fixed-template gpt-5.4 call, not a letter-parse over a free-form model generation. No regex `[A-D]` parsing anywhere in the pipeline.
