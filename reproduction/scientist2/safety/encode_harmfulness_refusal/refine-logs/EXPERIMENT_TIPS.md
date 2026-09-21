# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-block-selection
  - steering-coefficient-tuning

## Matches

1. **steering-block-selection** — Plan runs a full layer sweep (M-prep) and a position ladder (M2) for choosing where h and r live; M3 additive steering hooks act at that chosen layer × position.
   - convention to adopt: pre-screen layers by an activation signal (probe AUROC — already the plan's diagnostic); prefer mid-to-late layers; if the single best layer shows no M3 effect, widen to a 3–5-layer window (per tip's fallback). Match-to-claim rule: the plan's regional claim is single-position × single-best-layer, so use the single-site hook M3 declares — no ad-hoc widening unless the single-site null triggers.

2. **steering-coefficient-tuning** — Plan pins `α ∈ {-2, -1, -0.5, 0, +0.5, +1, +2}` in units of `‖d‖` (the direction's L2 norm).
   - convention to adopt: (a) also compute `σ_l = std(h_l^T u_l)` on the direction-extraction split and report a companion `α_σ = α_‖d‖ × ‖d‖ / σ_l` for cross-layer / cross-direction comparability; (b) plan already includes α=0 baseline; (c) log a **fluency / general-ability metric** alongside the refusal metric in M3 (per-token mean log-prob, or completion-length + repetition rate) to detect off-distribution collapse; (d) prefer the smallest sufficient `|α|` when reporting the "target axis flipped" result; (e) if α=+2 already produces gibberish / repetition (fluency collapse), that grid point is off-distribution and its target-axis reading is an artifact — flag in M3 verdict.

## No-match log
- Tip 1 (ImageNet Eval Preprocessing) — no vision backbone, no torchvision transforms. Not applicable.
- Tip 4 (Fine-Tuning Hyperparameter Sweep) — no fine-tuning in the plan (only forward passes + linear probes + additive-steering hooks). Not applicable.
- Tip 5 (Multiple-Choice Evaluation) — no A/B/A-D letter-parse on free-form generation. Refusal detection uses a canonical-string classifier + Llama Guard 3 8B judge on the full response, both scored as binary; no letter regex, no MCQ. Not applicable.
