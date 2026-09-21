# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - finetune-hyperparameter-sweep
  - steering-block-selection

## Matches

1. **finetune-hyperparameter-sweep** — Plan runs two LoRA-SFT fine-tunes (M2 R2D2-style adversarial baseline B1, M3 RR fine-tune) with rank/α hardcoded (r=16, α=32, 500 steps) and NO learning rate specified. Fires the "hard-coded config without a sweep" trigger.
   - convention to adopt: **LR-first sanity check** per fine-tune with the tip's LoRA-SFT grid `{5e-5, 1e-4, 2e-4, 5e-4, 1e-3}`. To stay in the 10 GPU-h budget, run one **sanity-checked** pilot at `lr=2e-4` (widely-validated LoRA-SFT modal on Llama-3-8B under `α/r=2`) per fine-tune, verify tip's Preflight + A-D signals, and only escalate to a 3-LR mini-grid if signals fire. Target modules = all-linears (not attention-only). Effective batch ≤ 32. Warmup 5%, cosine decay, bf16, AdamW, grad-clip 1.0, 1 epoch. Mark `sweep_status: sanity_checked` in the milestone hyperparameters block after pilot passes.

2. **steering-block-selection** — Plan intervenes on residual-stream sites `S` selected by mean-difference + linear-probe AUC at M1, then trains M3 RR against those sites. Fires the "target block / layer / site" trigger. Note: this is a *training-time* intervention, not additive steering — the tip's site-selection heuristic still applies.
   - convention to adopt: **Mid-to-late layers first, 3-5+ sites, match-to-claim.** The plan's default `k=6 contiguous middle-band` (layers 10-20 out of 32) satisfies the heuristic. M4's specificity control (random orthogonal direction `d_ctrl` at each site) satisfies the matched-null-control rule for a regional claim. Do NOT copy the layer index across models — for M6 (Mistral-7B-Instruct-v0.2, 32 blocks — same depth as Llama-3-8B) the mid-band still lands at ~10-20, but re-run M1's per-layer AUC on Mistral before locking sites.

## Mechanism General Rule (mandatory for every mechanism experiment)

Applied unconditionally per `experiment-tips/SKILL.md`:
- **Locate then intervene**: M1 (locate residual-stream sites via probe AUC + mean-diff) → M3 (RR fine-tune on those sites) → M4 (diagnostic on the located sites).
- **Measure target metric AND general ability in parallel**: M5 evaluates HarmBench ASR (target) alongside MT-Bench + MMLU (general ability). Never report ASR alone.

## No-match log
- Tip 1 (ImageNet preprocessing) — no torchvision / ImageNet pipeline.
- Tip 2 (steering-coefficient tuning) — the plan uses training-loss weights (α, β, λ_lm), not inference-time additive steering coefficients on activations. No CAA / DAS / SAE feature scaling in the pipeline.
- Tip 5 (MCQ letter-parse) — MMLU uses standard 5-shot log-likelihood eval (not free-form letter parsing); HarmBench uses a judge model (not a letter regex); MT-Bench uses judge scoring. None of the three eval flows exposes the letter-parse trap.
