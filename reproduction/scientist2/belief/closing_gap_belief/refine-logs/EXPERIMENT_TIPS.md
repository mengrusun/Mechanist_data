# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - general-rule-mechanism-interpretability
  - steering-block-selection
  - steering-coefficient-tuning

## Matches

1. **general-rule-mechanism-interpretability** — mechanism/interpretability experiment (linear probing + activation steering on residual stream)
   - convention to adopt: (a) locate the target function first (linear probes → v_c, v_v at L*), then intervene (steering at L*). (b) When steering, always measure a **general-ability / fluency metric** in parallel — the plan already ships a perplexity safety cap (>3× baseline → halve α); we ADD explicit unsteered-vs-steered mean token perplexity logging as the fluency metric, so a null cross-steering result cannot be confused with silent generation collapse.

2. **steering-block-selection** — B3 declares steering at site `L*` (single-block on residual stream)
   - convention to adopt: (a) L* selected by activation-based screening (mean of normalized AUROC across both binary probes) — matches plan §6.8 exactly. (b) Match-the-claim rule: the plan's C3a claim is layer-local (at L* + neighborhood L*±2), so a single-block steering site is the correct match — B2 already reports neighborhood-robustness in |cos| trajectory. (c) If single-block steering is inert (Δ_random_direction < 0.5 σ_probe_readout), the plan's absolute-effect criterion already kicks in — no need to widen to 3–5 layers preemptively.

3. **steering-coefficient-tuning** — B3 declares α ∈ {−1σ, 0, +1σ} × 3 directions × 500 samples = 4500 forward passes
   - convention to adopt: (a) α expressed in units of σ_L\* (std of residual-stream magnitude) — matches plan §6.9 exactly. (b) `α = 0` baseline included in the grid — matches plan (the 0 entry gives the unsteered reference). (c) Log a **fluency metric** (mean token perplexity, coherence-check) alongside the target Δ probe-readout — required by both this tip and the General Rule. If mean perplexity > 3× unsteered baseline, halve α and re-run (plan §6.9 perplexity safety cap). (d) Smallest-sufficient-β rule — the plan's α=±1σ is already at the small end of {0, ±0.25, ±0.5, ±1, ±2, ±4}; if the ratio criterion cannot fire because random-direction effect is too small, we do NOT auto-escalate to ±2σ — the plan's absolute-effect criterion handles it. This preserves the small-α discipline.

## Composition applied
- Site set (Tip 3) locked FIRST at L* (activation-screened) → coefficient sweep (Tip 2) at that locked site. Order matches the tip file's Composition rule.
- If a null cross-steering result appears at α=±1σ AND fluency is intact (no >3× perplexity blow-up), report it honestly per the General Rule ("target metric moves while general ability intact — valid result"). Fluency collapse would flag the run as `suspected_off_distribution`, not a supporting result.

## No-match log
- Tip 1 (ImageNet preprocessing): N/A — text-only QA.
- Tip 4 (fine-tuning hyperparameter sweep): N/A — no fine-tuning in the plan; only linear-probe fitting + additive steering. The plan explicitly rejects "Tuning & Editing" as a mechanism direction.
- Tip 5 (multiple-choice evaluation): N/A — TriviaQA is free-form short-answer, scored against alias sets (not MCQ letter-parse). The plan's failure-mode diagnostics (label-quality audit, parseable-rate check) already address the analogous silent-failure surface for free-form answers.
