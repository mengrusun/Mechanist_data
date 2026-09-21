# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - finetune-hyperparameter-sweep
  - multiple-choice-evaluation

## Matches

1. **finetune-hyperparameter-sweep** — LoRA-SFT fires on two distinct `(base_model, training_data)` pairs: (a) teacher SFT on `teacher_anchor_sft.json` (LR 2e-4, hard-coded, no sweep planned) and (b) student SFT on filtered teacher-generated corpus (wide LR sweep `[5e-6..2e-3]` × 3 seeds, PLUS a Ctrl-B student SFT on `filtered_base.jsonl` at top-3 LRs × 3 seeds). Same base (Qwen3.5-9B) but different data, so per Scope callout each is a distinct fine-tune with its own `sweep_status`. Student SFT already declares a full grid `swept`; teacher SFT is single-config so needs `sanity_checked` at pilot scale.
   - Convention to adopt: (i) student LR grid `[5e-6, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]` well covers the LoRA-SFT recommended band `{5e-5, 1e-4, 2e-4, 5e-4, 1e-3}` — proceed as planned (`sweep_status: swept`); (ii) teacher SFT at fixed LR 2e-4 must satisfy Diagnosing A–D on its training-side trace to earn `sweep_status: sanity_checked` — verify (a) smoothed-loss descent ≥ 30 % of initial loss, (b) grad-norm not dead, (c) not base-lookalike, (d) not verbatim-recall; (iii) LR-first iteration order — if M0 fails, adjust LR first; (iv) student SFT uses `α = 32, r = 16` (α/r = 2), matches non-negotiable `α/r ≥ 1`; (v) effective batch ≤ 32 for LoRA SFT, ensured by our batch config; (vi) universal: warmup 3-10%, bf16, AdamW, grad-clip 1.0 — enforced.

2. **multiple-choice-evaluation** — QA_I eval reads a letter out of a free-form generation from a multimodal Qwen3.5-9B with a gpt-5.4 judge already specified in the plan (task.md pins the judge). Fires per description "any per-choice rate computed from a letter parse". Composition rule fires: fine-tuned student + MCQ letter-parse eval → letter-only obedience often breaks post-SFT, exactly where regex would silently fail.
   - Convention to adopt: (i) three-way `{CORRECT, INCORRECT, OTHER}` verdict via gpt-5.4 judge (never coerce refusal / off-topic into INCORRECT); the loose1 sibling's `judge_qai_match` implementation already does this — reuse the shape; (ii) persist per-row `(prompt, gold_letter, raw_generation, judge_verdict)` so audits don't need re-inference; (iii) freeze judge config across arms (model=gpt-5.4, temperature=0.0); (iv) report OTHER-rate per arm as a diagnostic — a rising OTHER rate in treated is a coercion-artifact signal; (v) A/D orientation rotation is recommended but our QA_I parquet has fixed option orderings — instead we spot-check the effect direction with an ad-hoc 100-item rotation as a diagnostic, and flag `suspected_parser_artifact` if the direction flips (noted here as a mild risk; the primary MCQ safeguard is the LLM-judge with OTHER bucket, not orientation rotation).

## No-match log

- **image tip** — no ImageNet / torchvision backbone / T.Compose is used; QA_I loads its own images from parquet and passes them through the multimodal preprocessor.
- **steering-coefficient-tuning / steering-block-selection** — no steering intervention is in scope for Phase 2 (they will fire in M2 if M0 establishes; deferred to Phase 1.5 mechanism-routing time).
