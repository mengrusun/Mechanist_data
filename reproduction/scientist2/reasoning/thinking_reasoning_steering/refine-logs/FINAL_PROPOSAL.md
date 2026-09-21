# Final Proposal — Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill

**Date**: 2026-07-14
**Behavior-source**: given (from `task.md`)
**Mechanism**: discovery
**mechanism_strategy**:
```yaml
directions: [Location, Causal Intervention, Tuning & Editing]   # in execution order
rejected:
  - Formation Tracing — task.md makes no genesis claim; how the behaviour directions formed during training is out of scope and prohibitively expensive under the 10-hour GPU budget.
  - Unit Interpretation — the claim is about mean-difference / linear-probe direction control, not decoding a monosemantic feature; SAEs / auto-interp would add cost without changing any of C1–C4.
  - Decision Auditing — the claim is about behaviour amplitude and downstream accuracy, not about whether a specific decision is spurious.
note: The three chosen directions map 1:1 onto C1 (Location — linearity of behaviour direction), C2 (Location — small-pool extractability), C3 (Causal Intervention — dose-response steering), C4 (Tuning & Editing — usable knob vs. prompt engineering at preserved accuracy).
```

## Problem Anchor

Reasoning-tuned "thinking" LLMs (DeepSeek-R1 distills, o1-style, Qwen-Thinking) generate long chains of thought before answering. Distinct in-chain reasoning *behaviours* — expressing uncertainty, generating validation examples, backtracking, self-correction — heavily influence final accuracy but are hard to control from the prompt alone (prompting is coarse; retraining is expensive). `task.md` pins the hypothesis that each behaviour occupies an approximately linear direction in the residual stream, extractable from a small pool of contrastive activation pairs, and dose-response steerable via a single scalar α while preserving downstream reasoning accuracy — and finer-grained than prompt engineering. The final proposal *tests* that hypothesis; it does not re-invent it.

## Final Method Thesis

**Unified testing method: contrastive-pair activation steering with layer-swept mean-difference directions**, evaluated jointly for (a) linearity (linear probe + first-PC alignment), (b) small-pool extractability (n_pairs sweep + split-half stability), (c) dose-response causal control (α sweep + off-target specificity), and (d) usability vs. prompt engineering + Thinking Intervention on the same 500-task benchmark under a fixed reasoning-accuracy budget. The concrete extraction / intervention *submethod* (pure mean-of-differences CAA vs. logistic-probe direction vs. optimised single-vector) is deferred to `/mechanism-skills` at the experiment stage — the plan carries `method_sensitive: [n_pairs, sites, metric, gpu_hours]` on every intervention milestone to license the routing.

## Dominant Contribution

An end-to-end verification of the four sub-claims C1–C4 for the *reasoning-behaviour* class in one specific, publicly-available thinking model (DeepSeek-R1-Distill-Llama-8B), on a fixed 500-task reasoning benchmark, with the joint (behaviour-rate, accuracy) Pareto comparison against both natural-language prompt engineering and Thinking Intervention as competing controllers.

## Explicitly Rejected Complexity

- Sparse-autoencoder feature dictionaries and auto-interp on the behaviour directions (out of scope — not what the claim asserts).
- Training-time formation of behaviour directions across R1 distillation checkpoints (out of scope; not part of the claim; well beyond the 10-hour budget).
- Weight editing or LoRA-based behaviour control (Tuning & Editing here means the applied steering knob, not model editing).
- Attention-head localisation and circuit-discovery to layer-attribute the behaviour beyond a residual-stream direction (would extend Location into finer grain than the claim asks for).

## Key Claims (verbatim from task.md, captured in `idea-stage/IDEA_REPORT.md`)

- **C1** — Each behaviour maps onto an approximately linear direction in the residual stream.
- **C2** — Each behaviour direction is extractable from a small pool of contrastive activation pairs.
- **C3** — Adding / subtracting the vector at inference amplifies / suppresses the behaviour dose-responsively via a single scalar α.
- **C4** — Steering-vector control is finer-grained than prompt engineering, at preserved reasoning accuracy.

## Must-run Ablations / Controls (surfaced into `EXPERIMENT_PLAN.md`)

- **Layer sweep for L*** (M1) — establish that the linearity is not a lucky-layer artefact.
- **n_pairs sweep** (M2, C2) — establish sample efficiency.
- **α sign check + monotonicity + off-target specificity** (M3, C3) — establish causality of the direction.
- **Prompt-engineering + Thinking Intervention Pareto** (M4, C4) — establish granularity + accuracy preservation.
- **Behaviour taxonomy annotation is grounded once** (M1 sub-step) — the external LLM is the fixed annotator; behaviour-tag prompt is versioned once and reused everywhere so all rate comparisons share a common ruler.

## Remaining Risks

- **R1 — LLM-as-judge noise.** All behaviour-rate + free-form accuracy scores lean on the external LLM. Mitigation: fix the judge (Claude 3.5 Sonnet OR GPT-4o via `task.md`'s DMX endpoint), version the annotation prompt, and report inter-run judge stability on ≥ 10% of chains as a sanity check.
- **R2 — Steering saturation / gibberish at large |α|.** Mitigation: bound the α-grid at ±3σ (residual-scale) and report both perplexity + a "chain-is-still-coherent" LLM flag alongside the behaviour rate.
- **R3 — Behaviour cross-talk.** Off-target behaviour rate is measured in M3 to establish specificity; if a behaviour cannot be moved without moving another, the specificity claim is limited to the pairs that separate.
- **R4 — Compute headroom.** M1–M4 budget targets ≤ 5.5 GPU-hours (of 10 total), leaving ≥ 4.5 GPU-hours for the verify + iteration stages.

## Frontier Primitive Check

No frontier primitive (SAE / mixture-of-experts feature attribution / probing-classifier meta-tricks) is *required* by the claim — the mean-difference / linear-probe extractor is the appropriate simplicity, and the plan explicitly rejects heavier extractors unless the experiment-stage `/mechanism-skills` routing surfaces a signed reason to prefer one.

## Final Verdict

**READY** — proposal is claim-anchored, minimal, and downstream-plan-ready.
