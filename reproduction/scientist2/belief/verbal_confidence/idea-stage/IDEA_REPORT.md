# Idea Report — Captured Behavior

**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: task.md (faithful capture)
**Date**: 2026-07-13
**Pipeline**: research-lit → faithful behavior capture → mechanism-explore (strategy) → research-refine-pipeline

---

## Executive Summary

The task pins **one** behavioral claim about verbalized confidence in decoder-only LLMs — that when the model is asked to verbalize its confidence after answering, the confidence value is *not* freshly computed at the moment of verbalization but is *written into hidden states immediately following the answer* and *retrieved from that cache* when the model emits the confidence token. The claim is concrete and falsifiable and is captured faithfully below; the refined proposal (`refine-logs/FINAL_PROPOSAL.md`) and experiment plan (`refine-logs/EXPERIMENT_PLAN.md`) verify it end-to-end via a Location → Causal Intervention chain on Gemma-3-27B (62 layers) on TriviaQA.

---

## Literature Landscape

See `idea-stage/LANDSCAPE.md` for the full landscape and structural gaps. Highlights (allowed pre-cutoff literature only):

- **Behavioral background**: verbal confidence is a real, useful uncertainty signal (Tian 2023, Xiong 2023, Lin 2022, Yang 2024) but its internal mechanism has not been resolved in the allowed literature.
- **Probing precedent**: self-knowledge / truth signals are linearly decodable from hidden states (Kadavath 2022, Azaria & Mitchell 2023, ICR-Probe 2025).
- **Causal-intervention grammar**: ROME 2022 causal tracing + activation patching (Heimersheim & Nanda 2024) + activation steering (van der Weij 2024) provide the standard tools.
- **Nulls to rule out**: recall-strength confound (arXiv:2510.09033) and log-prob restatement confound.

---

## Claims to Verify

### Claim 1: Verbalized confidence is cached mid-generation, not computed on demand

**Original (verbatim excerpt from task.md):**
> When an LLM is asked to verbalize its confidence after answering, the confidence value is not freshly computed at the moment of verbalization; instead, it is written into hidden states immediately following the answer and is later retrieved from that cache when the model speaks the confidence token.

**Extracted statement**: For a decoder-only LLM asked to (i) answer a question then (ii) verbalize a confidence value about that answer, the residual-stream hidden states at the **post-answer positions** (the small window of tokens immediately following the answer, before the confidence token) already carry a representation that determines the eventual verbalized confidence value, and this representation is *causally used* by the confidence-generation step — as opposed to the confidence value being computed *at* the confidence-generation step from token log-probabilities or a fresh reading of the answer.

**Hypothesis**: H1 — There exists an answer-adjacent position band (post-answer boundary tokens, some layer range) at which a linear-decodable representation of the eventual verbalized confidence value is already present, and causal interventions at that band change the verbalized confidence in a directed, dose-responsive, specific way; interventions at matched non-cache positions do not.

**Measurable predicate**:
- (P1 — Location, correlational) A linear probe trained on residual-stream activations at post-answer positions predicts the model's verbalized confidence value with accuracy substantially above a chance / log-prob-only baseline (target: R² or Spearman ρ against verbal confidence noticeably higher than the log-prob-only baseline, on a held-out split of TriviaQA).
- (P2 — Sufficiency, causal) Patching the residual stream at the identified post-answer position band from a "corrupted" run (answer paraphrase / different-question run) into a "clean" run measurably shifts the verbalized confidence token distribution in the predicted direction.
- (P3 — Retrieval path, causal) Blocking attention from the post-answer band to the confidence-generation position collapses the verbalized confidence toward a prior (uniform-ish / mean); the same block on matched non-cache positions has a much smaller effect.
- (P4 — Steering, dose-response) Adding a scalar multiple of the identified confidence direction at the post-answer band produces a monotone dose-response in the verbalized confidence value, with the expected sign.
- (P5 — Specificity) The interventions above do not disrupt answer accuracy at rates comparable to their effect on verbalized confidence; matched non-cache-position patches produce small effects; recall-strength null and log-prob-restatement null are ruled out (see FINAL_PROPOSAL.md §Controls).

**Expected direction**:
- P1: probe R² clearly above the log-prob-only baseline.
- P2: post-answer patch shifts verbal confidence toward the patched-in prompt's confidence.
- P3: attention-block at cache → confidence collapse; block at controls → small change.
- P4: monotone dose-response with correct sign.
- P5: answer accuracy roughly preserved; recall-strength null and log-prob null falsified.

**Resources (preferred, cost-aware)**:
- Model: **gemma-3-27b-pt** (62 layers) — HARD-pinned for the main experiment. Do not swap or downscale.
- Dataset: **TriviaQA** — HARD-pinned for the main experiment. Do not swap.
- Layer coverage: 62-layer model → probe / intervene at layer indices covering early / middle / late bands. Percentile mapping: **~L{5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60}** (roughly deciles + finer sampling in mid-late). The exact set is bound at Phase 1.5 of the experiment stage per `method_sensitive` fields.
- Verify-stage cross-model swap candidate (informational; not for main): **Qwen 2.5 7B** (28 layers).
- `used_n` on TriviaQA: exact size resolved in Phase 4.5 / experiment stage; floor is a per-condition sample size that clears noise for a linear probe and lets patching effects reach significance (target ≥ ~500 examples per split, subject to the 10 GPU-hour budget).

**Status**: pending verification

**Notes**: Behavior taken as given by prior work (verbal confidence is a documented, reproducible black-box uncertainty signal — Tian 2023, Xiong 2023, Yang 2024). Because `BEHAVIOR_SOURCE=given`, no M0 phenomenon-validation gate is inserted; the plan goes straight to the mechanism ladder. Only one claim is extracted — task.md states a single, unified hypothesis with linked sub-parts (a) written-in and (b) retrieved. Splitting them at the *claim* level would over-decompose; they are handled as **milestone-level sub-parts** in `EXPERIMENT_PLAN.md`.

---

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach for the single claim)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim sub-parts each verifies)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md` (plan-level pending rows)

---

## Next Steps

- [ ] `/mechanism-skills` — route the Location + Causal-Intervention testing approach to a concrete mechanism family + submethod (Workflow 1.25)
- [ ] `/auto-experiment` — implement and run the verification suite (Workflow 1.5)
- [ ] `/auto-verify` — stress-test the verified claim under method / dataset / model swaps (Workflow 1.75; Qwen 2.5 7B available as the cross-model swap)
- [ ] `/auto-iteration-loop` — iterate the verification suite until reviewer-ready (Workflow 2)
- [ ] Or invoke `/auto` for the autonomous claim → routing → experiments → verify → review chain
