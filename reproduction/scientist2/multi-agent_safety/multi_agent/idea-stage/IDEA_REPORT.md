# Idea Report — Captured Behavior

**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: task.md (faithful capture — no ideation, no ranking, no novelty check)
**Date**: 2026-07-14

## Executive Summary

`task.md` states a single research hypothesis — *"Group-Aggregated Activation Probes for Multi-Agent Collusion Detection"* — decomposed into three interlocking sub-claims (C1 existence, C2 aggregation, C3 transfer). Because `BEHAVIOR_SOURCE=given`, this report captures the behavior verbatim, splits the paragraph-shaped `## Claim` into three individually-verifiable sub-claim predicates, and hands them to Phase 4.5 as a single bundle. No M0 phenomenon-validation gate is added (per `given` semantics — the behavior is assumed to hold at the claim stage; existence is exactly what C1 tests as a first-class sub-claim, not as a gate).

## Literature Landscape (from Phase 1)

See `idea-stage/LANDSCAPE.md`. Five structural gaps were identified — G1 (no prior group-level probe aggregation for collusion), G2 (zero-shot cross-scenario transfer of internal-state probes unmeasured), G3 (untested complementarity of aggregation techniques), G4 (underspecified text-only monitor baselines for multi-agent covert coordination), G5 (missing non-lying-deception and confound controls specific to collusion). The three sub-claims below each map onto one or more of these gaps.

## Resources (from task.md — cost-aware, not `strict` marker)

- **Primary model (experiment stage)**: `Qwen3-32B-AWQ` — HARD CONSTRAINT (task.md-marked primary; do not swap for a smaller model in the main experiment)
- **Primary dataset (experiment stage)**: `NARCBench-Core` — 50 scenarios × 17 domains, binary committee voting (may be adapted with grounded scientific reason per NOTICE)
- **Verify-stage candidates (informational, choose as needed)**:
  - models: `Llama-3.1-70B-Instruct-AWQ-INT4`, `DeepSeek-R1-Distill-Qwen-32B`, `GPT-OSS-20B`
  - datasets: `NARCBench-Transfer` (6 domain-different families), `NARCBench-Stego` (steganographic blackjack card-counting)
- **Scenario examples (task.md)**: S1 Multi-Domain Committee Deliberation, S2 Blackjack Card-Counting Steganography, S3 Code-Review Sabotage
- **External judge**: `gpt-5.4` via `https://www.dmxapi.cn/v1` (bypass proxy)
- **Compute budget**: 10 GPU-hours total (HARD CONSTRAINT), GPUs restricted to `gpu_id ∈ {1,2,3,5,6}`
- **Data / model roots**: `DATA_DIR=/data/zhenqian/data`, `MODEL_DIR=/data/zhenqian/models`
- **Inference framework**: vLLM
- **Env**: conda

## Claims to Verify

### Claim 1 (C1): Existence — internal signal that text-only monitoring misses

**Original (verbatim excerpt from task.md):**
> Model internals carry a collusion signal that text-level monitoring misses; lightweight probes on per-agent residual-stream activations can recover it.

**Extracted statement**: On Qwen3-32B-AWQ generating a multi-agent NARCBench-Core deliberation, a lightweight linear probe trained on per-agent residual-stream activations detects the ground-truth collusion label at an AUROC that materially exceeds a strong text-only judge baseline (external LLM-as-judge, gpt-5.4) applied to the same transcripts.

**Hypothesis**: H1 — The per-agent residual stream of a colluding agent contains a linearly recoverable feature that a text-only observer of the transcript cannot recover, so a linear activation probe strictly beats the strongest text-only monitor.

**Measurable predicate**: On NARCBench-Core, `probe_AUROC ≥ 0.75` AND `probe_AUROC − text_only_judge_AUROC ≥ 0.05` on a held-out test split.

**Expected direction**: up (probe AUROC > text-only judge AUROC).

**Resources**: model: Qwen3-32B-AWQ; dataset: NARCBench-Core (50 scenarios × 17 domains); external judge for baseline: gpt-5.4.

**Status**: pending verification.

**Notes**: Maps to Gap G1 (no prior group-level probe aggregation for collusion) and Gap G5 (non-lying-deception controls). Because C1 stands on the *existence* of a per-agent signal, it is the natural M1 milestone. The `given` regime intentionally does not gate the pipeline on a separate M0 — C1 itself is a first-class hypothesis test rather than a gate.

### Claim 2 (C2): Aggregation — group-level probes beat single-agent probes and text-only judges, and no single aggregation dominates

**Original (verbatim excerpt from task.md):**
> Detection requires aggregating signals across agents — group-aggregated probes outperform any single-agent probe and the text-only judge — and no single aggregation technique dominates across collusion types, motivating a family of complementary multi-agent probes.

**Extracted statement**: Given per-agent probe scores on a group of agents (typically K agents deliberating one scenario), there exists at least one group-aggregation function (e.g. mean-pool, max-pool, attention-pool, concat-then-MLP, best-of-N) whose group-level AUROC materially exceeds both the best single-agent probe (from C1) and the text-only judge baseline. Furthermore, *which* aggregation is best varies across the 17 NARCBench-Core domains — no single aggregation strictly dominates across all domains.

**Hypothesis**: H2a — At least one group-aggregation of per-agent probe scores exceeds the best single-agent probe by ≥ 0.05 AUROC. H2b — Under permutation across NARCBench-Core's 17 domains, no single aggregation attains rank 1 in > half of the domains (diversity-of-aggregation test).

**Measurable predicate**: (a) `best_group_AUROC − best_single_agent_AUROC ≥ 0.05` on the held-out test split; (b) among the 5 aggregation techniques, no single technique is top-ranked in ≥ 9 of the 17 domains (i.e. the domain-argmax over aggregations is genuinely diverse, not concentrated on one method).

**Expected direction**: (a) up (best aggregation > best single-agent probe); (b) diversity (no single-technique dominance).

**Resources**: model: Qwen3-32B-AWQ; dataset: NARCBench-Core.

**Status**: pending verification.

**Notes**: Maps to Gap G1 (no group-level probe aggregation prior work), Gap G3 (untested complementarity of aggregation techniques), and Gap G4 (needs strong text-only monitor baseline). The `no-single-dominance` sub-predicate is a *diversity* test; if C2 (a) holds but a single aggregation dominates all 17 domains, the family-of-aggregations motivation weakens — this should be reported honestly rather than suppressed.

### Claim 3 (C3): Zero-shot transfer to structurally different multi-agent scenarios

**Original (verbatim excerpt from task.md):**
> Activation probes transfer zero-shot, without retraining, to structurally different multi-agent scenarios (different channel, different domain), showing the learned signal is not tied to one collusion form.

**Extracted statement**: A probe (single-agent from C1, and/or the best aggregation from C2) trained on NARCBench-Core and applied *without any retraining or fine-tuning* to structurally different multi-agent scenarios — NARCBench-Transfer (6 domain-different families: code review, market manipulation, pipeline sabotage, knowledge poisoning, research framing, news narratives) and NARCBench-Stego (steganographic blackjack card-counting with covert signaling) — attains an AUROC well above chance on at least half of the transfer targets.

**Hypothesis**: H3 — The residual-stream direction that the probe reads is a *scenario-general* collusion feature rather than a NARCBench-Core-specific surface artifact, so it transfers to unseen coordination structures at above-chance AUROC.

**Measurable predicate**: On at least 4 of the 7 transfer targets (6 NARCBench-Transfer families + 1 NARCBench-Stego family), zero-shot transferred `AUROC ≥ 0.65`.

**Expected direction**: up (well above chance = 0.5 with margin).

**Resources**: model: Qwen3-32B-AWQ; datasets: NARCBench-Transfer (6 families), NARCBench-Stego (1 family — steganographic blackjack). Training happens on NARCBench-Core only; no retraining on the transfer families.

**Status**: pending verification.

**Notes**: Maps to Gap G2 (zero-shot cross-scenario transfer of internal-state probes unmeasured; the Marks & Tegmark follow-up quantifies a within-model within-topic transfer ceiling, but not cross-coordination-structure transfer). Threshold `0.65` is chosen well above chance (0.5) but below the C1 within-distribution bar (0.75) — a *transfer* is not required to match within-distribution performance to still validate scenario-generality.

## Next Steps

- Phase 4.5 refines a **unified proposal + experiment plan** covering all three sub-claims (no per-claim selection; every one is carried forward).
- Downstream: `/auto-experiment` (Workflow 1.5) implements + deploys; `/auto-verify` (Workflow 1.75) stress-tests via swap variants using the NOTICE-listed candidate models/datasets.
