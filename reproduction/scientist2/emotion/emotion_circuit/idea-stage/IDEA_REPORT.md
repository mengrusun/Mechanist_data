# Idea Report — Captured Behavior

**Behavior-source**: given
**Mechanism**: discovery (system routes; `/mechanism-explore` loaded — chosen strategy: Location + Causal Intervention + Tuning & Editing)
**Claim source**: task.md (faithful capture)
**Date**: 2026-07-13
**Pipeline**: research-lit → faithful behavior capture → research-refine-pipeline

## Executive Summary

Three claims are captured verbatim from `task.md` about emotion-specific circuits in a decoder-only LLM: (1) such circuits (MLP neurons, attention heads, layers) can be identified via a systematic framework (emotion-direction extraction → per-component identification → global circuit integration); (2) the circuits are stable across scenarios and across emotions, providing mechanistic evidence that emotion generation is supported by traceable, reusable circuits; (3) intervening on the circuit induces target emotions reliably across arbitrary inputs and beats both prompting and single-direction (activation-steering) baselines on emotion-expression accuracy. Verification uses Llama-3.2-3B-Instruct on the SEV dataset (480 events × 6 emotion variants); robustness verification (Verify stage) uses Qwen2.5-7B-Instruct on a held-out SEV split. The strategy shaping the plan is Location + Causal Intervention + Tuning & Editing.

## Literature Landscape

See `LANDSCAPE.md`. Retrieval was **partially degraded** (mechanic-db unavailable; arXiv/S2 rate-limited; several WebSearch pages voided by post-cutoff URL policy). The plan therefore leans on `task.md` as the authoritative source of claims and only on **pre-cutoff canonical** methods with confirmed clean references: RepE (Zou 2023, arXiv:2310.01405) and CAA (Rimsky 2023, arXiv:2312.06681) as the single-direction steering baselines; EmotionPrompt (Li 2023, arXiv:2307.11760) as the prompting baseline; ITI (Li 2023, arXiv:2306.03341) as the head-level probe+intervene template; ROME / knowledge-neuron / IOI / ACDC lines as the neuron-and-circuit template.

## Resources (binding for the reproduction combination — cost-aware here, but full-scale)

`task.md` names resources per-stage and explicitly writes: *"GPU budget is not a constraint — don't drop experiments just because they're compute-heavy. Choose the most suitable methods regardless of compute requirements."* Combined with the 10-GPU-hour envelope, this instructs the plan to run at **full scale on the named resources** — no cost-driven downscaling of models or data. Not the reproduction combo (`MECHANISM=discovery`), so the top-metadata marker `resource_fidelity: strict` is NOT stamped in `EXPERIMENT_PLAN.md`; however, the plan still records `used_n = available_n` on each captured milestone.

| Resource | Value | Where used |
|---|---|---|
| Main-experiment model | Llama-3.2-3B-Instruct (`/data/zhenqian/models/Llama-3.2-3B-Instruct`) | All main runs — Claims 1, 2, 3. |
| Main-experiment dataset | SEV — Scenario–Event with Valence, 480 events = 8 domains × 20 scenarios × 3 outcomes; 6 emotion variants per event (i.e., 2880 event-emotion pairs). Local: `/data/zhenqian/data/SEV/sev.json`. | Direction extraction, component identification, circuit integration, main control eval. |
| Verify-stage model | Qwen2.5-7B-Instruct (`/data/zhenqian/models/Qwen2.5-7B-Instruct`) | Robustness variant (model swap) in the Verify stage. |
| Verify-stage dataset | SEV held-out test set, 480 events (disjoint content). | Robustness variant (data swap) in the Verify stage. |
| GPU pin | `CUDA_VISIBLE_DEVICES=1,2,3,5,6` (subset OK) | All runs. |
| Env | conda env | All runs. |
| GPU budget | 10 GPU-hours total; **do not pause citing budget until actual usage reaches this**, and do not skip strong methods to save cost. | Whole pipeline. |
| Directory limits | Work dir + `/data/zhenqian/data` + `/data/zhenqian/models` only. | All runs. |
| API key (optional judge) | `<Your_api>` @ `https://www.dmxapi.cn/v1` model `gpt-5.4`; **bypass proxy**. | Optional external emotion-expression judge for accuracy scoring. |

## Claims to Verify

### Claim 1: Locatability of emotion-specific global circuits

**Original (verbatim excerpt from task.md):**
> We can identify emotion-specific global circuits in LLMs by constructing a systematic framework that integrates emotion direction extraction, component identification, and global circuit integration.

**Extracted statement**: On Llama-3.2-3B-Instruct evaluated on SEV, a systematic framework — (a) extract a per-emotion direction from paired-contrast activations, (b) identify MLP neurons and attention heads whose activations best carry that direction, (c) integrate the per-emotion components into a *global circuit* — produces an identifiable, per-emotion, sparse-yet-non-trivial set of components (neurons + attention heads + layers). "Identifiable" means the selection is (i) reproducible across seeds / event resamples above chance, and (ii) sparser than "everything" (a few percent of components suffice) while still causally sufficient (see Claim 2).

**Hypothesis**: H1 — There exist per-emotion component sets — MLP neurons and attention heads distributed across a handful of layers — that carry the emotion signal and can be recovered by a probe-then-select framework built from paired-contrast activation differences.

**Measurable predicate**: For each of the six emotions in SEV, the framework produces a component set `C_e = {neurons_e, heads_e, layers_e}` with (i) selection stability ≥ chance under seed / event-subset resampling (measured by Jaccard overlap of `C_e` across resample folds, ≥ some pre-registered threshold above a random-selection null), and (ii) sparsity — |neurons_e| and |heads_e| are each small enough (a few % of totals) that the "global circuit" is non-trivially localized. Downstream Claim 2 tests its causal sufficiency.

**Expected direction**: threshold — the framework returns a non-empty, sparse, and above-chance-stable set for every emotion.

**Resources**: model: Llama-3.2-3B-Instruct (full size); dataset: SEV full 480 events × 6 emotions = 2880 pairs (used_n = available_n = 2880); split — reserve at least a small held-out fold from the 480 for the causal / stability tests in Claim 2; storage of extracted directions and per-emotion component lists.

**Status**: pending verification

**Notes**: Framework decomposition inferred from the task.md sentence structure: "emotion direction extraction" → step (a); "component identification" → step (b); "global circuit integration" → step (c). Because `MECHANISM=discovery`, the *specific* mechanistic submethod (ITI-style head selection vs. ACDC-style circuit discovery vs. knowledge-neuron-style MLP scoring) is deliberately left to the experiment stage's `/mechanism-skills` routing — the claim only asserts the *kind* of circuit exists, not any specific head/neuron identity.

---

### Claim 2: Mechanistic evidence — traceable circuits stable across scenarios and emotions

**Original (verbatim excerpt from task.md):**
> We provide the mechanistic evidence that emotion generation in LLMs is supported by traceable circuits, which is stable across different scenarios and emotions.

**Extracted statement**: The circuits `C_e` identified in Claim 1 (a) **causally** carry emotion generation — ablating them weakens the target emotion, enhancing them strengthens it, with off-target emotions largely unaffected; and (b) are **stable** across (i) *scenarios* (Jaccard overlap of `C_e` selected from disjoint scenario partitions of SEV is high — the same neurons/heads are re-elected across the 8 domains × 20 scenarios) and (ii) *emotions* (per-emotion neuron sets show low overlap — indicating emotion-specific processing — while shared attention-head machinery shows moderate overlap — indicating shared contextual propagation, in line with the task's implicit framing).

**Hypothesis**:
- H2a (causality): For each emotion e, ablating `C_e` on prompts targeted at e reduces the target-emotion probability / expression score, and enhancing (scaling up) `C_e` increases it, both with a dose-response monotonic in intervention strength and both effects specific to e (off-target emotions e′ ≠ e are much less affected).
- H2b (scenario stability): The per-emotion component set `C_e^{S1}` extracted from one subset of SEV scenarios and `C_e^{S2}` extracted from a disjoint subset show Jaccard overlap significantly above a random-selection null.
- H2c (emotion structure): Across emotions, per-emotion **MLP-neuron** sets show *lower* pairwise overlap (per-emotion machinery) than per-emotion **attention-head** sets (shared machinery). Direction of the *difference* is the qualitative predicate; magnitudes are reported.

**Measurable predicate**:
- H2a: mean per-target Δ(emotion-expression) under ablation < 0 and under enhancement > 0, with a monotonic dose-response over ≥ 3 intervention strengths; specificity — average off-target Δ magnitude is smaller than target Δ magnitude at a matched strength.
- H2b: mean Jaccard(`C_e^{S1}`, `C_e^{S2}`) > matched-permutation null (95% CI does not cross the null), averaged over emotions.
- H2c: mean pairwise Jaccard over emotions is *smaller for neuron sets than for head sets*, reported as a directed inequality with bootstrap CIs.

**Expected direction**:
- H2a — ablation ↓, enhancement ↑, target > off-target (three directional predicates).
- H2b — above-null (up).
- H2c — neuron overlap < head overlap (directed inequality).

**Resources**: model: Llama-3.2-3B-Instruct (full size); dataset: SEV full 480 events × 6 emotion variants; split — the framework in Claim 1 fits `C_e` on a fit fold; ablation / enhancement is evaluated on a held-out fold; scenario stability uses disjoint scenario partitions from the 8-domain × 20-scenario grid; used_n = available_n; up to 3 seeds for stability CIs.

**Status**: pending verification

**Notes**: Task.md does not specify the exact overlap metric — Jaccard is chosen because it is the standard component-set overlap measure (dimension-free, symmetric, robust to set-size differences), matching a natural reading of "stable across scenarios and emotions". The overlap direction (H2c) is inferred from the sentence structure — task.md treats "circuits" (plural) as both "emotion-specific" (Claim 1) *and* "stable across emotions" (Claim 2), which is only internally consistent if per-emotion sets *share* some components (heads) while *differing* in others (neurons); we capture that as a directed inequality rather than a quantitative bound. The intervention specificity is what promotes "located" to "mechanistic evidence".

---

### Claim 3: Circuit-based control reliably induces target emotions, outperforming prompting and single-direction steering

**Original (verbatim excerpt from task.md):**
> a circuit-based control method can reliably induce target emotions across arbitrary inputs without relying on explicit instructions. It outperforms both prompting and direction-level steering on emotion-expression accuracy.

**Extracted statement**: Intervening on the circuit `C_e` from Claim 1 during generation on **arbitrary event inputs** (SEV event stems, without emotion-tagged prompt suffixes) induces the target emotion `e` in the generated continuation with an emotion-expression accuracy that is *higher* than (i) a **prompting** baseline (EmotionPrompt-style: append an emotion / stakes stimulus sentence to the prompt) and (ii) a **direction-level steering** baseline (single-direction RepE / CAA-style: extract one residual-stream direction for `e` and add it with a swept coefficient at the best single layer). "Arbitrary inputs" means event stems that are *not* the same items used to fit `C_e`. "Without explicit instructions" means the input does not name the target emotion.

**Hypothesis**: H3 — Circuit-based control achieves higher emotion-expression accuracy on held-out SEV event stems than the prompting baseline **and** than the direction-level steering baseline, under matched evaluation (same event stems, same accuracy judge, per-emotion aggregation).

**Measurable predicate**: On a held-out fold of SEV event stems (each stem run under six target-emotion conditions), emotion-expression accuracy (auto-judged by an external LLM judge and/or a trained classifier — see plan) satisfies
`accuracy(circuit) > accuracy(prompting)` **AND** `accuracy(circuit) > accuracy(single-direction steering)`,
with per-emotion means reported and paired-bootstrap CIs on the two pairwise differences that exclude 0 at the 95% level. Steering-baseline coefficient is tuned per-emotion by held-out sweep (same held-out mechanism the circuit method uses for its own strength knob).

**Expected direction**: up — circuit accuracy > baseline accuracy on both pairwise comparisons; per-emotion majority (≥ 5 of 6 emotions) rather than one aggregate.

**Resources**: model: Llama-3.2-3B-Instruct (full size); dataset: SEV held-out fold from the 480 events (used_n = available_n on the held-out fold); external LLM judge = `gpt-5.4` @ dmxapi (with proxy bypass) *or* a locally trained emotion classifier — plan resolves the exact judge in the experiment stage per the Judge-Reliability tips in `/experiment-tips` (agreement floor vs. human labels, judge-swap ablation). Verify stage will additionally re-check on Qwen2.5-7B-Instruct (model swap) and on the SEV held-out test set (data swap).

**Status**: pending verification

**Notes**: "Without relying on explicit instructions" is captured as the input contract for the eval fold — event stems without emotion tags. The comparison is a **matched-budget** comparison: each method (circuit / prompting / single-direction steering) gets the same held-out event stems, the same target-emotion conditions, and the same judge; direction-level steering must be tuned per-emotion at its best single layer + best α, otherwise the comparison is unfair. Choice of judge (external LLM vs. classifier) and its reliability check are deferred to Phase 4.5 / the experiment stage per `/experiment-tips` general policy on MCQ-vs-free-form parsing and judge audits.

## Cross-claim dependency

`C_e` is fit once in Claim 1; Claims 2 and 3 consume it. Claim 2's causal / stability tests validate that the framework is discovering a real, reusable circuit; Claim 3 shows that using it as a control knob wins over the two named baselines. No M0 phenomenon-validation gate is stamped because `BEHAVIOR_SOURCE = given` — task.md treats the behavior "LLMs can be steered to express emotions" as an established starting point, and the plan goes directly to the mechanism milestones (Location → Causal Intervention → Tuning & Editing).

## Verification milestone mapping (from EXPERIMENT_PLAN.md)

- **Claim 1** is verified by **M1** (Location — Stage A shortlist + Stage B causal ranker + Jaccard stability across event-subsample folds).
- **Claim 2** is verified by **M2** (Ablation + 3-α enhancement + three specificity controls: random-set null / targeted `C_{e'}` / off-target scoring + scenario stability + secondary cross-emotion structure).
- **Claim 3** is verified by **M3** (three-arm matched-budget comparison — Circuit / Prompting / Single-direction steering — with hidden-target 6-way judge and paired-bootstrap CIs) and additionally by **M4** (Verify-swap-lite on Qwen2.5-7B-Instruct + SEV held-out, Claim 3 primary comparison only).
- **M0.5** (data prep + judge audit) precedes all mechanism milestones and gates M3 / M4 via the judge-reliability check.

See `refine-logs/EXPERIMENT_PLAN.md` for the full plan and `refine-logs/EXPERIMENT_TRACKER.md` for the run-level breakdown.
