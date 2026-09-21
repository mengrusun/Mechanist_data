# Final Proposal — Group-Aggregated Activation Probes for Multi-Agent Collusion Detection

**Behavior-source**: given
**Mechanism**: discovery
**Direction**: (empty in args — sourced entirely from task.md)
**Date**: 2026-07-14
**Covers**: C1 (existence), C2 (aggregation), C3 (transfer) as captured in `idea-stage/IDEA_REPORT.md`.

---

## Metadata (top markers read by the experiment stage)

```yaml
behavior_source: given
mechanism: discovery
resource_fidelity: not-strict   # cost-aware routing allowed (not the reproduction combo)

mechanism_strategy:
  directions: [Location, Decision Auditing]   # in execution order
  rejected:
    - Causal Intervention — a *causal* intervention (patching / ablation / steering) on Qwen3-32B-AWQ residual streams would exceed the 10 GPU-hour budget once the AWQ-quantized activation extraction pass and the multi-scenario probe eval are already inside the plan; the three sub-claims are formulated as *detection/monitoring* predicates (AUROC / dominance / transfer), which Location + Decision Auditing already answer without a causal ablation.
    - Tuning & Editing — the goal is detection, not capability improvement; tuning the located component to make the model *more collusive* or *less collusive* is out of scope.
    - Formation Tracing — training-time genesis of the collusion feature is out of scope; the claim is about a *deployed*, already-trained model.
    - Unit Interpretation — decoding the exact human-readable concept in the probe direction (SAE / auto-interp) is an interesting follow-up but not required by C1/C2/C3; the sub-claims only require that the direction exists, aggregates, and transfers, not that a name is attached to it.
  note: >
    Location extracts per-agent residual-stream signals and their group-level aggregations (C1, C2);
    Decision Auditing uses the aggregated probes as monitors on group-level committee decisions
    and stress-tests transfer to structurally different coordination scenarios (C3).
```

Downstream: `/mechanism-skills` (Workflow 1.25) will bind a concrete family from the Location direction — expected to route to **linear-probe / representation-reading** (residual-stream linear classifiers) — and the experiment stage may re-bind method-sensitive fields (`n_pairs`, `sites`, `metric`, `gpu_hours`) without rewriting this plan.

---

## 1. Problem Framing

Multi-agent LLM systems are increasingly deployed for high-stakes committee-style tasks: multi-domain deliberation, code review, market decisions. Text-only monitoring — even paraphrase / CoT monitors — has been repeatedly shown to be insufficient against covert coordination (`Hidden in Plain Text`, arXiv:2410.03768; `Covert CoT Sandbagging`, arXiv:2508.00943; `Decision-Theoretic Steganography`, arXiv:2602.23163). Simultaneously, single-agent residual-stream linear probes for deception have matured (`Apollo`, arXiv:2502.03407; `RepE`, arXiv:2310.01405; `Geometry of Truth`, arXiv:2310.06824). The unresolved question is whether the same white-box tool generalizes from *single-agent lying* to *multi-agent collusion*. The `LANDSCAPE.md` structural gaps that this proposal targets:

- **G1** — no prior work aggregates per-agent activation-probe scores at the group level to detect collusion.
- **G2** — zero-shot transfer of internal-state probes across *structurally different* multi-agent scenarios is unmeasured.
- **G3** — the complementarity of aggregation techniques for group-level probes is untested.
- **G4** — a strong text-only monitor baseline for multi-agent covert coordination is underspecified.
- **G5** — non-lying-deception and confound controls specific to collusion are missing.

## 2. Method Sketch (unified across C1, C2, C3)

The plan has three components, executed as three main milestones (M1 → M3), all sharing the same primary model **Qwen3-32B-AWQ** (HARD CONSTRAINT).

**(a) Per-agent residual-stream activation extraction.** For every multi-agent scenario in NARCBench-Core, run the committee with **K = 3 agents** per scenario (the S1 committee-deliberation frame; adjustable per scenario), running each agent as one vLLM inference pass with `output_hidden_states` collected (or via a forward-hook wrapper if AWQ + vLLM cannot expose hidden states directly — fallback: run the same prompts under HF transformers with AWQ and collect activations). At each agent's *final assistant turn*, extract the last-token residual-stream hidden state at a small set of candidate layers (mid-late layers, e.g. layers {20, 28, 36, 44} of 48 for Qwen3-32B; final list resolved in M1 layer-selection sub-phase — layer choice is `method_sensitive`).

**(b) Group-level aggregation family.** For each scenario we get K per-agent activation vectors → K per-agent probe scores. The family of group-aggregations tested in M2:
1. **mean-pool** — average per-agent probe logits.
2. **max-pool** — take max per-agent probe logit.
3. **attention-pool** — weighted mean where weights are a small learned attention over the K per-agent activations.
4. **concat-then-MLP** — concatenate K per-agent activation vectors, feed to a small 2-layer MLP.
5. **best-of-N** — take the single agent's probe score whose confidence is highest (a max variant that drops the sign).

**(c) Linear probe head.** For each per-agent activation vector at each candidate layer, train a **logistic-regression probe** on labeled (collusive, honest) NARCBench-Core scenarios. The probe head is intentionally lightweight to guard against Farquhar-et-al.-style "probes capture the prominent feature not the target" failure (arXiv:2312.10029) — sanity checks: (i) label-permutation control, (ii) matched-length control (honest committee same length as collusive committee), (iii) topic-swap control (train on honest–vs–collusive within one domain, test held-out domains).

**(d) Evaluation as a Decision Auditor (mechanism direction 6).** The aggregated probe on each scenario is treated as a *decision auditor* over the committee's final vote — asking "should this collusive committee output be trusted?" This is the Decision Auditing framing from `/mechanism-explore`: the aggregated per-agent activations trace the evidence the committee relied on, and we check whether that evidence is *valid task-relevant reasoning* versus *coordinated covert signaling*.

## 3. Tie-in to the Three Sub-Claims

| Sub-claim | Component | Milestone |
|---|---|---|
| **C1 (existence)** — internal signal beats text-only judge | (a) extraction + (c) single-agent linear probe + strong text-only judge baseline (gpt-5.4 committee vote on transcripts) | M1 |
| **C2 (aggregation)** — group aggregation beats best single-agent probe; no single aggregation dominates across 17 domains | (b) five aggregation techniques applied on top of M1's per-agent probe scores + domain-argmax diversity test | M2 |
| **C3 (transfer)** — zero-shot to NARCBench-Transfer + NARCBench-Stego | (a) + best aggregation from M2 applied to unseen coordination structures without any retraining | M3 |

## 4. Method-Sensitive Fields (bind at experiment stage, not here)

- `n_pairs` — final probe train / eval sample size per class (subject to `/data-rule` floors); planned as **≥ 100 collusive + ≥ 100 honest** per domain (7 domains × 100 = 700 per class is aspirational; if NARCBench-Core is not that large, generate additional scenarios or reduce with power justification).
- `sites` — which candidate layers to probe (planned shortlist above; final choice = per-layer AUROC on a small held-out set).
- `metric` — primary AUROC; secondary at 1% FPR (parallel to Apollo's protocol).
- `gpu_hours` — the four fields above may drift once `/mechanism-skills` binds the concrete family; the estimate below (~10h total) is provisional.

## 5. Risks and Confound Controls

- **Probe captures topic/style, not collusion (G5).** Mitigated by (i) matched-length control, (ii) topic-swap control, (iii) label-permutation control.
- **Text-only judge baseline is a strawman (G4).** The judge is `gpt-5.4` (a frontier LLM) prompted with the full committee transcript and asked for a collusion probability with rubric — not a bag-of-words classifier.
- **Non-lying deception mode (G5).** Collusion may be *coordination without falsehood*; the probe is trained on behavior-labeled (collusive vs honest) not statement-labeled (true vs false) pairs, following Berger's arXiv:2603.10003 warning.
- **AWQ quantization + hidden-state access.** vLLM's AWQ path may not expose hidden states directly. Fallback: HF transformers + AWQ for activation extraction (~2× slower but works). Budget assumes fallback if needed.
- **No single-aggregation dominance sub-claim may be violated (C2b).** If one aggregation dominates all 17 domains, the family-of-aggregations motivation weakens; report this honestly.

## 6. Budget Sketch (HARD CONSTRAINT: 10 GPU-hours total, GPUs ∈ {1,2,3,5,6})

Preliminary allocation (refined in `EXPERIMENT_PLAN.md`):
- **M1** (C1 existence + text-only judge baseline + layer selection): ~4 h
- **M2** (5 aggregations across 17 domains): ~3 h
- **M3** (zero-shot transfer to NARCBench-Transfer + NARCBench-Stego): ~2 h
- **Contingency**: ~1 h

## 7. Deliverables

- `refine-logs/EXPERIMENT_PLAN.md` — milestone-level roadmap with grid, cmd templates, predicates.
- `refine-logs/EXPERIMENT_TRACKER.md` — plan-level tracker (populated by the experiment stage).
- Downstream: `/mechanism-skills` binds the concrete Location submethod; `/auto-experiment` runs; `/auto-verify` stress-tests C1–C3 with swap variants.
