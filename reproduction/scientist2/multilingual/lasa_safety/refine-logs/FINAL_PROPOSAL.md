# Final Proposal — Language-Agnostic Safety Alignment at the Semantic Bottleneck of LLaMA-3.1-8B-Instruct

**Date**: 2026-07-14
**Behavior-source**: given (behavior taken verbatim from `task.md` §Claim; no ideation, no M0)
**Mechanism**: discovery
**Chosen candidate**: F1 — layer-level bottleneck via per-layer semantic-vs-language-identity ratio (see `idea-stage/IDEA_REPORT.md` §Mechanism-Hypothesis Space)

## Metadata (machine markers)

```yaml
mechanism_strategy:
  directions: [Location, Causal Intervention, Tuning & Editing]   # in execution order
  rejected:
    - Formation Tracing — genesis of the bottleneck is not part of the claim; training-time attribution would blow the 10 h GPU budget.
    - Unit Interpretation — SAE-level feature naming is a costlier route to the same layer-selection answer (F3 in IDEA_REPORT); F1 gets there via cheap forward passes.
    - Decision Auditing — the claim is about mechanism, not per-decision trustworthiness of a fielded system.
  note: >
    The bottleneck-layer choice is turned into a quantitative per-layer diagnostic (Location) and confirmed by
    cross-lingual activation patching with matched-layer specificity controls (Causal Intervention), then that layer is
    used as the anchor for a representation-space safety-alignment procedure (Tuning & Editing) evaluated on MultiJail
    across seen (EN/ZH/KO) and unseen (7 other MultiJail languages) alignment-time languages.
```

Notes on other markers:
- `resource_fidelity`: **not stamped** — this combination is `BEHAVIOR_SOURCE=given` × `MECHANISM=discovery`, which is cost-aware, not the reproduction combo. However, the `task.md` HARD emphatic-positive requirements pin (a) the base model (LLaMA-3.1-8B-Instruct) and (b) the primary safety benchmark (MultiJail) — those may **not** be swapped or downscaled for cost. Training-data sizes may be scaled down only if the 10 h GPU budget genuinely does not cover the full run, in which case the downscale is declared, not silent.
- `chosen_mechanism`: n/a (`MECHANISM=discovery`; the concrete family is chosen later by `/auto-experiment` Phase 1.5 via `/mechanism-skills`).

## Problem Anchor

Multilingual safety-aligned LLMs (LLaMA-3.1-8B-Instruct in particular) exhibit a systematic cross-lingual safety gap: harmful prompts translated into low-resource languages jailbreak the model far more easily than the same prompts in English (MultiJail, XSafety, Cross-Lingual Pitfalls). Simultaneously, prior mechanistic work (Wendler 2024; Dumas 2024; Wu 2024; Wang 2025) shows that mid-network layers of multilingual LLMs carry a substantially language-agnostic representation of meaning. The behavior we are asked to test says: **(B1)** such an intermediate "semantic bottleneck" layer exists in LLaMA-3.1-8B-Instruct and its geometry is dominated by shared meaning rather than by language identity; and **(B2)** anchoring the safety-alignment signal at that layer produces stronger cross-lingual generalization (lower MultiJail ASR on unseen-in-training languages) than surface-level alignment on the same training data, without sacrificing general capability. This proposal freezes B1 and B2 as written and refines the *testing method*.

## Method Thesis (one sentence)

Given LLaMA-3.1-8B-Instruct, quantitatively locate a semantic-bottleneck layer by a per-layer diagnostic on parallel multilingual prompts, causally verify the diagnostic with matched-control cross-lingual activation patching, and then use that layer as the anchor point of a representation-space safety-alignment procedure trained on PKU-SafeRLHF (EN/ZH/KO only) + UltraFeedback — evaluated head-to-head against a surface-space DPO baseline on identical data via MultiJail ASR (per-language, per-tier, and worst-language) and capability retention (MMLU / M-MMLU / MGSM / MT-Bench).

## Dominant Contribution

Not "safety alignment works better"; the dominant contribution is a **falsifiable claim about where the safety signal should be attached**. Specifically: turning the "semantic bottleneck" from a *qualitative narrative* in prior mid-layer / concept-space work into (a) a **quantitative per-layer diagnostic** with an interior optimum, (b) a **causal test** with matched-control layers, and (c) a **downstream training-time consequence** on cross-lingual safety generalization. If B1's interior optimum is absent, or B2's advantage over surface DPO on the same data / same base model disappears, the claim is refuted — cleanly.

## Method Sketch (per direction)

### Step 1 — Location (B1 mechanism-existence probe)

Given LLaMA-3.1-8B-Instruct (32 layers), for each layer `l` compute two centroids on parallel multilingual sentence pairs:

- `Sem(l)` = mean cosine similarity between the last-token residual-stream hidden state of the *same-meaning* prompt across different languages (parallel MultiJail prompts across 10 languages).
- `Lang(l)` = mean cosine similarity between the last-token hidden state of *different-meaning* prompts within the *same* language (random pairs within one language).

Define `R(l) = Sem(l) / Lang(l)` and `D(l) = Lang(l) − Sem(l)`. B1 predicts an **interior maximum of `R`** (equivalently, interior minimum of `D`) in the mid-network band `l ∈ [ceil(0.25·L), floor(0.75·L)]`, i.e., `l ∈ [8, 24]` for LLaMA-3.1-8B-Instruct. Sanity band: at both the input side (`l = 0, 1, 2`) and near the output (`l = L-2, L-1`), we expect Lang > Sem or the ratio dropping. The *identified L\** is `argmax_l R(l)`.

**Cost / time**: forward-only, no training; ~few hundred prompt sets × 10 languages × 32 layers of hidden-state extraction. Estimated < 1 GPU-hour on a single GPU from {1,2,3,5,6}.

**Result form**: `results/M1_bottleneck_diagnostic.json` — per-layer `Sem(l)`, `Lang(l)`, `R(l)`, `D(l)`, chosen `L*`, plus a per-language decomposition of `Sem(l)` so we can see whether specific low-resource languages break the average.

### Step 2 — Causal Intervention (B1 confirmation)

Cross-lingual activation patching along the lines of Dumas 2024: given a pair (English prompt `p_en`, Chinese/Korean/other-lang prompt `p_x` conveying the same meaning), forward-pass `p_x` up to layer `l`, replace the residual-stream state at the last token with the corresponding state from `p_en` at layer `l`, and complete the forward pass to measure whether the target-language generation preserves meaning (vs. drifts to English / drifts semantically). Report the patching effect at `L*` vs. at surface controls (layer 2, layer 30 for the 32-layer model), for each language.

B1 predicts: patching effect at `L*` significantly larger (meaning-preservation higher) than at surface controls, matched across languages. Matched-control specificity: patching an *unrelated* prompt state at `L*` should not preserve meaning.

**Cost / time**: still forward-only interventions; ~same order as Step 1. Estimated < 1 GPU-hour on one GPU. Combined M1 (Steps 1+2) budget: ~2 GPU-hours.

**Result form**: `results/M1_patching.json` — per-language patching-effect at each of `L*`, layer 2, layer 30, plus matched-control.

**B1 verdict rule**: B1 is judged `supported` iff (a) an interior maximum of `R` exists in [8,24], (b) the patching effect at `L*` is materially larger than at surface controls with matched-control ≈ 0.

### Step 3 — Tuning & Editing (B2 payoff, gated)

If Step 2 supports B1 and the remaining GPU-budget headroom (10 h total minus M1 usage) covers B2, run the alignment head-to-head:

- **Bottleneck-anchored variant (B2-Method)**: SFT+DPO on LLaMA-3.1-8B-Instruct with PKU-SafeRLHF (EN/ZH/KO translations) + UltraFeedback, augmented with a hidden-state-level regularizer that anchors the chosen-vs-rejected representation gap at `L*` (representation-space DPO). Concretely: alongside the token-level DPO loss, add a term that penalizes divergence between chosen-response hidden states at `L*` across languages (a language-invariance loss) — this operationalizes "aligning safety at the bottleneck layer".
- **Surface DPO baseline (B2-Baseline)**: same base, same data, same optimizer, same steps — DPO without the L*-anchored regularizer. This isolates the "anchoring at L*" effect.
- **(Optional if budget allows)**: A second baseline — Aya-style translated multilingual DPO with the same data — to strengthen the head-to-head. If budget is tight, defer to `/auto-verify` swap variants.

**Evaluation** — MultiJail 10-language ASR via GPT-4o judge (task.md-specified `gpt-5.4` at `https://www.dmxapi.cn/v1`, proxy bypass); split by (i) seen-in-training (EN, ZH, KO) vs (ii) unseen (7 other MultiJail languages); report worst-language ASR. Capability retention: MMLU / M-MMLU (accuracy), MGSM (math accuracy), MT-Bench (LM-judge quality). All within the 10 h GPU budget.

**B2 verdict rule**: B2 is judged `supported` iff (a) average ASR on the 7 unseen languages of B2-Method is lower than B2-Baseline by a pre-registered relative margin (planned target ≥ 20 pp relative on the unseen slice; a smaller-but-consistent effect is `partially supported`), (b) worst-language ASR of B2-Method ≤ B2-Baseline, (c) MMLU / M-MMLU / MGSM / MT-Bench of B2-Method ≥ B2-Baseline − 2 absolute pp on every metric.

**Cost / time**: this is the expensive milestone. DPO on LLaMA-3.1-8B-Instruct with two variants + baseline evaluations is on the order of 6–8 GPU-hours on a single GPU from {1,2,3,5,6}. The **budget gate at M3-start** decides scale and whether to include the Aya baseline (see EXPERIMENT_PLAN.md M3).

## Expected Observations (falsifiers listed)

1. **B1 supports**: `R(l)` has an interior maximum in [8,24]; cross-lingual patching at `L*` restores meaning more than at surface layers; matched-control ≈ 0.
2. **B1 refutes** (any of): `R(l)` monotonic (no interior optimum); patching at `L*` no better than surface control; matched-control ≈ patching signal (specificity fail).
3. **B2 supports**: on identical training data, bottleneck-anchored DPO cuts unseen-language MultiJail ASR by ≥ ~20 pp relative to surface DPO, without capability collapse.
4. **B2 refutes** (any of): unseen-language ASR of the two variants is within noise; the anchored variant collapses capability (any of the four capability metrics drops > 2 absolute pp).
5. **B2 partially supports**: ASR effect present but < 20 pp relative, or worst-language ASR tie — reported as `partial` and flagged for `/auto-iteration-loop`.

## Rationale — Why F1 (not F2 / F3)

- F1's cheap-screen diagnostic is training-free and reuses only forward passes, so it fits in ~1–2 GPU-hours — leaving > 8 hours for B2's DPO training.
- The diagnostic matches task.md's B1 wording literally: **layer** with **hidden-state geometry** dominated by **shared meaning**. F2 rephrases B1 as a claim about a refusal *direction* (safety-specific), which partially pre-empts B2. F3 requires a multi-layer SAE, which the 10 h budget cannot comfortably support alongside B2 DPO.
- The causal-test primitive (activation patching) is textbook and well-controlled (Dumas 2024), so we do not invent new methodology for the B1 half — the novelty (relative to the LANDSCAPE) is the *quantitative* diagnostic + *matched-control* comparison + *hand-off* to the alignment stage.
- The `Tuning & Editing` extension (Step 3) is the minimum machinery that turns "there is a bottleneck" into "and it changes safety alignment when you use it" — which is what task.md's B2 asks.

## Downstream Handoff

`/mechanism-skills` (Workflow 1.25) will most likely route Location to the `representation-and-parameter-analysis` family (probing / geometry) and Causal Intervention to `causal-attribution` (activation patching). The Tuning & Editing step is a training-time procedure (representation-space DPO) that does not sit inside `/mechanism-skills`' interpretability families — `/auto-experiment` treats it as an alignment-training milestone rather than a mechanism-routing milestone. The two milestones share the same `L*` output artifact.
