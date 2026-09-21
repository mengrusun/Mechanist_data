# FINAL PROPOSAL — Emotional Framing in Prompts as a Weak, Input-Dependent Signal (with residual-stream localization)

**Behavior-source**: given (from `task.md`)
**Mechanism**: discovery
**Resource-fidelity**: cost-aware (`resource_fidelity: strict` NOT stamped — this is `given` + `discovery`, not the reproduction combo)
**Primary model / dataset (hard-pinned by task.md)**: Qwen3-14B / GSM8K
**Date**: 2026-07-13

---

## mechanism_strategy (mirrored from EXPERIMENT_PLAN.md top metadata)

```yaml
mechanism_strategy:
  directions: [Location, Causal Intervention]   # in execution order
  rejected:
    - Tuning & Editing — C4 (EmotionRL) is input-space adaptive selection, not weight-space editing; the mechanism claim is about the residual-stream frame direction that mediates C1–C3, not tuning Qwen3-14B.
    - Formation Tracing — training-time genesis is out of scope; too expensive for the 10-GPU-h budget on a 14B model.
    - Decision Auditing — no spurious-correlation reliability audit is requested; the phenomenon itself is the target.
  note: The mechanism claim is that some low-rank residual-stream direction (or a small attention-head set) on Qwen3-14B early-to-mid layers carries the emotional frame and causally modulates per-item task accuracy; Location + Causal Intervention are the shortest strategy that lands this claim.
  optional_add_on: Unit Interpretation (SAE / vocab projection) is a cheap add-on only if Location produces a clean shortlist.
```

---

## Problem Anchor

Prompt-engineering folklore claims that emotional prefixes ("This is very important to my career", "I feel anxious about getting this right") reliably improve LLM accuracy. The seminal *EmotionPrompt* paper (arXiv:2307.11760) reports large gains (8%–115%) but averages over the *maximum* of 11 emotional stimuli per task rather than reporting the per-stimulus distribution. Independent evidence (Sclar et al. 2310.11324; Kim et al. 2408.08631) suggests stylistic prefixes are often harmful and highly input-dependent, and the affect-as-information theory (Schwarz & Clore, 1983) predicts the effect should be much larger on socially grounded tasks than on math. The **problem anchor** is: *characterize the actual behavior of static basic-emotion prefixes as an input-dependent signal on Qwen3-14B, and locate the internal component in Qwen3-14B that carries the emotional frame and causally mediates the observed per-item accuracy shifts.*

The Problem Anchor never moves through refinement. The four claims below are taken **verbatim from task.md** (faithful capture, Phase 2). All refinement below only sharpens **how** we test them, never **what** we claim.

## Final Method Thesis

We hold Qwen3-14B fixed and treat the 13-way action space {6 emotions × 2 intensities} ∪ {neutral} × {human-written, LLM-generated} = **26 prompt conditions** as a controlled experimental grid. On GSM8K (primary), SocialIQA (social companion), and MedQA (factual companion), we measure per-condition macro accuracy and per-item Δaccuracy vs neutral to test the four behavioral claims C1–C4. We then use Qwen3-14B's cached activations from the GSM8K runs to (i) *locate* the residual-stream direction(s) and attention heads that linearly encode the emotion identity of the prefix, and (ii) *causally verify* that those components mediate the C1–C3 per-item Δaccuracy pattern via activation patching and dose-response steering with matched-length neutral controls. EmotionRL (C4) is trained as an adaptive per-query policy over the same 13-way action space using the Directional Stimulus Prompting architecture (arXiv:2302.11520).

**Dominant contribution**: the first paper to place emotional-prefix effects (i) in a FormatSpread-style noise-floor comparison across a task-family grid with the *distribution* per emotion×intensity×wording-source reported, (ii) with a causally-localized mechanism on the same model & task, and (iii) with a fair head-to-head against a learned adaptive policy over the same action space.

**Optional supporting contribution**: SAE / vocab-projection decoding of the located residual direction (Unit Interpretation) — added only if Location produces a clean shortlist.

**Explicitly rejected complexity**:

- Weight-space editing / task-vector tuning (out of scope; the mechanism claim is representation-level).
- Training-time formation tracing / influence functions (compute-prohibitive at 14B / 10 GPU-h).
- Free-form stimulus generation for EmotionRL (fixed 13-way discrete space per the task.md prompt-template spec).
- Cross-model comparison at this stage (Llama-3.3-70B-Instruct and DeepSeek-V3.2 are the verify-stage variants at `/auto-verify`, not here).
- Multi-turn / conversational emotion tracking (task.md scopes single-turn static prefixes only).

## Claims to Verify (verbatim from `idea-stage/IDEA_REPORT.md`)

**C1 (small input-dependent shift)** — for each of 12 emotion×intensity prefixes × {human, LLM-generated} + neutral, on Qwen3-14B / GSM8K, mean|Δaccuracy vs neutral| ≤ format-perturbation noise floor **and** sign-consistency of per-item Δ within [0.4, 0.6] per prefix.

**C2 (task-family ordering)** — inter-emotion spread of macro accuracy on SocialIQA ≥ 2 × spread on GSM8K, with ordering `spread_social > spread_factual > spread_math` observed in ≥ 2 of 3 comparisons.

**C3a (no consistent winner)** — no emotion is argmax across all task families; the best-emotion identity flips ≥ once across {math, social, factual}.
**C3b (no monotone intensity)** — for ≥ half of the 6 emotions on GSM8K, the paired Δ(intensity-2 − intensity-1) CI straddles 0 or reverses sign.

**C4 (adaptive > fixed)** — on held-out GSM8K, `acc(π_θ) > acc(neutral)` and `acc(π_θ) > acc(e*)` for the fixed-emotion argmax `e*`, both with lower 95% CI > 0 across ≥ 3 seeds. Policy: Llama-3.2-1B classifier over the 13-way action space; SFT on argmax-per-item labels from a labeled train subset, then a light RL fine-tune against Qwen3-14B GSM8K exact-match reward (Directional Stimulus Prompting, arXiv:2302.11520).

**Mechanism claim CM** *(shape only — the exact layer/head identities are discovered by M5/M6, not fixed here)*: some low-rank residual-stream direction (or small attention-head set) on Qwen3-14B early-to-mid layers carries the emotional frame identity and causally modulates per-item GSM8K accuracy, in the same direction as the C1/C3 per-item Δ pattern, with matched-length neutral controls showing no effect.

## Testing approach (unified across C1–C4 + CM)

1. **Corpus construction (M1)** — 6 emotions × 2 intensities × 2 wording sources = 24 emotional prefixes + 1 neutral + 1 length-matched non-emotional filler = **26 conditions**. Human-written variants adapted from EmotionPrompt (arXiv:2307.11760) and the affect-lexicon convention. LLM-generated variants produced via the dmxapi `gpt-5.4` service (task.md) with proxy bypass; each variant length-matched to its human counterpart within ±10 tokens.

2. **Behavioral evaluation (M2, M3)** — Qwen3-14B in vLLM bf16 with fixed decoding (temperature 0.0 for GSM8K CoT, temperature 0.0 with MCQ letter parsing for SocialIQA / MedQA). Sample size: 500 items per (condition, task) chosen to give per-condition macro-accuracy 95% CI width ≤ 4.5 pp under a p=0.5 binomial (with 500 items this is ±4.4 pp). Bootstrap CIs (1000 resamples) on paired per-item Δ.

3. **Noise-floor calibration (embedded in M2)** — parallel FormatSpread-style neutral-preserving format perturbations on the same 500 items: whitespace, list-marker, casing, and instruction-wording variants. The 90th-percentile format-perturbation |Δaccuracy| is C1's noise-floor threshold.

4. **C3 analysis (M4)** — no new runs; pure post-hoc analysis of M2 + M3 accuracy grids.

5. **Mechanism-Location (M5)** — during M2 & M3, cache Qwen3-14B residual-stream activations at the last prefix token for a 200-item paired subset per condition (every 4 layers, 10 layers total on a 40-layer model). Train logistic probes for emotion identity (6-way) and valence (positive/negative/neutral) on the cached activations. Rank layers by probe accuracy; take the top-2 layers as the "frame layers". Vocabulary-project the top singular directions of the 26×d activation matrix to name candidate frame directions.

6. **Mechanism-Causal (M6)** — on the identified frame layers: (i) activation-patch the frame direction from an emotional-prefix run into a neutral-prefix run on the same GSM8K item; measure Δaccuracy vs the paired neutral baseline; (ii) run a 4-point steering dose-response (α ∈ {−1, 0, +0.5, +1} × the frame direction on 200 items) and check the expected sign + monotonicity of the effect; (iii) matched-length non-emotional-filler control shows null effect. Specificity via off-target: an unrelated skill (e.g., factual-QA subset) is not affected.

7. **EmotionRL (M7)** — labeled train set: 2000 GSM8K train items × 13 candidate prefixes (compute reward once per item×prefix). Policy: `Llama-3.2-1B`-class classifier or a `bert-base`-class encoder over the 13-way action space (Phase 4.5 fixes: **Llama-3.2-1B** as the policy backbone, as it is available in `$MODEL_DIR` per task.md-referenced pattern and reuses vLLM infra). SFT: argmax label + label-smoothing; RL fine-tune: 1 epoch of REINFORCE with a learned baseline against the cached reward table (does not require additional Qwen3-14B forward passes). 3 seeds. Held-out eval on GSM8K test.

## Compute budget (10 GPU-h total; GPUs {1,2,3,5,6})

Reference numbers use Qwen3-14B bf16 in vLLM on an H100/A100-class GPU with batched generation and prefix-caching; per-item cost ≈ 0.3 s at 256 new tokens for GSM8K CoT and ≈ 0.15 s for MCQ tasks.

| Milestone | Description | GPU-h | Runs on |
|---|---|---|---|
| M1 | Corpus construction (LLM-gen via dmxapi; no GPU) | 0.0 | — |
| M2 | C1 primary — GSM8K × 26 conditions × 500 items × 0.3 s / 5 GPUs = **wall 0.22 h**; **GPU-h ≈ 1.08** | 1.10 | {1,2,3,5,6} parallel |
| M3 | C2 companion — (SocialIQA + MedQA) × 26 × 500 × 0.15 s / 5 GPUs; **GPU-h ≈ 1.08** | 1.10 | {1,2,3,5,6} parallel |
| M4 | C3 analysis (CPU only) | 0.0 | — |
| M5 | Location — activation dump embedded in M2; probe training CPU-cheap; ≈ 0.5 GPU-h for vocab projection sweep | 0.5 | 1 GPU |
| M6 | Causal — patching + steering on 200 paired items × 5 sites × 4 coefficients = 4000 forwards × 0.3 s ≈ 0.34 h | 0.6 | 1–2 GPUs |
| M7 | EmotionRL — reward-table build: 2000 items × 13 prefixes × 0.3 s / 5 GPUs ≈ 0.43 h; **GPU-h ≈ 2.17**; policy SFT+RL on CPU/1-GPU ≈ 0.2 h × 3 seeds = 0.6 GPU-h; eval on 500 held-out × 3 seeds ≈ 0.15 GPU-h | 2.90 | {1,2,3,5,6} parallel |
| **Sum** | | **7.20** | Headroom **~2.8 GPU-h** for retries / SAE add-on |

**Downscaling policy**: none until actual usage reaches 10 GPU-h (task.md hard constraint). If M7's reward-table build overruns, reduce train subset from 2000 to 1000 items *only after* the 10-h wall is reached.

## Reviewer concerns still live for the plan (Planning Gate)

1. **Length / lexical / frequency confounds** — emotional prefixes are longer and use rarer words than neutral. *Handled* via a length-matched non-emotional filler (M2), token-frequency stratification in M5's probe controls, and a paraphrase-based format-noise floor in M2.
2. **CoT decoding drift on Qwen3-14B** — fixed decoding at T=0.0 for GSM8K, seed = 42 for shuffling only; each run reports per-item logs so per-item Δ is deterministic.
3. **Prefix-caching bias** — vLLM's prefix cache is disabled across conditions to prevent inter-condition contamination.
4. **Answer parser** — GSM8K uses the standard *"The answer is X"* regex (`##\s*(-?\d+)`); items where neither the neutral nor the emotional run parses successfully are dropped pairwise, not per-condition.
5. **EmotionRL leakage** — GSM8K test items are strictly held out of M7's train subset and reward-table cache.
6. **Mechanism specificity** — M6 mandates the matched-filler control, an off-target task, and a sign-consistency check on the dose-response; a positive M6 without all three does not verify CM.

## Final Verdict: READY

The problem anchor is stable, the four behavioral claims are faithfully captured, the mechanism claim is stated at the right altitude (kind of component, not specific identity), the testing method has been sharpened around each claim, and the compute plan fits the 10 GPU-h envelope with a ~30% headroom. The plan is ready to hand off to `/mechanism-skills` (Workflow 1.25) and `/auto-experiment` (Workflow 1.5).
