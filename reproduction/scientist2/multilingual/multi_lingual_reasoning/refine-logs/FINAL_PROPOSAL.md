# Final Proposal — Unified Testing Approach for the Language-Agnostic vs Language-Specific Subspace Hypothesis

```
behavior_source: given
mechanism: discovery
mechanism_strategy:
  directions: [Location, Causal Intervention, Tuning & Editing]
  rejected:
    - Formation Tracing — no training-time / origin claim in task.md
    - Unit Interpretation — task.md operates at subspace level, not per-neuron/SAE-feature naming
    - Decision Auditing — no trustworthiness / spurious-vs-valid framing in task.md
  note: Locate a language-specific subspace via probing (Claim 1), causally test its role via signed steering + null-space projection (Claims 2 & 3), and use the projection as an applied editing method benchmarked against multilingual SFT (Claim 4).
project_hard_constraints:
  gpu_budget_hours: 10
  gpu_allowlist: [1, 2, 3, 5, 6]
  dir_allowlist: [/data/zhenqian/Reproduction1/mechanica/multilingual/multi_lingual_reasoning, /data/zhenqian/data, /data/zhenqian/models]
  main_model: Qwen-3-4B-Thinking
  main_dataset: MGSM
  language_identifier: GlotLID
  target_languages: [En, Es, Fr, De, Zh, Jp, Ru, Th, Te, Bn, Sw]
verify_stage_awareness:
  candidate_models: [Qwen-2.5-Instruct-3B, Qwen-2.5-Instruct-7B, Qwen-3-1.7B-Thinking, Qwen-3-8B-Thinking, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-LLaMA-8B, DeepSeek-R1-Distill-Qwen-14B, GLM-Z1-9B, QwQ-32B]
  candidate_datasets: [XWinograd, M-MMLU]
```

## Problem Anchor (frozen)

LLMs reason worse in low-resource languages than in high-resource ones, even though the underlying logical content of a problem is language-independent. Task.md advances four hypotheses about the *mechanism* of this gap in a multilingual reasoning model (Qwen-3-4B-Thinking on MGSM across 11 languages spanning high / mid / low-resource):

- **Claim 1 (Location)** — the hidden state decomposes into a *language-specific subspace* and an approximately orthogonal *language-agnostic subspace*, and the language-specific subspace can be identified from a *small* multilingual probe set.
- **Claim 2 (Causal Intervention — null-space projection)** — suppressing the language-specific subspace at inference time consistently improves multilingual reasoning accuracy across languages and reasoning tasks, while GlotLID-measured output-language fidelity remains acceptable *when upper layers are left intact*.
- **Claim 3 (Causal Intervention — signed dose-response)** — the strength of language-specific activation is negatively correlated with reasoning accuracy: amplifying it degrades reasoning; removing it improves reasoning.
- **Claim 4 (Tuning & Editing — competitive cost efficiency)** — this training-free intervention matches or exceeds multilingual post-training (supervised fine-tuning, reinforcement learning) at a small fraction of the compute.

The four claims are *given*; this proposal never refines them, only refines *how to test them*.

## Method Thesis (one sentence)

A single unified testing pipeline — (i) fit a language-specific subspace `V_lang` from a small multilingual probe set via language-mean-difference SVD, then (ii) intervene at inference on Qwen-3-4B-Thinking's residual stream at non-upper layers with `h ← h + α · (Π_lang · h)` sweeping α across a signed range, and (iii) compare the α = −1 (null-space projection) setting against a matched-compute multilingual SFT baseline — jointly verifies the four claims by decomposing them into the ladder-of-evidence chain **Location → Causal Intervention → Tuning & Editing** and reporting matched-control specificity + off-target-language checks at each rung.

## Dominant Contribution (as-tested)

The core contribution the plan tests is that a **single, small-probe, low-rank, inference-time linear operator** (null-space projection onto the language-agnostic subspace at Qwen-3-4B-Thinking's *non-upper* layers) simultaneously (a) raises MGSM accuracy across all 11 target languages, (b) preserves GlotLID output-language fidelity, (c) exhibits a signed monotone dose-response, and (d) matches multilingual SFT at ≤ 10% of the SFT compute. Any single one of these four legs is a testable claim in its own right; the plan is structured so that failure of one leg still delivers verdicts for the others (each milestone stands alone).

## Explicitly Rejected Complexity (as-scoped in the plan)

The plan intentionally does **not**:

- introduce a mechanism family beyond linear subspace projection (no SAE features named, no neuron-level LAPE editing, no attention-head circuit surgery — these are candidate submethods that `/mechanism-skills` may re-bind at routing time via the `method_sensitive` field, but the *strategy* stays at the subspace level).
- introduce trainable parameters into the intervention path (no soft-prompt / bridge / adapter — the "training-free" wording of Claim 4 is preserved).
- expand beyond Qwen-3-4B-Thinking + MGSM at the main-experiment stage (task.md HARD constraints). Cross-model / cross-task generalization is a **verify-stage** contract via the NOTICE candidate list (Qwen-2.5, R1-Distill family, GLM-Z1, QwQ; XWinograd, M-MMLU) and belongs to `/auto-verify`, not the main experiment plan.
- open with an M0 phenomenon-validation gate (`BEHAVIOR_SOURCE = given` — the behavior is taken as given by prior work; the LENS / LAPE / MEXA / Wendler-et-al. landscape already establishes the subspace/pivot picture at broad scale).

## Testing Approach — Unified Across the Four Claims

### Step T1 — Locate `V_lang` (serves Claim 1)

- Probe set: parallel sentences across the 11 target languages, sourced from FLORES-200 dev + a light MGSM-training-shot pool held out from evaluation. Sweep probe-set size n_probe ∈ {50, 100, 250, 500, 1000} sentences per language.
- Extraction: for each candidate intervention layer `ℓ`, compute mean residual-stream activation μ_L,ℓ per language over the probe set. Stack the mean-difference matrix M_ℓ = [μ_L,ℓ − μ_avg,ℓ]_L; take its top-r SVD components as `V_lang,ℓ` with rank r ∈ {1, 2, 4, 8, 16, 32}.
- Verification (Claim 1 predicate):
  - Held-out language classifier trained on the projection of held-out MGSM prompts onto `V_lang,ℓ` reaches ≥ 0.90 accuracy across 11 languages;
  - Held-out language classifier trained on the orthogonal-complement projection collapses toward chance (~1/11);
  - Principal-angle cosine between `V_lang,ℓ` and a content-probe subspace (e.g., subject-identity probe fitted on English-only MGSM) has median ≤ 0.2 (approximate orthogonality).
- Baselines / matched controls: (a) random-r-dim subspace at layer ℓ (must NOT produce language classifier ≥ 0.90 on held-out); (b) LSAR-style unsupervised SVD over pooled monolingual corpora (sanity-check the SVD-mean-difference variant reproduces the LSAR-style structure on the reasoning model).

### Step T2 — Causal Intervention: null-space projection at non-upper layers (serves Claim 2)

- Intervention op: `h ← h − Π_lang · h` (equivalently α = −1) at layers ℓ ∈ intervention_range. Sweep intervention_range = {early = [0, L/3), mid = [L/3, 2L/3), all-non-upper = [0, L − k_top)} for Qwen-3-4B-Thinking (L = 36 layers; k_top ∈ {0, 4, 8, 12}). k_top > 0 tests the "leave upper layers intact" clause of Claim 2.
- Evaluation: MGSM in all 11 languages, 3-shot (task standard). Report per language: exact-match accuracy A(L) and GlotLID output-language fidelity F(L). Statistical test: paired bootstrap over problems, 95% CI on mean accuracy gain and mean fidelity change.
- Predicate (Claim 2):
  - Aggregate mean accuracy gain ≥ +3 pp over baseline with 95% CI > 0;
  - Per-language: accuracy improves or does not regress beyond an ε_slack = 1 pp margin in ≥ 8 of 11 languages;
  - Language fidelity: mean fidelity within ≤ 5 pp of baseline for k_top ≥ 4 (the "acceptable when upper layers intact" clause);
  - The cross-condition curve F vs k_top should be monotone-improving as k_top grows.
- Specificity controls: (a) matched-rank random-subspace projection (rank r same as `V_lang`); (b) leave-one-language-out fit of `V_lang` — the intervention should still improve accuracy on the held-out language (probe-set generalization to unseen languages); (c) English-only MGSM: intervention must NOT degrade English accuracy beyond a small ε_off = 1 pp (off-target check — the language-agnostic reasoning must stay intact).

### Step T3 — Causal Intervention: signed α-sweep (serves Claim 3)

- Family: `h ← h + α · Π_lang · h`, α ∈ {−1.5, −1.0, −0.5, 0, +0.5, +1.0, +1.5}. Fix the best (intervention_range, r) from T2. Evaluate on full MGSM 11-language.
- Predicate (Claim 3):
  - Sign: Spearman correlation of α with mean-across-language A(α) is significantly negative (p < 0.05);
  - Monotone check-points: A(−1) > A(0) > A(+1) with each strict inequality passing paired bootstrap 95% CI;
  - Per-language robustness: sign holds for ≥ 8 of 11 languages individually.
- Specificity control: run the same α-sweep with the *matched random* subspace from T2 controls; correlation should be null.

### Step T4 — Tuning & Editing: competitive comparison vs multilingual SFT (serves Claim 4)

- Baseline B1 — multilingual SFT: fine-tune Qwen-3-4B-Thinking on MGSM8KInstruct-style data (translated GSM8K in the 11 target languages). Use LoRA to stay within the 10-hour GPU budget; report the LoRA-SFT compute (GPU-hours). If a full-parameter SFT is affordable, report that instead; otherwise LoRA-SFT is the actually-run competitor and this is noted.
- Baseline B2 (optional, budget-permitting) — RL: skipped by default (does not fit in the residual budget after T1–T4 unless full-SFT is dropped); if run, use a light RL objective on multilingual reasoning traces via GRPO with a rule-based reward.
- Predicate (Claim 4):
  - Accuracy match: mean_L A_edit(L) ≥ mean_L A_SFT(L) − ε_match (ε_match = 0 as the strict target; report a robustness curve with ε_match ∈ [0, 2] pp);
  - Compute ratio: (edit_compute + probe-fit compute + inference overhead) / (SFT_training_compute + SFT_inference_compute) ≤ κ, with κ ≤ 0.10;
  - Both metrics reported side-by-side per language + aggregate.
- If B2 is skipped, the "matches or exceeds RL" side of Claim 4 is recorded as *unattempted, out of budget*, not silently claimed.

### Common protocol details

- **Language identifier**: GlotLID V3 (fastText, 2102 labels). Run on the generated *continuation* (the response after the CoT trigger); a generation counts as language-faithful iff GlotLID predicts the source language of the MGSM problem.
- **Data**: MGSM — 250 problems × 11 languages = 2750 evaluations per condition. Use 3-shot exemplars from the standard MGSM development set (in the source language; standard CoT prompting).
- **Backend**: vLLM (batched-inference engine) on the allowed GPU pool `{1, 2, 3, 5, 6}`. Load Qwen-3-4B-Thinking once per run; iterate α / layer / rank without reload where possible.
- **Reporting**: every table reports mean ± bootstrap 95% CI over MGSM problems (per language and pooled). Every plot annotates the specific claim it addresses.
- **Compute accounting for Claim 4**: measure wall-clock GPU-hours (verified against `nvidia-smi` / process wall-clock) *and* estimated forward+backward FLOPs. Report both; the compute ratio predicate uses GPU-hours as the primary metric.

## Reviewer-Concern Coverage (single self-review round, no external LLM chat)

- *"How do you know V_lang is the language subspace and not the 'style' or 'script' subspace?"* → T1 held-out language classifier + orthogonal-complement classifier + principal-angle-to-content-subspace triple; T2 leave-language-out fit tests generalization to *unseen* languages.
- *"How do you know accuracy gains are not from the projection acting as a regularizer?"* → T2 matched-rank random-subspace control (same-rank random operator should not improve accuracy).
- *"Is the fidelity metric fair — could GlotLID be miscalling low-resource languages?"* → GlotLID V3 covers all 11 targets including Bn/Sw/Te; report GlotLID's top-1 accuracy on FLORES-200 dev as a calibration reference in the appendix.
- *"Compute comparison in Claim 4 could be gamed by LoRA vs full SFT choice"* → report *both* the actually-run baseline compute and the extrapolated full-SFT compute (a documented estimate from MathOctopus/LinguaLIFT literature), and be honest about which is which.

## Remaining Risks (documented, not silenced)

- **Layer-scope disagreement**: Landscape has two views — LAPE says language neurons are top+bottom (support: leave top intact suppress bottom); Wendler / Zhao NeurIPS say English pivot is at middle. On Qwen-3-4B-Thinking (a reasoning-tuned distilled model) the answer may differ from vanilla Llama-2; T2's layer-range sweep tests this directly.
- **10-hour GPU budget vs full SFT**: full-parameter SFT on Qwen-3-4B-Thinking + 11-language MGSM8KInstruct is tight even with 5 GPUs. The plan defaults to LoRA-SFT with an extrapolated-full-SFT annotation, not silent full-SFT skipping.
- **Probe-set-size stability**: below n_probe ≈ 100 the SVD subspace may be noisy. T1's n_probe sweep quantifies this; the "small probe set" claim is verified operationally by the smallest n_probe at which held-out language classification hits ≥ 0.90.

## Verdict

**READY** for experiment planning. The problem anchor is frozen (the four verbatim task.md claims). The testing approach commits to the smallest adequate mechanism family (linear subspace projection) and defends the choice against LENS / LAPE alternatives by construction of the matched controls in T1/T2/T3. The plan is inside the HARD 10-hour GPU budget with LoRA-SFT as the actually-run Claim-4 baseline (RL as budget-permitting).
