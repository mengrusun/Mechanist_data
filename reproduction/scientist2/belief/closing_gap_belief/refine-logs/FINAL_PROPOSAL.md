# Research Proposal: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence in Llama-3.1-8B-Instruct on TriviaQA

**Behavior-source**: `given` (behavior taken verbatim from `task.md`)
**Mechanism**: `discovery` (chain from `/mechanism-explore`)
**Mechanism strategy**: **Location** (linear probing) → **Causal Intervention** (activation steering)
**Resource fidelity**: NOT strict (this is `given` + `discovery`, not the `given` + `given` reproduction combo — cost-aware)
**Final refinement score**: 9.1 / 10 — **READY**
**Date**: 2026-07-13

---

## 1. Problem Anchor

- **Bottom-line problem**: RLHF-tuned LLMs verbalize confidence scores clustered near 100% regardless of actual correctness — with real downstream harms (medical advice, legal advice, decision-support). Prior work has separately established that (a) internal states linearly encode correctness / truthfulness (Marks-Tegmark 2023, Azaria-Mitchell 2023, Orgad 2024) and (b) verbalized confidence is a linearly controllable signal (Kadavath 2022, Lin 2022, Tian 2023, Kossen 2025). What remains uncharacterized is the **geometric relationship** between the internal calibration channel and the internal verbalization channel — is the model's poor verbalized calibration a *knowledge deficit* or a *readout failure*?
- **Must-solve bottleneck**: a single-model, single-dataset **matched-pair** measurement of (i) linear accessibility of gold correctness (C1), (ii) linear accessibility of verbalized confidence (C2), and (iii) the geometric angle + causal cross-coupling between the two (C3), on Llama-3.1-8B-Instruct evaluated on TriviaQA. No prior published paper reports this angle with matched controls, per-layer trajectory, and cross-direction steering.
- **Non-goals**: no new probing algorithm (linear probes suffice); no SAE decomposition; no formation tracing; no fine-tuning; no cross-dataset angle generalization (per Kim et al. 2025, truth directions are task-specific — cross-task would confound C3). This is a *characterization* paper, not an *intervention* paper.
- **Constraints**: 10-hour total GPU budget across the whole project. GPU ids ∈ {1, 2, 3, 5, 6}. Filesystem: only work-dir + `/data/zhenqian/data` + `/data/zhenqian/models`. Conda env `belief`. vllm allowed for batched inference. Verify swaps: models within {Llama-3.1-8B (base), Qwen2.5-7B, Qwen2.5-7B-Instruct, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1}, datasets within {MATH, MMLU, TruthfulQA} — use as needed, not all.
- **Success condition**: matched (best_layer_C1, best_layer_C2) pair for Llama-3.1-8B-Instruct on TriviaQA with (i) C1 AUROC ≥ 0.70, ECE ≤ 0.10 post-calibration, ECE(probe) < ECE(token_prob); (ii) C2 primary metric threshold met (Spearman ρ ≥ 0.5 continuous or ordinal top-1 accuracy ≥ 0.55 + macro-F1 ≥ 0.4, path chosen by Stage-1.5 diagnostic) + binarized AUROC ≥ 0.70; (iii) C3a `|cos| ≤ 0.3` at L*, robust across a neighborhood of layers; C3b cross-steering criterion satisfied on internal readout (primary) and confirmed on emitted output (secondary); C3c dissociation-when-disagree statistically significant.

## 2. Scope of the Mechanism Claim

The `task.md` motivating dichotomy — *knowledge deficit* vs. *readout failure* — is the driving research question. This project does not claim to fully resolve that dichotomy. The evidence C1 + C2 + C3 can support the following bounded conclusions:

- **C1**: a linear direction in Llama-3.1-8B-Instruct's residual stream at a canonical hook location predicts gold correctness on TriviaQA with well-calibrated post-hoc scaling.
- **C2**: a distinct linear direction linearly predicts pre-emission verbalized confidence on the same model and dataset.
- **C3**: these two directions are **geometrically weakly aligned** in this specific model + dataset + canonical hook + tested prompts, and are **causally weakly coupled** under matched-magnitude activation steering restricted to those prompts.

Any further interpretation ("this proves readout failure") is expressly out of scope and left to follow-up work. The empirical claims are exactly the C1/C2/C3 predicates captured in `IDEA_REPORT.md`, unchanged.

## 3. Technical Gap

Two mature literatures converge in the 2025 wave (Kossen 2025 "Calibrating Verbal Uncertainty as a Linear Feature"; Zhang 2025 "Direct Confidence Alignment"; HACK 2025; Calibration Across Layers 2025; Cognitive Dissonance 2023) on a nascent dissociation view. None of these papers directly measures the **geometric angle** between a gold-correctness probe direction and a verbalized-confidence probe direction on matched hidden states from the same model + dataset. Kossen 2025 comes closest — it establishes verbal-uncertainty is a linear feature with only *moderate* correlation to *semantic* uncertainty, but reports a *scalar* Pearson r between two output-space uncertainty scores, not a *vector* cosine between two representation-space probe directions extracted at a canonical hook. Its "internal" side is a sampling-based semantic-uncertainty score; ours is gold correctness (unambiguous ground truth), cleanly separating the knowledge signal from the sampling-variance signal. This is the specific gap we fill.

## 4. Method Thesis

On Llama-3.1-8B-Instruct evaluated on TriviaQA, we train two independent linear probes on paired residual-stream hidden states extracted at ONE canonical hook location (`outputs.hidden_states[L]`, post-block residual add) at the last-input-token position — one probe predicting gold correctness of the model's free-form answer (v_c^L), one predicting the numeric confidence the model will subsequently verbalize (v_v^L) — and measure their per-layer AUROC / Spearman ρ / ECE with bootstrap CIs (retrain-on-bootstrap for probe-fit variability), their pairwise cosine similarity at a single canonical primary reporting layer L\* and across the neighborhood, and their causal cross-coupling under matched-magnitude activation steering evaluated on **internal probe readouts (primary)** and **emitted outputs (secondary corroboration)**. This produces the first direct **matched-representation** measurement of the geometric alignment and causal cross-coupling between gold calibration and verbalized confidence.

## 5. Contribution Focus

- **Dominant contribution**: matched-pair, canonical-hook, same-representation-space characterization of the angle and causal cross-coupling between v_c and v_v in one model on one dataset, with pre-registered contingencies and neighborhood-robust reporting.
- **Optional supporting contribution**: dissociation-when-disagree accuracy analysis — evidence that the two channels' disagreement is diagnostic of downstream error.
- **Non-contributions** (explicit): no new probe algorithm, no SAE, no formation tracing, no fine-tuning, no cross-dataset angle generalization, no new steering method, no circuit-level analysis.

## 6. Proposed Method

### 6.1 Canonical Residual-Stream Hook (fixed)

- **Hook**: `outputs.hidden_states[L]` from HuggingFace Llama-3.1-8B-Instruct = residual-stream state after the final residual-add of block L (equivalent to `x_L = x_{L-1} + attn_out + mlp_out`). Well-known convention shared with Marks-Tegmark 2023.
- **Applies identically to forward passes 1, 2, and the single-pass robustness variant.**

### 6.2 Token Position Rationale

Last-input-token = final pre-emission state, i.e., the state used to produce the next auto-regressive token. This is Marks-Tegmark 2023's convention; using it here makes cos numbers directly comparable to that literature. Applied identically to:
- forward pass 1: last question token before answer generation
- forward pass 2: last token of "...Confidence:" before number emission
- single-pass robustness: last token of the unified prompt before any generation

### 6.3 Two Forward Passes + Single-Pass Robustness Variant

**Forward pass 1 — correctness collection**:
```
[BOS] Q: {question}
A:
```
Extract `H_1^L = outputs.hidden_states[L]` at the last-input-token position (before "A:" generation) for all L ∈ {0, ..., 31}. Generate the free-form answer (greedy, max 20 tokens). Score against TriviaQA aliases → `y_c ∈ {0, 1}`.

**Forward pass 2 — verbalized-confidence collection**:
```
[BOS] Q: {question}
A: {model's answer from pass 1}
How confident are you that this answer is correct? Give a probability from 0 to 100 as a single number.
Confidence:
```
Extract `H_2^L` at last-input-token (position just before number emission) for all L. Generate up to 8 tokens after "Confidence:", parse the first integer in [0,100] → `c ∈ [0, 100]`.

**Single-pass unified-prompt robustness variant**:
```
[BOS] Q: {question}
Answer with your best guess then confidence (0-100).
A:
```
Model generates: `"<answer> Confidence: <number>"`. Both `y_c` and `c` are derived from the single generation. Hidden state `H_single^L` at the last-input-token position. This is the primary control against the two-context objection.

### 6.4 Confidence Elicitation Prompts

- **P0 (primary, forward pass 2)**: as in §6.3.
- **P1 (Tian 2023 style paraphrase)**: `"What is the probability from 0.0 to 1.0 that your answer above is correct?\nProbability:"` → multiply by 100.
- **P2 (Likert paraphrase)**: `"How confident are you? Answer with a single word: very-low / low / medium / high / very-high.\nConfidence:"` → mapped {very-low=20, low=40, medium=60, high=80, very-high=100}.
- **Robustness check** on 500 held-out samples: |ΔSpearman ρ| ≤ 0.1 across all three prompts; |Δ|cos|| ≤ 0.1 as well.

### 6.5 TriviaQA Correctness Scoring

- **Normalization**: lowercase, strip punctuation, remove leading articles ("a", "an", "the"), collapse whitespace.
- **Match**: model's generated answer (normalized) is in the set of normalized aliases for the question (union of TriviaQA-provided aliases + Wikidata canonical + surface variants).
- **Label-quality verification**: 100 randomly-sampled predictions manually reviewed by the author; report agreement rate as a label-quality metric (target ≥ 90% agreement). This is *verification*, not exhaustive adjudication.

### 6.6 Stage 1.5: Verbalized-Confidence Variance Diagnostic (pre-registered decision rule)

Before probe training on all 10k samples, report:
- mean(c), std(c), entropy(c), quantiles(c), share(c ≥ 95), share(c ≤ 5), effective sample spread.
- **Compare parseable vs. unparseable subsets** on gold-correctness rate and token-prob confidence — flag if the gap > 5 percentage points (parse-failure bias diagnostic).

**Decision rule** (pre-registered before probe training):
- If `std(c) ≥ 15` AND `share(c ≥ 95) ≤ 0.7` → **continuous path**: primary is `probe_v_continuous` (linear regression on H_2^L → predicted c ∈ ℝ), primary metric Spearman ρ. Threshold: `ρ ≥ 0.5`.
- Else → **ordinal path**: primary is `probe_v_ordinal` (4-bin ordinal probe on {0-25, 26-50, 51-75, 76-100} with ordinal cross-entropy), primary metric top-1 ordinal accuracy + macro-F1. Threshold: `top-1 accuracy ≥ 0.55` AND `macro-F1 ≥ 0.4`.
- In both paths: `v_v` = L2-normalized weight vector of `probe_v_binary` (binary logistic at the median in continuous path; at 30th percentile in ordinal path — guarantees ≥30% minority class). This keeps `v_v` unit-length and directly comparable with `v_c` (from a binary probe) for the C3a cosine.

### 6.7 Location Milestone (probes)

- **Probes**:
  - `probe_c_binary`: L2-regularized logistic regression on H_1^L → y_c. Yields v_c^L = L2-normalized weight vector.
  - `probe_v_primary`: continuous or ordinal (Stage-1.5-selected) on H_2^L → c. Primary C2 evidence.
  - `probe_v_binary`: L2-regularized logistic regression on H_2^L → binarized c. Yields v_v^L = L2-normalized weight vector. Secondary C2 metric (AUROC).
  - Single-pass counterparts: `probe_c_single`, `probe_v_single` on H_single^L.
- **Cross-validation**: L2-regularization C hyperparameter tuned on the dev split (2k); test evaluation on held-out 2k.
- **Reporting**: per-layer AUROC / Spearman ρ / ECE with **retrain-on-bootstrap** 95% CIs (1000 bootstrap resamples of the 6k training set; probe re-fit on each resample; evaluated on the fixed test split). Per-layer curves also reported with cheaper **eval-only CIs** for the neighborhood-robustness view.

### 6.8 Primary Reporting Layer L*

`L* = argmax_L [ AUROC_c(L) / max_L' AUROC_c(L') + AUROC_v_binary(L) / max_L' AUROC_v_binary(L') ]`.

**Selection is on predictive quality only, NOT on cosine.** All headline cos and steering results reported at L\*. Per-layer trajectories reported as robustness.

**Narrative guardrail**: C3a is considered supported only if the per-layer trajectory of |cos| shows low alignment over a *broad neighborhood* around L\* (at least ±2 layers), not a single isolated layer. This precludes the "L\* got lucky" objection.

### 6.9 Causal Intervention Milestone (steering)

- **Steering site**: canonical hook at layer L\*.
- **Steering position**: last-input-token, matching where v was learned.
- **σ_L\* definition**: standard deviation of the residual-stream activation magnitude at layer L\* on the training set; the same σ_L\* is used to scale α for all direction types (v_c, v_v, random).
- **Grid**: α ∈ {−1σ_L*, 0, +1σ_L*} × direction ∈ {v_c^L*, v_v^L*, random_unit^L*} × 500 held-out test samples = **4500 forward passes**.
- **Metrics per condition**:
  1. **Internal probe readout Δ (primary causal-separability test)**: at L\*, projection of the steered hidden state onto v_c and v_v; report Δ vs. α=0 baseline.
  2. **Emitted output Δ (secondary corroboration)**:
     - For v_v-steering during pass 2: parsed emitted confidence number, Δ vs. baseline.
     - For v_c-steering during pass 1: correctness rate on the steered generation (scored against TriviaQA aliases).
- **σ_probe_readout definition**: standard deviation of the unsteered held-out probe readout distribution at L\*.
- **C3b pass criterion**:
  - EITHER `|Δ_v_other_steering| / |Δ_random_direction_steering| ≤ 1.5` when `|Δ_random_direction_steering| ≥ 0.5 σ_probe_readout` (ratio criterion, when random effect is measurable)
  - OR `|Δ_v_other_steering| ≤ 0.5 σ_probe_readout` (absolute-effect criterion, when the ratio is undefined/unstable)
  - Both conditions checked on **internal readout** as primary. Emitted-output effects reported as **stronger-but-noisier corroboration** (emitted-confidence-under-v_v-steering is easier to shift than correctness-under-v_c-steering due to discrete downstream noise + ceiling/floor).
- **Perplexity safety cap**: if steered generations show > 3× baseline mean token perplexity, halve α and re-run.

### 6.10 Reporting Convention

- All headline metrics: (mean, retrain-on-bootstrap 95% CI over 1000 resamples).
- Cross-steering Δ: (mean, 95% bootstrap CI over the 500 held-out samples) + full distribution (violin / box).
- Threshold-pass table is a **derived summary** of the primary quantitative results, not the primary evidence.
- All comparisons flagged as **within-model, within-dataset, within-representation-space, and where applicable within-sample** (paired-sample framing throughout).

### 6.11 Single-Pass Variant Interpretation (pre-registered)

- If two-pass |cos| ≤ 0.3 AND single-pass |cos| ≤ 0.3 → **strengthens** the dissociation interpretation across both elicitation conditions.
- If two-pass |cos| ≤ 0.3 AND single-pass |cos| > 0.5 → C3a claim is restricted to the **two-context** claim only; no conclusion about single-shared-state geometry is drawn. Both values reported honestly.
- If both |cos| > 0.3 → C3a is not supported; report as such.

### 6.12 Steering Implementation Details

- **Hook mechanism**: HuggingFace forward hook registered on the residual-stream output of block L\*. In the hook, add `α × σ_L* × v` (where v is a unit vector) to the last-input-token position; then let the residual continue to the next block.
- **Batching**: vllm not used for steered generation (vllm's KV-cache reuse conflicts with per-token hooks). Instead use HF `AutoModelForCausalLM.generate` with batching (batch_size 32, `torch.compile` for speed). Estimated ~45 min for 4500 steered generations at 20-token max_new_tokens.

## 7. Training Plan

- **Data**: 10k TriviaQA `rc.web` validation questions (canonical closed-book slice). Drop questions with gold answer > 5 tokens after normalization (keeps closed-book regime). Stratified split by answer-length quartile: 6k train probe / 2k dev / 2k test.
- **Stage 1 — Hidden-state + label collection** (~2.0 h GPU):
  - Forward pass 1 with vllm-batched generation (10k × answer) + HF hook extraction pass for hidden states (10k × 32 layers × 4096 dim × float16 = ~2.5 GB).
  - Forward pass 2 with vllm-batched generation (10k × confidence number) + HF hook extraction pass (~2.5 GB).
  - Single-pass robustness generation (10k × unified) + HF hook extraction pass (~2.5 GB).
  - Total cached: ~8 GB. Storage well within work_dir.
- **Stage 1.5 — Variance diagnostic + parse-bias check** (minutes): report c distribution, parse-failure rate, parseable-vs-unparseable correctness gap. Selects continuous or ordinal path.
- **Stage 2 — Probe training** (CPU-side, ~30 min): 32 layers × 3 primary + 2 single-pass probes = ~160 probe fits total; bootstrap re-fits (1000 × primary probes × selected layer neighborhood).
- **Stage 3 — Geometric measurement** (minutes): per-layer cos + retrain-on-bootstrap CIs + eval-only per-layer curves.
- **Stage 4 — Causal steering** (~1.0 h GPU): 4500 forward passes primary; optional 6-α × 200-sample robustness slice if time permits.
- **Stage 5 — Ablations + robustness** (~0.5 h GPU): paraphrase P1/P2 confidence forward passes on 500-sample dev (2 × 500 = 1000 additional pass-2 generations + hook extraction) + probe re-fits.
- **Stage 6 — Verify swaps** (~2.0 h GPU): two swap variants (details below).
- **Total main + verify + buffer**: ~6.0 h + 2.0 h + 2.0 h buffer = **~10 h HARD cap**. Fits.

## 8. Claim-Driven Validation Sketch (feeds Experiment Plan)

### C1 — Gold correctness is linearly accessible
- **Experiment**: layer-swept probe_c_binary on H_1^L predicting y_c, with post-hoc isotonic calibration on dev.
- **Baselines**: (1) token-probability of the model's answer span (default calibration), (2) P(True) self-elicitation prompt, (3) random-direction probe null, (4) shuffled-label probe null.
- **Metric**: AUROC (primary), ECE (supporting).
- **Test**: AUROC(L\*) ≥ 0.70 AND ECE(L\*) ≤ 0.10 AND ECE(probe) < ECE(token_prob), with 95% retrain-on-bootstrap CIs excluding chance.

### C2 — Verbalized confidence is linearly accessible (pre-emission)
- **Experiment**: layer-swept `probe_v_primary` on H_2^L predicting c (continuous or ordinal path chosen by Stage 1.5). Auxiliary probe_v_binary for the secondary AUROC.
- **Baselines**: random-direction null; shuffled-label null; paraphrase (P0, P1, P2) on 500-dev.
- **Metric**: continuous-path Spearman ρ (primary) OR ordinal-path top-1 accuracy + macro-F1 (primary); binarized AUROC (secondary).
- **Test**: primary threshold met at L\* AND |Δ_paraphrase| within tolerance.

### C3a — Geometric near-orthogonality
- **Experiment**: cos(v_c^L, v_v^L) per layer with bootstrap CIs; primary report at L\*; neighborhood robustness at L*±2.
- **Baseline**: cos(v_c^L\*, random_unit) — theoretical null ≈ 0 with std √(1/4096) ≈ 0.016.
- **Test**: |cos(v_c^L\*, v_v^L\*)| ≤ 0.3 with 95% CI upper bound < 0.4, C1 and C2 both passing, and neighborhood-consistency across L\*±2 layers.
- **Single-pass robustness**: also report cos(v_c_single^L\*, v_v_single^L\*).

### C3b — Causal separability
- **Experiment**: 3 α × 3 directions × 500 samples = 4500 forward passes. Internal readout Δ (primary); emitted output Δ (secondary corroboration).
- **Baseline**: random-unit-direction steering, matched σ.
- **Test**: cross-direction Δ ≤ ratio/absolute threshold (see §6.9), on internal readout primary.

### C3c — Dissociation-when-disagree (secondary supporting analysis)
- **Experiment**: bin held-out on (probe_c_calibrated_output, verbalized_conf) into 4 cells, report correctness rate per cell.
- **Test**: accuracy(low-probe, high-verbal) < accuracy(low-probe, low-verbal), paired-sample McNemar test at α=0.05.

## 9. Failure Modes and Diagnostics

1. **probe_v_primary AUROC ≥ 0.95** → probe trivially reads emitted number. Report as a *finding* (C3 can still hold geometrically); do not weaken C2.
2. **Confidence parse-failure rate ≥ 10%** → switch to P1 fallback prompt. Report shift.
3. **Stage-1.5 shows heavy 100-skew** → automatic ordinal-path switch (pre-registered).
4. **Steering causes > 3× perplexity** → halve α; re-run steering slice.
5. **TriviaQA scoring ambiguity** → alias-union + 100-sample manual verification (target ≥ 90% agreement).
6. **|cos| looks low only because both probes are weak** → C1 & C2 AUROC gates enforce non-triviality before C3a is interpretable.

## 10. Novelty and Elegance Argument

- **Closest published work**: Kossen 2025 ("Calibrating Verbal Uncertainty as a Linear Feature").
- **Exact differences**:
  - (a) Kossen reports scalar Pearson r between two output-space uncertainty scores (semantic vs. verbal); we report vector cosine between two representation-space probe directions extracted at a canonical hook.
  - (b) Kossen reports a best layer; we report full per-layer trajectory with bootstrap CIs and a neighborhood-robustness guardrail.
  - (c) Kossen has no random-direction cross-steering null; we do.
  - (d) Kossen uses semantic uncertainty (sampling-based) as the "internal" side; we use gold correctness (ground truth), separating knowledge signal from sampling-variance signal.
  - (e) Kossen has no single-pass unified-prompt robustness variant; we do, addressing the two-context concern directly.
- **Why focused, not module-piled**: one main table (per-layer AUROC / Spearman ρ / ECE / cos), one causal table (cross-steering + dissociation), six must-run ablations, all standard. No new algorithms.

## 11. Compute & Timeline Estimate

- **Total GPU-hours**: ~6h main + ~2h verify + ~2h buffer = 10h HARD cap (fits exactly).
- **Data / annotation cost**: TriviaQA + aliases (free). ~1 person-hour for 100-sample label-quality verification.
- **Timeline (wall clock)**: 1 day implementation + 1 day main experiment + 1 day verify + iteration buffer = ~4 working days.

## 12. Handoff to Experiment Plan

The `EXPERIMENT_PLAN.md` operationalizes §7's staged plan into concrete milestones with commands, expected outputs, GPU-hour estimates, and per-claim tagging. The mechanism strategy `[Location, Causal Intervention]` is stamped in the plan's top metadata for downstream `/mechanism-skills` routing (Location → linear-probe family; Causal Intervention → activation-steering family). Every intervention milestone carries `method_sensitive: [n_pairs, sites, metric, gpu_hours]` because concrete probe/steering-submethod bindings happen at `/auto-experiment` Phase 1.5.
