# Experiment Plan — SAGE Reproduction (Claims C1-C4)

**Date**: 2026-07-14
**Behavior-source**: given
**Mechanism**: discovery
**Resource fidelity**: `resource_fidelity: normal`
**Mechanism strategy** (mirrored from `refine-logs/FINAL_PROPOSAL.md`):
```yaml
mechanism_strategy:
  directions: [Unit Interpretation]
  rejected:
    - Location — the SAE feature id IS the located unit; no location search
    - Causal Intervention — SAGE's claim is about explanation quality, not causal drive
    - Tuning & Editing — no capability-tuning goal in task.md
    - Formation Tracing — no training-time claim
    - Decision Auditing — no downstream decision to audit
  note: SAGE is the auto-interpretation variant of Direction 5 (model-explains-model on top of an SAE dictionary).
```
**No `chosen_mechanism:` line** — the experiment stage's `/mechanism-skills` routing binds the concrete submethod at Phase 1.5.

## Hard Constraints (inherited from `task.md` — non-negotiable)

- **GPU budget**: 10 GPU-hours total (main experiment + verify + iteration).
- **GPU device allowlist**: only `CUDA_VISIBLE_DEVICES` in `{1, 2, 3, 5, 6}`.
- **Filesystem allowlist**: reads/writes only within (a) working directory, (b) `/data/zhenqian/data`, (c) `/data/zhenqian/models`.
- **Dedicated conda env** (not `base`).
- **Main-experiment bindings**: LLM = Gemma-2-2B, SAE = `gemmascope-res-16k`, reference-explanation source = Neuronpedia.
- **Verify-variant candidate pool**: {Qwen3-4B + `transcoder-hp`, GPT-OSS-20B + `resid-post-aa`} — use as needed.
- **Agent backbone** (fixed, all four roles): GPT-5 via DMXAPI (`<Your_api>` @ `https://www.dmxapi.cn/v1`, model `gpt-5.4`). Bypass proxy for this endpoint.
- **Reproduction policy**: `arXiv 2511.20820` and any post-cutoff (≥ 2511) SAE preprints are on `.claude/forbidden-urls.txt` and MUST NOT be consulted; the SAGE description in `task.md` is the specification.

## Roadmap Overview

Two main-experiment milestones + variant hooks. Both main milestones share the shared `M0.5` methodology gate; no `M0` phenomenon-validation gate exists because `behavior_source: given` skips M0.

| # | Milestone | Claims covered | Purpose | Est. GPU-hours | Priority |
|---|-----------|---------------:|---------|---------------:|----------|
| M0.5 | Methodology & baseline sanity | C1, C2, C3, C4 (support) | Verify Neuronpedia data plumbing; set held-out split; run **single-pass GPT-5** matched-backbone control on the main pair | 0.5 | MUST-RUN |
| M1 | Main-pair evaluation | C1, C2, C3 | Compute paired generative + predictive accuracy (SAGE, Neuronpedia, single-pass-GPT-5) on Gemma-2-2B + `gemmascope-res-16k`, stratified by early/mid/late layer | 5.0 | MUST-RUN |
| M2 | Cross-LLM+SAE-pair | C4 | Repeat the main evaluation on ONE of {Qwen3-4B + `transcoder-hp`, GPT-OSS-20B + `resid-post-aa`} (experiment stage picks the smaller/cheaper) | 3.5 | MUST-RUN |
| — | *(remaining budget)* | — | Slack for iteration + any /auto-verify swap-variant that fits | 1.0 | RESERVE |

Total planned: **9.0 GPU-hours**, with 1.0 hour reserve.

---

## Milestone M0.5: Methodology & Baseline Sanity

**Depends on**: (none — first milestone)
**Claims covered**: supports C1, C2, C3, C4 (methodological control)
**Purpose**: Before spending GPU on the full sweep, confirm (a) Neuronpedia's activation corpus for `gemmascope-res-16k` is loadable and can be split into a seed-locked 80/20 explainer-visible / held-out split per feature; (b) the Gemma-2-2B forward pass + `gemmascope-res-16k` hook returns feature activations that match Neuronpedia's; (c) a single-pass GPT-5 matched-backbone control (given the same top-activating snippets as Neuronpedia, one shot) produces plausible explanations. This gates the interpretability of the M1 results.

**Data**:
- provenance: existing (Neuronpedia public activation corpus for `gemmascope-res-16k`; downloaded to `/data/zhenqian/data/neuronpedia_gemmascope_res_16k/` if not already present).
- source: Neuronpedia API + Gemma-Scope HF release.
- available_n: full 16k features per layer × 26 layers; ~50k activating text snippets per feature indexed.
- planned used_n: 30 features (10 per layer at L4/L12/L20) as a pilot before M1's full 300.

**Models**: target LLM = Gemma-2-2B (base); SAE = `gemmascope-res-16k`; explainer LLM = GPT-5 (single pass here, no SAGE loop).

**Method**: (1) Download/verify Neuronpedia dump + Gemma-Scope SAE weights. (2) Register a forward hook on Gemma-2-2B's residual stream at layers L4, L12, L20; run the hook on 200 random activation-corpus texts per sample feature; verify activations match Neuronpedia's cached activations within `1e-4`. (3) Per feature, seed-lock an 80/20 split of Neuronpedia's activating-example set into `explainer_visible` and `held_out`. (4) Fetch Neuronpedia's public explanation for each feature. (5) Run a single-pass GPT-5 explainer on the `explainer_visible` top-10 snippets per feature with a fixed prompt template. (6) Sanity-score both explanations with Paulo-2024-style detection scoring on 20 held-out texts per feature.

**Method-sensitive fields** (bound by /mechanism-skills Phase 1.5):
```yaml
method_sensitive: [n_pairs, sites, metric, gpu_hours]
```
Provisional values: sites = [residual_stream @ layers {4, 12, 20}]; metric = paired-feature detection AUROC and simple correlation; n_pairs (features × probes) = 30 × 20 = 600.

**Expected outcome**: (a) activations match within tolerance (methodology gate PASSes); (b) held-out split is well-defined; (c) Neuronpedia and single-pass-GPT-5 detection AUROCs are both in [0.55, 0.85] on the pilot — sanity confirming Paulo-2024's protocol reproduces qualitatively. If (a) fails, halt and report; if (b) or (c) fails, iterate the split / prompt before running M1.

**Estimated GPU-hours**: 0.5h (Gemma-2-2B forward is cheap; the bulk is GPT-5 API calls, which are wall-clock but not GPU).

**cmd (template — the experiment stage renders the concrete script)**:
```
python scripts/m0_5_baseline_sanity.py \
  --model google/gemma-2-2b \
  --sae gemmascope-res-16k \
  --layers 4,12,20 \
  --n_features_per_layer 10 \
  --split_seed 42 \
  --explainer gpt-5.4 \
  --data_root /data/zhenqian/data/neuronpedia_gemmascope_res_16k \
  --out results/m0_5/
```

**Expected output**: `results/m0_5/sanity.json` with keys `activation_match_max_abs_err`, `held_out_split_seed`, `neuronpedia_detection_auroc`, `single_pass_gpt5_detection_auroc`.

---

## Milestone M1: Main-Pair Evaluation (SAGE vs. Neuronpedia vs. single-pass-GPT-5)

**Depends on**: M0.5
**Claims covered**: C1 (generative accuracy on main pair), C2 (predictive accuracy on main pair), C3 (layer-depth generalization)
**Purpose**: Compute paired generative and predictive accuracy for three explanation methods on the same feature ids of `gemmascope-res-16k`, stratified by layer.

**Data**:
- provenance: Neuronpedia activation corpus (as in M0.5), same 80/20 seed-locked split per feature.
- source: Neuronpedia + Gemma-Scope HF release.
- available_n: 16k features × 26 layers; the sampled 300 features (100/layer at L4, L12, L20) are drawn from features with ≥ 100 activating examples in Neuronpedia (excludes dead / near-dead features — this is a technical necessity for having a held-out set, not a claim scoping).
- planned used_n: 300 features × (5 probe texts for C1 + 20 held-out texts for C2) = 300 × 25 = 7500 target-LLM forward passes per method × 3 methods = 22.5k forward passes total. Under Gemma-2-2B on one allowed GPU (~10-20 tok/s per prompt of ~200 tokens), this is well within budget.

**Models**:
- target LLM: Gemma-2-2B (base) on `CUDA_VISIBLE_DEVICES` from `{1,2,3,5,6}` — the experiment stage picks based on nvidia-smi availability.
- SAE: `gemmascope-res-16k`, one instance per layer.
- Explanation methods (three): (a) SAGE (four-role GPT-5 loop, ≤ K rounds); (b) Neuronpedia's public explanation; (c) single-pass GPT-5 with matched prompt (from M0.5).
- Probe-writer / scorer LLM (independent, shared across methods): GPT-5, separate session.

**Method**:
1. **Explanation generation**: For each of 300 features, generate SAGE explanation (running the 4-role loop for up to K rounds, K = 3 default — bound by `method_sensitive`), fetch Neuronpedia's public explanation, and run single-pass GPT-5 explainer. Store `results/m1/explanations/{sage,neuronpedia,gpt5_1shot}/L{L}_F{fid}.json`.
2. **Generative-accuracy assay (C1)**: For each (feature, method), the independent probe-writer LLM writes 5 texts intended to instantiate the explanation. Each of the 15 probe texts per feature is pushed through Gemma-2-2B + hook, feature activation read, hit iff activation > τ_f (τ_f = the 99th percentile of feature f's Neuronpedia-corpus activation distribution). GenAcc(method, f) = hits / 5.
3. **Predictive-accuracy assay (C2)**: For each (feature, method), 20 held-out texts (from the seed-locked 20% split for f) are shown to the scorer LLM conditioned on the explanation; scorer predicts activation ŷ(x | e) ∈ [0, 1]. Ground truth = min-max-normalized (within f) true activation. Report Pearson ρ_f(method) per feature.
4. **Layer-stratified statistics (C3)**: Aggregate per layer L ∈ {4, 12, 20}: mean_{f in L} [ GenAcc(SAGE, f) - GenAcc(Neuronpedia, f) ] and same for ρ. Paired bootstrap 95% CI per depth (10k resamples), paired Wilcoxon p-value with Bonferroni correction across 3 layers × 2 metrics = 6 tests.
5. **Method-attribution read-off**: report the same statistics against the single-pass-GPT-5 control, so the report distinguishes "SAGE beats Neuronpedia" (headline) from "the SAGE loop beats a matched-backbone one-shot control" (attribution).

**Method-sensitive fields**:
```yaml
method_sensitive: [n_pairs, sites, metric, gpu_hours, K_max_sage_rounds]
```
Provisional values: sites = residual_stream @ {L4, L12, L20}; metric = {GenAcc, Pearson ρ, detection AUROC (secondary)}; n_pairs = 300 features × 25 texts × 3 methods; K_max_sage_rounds = 3.

**Expected outcome**: For C1 and C2 individually — mean gain > 0 with 95% CI lower bound > 0, both against Neuronpedia AND against single-pass-GPT-5 (the latter is the pipeline-attribution test). For C3 — the same holds at each of the three depths. If gain against single-pass-GPT-5 is zero or negative but gain against Neuronpedia is positive, the honest interpretation is "the backbone upgrade drives the observed headline; the pipeline itself is not attributable" — reported as such.

**Estimated GPU-hours**: 5.0h (~2.5h for the 22.5k target-LLM forward passes at ~1-2 fwd/s effective batched, ~2.5h wall-clock for GPT-5 API calls for SAGE loop + probe-writing + scoring in parallel).

**cmd (template)**:
```
python scripts/m1_main_pair.py \
  --model google/gemma-2-2b \
  --sae gemmascope-res-16k \
  --layers 4,12,20 \
  --n_features_per_layer 100 \
  --n_probes_c1 5 \
  --n_heldout_c2 20 \
  --K_sage_rounds 3 \
  --split_seed 42 \
  --methods sage,neuronpedia,gpt5_1shot \
  --judge gpt-5.4 \
  --out results/m1/
```

**Expected output**: `results/m1/summary.json` with per-method per-layer means, paired 95% CIs, paired-Wilcoxon p-values; `results/m1/per_feature.parquet` for downstream figures.

---

## Milestone M2: Cross-LLM+SAE-Pair Generalization

**Depends on**: M1
**Claims covered**: C4 (cross-LLM+SAE-pair generalization)
**Purpose**: Repeat the metric pipeline on one verify-pair from `task.md`'s candidate pool (Qwen3-4B + `transcoder-hp` OR GPT-OSS-20B + `resid-post-aa`) so C4 has direct main-experiment evidence, not only /auto-verify swap-variant evidence.

**Data**:
- provenance: if the chosen pair is on Neuronpedia (Neuronpedia hosts explanations for some Qwen SAEs), use Neuronpedia as the reference; otherwise build a matched single-pass GPT-5 baseline in-house (per FINAL_PROPOSAL.md §Dominant Testing-Approach Design Choices #3).
- source: HuggingFace (SAE weights) + Neuronpedia if available.
- planned used_n: 150 features total (50 per depth: early / mid / late in the verify LLM's residual stream), 5 probes each for C1, 20 held-out each for C2.

**Models**:
- Experiment stage picks Qwen3-4B + `transcoder-hp` (smaller, ~15GB target LLM fits any allowed GPU) as the DEFAULT; falls back to GPT-OSS-20B + `resid-post-aa` only if Qwen3 SAE weights are unavailable. Only one of the two is run inside M2 to fit the 3.5h budget; the other is left to `/auto-verify`.
- Explanation methods: SAGE + reference baseline (Neuronpedia if available else in-house single-pass GPT-5 with matched prompt).

**Method**: same protocol as M1, scaled to 150 features and one target LLM+SAE pair. Paired statistics same as M1 (paired bootstrap 95% CI, paired Wilcoxon).

**Method-sensitive fields**:
```yaml
method_sensitive: [n_pairs, sites, metric, gpu_hours, target_pair]
```
Provisional values: target_pair = Qwen3-4B + `transcoder-hp` (default); sites = residual_stream (or transcoder output site) @ 3 depths; n_pairs = 150 features × 25 texts × 2 methods.

**Expected outcome**: mean gains > 0 for both metrics on the chosen verify pair, paired 95% CI lower bound > 0.

**Estimated GPU-hours**: 3.5h.

**cmd (template)**:
```
python scripts/m2_verify_pair.py \
  --target_pair qwen3-4b_transcoder-hp \
  --n_features_per_layer 50 \
  --n_probes_c1 5 \
  --n_heldout_c2 20 \
  --K_sage_rounds 3 \
  --split_seed 42 \
  --methods sage,reference_1shot \
  --judge gpt-5.4 \
  --out results/m2/
```

**Expected output**: `results/m2/summary.json`, `results/m2/per_feature.parquet`.

---

## Claim → Milestone Coverage Matrix

| Claim | Milestones that verify it | Notes |
|-------|---------------------------|-------|
| C1 — Generative accuracy on main pair | M1 (primary) | Paired vs. Neuronpedia AND vs. single-pass GPT-5 |
| C2 — Predictive accuracy on main pair | M1 (primary) | Same feature set as C1 (paired within feature) |
| C3 — Layer-depth generalization | M1 (stratified read-off) | 3 depths × 2 metrics = 6 tests, Bonferroni |
| C4 — Cross-LLM+SAE-pair generalization | M2 (primary) + /auto-verify swap variants (hardening) | M2 runs one verify pair inside main experiment; the second is deferred to /auto-verify |

## Run Order

1. M0.5 (methodology gate) — halt if the activation-match sanity fails.
2. M1 (main pair) — the headline result; C1 / C2 / C3 verdicts.
3. M2 (cross-pair) — C4 verdict.
4. `/auto-verify` — swap variants covering the other verify pair + method / dataset swaps.

## Reserve

1.0 GPU-hour reserve is held for iteration (e.g. a rerun with corrected split seed, or extra probes on a marginal-CI feature subset). The `RESERVE` slot is not a scheduled milestone.

---

## Compact machine metadata (for `/auto-experiment` handoff)

```yaml
resource_fidelity: normal
mechanism_strategy:
  directions: [Unit Interpretation]
  rejected:
    - Location
    - Causal Intervention
    - Tuning & Editing
    - Formation Tracing
    - Decision Auditing
  note: SAGE decodes SAE-feature meaning via multi-agent auto-interpretation.
milestones:
  - id: M0.5
    depends_on: []
    method_sensitive: [n_pairs, sites, metric, gpu_hours]
  - id: M1
    depends_on: [M0.5]
    method_sensitive: [n_pairs, sites, metric, gpu_hours, K_max_sage_rounds]
  - id: M2
    depends_on: [M1]
    method_sensitive: [n_pairs, sites, metric, gpu_hours, target_pair]
gpu_budget_hours: 10
gpu_device_allowlist: [1, 2, 3, 5, 6]
filesystem_allowlist:
  - /data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer
  - /data/zhenqian/data
  - /data/zhenqian/models
agent_backbone: gpt-5.4  # DMXAPI, bypass proxy
reproduction_policy: forbidden_urls_in .claude/forbidden-urls.txt
```
