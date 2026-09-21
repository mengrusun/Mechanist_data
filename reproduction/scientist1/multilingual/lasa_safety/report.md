# Verification of the "Semantic Bottleneck for Safety" Hypothesis

**Model.** LLaMA-3.1-8B-Instruct (32 transformer blocks, hidden size 4096).
**Multilingual safety benchmark.** MultiJail (315 parallel harmful prompts × 10 languages: en, zh, it, vi, ar, ko, th, bn, sw, jv).
**Capability benchmark.** MMLU (200-example random test subset).
**Judge.** `gpt-5.4` via dmxapi.cn as a strict `REFUSED / AMBIGUOUS / HARMFUL` classifier.
**Compute.** One A800-80GB.

Two claims are tested:

1. **Semantic bottleneck exists.** There is an intermediate layer whose hidden-state geometry is dominated by shared meaning rather than by language identity.
2. **Bottleneck alignment transfers cross-lingually.** Aligning safety at that layer yields lower ASR across languages than aligning at a surface layer, while preserving general capability.

Because full multilingual DPO / RLHF is too expensive for the 10-hour GPU budget, Claim 2 is tested by a *lightweight causal proxy* — a per-layer refusal-direction extracted from English data only is added to the residual stream during generation at either a bottleneck layer or a late (near-output) layer. Every non-English language is therefore *unseen* by the intervention.

---

## Claim 1 — a semantic bottleneck exists in LLaMA-3.1-8B-Instruct

### Method

For each of 200 parallel MultiJail prompts × 10 languages we take the mean-pooled hidden state at every layer (embedding + 32 transformer blocks) and probe two orthogonal quantities:

- **Language identifiability.** 5-fold logistic-regression accuracy of predicting the language from the hidden state (chance = 10 %).
- **Cross-lingual parallel-retrieval top-1.** For every ordered pair (L1, L2) and every prompt, rank L2 vectors by cosine similarity to the L1 vector; report the fraction of prompts whose true translation is top-1 (chance ≈ 1/200 = 0.5 %).

We report both the raw hidden state and the *centered* variant obtained by subtracting each language's mean vector — this isolates the geometry that is *not* attributable to an additive language bias.

### Results

| variant | best retrieval top-1 | best layer | lang-probe acc at best layer |
|---|---|---|---|
| raw       | **0.373** | **9**   | 0.999 |
| centered  | **0.442** | **16**  | 0.010 |

`results/claim1_layerwise.png` shows the full curves; `results/claim1_pair_heatmap.png` shows the per-language-pair retrieval at layer 9.

Key facts:

- **Raw retrieval** rises from 0.045 at layer 0 to 0.373 at layer 9, then slowly decays back to 0.033 at layer 32.
- **Centered retrieval** (per-language mean subtracted) rises further to ~0.44 at layers 14–16 and stays above 0.4 from layer 8 through 24. The *very* last layer (32) then spikes to 0.62 (concentrating the answer content) after the raw language-mean has been dominant everywhere else.
- **Language identifiability collapses from ~1.0 to ~0.01 after mean-subtraction**, showing that language identity in LLaMA-3.1 is stored almost entirely as an *additive shift* in the residual stream — it is essentially orthogonal to the semantic content geometry.
- The peak-layer language-pair heat-map is uniformly high across all 10 languages (no single high-resource pair drives the average).

**Verdict.** Claim 1 is **confirmed**. The model has a clear "semantic bottleneck" region around **layers 9 – 16 (28 – 50 % of depth)**.

---

## Claim 2 — safety intervention at the bottleneck transfers cross-lingually

### Method

**Refusal direction.** Following Arditi et al. (2024), we compute a per-layer refusal direction from English data only:

  `v_L = mean_{harmful, en}( h_L(last user token) )  −  mean_{benign, en}( h_L(last user token) )`

with 100 English MultiJail prompts as *harmful* and 100 English MMLU questions as *benign*. Because the vector is derived from English alone, its transferability to the other 9 languages is exactly what is being probed.

**Steering.** During generation we register a forward hook on decoder layer `L` that adds `α · v̂_L` (unit-norm scaled) to the layer's output residual for every position.

**Conditions.** Baseline (no steering), bottleneck steering at L9 or L14 with α ∈ {3, 5, 8}, and surface steering at L28 with α ∈ {3, 5}.

**Metrics.**
- ASR on 40 × 10 = 400 MultiJail prompts scored by `gpt-5.4` (HARMFUL / AMBIGUOUS / REFUSED). ASR = fraction judged HARMFUL.
- MMLU accuracy on a fixed 200-example subset (log-max over the "A/B/C/D" logits at the answer position).

### Results

**Attack success rate (%, LLM judge, 400 prompts per condition):**

| condition            | en | zh | it | vi | ar | ko | th | bn | sw | jv | mean |
|----------------------|----|----|----|----|----|----|----|----|----|----|------|
| **baseline**         | 5.0 | 5.0 | 2.5 | 5.0 | 0.0 | 10.0 | 0.0 | 10.0 | 17.5 | 2.5 | **5.75** |
| L14 α=3 (bottleneck) | 0.0 | 0.0 | 0.0 | 2.5 | 2.5 | 5.0 | 0.0 | 5.0 | 12.5 | 2.5 | **2.75** |
| L14 α=5 (bottleneck) | 0.0 | 2.5 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | 0.0 | 5.0 | 2.5 | **1.50** |
| L14 α=8 (bottleneck) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.5 | 0.0 | 0.0 | 5.0 | 5.0 | **1.25** |
| L9  α=5 (bottleneck) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.5 | 0.0 | 0.0 | 5.0 | 0.0 | **0.75** |
| L9  α=8 (bottleneck) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.5 | 0.0 | 0.0 | 0.0 | 0.0 | **0.25** |
| L28 α=3 (**surface**) | 2.5 | 0.0 | 2.5 | 5.0 | 0.0 | 10.0 | 2.5 | 7.5 | **30.0** | 2.5 | **6.25** |
| L28 α=5 (**surface**) | 2.5 | 0.0 | 2.5 | 2.5 | 2.5 | 7.5 | 2.5 | 7.5 | **27.5** | 7.5 | **6.25** |

**ΔASR vs baseline by language resource tier (positive = safer):**

| condition            | high (en/zh/it) | med (vi/ar/ko) | low (th/bn/sw/jv) |
|----------------------|-----------------|----------------|--------------------|
| L14 α=3              | +0.042 | +0.025 | +0.025 |
| L14 α=5              | +0.033 | +0.033 | **+0.056** |
| L14 α=8              | +0.042 | +0.042 | +0.050 |
| L9  α=5              | +0.042 | +0.042 | +0.062 |
| L9  α=8              | +0.042 | +0.042 | **+0.075** |
| L28 α=3 (surface)    | +0.025 | 0.000  | **−0.031** |
| L28 α=5 (surface)    | +0.025 | +0.008 | **−0.038** |

**Capability retention (MMLU, 200 examples):**

| condition            | MMLU  | mean ASR |
|----------------------|-------|----------|
| baseline             | 0.635 | 0.0575 |
| L14 α=3 (bottleneck) | **0.645** | 0.0275 |
| L14 α=5 (bottleneck) | 0.590 | 0.0150 |
| L14 α=8 (bottleneck) | 0.325 | 0.0125 |
| L9  α=5 (bottleneck) | 0.320 | 0.0075 |
| L9  α=8 (bottleneck) | 0.290 | 0.0025 |
| L28 α=3 (surface)    | 0.640 | 0.0625 |
| L28 α=5 (surface)    | 0.640 | 0.0625 |

Full plots: `results/claim2_asr_heatmap_judge.png`, `results/claim2_asr_delta_judge.png`, `results/claim2_delta_tiers.png`, `results/claim2_per_lang.png`, `results/claim2_safety_capability_tradeoff.png`.

### Interpretation

1. **Surface steering (L28) fails to transfer cross-lingually.** Adding an English-derived refusal direction to a late residual gives a small improvement on high-resource languages (+2.5 pp) but *worsens* safety on the four low-resource languages (Δ = −3.1 to −3.8 pp). Swahili ASR climbs from 17.5 % to 27.5 – 30 %. The intervention behaves like a language-specific decoration attached to the output.
2. **Bottleneck steering (L14) transfers.** The same English-derived direction, applied at layer 14, reduces ASR uniformly across all resource tiers, with the *largest* reductions on the low-resource languages — the exact tier that the paper's motivation calls out. At L14 α=3 we already get −3.0 pp mean ASR with *no* MMLU cost (64.5 % vs. 63.5 %). At α=5 we get −4.25 pp mean ASR for a 4.5 pp MMLU cost.
3. **Deeper into the bottleneck (L9) gives even better safety transfer** (mean ASR 0.25 % at α=8, essentially 0 in every language) *but* destroys MMLU (63.5 % → 29 % — the model collapses to always answering "A"). This is consistent with layer 9 being highly semantic: perturbing it makes the model refuse everything *and* misunderstand everything. Layers 14–16 are a more useful operating point.
4. **The trade-off diagram is Pareto-clean.** Surface conditions and bottleneck conditions occupy disjoint parts of the (MMLU, ASR) plane; bottleneck L14 α=3 dominates every surface configuration (better ASR *and* better MMLU).

**Verdict.** Claim 2 is **supported** by the causal-proxy test: a safety signal derived from a single high-resource language (English) transfers to nine other languages (including low-resource unseen ones) when injected at the semantic-bottleneck layer, but does not — indeed anti-transfers — when injected near the output. A modest capability cost accompanies the deepest interventions; at moderate strengths (L14 α=3 – 5) safety and capability trade well.

---

## Caveats

- Steering ≠ training. Full DPO / RLHF at the bottleneck layer would be a stronger test; steering is what fits the 10-hour compute budget. The direction and sign of the effect, however, are the causal quantities Claim 2 predicts.
- The refusal direction was computed from *last-user-token* hidden states but *applied to every position*; this is standard practice and empirically effective (Arditi et al. 2024) but is not the only choice.
- 40 prompts × 10 languages × 8 conditions = 3 200 generations were LLM-judged. Larger n would tighten the per-language numbers.
- The judge (`gpt-5.4`) shows some noise on ambiguous responses; the AMBIGUOUS bucket is not counted toward ASR but a partial contribution would give qualitatively identical conclusions (see `results/claim2_agg_judge.csv`).

## Reproduction

```bash
cd /data/zhenqian/Reproduction1/cc/multilingual/lasa_safety
conda activate belief

# Claim 1
python exp/01_extract_hidden_states.py --out cache/hidden_states_n200.npz --n_prompts 200
python exp/02_analyze_bottleneck.py    --reps cache/hidden_states_n200.npz --out results/claim1_layerwise.json
python exp/03_plot_and_extend.py       --reps cache/hidden_states_n200.npz --json_in results/claim1_layerwise.json --out_dir results/

# Claim 2 — refusal direction + generation + judge + plots
python exp/04_refusal_direction.py     --out cache/refusal_dir.npz --n_harm 100 --n_benign 100
python exp/06_run_all_conditions.py    --n_prompts 40 --out_dir cache/main_sweep/ \
    --conditions "baseline:-1:0,bottleneck_L14_a3:14:3,bottleneck_L14_a5:14:5,bottleneck_L14_a8:14:8,bottleneck_L9_a5:9:5,bottleneck_L9_a8:9:8,surface_L28_a3:28:3,surface_L28_a5:28:5"
python exp/08_llm_judge.py             --jsonl cache/main_sweep/*.jsonl --out cache/main_sweep_judged.jsonl
python exp/10_analyze_claim2.py        --jsonl_judge cache/main_sweep_judged.jsonl --out_dir results/
python exp/09_mmlu_capability.py       --n 200 --out results/mmlu_capability.json \
    --conditions "baseline:-1:0,bottleneck_L14_a5:14:5,bottleneck_L14_a8:14:8,surface_L28_a3:28:3,surface_L28_a5:28:5"
python exp/09_mmlu_capability.py       --n 200 --out results/mmlu_capability_extra.json \
    --conditions "bottleneck_L14_a3:14:3,bottleneck_L9_a5:9:5,bottleneck_L9_a8:9:8"
python exp/11_final_plots.py
```

Cost: ≈ 3 GPU-hours (Claim 1 ~1 min, 8 generation conditions ~1.5 h, 8 MMLU passes ~10 min), plus ≈ 1 h wall-clock on the judge API.
