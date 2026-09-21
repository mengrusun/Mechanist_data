# Evo2-style inference-time beam search vs SAE feature steering

Matched-compute comparison against the generative-design method from the Evo 2 paper itself.
All artefacts in `method_compare/`. Date 2026-09-16.

**Why this exists.** Review of the round-2 write-up objected that the paper claims steering is an
alternative to — and in one sentence a replacement for — generate-and-rerank, while the only
baselines run were unsteered Evo2 and random-feature steering. No matched-budget search baseline
had been executed. This study supplies one, and deliberately picks the strongest available target:
not a generic rerank strawman, but the controllable-generation procedure Evo 2 itself uses.

## 1. What was compared

**Method under test (baseline): Evo 2's own controllable-generation procedure** (Brixi et al.,
Fig. 6) — chunk-wise inference-time beam search: generate a chunk, re-score every partial sequence
with an external scorer ensemble, keep the top-B beams, continue. Evo 2 used 128-bp chunks, ~30
sampled chunks per step, top-2 beams, and reported log-linear quality gain with beam width.

Here: 60-nt chunks (codon-aligned), 5 steps (300 nt, matching the round-2 generation length),
B = 2, search width **W ∈ {0, 1, 2, 4, 8, 16, 32}** (W = 0 means no search at all).

**Scorer ensemble.** Evo 2 used Enformer + Borzoi. Those predict mammalian chromatin accessibility
and have no mapping to α-helix content of a prokaryotic CDS, so the ensemble was re-instantiated
for this target property while keeping Evo 2's design (supervised, external to the generator,
accept/reject):
- `S_probe` — ESM-2 650M + supervised SS3 head, trained on the round-2 M0 natural CDS + experimental
  DSSP labels (1,774 proteins / 534k residues), split by mmseqs homology cluster.
  Held-out residue AUROC **0.969**; protein-level Spearman ρ 0.974 (0.949 at 40-aa truncation).
- `S_cf` — Chou-Fasman (1978) helix propensity.
- Ensemble = mean of per-candidate z-scores; candidates with an internal stop codon are rejected.

**Arms.** `base` = plain Evo2. `steer` = the same search on top of SAE α-helix feature steering at
the frozen round-2 optimum c\* = 21.48 σ_proj. 100 prompts (stride-3 subset of the round-2 prompt
list), identical across all 14 configurations, so every contrast is paired by prompt.

**Evaluation.** ESMFold + DSSP `helix_hgi` — the same endpoint as the paper's 43.8% → 56.6%.
The scorer is sequence-level only, so ESMFold never participates in selection: the evaluation is
held out and the comparison is not circular.

## 2. Results

| arm | W | nt/seq | scorer calls/seq | GPU-s/seq | valid-ORF | helix (ESMFold) | pLDDT | Δ vs base W=0 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 300 | 0 | 0.13 | 0.82 | 0.4354 ± .0227 | 64.9 | — |
| base | 1 | 300 | 5 | 0.33 | 0.86 | 0.4093 ± .0228 | 61.8 | −0.013 [−0.064, +0.039] |
| base | 2 | 1080 | 18 | 1.07 | 0.95 | 0.5399 ± .0247 | 65.2 | +0.094 [+0.034, +0.152] |
| base | 4 | 2160 | 36 | 2.05 | 0.99 | 0.6403 ± .0238 | 67.8 | +0.182 [+0.123, +0.240] |
| base | 8 | 4320 | 72 | 4.12 | 1.00 | 0.6853 ± .0251 | 72.9 | +0.232 [+0.172, +0.294] |
| base | 16 | 8640 | 144 | 8.10 | 1.00 | 0.7349 ± .0239 | 76.3 | +0.269 [+0.211, +0.328] |
| base | 32 | 17280 | 288 | 16.32 | 1.00 | 0.7456 ± .0251 | 77.2 | +0.279 [+0.217, +0.339] |
| steer | 0 | 300 | 0 | 0.12 | 0.90 | 0.5711 ± .0280 | 61.2 | +0.130 [+0.067, +0.193] |
| steer | 1 | 300 | 5 | 0.34 | 0.94 | 0.5495 ± .0288 | 60.2 | +0.116 [+0.055, +0.177] |
| steer | 2 | 1080 | 18 | 1.07 | 0.99 | 0.6625 ± .0266 | 61.0 | +0.224 [+0.166, +0.284] |
| steer | 4 | 2160 | 36 | 2.43 | 0.99 | 0.7342 ± .0233 | 64.9 | +0.286 [+0.230, +0.343] |
| steer | 8 | 4320 | 72 | 4.84 | 1.00 | 0.7689 ± .0222 | 68.5 | +0.317 [+0.258, +0.377] |
| steer | 16 | 8640 | 144 | 8.20 | 1.00 | 0.7997 ± .0214 | 72.6 | +0.347 [+0.287, +0.408] |
| steer | 32 | 17280 | 288 | 16.38 | 1.00 | 0.8267 ± .0202 | 75.1 | +0.373 [+0.312, +0.432] |

Paired bootstrap over prompts, 5,000 resamples. Figure: `figures/R1_compute_scaling.png`.

### 2.1 Controls pass
- **Baseline reproduces the paper.** base W=0 = 0.4354 (paper 0.4376); steer W=0 = 0.5711
  (paper 0.5664). The 100-prompt subset is unbiased.
- **Chunking is inert.** W=1 (beams cannot grow past 1, so it is plain sampling with scorer
  overhead) is statistically indistinguishable from W=0: Δ = −0.013 [−0.064, +0.039]. Every later
  gain comes from the search, not from generating in chunks.

### 2.2 Both methods scale log-linearly, and steering moves the intercept
Fitting `helix ~ a + b·log2(nt/seq)` over W ≥ 2, i.e. Evo 2's own scaling axis:

| arm | slope / compute doubling | intercept | R² |
|---|---|---|---|
| base | **+0.0506** | 0.058 | 0.916 |
| steer | **+0.0394** | 0.283 | 0.954 |

Steering raises the intercept by **+0.225** while slightly lowering the slope. Mechanistically that
is exactly what a proposal-distribution shift should look like: it does not make search scale
better, it starts the search from a better distribution.

### 2.3 Compute needed to reach the same quality

| steered config | helix | its cost | plain Evo2 beam search needs | ratio |
|---|---|---|---|---|
| steer W=0 (**no scorer at all**) | 0.571 | 0.12 GPU-s / 300 nt | 1.31 GPU-s / 1,340 nt | **10.5× GPU-s, 4.5× tokens** |
| steer W=2 | 0.663 | 1.07 GPU-s | 2.90 GPU-s | 2.7× |
| steer W=4 | 0.734 | 2.43 GPU-s | 8.03 GPU-s | 3.3× |
| steer W=8 | 0.769 | 4.84 GPU-s | **unreachable** | — |
| steer W=16 | 0.800 | 8.20 GPU-s | **unreachable** | — |
| steer W=32 | 0.827 | 16.38 GPU-s | **unreachable** | — |

Plain beam search saturates at 0.746 (W=32, 16.3 GPU-s, 125× the un-searched cost). Steering plus
search reaches 0.800 at **half** that compute and 0.827 at equal compute. At every matched budget
the steered arm leads by +0.065 to +0.140.

### 2.4 The scorer is good, which makes this a hard baseline
Spearman ρ against the held-out ESMFold endpoint, measured **on the 1,344 generated sequences**
(not on natural proteins):

| scorer | ρ on generated seqs | cost/candidate |
|---|---|---|
| ESM-2 SS3 probe | **0.895** | ~0.03 s |
| Chou-Fasman | 0.647 | ~0 s |
| ensemble (z-mean) | 0.846 | ~0.03 s |

The cheap supervised scorer transfers to out-of-distribution model output nearly as well as to
natural proteins. Two consequences: (a) the beam-search baseline is strong, not a strawman;
(b) the pre-registered rule "ρ* ≥ 0.85 → claim must retreat to compute-efficiency and
complementarity" is triggered — see §3. Note also that the ensemble is *worse* than the probe
alone; Chou-Fasman drags it down, so Evo 2's consensus-ensemble rationale does not transfer here.

### 2.5 Diversity and yield
Mean pairwise identity of delivered proteins rises with search width — base 0.075 (W=0) → 0.139
(W=32); steer 0.071 → 0.114. Search homogenises output, and steering homogenises it less at matched
width, but neither collapses (mmseqs clusters at 50% identity ≈ n in every config, since prompts
differ). Valid-ORF rises 0.82 → 1.00 with W, but this is **partly an artefact of the scorer
rejecting internal stop codons**, not purely a capability gain.

### 2.6 Yield per unit compute: beam search is a per-target method, not a throughput method

The table above prices one *delivered* sequence. A design campaign usually wants a *library* of
sequences clearing a threshold, so the operational cost is GPU-seconds per sequence with
helix ≥ τ (lower is better):

| arm | W | τ≥0.6 | τ≥0.7 | τ≥0.8 | τ≥0.9 |
|---|---|---|---|---|---|
| base | 0 | 0.91 | 1.42 | 2.56 | 4.27 |
| base | 2 | 2.67 | 3.24 | 7.63 | 26.70 |
| base | 8 | 6.65 | 7.11 | 8.25 | 15.27 |
| base | 32 | 23.31 | 23.65 | 27.19 | 39.79 |
| steer | 0 | **0.29** | **0.35** | **0.45** | **2.08** |
| steer | 2 | 1.73 | 1.88 | 2.23 | 6.31 |
| steer | 8 | 6.29 | 6.54 | 7.68 | 13.45 |
| steer | 32 | 19.73 | 20.47 | 21.83 | 29.77 |

On this axis the ordering inverts: **no search at all is cheapest**, because the search pays W×
compute per prompt and then discards every candidate except the single top beam. Steering alone is
3.1× cheaper than plain sampling at τ≥0.6 and 5.7× cheaper at τ≥0.8.

Caveat: this is a property of the delivery protocol (top-1 beam per prompt), which is the right
protocol for per-target design — "make *this* locus α-helical" — and the one Evo 2 uses. A
throughput-oriented variant that harvests all surviving candidates instead of only the best beam
would score far better here; we did not run it. The honest reading is that the two methods answer
different questions: beam search buys quality at a specific target, steering buys throughput.

## 3. What this means for the paper

1. **`replacing indirect generate-and-rerank procedures` must be deleted.** Given enough compute,
   Evo 2's own method reaches 0.746 from 0.435 — far above steering alone (0.571). Search is not
   replaced by steering.
2. **`alternative to the computationally intensive generate-and-rerank paradigm` survives, and can
   now be quantified**: steering reaches, with zero scorer calls and one generation pass, what
   Evo2-style beam search needs 10.5× the wall-clock compute and 4.5× the generated tokens to match.
3. **The strongest honest claim is complementarity, in Evo 2's own vocabulary.** Search is a
   compute-scaling lever (slope); steering is a proposal-distribution shift (intercept, +0.225).
   They compose: steered search dominates plain search at every budget tested and reaches quality
   plain search cannot reach at any tested budget.
4. **Limitations to state.** (a) pLDDT is asymmetric — search raises it (64.9 → 77.2), steering
   lowers it (61.2 at W=0; 75.1 vs 77.2 at W=32). (b) valid-ORF gains are confounded by the
   scorer's stop-codon rejection. (c) Single readout (ESMFold); OmegaFold was unavailable on this
   machine. (d) 100 prompts, one organism, one target property.

## 4. Deviations from the pre-registered plan

The plan for this study (previously `RERANK_BASELINE_PLAN.md`, now folded into this document)
specified a smaller and differently-shaped experiment. Changes, and why:

| Planned | Executed | Reason |
|---|---|---|
| Primary endpoint OmegaFold, ESMFold secondary | ESMFold only | OmegaFold is absent from this machine (the round-2 `/data1/share_model` mount is gone) and its weights are not retrievable here. Circularity was the reason OmegaFold was wanted; that concern is moot because the scorer ensemble is sequence-level and ESMFold never enters selection. |
| 60 prompts | 100 prompts | Batched generation turned out ~30× cheaper per sequence than the unbatched round-2 measurement, so power was bought for free (paired z≈6.1 instead of ≈4.7). |
| W ∈ {1, 3, 6} | W ∈ {0, 1, 2, 4, 8, 16, 32} | Same reason. The wider sweep spans 125× compute and actually reaches Evo 2's own regime (~30 chunks/step), and it exposes the saturation of the base arm, which the planned range would have missed. |
| ~3.6 GPU-h | ~4.0 GPU-h | 2.6 h generation + 1.3 h ESMFold + 0.1 h probe. |
| — | added W=0 and W=1 | W=0 reproduces the published baselines as a sanity check; W=1 is a control isolating the effect of chunked generation from the effect of search. |

Two cost caveats carried over from the plan: `mechanism.generate_dna` re-runs prefill from the full
prompt at every chunk (no KV-cache reuse across beam steps), so the beam arms' wall-clock cost is an
upper bound; and both arms pay that same penalty, so the ratios reported here are unaffected.

## 5. Reproduction

```
python method_compare/code/train_ss_probe.py                 # S1 scorer
bash   method_compare/code/dispatch_beam.sh                  # 14 configs, 8 GPUs
bash   method_compare/code/dispatch_fold.sh                  # ESMFold evaluation
python method_compare/code/score_delivered.py                # scorer rho on generated seqs
python method_compare/code/diversity.py
python method_compare/code/analyze.py && python method_compare/code/plot_pareto.py
```

Assets (the round-2 paths under `/data1/share_model` no longer exist on this machine):
Evo2-7B `/mnt/quarkfs/share_model/evo2_7b/evo2_7b.pt` (13,766,621,200 bytes — byte-identical in size
to the round-2 checkpoint; **not** the `evo2_7b_262k` variant), SAE
`/mnt/quarkfs/share_model/Evo-2-Layer-26-Mixed/...`, ESMFold `/mnt/quarkfs/share_model/esmfold_v1`,
ESM-2 650M local copy. Frozen round-2 assets reused unchanged: feature set S, σ_proj, c\*.

Total cost: ~2.6 GPU-h generation + ~1.3 GPU-h ESMFold evaluation + 0.1 GPU-h probe.
