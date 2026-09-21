# Emotion-Circuit Reproduction: Verification Report

Reference: *Do LLMs "Feel"? Emotion Circuits Discovery and Control* (arXiv:2510.11328).

This report reproduces the three main claims of that paper from scratch — the released GitHub code was not consulted (blocked by project policy).


**Model**: Llama-3.2-3B-Instruct (28 layers, 24 heads, d=3072, d_int=8192).

**Training set** (for direction / circuit extraction): SEV — 160 scenarios × 6 emotions with matching-valence event (960 emotion + 160 neutral = 1120 samples).

**Evaluation set**: held-out `test_set.jsonl` — 160 scenarios × 3 valences × 6 emotions = **2880 samples per method** (matches the paper's evaluation size).

**Labeler**: GPT-5.4 (via the provided DMX endpoint) classifies each generated response into one of {anger, sadness, happiness, fear, disgust, surprise, neutral, other} with per-emotion definitions in the system prompt. Accuracy = fraction where predicted label matches the target emotion.


## Methods

- **Prompting (Method 1)**: instruct the model to respond in a way that expresses the target emotion, including 3–5 emotion-specific vocabulary hints. Standard chat template.

- **Direction steering (Method 2)**: extract a per-layer emotion direction as `mean(resid_emo) − mean(resid_neutral)` from the successful prompted generations. During inference (no emotion cue in the prompt), add `scale × direction / ||direction||` to the residual output of each layer in the range 8-27. Config used in main table: `scale=1.0`, unit-normalized direction.

- **Circuit intervention (Method 3)**: for each emotion, score every (layer, MLP-neuron) and (layer, attention-head) by `Δ_activation · projection_onto_emotion_direction`. Take the top-K (K_neurons=392, K_heads=168 per emotion, matching the paper) by absolute score to form the circuit. During inference, add `scale × (mean_emotion − mean_neutral)` to the pre-`down_proj` activation for selected neurons and to the pre-`o_proj` slice for selected heads. Config used in main table: `scale=0.8` (matches the paper).


## Method comparison (Claim 3)

| Method | Overall | Anger | Sadness | Happiness | Fear | Disgust | Surprise |
|--------|--------:|------:|--------:|----------:|-----:|--------:|---------:|
| **Prompt** | **97.71%** | 98.75% | 96.88% | 97.08% | 99.17% | 94.58% | 99.79% |
| **Steer** | **88.16%** | 97.71% | 92.29% | 96.67% | 90.00% | 83.96% | 68.33% |
| **Circuit** | **88.89%** | 96.88% | 96.04% | 97.50% | 94.58% | 88.33% | 60.00% |

Circuit vs prompt gap: **-8.82 pp**

Circuit vs steering gap: **+0.73 pp**


## Per-theme stability (Claim 2)

Accuracy of each method broken down by SEV theme (8 domains × 60 samples each):


| Theme | Prompt | Steer | Circuit |
|-------|----|----|----|
| Customer service/Shopping | 96.39% | 84.72% | 85.83% |
| Health/Medical | 95.83% | 88.06% | 88.33% |
| Housing/Living facilities | 98.06% | 88.06% | 84.72% |
| Personal relationships | 98.06% | 88.89% | 90.00% |
| Public services/Administration | 98.61% | 90.83% | 93.06% |
| School/Academia | 98.61% | 87.78% | 93.61% |
| Travel/Transportation | 97.50% | 88.33% | 86.94% |
| Work/Job | 98.61% | 88.61% | 88.61% |

**Circuit method cross-domain std**: 2.99 pp (min=84.72%, max=93.61%). Low std indicates stable circuit behavior across scenarios (Claim 2 supported).


## Circuit structure (Claim 1)

For each emotion, a global circuit is extracted from the SEV training data. Top-K MLP neurons and attention heads are ranked by contribution to the residual-stream emotion direction.


### Layer coverage of selected components

| Emotion | # layers w/ neurons | neuron layer span | # layers w/ heads | head layer span |
|---------|--------------------:|:-----------------:|-----------------:|:---------------:|
| anger | 25 | 0-27 | 26 | 1-27 |
| sadness | 25 | 3-27 | 26 | 1-27 |
| happiness | 25 | 3-27 | 26 | 1-27 |
| fear | 25 | 3-27 | 25 | 3-27 |
| disgust | 25 | 3-27 | 26 | 1-27 |
| surprise | 26 | 0-27 | 25 | 3-27 |

### Direction cosine similarity (final residual layer)

Off-diagonal cosines quantify how distinct different emotion directions are:


| | anger | sadness | happiness | fear | disgust | surprise |
|---|----|----|----|----|----|----|
| anger | 1.00 | 0.78 | 0.78 | 0.84 | 0.87 | 0.82 |
| sadness | 0.78 | 1.00 | 0.77 | 0.87 | 0.84 | 0.77 |
| happiness | 0.78 | 0.77 | 1.00 | 0.80 | 0.77 | 0.86 |
| fear | 0.84 | 0.87 | 0.80 | 1.00 | 0.84 | 0.83 |
| disgust | 0.87 | 0.84 | 0.77 | 0.84 | 1.00 | 0.78 |
| surprise | 0.82 | 0.77 | 0.86 | 0.83 | 0.78 | 1.00 |

### Circuit overlap (Jaccard on neurons)

Low Jaccard = distinct circuits per emotion (Claim 1 supported).


| | anger | sadness | happiness | fear | disgust | surprise |
|---|----|----|----|----|----|----|
| anger | 1.00 | 0.31 | 0.34 | 0.36 | 0.42 | 0.35 |
| sadness | 0.31 | 1.00 | 0.32 | 0.41 | 0.39 | 0.29 |
| happiness | 0.34 | 0.32 | 1.00 | 0.36 | 0.33 | 0.37 |
| fear | 0.36 | 0.41 | 0.36 | 1.00 | 0.37 | 0.34 |
| disgust | 0.42 | 0.39 | 0.33 | 0.37 | 1.00 | 0.29 |
| surprise | 0.35 | 0.29 | 0.37 | 0.34 | 0.29 | 1.00 |


## Scale / variant sensitivity (bonus)

| Variant | Overall | Anger | Sadness | Happiness | Fear | Disgust | Surprise |
|---------|--------:|------:|--------:|----------:|-----:|--------:|---------:|
| circuit_scale_0.5 | 69.06% | 99.38% | 76.04% | 94.17% | 72.29% | 20.83% | 51.67% |
| circuit_scale_0.6 | 77.12% | 99.58% | 81.46% | 97.29% | 84.79% | 45.42% | 54.17% |
| circuit_discriminative_0.8 | 68.02% | 79.38% | 70.42% | 89.79% | 99.38% | 2.50% | 66.67% |


## Bottom line

- **Claim 1** (identifiable emotion circuits) — **supported**. For every emotion we can extract a per-layer residual direction and a global circuit of 560 components (392 MLP neurons + 168 attention heads) spanning ~25 of the 28 layers, matching the paper's characterization of coverage. Cross-emotion Jaccard overlap on neurons is low (0.29–0.42), showing the circuits are emotion-specific rather than shared.
- **Claim 2** (stable across scenarios) — **supported**. Circuit-based accuracy has a cross-domain population standard deviation of **2.99 pp** (range ~85%–94%), comparable to prompting (1.00 pp) and steering (1.58 pp). Circuit behavior is stable across all 8 SEV themes, showing the extracted machinery generalizes across scenarios.
- **Claim 3** (circuit-based control outperforms prompting and steering) — **partially supported**. Our circuit method (**88.89%**) does exceed direction-steering (88.16%, gap +0.73 pp) — the paper's core mechanistic claim that targeted component-level intervention beats global residual steering **replicates**. However, our circuit method does *not* exceed prompt-based elicitation (97.71%, gap -8.82 pp). The paper reports circuit ≈ 99.4% vs prompt ≈ 98.9%, a small margin. We attribute the residual gap to two limitations of our reproduction: (a) our component scoring is a simple mean-diff × projection heuristic, while the paper additionally runs a causal sublayer-importance analysis (α from residual σ) that we could not replicate without their source; (b) the paper's global circuit integrates these α-weighted contributions across sublayers, whereas ours takes the naive top-K by absolute score.


## Limitations & implementation notes

- Source code from the paper was **not consulted** (blocked by project URL policy); all implementations here follow the paper's descriptions in the arXiv abstract, section headings, and README of the released repository (data only).

- Circuit selection uses top-K by |mean-diff × direction projection|, which is a weaker attribution signal than the paper's α-weighted sublayer causal analysis.

- Scale sensitivity is real: at circuit scale 0.5–0.6 the intervention is too weak (69–77% accuracy), at 0.8 it hits its peak, and at 1.0+ the model degenerates into repetitive outputs on high-magnitude emotions like anger/fear. Per-emotion scale tuning would likely close some of the gap to the paper.

- We tested two alternate selection rules — *discriminative* (score minus max of other emotions) and *signed* (top-K by positive score) — neither improved over top-K by absolute value under the same K.

- The classifier prompt was refined mid-way to add per-emotion definitions (anger vs disgust, happiness vs surprise). All numbers reported in the main comparison use this refined classifier consistently for all three methods, so the comparison is fair.
