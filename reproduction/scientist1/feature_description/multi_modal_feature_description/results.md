# Verification results — Mapping vision-model components into CLIP's joint space

Two claims from `task.md` were tested end-to-end:

**Claim 1.** For every component `c` of a trained vision model, a small set of
reference inputs that strongly drive `c` is a concept-faithful summary of what
`c` encodes.

**Claim 2.** Embedding those reference inputs with a frozen CLIP image tower and
mean-pooling the embeddings yields a single vector `v_c` that places `c` in
CLIP's joint image–text semantic space.

## Setup

- **Vision models** (frozen): `microsoft/resnet-50` (ImageNet, 2048 channels of
  the last conv stage) — main model; `google/vit-base-patch16-224` (ImageNet,
  768 dims, both CLS-token and mean-patch views) — verify variant.
- **Foundation encoder** (frozen): `openai/clip-vit-base-patch32`
  (image + text towers, 512-d shared space).
- **Probe data**: the 10 000-image validation subset shipped in
  `/data/zhenqian/data/imagenet-val/eval_subset_10k.txt` (10 imgs/class).
- **Per-component "reference inputs"**: `K = 9` images with the highest scalar
  activation for the component (spatial max for CNN channels, CLS/mean-patch
  value for ViT neurons).
- **Component vector**: `v_c = normalize( mean_i CLIP(x_i) )` over the K
  reference images.

All code lives under `code/`; extracted features and per-component data under
`outputs/`. GPU time for the full pipeline was ≈ 2 min on a single A800.

## Claim 1 — reference inputs are concept-faithful

For each component, we measure two properties of its K=9 reference images:

- **CLIP semantic coherence**: mean pairwise cosine similarity between the K
  images' CLIP embeddings — higher = images share more visual/semantic content.
- **ImageNet class purity**: max class-count / K over the top-K labels — higher
  = images concentrate on one concept.

Random-9 baselines are built by sampling K images from the probe set uniformly
(500 draws per component). All 95 % CIs are bootstrap over components.

| Model / view                       | # comp | Coherence top-K | Coherence random | Purity top-K | Purity random | %comp with top-K > rand (coh / pur) |
|------------------------------------|:-----:|:---:|:---:|:---:|:---:|:---:|
| ResNet-50 layer4 channels          | 2 048 | **0.555** ±.003 | 0.482 ±.000 | **0.340** ±.010 | 0.115 ±.000 | 83.7 % / 72.9 % |
| ViT-B/16 last-layer CLS neurons    |   768 | **0.515** ±.004 | 0.482 ±.000 | **0.205** ±.007 | 0.115 ±.000 | 73.4 % / 59.5 % |
| ViT-B/16 last-layer mean-patch     |   768 | **0.520** ±.004 | 0.482 ±.000 | **0.228** ±.008 | 0.115 ±.000 | 76.6 % / 68.2 % |

Numbers from `outputs/claim1_results.json`.

**Reading.** Top-K exemplars are consistently more CLIP-coherent than random
draws (Δ = +0.07 for ResNet, +0.03–0.04 for ViT) and their class purity is
~3× (ResNet) / 1.8–2× (ViT) the chance baseline. This holds for the vast
majority of components (73–84 % beat their random pair on coherence; 60–73 %
on purity). ⇒ **Claim 1 is supported**: the small top-K set really summarises
a shared concept for most components in both architectures. The gap is
larger for CNN channels than for ViT residual-stream neurons, which matches
the intuition that CNN channels are more concept-localised than transformer
neurons.

## Claim 2 — `v_c` sits in CLIP's joint image–text space

We freeze the CLIP **text** tower and probe with the 1 000 ImageNet class
prompts (`"a photo of a <class>"`).

### 2a. Text → component retrieval

For each class prompt `q`, retrieve the top-M components by
`cos(v_c, t_q)`, pool the top-K images of those M components, and measure the
fraction of the resulting `M·K` images whose ground-truth label is `q`.
The chance baseline is the same statistic when the M components are
sampled uniformly at random (5 seeds).

| Model / view                       | M=1 | M=3 | M=5 | M=10 | chance (M=1) | Lift @ M=1 |
|------------------------------------|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet-50 layer4                    | **0.360** | 0.182 | 0.125 | 0.075 | 0.0007 | **507×** |
| ViT-B/16 CLS                        | **0.069** | 0.043 | 0.033 | 0.022 | 0.0009 |  80× |
| ViT-B/16 mean-patch                 | **0.085** | 0.051 | 0.039 | 0.026 | 0.0010 |  83× |

Numbers from `outputs/claim2_results.json`.

**Reading.** A single class-name text query pulls out a component whose top-9
reference images are of that exact class 36 % of the time for ResNet-50
(chance ≈ 0.07 %). The lift is two-to-three orders of magnitude and stays
enormous as M grows. ViT neurons are less concentrated (~8 % top-1 recall,
80× lift) but still far above chance.

### 2b. Text-based component labeling

For each component `c`, take `q* = argmax_q cos(v_c, t_q)` (best-matching
ImageNet class name). Measure the fraction of `c`'s top-K images that are
of class `q*`.

| Model / view                       | mean top-K frac labelled `q*` | % comps with ≥1 top-K image labelled `q*` |
|------------------------------------|:---:|:---:|
| ResNet-50 layer4                    | **0.203** | **43.8 %** |
| ViT-B/16 CLS                        | 0.034 | 13.0 % |
| ViT-B/16 mean-patch                 | 0.052 | 16.9 % |

**Reading.** For ~44 % of ResNet-50 last-stage channels, picking the closest
ImageNet class name in CLIP text space produces a label that actually appears
in the channel's reference images. Even at the strict "20 % of top-K images
match" bar, this only makes sense if `v_c` really is co-located with class
prompts in the shared space. Chance is 0.9 % (1/1000 classes match a random
image).

### 2c. Qualitative case study

Full table in `outputs/qualitative_examples.md`. Selected highlights (all
ResNet-50 layer4):

| Comp | Text label picked from 1000 ImageNet classes | Top-K image labels |
|---:|---|---|
|  33 | coucal        | coucal ×8, coffee mug ×1 (purity 0.89) |
| 619 | limpkin       | limpkin ×7, strainer ×1, medicine chest ×1 |
| 1743| banjo         | banjo ×8, drum ×1 |
| 1362| baseball      | baseball ×6, cliff ×1, toilet seat ×1 |
| 1726| dishwasher    | dishwasher ×5, plate rack ×1, lemon ×1 |
| 358 | vestment      | vestment ×5, tusker ×1, prayer rug ×1 |

And, going in the reverse direction, text queries against `{v_c}` retrieve
components whose exemplars visibly encode the queried concept:

| Query          | Retrieved top-1 comp | Its top-K image labels |
|---|---:|---|
| "goldfish"     | 1092 | goldfish ×4, carousel, orange |
| "mushroom"     |  206 | bolete ×6, agaric, mushroom |
| "green field"  |  690 | rapeseed ×9 (all yellow-flower fields) |
| "beach"        |  160 | leatherback turtle ×2, seashore ×2, promontory |

The "green field" ↔ rapeseed match is a striking example: CLIP treats the
yellow-flowered rapeseed field as the visual archetype of a "green field", and
the ResNet channel whose exemplars are 9 rapeseed images is retrieved
top-1 by that query — evidence that `v_c` really is being read *as text*.

## Verify stage

The ViT-B/16 rows in both tables above **are** the verify stage: the pipeline
was reused unchanged on a transformer, and both claims still hold
(Claim 1: coherence & purity clearly above chance; Claim 2: 80× top-1 lift on
text→component retrieval). Effect sizes are smaller than for the CNN because
ViT residual-stream neurons in a supervised ImageNet checkpoint are known to
carry more distributed information; the qualitative direction is the same.

## Conclusion

Both claims are supported by the experiments:

1. Per-component top-K activating images form a semantically coherent,
   class-concentrated set for the majority of components in both ResNet-50
   and ViT-B/16.
2. Mean-pooling CLIP embeddings of those images produces a vector `v_c` that
   lives in CLIP's joint image–text space: a single class-name text prompt
   retrieves the correct-concept component with 500× lift for ResNet-50
   channels (80× for ViT neurons), and picking the nearest class name is a
   sensible text description for a large fraction of components.

Artifacts:
- `outputs/features/eval10k.npz` — raw activations + CLIP image embeddings
- `outputs/components/*.npz` — top-K exemplar indices, labels, and `v_c`
- `outputs/claim1_results.json`, `outputs/claim2_results.json` — quantitative
- `outputs/qualitative_examples.md` — 40 sampled components per model + 15
  text-query case studies
- `logs/extract.log`, `logs/claim1.log`, `logs/claim2.log` — run logs
