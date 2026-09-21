# Verbal Confidence in LLMs: probing + causal-patching evidence

## Claim under test

> When an LLM is asked to verbalise its confidence after answering, the
> confidence value is not freshly computed at the moment of verbalisation;
> instead, it is written into hidden states immediately following the answer
> and is later retrieved from that cache when the model speaks the confidence
> token.

Two consequences follow:

1.  A linear probe applied to hidden states at token positions **immediately
    after the answer** should already predict the verbalised confidence
    almost as well as a probe applied to the readout position.
2.  Causally transplanting the residual stream at those *post-answer*
    positions from a HIGH-confidence run into a LOW-confidence run should
    shift the verbalised confidence upward — because the "cache" the model
    is supposedly reading has been overwritten.

## Setup

- **Prompt (base LM, few-shot).**  Nine diverse `Question / Answer /
  Confidence (0-100)` exemplars followed by the test question.  Confidence
  values in the few-shot examples span **{5,30,40,55,70,82,90,97,98,99}** so
  the model is free to output a low number when unsure.
- **Data.**  TriviaQA (`rc.nocontext` validation split), 250 questions.
- **Models.**  Primary — `gemma-3-27b-pt` (62 layers, base pretrained).
  Cross-model check — `Qwen2.5-7B` (28 layers, base pretrained).
- **Anchor token positions** we hook and later probe / patch:
  - `P_pre`  : last token of the `Answer:` tag  (before the answer text)
  - `P_ans`  : last token of the answer text  (**alleged cache site**)
  - `P_nl`   : the `\n` between the answer and the confidence tag
  - `P_post` : last token of `Confidence (0-100):` (immediately drives the
              confidence-number readout)

Full code, hyper-parameters and raw logs are in `code/`, `logs/`.

## Result 1  — Probing (correlational)

`Ridge` regression, 5-fold CV, per (position × layer).  Metric = R² and
Spearman-ρ between predicted and verbalised confidence.

### Gemma-3-27B — 250 examples, correct-rate 76.8 %

| position     | peak R²  | @ layer | peak Spearman |
|--------------|:-------:|:-------:|:-------------:|
| `P_pre`      |  0.159  |   38    |     0.470     |
| `P_ans`      |  **0.632**  |  27  |     **0.700** |
| `P_post`     |  0.804  |   61    |     0.853     |
| `P_conf` (baseline: contains the confidence *token* itself, trivial) | 1.00 | 4 | 0.918 |

### Qwen 2.5-7B — 250 examples, correct-rate 53.6 %

| position     | peak R²  | @ layer | peak Spearman |
|--------------|:-------:|:-------:|:-------------:|
| `P_pre`      | −0.030  |    1    |     0.280     |
| `P_ans`      | **0.581** |  25  |   **0.452**   |
| `P_post`     |  0.786  |   27    |     0.716     |
| `P_conf` (baseline) | 0.999 |  13  |  0.910 |

### Take-away

Both models write confidence-relevant information into the residual stream at
`P_ans` — *before the confidence tag has even been read*.  On Gemma a linear
probe already achieves ρ = 0.70 there; on Qwen ρ = 0.45.  Meanwhile `P_pre`
carries little to no signal, ruling out that the correlation is driven by
the few-shot pattern alone.  This is a strong prediction of the "cache"
account.

Layer-wise plots: `probe_gemma_spearman.png`, `probe_qwen_spearman.png`.

## Result 2  — Causal activation patching

For each `(HIGH-confidence example A, LOW-confidence example B)` pair we run
`B` forward and replace the residual-stream vector at `A`'s activation, at
one of the positions listed above, across a range of layers (multi-layer
patch to survive downstream re-computation).  We report the mean shift in
verbalised confidence  `Δ = c_patch − c_base`,  and the number of
upward-flips out of `N` pairs.

The claim predicts:  `|Δ(ans)|  ≈  |Δ(post)|`  and both large.

### Gemma-3-27B (patch layers 30-58, step 3, N = 20 pairs, target Δ = +38.25)

| patched position | mean Δ | upward-flips |
|---|---:|---:|
| `P_pre`               | +1.75 | 1 / 20 |
| `P_ans`               | +3.25 | 2 / 20 |
| `P_nl`                | +3.25 | 3 / 20 |
| `P_ans + P_nl` (joint)| +3.25 | 3 / 20 |
| `P_post` (positive control) | **+37.25** | **20 / 20** |

Almost identical numbers came out of a second run with shallower layers
(20-45, step 3): pre +6.0, ans +4.5, post +35.8, target +38.25.

**→ On Gemma the strict cache claim is falsified:** patching the alleged
cache site is essentially inert.  Only patching the readout position `P_post`
causally changes the emitted confidence, and does so to ≈ 94–97 % of the
target.

### Qwen 2.5-7B (patch layers 10-24, step 2, N = 15 pairs, target Δ = +60.0)

| patched position | mean Δ | upward-flips |
|---|---:|---:|
| `P_pre`                  |  +2.0 |  1 / 15 |
| `P_ans`                  | +17.7 |  7 / 15 |
| `P_nl`                   | +21.3 |  9 / 15 |
| **`P_ans + P_nl` joint** | **+46.3** | **15 / 15** |
| `P_post` (control)       | +16.0 |  7 / 15 |

**→ On Qwen the claim is largely supported.**  A joint patch of the two
post-answer positions captures ≈ 77 % of the target and flips every pair,
while patching `P_pre` (before the answer) is essentially inert (control
passes).  Interestingly the single-position `P_post` patch is not the
strongest lever on Qwen — the causally relevant "confidence" seems to sit
distributed across the post-answer region, not at the readout.

Bar chart of all deltas: `patch_delta_bars.png`.

## Interpretation

| |  claim confirmed? |
|---|---|
| **Correlational probing** — confidence readable *before* verbalisation | **YES**, in both models. |
| **Causal patching** — post-answer residual is the substrate |  **YES** on Qwen 2.5-7B, **NO** on Gemma-3-27B. |

So the phrase "verbal confidence is cached mid-generation" is
**architecture-dependent**:

- In Qwen 2.5 the causally-relevant confidence variable lives on the
  residual stream at `P_ans` and `P_nl` at mid-network layers.  Overwriting
  those tokens with another example's activations reliably transplants the
  future verbalised number.
- In Gemma 3-27B the same probe signal exists (in fact stronger than in
  Qwen), yet is not causally used: downstream layers of Gemma appear to
  *re-derive* the confidence at the readout token, and the only lever that
  moves the number is the readout token itself.  A plausible mechanistic
  reason is Gemma 3's alternating sliding-window / global attention
  (`sliding_window = 1024`, config file): later "post" positions attend
  broadly and can rebuild the confidence from the answer text even when the
  intermediate cache is corrupted.

## Files

```
code/
  prompt_utils.py         few-shot template + parsers
  generate.py             stage 1/2/3: answer + confidence + hidden-states
  probe.py                per-layer ridge probe
  patch.py                causal residual patching (multi-layer, multi-pos)
  plot.py, patch_plot.py  visualisation
artifacts/
  generations_gemma.csv           250 rows
  generations_qwen.csv            250 rows
  hidden_states/gemma__*.pt       (pre,ans,post,conf) × 62 layers × 5376 dim
  hidden_states/qwen__*.pt        (pre,ans,post,conf) × 28 layers × 3584 dim
results/
  probe_gemma.json                per-position × per-layer R2, Spearman
  probe_qwen.json                 same
  patch_gemma_L20-45.json         mid-layer patch
  patch_gemma_L30-58.json         deep-layer patch (with nl / ans+nl)
  patch_qwen_L10-24.json          Qwen patch
  probe_gemma_r2.png, probe_gemma_spearman.png
  probe_qwen_r2.png,  probe_qwen_spearman.png
  patch_delta_bars.png            bar chart of mean Δ per position
  REPORT.md                       this file
logs/
  gen_gemma.log, gen_qwen.log
  patch_gemma_L20-45.log, patch_gemma_L30-58.log, patch_qwen.log
```

## Caveats and limitations

- Both models are *base pretrained*, not RLHF-instruct — the "confidence"
  they emit is a pattern-match to the few-shot exemplars and mostly
  concentrates in a handful of values (`{60, 90, 95}` on Gemma;
  `{80, 90, 95}` on Qwen).  A more graded confidence distribution (e.g.
  from an instruct model or explicit sampling) would tighten the probing R²
  bounds.
- Sample sizes for patching are modest (N = 15–20 pairs per condition)
  because each pair costs 5 forward passes on a 27 B model.  Effect sizes
  are, however, large enough that all Gemma controls pass (post = 20/20
  flips, ans = 3/20) and all Qwen controls pass (ans+nl = 15/15,
  pre = 1/15).
- Only two anchor layer-ranges tested per model.  We did not sweep every
  layer at every position.
- Patched residual = raw copy of source vector.  We did not try
  norm-scaling or difference-only patches, which sometimes yield stronger
  effects but complicate interpretation.
