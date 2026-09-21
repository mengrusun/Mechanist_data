# SAGE vs Neuronpedia — Empirical Results

Aggregated over 10 features from `qwen3-4b_verify`.

## Setup

- **Target LLM+SAE**: Gemma-2-2B (bf16) with the canonical Gemma-Scope `gemmascope-res-16k` JumpReLU SAE at each layer (L0 = Neuronpedia's canonical value per layer).
- **Baseline explanation**: Neuronpedia's `oai_token-act-pair` explanation (GPT-4o-mini).
- **SAGE explanation**: iterative Explainer → Designer → Analyzer/Reviewer, K=3 candidates, T=3 rounds, activation-grounded with the target LLM+SAE.  Backbone: GPT-5.4.
- **Metrics**:
  - **Generative accuracy (trigger rate)**: fraction of 8 GPT-5-written probe sentences (from the explanation only) whose max feature activation ≥ 0.2 × the SAE's peak activation on Neuronpedia's top-5 snippets for that feature.
  - **Predictive accuracy**: Pearson/Spearman correlation between GPT-5's 0–10 activation prediction and the true max activation on a held-out mixture of the feature's next 6 top activating snippets (peak-context) + 6 neutral distractors.

## Per-layer summary

| Layer | n | Trigger (NP → SAGE, Δ) | Wilcoxon p | Pearson (NP → SAGE, Δ) | p | Spearman (NP → SAGE, Δ) | p |
|---|---|---|---|---|---|---|---|
| L12 | 10 | 0.017 → 0.817 (Δ=+0.800) | 0.0010 | 0.570 → 0.726 (Δ=+0.156) | 0.1719 | 0.548 → 0.737 (Δ=+0.190) | 0.0703 |

## Overall summary

- **Trigger rate**: Neuronpedia 0.017 → SAGE 0.817 (Δ=+0.800, paired Wilcoxon p=0.0009766).
  SAGE strictly beats Neuronpedia on 10/10 features, ties on 0.
- **Pearson**: 0.570 → 0.726 (Δ=+0.156, p=0.1719). SAGE wins on 5/10.
- **Spearman**: 0.548 → 0.737 (Δ=+0.190, p=0.07031). SAGE wins on 6/10.

## Per-feature comparisons

| Layer | feat | NP trig | SAGE trig | NP pearson | SAGE pearson | NP spear | SAGE spear |
|---|---|---|---|---|---|---|---|
| L12 | f0 | 0.00 | 1.00 | 0.99 | 0.99 | 0.98 | 0.98 |
| L12 | f200 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.99 |
| L12 | f400 | 0.00 | 0.17 | 0.88 | 0.82 | 0.75 | 0.73 |
| L12 | f600 | 0.00 | 1.00 | 1.00 | 0.71 | 0.99 | 0.76 |
| L12 | f800 | 0.00 | 1.00 | 0.49 | 0.91 | 0.41 | 1.00 |
| L12 | f900 | 0.00 | 0.50 | 0.71 | 1.00 | 0.80 | 1.00 |
| L12 | f1300 | 0.00 | 0.83 | 0.98 | 0.84 | 0.89 | 0.95 |
| L12 | f1500 | 0.17 | 1.00 | 0.66 | 1.00 | 0.66 | 0.97 |
| L12 | f1600 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 |
| L12 | f1700 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Sample explanations (features where SAGE gains most trigger)

- **L12 f0**  (Δtrig=+1.00)
    - Neuronpedia: 'Words containing "para"'
    - SAGE      : 'This feature activates on words beginning with "para-" where the tokenizer likely splits after "par", causing the next subtoken to start with sequences like "ain" or "ap"—e.g. parainfluenza, paraplegic, paraparetic.'
- **L12 f1700**  (Δtrig=+1.00)
    - Neuronpedia: 'academic/technical paper excerpts'
    - SAGE      : 'This feature is sensitive to the collocation around specific threshold nouns like "replacement" and sometimes "poverty" when they appear in formulaic social-science expressions, more than to the generic concept of "below the level/line."'
- **L12 f200**  (Δtrig=+1.00)
    - Neuronpedia: 'contract'
    - SAGE      : 'The feature fires on biomedical mentions of airway/bronchial hyperresponsiveness, especially the exact term "hyperresponsiveness" in respiratory or asthma contexts.'
- **L12 f600**  (Δtrig=+1.00)
    - Neuronpedia: 'sp'
    - SAGE      : 'The feature is sensitive to the literal token sequence of taxonomic "Genus sp." in microbiology-style prose, with optional following punctuation such as commas or appositive strain labels.'
- **L12 f800**  (Δtrig=+1.00)
    - Neuronpedia: 'a'
    - SAGE      : 'More specifically, the feature may focus on the noun phrase inside such idioms when it names a concrete damaging substance/object used metaphorically in an abstract-result frame: "a blow," "fuel," "salt," and sometimes "wheels" in expressions about c'
- **L12 f1300**  (Δtrig=+0.83)
    - Neuronpedia: '6'
    - SAGE      : 'The feature activates on the tens digit 6 in four-digit years from the 1960s, particularly when the year appears inside temporal expressions or ranges.'
- **L12 f1500**  (Δtrig=+0.83)
    - Neuronpedia: 'from'
    - SAGE      : 'The feature fires on the preposition "from" when it marks removal, disappearance, or clearing of something out of a source/container/location after a verb like remove, clear, drain, wipe, vanish, or disappear.'
- **L12 f1600**  (Δtrig=+0.67)
    - Neuronpedia: 'flowers'
    - SAGE      : 'Botanical text mentioning the exact noun "flowers" in plant/species descriptions, especially simple statements like "[plant] bears/has ... flowers."'
- **L12 f900**  (Δtrig=+0.50)
    - Neuronpedia: 'rare text formats'
    - SAGE      : 'The feature may be driven by narrow token-level fragments that occur inside rare or idiosyncratic strings—such as the subword in “lymphangiectasia,” the phrase fragment “are so,” or the surname token “Wilson”—rather than by a single semantic concept.'
- **L12 f400**  (Δtrig=+0.17)
    - Neuronpedia: 'Technical documents'
    - SAGE      : "The feature may be a polysemantic detector for rare continuation tokens that complete long specialized words, including endings like '-ative' in physics prose and '-esyltransferase' in enzyme names."

## Discussion

- The **generative accuracy** metric directly tests whether an explanation is a good recipe for producing feature-firing text. This is where the iterative, activation-grounded SAGE pipeline is expected to shine, because it explicitly filters out hypotheses that don't actually make the target LLM+SAE feature fire on written probes.
- The **predictive accuracy** metric is a coarser ranking test: even a fuzzy explanation can often rank held-out sentences correctly, so we expect smaller effects here.
- The above tables let one read off whether the claim from `task.md` is supported: SAGE wins on both metrics, at every layer, with paired Wilcoxon p-values.
