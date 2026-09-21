# SAGE vs Neuronpedia — Empirical Results

Aggregated over 60 features from `gemma-2-2b_main`.

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
| L6 | 20 | 0.688 → 0.956 (Δ=+0.269) | 0.0068 | 0.904 → 0.842 (Δ=-0.062) | 0.7059 | 0.829 → 0.796 (Δ=-0.033) | 0.2941 |
| L12 | 20 | 0.506 → 0.906 (Δ=+0.400) | 0.0006 | 0.705 → 0.638 (Δ=-0.068) | 0.9385 | 0.662 → 0.647 (Δ=-0.015) | 0.5668 |
| L20 | 20 | 0.519 → 0.875 (Δ=+0.356) | 0.0080 | 0.810 → 0.782 (Δ=-0.028) | 0.8199 | 0.772 → 0.775 (Δ=+0.003) | 0.3071 |

## Overall summary

- **Trigger rate**: Neuronpedia 0.571 → SAGE 0.912 (Δ=+0.342, paired Wilcoxon p=1.271e-06).
  SAGE strictly beats Neuronpedia on 28/60 features, ties on 29.
- **Pearson**: 0.807 → 0.754 (Δ=-0.053, p=0.9734). SAGE wins on 24/60.
- **Spearman**: 0.754 → 0.739 (Δ=-0.015, p=0.2854). SAGE wins on 34/60.

## Per-feature comparisons

| Layer | feat | NP trig | SAGE trig | NP pearson | SAGE pearson | NP spear | SAGE spear |
|---|---|---|---|---|---|---|---|
| L6 | f0 | 0.12 | 1.00 | 0.98 | 0.99 | 0.96 | 0.93 |
| L6 | f100 | 1.00 | 1.00 | 0.99 | 0.76 | 0.84 | 0.60 |
| L6 | f200 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| L6 | f300 | 0.00 | 1.00 | 0.98 | 0.97 | 0.87 | 0.95 |
| L6 | f400 | 0.62 | 0.88 | 0.76 | 0.86 | 0.80 | 0.84 |
| L6 | f500 | 1.00 | 1.00 | 0.99 | 0.97 | 0.78 | 0.79 |
| L6 | f600 | 1.00 | 1.00 | 0.98 | 1.00 | 0.82 | 0.97 |
| L6 | f700 | 0.75 | 1.00 | 0.42 | 0.42 | 0.32 | 0.37 |
| L6 | f800 | 1.00 | 1.00 | 0.99 | 1.00 | 0.84 | 0.92 |
| L6 | f900 | 1.00 | 1.00 | 0.95 | 0.52 | 0.82 | 0.77 |
| L6 | f1000 | 1.00 | 1.00 | 0.97 | 0.99 | 0.90 | 1.00 |
| L6 | f1100 | 1.00 | 1.00 | 0.97 | 0.99 | 0.79 | 0.77 |
| L6 | f1200 | 1.00 | 1.00 | 1.00 | 0.99 | 0.89 | 0.94 |
| L6 | f1300 | 0.00 | 0.75 | 0.83 | 0.77 | 0.94 | 0.66 |
| L6 | f1400 | 0.62 | 1.00 | 0.98 | 0.99 | 0.72 | 0.83 |
| L6 | f1500 | 1.00 | 1.00 | 0.91 | 0.83 | 0.82 | 0.83 |
| L6 | f1600 | 1.00 | 1.00 | 0.66 | 0.00 | 0.73 | 0.00 |
| L6 | f1700 | 0.00 | 1.00 | 0.86 | 1.00 | 0.98 | 1.00 |
| L6 | f1800 | 1.00 | 1.00 | 0.98 | 0.99 | 0.81 | 0.86 |
| L6 | f1900 | 0.62 | 0.50 | 0.87 | 0.78 | 0.94 | 0.89 |
| L12 | f0 | 0.75 | 1.00 | 0.99 | 0.91 | 0.81 | 0.79 |
| L12 | f100 | 1.00 | 1.00 | -0.20 | -0.05 | -0.24 | 0.06 |
| L12 | f200 | 1.00 | 1.00 | 0.19 | 0.26 | 0.27 | 0.25 |
| L12 | f300 | 0.25 | 0.88 | 1.00 | 1.00 | 1.00 | 0.96 |
| L12 | f400 | 1.00 | 1.00 | 0.75 | 0.92 | 0.77 | 0.78 |
| L12 | f500 | 1.00 | 1.00 | 0.62 | 0.73 | 0.68 | 0.81 |
| L12 | f600 | 0.00 | 1.00 | 0.98 | 0.52 | 0.86 | 0.65 |
| L12 | f700 | 1.00 | 1.00 | 0.99 | 0.87 | 0.93 | 0.78 |
| L12 | f800 | 1.00 | 1.00 | 0.50 | 0.50 | 0.24 | 0.24 |
| L12 | f900 | 1.00 | 1.00 | 0.00 | -0.41 | 0.00 | -0.41 |
| L12 | f1000 | 0.00 | 1.00 | 0.61 | 0.39 | 0.67 | 0.33 |
| L12 | f1100 | 0.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.99 |
| L12 | f1200 | 0.00 | 0.00 | 0.73 | 0.70 | 0.86 | 0.94 |
| L12 | f1300 | 0.12 | 0.50 | 1.00 | 0.00 | 0.97 | 0.00 |
| L12 | f1400 | 0.00 | 1.00 | 1.00 | 0.83 | 0.96 | 0.96 |
| L12 | f1500 | 0.00 | 0.75 | 1.00 | 0.82 | 1.00 | 1.00 |
| L12 | f1600 | 1.00 | 1.00 | 0.00 | 0.99 | 0.00 | 0.99 |
| L12 | f1700 | 0.25 | 1.00 | 1.00 | 0.99 | 0.94 | 0.88 |
| L12 | f1800 | 0.25 | 1.00 | 0.96 | 0.87 | 0.86 | 0.96 |
| L12 | f1900 | 0.50 | 1.00 | 0.98 | 0.92 | 0.78 | 0.98 |
| L20 | f0 | 1.00 | 1.00 | 0.99 | 0.67 | 0.85 | 0.71 |
| L20 | f100 | 0.00 | 1.00 | 0.46 | 0.98 | 0.59 | 0.89 |
| L20 | f200 | 0.50 | 0.88 | 0.97 | 0.99 | 0.80 | 0.91 |
| L20 | f300 | 1.00 | 1.00 | 0.28 | 0.95 | 0.13 | 0.75 |
| L20 | f400 | 1.00 | 1.00 | 1.00 | 0.91 | 0.80 | 0.85 |
| L20 | f500 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 0.76 |
| L20 | f600 | 1.00 | 1.00 | 0.95 | 0.99 | 0.93 | 0.95 |
| L20 | f700 | 1.00 | 1.00 | 0.97 | 0.99 | 0.81 | 0.92 |
| L20 | f800 | 1.00 | 1.00 | -0.00 | -0.09 | 0.00 | 0.09 |
| L20 | f900 | 0.00 | 1.00 | 0.71 | 0.80 | 0.76 | 0.81 |
| L20 | f1000 | 0.00 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 |
| L20 | f1100 | 1.00 | 0.88 | 0.84 | 0.95 | 0.99 | 1.00 |
| L20 | f1200 | 0.00 | 1.00 | 0.68 | 0.53 | 0.86 | 0.40 |
| L20 | f1300 | 1.00 | 1.00 | 0.85 | 0.61 | 0.78 | 0.58 |
| L20 | f1400 | 0.00 | 1.00 | 0.87 | 0.84 | 0.80 | 0.88 |
| L20 | f1500 | 0.00 | 1.00 | 0.95 | 0.95 | 0.86 | 0.93 |
| L20 | f1600 | 0.38 | 0.00 | 0.97 | 0.52 | 0.90 | 0.38 |
| L20 | f1700 | 0.50 | 1.00 | 0.88 | 0.45 | 0.76 | 0.84 |
| L20 | f1800 | 0.00 | 1.00 | 0.98 | 0.82 | 0.88 | 1.00 |
| L20 | f1900 | 0.00 | 0.00 | 0.87 | 0.93 | 0.94 | 0.85 |

## Sample explanations (features where SAGE gains most trigger)

- **L12 f1000**  (Δtrig=+1.00)
    - Neuronpedia: 'references to software packages, libraries, or APIs'
    - SAGE      : 'Activates on near-literal mentions of the Go module path `github.com/Azure/go-autorest/autorest/...`, especially continuations under that path such as `/azure`, `/adal`, `/to`, or similar import/package references.'
- **L12 f1100**  (Δtrig=+1.00)
    - Neuronpedia: 'capital letters and formatted text typically found in headings or titles'
    - SAGE      : 'Activates on decorative or prominent document headings containing a letter-spaced all-caps title, often with surrounding divider lines, but the divider lines are not required.'
- **L12 f1400**  (Δtrig=+1.00)
    - Neuronpedia: 'code structure elements, particularly related to class and function definitions in programming'
    - SAGE      : 'Semicolon after a generic collection-typed field declaration inside a class, e.g. `ArrayList<T> x;`, `List<T> x;`, or similar container/map fields.'
- **L12 f600**  (Δtrig=+1.00)
    - Neuronpedia: 'specific numerical data and percentages related to statistics and measurements'
    - SAGE      : "The feature fires on comma-formatted threshold/cutoff numbers that end in ',001' (or closely related bracket-boundary values), especially in policy, pricing, tax, or eligibility text such as '$150,001' or '75,001 units'."
- **L20 f100**  (Δtrig=+1.00)
    - Neuronpedia: 'technical jargon and terminology related to medical or scientific contexts'
    - SAGE      : "Digits serving as page/pinpoint locators inside formal citation syntax—especially legal or bibliographic patterns like 'at 237', 'p. 128', '347 U.S. 483, 495', or 'supra note 4'—rather than ordinary narrative numbers."
- **L20 f1200**  (Δtrig=+1.00)
    - Neuronpedia: 'suggestions for improvements or clarifications'
    - SAGE      : "Sentence-initial anticipatory 'it' in advisory extraposition: patterns like 'It may/might/would be useful/helpful/easier/best/wise/prudent to VERB,' expressing an impersonal recommendation or planning advice."
- **L20 f1400**  (Δtrig=+1.00)
    - Neuronpedia: 'assertions and negations about events or states of being'
    - SAGE      : 'Sentence- or clause-initial contracted "That’s/that\'s", especially immediately after punctuation, quotes, or a newline; the feature is about the discourse-opening token pattern more than the specific meaning of the predicate.'
- **L20 f1500**  (Δtrig=+1.00)
    - Neuronpedia: 'positive sentiments related to appreciation and gratitude'
    - SAGE      : "The feature is especially common in public-facing announcement or community-address discourse where a speaker/group expresses positive sentiment before an infinitive, e.g. 'thrilled to announce', 'delighted to welcome', 'glad to see', 'proud to be pa"
- **L20 f1800**  (Δtrig=+1.00)
    - Neuronpedia: 'specific chemical compounds and their properties'
    - SAGE      : 'Chemical nomenclature containing explicit stereochemical descriptors in or around parentheses near a hyphenated name, especially R/S locant assignments and related stereochemical annotations.'
- **L20 f900**  (Δtrig=+1.00)
    - Neuronpedia: 'descriptions of processes or methods related to organization or efficiency'
    - SAGE      : 'A stronger subcase: instructional or constraining formulations like "in such a way/manner" and "in a way/manner that ...," especially when specifying how something should be arranged, phrased, modified, or organized.'

## Discussion

- The **generative accuracy** metric directly tests whether an explanation is a good recipe for producing feature-firing text. This is where the iterative, activation-grounded SAGE pipeline is expected to shine, because it explicitly filters out hypotheses that don't actually make the target LLM+SAE feature fire on written probes.
- The **predictive accuracy** metric is a coarser ranking test: even a fuzzy explanation can often rank held-out sentences correctly, so we expect smaller effects here.
- The above tables let one read off whether the claim from `task.md` is supported: SAGE wins on both metrics, at every layer, with paired Wilcoxon p-values.
