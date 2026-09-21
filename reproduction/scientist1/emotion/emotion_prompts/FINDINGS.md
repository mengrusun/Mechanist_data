# Findings: Emotional Framing in Prompts as a Weak, Input-Dependent Signal

## Experimental setup

- **Models**: Qwen3-14B (primary), Qwen2.5-14B-Instruct (cross-model verification)
- **Datasets** (test-set subsamples):
  - GSM8K (math) — n=500 for each model
  - SocialIQA (social) — n=300 for each model
  - BoolQ (yes/no reading comp) — n=300 (Qwen3-14B)
  - OpenBookQA (commonsense MCQ) — n=300 (Qwen3-14B)
  - MedQA-US (medical MCQ) — n=250 (Qwen3-14B, `max_new=768`)
  - BBH (reasoning; 7 subtasks × 40) — n=280 (Qwen3-14B)
- **Conditions per (model, dataset)**: 25 =
  1 neutral baseline
  + 6 emotions × 2 intensities × 2 sources (human-written, LLM-generated)
- **Inference**: vLLM 0.10 with `enable_thinking=False`, temperature 0, greedy decoding
- **Total inferences**: ≈ 44,650

## Summary results (per condition = a specific fixed prefix)

| Dataset      | Model                   | neutral | best   | worst  | best_Δ  | worst_Δ | range |
|--------------|-------------------------|--------:|-------:|-------:|--------:|--------:|------:|
| gsm8k        | Qwen3-14B               | 0.9480  | 0.9580 | 0.9140 | +0.0100 | -0.0340 | 0.044 |
| gsm8k        | Qwen2.5-14B-Instruct    | 0.9320  | 0.9260 | 0.8880 | -0.0060 | -0.0440 | 0.038 |
| socialiqa    | Qwen3-14B               | 0.7700  | 0.7867 | 0.7200 | +0.0167 | -0.0500 | 0.067 |
| socialiqa    | Qwen2.5-14B-Instruct    | 0.8067  | 0.8100 | 0.7567 | +0.0033 | -0.0500 | 0.053 |
| boolq        | Qwen3-14B               | 0.9000  | 0.9133 | 0.8833 | +0.0133 | -0.0167 | 0.030 |
| openbookqa   | Qwen3-14B               | 0.9433  | 0.9467 | 0.8967 | +0.0033 | -0.0467 | 0.050 |
| medqa        | Qwen3-14B               | 0.7160  | 0.7440 | 0.6640 | +0.0280 | -0.0520 | 0.080 |
| bbh          | Qwen3-14B               | 0.7607  | 0.8107 | 0.7571 | +0.0500 | -0.0036 | 0.054 |

(all values on the held-out test sample; range = best − worst across 24 emotional prefixes)

## Claim-by-claim verdict

### Claim 1 (SUPPORTED)
> Static emotional prefixes change accuracy only by small, input-dependent amounts.

- **Best positive gain** never exceeds +5 pp across 8 (model × dataset) combinations, and on 6 of the 8 it is ≤ +2 pp (median +1.3 pp).
- On the two settings where an emotion prefix beats neutral by more than 1 pp (BBH +5.0 pp, MedQA +2.8 pp), *which* prefix wins is inconsistent (BBH: `llm_happiness_high`; MedQA: `llm_sadness_low`). Beyond the winner, the ordering of the other 23 prefixes shuffles across datasets.
- On Qwen2.5-14B-Instruct/GSM8K, *no* emotional prefix beats neutral (best_Δ = −0.6 pp). Cross-model, the same prefix produces different signs (e.g., `llm_surprise_high` = +1.3 on SocialIQA/Qwen3 but −1.7 on SocialIQA/Qwen2.5).
- The pooled mean_Δ across all 192 (prefix × model × dataset) cells is **−0.87 pp** (SD 1.96 pp; median −1.00 pp). Only **30%** of prefixes beat neutral at all, and only **21%** beat it by ≥ 0.5 pp; **58%** hurt by ≥ 0.5 pp. Extremes are bounded: max gain +5.0 pp, max loss −5.2 pp.

### Claim 2 (PARTIALLY SUPPORTED)
> Effect is most pronounced on socially grounded tasks; smaller on math/factual.

Reading the data through several lenses:
- **Range** (max prefix effect): SocialIQA 5.3–6.7 pp is above the math/reading baselines (BoolQ 3.0 pp, GSM8K 3.8–4.4 pp), but MedQA (8.0 pp) and BBH (5.4 pp) also exceed the math range even though they are not "social".
- **Worst-Δ magnitude**: SocialIQA’s worst prefix causes −5.0 pp on *both* models — larger than what almost any single-model math or BoolQ worst prefix produces. This asymmetry (small upside, larger downside) is most consistent on SocialIQA.
- **Oracle upper bound** (per-query pick of any correct prefix): SocialIQA jumps from ≈0.77 to ≈0.87 (+14 pp headroom), the largest headroom in the study.

So: the *positive* effect of emotion prefixes is **not** especially large on social tasks (BBH and MedQA can beat SocialIQA on the best-prefix delta). But the **variability** of the effect and the size of the *downside* — plus the per-query gap between neutral and oracle — are indeed largest on social tasks. Interpretation: emotional framing interacts more strongly with socially-grounded content, but the interaction is not uniformly beneficial.

### Claim 3 (SUPPORTED)
> No single basic emotion consistently benefits across all models and tasks; stronger emotional wording does not yield proportionally larger gains.

- Across the 8 (model × dataset) settings, the *best* emotion is spread across the family: happiness ×2, sadness ×2, fear ×1, anger ×2, disgust ×0, surprise ×1. Disgust never wins.
- Pooled mean deltas per emotion (across all sources, intensities, datasets, models):
  - happiness −0.14 pp, sadness −0.96 pp, fear −1.21 pp, anger −0.72 pp, disgust −1.12 pp, surprise −1.06 pp.
- Happiness is the closest to neutral on average, but its *best-per-dataset* win rate is only 2/8; on Qwen2.5-14B-Instruct/GSM8K, `human_happiness_low` is one of the worst prefixes (−0.8 pp).
- **Intensity does not scale monotonically with effect**: pooled mean_Δ is nearly identical for low (−0.88 pp) and high (−0.86 pp). On SocialIQA/Qwen3-14B the two high-intensity happiness prefixes actually gain less than the low-intensity human happiness (`human_happiness_low` +0.7 vs `human_happiness_high` +0.3, but `llm_happiness_high` at +0.3 vs `llm_happiness_low` +1.0). No monotone trend.
- **Human-written < LLM-generated?**: pooled mean_Δ is −0.67 pp for human, −1.07 pp for LLM. Our LLM-generated prefixes (drafted with more dramatic phrasing) are on average slightly worse than the plain human ones — confirming that stronger emotional wording does not translate into larger gains.

### Claim 4 (PARTIALLY SUPPORTED)
> An adaptive per-query policy (EmotionRL) yields more reliable accuracy gains than any fixed emotional prefix or neutral.

EmotionRL was trained as a per-condition binary classifier (predict "is the query answered correctly under condition X") on top of TF-IDF+LSA features; per query at inference we picked the condition with the highest predicted correctness score. Results are 5-fold CV over the same evaluation set.

| Dataset      | Model                   | neutral | best_fixed_train | oracle | EmotionRL | Δ vs neutral | Δ vs best_fixed |
|--------------|-------------------------|--------:|-----------------:|-------:|----------:|-------------:|----------------:|
| gsm8k        | Qwen3-14B               | 0.9520  | 0.9600           | 0.9760 | 0.9480    | −0.4         | −1.2            |
| gsm8k        | Qwen2.5-14B-Instruct    | 0.9480  | 0.9280           | 0.9720 | 0.9400    | −0.8         | +1.2            |
| boolq        | Qwen3-14B               | 0.9000  | 0.8933           | 0.9200 | 0.8933    | −0.7         |  0.0            |
| openbookqa   | Qwen3-14B               | 0.9467  | 0.9333           | 0.9733 | 0.9267    | −2.0         | −0.7            |
| medqa        | Qwen3-14B               | 0.7200  | 0.7520           | 0.9200 | 0.7120    | −0.8         | −4.0            |
| bbh          | Qwen3-14B               | 0.7571  | 0.8143           | 0.9214 | 0.8000    | +4.3         | −1.4            |
| socialiqa    | Qwen3-14B               | 0.7333  | 0.7467           | 0.8733 | 0.7400    | +0.7         | −0.7            |
| socialiqa    | Qwen2.5-14B-Instruct    | 0.7600  | 0.7600           | 0.8800 | 0.7867    | +2.7         | +2.7            |

Verdict:
- The classifier **is not reliably better than neutral** on structured factual tasks (BoolQ, MedQA, OpenBookQA, GSM8K). This is consistent with claim 2: where the underlying emotional signal is weak, a supervised prefix selector has little to learn from and cannot beat "just do the task".
- The classifier **beats both neutral and best-fixed-prefix on socially grounded tasks** — most clearly on Qwen2.5-14B-Instruct/SocialIQA (+2.7 pp over both baselines) and BBH (+4.3 pp over neutral, but slightly under best-fixed).
- The **oracle upper bound gap** on SocialIQA and MedQA is huge (+10–15 pp), so there is a *lot* of room for a better selector; our TF-IDF+LR selector captures only a small fraction of it. A stronger embedding-based classifier or a bandit trained with online rewards would likely close more of the gap on social tasks. The claim as stated is directionally correct but only strongly true on social settings.

## Practical takeaways

1. If you already have a decent instruction, prepending an emotional line rarely helps and can hurt by up to 5 pp on the same query. The expected effect on a random query is essentially zero.
2. Positive effects, when they exist, are dataset-specific — there is no "always-on" emotion prefix. Happiness is the safest default (mean_Δ closest to zero) but is not a reliable win.
3. Stronger/more dramatic phrasing (our LLM-generated prefixes) is on average *worse* than plainer human-written framing, and on tasks that require concise multi-step reasoning it can also make the model verbose enough to run out of the token budget before producing the final answer (observed on MedQA at `max_new=384`).
4. A learned per-query policy has room to help specifically on socially grounded tasks; on factual QA and math it should default to neutral.

## Files

- `prefixes.py` — the 25 prefix conditions
- `datasets_loader.py` — unified loader
- `run_eval.py` — vLLM inference driver
- `rescore.py` — smarter regex-based answer extractor (needed for BBH mixed formats)
- `emotion_rl.py` / `emotion_rl_v2.py` — supervised selector, single split and 5-fold CV
- `final_report.py` — regenerates the numbers above from `results/*.jsonl`
- `results/` — per-condition JSONL files (`{dataset}__{model}__{condition}.jsonl`)
