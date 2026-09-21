# Scoring and Aggregation Rules (LLM Judges / Human Judges)

This document defines how each evaluation JSON, containing scores for nine dimensions, is aggregated into comparable total scores and statistical tables.

## 1. Data Sources

- Each experiment (a leaf subfolder such as `belief/verbal_confidence`) has evaluation files for `scientist1`, `scientist2`, and `scientist3`.
- Files follow `{exp_dir}-scientist{N}-{judge}.json`, where `judge` is the scoring model or expert name.
- One experiment/scientist pair may have multiple scoring files.

## 2. Individual Scores

| Raw value | Treatment |
|---|---|
| Numeric 1/2/3/4/5 | Count the value directly |
| `0` | Count as 0; it represents a real failure, not an inapplicable item |
| `"n/a"` | Exclude from both numerator and denominator |
| `null`, empty, or unparseable | Treat as missing and exclude |

## 3. Dimensions 1–8

Each dimension has a maximum of 5 points. The applicable scores are summed as `Σ(dim1–8)`.

## 4. Dimension 9: Reproduction Fidelity

Dimension 9 is split by claim into three independent subdimensions: `method`,
`experiment`, and `result`.

1. Average each subdimension across all applicable claims to obtain
   `method_avg`, `experiment_avg`, and `result_avg`.
2. Skip claims scored `n/a`; if a subdimension is `n/a` for every claim, its
   average is empty and is omitted.
3. Set `dim9_final` to the average of the non-empty subdimension averages.

For example, if the three averages are 4.2, 2.6, and 3.6, then
`dim9_final = (4.2 + 2.6 + 3.6) / 3 = 3.47`.

## 5. Raw Total and Maximum Baseline

```text
raw = Σ(applicable dim1–8) + dim9_final
max_base = 5 × (number of applicable dimensions among dim1–8)
            + 5 if any dim9 subdimension is applicable, otherwise + 0
total score = raw / max_base × 100
```

`n/a` dimensions are removed from both numerator and denominator, so the
baseline decreases accordingly. With no `n/a` values, `max_base = 8×5 + 5 = 45`.

## 6. Multiple Judges or Experts

For an experiment/scientist pair with multiple scoring files, calculate all
metrics independently for each file, then average each metric across files.
The table's `n` column records the number of files contributing to the average.

## 7. Statistical Table Columns

Each row represents one experiment/scientist pair. Columns include category,
experiment, scientist, `n`, total score (0–100), raw total, maximum baseline,
`d1`–`d8`, the first-eight subtotal, `dim9_method`, `dim9_exp`,
`dim9_result`, and `dim9_final`.

The rules deliberately treat `n/a` as excluded and `score=0` as a real zero;
an experiment cannot gain credit by avoiding an applicable dimension.
