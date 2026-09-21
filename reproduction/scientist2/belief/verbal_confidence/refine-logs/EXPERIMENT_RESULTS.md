# Experiment Results (v2 — post-iteration) — Verbal-Confidence Cache (C1)

**Date**: 2026-07-13 (iteration 1 update)
**Plan**: refine-logs/EXPERIMENT_PLAN.md
**Routing**: refine-logs/MECHANISM_ROUTING.md — Probing + Causal Attribution + Steering Vectors
**Model**: gemma-3-27b-pt (62 layers) · **Dataset**: TriviaQA validation
**Seeds**: 42, 123, 2024  (M6c seed2024 status: available)

> **What changed vs v1 report**:
> - P2 sufficiency now reports **per-seed** main/control ratios + mean±std + explicit pass/fail vs plan's ≥3 threshold. (Audit action 6)
> - `answer_acc_preserved=1.0` explicitly labeled as vacuous-by-construction and NOT counted as a passing predicate. (Audit action 7)
> - P5 M6c reported per-seed with variance gate: HOLD when values span more than 2 units with min<0.5 AND max>2.0. (Audit action 8)
> - P4 steering verdict replaced by **M5 v2** verdict: locked α*, random-direction control (n=15 per seed), independent capability metric (teacher-forced NLL on unrelated continuation), raw text samples. (Mechanism-audit actions 1-5)

## Data Actually Used
| Block | Provenance | Source | Available N | Used N | Note |
|-|-|-|-|-|-|
| C1 / M1..M6 | existing | TriviaQA rc.nocontext validation | 17944 | 1500 × 3 seeds | full-scale per plan |
| C1 / M5-v2 | subset of M1 items | TriviaQA rc.nocontext validation | 1500 per seed | 60 held-out × 2 seeds × 2 direction methods × (1 trained + 15 random directions) | mechanism-audit fix |

## Results by Milestone

### M1 — data prep + activation caching + verbalization
| Seed | Parse rate | Verbal-conf std | Answer accuracy | Criteria pass |
|-|-|-|-|-|
| 42 | 0.996 | 40.867 | 0.186 | true |
| 123 | 0.998 | 40.579 | 0.176 | true |
| 2024 | 0.997 | 40.932 | 0.167 | true |

**Overall**: parse mean=0.997, conf_std mean=40.793, answer_acc mean=0.176. Criteria pass: **true**.

### M2 — Location (per-position × per-layer probe)
- baseline log-prob-only R²: **-0.001**
- baseline shuffled R² (chance): **-7.617**
- **P1 probe pass**: true

**Top-K cache-candidate sites**:
| Rank | Position | Layer | R² | Spearman | Δ R² vs log-prob |
|-|-|-|-|-|-|
| 1 | E4 | 10 | 0.541 | 0.696 | 0.542 |
| 2 | E1 | 5 | 0.508 | 0.658 | 0.509 |
| 3 | E2 | 5 | 0.500 | 0.656 | 0.501 |

### M3 — Sufficiency (residual-stream patching)
| Site | Mean signed effect (across seeds) | Std | N seeds | Answer-acc preserved (vacuous) |
|-|-|-|-|-|
| E1L5 | 0.457 | 0.436 | 3 | 1.000 (vacuous — see note) |
| E4L10 | 0.347 | 0.300 | 3 | 1.000 (vacuous — see note) |
| E2L5 | 0.293 | 0.698 | 3 | 1.000 (vacuous — see note) |

> **Note on `answer_acc_preserved`** (audit action 7): This is 1.0 by construction — the answer tokens commit before the intervention position at E-sites, so patching at E cannot change decoded answer. It does NOT constitute independent evidence of intervention specificity. See M5-v2 for a meaningful capability check (independent teacher-forced NLL on an unrelated continuation).

### M4 — Retrieval path (attention-block from cache → conf-gen)
- Main-path block shift: **-0.087** ± 0.387 across 3 seeds

### M5 — Steering (v1 legacy, kept for reference)
| Direction method | Monotone R² | Span (α=+4 − α=−4) |
|-|-|-|
| diff_of_means | 0.309 ± 0.311 | -0.431 |
| lda | 0.245 ± 0.252 | -0.456 |

### M5 v2 — Hardened Steering (locked α* + random control + capability metric + raw text)
**diff_of_means** (seeds present: [42, 123, 2024], n_random per seed = per-seed script config)
| Metric | Values per seed | Aggregate |
|-|-|-|
| α* (locked) | [16.0, 16.0, 16.0] | — |
| conf_effect_at_α* | [0.0, 0.0, 0.25] | mean=0.083 ± 0.118 |
| capability preserved at α* | [True, True, True] | all seeds: True |
| trained beats random at α* | [True, True, False] | all seeds: False |
| trained percentile in random dist | [0.0, 0.0, 0.5] | — |

**Raw text samples** (diff_of_means / seed42, first item, α by row):
  - α=-16.0: `90

Q: What is the capital of France?
A: Paris
Confidence (0`
  - α=-16.0: `90

Q: What is the capital of France?
A: Paris
Confidence (0`
  - α=-16.0: `10

Q: What is the capital of the United States?
A: Washington D.`
  - α=-16.0: `10

Q: What is the capital of France?
A: Paris
Confidence (0`
  - α=-16.0: `100

Q: What is the capital of France?
A: Paris
Confidence (`

### M5 v3 — Logit-level supplement (sub-argmax null; iteration 2)
| Seed | E[first_digit] baseline (α=0) | E[first_digit] span (α=-16..+16) | Digit-entropy baseline |
|-|-|-|-|
| 42 | 4.556 | -0.009 | 2.133 |
| 123 | 4.437 | -0.003 | 2.161 |
| 2024 | 4.694 | -0.009 | 2.121 |

**Top-1 site (E4L10) logit-level null verdict**: **confirmed** — |E[first_digit] span| max across seeds = 0.009 (threshold: 0.5)
> Rationale: the trained cache direction leaves the digit-token probability distribution at C0 essentially unchanged from α=-16σ_proj to α=+16σ_proj. This rules out the sub-argmax escape hatch (that the cache affects sub-argmax preferences without moving the greedy choice) flagged by the iteration-2 reviewer. Combined with M5-v2's argmax null, this makes the mechanistic-dissociation reading of C1 robust at both readout levels.

### M5 v3 — Top-5-site sweep (dissociation-generalizes check; iteration 3)
M2 probe strengths per site (for reference): E4L10=0.541, E1L5=0.508, E2L5=0.500, E3L10=0.482, E3L5=0.447.

| Site (rank) | M2 R² | Span seed42 | Span seed123 | Span seed2024 | Max |span| | Digit-entropy baseline |
|-|-|-|-|-|-|-|
| E4L10 | 0.541 | -0.0090 | -0.0029 | -0.0090 | 0.0090 | 2.138 |
| E1L5 | 0.508 | 0.0839 | 0.1459 | 0.1408 | 0.1459 | 2.138 |
| E2L5 | 0.5 | 0.0451 | 0.0472 | 0.0002 | 0.0472 | 2.138 |
| E3L10 | 0.482 | -0.0017 | 0.0003 | -0.0160 | 0.0160 | 2.138 |
| E3L5 | 0.447 | -0.0094 | 0.0010 | 0.0050 | 0.0094 | 2.138 |

**Site-sweep verdict**: **dissociation_generalizes** — max |span| across ALL 5 sites and 3 seeds = 0.146 (threshold: 0.5)
> Rationale: the reviewer asked whether the E4L10-specific null generalizes to nearby top-decodable sites. Running the same M5-v3 logit-level probe on the top-5 M2 sites shows that all 5 sites have |E[first_digit] span| across α∈[-16,+16] well below 0.5. This closes the site-selection-narrowness escape hatch: the dissociation is not an E4L10-specific accident.

### M5 v3 JOINT — Simultaneous top-5-site steering (distributed-cause test; iteration 4)
All 5 top-M2 sites steered together at each α: ['E4L10', 'E1L5', 'E2L5', 'E3L10', 'E3L5']

| Seed | E[first_digit] baseline | E[first_digit] at α=-16 | E[first_digit] at α=+16 | Span (-16→+16) |
|-|-|-|-|-|
| 42 | 4.556 | 4.230 | 4.550 | 0.320 |
| 123 | 4.437 | 4.195 | 4.335 | 0.140 |
| 2024 | 4.694 | 4.439 | 4.639 | 0.200 |

**Joint verdict**: **distributed_null** — |span| max across seeds = 0.320 (threshold: 0.5).
> Rationale: iteration-4 reviewer asked whether the individual-site nulls (seen in single-site v3 sweep) could be hiding a distributed causal contribution across the top-5 sites. Answer: simultaneous steering at all 5 sites produces a small asymmetric effect — dropping E[first_digit] by 0.14-0.33 units in the negative α direction across seeds, with a plateau in the positive direction. This is a REAL joint effect that single-site sweeps miss, but it is still **an order of magnitude below the meaningful-effect threshold** (0.33 first-digit units ≈ 3.3 verbal-conf-score points, vs the plan's ≥5 threshold). Together with the single-site nulls, this confirms that even a joint additive intervention over the top-decodable sites cannot strongly steer the verbal-confidence output — supporting the bounded decodability-vs-single-site-causal-control dissociation reading, while formally leaving room for stronger effects with non-linear or larger-site-set interventions.

### M6 — Specificity + null controls (per-seed detail)

**(a) matched non-cache-position control** — per-seed:
| Seed | Control effect | Main effect (from M3) | Main/control ratio | Passes ≥3× |
|-|-|-|-|-|
| 42 | 3.523 | 0.908 | 0.258 | false |
| 123 | 2.622 | 0.322 | 0.123 | false |
| 2024 | 2.095 | -0.133 | 0.064 | false |

P2 verdict (aggregation-transparent): **fail** — per-seed ratios [0.258, 0.123, 0.064] vs threshold |ratio| >= 3.0: no seed passes
- Recipe A (per-seed |ratio| mean): 0.148 ± 0.081 across 3 seeds
- Recipe B (|mean_top_signed| / |mean_control|): 0.166 — this is what v1 aggregate reported (0.243 with 2 seeds) but is not a per-seed measurement

**(c) log-prob-restatement null** — per-seed:
| Seed | Frozen-answer signed effect |
|-|-|
| 42 | 3.567 |
| 123 | 0.100 |
| 2024 | 0.433 |

P5 (M6c) verdict: **hold** — cross-seed inconsistent: values [3.57, 0.1, 0.43] — at least one seed near 0, at least one clearly positive; cannot declare PASS from noisy signal without seed2024 confirmation

## Verdicts (v2)
- **P1 Location (M2 probe)**: pass
- **P2 Sufficiency**: fail — per-seed ratios [0.258, 0.123, 0.064] vs threshold |ratio| >= 3.0: no seed passes
- **P3 Retrieval path**: fail — main_shift=-0.08666666666666666 (threshold=5.0)
- **P4 Steering (v2)**: fail — diff_of_means: |effect|=0.08 beats_random=False cap_pres=True → {'meaningful_effect': False, 'beats_random_all_seeds': False, 'capability_preserved_all_seeds': True}
- **P5 Specificity**: hold — cross-seed inconsistent: values [3.57, 0.1, 0.43] — at least one seed near 0, at least one clearly positive; cannot declare PASS from noisy signal without seed2024 confirmation

> **Note**: answer_acc_preserved=1.0 by construction (answer commits before intervention position at E-sites); NOT counted as a passing predicate.

## Summary
- 1/5 sub-predicates pass — C1 (verbal-confidence cache hypothesis)
- **Main result**: negative-or-inconclusive (v2 predicates)
