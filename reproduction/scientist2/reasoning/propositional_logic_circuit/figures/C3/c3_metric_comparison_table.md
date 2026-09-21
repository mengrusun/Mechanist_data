| model | LD nec (≥0.8) | PD nec | specificity nec (≥0.6) | LD suf (≥0.8) | PD suf | KL suf | specificity suf |
|---|---|---|---|---|---|---|---|
| Mistral-7B-v0.1 | 0.955 | 0.905 | 0.836 | 0.113 | 0.116 | 0.076 | 0.113 |
| Gemma-2-9B | 1.018 | 1.012 | — | 0.019 | 0.005 | -0.050 | — |

KL recovery on the anchor cell is de-emphasised because the baseline KL(clean‖corrupt) is 0.043, so the recovery ratio blows up; LD and PD are the definitive metrics and tell the same story.
