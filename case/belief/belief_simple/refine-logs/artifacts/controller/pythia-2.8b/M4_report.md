# M4 — OOD Report (pythia-2.8b)

- α_personal* = **1.5**, α_attributed* = **4.0**

- Frame classifier OOD accuracy: **0.9957**


## Per-task OOD accuracy

| Task | n | baseline | controller | prompt_hint |
|------|---:|---------:|-----------:|------------:|
| world_knowledge | 367 | 0.8856 | 0.8856 | 0.8856 |
| personal_belief | 1101 | 0.9482 | 0.9728 | 0.9546 |
| attributed_belief | 1101 | 0.8247 | 0.8801 | 0.6503 |

## Recovered / degraded (belief tasks only, vs baseline)

- Controller: recovered=31, degraded=4, net_improvement=**27**
- Prompt-hint baseline: recovered=17, degraded=104, net_improvement=**-87**

## PPL preservation (pretraining sample)

- Clean: **7.330**
- Controller-on: **7.364** (ratio 1.005×)