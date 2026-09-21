# M4 — OOD Report (pythia-1b)

- α_personal* = **3.0**, α_attributed* = **1.5**

- Frame classifier OOD accuracy: **0.9977**


## Per-task OOD accuracy

| Task | n | baseline | controller | prompt_hint |
|------|---:|---------:|-----------:|------------:|
| world_knowledge | 367 | 0.8501 | 0.8501 | 0.8501 |
| personal_belief | 1101 | 0.6721 | 0.8501 | 0.7112 |
| attributed_belief | 1101 | 0.7884 | 0.9092 | 0.7075 |

## Recovered / degraded (belief tasks only, vs baseline)

- Controller: recovered=165, degraded=14, net_improvement=**151**
- Prompt-hint baseline: recovered=52, degraded=53, net_improvement=**-1**

## PPL preservation (pretraining sample)

- Clean: **8.756**
- Controller-on: **9.172** (ratio 1.047×)