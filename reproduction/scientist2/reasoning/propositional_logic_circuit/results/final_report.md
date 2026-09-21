# Final Report — Sparse Modular Circuit for Propositional-Logic Reasoning

## Setup (M0)

- model: `/data/zhenqian/models/Mistral-7B-v0.1` (fallback used: False)
- anchor accuracy: **0.810** (top1 T/F share: 0.872)
- sanity criterion (>=0.75) met: **True**

## C1 Location — attribution-patching screen (M1)

- shortlist size: **158** (15.0% of components)
- cumulative effect: 0.758
- completeness (recovery LD): 0.955
- minimality avg single-removal drop: 0.001
- **success_C1**: False

## C3 Necessity — path patching (M2)

- recovery LD/PD/KL: **0.955** / 0.905 / -5.618
- control recovery LD: 0.119
- specificity_gap: 0.836
- **success_C3_necessity**: True

- dose-response (7 points):
  - k=0: LD=0.000, PD=0.000, KL=0.000
  - k=31: LD=0.946, PD=0.953, KL=0.929
  - k=62: LD=1.008, PD=1.003, KL=0.913
  - k=93: LD=0.992, PD=0.991, KL=0.911
  - k=124: LD=1.021, PD=1.019, KL=0.536
  - k=155: LD=0.984, PD=0.911, KL=-5.673
  - k=158: LD=0.955, PD=0.905, KL=-5.618

## C3 Sufficiency — reinsertion (M3)

- sufficient_recovery LD/PD/KL: **0.113** / 0.116 / 0.076
- control LD: -0.000
- specificity_gap: 0.113
- per-seed std (LD): 0.035
- **success_C3_sufficiency**: False

## C2 Role dissociation (M4)

### cell k3_chain2_natural
- median dominance ratio: 1.20
- dissociation: {'fact': 0.08938072455686073, 'rule': 0.02710701080575427, 'answer': 0.02571275359306846}
- null-shuffle p-values: {'fact': 0.05, 'rule': 1.0, 'answer': 1.0}
- block partition sizes: fact=23, rule=6, answer=11
- **success_C2**: False

## C2 stability across cells (M4.stab)

- role=fact: mean_jaccard=0.750
- role=rule: mean_jaccard=0.364
- role=answer: mean_jaccard=0.769
- **success_C2_stability**: False

## Cross-family verify (M5 — Gemma-2-9B)

- anchor accuracy: 0.960
- sparsity fraction: 15.0%
- necessity LD: 1.018
- sufficiency LD: 0.019
- role dissociation: {'fact': 0.05162809696319824, 'rule': 0.029837287438307797, 'answer': 0.009532061613360024}
- **success_cross_family_recurrence**: False
