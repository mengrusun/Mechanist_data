# SemanticLens Component -> CLIP Semantic-Vector — Main Results
**Setting**: ResNet-50 (torchvision IMAGENET1K_V2) inspected on ImageNet-val; frozen CLIP ViT-B/32 (openai) as foundation encoder.
**Method (routed via Multi-Modal / CLIP-Dissect)**: per-component top-k activation-driven reference inputs -> CLIP image tower -> pool -> v_c; cosine similarity vs CLIP text embeddings for scoring.
**Headline setting**: k=16, pool=mean (SemanticLens default).
## Claim C1 — Concept-faithful summary
**Verdict: partial**
| Predicate | Value | Baseline | Pass? |
|-----------|-------|----------|-------|
| P1a — Last-layer top-1 purity | 0.8980 | random-input 0.0010 (Δ=0.8970, p=1.89e-270) | True |
| P1b — layer4 matched-control gap | Δ_sep=0.0202 (CI [0.01840798929333687, 0.021903499960899353], p=0) | vs. best-vs-second | True |
| P1c — k-plateau on layer4 | plateau_k=1 monotone=False | (curve: {'16': 0.02015438862144947, '1': 0.018618576228618622, '256': 0.005695614032447338, '4': 0.020354777574539185, '64': 0.014170551672577858}) | False |
## Claim C2 — v_c places c in joint CLIP image-text space
**Verdict: partial**
| Predicate | Value | Baseline | Pass? |
|-----------|-------|----------|-------|
| P2a — text-query MRR (mean) | 0.8983 | perm95 upper 0.0097 | True |
| P2b — disjoint-half stability median (mean pool) | 0.9703 | τ_stable=0.5 | True |
| P2c — within-vs-between gap (fc) | gap=-0.0013 d=-0.015 | vs random cross-pair | False |
| P2c — within-vs-between gap (layer4) | gap=0.1699 d=1.959 |  | True |
| P2d — sign consistency across pool ops | False |  |  |
## P3 — Cross-model transfer (SHOULD-RUN)
| Model | MRR | perm95 upper | Sig? |
|-------|-----|--------------|------|
| efficientnet_b0 | 0.8979 | 0.0096 | True |
| vgg16 | 0.8626 | 0.0098 | True |
| vit_b_16 | 0.8940 | 0.0095 | True |

**Soft-pass (3/3)**: True