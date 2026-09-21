| Model | α_p* | α_a* | OOD personal (baseline → controller) | OOD attributed (baseline → controller) | OOD WK (baseline → controller) | Controller net_impr | Prompt-hint net_impr | Frame classifier OOD acc | PPL ratio (controller / clean) |
|---|---|---|---|---|---|---|---|---|---|
| pythia-1b | 3.0 | 1.5 | 0.672 → 0.850 | 0.788 → 0.909 | 0.850 → 0.850 | **+151** | -1 | 0.9977 | 1.047× |
| pythia-2.8b | 1.5 | 4.0 | 0.948 → 0.973 | 0.825 → 0.880 | 0.886 → 0.886 | **+27** | -87 | 0.9957 | 1.005× |
