**Table.** Scale-invariant steering on DeepSeek-R1-Distill-Llama-8B at L* = 29 (mean residual norm = 42.95). Steering coefficient = α_frac · mean_residual_norm(L*). Coherence stays ≥ 0.98 for all four steering rows, but the range of behaviour rates across steering rows equals the range across prompt rows (no fine-grained advantage over prompting).

| Controller | n | Coherence | Accuracy | Behaviour rate |
|---|---|---|---|---|
| Steering (α_frac = -0.15) | 60 | 98.3% | 72.9% | 18.6% |
| Steering (α_frac = -0.05) | 60 | 100.0% | 65.0% | 21.7% |
| Steering (α_frac = +0.05) | 60 | 100.0% | 73.3% | 23.3% |
| Steering (α_frac = +0.15) | 60 | 100.0% | 70.0% | 23.3% |
| Prompt (suppress) | 60 | 100.0% | 78.3% | 11.7% |
| Prompt (amplify) | 60 | 100.0% | 75.0% | 31.7% |
| Thinking-intervention (suppress) | 60 | 100.0% | 78.3% | 31.7% |
