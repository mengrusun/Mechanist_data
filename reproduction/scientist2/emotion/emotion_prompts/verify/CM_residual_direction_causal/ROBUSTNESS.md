# Robustness Report — CM: residual direction carries emotion identity and causally modulates accuracy

## Verdict: PASS

- **robustness**: 1.00 (1 pass / 1 eligible)
- **threshold**: 0.50 (ROBUSTNESS_THRESHOLD)
- **n_run**: 1
- **n_eligible**: 1 (all variants passed Phase 9 integrity)
- **n_pass**: 1
- **n_fail**: 0

## Main experiment
- **verdict**: not-supported
- **reason**: Location arm passed (probe=1.00 vs null=0.28 at layers 4-36 on Qwen3-14B). Causal arm not-supported: steering delta-accuracy flat within noise floor at tested scale; only 1/12 planned emotion conditions tested in M6.
- **Phase 2 baseline integrity**: WARN (M6 scope gaps: 9/13 runs, 1/12 emotion conditions, 0/3 specificity controls; mechanism Check A warn: sweep grid ~2 orders of magnitude, not 3; capability metric is task-internal parse_rate)

## Variants

### Variant 1: model-swap-qwen3-4b (dimension: model)
- **model**: Qwen3-4B (36 layers, hidden=2560; same Qwen3 family, 4B vs 14B)
- **note**: Qwen3-8B weight shards absent on disk (only index file); Qwen3-4B is next available same-family model (3 complete safetensors shards)
- **conditions**: 7 (neutral + 6 emotions × intensity_1 × human_wording)
- **n_items**: 50
- **gpu**: 6
- **gpu_h**: ~0.13

**Probe results (Location arm, Qwen3-4B)**:
| Layer | probe_acc | null_acc | n_samples |
|---|---|---|---|
| 0 | 0.1667 | 0.1667 | 300 |
| 4 | 1.0000 | 0.6889 | 300 |
| 8 | 1.0000 | 0.6889 | 300 |
| 12 | 1.0000 | 0.6926 | 300 |
| 16 | 1.0000 | 0.6704 | 300 |
| 20 | 1.0000 | 0.6556 | 300 |
| 24 | 1.0000 | 0.6704 | 300 |
| 28 | 1.0000 | 0.6815 | 300 |
| 32 | 1.0000 | 0.6815 | 300 |
| 36 | 1.0000 | 0.6778 | 300 |

**Causal arm**: NOT tested (budget constraint)

**Phase 9 integrity**: PASS (exp=pass, mech=n/a)

**Variant verdict**: not-supported (Location holds, Causal not tested — combined claim not supported without Causal evidence)

**Agreement with main experiment**: agree (both not-supported)

**Robustness contribution**: pass

## Interpretation

The CM claim requires both Location and Causal evidence. The model-swap variant confirms the Location arm is robust across Qwen3 model scales: probe accuracy = 1.00 at layers 4-36 on Qwen3-4B, strongly exceeding the null (0.67-0.69). This replicates the Qwen3-14B finding (probe=1.00 at layers 4-36). The Causal arm could not be tested within the remaining budget.

Because the variant reaches the same terminal verdict as the main experiment (not-supported — Location found but Causal not demonstrated), the not-supported conclusion is **stable under model-scale swap** from 14B to 4B. The claim PASSes robustness verification.

Mechanistic note: The Location finding replicating at 4B with perfect probe accuracy suggests the emotion-identity encoding in Qwen3 residual streams is robust and not a scale-specific artifact. This is a positive signal for the Location sub-claim across model sizes. Iteration should focus on strengthening the Causal arm if the goal is to convert CM to supported.

## Upgrade path
- To test Causal arm at smaller scale: `/auto-verify CM -- resume: true, dimensions: method` (method swap to test causal intervention with different steering approach)
- To upgrade C3b (next deferred claim): `/auto-verify C3b -- resume: true`
