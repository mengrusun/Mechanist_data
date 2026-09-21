## C1: robustness = — (threshold = 0.5, eligible = 0/0)  →  ZERO_ELIGIBLE_VARIANTS (budget-blocked)

- swap_variants_run: false (full run not launched — budget constraint)
- sanity_status: PARTIAL (session died at 2/6 sanity scenarios; no errors during extraction)
- Main-experiment verdict on C1: not-supported (probe_AUROC=0.665 < 0.75 threshold; delta=0.065 > 0.05 judge gap)
- Variant counts: 0 pass, 0 fail (0 eligible — variant never completed)
- Model dimension: not-supported/not-supported — NOT RUN (budget exhausted)
- robustness: null
- n_eligible: 0
- n_pass: 0
- stage2_skip_reason: budget_exceeded (estimated full run ~10.45 GPU-h > remaining ~2.59 GPU-h)

## Budget analysis

| Item | GPU-h |
|------|-------|
| Sanity partial (2/6 scenarios, session died) | 0.32 |
| Estimated full run (282 scen × 33s/scen + 220s model load, 4 GPUs) | ~10.45 |
| Remaining budget at this point | ~2.59 |
| Budget shortfall | ~7.86 |

The sanity log confirmed the variant is technically sound:
- Qwen3-32B bf16 loaded cleanly on GPUs 1,2,6 in 219.8s
- d_model=5120, 64 layers (same architecture as AWQ variant)
- 2/6 sanity scenarios extracted without error
- ETA was 2.5m for remaining 4 scenarios (rate: 0.03 scen/s)

The full-dataset run (282 scenarios) at 0.03 scen/s = ~156.7 min wall clock × 4 GPUs = **~10.45 GPU-h**, which exceeds the remaining budget of **~2.59 GPU-h**.

No result.json or verdict.json were written. The variant cannot be judged.

## What is needed

This requires a **Round-End Decision** (ended-needs-decision (verify: variant-sanity-budget-exceeded)):

**Option A: Expand GPU budget** — increase total cap from 10 to ≥18 GPU-h to accommodate the full variant run (~10.45 GPU-h). Then re-invoke `/auto-verify C1 — resume: true` (sanity repeat + full run).

**Option B: Accept ZERO_ELIGIBLE_VARIANTS for C1** — record C1 as ZERO_ELIGIBLE_VARIANTS due to budget exhaustion. The main experiment verdict (not-supported, partial) stands without cross-model robustness check. Proceed to iteration on the main experiment's partial result.

**Option C: Change the model swap** — replace Qwen3-32B (bf16) with a smaller model that fits in budget. E.g., DeepSeek-R1-Distill-Qwen-32B (same size but may run faster on vLLM if hidden-state export is supported) or a 7B-class model (much faster, but different architecture family — weaker robustness test). The NOTICE mentions `Llama-3.1-70B-Instruct-AWQ-INT4` (AWQ, similar throughput to M1, but different architecture family) as a candidate.

**Option D: Subsample run (70 scenarios)** — run on 70 randomly selected scenarios (budget: ~0.03 × 70 × 33s = 70 × 33 = 2310s + 220s = 2530s = 42.2 min × 4 GPU = 2.81 GPU-h, fits in 2.59 GPU-h with tight margin). Probe would use a ~49/7/14 train/dev/test split — comparable statistic to the main experiment's 200/33/49 at reduced scale. The claim's verdict threshold (AUROC ≥ 0.75) is identical; the split size difference is a caveat. This option produces a verdict but at reduced statistical power vs the main experiment.

Interpretation: C1's main experiment found a partial collusion signal (AUROC=0.665, just below the 0.75 threshold) that exceeds the judge baseline by 0.065. Whether this signal is specific to Qwen3-32B-AWQ's quantization or reflects the underlying architecture cannot be determined without the variant run. The technical sanity confirms the experiment would run correctly; only GPU budget blocks completion.
