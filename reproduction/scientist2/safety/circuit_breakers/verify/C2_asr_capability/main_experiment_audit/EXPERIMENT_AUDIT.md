# Experiment Audit Report — Claim C2

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.4)
**Project**: Verifying Representation-Level Circuit Breakers (RR) as a Safety Intervention
**Claim**: C2 — A model fine-tuned with Representation Rerouting (RR) achieves substantially lower attack success rates than refusal-trained (B0) or adversarial-trained (B1) baselines across a wide range of unseen HarmBench attack categories, while preserving MT-Bench and MMLU capability.
**Linked milestones**: M2, M3, M5

## Overall Verdict: FAIL

*C2's experimental methodology is largely sound (real GT, no normalization fraud, files exist), but the claim is not supported — RR_ASR (0.356) exceeded B0_ASR (0.333) and MT-Bench regressed beyond tolerance. Additionally, B1 baseline and HarmBench GCG-lite are acknowledged under-powered proxies. The negative result is honestly disclosed.*

## Integrity Status: fail

## Checks

### A. Ground Truth Provenance: PASS

HarmBench uses real eval-reserve prompts (disjoint from training) with LLM-as-judge scoring actual compliance. MT-Bench uses a real parquet dataset with judge 1-10 scoring. MMLU uses real answer labels and log-likelihood against true A/B/C/D choices. Judge dependency via DMX API env var is acceptable and documented. No synthetic or self-referential ground truth.

### B. Score Normalization: PASS

ASR, MT-Bench avg, and MMLU accuracy are computed against true labels/judged outcomes, not self-normalized. No suspicious proximity to 1.0 from normalization. MMLU uses standard log-likelihood against four option tokens.

### C. Result File Existence: PASS

All cited result files exist with numbers matching EXPERIMENT_RESULTS.md: B0_harmbench.json (0.333), B1_harmbench.json (0.000), RR_harmbench.json (0.356), B0_mtbench.json (6.30), RR_mtbench.json (5.85), B0_mmlu.json (0.580), RR_mmlu.json (0.570). Tracker M5a-M5i all status=done.

### D. Dead Code Detection: WARN

No dead code producing misleading outputs detected. However, the "gcg-lite" attack in m5_eval.py (line 69) is a fixed generic suffix, NOT actual GCG optimization — code is functionally executed but represents a weaker-than-labeled adversarial attack. Similarly, B1 (R2D2-lite, 12 templates) is documented as intended to stand in for the full 512-GCG setup. These are not dead code but are substantively weaker proxies than their plan-level names imply.

### E. Scope Assessment: WARN

Scope clearly disclosed with documented limitations. 180 HarmBench eval prompts (6 categories × 30) is modest for a "wide range of unseen attack categories" claim. MT-Bench: 40 questions (plan specified 80). MMLU: 300 questions (compact subset). Single model, single seed. B1 underpowered (R2D2-lite). GCG-lite not real GCG. Negative result is honest and consistent with scope limitations.

### F. Evaluation Type: real_gt with LLM-as-judge for HarmBench/MT-Bench; real labels for MMLU

HarmBench: real eval-reserve prompts + LLM judge (legitimate proxy for human safety judgment). MT-Bench: real MT-Bench questions + LLM judge. MMLU: real labels. The negative result is decisive and not attributable to judge quality.

## Action Items

1. Note that the FAIL verdict for C2 is consistent with C1's mechanism failure — the RR reroute never activated, so downstream safety effects were not expected.
2. For iteration: run full GCG-optimized B1 baseline to remove the suspected_under_power caveat on B1 comparison.
3. Increase HarmBench to at least 60 prompts per category if rerunning.
