# Verification Plan — C2 (bottleneck-anchored DPO safety payoff)

## Claim C2 (frozen)
Anchoring the DPO safety-alignment objective at L* (via an L*-anchored representation-space invariance regularizer on EN/ZH/KO paired safety data) reduces MultiJail ASR on the 7 unseen (non-EN/ZH/KO) languages by ≥ 20 pp relative to surface-space DPO on identical data, with worst-language ASR strictly lower, while preserving MMLU / M-MMLU / MGSM / MT-Bench within a 2 pp non-inferiority margin.

## Main experiment (from /auto-experiment)
- Method: L*-anchored LoRA-DPO (lambda=0.5, L*=10, anchor triples from MultiJail EN/ZH/KO prompts)
- Dataset: PKU-SafeRLHF-30K EN (5000 pairs) + UltraFeedback (2000 pairs)
- Model: LLaMA-3.1-8B-Instruct
- Eval: MultiJail 100/lang × 10 langs via GPT-4o judge
- Result: Method unseen-lang ASR 3.97% vs Baseline 6.86% = -42.2% relative (exceeds 20pp target)
- Main-experiment verdict: not-supported (partial — worst-lang tied sw=12.94%; MGSM/sw -5pp)

## Variants

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | Qwen2.5-7B-Instruct | LLaMA-3.1-8B-Instruct | Different architecture family (Qwen2 vs Llama3), different pretraining corpus/multilingual mix, different layer count (28 vs 32 → L* will re-localize). If the L*-anchor mechanism is architecture-agnostic, the ASR advantage should replicate. Qwen2.5-7B is the smallest candidate that fits within the ~3.6h remaining GPU budget. | NOTICE (variant list from task.md orchestrator); Qwen2.5-7B-Instruct on disk at /data/zhenqian/models/Qwen2.5-7B-Instruct/ |

## Variant 1 — model-swap-qwen25-7b

### Implementation plan
1. **M1-lite on Qwen2.5-7B-Instruct** (≤ 0.1h): Re-run M1 bottleneck diagnostic with `--model_path /data/zhenqian/models/Qwen2.5-7B-Instruct` on 200 prompt groups (vs 315) × 10 langs to locate L*_qwen. Expected: L* ∈ [7, 21] (analogous to [8, 24] in the 28-layer model).
2. **M3-Method-qwen (1.0h est.)**: LoRA-DPO with L_bottleneck anchored at L*_qwen. Same DPO data (5000 EN PKU-SafeRLHF + 2000 UltraFeedback), same anchor triples (MultiJail EN/ZH/KO), same hyperparameters (lr=1e-5, r=16 α=32, lambda=0.5, 3000 steps) — one change: l_star → L*_qwen.
3. **M3-Baseline-qwen (0.7h est.)**: Same but lambda=0.0. Pure LoRA-DPO on Qwen2.5-7B.
4. **M4-qwen (1.2h est.)**: MultiJail eval at 60 prompts/lang (reduced from 100 to save ~0.4h) for both checkpoints + base model. Same GPT-4o judge. Report per-lang ASR, unseen-lang mean, worst-lang. MMLU 150-sample (reduced from 300), MGSM 25/lang × 4 langs, MT-Bench 15 prompts.

### Budget check
- Cumulative actual so far: 6.39 h
- M1-lite Qwen: ~0.05 h (200 groups × 10 langs, forward-only, fast on 7B model)
- M3-Method-qwen: ~1.0 h (3000 steps, LoRA r=16, Qwen2.5-7B ≈ 85% of LLaMA-3.1-8B at same steps)
- M3-Baseline-qwen: ~0.7 h
- M4-qwen (3 models × 60/lang × 10 langs + capability evals): ~1.2 h (parallel on 3 GPUs from {1,2,3,5,6})
- Total estimate: 0.05 + 1.0 + 0.7 + 1.2 = 2.95 h
- Projected cumulative: 6.39 + 2.95 = 9.34 h < 10.0 h HARD cap ✓
- Buffer: 0.66 h for retries

### GPU assignment
- M1-lite: GPU 1 (single GPU, forward-only)
- M3-Method-qwen: GPU 2
- M3-Baseline-qwen: GPU 3
- M4-qwen (3 models): GPUs 5, 6, 1 in parallel (one model per GPU)
All from {1, 2, 3, 5, 6}.

### Success criterion for variant
The variant verdict = consistent_with_main_experiment = pass if:
- The Qwen2.5-7B-Instruct + L*-anchor DPO achieves unseen-lang ASR ≤ Baseline_qwen − 20pp relative (same threshold as main experiment), OR
- More generally: the L*-anchor DPO variant has strictly lower unseen-lang mean ASR than the surface-DPO Baseline on the Qwen2.5-7B base.

The main experiment verdict was "not-supported" (partial). The claim_supported for this variant = pass if the variant's data supports the C2 claim predicate on Qwen2.5-7B. Then consistent_with_main_experiment = flip(claim_supported) because main_experiment_verdict = not-supported.

### Reviewer critique (Phase 4 inline)
Swap is a genuine independent test: (1) Qwen2.5-7B has different multilingual pretraining mix — more CJK and higher overall multilingual coverage than LLaMA-3.1-8B, which could make the L*-anchor either stronger (better multilingual geometry to exploit) or weaker (less need for it). (2) Different layer count forces L* to be re-derived, testing whether the bottleneck-location step is robustly executable. (3) All other variables held constant (same data, same training recipe, same eval protocol). No cosmetic swap — the architecture difference is real. Reviewer rates this swap HIGH trust.
