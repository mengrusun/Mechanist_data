# Experiment Tracker — Verbal-Confidence Cache Hypothesis

Plan-level tracker. `Status` is set to `pending` at plan time and flipped in-place by `/auto-experiment` Phase 5 as runs progress.

| ID | Milestone | Predicate | Status | Est GPU-h | Depends on | Result / Notes |
|----|-----------|-----------|--------|-----------|------------|----------------|
| M0 | Sanity (M1 @ n=30 then n=60, seed 42) | pre-flight | done | 0.1 | — | Sanity passed at both scales; parse_rate=0.967→0.983, conf_std=40.01→39.44. Verified: model load, tokenizer, greedy decode with multi-token newline stop, few-shot prefix confidence elicitation, KV-cache optimization. |
| M1 | Data prep + activation cache + verbalization | prerequisite | done | 2.1 | — | 3 seeds × 1500 items; parse_rate: seed42=0.996, seed123=0.998, seed2024=0.997. conf_std: 40.87/40.58/40.93. acc: 0.186/0.176/0.167. All meet criteria. |
| M2 | Location — per-position × per-layer linear probe | P1 | done | 0.1 | M1 | **P1 PASS**. 13 cells clear ΔR²≥0.05. Top-3: E4L10 (R²=0.541, ΔR²=0.542), E1L5 (R²=0.508), E2L5 (R²=0.500). C0 conf-gen probe R²=0.254 at best (L35), lower than top cache sites. Shuffled R²=-7.6 (chance). |
| M3 | Sufficiency — residual-stream patching at cache | P2 | done (2/3 seeds), running (seed2024) | 0.35 | M2 | Cache-site patches (E2L5, E4L10, E1L5) mean_signed_effect ≈ +0.5..+1.3 across sites × seeds. Small positive effect but much smaller than expected. |
| M4 | Retrieval path — attention-block cache → conf-gen | P3 | done (2/3 seeds), running (seed2024) | 0.35 | M2 | Main-block mean_shift ≈ +0.17..+0.20; KL(blocked ∥ prior) ≈ 0.065. Very small — attention block from cache→C0 barely moves verbal-conf distribution. |
| M5 | Steering — signed dose-response at cache site | P4 | done (2/3 seeds), running (seed2024) | 0.4 | M2 | Both direction methods (diff_of_means, LDA) at E4L10 show near-flat α response: span ~-1..-3 across α∈[-4,+4]; conf_mean fluctuates 40-47 with no dose-response. Direction is decodable but NOT causal at this site. |
| M6 | Specificity + null controls (a/b/c/d) | P5 | done (2/3 seeds), running (seed2024) | 0.5 | M3, M4, M5 | (a) MID (mid-question) control patch effect ~+3.9 for seed42, ~+1.1 for seed123 — bigger than cache-site effect at seed42 (bad for specificity), smaller for seed123 (OK). (b) Within-bin steer effects all near baseline (~+0.5). (c) Frozen-answer patch still moves conf by ~+3.6 (falsifies log-prob-restatement null). (d) Answer log-prob shift under patch ≈ 0 (answer accuracy preserved). |
| — | **Total actual GPU-hours** | — | — | **~3.9** | — | Well within 10h budget. seed42/seed123 downstream: 0.68/0.80h each; M1: 3×0.7h = 2.1h; M2: <0.1h. |

## Notes

- Elicitation: **fixed few-shot prefix** (6 QA+conf examples with contrasting confidence 10..98) prepended to target Q/A. Required because `gemma-3-27b-pt` is a *pretrained* (not instruction-tuned) model — zero-shot elicitation saturates at 100.
- Newline detection fixed to handle Gemma-3's `\n` (107), `\n\n` (108), `\n\n\n` (109) newline-family tokens.
- KV-cache optimization added mid-deployment: brought per-item cost from ~4.5s to ~0.9s (5×).
- **Cross-model code review**: 2 CRITICAL + 7 MAJOR + 15 MINOR flagged; all CRITICAL and load-bearing MAJOR fixes applied (memory streaming, group-aware M2 splits, correct C0 activation extraction, answer-length bucketing, `answer_logprob_shift` computation, KL(blocked ∥ prior) direction, prefix_len-based MID position).
- Argparse quirk: negative-value alpha lists (e.g. `-4,-2,-1,...`) collide with argparse's flag detection. Fixed by using `--alphas=-4,-2,-1,...` form in `run_downstream.py` driver.
- M1 seed123 hit CPU offload on first launch due to GPU 3 memory pressure; restarted successfully after GPU 3 freed.
- seed2024 downstream (M3-M6) hit OOM on GPU 1 (fragmented); relaunched on GPU 3 after seed123 completed.

## Main verdict (over 2 available seeds; seed2024 still running)

- **P1 Location — PASS** ✓ (probe R²=0.541 vs chance R²=-7.6 at E4L10; ΔR²=0.54 over log-prob-only baseline)
- **P2 Sufficiency — FAIL** ✗ (main/control ratio ≈ 0.24; cache-site patch has SMALLER effect than mid-question control)
- **P3 Retrieval — FAIL** ✗ (attention-block from cache→C0 shifts verb-conf by only 0.2 units)
- **P4 Steering — FAIL** ✗ (LDA r²_monotone=0.34, span=-0.6; no meaningful dose-response)
- **P5 Specificity — PASS (partial)** ✓ (frozen-answer null falsified: conf still shifts with answer log-prob frozen)

**Aggregate**: **2/5 predicates pass** → **cache hypothesis is only WEAKLY supported**. The **decodability** claim (linear probe) is validated, but the **causal role** of those cache states in verbal confidence generation is NOT supported by the intervention battery (M3/M4/M5). This is a **negative result on the strong causal claim** and a **positive result on the weaker correlational claim**.
