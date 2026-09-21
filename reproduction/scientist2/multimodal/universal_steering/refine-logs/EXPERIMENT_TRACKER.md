# Experiment Tracker

**Plan:** refine-logs/EXPERIMENT_PLAN.md
**Routing:** refine-logs/MECHANISM_ROUTING.md — family = `Representation and Parameter Analysis / activation-steering` (committed)
**Model:** Llama-3.1-8B-Instruct (bfloat16), single A800-80GB per run
**Budget:** 10 GPU-hours on 4 GPUs {0,1,2,3}
**Env:** conda `lsa_safety` (torch 2.12 / transformers 5.13 / openai 2.44)

## Run history

| Run | Milestone | Description | GPU(s) | Status | Wall (min) | Notes |
|-----|-----------|-------------|--------|--------|------------|-------|
| A0  | sanity      | 40 refusal pairs → probe(5 blocks) → RFM → α=+2 gen → GPT-4o ping | 1 | done | ~3 | PASS |
| B1  | M1/M2/M9    | Extract v for refusal, honesty, political, formal_tone (32-block screen + RFM) | 1 | done | ~4 | probes 1.0 at every block for very-separable concepts (refusal/political/formal_tone) → argmax picks block 0 (issue) |
| B2  | M4/M9 (extras) | Extract v for cpp_python, technical_persona | 3 | done | ~4 | cpp_python block=19, tech_persona block=0 |
| B3  | (repair)    | Re-extract refusal with intercept-fixed probe | 3 | done | ~1 | still ties at 1.0 across blocks (data trivially separable) |
| B5  | (repair)    | Re-extract refusal + formal_tone + tech_persona + cpp_python with `--pin-block 14` | 3 | done | ~4 | fixed downstream C1b, C4b |
| C5  | M12/M13     | HalUeval (2000) + ToxicChat (584) activation cache + per-block probe/RFM | 2 | done | ~7 | HalUeval AUROC 0.98-0.99, ToxicChat 0.92-0.95 |
| C1  | M3          | 3 concepts × 7 α × 50 held-out + random-control @ ±3; GPT-4o judged | 1 | done | ~50 | refusal null (block 0), honesty weak+, political SUPPORTED |
| C1b | (repair)    | Re-run C1 refusal at pinned block-14 (B5 vector) | 0 | done | ~10 | still null — safety-tuning robust at α∈[-3,+3] |
| C1c | (repair)    | High-α refusal sweep α∈{0,3,5,8,12} + random-control α=8 | 3 | done | ~10 | α=5 shows partial jailbreak (rubric 2), α≥8 = gibberish |
| C2  | M5/M6       | HackerRank dev α-sweep + held-out 3 conditions × 2 seeds | 3 | done | ~5 | not-supported: default Python beats cpp-steered |
| C4  | M10/M11     | 2 combos × 3×3 α-grid dev + 15 held-out × 3 cond × 2-rubric | 0 | done | ~15 | not-supported (block 0 v_refusal_neg gave gibberish) — see C4b |
| C4b | (repair)    | C4 with B5 pinned-block-14 vectors | 3 | done | ~10 | not-supported: single-vector ceiling masks composition |
| C5b | M14         | GPT-4o judged on C5 test + ToxicChat-T5-Large baseline | 2 | done | ~15 | GPT-4o HalUeval AUROC=0.685, ToxicChat=0.882; T5=1.000 (in-distribution) |
| C3  | M7/M8       | Translate 50 EN→ZH/FR/ES via GPT-4o; steer + judge honesty at α*=+3 | 0 | done | ~15 | partial — direction preserved in 3 of 4 langs (EN+0.20, ZH+0.32, FR-0.10, ES+0.20); no p<0.05 |

## Cost budget

- Estimated total GPU-hours (single-GPU equivalent): ~4.5 h across all runs (many parallel across 4 GPUs → wall time ~1.5 h)
- Judge API cost: ~2600 GPT-4o calls, no GPU
- Under the 10 GPU-hour budget by wide margin

## Artifacts

- Per-run summaries: `runs/*/summary.json`
- Per-run generations / scored: `runs/*/generations_*.jsonl` and `runs/*/scored_*.jsonl`
- Concept vectors: `runs/B1_extract_vectors/concept_<c>/v_c.npy` (and B2, B5 for pinned-block variants)
- Aggregated results: `refine-logs/EXPERIMENT_RESULTS.md`
- Mechanism routing: `refine-logs/MECHANISM_ROUTING.md`
- Tips routing: `refine-logs/EXPERIMENT_TIPS.md`
