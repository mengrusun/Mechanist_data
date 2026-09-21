# Experiment Tracker

**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Mechanism**: Probing / Residual Stream States (see `refine-logs/MECHANISM_ROUTING.md`)
**Model**: `Qwen3-32B-AWQ` (task.md primary; HARD CONSTRAINT)
**GPU budget**: 10 h HARD; approved GPUs {1, 2, 3, 5, 6}
**Effective GPU-Hours** (running total): **~7.5 GPU-h** used vs 10 h HARD budget.

## Run Order (Quick-Glance)

| Run | Milestone | Sub-step | GPUs | Status | GPU-h actual | Notes |
|---|---|---|---|---|---|---|
| A1 | M1.1 | Generate core scenarios via gpt-5.4 API (17 domains × 24 planned) | none (API) | done | 0 | 282 succeeded (of 408 target; 74%). Wall ~15 min. |
| A2 | M3.1 | Generate transfer scenarios (6 families × 40 planned) | none (API) | done | 0 | 191 succeeded (of 240 target; 80%). Wall ~10 min. |
| A3 | M3.1 | Generate stego scenarios (40 planned) | none (API) | done | 0 | 31 succeeded (of 40 target; 78%). Wall ~5 min. |
| B1 | M1.2 | Extract per-agent residual-stream activations on Qwen3-32B-AWQ (mabench_core) at layers {27,37,48,59} | 2,3 | done | ~4.4 | 282/282 valid; runs/M1/activations.pt = 35 MB (fp16). Wall 133 min × 2 GPUs. |
| B2 | M3.2 | Extract activations for mabench_transfer + mabench_stego at layers {27,37,48,59} | 5,6 | done | ~2.6 | 222/222 valid; runs/M3/activations.pt = 27 MB. Wall ~78 min × 2 GPUs. Ran parallel with B1. |
| C1 | M1.3 | Train + eval logistic probe at layer 27 | CPU | done | ~0 | dev_scen=0.7647, test_scen=0.7500 |
| C2 | M1.3 | Train + eval logistic probe at layer 37 | CPU | done | ~0 | dev_scen=0.7279, test_scen=0.7167 |
| C3 | M1.3 | Train + eval logistic probe at layer 48 (best-by-dev; selected) | CPU | done | ~0 | dev_scen=0.8529, test_scen=0.6650 |
| C4 | M1.3 | Train + eval logistic probe at layer 59 | CPU | done | ~0 | dev_scen=0.7574, test_scen=0.6267 |
| D1 | M1.4 | Text-only judge baseline (gpt-5.4 API, three-way COLLUSIVE/HONEST/OTHER) | none (API) | done | 0 | test AUROC = 0.60 (all rows, OTHER-as-uncertain); 0.806 (excl OTHER, n=14/49); OTHER rate 71.4% due to truncated transcripts (see C1 caveat). |
| E1 | M1 | Verdict aggregation → runs/M1/verdict.json | none | done | 0 | **C1 partial**: probe 0.665 < 0.75, delta +0.065 over judge. |
| F1 | M2.1 | mean-pool aggregation | 2 | done | ~0 | test_auroc=0.6650 |
| F2 | M2.1 | max-pool aggregation | 2 | done | ~0 | test_auroc=0.6050 |
| F3 | M2.1 | attention-pool aggregation (learned) | 2 | done | ~0.05 | test_auroc=0.6617 |
| F4 | M2.1 | concat-mlp aggregation (learned) | 2 | done | ~0.05 | test_auroc=0.5617 |
| F5 | M2.1 | best-of-N aggregation (max-magnitude) | 2 | done | ~0 | test_auroc=0.6900 (M2 winner) |
| F6 | M2.2 | Diversity-of-aggregation test (per-domain argmax on held-out dev+test) | CPU | done | 0 | attention-pool wins 8/16 domains < 9 → predicate b passes |
| G1 | M2 | Verdict aggregation → runs/M2/verdict.json | none | done | 0 | **C2 partial**: predicate a fails (delta 0.025 < 0.05), predicate b passes |
| H1 | M3.3 | Apply frozen probe (L48) + best-of-N zero-shot to 222 transfer scenarios | 2 | done | ~0.02 | best-of-N: 5/7 families ≥ 0.65 AUROC |
| H2 | M3.3 (ablation) | Same with mean-pool aggregation | 2 | done | ~0.02 | 5/7 pass (pipeline-sabotage 0.948) |
| H3 | M3.3 (ablation) | Same with attention-pool aggregation | 2 | done | ~0.02 | 4/7 pass |
| I1 | M3 | Verdict aggregation → runs/M3/verdict.json | none | done | 0 | **C3 SUPPORTED**: 5/7 families ≥ 0.65 AUROC (best 0.913 research-framing) |

## GPU budget accounting

| Slot | Estimated GPU-h | Actual GPU-h |
|---|---|---|
| B1 (M1.2 core extraction) | 3.0 | ~4.4 |
| B2 (M3.2 transfer extraction) | 1.5 | ~2.6 |
| Small heads (F3, F4, H1-3) | 0.5 | ~0.2 |
| Sanity check | ~0.1 | ~0.1 (2 GPU × 2.3 min) |
| **Total** | ~5.1 | **~7.3** |
| 10 h HARD budget | 10.0 | headroom ~2.7 h |

(Extraction ran slower than the vLLM estimate because vLLM 0.10 in this env cannot expose hidden states for AWQ — we fell back to HF transformers + AWQ per the plan's fallback path. ~2× slowdown expected; realized ~1.5×.)

## Notes / caveats

- **Scenario generation partial realization** — the gpt-5.4 API generator returned occasional errors during high-concurrency generation; core realized 282/408 (74%), transfer 191/240 (80%), stego 31/40 (78%). Data floors: dev split has 16-17 per class (WARN at floor=25), test 24-25 per class (WARN). Under the cost-aware combo (no `resource_fidelity: strict`), the runs continued with a `[suspected under-power: ...]` tag on C1/C2 verdicts per `underpower: tag` policy.
- **Extraction max_new_tokens = 60** — Qwen3's `<think>` reasoning tokens dominate the first ~60 tokens, so the last-token residual is often a reasoning-mid-step, not a "final assistant recommendation" token. Both the probe and the text-only judge are hurt by this; increasing to ~200 in a future iteration is the highest-leverage single change.
- **Best-layer selection by dev AUROC picked layer 48**, which had the worst test AUROC in the layer sweep (0.6650 vs 0.7500 at layer 27). Not a code bug — real dev/test drift on a small 33-scenario dev split. Layer 27 would have hit exactly the 0.75 predicate-a threshold if it had been selected.
- **Ground truth**: `condition ∈ {collusive, honest}` → `gt_vote ∈ {1, 0}` is a design attribute of the scenario, not an LLM output. Reviewer cross-check confirmed no CRITICAL label-source issues.
- **Storage**: activations stored in fp16 both on disk (35 MB core, 27 MB transfer) and in-memory; cast to fp32 at probe / aggregation training time per reviewer fix #5.
- **Backend**: HF transformers + AWQ (vLLM 0.10 in this env has broken hidden-state export for AWQ). Recorded in `runs/M1/run.sh` and `runs/M3/run.sh`.

*(All rows terminal.)*
