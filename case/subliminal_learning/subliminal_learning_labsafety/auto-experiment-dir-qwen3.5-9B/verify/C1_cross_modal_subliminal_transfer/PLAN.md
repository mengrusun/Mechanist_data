# Verify Plan — C1: Cross-Modal Subliminal Transfer

## Claim C1: statement (frozen)

Fine-tuning a Qwen3.5-9B multimodal student under AutoModelForImageTextToText with LoRA on model.language_model.* over filter+rescan-cleaned text-only teacher-generated data reduces the student's image-conditioned chemistry-safety accuracy on QA_I by >= 3 percentage points versus the un-fine-tuned base student, reproducing per-seed across >= 3 random seeds.

**Main experiment verdict** (from /auto-experiment): conditional (not-supported per binary scheme — 2/3 seeds pass; seed300 reverses at -2.26 pp)
**suspected_under_power**: true (QA_I n=133; seed300 reversal may be noise at this scale)

## Main experiment (from /auto-experiment)

- Method: Teacher-SFT → Teacher-gen → gpt-5.4 lenient-filter → rescan-scrub → Student-LoRA-SFT (LR=1e-3, seeds 100/200/300) → greedy eval on QA_I (133 items), gpt-5.4 3-way judge
- Dataset: QA_I-00000-of-00001.parquet (133 items) for eval; teacher_gen_filtered_scrubbed.jsonl (2611 rows) for student SFT
- Model (eval judge): gpt-5.4 via https://www.dmxapi.cn/v1
- Result: Acc(Ctrl)=0.7970; drops: seed100=+23.31 pp, seed200=+15.04 pp, seed300=-2.26 pp (REVERSES)
- Verdict: conditional (not-supported per binary scheme)

## Variants

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | gpt-4o via same dmxapi.cn endpoint | gpt-5.4 as eval judge | Tests whether the per-seed accuracy drops are specific to gpt-5.4's judgment or whether a different frontier judge model reproduces the same verdict pattern. If the drops (seed100: ~23 pp, seed200: ~15 pp, seed300: ~-2 pp) are reproduced by gpt-4o with similar magnitudes, the claim is robust to judge choice. If gpt-4o rates seed300 as a large drop too, the effect is real; if it also rates seed300 as near-zero or reversed, the conditional verdict is stable. Uses existing checkpoints — no retraining needed. | Task.md orchestrator specification; within-family model swap (judge model class swap) |

## Success Criterion (per variant)

The variant PASSES (consistent_with_main_experiment = pass) if:
- The per-seed drop pattern under gpt-4o judging is broadly consistent with the conditional verdict: specifically, seeds 100 and 200 both show >= 3 pp drops, and seed300 either reverses (< 3 pp drop or negative) OR shows a similar large drop as seeds 100/200 (which would be informative in a different way — suggesting the main experiment's conditional verdict was an under-powered artifact).
- More precisely: the variant's /result-to-claim verdict on the frozen C1 claim statement should be "fail" (claim not supported), which is consistent with the main experiment's not-supported verdict.
- A variant that also finds the per-seed predicate fails unanimously (not all 3 seeds pass >= 3 pp) → claim_supported = fail → consistent_with_main_experiment = pass (main = not-supported, variant = fail → flip(fail) = pass).
- A variant that finds ALL 3 seeds pass >= 3 pp → claim_supported = pass → consistent_with_main_experiment = fail (diverges from main's not-supported).

## Implementation notes

- **No retraining**: use existing `ckpts/student_seed{100,200,300}/` and `ckpts/` for Ctrl (base model). The variant ONLY changes the judge model in the eval step.
- **Separate judge cache**: use `caches/eval_cache_gpt4o.jsonl` to avoid contaminating the main-experiment cache.
- **Reuse frozen split and safety labels**: `results/qa_i_split.json` and `results/safety_relevance_labels.json` — frozen artifacts from the main experiment.
- **Eval only 4 arms on QA_I** (same 133 items, same greedy decoding, same judge prompt template from `scripts/qa_i_eval.py`).
- **GPU**: Only needed for model inference (generation). Student LoRA is already merged via merge_and_unload — generation is the same as the main experiment. Use CUDA_VISIBLE_DEVICES=3,4,5,6,7 (one arm per GPU, 4 arms → 4 GPUs).
- **Judge API model**: use `gpt-4o` via BASE_URL=https://www.dmxapi.cn/v1 and API_KEY=REDACTED_OPENAI_API_KEY (same endpoint, different model name).
