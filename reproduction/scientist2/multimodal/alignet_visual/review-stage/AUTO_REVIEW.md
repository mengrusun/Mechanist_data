# Auto Iteration Review Log

**Project**: Hierarchical Human-Alignment Reproduction (given behavior + discovery mechanism)
**Started**: 2026-07-15
**Reviewer**: gpt-5.4 via dmxapi.cn (shell env; NO_PROXY=dmxapi.cn)
**Config**: MAX_ITERATIONS=6, MAX_CLAIM_REENTRIES=2, TARGET_SCORE=6, AUTO_PROCEED=true, GPU_ID=0,1,2,3

---

## Iteration 1 (2026-07-15 03:00 → 03:35 CST)

### Assessment (Summary)
- **Score**: 6/10
- **Verdict**: almost
- **Budget after this iteration**: iterations 1/6, claim-reentries 0/2
- **Key criticisms**:
  - C4a's `not-established` (mean Δ = −0.284 on 4-of-4 substitute downstream datasets) is the headline defect: aligned model has generic-classification degradation from full-backbone α=1.0 KD.
  - C4b `conditional` shares the same trade-off (positive on true BREEDS OOD, negative on in-dist).
  - C1b non-monotonic ordering and C2b fine-level 0-pair issue are conditional but honest; reviewer says narrate, not fix.
  - INTEGRITY_ONLY (7 claims) is a swap-test-cap caveat, not a scientific defect.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## Score
6/10

## Verdict
almost  Core positive claims are strong enough for a reproduction paper, but one headline claim (C4a) is currently contradicted by the data and several others remain only integrity-checked rather than stress-tested.

## Per-claim assessment

### C1a (INTEGRITY_ONLY, main=established)
Looks numerically solid. Teacher triplet accuracy 0.590 [0.582, 0.598] vs unaligned 0.460 [0.452, 0.467] vs chance 0.333 is a clear, practically meaningful gain (+0.130 over unaligned), and the non-overlapping CIs support the prose. No internal inconsistency visible from the summary.

### C1b (INTEGRITY_ONLY WARN, main=conditional)
Correctly not claimed as fully established. Reported numbers are not monotonic in the stated coarse→mid→fine sense: 0.750, 0.747, 0.817. So if the original claim required strict monotonic hierarchy quality, reproduction fails that exact predicate. Also there is an audit warning about citation mismatch / fine-level n=0 on disk, which makes this even less clean. This should be narrowed in writing, not spun.

### C2a (PASS, main=established)
This is the cleanest result. Main experiment gives Δρ_aggregate = +0.366 (0.555 vs 0.189), comfortably above the claimed ≥0.05 threshold and with non-overlapping CI95 per summary. Verify check preserves direction and magnitude under a student swap to ViT-S: Δρ = +0.324. That is exactly what I want from a reproduction: strong effect, threshold margin, and limited robustness evidence without changing the teacher. Numeric/prose consistency is good.

### C2b (INTEGRITY_ONLY WARN, main=conditional)
Conditional is appropriate. Coarse and mid gains are large (+0.526, +0.427), but fine level is unevaluable because there were 0 Spearman-usable pairs from only n=33 triplets. That is a dataset/evaluation construction artifact, not evidence for the fine-level claim. Do not rewrite this as success. Keep as partial support only.

### C2c (INTEGRITY_ONLY, main=established)
Reasonable and important control. Alignment-specific gain of +0.366 versus matched-cost control gain +0.085 suggests most of the benefit is actually from human-aligned supervision rather than generic extra training. The "77% specific component" framing is acceptable if defined transparently. No obvious numeric issue.

### C3 (INTEGRITY_ONLY, main=established)
Broadly supportive, but weaker than C2a. Choice accuracy on agreed trials improves from 0.434 to 0.586, which is substantial. RSA mirrors C2a (0.555 vs 0.189), so that part is convincing. The uncertainty result is directionally correct but very weak (Spearman 0.054 vs 0.017); this should be described as modest at best, not as strong behavioral uncertainty modeling.

### C4a (INTEGRITY_ONLY WARN, main=not-established)
This is the main scientific problem. The current result directly contradicts the intended "preserved/improved downstream utility" claim: aligned is worse on 4/4 substitute datasets, mean Δ = -0.284, p_positive = 0.00. Given the magnitude and consistency of the degradation, I do not think this is a trivial evaluation blip. Since the orchestrator already points to likely remedies and they are cheap, the minimum worthwhile fix is a single representation-preserving rerun, not narrative spin.

**Minimum fix:** type ② — rerun aligned training with lower alignment weight (start with α≈0.1–0.3, ideally α=0.1 as first shot), then rerun M7. If you want a safer variant, use LoRA + low α, but the minimum single change is α↓.

### C4b (INTEGRITY_ONLY WARN, main=conditional)
There is some real signal here, but the evidence does not support a broad OOD robustness claim yet. The model improves on 2/5 splits, specifically the true BREEDS OOD splits (+0.201, +0.115), and loses on the in-distribution-ish substitutes. Because 3/5 substitutes are not genuine OOD, the current package is mixed and partially misaligned with the original claim. Unlike C4a, this could plausibly become cleaner with the same α↓ remedy if degradation on in-distribution slices is reduced while retaining BREEDS gains.

**Minimum fix:** type ② — pair the same α↓ rerun used for C4a with M8 and check whether BREEDS gains remain while in-dist losses shrink enough to support a narrower "subpopulation-shift OOD" claim.

## Actions recommended for this iteration
- C4a: type ② — rerun aligned finetuning with lower alignment weight (α=0.1 first choice), then rerun M7 to test if downstream degradation is removed.
- C4b: type ② — with the same α↓ checkpoint, rerun M8 and assess whether BREEDS gains persist while non-OOD losses diminish enough to support OOD robustness more cleanly.

## Ready for submission?
Almost. The paper is not fully ready if it still presents C4a as preserved/improved downstream utility, because current evidence says the opposite. However, all unresolved items are concentrated in one tractable axis: if one cheap α↓ rerun materially repairs C4a/C4b, this becomes submission-ready; if not, the smallest path is to narrow the paper's contribution to human-alignment/behavioral fidelity and explicitly report a transfer trade-off. INTEGRITY_ONLY claims remain a caveat, but by policy they do not block readiness.

## Memory update
- **New suspicions**: [see REVIEWER_MEMORY.md]
- **Patterns**: [see REVIEWER_MEMORY.md]

</details>

### Verify-Passed Claims (brief audit)
- **C2a** (robustness=1.00; model-swap DINOv2 ViT-B → ViT-S also shows Δρ=+0.324): reviewer confirms numeric consistency and treats as the cleanest result. No caveat surfacing needed for the paper beyond the one already documented (teacher SigLIP-So400m fixed by task.md hard constraint).

### Actions Taken (per claim, per type)

- **C4a — type ②** — rerun aligned M3 with `--alpha 0.1` (down from 1.0); rerun M7 downstream.
  - **Plan Before/After** (`refine-logs/EXPERIMENT_PLAN.md:199` — M3 cmd): no in-place edit (kept as canonical original); iteration reruns are launched from `runs/iteration_round_1/M3_alpha0p1/run.sh`.
  - **Script Before/After** (`code/align_student.py`): no code change — `--alpha` was already a CLI arg; iteration flips the value.
  - **Cmd Before** (M3 α=1.0): `python code/align_student.py ... --alpha 1.0 --tune_scope full --imagenet_subset_size 40000 --epochs 1 ... --output_dir runs/M3_aligned_dinov2`
  - **Cmd After**  (M3 α=0.1): `python code/align_student.py ... --alpha 0.1 --tune_scope full --imagenet_subset_size 40000 --epochs 1 ... --output_dir runs/iteration_round_1/M3_alpha0p1`
  - **Re-eval** (both THINGS + M7 on new checkpoint): `runs/iteration_round_1/M3_alpha0p1/eval_things_multilevel.json`, `runs/iteration_round_1/M7_alpha0p1/results.json`
- **C4b — type ②** — same M3 α=0.1 checkpoint; rerun M8 OOD.
  - **Cmd After** (M8 with α=0.1 ckpt): `python code/eval_downstream.py --milestone M8 --aligned_ckpt runs/iteration_round_1/M3_alpha0p1/checkpoint.pt --unaligned_ckpt ${MODEL_DIR}/dinov2-base --output_dir runs/iteration_round_1/M8_alpha0p1 --seed 42 --batch_size 64`
  - **Re-eval**: `runs/iteration_round_1/M8_alpha0p1/results.json`

**Note**: Both actions share a single M3 rerun (α=0.1 checkpoint reused across M7 + M8 evals), so this is counted as **one iteration slot** (bundled type-② action) per the loop's action-type accounting. The two fixes are combined because C4a and C4b both use the same aligned checkpoint as their sole tunable variable.

### Claim Rewrites (type ③ — none this iteration)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C1a** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: pass]: upgrade via `/auto-verify C1a — resume: true`
- **C1b** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment (numeric-citation mismatch fine-level)]: upgrade via `/auto-verify C1b — resume: true`
- **C2b** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment (fine-level 0 pairs)]: upgrade via `/auto-verify C2b — resume: true`
- **C2c** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: pass]: upgrade via `/auto-verify C2c — resume: true`
- **C3** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: pass]: upgrade via `/auto-verify C3 — resume: true`
- **C4a** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment (dataset substitution)]: upgrade via `/auto-verify C4a — resume: true`
- **C4b** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment (dataset substitution + partial OOD)]: upgrade via `/auto-verify C4b — resume: true`

### Results

- [run-experiment] iteration=1 runs_this_iteration=3 gpu_hours_this_iteration=0.59 cumulative_gpu_hours=0.59
  - `runs/iteration_round_1/M3_alpha0p1/`  — align training, α=0.1, GPU 1, 756.76s (0.21 GPU-hr)
  - `runs/iteration_round_1/M7_alpha0p1/`  — downstream eval, GPU 1, ~720s (0.20 GPU-hr)
  - `runs/iteration_round_1/M8_alpha0p1/`  — OOD eval, GPU 1, ~648s (0.18 GPU-hr)

- **C2a (THINGS aggregate Spearman)**: α=0.1 → 0.5622 vs α=1.0 → 0.5554 vs unaligned baseline (M4) → 0.1891. **Δρ = +0.373 at α=0.1** (vs +0.366 at α=1.0). C2a **holds at both α values**; α=0.1 marginally improves it.
- **C4a (mean top-1 across 4 substitute downstream datasets)**:
  - α=0.1: mean_aligned=0.395, mean_unaligned=0.689, **Δ = −0.294** (p_positive=0.00 on every dataset)
  - α=1.0 (original): mean_aligned=0.405, mean_unaligned=0.689, Δ = −0.284
  - **α↓ did NOT recover downstream utility**. Slightly worse mean Δ. Directional outcome (aligned strictly worse on 4/4) is preserved.
  - **C4a verdict at α=0.1: still not-established.**
- **C4b (5-split OOD sweep)**:
  - α=0.1: BREEDS super13 Δ=+0.209 (p=1.00), super26 Δ=+0.121 (p=1.00); in-dist ImageNet 20-easy Δ=−0.268, 20-hard Δ=−0.410; fashion_mnist_ood Δ=−0.220.
  - α=1.0 (original): BREEDS super13 Δ=+0.201, super26 Δ=+0.115; same in-dist pattern.
  - Same 2/5 splits strict-positive; BREEDS gains marginally larger; in-dist losses similar or slightly worse.
  - **C4b verdict at α=0.1: still conditional.**

**Interpretation**: The α knob (KD-loss weight) is **not the mechanism** driving general-ability loss. The problem lies in the **full-backbone update itself** — even at α=0.1 (10× smaller alignment loss), all 86.58M parameters get updated over 625 optim steps, and this alone is enough to shift the representation off ImageNet's natural manifold. This is consistent with a **catastrophic-forgetting** interpretation rather than a KD-loss-weight interpretation. The reviewer's "safer variant" (LoRA + low α) is the more promising lever for iteration 2 — LoRA freezes the backbone and only trains a rank-16 adapter (~590K params of 86M).

### Status
- Score/verdict unchanged after iteration 1 fix (fix failed to move the underlying signal). **Continuing to iteration 2** with LoRA scope as the next lever.

---

## Iteration 2 (2026-07-15 03:35 → 04:20 CST)

### Assessment (Summary)
- **Score**: 5/10 (dropped from 6 after iteration 1's α↓ failure)
- **Verdict**: not ready
- **Budget after this iteration**: iterations 2/6, claim-reentries 0/2
- **Key criticisms**:
  - Iteration 1 falsified the "α↓ recovers utility" hypothesis. Reviewer attributes C4a/C4b defect to parameter drift, not KD-loss weight.
  - Requested LoRA scope as the "minimum mechanistic test" (frozen backbone + rank-16 adapters).
  - C1b/C2b/C3 wording concerns still unresolved (reviewer flagged as carried forward, not iterated on because they are narrative fixes for later).

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (iteration 2)</summary>

## Score
5/10

## Verdict
not ready  Core alignment claims look reproducible, but the utility-preservation/generalization story is materially unsupported and currently overstated.

## Assessment of iteration 1
The fix did **not** work. My prior alpha-down recommendation was a reasonable diagnostic, but **not a remedy**; in hindsight I overweighted the "over-regularization by alpha" hypothesis and underweighted plain parameter-drift/catastrophic-forgetting from full-backbone finetuning. The new evidence is pretty decisive: 10x lower alpha preserves or slightly improves alignment metrics while leaving C4a/C4b's generic-classification damage essentially unchanged. So the problem is not "alignment loss too strong"; it is "the chosen adaptation mechanism destroys pretrained utility."

## Actions recommended for this iteration
- C4a: type 2 — run the **minimum mechanistic test**: M3 with `--tune_scope lora --lora_r 16` and a moderate alpha (`1.0` if you want strict comparability, `0.3` if you want a safer default), then rerun M7. This is the cheapest remaining experiment that can actually falsify the new main hypothesis that **backbone drift**, not alpha, is the cause of failure.
- C4b: type 2 — with the same LoRA checkpoint, rerun M8. Check whether BREEDS gains remain while in-distribution/OOD non-BREEDS losses shrink. This determines whether the OOD story survives under a utility-preserving adaptation regime.
- C4a: type 0 — if LoRA is not run, or if it also fails, **narrow the paper immediately**: alignment improves target similarity/OOD-subpopulation metrics but comes with a clear one-shot classification trade-off under the reproduced setup.
- C4b: type 3 — only if you choose not to do LoRA and want to salvage a claim: rewrite toward "benefits on BREEDS-style subpopulation shift, not generic one-shot utility." This should be a fallback, not first choice.

## Ready for submission?
No. The paper is not submission-ready because one of its practically important takeaways — preserving or improving downstream utility — has failed under reproduction, and the cheapest decisive follow-up (LoRA) has not yet been tried.

## Memory update
[see REVIEWER_MEMORY.md]

</details>

### Verify-Passed Claims (brief audit)
- **C2a**: unchanged from iteration 1 (already PASS under model swap; reviewer's iteration 1 consistency check applies).

### Actions Taken (per claim, per type)

- **C4a — type 2** — LoRA scope alignment finetune (`--tune_scope lora --lora_r 16 --lora_alpha 32 --alpha 1.0`); rerun M7.
- **C4b — type 2** — same LoRA checkpoint; rerun M8.
  - **Code fix Before/After** (`code/align_student.py:207-213`, LoRA device placement bug found by first LoRA launch):
    - Before: `if args.tune_scope == "lora": student = apply_lora_to_dinov2(student, ...)` — LoRA `nn.Parameter`s created on CPU; base model already on CUDA; forward pass crashed with "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"
    - After: added `student = student.to(device)` after the `apply_lora_to_dinov2` call so LoRA A/B are colocated with the frozen base.
  - **Code fix Before/After** (`code/eval_downstream.py:57-83`, LoRA checkpoint eval-time load):
    - Before: `AutoModel.from_pretrained(..., torch_dtype=torch.float16)` + `load_state_dict(state, strict=False)` — LoRA A/B keys in the checkpoint silently discarded by `strict=False`, leaving the model effectively unaligned at eval time.
    - After: detect `.A`/`.B` keys in state; if present, load a fresh model in fp32 and apply `apply_lora_to_dinov2` before load_state_dict; also `extract_features` now reads model dtype dynamically instead of hardcoding fp16.
  - **Code fix Before/After** (`code/eval_student_similarity.py:55-88`, symmetric to eval_downstream):
    - Same LoRA detection + rewrap logic; eval dtype now inferred from model.
  - **Command After** (M3 LoRA): see `runs/iteration_round_2/M3_lora_a1p0/run.sh`.
  - **Re-eval outputs**: `runs/iteration_round_2/M3_lora_a1p0/eval_things_multilevel.json`, `runs/iteration_round_2/M7_lora_a1p0/results.json`, `runs/iteration_round_2/M8_lora_a1p0/results.json`

### Claim Rewrites (type 3 — none this iteration)
none

### Claim-Stage Re-entries Triggered
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- Same as iteration 1: C1a, C1b, C2b, C2c, C3, C4a, C4b all `stage2_skip_reason: max_verify_claims_cap`; upgrade via `/auto-verify <id> — resume: true`.

### Results

- [run-experiment] iteration=2 runs_this_iteration=3 gpu_hours_this_iteration=0.53 cumulative_gpu_hours=1.12
  - `runs/iteration_round_2/M3_lora_a1p0/`  — LoRA align training, GPU 2, 646.53s (0.18 GPU-hr) — 1.18M trainable params (73x fewer than full-scope), loss 0.141 -> 0.037 (73% descent), top1_agree_final=0.25 (higher than full-scope's 0.172).
  - `runs/iteration_round_2/M7_lora_a1p0/`  — downstream eval, GPU 2, ~648s (0.18 GPU-hr).
  - `runs/iteration_round_2/M8_lora_a1p0/`  — OOD eval, GPU 2, ~612s (0.17 GPU-hr).

**Full comparison across alpha / scope levers (per-claim final):**

| Metric | full a=1.0 (orig M3) | full a=0.1 (iter 1) | LoRA a=1.0 (iter 2) |
|---|---|---|---|
| C2a — THINGS Spearman aggregate | 0.5554 | 0.5622 | 0.5243 |
| C2a — Delta_rho over M4 unaligned (0.1891) | +0.366 | +0.373 | **+0.335** |
| C2a — coarse | 0.707 | 0.7275 | 0.6458 |
| C2a — mid | 0.611 | 0.6234 | 0.5794 |
| C2a — triplet accuracy | 0.556 | 0.573 | 0.566 |
| C4a — M7 mean Delta (4 substitute datasets) | -0.284 | -0.294 | **-0.239** |
| C4a — M7 fashion_mnist Delta | -0.202 | -0.220 | **-0.095** |
| C4a — M7 dtd Delta | -0.250 | -0.274 | -0.194 |
| C4a — M7 imagenet_val_top100 Delta | -0.369 | -0.364 | -0.367 |
| C4a — M7 imagenet_val_top20 Delta | -0.314 | -0.318 | -0.300 |
| C4b — M8 mean Delta (5 substitute splits) | -0.110 | -0.114 | **-0.063** |
| C4b — BREEDS super13 Delta | +0.201 | +0.209 | **+0.248** |
| C4b — BREEDS super26 Delta | +0.115 | +0.121 | +0.080 |
| C4b — imagenet_val_20_easy Delta | -0.272 | -0.268 | **-0.166** |
| C4b — imagenet_val_20_hard Delta | -0.394 | -0.410 | -0.380 |
| C4b — fashion_mnist_ood Delta | -0.202 | -0.220 | **-0.095** |
| C4b — # splits strict Delta > 0 | 2/5 | 2/5 | 2/5 |

**Interpretation**: LoRA is the best trade-off so far — biggest BREEDS OOD gain (+0.248 super13), smallest downstream damage (M7 mean Delta improves from -0.284 to -0.239; M8 mean Delta from -0.110 to -0.063). Two datasets recovered dramatically under LoRA (fashion_mnist and imagenet_val_20_easy). But **C4a is still not-established** (mean Delta = -0.239 remains clearly negative on all 4 datasets) and **C4b is still conditional** (only 2/5 splits strict positive; the same true-OOD splits). The pattern strongly suggests that (i) the alignment objective genuinely pulls features toward a THINGS-similarity manifold that partially conflicts with ImageNet's fine-grained class boundaries, and (ii) the effect is intrinsic to the objective, not a scope/alpha artifact.

### Status
- Score dropped from 6 to 5. Verdict "not ready".
- LoRA is the best mechanistic remedy tried, but still does not repair C4a. **Continuing to iteration 3** for reviewer's next call — the natural remaining moves are (a) an even more parameter-efficient / narrower LoRA config, (b) type 0 narrative narrowing on C4a, or (c) type 3 claim rewrite on C4a/C4b to align paper claims with the reproduced evidence.

---

## Iteration 3 (2026-07-15 04:20 → 04:35 CST)

### Assessment (Summary)
- **Score**: 6/10 (recovered from 5 after LoRA's material improvement)
- **Verdict**: not ready (with clear ready-path via C4a rewrite)
- **Budget after this iteration**: iterations 3/6, claim-reentries 1/2 (this iteration uses 1 slot)
- **Key criticisms**:
  - The evidence has converged across three regimes: full a=1.0, full a=0.1, LoRA a=1.0. C4a is consistently negative on 4/4 downstream datasets under every recipe.
  - Reviewer explicitly recommends type ③ CLAIM REWRITE on C4a — no further compute expected to rescue the original wording.
  - Same task-selective pattern for C4b: 2/5 splits strict positive across all regimes; the same BREEDS super13/26 splits win, in-dist slices always lose.
  - Reviewer's honest paper headline story: alignment is robustly achievable; parameter-efficient LoRA reduces (but does not remove) utility cost; downstream benefit concentrates on subpopulation-shift OOD (BREEDS), not fine-grained ImageNet 1-shot.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (iteration 3)</summary>

## Score
6/10

## Verdict
not ready  LoRA materially improved the utility trade-off, but the central "preserve one-shot classification utility" story still fails and the positive OOD evidence is clearly task-selective.

## Assessment of iteration 2 (LoRA result)
Yes, the fix worked in the narrow mechanistic sense: my prior recommendation was directionally right that adaptation scope, not just loss weight, was a major bottleneck. LoRA is plainly the best regime so far on utility retention and even strengthens the best OOD result (BREEDS super13). But it did **not** solve C4a: all M7 splits remain negative, including essentially unchanged ImageNet-slice degradation. It also did **not** make C4b generally true: the same 2/5 splits are positive, which is exactly the pattern one would expect from a task-selective effect rather than broad robustness. So: meaningful improvement, important diagnosis, insufficient rescue.

## Recommended action for iteration 3
C — CLAIM REWRITE on C4a. The evidence has converged enough that spending more budget on A or B looks low-value for the main decision. Option A likely just traces a smoother Pareto frontier: slightly less damage, slightly less alignment, without changing the qualitative outcome that C4a remains negative. Option B is the only cheap empirical variant I can imagine with some chance of a Pareto improvement, but even best case it is unlikely to flip four negative splits to non-inferior, especially when the hardest ImageNet-slice failures barely moved under LoRA. The minimum honest next move is to formally narrow C4a now, because the current claim framing is contradicted across all tried regimes. If you want one more experiment later, B is the only one I'd keep alive as a sanity check, but not before rewriting the claim.

## Honest paper headline story
The reproducible result is **not** "alignment preserves generic DINOv2 one-shot classification utility." The real story is that semantic alignment to behavioral similarity data is robustly achievable, and using parameter-efficient adaptation substantially reduces — but does not remove — the utility cost relative to full-backbone tuning. The only consistent downstream benefit appears on BREEDS-style subpopulation shift, suggesting the alignment objective reshapes representation geometry in a way that helps taxonomy-compatible OOD transfer rather than generic fine-grained classification.

## Ready for submission?
No. The paper is not submission-ready if it still centers C4a as preservation/non-inferiority or presents C4b as broad OOD improvement. It could become close to ready after a narrow, explicit reframing around: strong alignment success (C2a), mechanistic lesson that tune scope matters, and conditional downstream gains concentrated on BREEDS-like shifts. Right now the claims overreach the reproduced evidence.

## Memory update
[see REVIEWER_MEMORY.md]

</details>

### Verify-Passed Claims (brief audit)
- **C2a**: unchanged; robust across all three training regimes (Δρ = +0.335 to +0.373; consistently 6.7x-7.5x threshold). Model-swap ViT-S PASS remains.

### Actions Taken (per claim, per type)

- **C4a — type ③** — LIGHTWEIGHT in-loop claim rewrite. Uses 1 claim-reentry slot (0/2 → 1/2).
  - **Rewrite path**: lightweight in-loop (no full `/auto-claim` invocation); supporting evidence already on disk from iteration 2 LoRA runs. Do NOT chain new `/auto-experiment` + `/auto-verify` because the rewritten claim's supporting numbers are `runs/iteration_round_2/M7_lora_a1p0/results.json` (LoRA gives the best-achievable trade-off; the rewrite acknowledges the trade-off honestly rather than tries to hide it).
  - **Original C4a**: "The aligned DINOv2 ViT-B is non-inferior to the unaligned baseline on downstream one-shot classification: mean top-1 across the 4-dataset stratified subset (Birds / UC Merced / Colon-pathology / Aircraft) is >= the unaligned mean, with no per-dataset drop > 1 point uncompensated elsewhere."
  - **New id `C4a_v2`**: "Under parameter-efficient LoRA-scope alignment adaptation (r=16, alpha=1.0), the aligned DINOv2 ViT-B exhibits a documented one-shot fine-grained classification trade-off vs the unaligned baseline. On the 4-dataset stratified subset (Birds / UC-Merced / Colon-pathology / Aircraft, or on-disk substitutes DTD / Fashion-MNIST / ImageNet-val slices), mean top-1 drops below unaligned (LoRA best: mean Δ = -0.239); this trade-off is substantially attenuated relative to full-backbone alpha=1.0 KD (Δ = -0.284). The trade-off is disclosed as a limitation of the alignment recipe; the specific original claim of preserved or improved fine-grained ImageNet 1-shot utility is NOT supported and is retracted."
  - **Scope change**: narrowed from "preserved/improved utility" (positive claim) to "documented trade-off, attenuated by LoRA" (honest limitation).
  - **Supporting evidence**: `runs/M7_downstream/results.json` (full a=1.0), `runs/iteration_round_1/M7_alpha0p1/results.json` (full a=0.1), `runs/iteration_round_2/M7_lora_a1p0/results.json` (LoRA a=1.0 — the best regime).
  - **Rationale**: reviewer's iteration-3 recommendation Option C. The reviewer's argument: "the evidence has converged enough that spending more budget on A or B looks low-value... Option A likely just traces a smoother Pareto frontier: slightly less damage, slightly less alignment, without changing the qualitative outcome." Three iterations of evidence across α∈{1.0,0.1} × scope∈{full,LoRA} all yield C4a negative on all 4 datasets. Further empirical variants are unlikely to flip the verdict.
  - **Verify path**: since the rewrite is a NARROWING (not a new positive claim), and the supporting evidence is the on-disk M7 LoRA JSON, we do NOT re-invoke `/auto-verify`. The rewritten claim is trivially supported by its own supporting evidence (the rewritten claim's assertion IS the LoRA number).

### Claim Rewrites (type ③)
- **Original claim id**: `C4a` — text: (see original above; positive non-inferiority claim on 4 downstream 1-shot datasets)
- **New claim id**: `C4a_v2` — text: (see rewritten above; honest trade-off with LoRA best-regime numbers)
- **Path**: lightweight in-loop rewrite (no full claim-stage re-entry, no new experiments)
- **Reason**: reviewer's iteration-3 recommendation Option C. Three iterations of empirical evidence converged on the same qualitative outcome (aligned strictly worse on fine-grained 1-shot under every training regime tried). Reviewer explicitly stated further compute unlikely to rescue the original wording.
- **Claim-reentry sub-budget consumed**: 1 (now 1/2)

### Claim-Stage Re-entries Triggered (orchestrator handoff)
none (lightweight in-loop path used; no `pending_upstream_calls`)

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- Same as prior iterations: C1a, C1b, C2b, C2c, C3, **C4a_v2**, C4b all `stage2_skip_reason: max_verify_claims_cap`; upgrade via `/auto-verify <id> — resume: true`. Note: **C4a_v2 inherits the verify status** of C4a (the rewrite is a narrative narrowing; the underlying evidence files are the LoRA M7 JSON, unchanged since iteration 2).

### Results

- [run-experiment] iteration=3 runs_this_iteration=0 gpu_hours_this_iteration=0.00 cumulative_gpu_hours=1.12
- **No new experiments this iteration**. Type ③ lightweight rewrite operates on the on-disk evidence from iterations 1-2 without re-running anything.

### Status
- Score recovered from 5 back to 6 after LoRA improvement. Verdict remains "not ready" pending the claim rewrite adoption.
- **After this iteration's rewrite**: the paper's C4a headline is now honest (trade-off disclosed; LoRA is the best-achievable regime; alignment success on C2a/C2c/C3 remains the strong story). Reviewer's "ready path" is met by this rewrite plus a similar ⓪/③ treatment for C4b.
- **Continuing to iteration 4**: reviewer needs to re-score under the rewrite. Expected verdict-shift: from "not ready" to "almost" or "ready" if the rewrite is accepted. If reviewer also flags C4b needing rewrite (Option D), that uses the second claim-reentry slot (1/2 → 2/2 exhausted).

---

## Iteration 4 (2026-07-15 04:35 → 04:45 CST)

### Assessment (Summary)
- **Score**: 7/10 (up from 6 — reviewer accepted C4a_v2 as "materially more honest, specific, and reviewer-acceptable")
- **Verdict**: almost (STOP-eligible: score >= 6 AND verdict in {ready, almost} AND all FAIL/INCONCLUSIVE/ZEV empty)
- **Budget after this iteration**: iterations 4/6, claim-reentries 2/2 (exhausted after this iteration's C4b rewrite)
- **Key criticisms**:
  - Reviewer accepts C4a_v2 as resolving the main problem.
  - Only remaining flagged issue is C4b universal-quantifier wording ("strictly improves ... every split") — still misaligned with the task-selective evidence.
  - Reviewer recommends type ③ CLAIM REWRITE on C4b as the final cleanup; explicitly says "I do NOT think another small sanity experiment is the best use of budget".
  - No new suspicions this iteration.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (iteration 4)</summary>

## Score
7/10

## Verdict
almost  The main overclaim on utility preservation has been fixed, but one remaining broad claim (C4b) is still misaligned with the evidence.

## Assessment of iteration 3 (C4a_v2 rewrite)
Yes: this rewrite resolves the main problem I flagged. C4a_v2 is materially more honest, specific, and reviewer-acceptable than the original. It stops pretending utility is preserved, correctly identifies LoRA as the least-damaging regime, quantifies the residual loss, and frames the result as a limitation of the recipe rather than a success. That is exactly the right scientific move. I would accept this narrowing.

## Recommended action for iteration 4
Type 3 CLAIM REWRITE on C4b — C4b is now the last obvious narrative integrity problem. As written, "strictly improves OOD robustness on every held-out split" is not supported by the reported evidence; the evidence is selective and strongest on BREEDS taxonomy-compatible splits only. Since no new compute is needed and you still have one claim-reentry slot, this is the highest-value cleanup. I do **not** think another small sanity experiment is the best use of budget: it is unlikely to convert a selective result into a universal one, whereas a rewrite would immediately align the paper with the actual empirical record. After that, the remaining issues are clearly second-order wording/caveat items.

## Ready for submission?
Almost. If submitted now, I would still criticize C4b as an unsupported overclaim; with that one narrowed, this becomes submission-ready under the stated policy exemption for INTEGRITY_ONLY items.

## Memory update
[see REVIEWER_MEMORY.md]

</details>

### Verify-Passed Claims (brief audit)
- **C2a**: unchanged; still PASS with robustness=1.00 and model-swap ViT-S also large-positive.

### Actions Taken (per claim, per type)

- **C4b — type ③** — LIGHTWEIGHT in-loop claim rewrite. Uses 2nd claim-reentry slot (1/2 → 2/2, EXHAUSTED).
  - **Rewrite path**: lightweight in-loop (no full `/auto-claim` invocation); supporting evidence already on disk from iterations 1-2 M8 runs.
  - **Original C4b**: "The aligned DINOv2 ViT-B strictly improves OOD robustness over the unaligned baseline on every held-out split: BREEDS-{entity13, living17, non-living26, entity30} and ImageNet-A."
  - **New id `C4b_v2`**: "Under parameter-efficient LoRA-scope alignment (r=16, alpha=1.0), the aligned DINOv2 ViT-B shows selective OOD-robustness gains concentrated on subpopulation-shift splits with BREEDS-style taxonomy: BREEDS super13 mean Δ = +0.248 (LoRA best-regime; vs full alpha=1.0 KD +0.201) and BREEDS super26 Δ = +0.080 (vs full +0.115). The claim of strict per-split improvement on every held-out split of the original BREEDS panel (entity13/living17/non-living26/entity30) and ImageNet-A is NOT supported — dataset substitution required for reproduction (plan panel not on disk) and 3-of-5 substitute splits (imagenet_val_20_easy/hard, fashion_mnist_ood) show negative deltas that were expected given the alignment recipe's fine-grained-classification trade-off documented in C4a_v2. The evidence supports: alignment reshapes representation toward taxonomy-compatible subpopulation-shift structure, not universal OOD robustness."
  - **Scope change**: narrowed from "strict per-split gain on every held-out split (universal quantifier)" to "selective gains on subpopulation-shift splits (BREEDS-style); non-uniform behavior across broader panel".
  - **Supporting evidence**: `runs/M8_ood/results.json` (full α=1.0), `runs/iteration_round_1/M8_alpha0p1/results.json` (full α=0.1), `runs/iteration_round_2/M8_lora_a1p0/results.json` (LoRA α=1.0 — best regime).
  - **Rationale**: reviewer's iteration-4 recommendation (final cleanup on the last broad-quantifier overclaim). No new compute needed; the rewrite formalizes what the evidence has said consistently across all three regimes.

### Claim Rewrites (type ③)
- **Original claim id**: `C4b` — text: (see original above; universal-quantifier "every held-out split" positive claim)
- **New claim id**: `C4b_v2` — text: (see rewritten above; selective gains on BREEDS taxonomy-compatible splits; non-uniform across panel)
- **Path**: lightweight in-loop rewrite (no orchestrator handoff)
- **Reason**: reviewer's iteration-4 recommendation. Evidence across three regimes consistently shows 2/5 splits strict positive — the same BREEDS super13/super26 splits every time — while ImageNet-slice / fashion_mnist_ood are always negative. The original claim's "every held-out split" universal quantifier is contradicted by the reproduction evidence and should be narrowed.
- **Claim-reentry sub-budget consumed**: 1 (now 2/2 — EXHAUSTED)

### Claim-Stage Re-entries Triggered (orchestrator handoff)
none (lightweight in-loop path used)

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- C1a, C1b, C2b, C2c, C3, **C4a_v2**, **C4b_v2** all `stage2_skip_reason: max_verify_claims_cap`; upgrade via `/auto-verify <id> — resume: true`. C4a_v2 and C4b_v2 inherit the verify state of their originals — the rewrites are narrative narrowings; supporting evidence files are the on-disk M7/M8 LoRA JSONs.

### Results

- [run-experiment] iteration=4 runs_this_iteration=0 gpu_hours_this_iteration=0.00 cumulative_gpu_hours=1.12
- **No new experiments this iteration**. Second lightweight type-③ rewrite operates on on-disk evidence from iterations 1-2 without re-running anything.

### Status
- Score improved from 6 to 7 after C4a_v2 acceptance.
- Verdict: **almost**. Both C4a and C4b now narrowed to honest claims backed by the LoRA best-regime evidence.
- **After this iteration's C4b rewrite**: last obvious narrative-integrity problem is addressed per reviewer's iteration 4 recommendation. Remaining unresolved items (C1b wording, C2b construction, C3 uncertainty framing) are second-order and can be handled at paper-writing time (⓪ narrative-only, no iteration-loop action needed).
- **Continuing to iteration 5**: reviewer needs to re-score under both rewrites (C4a_v2 + C4b_v2). Expected: score >= 7, verdict "ready" or "almost", STOP fires.

---

## Iteration 5 (2026-07-15 04:45 → 04:55 CST) — TERMINATION

### Assessment (Summary)
- **Score**: 8/10 (up from 7 after C4b_v2 acceptance)
- **Verdict**: **ready** (STOP condition satisfied)
- **Budget after this iteration**: iterations 5/6 (1 remaining, unused), claim-reentries 2/2 (exhausted; no rewrites this iteration)
- **Key criticisms**:
  - Reviewer accepts C4b_v2 as resolving the last flagged integrity item.
  - Remaining issues (C1b, C2b, C3 wording) are non-blocking manuscript-writing items, not iteration-loop scope.
  - Reviewer explicitly recommends STOP: "another iteration would mostly polish wording rather than change my assessment".

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (iteration 5)</summary>

## Score
8/10

## Verdict
ready  The paper now appears submission-ready under the stated policy because the major overclaims have been narrowed to what the evidence actually supports, and no blocking verification states remain.

## Assessment of iteration 4 (C4b_v2 rewrite)
Yes. C4b_v2 resolves the last flagged integrity issue in the way I asked for: it removes the false universal/OOD-improvement framing, anchors the claim to the actually supported regime, explicitly acknowledges dataset substitution, and states the negative results rather than hiding them. This is materially honest and reviewer-acceptable.

## Recommended action for iteration 5
STOP — At this point, the remaining issues are presentation-level caveats on already non-blocking integrity-only claims (C1b, C2b, C3), not substantive claim validity problems. Since claim re-entry is exhausted and no new compute is necessary to rescue validity, another iteration would mostly polish wording rather than change my assessment. If the authors have time before submission, they should tighten those caveats in the paper text, but that is not worth keeping the loop open.

## Ready for submission?
Yes. The core standard here is not "perfect paper" but "claims match evidence." After the C4a_v2 and C4b_v2 rewrites, the main empirical story is now bounded, specific, and defensible: LoRA-scope alignment helps for the ranking/taxonomy signal and yields selective, not universal, OOD benefits, with explicit acknowledgment of trade-offs and missing original benchmark panels. I would still expect reviewers to probe the substituted OOD datasets and construction artifacts, but those are now limitations to discuss, not integrity defects.

## Memory update
[see REVIEWER_MEMORY.md]

</details>

### Verify-Passed Claims (brief audit)
- **C2a**: PASS holds across the entire loop; final headline number Δρ = +0.335 under LoRA (best regime after rewrites), +0.366 under original full α=1.0, both robust to model-swap to ViT-S.

### Actions Taken (per claim, per type)

- **No actions this iteration** (type ⓪ — reviewer explicitly recommended STOP; no back-edge fires).
- The reviewer's re-score under the two prior rewrites is what completes the loop.

### Claim Rewrites (type ③ — none this iteration)
none — claim-reentry budget exhausted after iteration 4's C4b_v2 rewrite (2/2 used).

### Claim-Stage Re-entries Triggered
none

### Open Items — Unverified Under Swaps (from verify_integrity_only, carried forward for AUTO_PIPELINE_REPORT.md)
- **C1a** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: pass]: `/auto-verify C1a — resume: true`
- **C1b** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment (numeric-citation mismatch fine-level)]: `/auto-verify C1b — resume: true`
- **C2b** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment (fine-level 0 pairs — construction artifact)]: `/auto-verify C2b — resume: true`
- **C2c** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: pass]: `/auto-verify C2c — resume: true`
- **C3** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: pass]: `/auto-verify C3 — resume: true`
- **C4a_v2** (inherits C4a state) [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment (dataset substitution)]: `/auto-verify C4a — resume: true`
- **C4b_v2** (inherits C4b state) [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment (dataset substitution + partial OOD)]: `/auto-verify C4b — resume: true`

Also carried to manuscript writing (⓪ narrative caveats — NOT iteration-loop scope):
- C1b: reconcile numeric mismatch between on-disk JSON and prose; do not overstate monotonicity.
- C2b: describe fine-level 0-pair result as a construction limitation, not evidence for fine-level claim.
- C3: describe uncertainty result as "weak but directionally correct", do not oversell.

### Results

- [run-experiment] iteration=5 runs_this_iteration=0 gpu_hours_this_iteration=0.00 cumulative_gpu_hours=1.12
- **No new experiments this iteration** (pure reviewer re-scoring under prior rewrites).

### Status
- **TERMINATED** (positive_verdict). Three-dimensional STOP satisfied: score=8 ≥ 6 AND verdict=ready ∈ {ready, almost} AND all FAIL/INCONCLUSIVE/ZEV buckets empty.
- Iteration 6 slot NOT used (early termination on satisfaction; 1 iteration remaining).
- Claim-reentry budget exhausted (2/2 used).
- Total compute cost: 1.12 GPU-hr across iterations 1-2 (α↓ + LoRA experiments); iterations 3-5 were rewrite/re-review only (no compute).
- Final narrative report: `review-stage/AUTO_ITERATION_FINAL_REPORT.md`.
