# FINAL PROPOSAL — RFM Concept-Vector Steering & Monitoring on Llama-3.1-8B-Instruct

**Mode**: `BEHAVIOR_SOURCE = given` × `MECHANISM = discovery`
**Language**: English (matches `task.md`)
**Not stamped**: `resource_fidelity: strict` (this is not the reproduction combo `given + given`); `chosen_mechanism:` (MECHANISM=discovery — the experiment stage commits the family via `/mechanism-skills`).

---

## Problem

Modern LLMs encode a large amount of concept knowledge in their residual-stream activations, yet extracting *specific* concept directions on demand — and deploying them at scale — has been fragmented:

- **Unsupervised methods** (SAEs, dictionary learning) cannot be steered to surface a *chosen* concept of interest.
- **Supervised mean-difference methods** (CAA, RepE, ActAdd) work for well-attested behaviors (refusal, sycophancy) but are documented to be brittle on subtler concepts (`LANDSCAPE.md` Papers #4/#17/#18).
- **Linear-probe methods** (CAV, ITI) recover a direction but usually per-behavior artisanally, not as a uniform *pipeline* that works for hundreds of concepts.

`task.md` posits that **Recursive Feature Machines (RFM)** — a supervised, nonlinear feature learner (kernel machine + Average Gradient Outer Product / AGOP reweighting) whose *output* is nevertheless a **single linear direction per block** (top AGOP eigenvector) — resolves this: it is fast (per public reports, under a minute per concept), universal (the same recipe over 512 concepts / 5 concept classes), and dual-use (the same vector serves both **steering** — added additively at inference — and **monitoring** — used as a linear classifier on activations).

The reproduction target is to demonstrate all five capabilities on the fixed base model `Llama-3.1-8B-Instruct` (32 blocks) within a 10-hour GPU budget, using the specified GPT-4o judge and public benchmarks (HackerRank, HaluEval, ToxicChat).

## Approach

The mechanism strategy is **Tuning & Editing → Location** (see `IDEA_REPORT.md § Mechanism Strategy`). Concretely, every claim decomposes into a common four-step recipe with a divergence only at the final step:

1. **Screen (Location sub-direction)** — For each target concept, sweep all 32 residual-stream blocks of `Llama-3.1-8B-Instruct`; for each block, extract per-token or per-sequence residual activations on a paired concept-positive / concept-negative dataset; fit a linear-probe accuracy score. Pick the best block(s) per concept.
2. **Extract (RFM step)** — At the chosen block, run the RFM procedure: alternate between fitting a kernel-machine predictor and reweighting features via AGOP; the top eigenvector of the final AGOP is the per-block linear concept direction `v_c`.
3. **Intervene (Tuning & Editing step)** — Additive residual-stream steering: `h_l ← h_l + α · v_c` at the chosen block `l`, applied at every generated-token position after the prompt. Sweep `α ∈ {-3, -2, -1, 0, +1, +2, +3}` on a dev split; pick the best-effect-without-degradation α on a held-out split.
4. **Evaluate (task scorer)** — C1: GPT-4o (`gpt-4o-2024-11-20`) rubric-judge on 50 held-out prompts per concept + matched-random-direction control. C2: test-case pass rate on HackerRank problems in C++ vs Python. C3: same GPT-4o judge on translated (ZH/FR/ES) prompts. C4: two-vector combo `α₁·v_{c₁} + α₂·v_{c₂}`, judge each target effect independently.

**C5 diverges at step 3–4**: instead of injecting the vector, use `⟨h_l, v_c⟩` (or a fitted linear probe on the same block's activations) as a **classifier score** on the model's residual stream while it processes a candidate output — compute AUROC vs a labeled hallu/tox split.

This one recipe covers all five claims; the only per-claim differences are (i) the concept-pair dataset, (ii) the chosen block, and (iii) the final scorer. Steering coefficient α is tuned per (concept, block) on a dev split — this is the mandatory dose-response control (`experiment-tips` steering-coefficient tip).

## Claims

> **Claim-status update (iteration 2 of `/auto-iteration-loop`, 2026-07-15)**: after empirical results from experiment + verify + iteration-1 stress-tests, the claim set is restructured into ONE headline finding (C5, narrowed and rewritten as `C5_v2`) plus FOUR exploratory / preliminary sub-findings (`C1_exp`, `C2_exp`, `C3_exp`, `C4_exp` — demoted from headline status per reviewer feedback in iteration 2). Rationale is documented in `review-stage/AUTO_REVIEW.md § Iteration 2` and `review-stage/REVIEWER_MEMORY.md`. The original C1-C5 claim ids remain as historical references below.

### Headline claim (post-iteration-2 rewrite)

- **`C5_v2`** (internal-feature monitoring beats LLM judge — **narrowed to RLHF-tuned 8B instruct models with per-benchmark scope**) —
  > **Factuality (HaluEval-General)**: RFM-based / linear-probe internal-feature monitors on 8B instruct-LLM residual-stream activations beat GPT-4o-2024-11-20 (used as a black-box output judge) in AUROC on HaluEval-General, robustly across at least 4 tested 8B models spanning both RLHF-tuned (Llama-3.1-8B-Instruct, Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2) and reasoning-distilled (DeepSeek-R1-Distill-Llama-8B) training regimes.
  >
  > **Toxicity (ToxicChat)**: The same internal-feature monitors beat GPT-4o-2024-11-20 on ToxicChat for the tested **RLHF-tuned 8B instruct models specifically** (Llama-3.1, Meta-Llama-3, Mistral-Instruct — all 3 pass), and this advantage does NOT extend to the tested reasoning-distilled non-RLHF model (DeepSeek-R1-Distill-Llama-8B fails ToxicChat by −0.024 AUROC). The boundary condition is stated explicitly: the toxicity advantage of internal monitors over GPT-4o judges is **contingent on RLHF-style harmlessness fine-tuning being present in the small monitored model**.
  >
  > `ToxicChat-T5-Large` (a specialised, in-domain fine-tuned toxicity classifier) hits AUROC 1.00 on ToxicChat and is not beaten by the internal 8B monitors; this is expected and reported as a task-specific reference, not the primary claim.

**Rationale for the rewrite (from iteration-2 reviewer):** The original C5 wording ("beat GPT-4o … on both benchmarks") was falsified by the DeepSeek variant on ToxicChat. Two additional RLHF-tuned model swaps in iteration 1 (Llama-3-8B-Instruct on the same Llama arch; Mistral-7B-Instruct-v0.2 on a different arch family) demonstrated 2/2 RLHF variants pass both benchmarks strictly, while the 1 non-RLHF variant fails ToxicChat. The rewritten claim states these boundaries explicitly rather than relegating them to a footnote. Robustness under the narrowed frame is 2/2 RLHF-tuned = 100% on both benchmarks; robustness under the broader "any 8B model" frame is 2/3 = 0.667. The paper adopts the narrowed frame.

### Exploratory / preliminary sub-findings (demoted from headline status)

The following four items were originally listed as headline claims (C1–C4) but each has documented substantive threats to validity (see `verify/INTEGRITY_AUDIT.md` and iteration-2 reviewer feedback). Per iteration-2 reviewer requirement, they are demoted from headline contribution status and re-framed as **exploratory / preliminary case studies** for a paper-appendix section (not the abstract or headline results). Their bold-claim originals are removed from the contribution list.

- **`C1_exp` (exploratory)** — On `Llama-3.1-8B-Instruct`, per-block RFM concept vectors produce a measurable, controlled shift in the political-stance direction (α=-3→3.08, α=+3→4.02, range 0.94 rubric points versus matched-random-control 0.30/0.12 at same |α·v|). This provides a **case study** demonstrating RFM extracts semantically meaningful directions for concepts that are (a) well-attested in the training data and (b) not defended by safety fine-tuning. The two other originally-bundled concepts do NOT support the claim as originally stated: **refusal** — the block-selection procedure is degenerate on this dataset (all 32 blocks tie at probe accuracy 1.0 on assistant-text; auto-argmax picked block 0; pinned block-14 re-run showed no jailbreak up to α=+12); the correct reading is that this dataset does not admit meaningful location-screen inference on this model. **Honesty** — Δ=+0.14 at α=+3 versus baseline 4.26 is within GPT-4o judge noise (baseline std=1.25). Both are documented as **null / non-diagnostic case studies**, not evidence for the bundled steering claim.

- **`C2_exp` (exploratory / negative)** — On the 10 held-out HackerRank problems (originally planned 30; only 20 available in `eval_set.jsonl` locally, split into 10 dev + 10 held-out × 2 seeds), the RFM C++ concept vector at α=+3 (best on the 10 dev problems) **did not switch generation language** (`cpp_frac=0.00` — outputs remain Python) and its pass rate (0.567) is BELOW the default-Python baseline (0.60). This is a **negative / non-diagnostic finding**: the vector appears to encode the language-tag prefix ("Language: X.\n") in the training data rather than an intrinsic C++ representation, and additive steering at reasonable α cannot induce a mid-generation language switch. Statistical power is under-powered relative to the plan (n=10 vs 30). Presented as an exploratory negative case, not a supported claim.

- **`C3_exp` (exploratory / partial)** — On the reused honesty vector, the direction shift is **positive in 3 of 4 tested languages** (EN +0.20, ZH +0.32, ES +0.20) but **reverses sign in French** (−0.10). No language reaches p<0.05 at n=50/language (no multiple-comparison correction applied, which if applied would further widen). This is presented as an exploratory transfer case study, NOT as a "transfers to ZH/FR/ES" claim. The FR reversal is documented as a boundary condition worth understanding. Underlying signal weakness in the honesty vector itself (Δ=+0.14 monolingually on EN) likely explains the sub-significance in every language.

- **`C4_exp` (exploratory / non-diagnostic due to ceiling)** — Two 2-vector combinations tested (honesty + refusal-neg; formal-tone + technical-persona) at α=(1,1). On the tested prompt sets both primary rubrics saturate at ~5.0 for single vectors, leaving no headroom to detect compositional gain. The compositionality question is **untested here**, not falsified — the experiment as designed is non-diagnostic. Missing matched-random-combination direction control (present for C1). Presented as an exploratory finding with an acknowledged experimental-design flaw; re-testing on non-saturating prompt sets with random-combo controls is left to future work.

### Headline contribution (post-rewrite)

The paper's headline contribution is now **C5_v2 alone**: a strong internal-monitor result for hallucination detection that generalizes across 8B model architectures and training regimes, plus a regime-bounded toxicity-monitor result for RLHF-tuned instruct models with the boundary condition (RLHF vs reasoning-distilled) explicitly stated and empirically demonstrated. The four exploratory case studies live in an appendix and are honestly framed as preliminary / null / non-diagnostic.

## Evaluation

| Claim | Dataset / Prompts | Baselines | Scorer | Success predicate |
|-------|-------------------|-----------|--------|-------------------|
| C1 | 3 concepts × 200-pos / 200-neg paired train + 50 held-out prompts each. Refusal from Harmful/Harmless instructions; political from generated pairs (DMX gpt-5.4); honesty from TruthfulQA. | Unsteered baseline; matched-random-direction control at same ‖α·v‖. | GPT-4o (`gpt-4o-2024-11-20`) rubric on 5-point steering-effect scale. | Mean rubric shift > baseline & > random control at chosen α (dev-selected). |
| C2 | HackerRank 50-problem subset (`task.md § Verify stage`). 20-problem dev split for α sweep, 30 held-out for the reported number. | (a) Default (Python), (b) prompt-only "Answer in C++." | Automatic test-case pass rate via a language-appropriate execution harness (C++ compiled with g++ -O2, Python via CPython). | Steered > max(default, prompt-only) by > run-to-run noise band. |
| C3 | Reuse C1 (honesty or refusal); 50 EN prompts translated once via GPT-4o into ZH/FR/ES. | Unsteered baseline in each target language. | GPT-4o multilingual judge (same rubric as C1). | Steering effect direction preserved (sign + significant magnitude) in each of ZH/FR/ES. |
| C4 | 20 prompts × 2 combos: (i) `honesty + refusal-negative`, (ii) `formal-tone + technical-persona`. | Each single vector alone; unsteered. | GPT-4o judged **twice** per output, one for each target concept independently. | Both target-concept rubrics improved simultaneously beyond either single-vector baseline. |
| C5 | HaluEval-General 2000-sample subset; ToxicChat 1000-sample subset. | (a) GPT-4o judge on outputs; (b) `ToxicChat-T5-Large` (toxicity only); (c) linear-probe (non-RFM) baseline. | AUROC (HaluBench convention). | RFM-monitor AUROC strictly greater than GPT-4o judge AUROC on each benchmark. |

**Cross-claim controls (from `experiment-tips` General Rule for mechanism/interpretability + steering-coefficient tip + `data-rule`)**:
- Every claim uses a **matched-random-direction control** (or an unsteered baseline) so that any observed effect is not attributable to activation noise or ‖α·v‖ magnitude alone.
- α is dose-swept on a **dev split** and locked before the reported held-out numbers.
- Every dataset has an explicit **train / dev / held-out** split; probe accuracy for block selection is measured on the dev split, never on held-out.
- Sample sizes exceed the `data-rule` floor: ≥50 held-out per condition for judged evaluations, ≥1000 for AUROC evaluations.

## Compute

**Budget**: 10 GPU-hours on 4×H100 (GPU IDs 0–3, per HARD constraints). `Llama-3.1-8B-Instruct` FP16 fits in ~16 GB on a single H100 — parallelizable across the 4 GPUs.

| Phase | GPU-hours (est) |
|-------|-----------------|
| Screen (32 blocks × ~15 concepts × ~400 paired sequences) | ~2.0 h |
| Extract (RFM at chosen block × ~15 concepts) | ~0.5 h |
| C1 steer + judge (3 concepts × 7 α × 50 prompts × 2 controls) | ~1.5 h |
| C2 steer + compile/test (50 HackerRank × ~7 α on dev + 3 conditions on held-out) | ~1.5 h |
| C3 cross-lingual (2 concepts × 3 langs × 50 prompts) | ~0.5 h |
| C4 compositional (2 combos × 20 prompts + α₁α₂ mini-sweep) | ~0.5 h |
| C5 monitor (HaluEval 2000 + ToxicChat 1000 forward passes + probe fit) | ~1.5 h |
| Judge calls (GPT-4o via DMX API — no GPU) | 0 h |
| Buffer / re-runs | ~2.0 h |
| **Total** | **~10 h** |

The env constraint is **conda** (per HARD `## Notice`). All models / datasets live under `/data/zhenqian/models` and `/data/zhenqian/data` (per HARD directory constraint); symbolic links from the working directory are used to reference them.

**Blind-reproduction posture**: `.claude/forbidden-urls.txt` voids the target paper's identifier and code repository; every design choice above is derived from foundational pre-cutoff works (RepE, CAA, ActAdd, ITI, SAPLMA, CAV, RFM algorithmic reference) enumerated in `RESEARCH_LIT.md`.
