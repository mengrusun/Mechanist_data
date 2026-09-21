# Final Proposal — Unified Verification Suite for the Five Claims in `task.md`

**Date**: 2026-07-15
**Behavior-source**: given (five claims from `task.md § Claim` — verbatim, 1:1)
**Mechanism**: discovery (system routes the family via `/mechanism-skills` later; strategy fixed here)
**Claim source**: `task.md § Claim` — see `idea-stage/IDEA_REPORT.md` for the verbatim capture with faithfulness audit.

## Top Metadata (machine fields — English, verbatim)

```yaml
mechanism_strategy:
  directions: [Location, Causal Intervention]
  rejected:
    - Tuning & Editing — Claim 5's harmfulness-direction probe is a diagnostic monitor, not a downstream-capability edit.
    - Formation Tracing — task.md makes no origin claim (how/when the two directions form during safety fine-tuning).
    - Unit Interpretation — the directions are already named by contrastive construction (harmfulness / refusal).
    - Decision Auditing — Claim 5's evaluation is a straight AUROC/FPR comparison, not a per-decision evidence audit.
  note: Localize the two candidate residual-stream directions at their respective token positions (Claims 1–2), then causally dissociate them via additive steering with dose-response + specificity control (Claim 3), leverage the dissociation to explain a class of jailbreaks (Claim 4), and use the harmfulness axis as a lightweight monitor (Claim 5).

# No resource_fidelity marker (given + mechanism:discovery is cost-aware, not the reproduction combo).
# No M0 milestone (given behavior — mechanism milestones do NOT declare depends_on: [M0]).
```

## Problem Anchor
`task.md` names one research hypothesis — *harmfulness perception and refusal execution are two dissociable linear directions in the residual stream of an instruction-tuned LLM* — and states five claims (existence + linearity + independence; position dissociation; causal dissociation via additive steering; jailbreak signature; a lightweight diagnostic). The five claims are the deliverable; the pipeline below is the smallest verification suite that lands them within the HARD 10-hour GPU budget on Llama-3-8B-Instruct + AdvBench.

## Method Thesis (one sentence)
Extract h (harmfulness) at the final instruction-token position and r (refusal) at the position immediately post-instruction via difference-in-means on Llama-3-8B-Instruct residual-stream activations for paired (AdvBench-harmful, matched Alpaca-benign) prompts; run a five-milestone pipeline of correlational probes (M1, M2), causal additive steering with a specificity control (M3), paired-attack projection deltas across two attack families (M4), and a head-to-head classifier comparison at matched FPR against Llama Guard 3 8B (M5).

## Dominant Contribution
A **claim-driven verification** of the two-direction decomposition on Llama-3-8B-Instruct that:
1. Localises h and r at two specific token positions using the standard difference-in-means recipe (Arditi 2024 / CAA / RepE — Location direction).
2. Cleanly dissociates them causally with additive steering + dose-response + specificity control (Causal Intervention direction), instead of just reporting a single-direction ablation.
3. Confirms the *refusal-suppressed / harmfulness-preserved* signature on paired successful-jailbreak samples across **both** an optimisation-based (GCG) and a persuasion-based (PAP) attack family, closing pre-cutoff gap **G2**.
4. Runs a head-to-head between a lightweight harmfulness-direction probe and Llama Guard 3 8B at matched FPR — a comparison no pre-cutoff paper does jointly on jailbreak detection (gap **G3**).

## Optional Supporting Contribution
Cross-model / cross-dataset generalisation of the two-direction decomposition is enumerated as **verify-stage variants** (below) and consumed by `/auto-verify`, not by the main-experiment budget.

## Explicitly Rejected Complexity
- Sparse-autoencoder feature-dictionary decomposition of h and r — the "approximately linear direction" wording in `task.md` does not require it; a linear/shallow readout is the intended level.
- Formation tracing (when / how h and r emerge during safety fine-tuning) — `task.md` makes no origin claim.
- Per-decision evidence auditing — Claim 5 is an aggregate AUROC / FPR head-to-head, not a per-instance decision audit.

## Must-Prove Claims (verbatim from `idea-stage/IDEA_REPORT.md`)
- **Claim 1** — Two distinct, approximately-linear, independently-recoverable directions in the residual stream of Llama-3-8B-Instruct: h (harmfulness) and r (refusal).
- **Claim 2** — Position dissociation: h at the final instruction-token position, r just after the instruction (position crossover on the AUROC curve).
- **Claim 3** — Additive steering yields dissociated effects: h flips the internal harmfulness readout without changing refusal; r flips refusal without changing the internal harmfulness readout. Dose-response monotone; specificity controls confirm axis-selectivity.
- **Claim 4** — A notable class of successful jailbreaks (GCG-style suffixes and persuasion / adversarial templates) suppresses r while h is preserved; a hidden-state probe on h picks up the signature.
- **Claim 5** — A lightweight harmfulness-direction probe matches or beats Llama Guard 3 8B at flagging jailbreaks, at a fraction of the compute.

## Terminology Note (project-specific)
`task.md`'s Claim 5 uses the phrase "**Latent Guard**". That exact term is on the project's forbidden list (`.claude/forbidden-urls.txt`). In **artifact prose only** we call the same classifier the **harmfulness-direction probe**; the underlying object and the semantics of Claim 5 are unchanged and remain verbatim in `idea-stage/IDEA_REPORT.md`.

## Constraints
- **GPU budget**: 10 GPU-hours total (HARD; from `task.md`). Plan sums to ≈ 9.4 GPU-h.
- **GPU pin**: `CUDA_VISIBLE_DEVICES ∈ {0,1,2,3}` (or any subset). HARD.
- **Directory allowlist**: working dir + `/data/zhenqian/data` + `/data/zhenqian/models`. HARD.
- **Environment**: conda env.
- **Model paths**: `MODEL_DIR=/data/zhenqian/models`, `DATA_DIR=/data/zhenqian/data`.
- **Main-experiment resources** (from `task.md`): Llama-3-8B-Instruct at `/data/zhenqian/models/Meta-Llama-3-8B-Instruct`; AdvBench at `/data/zhenqian/data/AdvBench`. Fixed resources: Llama Guard 3 8B (baseline judge for Claim 5); GCG-style suffixes; PAP-style persuasion / adversarial templates (both for Claim 4).

## Verify-Stage Variants (out of main-experiment budget)
The following are NOT in the main plan and are NOT charged against the 10-hour budget; they are the swap axes `/auto-verify` will explore in its own budget:
- **Model swaps**: `Llama-2-Chat-7B`, `Qwen2-Instruct-7B` — re-run M1–M2 (and M3 if time allows) to test cross-model generalisation of the two-direction decomposition.
- **Dataset swaps**: `JailbreakBench (JBB)`, `Sorry-Bench`, `CATQA` (additional harmful/jailbreak benchmarks); `Alpaca` (benign contrast for over-refusal analysis); `XSTest` (exaggerated-refusal probe on benign lookalikes).
- **Attack-family swap**: within M4, an optional third attack family may be added if warranted.

`/auto-verify` will pick from these as needed under its own robustness threshold; it never spends the main-experiment budget.

## References to Working Artifacts
- Ranked claims + faithfulness audit: `idea-stage/IDEA_REPORT.md`.
- Landscape (pre-cutoff foundational literature only — target paper excluded by project policy): `idea-stage/LANDSCAPE.md`.
- Raw retrieval dump: `idea-stage/RESEARCH_LIT.md`.
- Experiment plan (milestone details, grids, budgets): `refine-logs/EXPERIMENT_PLAN.md`.
- Plan-level run tracker: `refine-logs/EXPERIMENT_TRACKER.md`.
