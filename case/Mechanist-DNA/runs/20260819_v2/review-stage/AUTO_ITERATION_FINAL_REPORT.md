# Auto Iteration Final Report — Causal α-helix steering in Evo2-7B

- **Generated**: 2026-08-19
- **Iterations consumed**: 2 / 6  (① 0, ② 1, ③ 1; plus one budget-free ⓪ narrative refinement)
- **Claim-reentries consumed**: 1 / 2
- **Final reviewer score**: 5 / 10
- **Final canonical verdict**: almost
- **Termination reason**: stalled / converged below target — all claims resolved & honest, no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE remaining, but final score 5 < TARGET 6 and the gap is NOT closable by any actionable back-edge (structural validation is infrastructure-blocked; composition confound is intrinsic to the mechanism)
- **Cumulative cost**: runs_total=1, gpu_hours_total=0.012
- **Reviewer**: gpt-5.6-luna (external, via dmxapi)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md)

---

## Executive Summary

The loop entered with C1 (PRIMARY causal α-helix steering) at INCONCLUSIVE (verify Phase-2 main-experiment-integrity FAIL) and C2 (eval-harness fidelity) at PASS. A type-② main-experiment fix (milestone **M5**) hardened C1's PRIMARY %H endpoint by re-scoring the existing 400 steered/baseline generations with two ESM2-independent predictors (GOR-windowed, Chou-Fasman) plus GC/amino-acid-composition/low-complexity controls and an honest non-monotonicity analysis — no regeneration, because the steering itself was sound and only the %H *evaluation* was broken. The repair was decisive and two-sided: the +14–16 pt %H uplift **reproduces across all three predictors** (so it is not a single-probe artifact), **but** it co-emerges with a severe, near-non-overlapping composition collapse (≈40% Lysine, low-complexity, GC 0.44→0.13), does **not** survive an (underpowered, 5/200-overlap) composition-matched test, and has **no structural validation** (ESMFold CDN-blocked). With the endpoint now trustworthy, the frozen strong C1 became assessable and **not-supported**, so a type-③ lightweight in-loop narrowing rewrote it to **C1_v2** — an intervention-specific, predictor-consistent bias in *predicted* helix-propensity/composition, explicitly **not** physical α-helix structure. C1_v2 is fully supported by the existing evidence and passes with a major caveat; C2 holds as PASS. The loop converged at score 5/"almost": the original strong claim was falsified and honestly narrowed, and the remaining gap to a top-venue score is an inherent scientific limitation (intrinsic composition confound + infrastructure-blocked structural validation) that no back-edge can close.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 1 (C2) | 1 PASS (held) |
| FAIL                     | 0 | — |
| INCONCLUSIVE             | 1 (C1) | strong C1 **falsified**; narrowed via ③ to **C1_v2 = PASS-with-major-caveat** |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| DEFERRED                 | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `C2` — the fast ORF→translate→SS-predict→%H harness agrees with structure-based DSSP
- **Original robustness signal**: robustness=1.00; method-swap (independent GOR) r=0.85, dataset-swap (fresh 310 proteins) r=0.979; main-experiment integrity WARN (scope only).
- **Reviewer consistency check**: consistent — r=0.987 vs experimental DSSP on 200 held-out natural proteins (r²≈0.974), Q3=0.858, frame recovery 1.0.
- **Touched in iterations**: [1, 2, 3] (brief audit only — no back-edge).
- **Final status**: PASS (held).
- **Notes for downstream**: **Caveat (must travel with the claim):** the evaluator is validated on NATURAL proteins only; it does NOT validate %H on the steered, GC-collapsed OOD sequences used for C1_v2. Do not cite C2 as evidence that the steered sequences contain physical α-helices.

---

## Section 2 — FAIL Claims (full journey)

(No claim entered the loop as FAIL. C1 entered as INCONCLUSIVE and is documented in Section 3; the ③ narrowing it triggered produced C1_v2.)

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

### 3.1 `C1` → `C1_v2` — causal α-helix steering in Evo2-7B

**Original INCONCLUSIVE reason** (from `verify/C1_causal_helix_steering/ROBUSTNESS.md`, pre-fix):
- main-experiment integrity broken (Phase 2): experiment-audit FAIL — the generated-sequence %H endpoint is an ESM2-650M+probe proxy validated only on natural proteins (r=0.987) but applied to heavily-steered, GC-collapsed (0.40→0.13), OOD sequences with NO structural validation; dose curve non-monotonic (α=4 dip); gain confounded with the GC collapse. Mechanism-audit WARN. Variants never ran.

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 1 | ② | endpoint untrustworthy (Check E scope + Check F synthetic_proxy) | added milestone M5: independent-predictor triangulation + GC/composition control + honest non-monotonicity on the existing 400 generations; re-audited the corrected main experiment | Check F REPAIRED fail→warn; verdict now computable = **not-supported**; residual FAIL = Check-E claim-scope overclaim → INCONCLUSIVE→**FAIL (claim-support)**; recommended narrow_claim |
| 2 | ③ | frozen claim overclaims vs repaired evidence | lightweight in-loop narrowing → **C1_v2** (no new experiments; existing evidence supports it) | strong C1 falsified; C1_v2 = **PASS-with-major-caveat**; score 4→"almost" |
| 3 | ⓪ | final wording/statistical-presentation tweaks | applied (intervention-specific; consistent-across-3-predictors; random-control equivalence; CIs; "shows" not "establishes") | score 5/"almost"; no actionable back-edge remains → termination |

**Path taken (summary)**: main-experiment integrity fix (②) → endpoint trustworthy → claim-stage re-entry / narrowing (③) → narrative finalization (⓪). Claim-reentry sub-budget used: 1.

**Experiment & script modifications (cumulative)**

| Iter | Component | Before | After |
|---|---|---|---|
| 1 | `refine-logs/EXPERIMENT_PLAN.md` | (no composition/independent-predictor control) | new **M5** milestone (independent-predictor triangulation + GC/composition control + honest non-monotonicity); `resource_fidelity` marker preserved verbatim |
| 1 | `experiments/run_M5_structural_gc_control.py` | — | NEW — re-scores existing generations with GOR + Chou-Fasman, GC-stratified/matched analysis, composition/length/low-complexity controls, bootstrap CIs, dose-curve honesty |
| 1 | `verify/C1_causal_helix_steering/main_experiment_audit/EXPERIMENT_AUDIT.{json,md}` | overall FAIL (Check E fail, Check F fail=synthetic_proxy) | overall FAIL (Check E fail=**claim-scope overclaim only**; Check F **warn** — repaired); verdict computable = not-supported; recommend narrow_claim (stale copies archived `*.stale.md`) |
| 1 | `refine-logs/EXPERIMENT_RESULTS.md` | "Claim 1 SUPPORTED — causally and monotonically raises %H" | M5 section + narrowed Claim-1 summary |

**Re-experiment outcome**

| Iter | Path | New runs | Result |
|---|---|---|---|
| 1 | lightweight ② (re-score existing generations; no regeneration) + `/experiment-audit` re-check | `runs/iteration_round_1/M5_structural_gc_control.json` | endpoint repaired (Check F warn); frozen strong C1 not-supported |

**Key M5 numbers** (`runs/iteration_round_1/M5_structural_gc_control.json`):
- Uplift (steered−baseline, bootstrap 95% CI): ESM2 +15.5 [9.3,21.6]; GOR +15.0 [9.5,20.8]; Chou-Fasman +13.6 [9.3,18.1]; all MWU p<1e-6 → not a single-probe artifact.
- Confound: AA-shift dominated by Lysine +0.396; low-complexity max-AA-fraction 0.21→0.43; length 69→46; GC 0.44→0.13; within-baseline %H anti-correlated with GC (ESM2 r=−0.17 p=0.017; CF r=−0.36 p=2e-7).
- Composition control underpowered / non-surviving: only 5/200 baseline generations overlap the steered GC band [0.11,0.20]; GC-matched & NN-matched deltas' 95% CIs span zero.
- No structural grounding (ESMFold CDN-blocked) → "sequence-predictor-supported, NOT structurally confirmed."
- Dose non-monotonic (full-range Spearman 0.54–0.86; only α≥8 monotone across all predictors).

**Claim modifications**
- Original claim id `C1`: "A localizable internal component of Evo2-7B (block-26 linear-probe α-helix direction) causally and specifically increases generated α-helical content (%H)." → **FALSIFIED** (strong form; composition-confounded, no structural validation, underpowered control).
- After iteration 2 → new claim id `C1_v2`: "Adding a block-26 linear-probe-derived residual-stream direction to Evo2-7B during autoregressive generation produces an **intervention-specific** shift (vs the tested norm-matched random-direction control, which shows no measured gain) toward higher **predicted** α-helix propensity; at a prespecified high-dose regime (α≥8; non-monotonic below) the estimated increase is ≈+14–16 pt of predicted %H (bootstrap 95% CIs given), **consistent across three distinct, heterogeneously-implemented sequence-based predictors** (ESM2-probe, GOR, Chou-Fasman), with ORF-validity and coding-likelihood preserved/improved under the tested conditions; the increase is **strongly accompanied by, and not shown separable from,** an AT-rich/lysine-rich/low-complexity composition change (GC 0.44→0.13; composition-matched test underpowered, 5/200 overlap); the result **shows** an intervention-specific, predictor-consistent bias in **predicted** helix-propensity/composition — **NOT** increased physical α-helical structure or a composition-independent mechanism (no structural/folding validation possible)."
- Scope change: dropped the physical-structure and composition-independence assertions; restricted the quantitative claim to α≥8; foregrounded the composition confound and the absence of structural validation.

**Final experiment summary**
- New runs cited: `runs/iteration_round_1/M5_structural_gc_control.json`.
- Final status: **C1 strong = FALSIFIED; C1_v2 = PASS-with-major-caveat** (`verify/C1_causal_helix_steering/ROBUSTNESS_C1_v2.md`).

**Reviewer memory thread (C1)**
- Proxy/compositional confounding suspicion (iter 1): confirmed real; the repaired endpoint shows it is intrinsic → drove the falsification/narrowing.
- "Independent predictors ≠ structural proof" (iter 1): upheld — three predictors agree but may share composition sensitivity; explicitly reflected in C1_v2's wording ("consistent across", not "independent"/"structural").
- Resolved: single-probe-artifact concern (3 predictors); non-monotonicity (honest, α≥8); evaluator-domain concern (C2 caveat separated).
- Unresolved (inherent): composition confound not separable; no structural validation; physical α-helix structure unverified.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims

None.

---

## Section 5 — Legacy DEFERRED Claims

None (empty under current architecture).

---

## Section 6 — Cross-Cutting Patterns

- **Proxy-validity / OOD-generalization gap** (all iterations): the primary metric was evaluated far outside its validated (natural-protein) domain, exactly where the apparent effect co-emerges with a compositional shift. Touched C1; resolved by hardening the endpoint (③ narrowing keeps the claim inside what the metric can support).
- **Metric-robustness ≠ construct validity**: three heterogeneous predictors agreeing on a +15 metric gain does not establish the underlying biological construct (physical helix) when all predictors can be fooled by the same composition artifact. Reflected in the final claim wording.
- **Steering-induced distribution collapse**: the intervention that produces the effect also collapses composition/complexity (poly-Lysine, GC 0.44→0.13), so the effect and its confound are inseparable at the doses where the effect appears — a genuine dead-end for a composition-independent claim.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 2 / 6  (② ×1, ③ ×1; plus one budget-free ⓪)
- **Claim-reentries consumed**: 1 / 2
- **Iteration `/run-experiment` calls**: runs_total = 1
- **Iteration GPU-hours**: gpu_hours_total = 0.012

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ② main_experiment_fix | C1 | — | 1 | 0.012 | 4 | not ready |
| 2 | ③ claim_reentry (lightweight) | C1 | C1_v2 | 0 | 0.0 | 4 | almost |
| 3 | ⓪ narrative_only (no budget) | C1_v2 | — | 0 | 0.0 | 5 | almost |

---

## Section 8 — Open Items for Human Reviewer

- **Still-FAIL claims**: none.
- **Still-INCONCLUSIVE claims**: none (C1's INCONCLUSIVE was cleared by the ② endpoint repair; the strong claim was then falsified and narrowed to C1_v2).
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none.
- **INTEGRITY_ONLY claims**: none.
- **Scientific limitations capping the score (NOT closable in-loop)**:
  1. **No structural validation of steered sequences** — ESMFold (2.7 GB) and the Goodfire SAE (537 MB) were HF-mirror-CDN-blocked at experiment time; no local folder was available. C1_v2's endpoint is sequence-predictor-based only. A direct HF endpoint (with token) for large files, or a local/open-source structure predictor / orthogonal biophysical assay, would be needed to test physical α-helix content — but per the reviewer this would **not** resolve the composition confound.
  2. **Composition confound is intrinsic** — the block-26 steering that raises predicted %H also collapses composition (≈40% Lys, low-complexity, GC 0.44→0.13); a composition-independent helix claim cannot be recovered from the current data (no α with both a real effect and adequate GC-overlap; the composition-matched test has only 5/200 overlap).
  3. **Underpowered composition matching** — a composition-decoupled steering protocol (e.g. steering with a composition/GC penalty, or a paired synonymous-recoding design) producing adequate GC/length/composition overlap at the high-dose regime would be required to test separability. Speculative; likely a new research direction, not an in-loop fix.
- **C2 caveat to carry into the paper**: evaluator validated on natural proteins only — not validated on steered OOD sequences.
- **Recurring unresolved pattern**: every stronger biological interpretation rests on the unclosed OOD + composition-confounding gap.
- **Claim-reentry refusals**: none (sub-budget not exhausted; 1/2 used).
- **Recommended framing for any submission**: position the contribution as *controlled generative steering of a DNA language model with rigorous confound/negative analysis* (an honest methods + confound-analysis result), NOT as protein-structure discovery.
