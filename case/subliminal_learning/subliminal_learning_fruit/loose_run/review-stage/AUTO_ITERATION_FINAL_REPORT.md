# Auto Iteration Final Report — Subliminal Learning in Diffusion Image Models (Qwen-Image)

- **Generated**: 2026-07-20T23:35:00
- **Iterations consumed (back-edge count)**: 0 / 2   (all iteration actions were type ⓪ narrative-only or pure re-review; 2 reviewer cycles were completed)
- **Claim-reentries consumed**: 0 / 1
- **Final reviewer score**: 5 / 10
- **Final canonical verdict**: almost
- **Termination reason**: `iterations_exhausted` (by task-declared reviewer-cycle cap; formal back-edge count is 0/2 — see Termination Note below)
- **Cumulative cost**: runs_total = 0, gpu_hours_total = 0.0
- **Reviewer model**: `gpt-5.4` via `<REDACTED_API_BASE_URL>` (source: shell env)
- **GPU allocation**: `CUDA_VISIBLE_DEVICES=4,5,6,7` (not exercised — no runs dispatched)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md), [`REVIEW_STATE.json`](./REVIEW_STATE.json)

**Termination Note**: MAX_ITERATIONS=2 was set as a **tightened reviewer-cycle cap** in the orchestrator task instructions ("reduced from default 6 due to HARD-budget crisis (experiment already 5.5h over)"). The formal back-edge counter `iterations_consumed = 0` because both reviewer cycles concluded with **"none pending"** for FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS fixes — every actionable concern was addressable via ⓪ narrative-only edits (which do not consume budget) or was contract-blocked (INTEGRITY_ONLY claims permit no back-edge action). The one substantive fix the reviewer flagged — a ~0.1 GPU-h sign-orientation rerun for C3's amplification arms — is affordable within the 0.8 GPU-h cap but targets an INTEGRITY_ONLY claim, so it is contract-blocked inside the iteration loop; the correct upgrade path is a standalone `/auto-verify C3 -- resume: true` (potentially preceded by a one-line sign-orient patch to `src/mechanism/mechanism_location.py` L228-230) run outside this loop.

---

## Executive Summary

The review loop ran two reviewer cycles under a HARD-budget-crisis-tightened iteration cap and consumed **zero GPU-hours**. Iteration 1 applied ⓪ narrative-only refinements (paper-framing caveats for C1's residue overreach risk and C3's negative-result reframing) plus an iteration-1 code audit of `src/mechanism/mechanism_intervene.py` and `mechanism_location.py` that identified an **unresolved SVD sign-ambiguity** in the `banana_direction.pt` extraction (the intervention hook itself is correct). Iteration 2 confirmed the C1 framing caveat as **addressed** and rated the C3 code-audit response as **partially addressed** (the confirmatory ~0.1 GPU-h sign-orient rerun was deferred — this is the reviewer's one remaining substantive concern). Score moved 4/10 → 5/10 (attributable purely to framing discipline; underlying evidence unchanged). Verdict remained `almost` throughout. The reviewer explicitly separated **internal-READY** (satisfied: no FAIL/INCONCLUSIVE/ZEV claims remain) from **top-venue-mechanism-paper READY** (not satisfied — C3's negative causal-handle result is a scientific reality unfixable within budget). Under an honest phenomenon-paper framing (C1 = strong phenomenon; C2 = suggestive late-layer LoRA-subspace signature; C3 = negative result against a single-direction causal handle), the work is `almost` submission-ready.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---:|---|
| PASS                     | 1 | 1 PASS held (C1) — framing caveat added; residue-overreach risk closed |
| FAIL                     | 0 | — |
| INCONCLUSIVE             | 0 | — |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| INTEGRITY_ONLY           | 2 | 2 carried forward (C2, C3) — no back-edge dispatched per verify contract; standalone `/auto-verify <id> -- resume: true` is the upgrade path |
| DEFERRED (legacy)        | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `C1` — Subliminal banana preference transfers from teacher to student
- **Original robustness signal**: robustness = 1.00 (variants_passed = 1/1); main-experiment Phase 2 integrity = WARN (M0 verdict override `inconclusive → conditional`)
- **Reviewer consistency check** (both iterations): numeric consistency confirmed — mean_teacher=0.680, max_control=0.038 → 64.2pp gap, 8/8 seeds positive, judge_recall=1.000, method-swap variant corroborates at 50.2pp binary. **Effect is unambiguously real; no numeric inconsistency.**
- **Touched in iterations**: [1, 2]
- **Final status**: **PASS (held)**
- **Notes for downstream paper**: Iteration-1 ⓪ paper-framing caveat added to `CLAIMS_LEDGER.md` C1 Caveats — the paper MUST NOT phrase this as "residue-free subliminal transfer". Correct phrasing: *"subliminal transfer with judge-stochasticity-level residue (≤ 3.3% under both MCQ and binary judge templates), corroborated by method-swap variant with a 50.2pp binary-judge gap"*. The post-hoc `inconclusive → conditional` override MUST be flagged in Methods, not buried. Iteration-2 reviewer rated this caveat as **addressed**.

---

## Section 2 — FAIL Claims (full journey)

None. `verify_failed = []`.

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

None. `verify_inconclusive = []`.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

None. `verify_zero_eligible_variants = []`.

---

## Section 4b — INTEGRITY_ONLY Claims (no-action, carried forward)

Two claims arrived in INTEGRITY_ONLY with `stage2_skip_reason = max_verify_claims_cap`. Per verify contract, the iteration loop is not permitted to dispatch back-edges on these. Both are surfaced here for the orchestrator's `open_items[]`.

### 4b.1 `C2` — Location of banana signal in DiT
- **`main_experiment_integrity`**: `pass`
- **Stage 1 audit**: PASS
- **Stage 2 skip reason**: `max_verify_claims_cap` (MAX_VERIFY_CLAIMS=1; C1 picked over C2 by importance)
- **Evidence on file (unchanged by iteration)**: b* = block 47/60 (78% depth), target_module = `attn.to_out.0`, top-block ratio = 11.30×, top-1 SVD variance fraction = 0.30
- **Hypothesis discrimination**: LoRA-artifact = consistent; single-steering-vector = not-clearly-supported; divergence-latent + early-layer = falsified
- **Iteration verdict**: not a fully verified mechanistic result (diagnostic/correlational in weight space; no causal control established here — that was M2/C3's job)
- **Upgrade path (outside iteration loop)**: `/auto-verify C2 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run)

### 4b.2 `C3` — Causal intervention on located sites
- **`main_experiment_integrity`**: `warn` (`warn_source: experiment+mechanism` — single-seed intervention + incomplete dose grid; random-direction ablation exceeds target-direction)
- **Stage 1 audit**: WARN
- **Stage 2 skip reason**: `max_verify_claims_cap`
- **Evidence on file (unchanged by iteration)**:
  | Intervention | P(banana) | Δ vs baseline |
  |---|---:|---:|
  | baseline | 0.744 | — |
  | ablate | 0.713 | −3pp |
  | amplify_x2 | 0.675 | −7pp (wrong sign observationally — see sign-ambiguity caveat) |
  | amplify_x4 | 0.656 | −9pp (wrong sign observationally — see sign-ambiguity caveat) |
  | random_ablate | 0.706 | −4pp (exceeds target ablation) |
  | matched_control_ablate | 0.719 | −2pp (comparable to target) |
- **Iteration-1 ⓪ code audit finding**: intervention hook (`src/mechanism/mechanism_intervene.py` L127-167) is correctly operationalized — no sign flip bug. Direction extraction (`src/mechanism/mechanism_location.py` L228-230, `direction_out = U_m[:, 0]`) saves the raw top-1 left singular vector with **no sign-orientation step**; SVD sign is arbitrary. The amplify-decreases-P(banana) observation is therefore **not diagnostic** between (a) SVD sign ambiguity and (b) real distributed-subspace effect. Ablate / random_ablate / matched_control_ablate results are sign-independent and remain valid signals of a weak/non-specific causal handle.
- **Iteration-2 reviewer rating**: code-audit response = **partially addressed** — the hook is confirmed clean, but the practical amplification-result interpretability remains open until the ~0.1 GPU-h confirmatory rerun is performed.
- **Iteration verdict**: negative result against a single-direction causal handle at DiT block 47 attn.to_out.0. Consistent with (but not positive evidence for) the distributed LoRA-artifact account per Nief 2606.00831.
- **Upgrade path (outside iteration loop)**:
  1. Add sign-orient step in `mechanism_location.py` L228-230 (flip `direction_out` iff `<mean_teacher_arm(ΔW) @ direction_out, teacher_ref>` < 0) before writing `banana_direction.pt`.
  2. Re-run only the two amplify arms in `mechanism_intervene.py` (~0.1 GPU-h).
  3. Then upgrade the swap test: `/auto-verify C3 -- resume: true`.

---

## Section 5 — Legacy DEFERRED Claims (empty)

None. `deferred_claims = []` (new-architecture verify does not populate this bucket).

---

## Section 6 — Cross-Cutting Patterns

Patterns the reviewer flagged across both iterations (from `REVIEWER_MEMORY.md`):

- **Framing risk cluster** (iteration 1) → *substantially addressed by iteration 2*. Three flags (C1 residue, C3 amp-sign, C3 as "partial") were about presentation, not numbers. All three received ⓪ narrative fixes; C1 rated `addressed`, C3 rated `partially addressed`.
- **Budget-blocked cluster** (iteration 1) → *unresolved and unfixable within iteration scope*. Multi-seed C3 (~1.4h) and full LR sweep (~2.4h) both exceed the 0.8h iteration cap.
- **Narrative repair helped substantially (+1 score point), but evidence did not improve** (iteration 2) — this is the correct summary of the whole loop's arc.
- **C3 remains the central blocker for ambitious claims** (iteration 2) — the code audit removed one possible excuse (implementation bug) but left the causal picture weak; this is the dominant reason the work is not top-tier mechanism-ready.
- **Internal-ready vs venue-ready distinction** (both iterations) — internal contract is satisfied (no FAIL/INCONCLUSIVE/ZEV); top-venue mechanism-paper standards are stricter. This distinction persists at termination.
- **Execution-choice bias signal** (iteration 2, new) — the affordable ~0.1 GPU-h C3 sign-orient rerun was deferred; reviewer flagged this as possible selective tolerance for ambiguity when the result might remain unfavorable. The contract-block on INTEGRITY_ONLY back-edges is a technical justification but does not fully dissolve the reviewer's concern.
- **Risk of laundering weak C3 through "distributed subspace" rhetoric** (iteration 2, new) — the ledger now says the result is "consistent with the distributed LoRA-artifact account." Consistency is cheap; the data do NOT positively establish a distributed-subspace mechanism. Watch for wording drift in the manuscript.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed (back-edge count)**: 0 / 2 (both cycles were ⓪-only / pure re-review)
- **Claim-reentries consumed**: 0 / 1
- **Iteration `/run-experiment` calls**: runs_total = 0
- **Iteration GPU-hours**: gpu_hours_total = 0.0 (well within the 0.8 GPU-h cap; the whole HARD-crisis budget guard was respected)

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ⓪ narrative_only + code audit | C1, C3 | — | 0 | 0.0 | 4 | almost |
| 2 | pure re-review (no on-disk changes) | — | — | 0 | 0.0 | 5 | almost |

---

## Section 8 — Open Items for Human Reviewer

Items the loop could not close. These need a human or a separate `/auto-verify` upgrade to resolve.

- **Still-FAIL claims**: none
- **Still-INCONCLUSIVE claims**: none
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:
  - **C2** [stage2_skip_reason: `max_verify_claims_cap`; `main_experiment_integrity: pass`]: upgrade via `/auto-verify C2 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run).
  - **C3** [stage2_skip_reason: `max_verify_claims_cap`; `main_experiment_integrity: warn`; `warn_source: experiment+mechanism`]: upgrade via `/auto-verify C3 -- resume: true`. **Additional cheap-fix recipe from iteration-1 code audit** (to resolve the amplification sign-ambiguity before swap-testing): add a one-line sign-orient step to `src/mechanism/mechanism_location.py` L228-230, then re-run only the two amplify arms in `src/mechanism/mechanism_intervene.py` (~0.1 GPU-h).
- **Deferred out-of-budget**:
  - Multi-seed C3 replication (2 additional seeds; ~1.4 GPU-h; exceeds 0.8h iteration cap by 1.75×)
  - Full LR sweep completion (lr=1e-5, 5e-6; ~2.4 GPU-h; not affordable — exceeds cap by 3×)
- **Legacy deferred claims**: none
- **Recurring unresolved patterns**:
  - Internal-READY ≠ top-venue-READY — the phenomenon story is publishable somewhere under honest narrow framing; the mechanism paper story is not.
  - Execution-choice bias signal — the affordable C3 sign-orient rerun was deferred; a human reviewer may want to reconsider this choice outside the iteration-loop contract constraints.
- **Claim-reentry refusals** (③ requested but sub-budget exhausted): none — reviewer never proposed ③ (no FAIL claims existed).

---

## Section 9 — Recommended Next Actions (for orchestrator / human)

1. **Adopt the paper-framing guidance from `CLAIMS_LEDGER.md`** (C1 Caveats + C3 Caveats + Open Items) verbatim into the manuscript's Introduction, Methods (residue-override disclosure), and Discussion (C3 as negative result, not "partial support").
2. **Run `/auto-verify C3 -- resume: true` outside this loop** with the sign-orient patch applied first (`mechanism_location.py` L228-230). Total additional cost: ~0.1 GPU-h + swap-variant cost.
3. **Optionally run `/auto-verify C2 -- resume: true`** to convert C2 from INTEGRITY_ONLY to PASS/FAIL/etc. — this is not required for a phenomenon-paper submission but strengthens the "suggestive late-layer LoRA-subspace signature" story.
4. **Do NOT attempt** the multi-seed C3 replication or the full LR sweep within the HARD budget — both are documented as "deferred to a future compute cycle".
