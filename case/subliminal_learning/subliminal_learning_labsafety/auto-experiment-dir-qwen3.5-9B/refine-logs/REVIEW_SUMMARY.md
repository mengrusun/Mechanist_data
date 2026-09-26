# Review Summary — Given-Validation × Discovery Testing Approach

**Date**: 2026-07-09
**Behavior-source**: given-validation (claims frozen from task.md — never edited)
**Mechanism**: discovery (Location + Causal Intervention, family routed at experiment stage)

## What was reviewed
The unified testing approach that verifies three claims from `idea-stage/IDEA_REPORT.md`:
- Claim 1 (M0 primary): cross-modal subliminal transfer, ≥3 pp Acc drop across ≥3 seeds
- Claim 2 (data-purity precondition): 0 unsafe rows after rescanning
- Claim 3 (mechanism, kind-level): a low-dim safety-relevant subspace in the Qwen3.5 language tower causally mediates the drop

## Design decisions the reviewer challenged, and the response

1. **"3% is a weak threshold — could be noise."**
   - Response: enforce **per-seed** (each of 3 seeds meets ≥ 3 pp), not just average. Report per-seed values and mean±std. Also add: **bootstrap CI on the paired difference** at the sample level (paired by QA_I item), so the effect is defended twice — across-seed variance AND within-seed sampling variance.

2. **"LR sweep + 3 seeds could inflate a false positive."**
   - Response: the LR sweep is done **on a held-out seed dev-slice** (seed 42 with a fixed LR grid), then the winning LR is **frozen** and applied identically to the ≥3 reproduction seeds. Report both dev-LR-picking curve and per-seed results at the winning LR. Rules out garden-of-forking-paths / LR cherry-picking.

3. **"Trivial explanations (tokenizer / LoRA-target-module trap) will masquerade as the effect."**
   - Response: dedicated pre-M0 sanity milestone `M-1` (zero-shot trivial-explanation checks). It exercises: (a) tokenizer round-trip stability on QA_I items; (b) that `AutoModelForImageTextToText` is actually the class loaded and LoRA is attached under `model.language_model.*` (assertion-level check on the PEFT target patterns); (c) that image encoder outputs are non-degenerate under the treated-arm forward. Failure of any = stop, do not run M0.

4. **"Paraphrase/decoding stability was not in task.md — do we really need it for M0?"**
   - Response: task.md's minimum is seed-only reproduction. `/mechanism-explore`'s M0 discipline additionally requires paraphrase + decoding stability. **We honor both**: the primary predicate (≥ 3 pp per seed on the exact QA_I default protocol) is what M0 votes on; paraphrase + decoding stability are **auxiliary robustness checks** reported inside M0 that promote the verdict from `conditional` to `established`. This satisfies task.md without weakening rigor, and satisfies `/mechanism-explore` without adding predicates task.md didn't request.

5. **"Mechanism claim assumes a low-dim direction. What if the trait is high-dim?"**
   - Response: the claim is at *kind-of-component* altitude per `/mechanism-explore` — it asserts the KIND (low-dim direction / small feature set), not identity. The Location milestone reports the *rank* of the treated–Ctrl activation subspace (e.g., cumulative-variance-explained curve, or effective rank of the residual-stream difference); if effective rank is large (say > 32 across all safety-relevant layers), the mechanism claim is refuted at that level and the plan writes a "distributed rewrite" negative result — a legitimate finding, not a failure.

6. **"Specificity — how do you rule out generic capability collapse?"**
   - Response: (a) run **matched-control directions** (random or non-safety-relevant probe directions) at identical dose; require < 1/3 of the effect. (b) run a **general-capability control benchmark** (MMLU-style subset via image-conditioned prompts, or a QA_I-shaped non-safety subset if such exists in QA_I — else a text-only MMLU slice on the language tower with images blanked). Require intervention drops ≤ 2 pp on that control.

## Final verdict
**READY** — the testing approach cleanly gates on M0, defends each claim with its own controls, respects every task.md constraint, and stays at the correct claim altitude for `discovery` mechanism.

## Remaining risks
- The general-capability control benchmark is not named by task.md; the plan uses a text-only MMLU slice on the language tower + a random-image QA_I subset as fallback. This is a soft weakness — flagged as *needs manual review* if the reviewer wants a specific control.
- Under `MECHANISM=discovery`, the `n_pairs`, `sites`, `metric`, `gpu_hours` fields on intervention milestones are provisional and tagged `method_sensitive`; the experiment stage's `/mechanism-skills` routing will finalize them.
