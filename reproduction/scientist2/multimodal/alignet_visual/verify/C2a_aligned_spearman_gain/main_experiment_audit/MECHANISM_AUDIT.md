# Mechanism Audit — C2a: Aligned DINOv2 Improves Aggregate Spearman

**Claim**: C2a — Alignment finetune of DINOv2 ViT-B increases aggregate Spearman by Δρ ≥ 0.05.

**Committed mechanism family**: Representation and Parameter Analysis / Parameter-Space Task Vectors
**Mechanism role**: Verify step — M3 full-backbone KD finetune produces `τ_aligned = θ_M3 − θ_pretrained` (the human-alignment task vector); M4 provides the baseline `θ_pretrained` evaluation. The task-vector sufficiency test is `apply_to(θ_pretrained, τ_aligned, coef=1.0)` evaluated on THINGS.

---

## Check A — Steering Coefficient Sweep

**Finding**: N/A with contextual note. The Parameter-Space Task Vectors family uses `apply_to(base_ckpt, scaling_coef)` as its primitive. In M3/M4, `scaling_coef = 1.0` (i.e., `θ_M3 = θ_pretrained + τ_aligned`) is the only evaluated point — the full task vector is applied directly. This is NOT a steering coefficient sweep in the mechanistic-interpretability sense (no α sweep across ≥3 orders of magnitude, no capability metric at each point, no plateau locking).

However, this is expected and appropriate for the current claim: C2a is about the behavioral effect of applying `τ_aligned` at `coef=1.0` on THINGS — a sufficiency test, not a dose-response characterization. The MECHANISM_ROUTING.md explicitly notes that the task-vector framing is the mechanism handle and that `scaling_coef ∈ {0.25, 0.5, 1.0, 2.0}` sweeping is designated for `/auto-verify` stress testing, not the main plan.

The main experiment does NOT:
- Sweep `scaling_coef` across ≥3 orders of magnitude
- Log a capability metric at each coefficient point
- Lock `coef` mid-plateau based on capability/coherence

But this is because C2a's mechanism claim is "does `τ_aligned` exist and improve THINGS Spearman when applied at coef=1.0?" — not "what is the dose-response relationship?" The task-vector framing's sufficiency test is the intended mechanism check, not a coefficient sweep.

Given that the family's mechanism primitive does not inherently require α-style sweeping (the task-vector itself is the intervention, and the claim is about applying it at coef=1.0), the steering coefficient check is N/A for C2a's main experiment.

**Severity**: N/A

---

## Checks B–F (Reserved)

not_implemented

---

## Overall Verdict

**overall_verdict: n/a**

C2a's main experiment (M3+M4) is a task-vector sufficiency test in the Parameter-Space Task Vectors family. The mechanism intervention is the full-backbone alignment finetune that produces `τ_aligned`, applied at `scaling_coef=1.0`. This does not require a steering coefficient sweep (the task vector IS the aligned checkpoint delta, not an additive direction applied at varying strengths). No mechanism rigor issues found.
