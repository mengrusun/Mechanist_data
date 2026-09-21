# Mechanism Audit — C1a: Teacher Head Beats Unaligned SigLIP + Chance

**Claim**: C1a — THINGS-fit SigLIP-So400m teacher head strictly beats unaligned SigLIP + chance on held-out THINGS triplets.

**Committed mechanism family**: Representation and Parameter Analysis / Parameter-Space Task Vectors

**Scope**: M1 — this milestone is the "Screen" step in the Screen → Decode → Verify → Recover composition. It produces a teacher head (small MLP) on frozen SigLIP-So400m features — no mechanism intervention on internal representations; it is a discriminative fitting step, not a causal perturbation or direction extraction.

---

## Check A — Steering Coefficient Sweep

**Finding**: N/A. Claim C1a concerns the teacher head fitting step (M1), which involves:
1. A small alignment MLP head trained on frozen SigLIP-So400m features
2. Evaluation of triplet accuracy on held-out THINGS data

There is NO additive intervention on internal representations (no α-scaled direction addition, no activation patch, no causal intervention of any kind). The claim is about evaluating whether the head-projected features agree more with human triplet labels than unaligned pooler output — this is behavioral comparison, not mechanism intervention. The steering coefficient sweep check is not applicable (N/A).

No subpoint (σ_proj-scaling, plateau-locking, random-direction control, sign pattern) is relevant because no additive steering is performed.

**Severity**: N/A

---

## Checks B–F (Reserved)

**Finding**: not_implemented — reserved for future mechanism-rigor checks.

---

## Overall Verdict

**overall_verdict: n/a**

C1a uses no additive intervention on internal representations. The mechanism check does not apply. This is expected: the Screen step of the Parameter-Space Task Vectors family produces a teacher head via discriminative fitting, not via weight-space perturbation. C1a is not penalized by the mechanism audit.
