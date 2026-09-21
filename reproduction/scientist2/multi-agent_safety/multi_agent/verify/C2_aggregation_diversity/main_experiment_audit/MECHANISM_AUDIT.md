# Mechanism Audit — C2: Group Aggregation Dominates Single-Agent

**Claim**: C2 — Probing / Residual Stream States (Location + Decision Auditing directions). Group-level aggregation over per-agent probe scores.

**Audit date**: 2026-07-14

---

## Check A — Steering Coefficient Sweep

**Finding**: C2 uses M1's frozen probe scores as inputs and applies five deterministic or small-head aggregation functions. No additive intervention on internal representations — this is a pure post-hoc aggregation over cached logistic-regression scores. No steering coefficient used anywhere in `train_aggregation.py` or `aggregation_diversity.py`.

**Verdict**: N/A — no mechanism intervention.

---

## Checks B–F — Reserved

Not implemented.

---

## Overall Verdict

**overall_verdict: n/a** — Passive aggregation over probe outputs. No causal intervention. N/A contributes no severity to the combined gate verdict.
