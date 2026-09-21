# Mechanism Audit — C3: Zero-Shot Transfer to Structurally Different Multi-Agent Scenarios

**Claim**: C3 — Probing / Residual Stream States (Decision Auditing direction). Frozen probe applied zero-shot.

**Audit date**: 2026-07-14

---

## Check A — Steering Coefficient Sweep

**Finding**: C3 applies the frozen M1 probe (logistic regression) and M2 best aggregation (best-of-N) to new scenarios — no additive intervention on representations. `apply_probe_zero_shot.py` performs a pure forward-pass and read-out, identical to the M1 probe application but on transfer data. No steering coefficient.

**Verdict**: N/A — no mechanism intervention.

---

## Checks B–F — Reserved

Not implemented.

---

## Overall Verdict

**overall_verdict: n/a** — Passive frozen probe read-out. No causal intervention. N/A contributes no severity.
