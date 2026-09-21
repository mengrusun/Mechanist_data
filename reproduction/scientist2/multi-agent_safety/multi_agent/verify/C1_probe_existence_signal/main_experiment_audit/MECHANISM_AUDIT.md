# Mechanism Audit — C1: Existence of Internal Collusion Signal

**Claim**: C1 — Probing / Residual Stream States (Location direction).

**Audit date**: 2026-07-14

---

## Check A — Steering Coefficient Sweep

**Finding**: C1 uses the Probing / Residual Stream States method family (per `MECHANISM_ROUTING.md`). This is a passive read-out mechanism — a linear probe trained on cached residual-stream activations. There is **no additive intervention on internal representations** (no steering vector, no activation addition, no causal patch). The mechanism is purely correlational / decoding: we check whether a linear classifier can recover the collusion label from the residual stream, not whether adding a direction changes model behavior.

No steering coefficient (`α`) is used anywhere in the pipeline (`extract_activations.py`, `train_probe.py`, `train_aggregation.py`, `apply_probe_zero_shot.py`, `verdicts.py`). Therefore Check A returns N/A.

**Verdict**: N/A — no mechanism intervention used; probing is a passive read-out.

---

## Checks B–F — Reserved

Not implemented in current protocol.

---

## Overall Verdict

**overall_verdict: n/a** — The experiment uses a passive probing method (residual-stream linear probe) with no steering, ablation, or other causal intervention. Mechanism rigor checks for coefficient sweep and related controls are inapplicable. Per the audit protocol, n/a is treated as severity=0 and contributes no penalty to the combined Phase 2 gate verdict.
