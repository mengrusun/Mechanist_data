# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
routing: pending
committed: false

## Rationale

Phase 1.25's phenomenon-validation gate (M0) returns `phenomenon_status: established` at LR 1e-3 — the treated arm reproduces the ≥ 3 pp double-drop vs Ctrl-A AND Ctrl-B on all 3 seeds (Δ_A +32–38 pp, Δ_B +27–39 pp), while Ctrl-B at the same LR/recipe shows no drop, isolating a data-specific transfer. C2 (filter-cleanliness) passed; C3 (no-collapse on the task.md indicators — loss NaN/divergence, repetition spike, degenerate output) passed at LR 1e-3.

With M0 established, the downstream mechanism milestones (M1 Location, M2 Causal Intervention) are **unblocked** but **not yet routed / run**. Recommended entry point: evaluate the mechanism chain at the established operating point (LR 1e-3), Treated vs Ctrl-B. `/auto-verify` and `/auto-iteration-loop` have likewise not yet been run on this newly-established result.
