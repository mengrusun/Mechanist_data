# Reviewer Memory

Persistent adversarial suspicion log across iterations of `/auto-iteration-loop`.
Append-only; each iteration's Phase B.5 adds a new section.

## Iteration 1 — Score: 4/10, Verdict: not ready

- **New suspicions**:
  - The C2 sign inconsistency (median Spearman ρ = +0.371, two seeds strongly positive) may reflect more than a bookkeeping error: (a) direction convention reversed, (b) intervention hook applied with opposite sign, or (c) extracted direction is not behavior-specific and increasing "treated-like" activation worsens QA_I in the wrong way. seed2026 ρ = +0.786 is a big signal.
  - Without `other_rate` capability metric, the "null" recovery could be partly an OOD-collapse artifact at large |α|. α-wise degeneration curves needed.
  - `n_random = 1` for M2.2c is too weak to trust the reported specificity, even though observed deltas (0.3–0.9 pp) are small. Could be lucky draws.
  - C1 (phenomenon) is stronger than C2 (mechanism). The project may support a phenomenon-only paper if the mechanistic rerun doesn't clean up.
  - If `-v` restores monotonicity, the current mechanism result should be treated as implementation/documentation error, not evidence for null localization. Would require careful rewriting of mechanistic interpretation.
  - If neither sign works AND capability remains stable near moderate α, then BOUNDED_NULL becomes substantially more credible.

- **Previous suspicions addressed?**: n/a (first iteration).

- **Unresolved (carried forward)**:
  - Sign inconsistency in C2 (needs bidirectional α sweep to resolve).
  - Capability/OOD-collapse confound at α extremes (needs other_rate at every α).
  - Underpowered specificity control (needs n_random ≥ 30).

- **Patterns**:
  - Phenomenon-vs-mechanism asymmetry: C1 is clean and passes robustness under method-swap; C2's audit failures are all on the mechanism-rigor side, not the phenomenon side. Watch for the paper trying to over-narrate C2 relative to actual mechanistic evidence.
  - The BOUNDED NULL verdict itself is scientifically honest and pre-registered — the mechanism audit FAIL is a rigor/protocol issue, not a scientific fraud issue. Fixes should preserve this framing.
