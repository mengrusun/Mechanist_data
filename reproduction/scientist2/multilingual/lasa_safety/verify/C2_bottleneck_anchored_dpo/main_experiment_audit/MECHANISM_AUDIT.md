# Mechanism Audit — C2 (main experiment)

**Claim**: C2 — Safety payoff of L*-anchored DPO vs surface DPO.

**Mechanism family**: Representation and Parameter Analysis / representation-engineering (M3) — LoRRA-style representation-space regularizer at L* on top of DPO.

**Audit date**: 2026-07-14

---

## Check A — Steering Coefficient Sweep

**Applicability**: C2's M3 training uses a regularizer coefficient λ (lambda_bottleneck = 0.5 for M3-Method, 0.0 for M3-Baseline). This λ is analogous to a steering coefficient in that it controls the strength of the representation-space invariance signal at L*.

**Sweep assessment**:
- λ was NOT systematically swept across ≥ 3 orders of magnitude. A single value λ=0.5 was used for the Method variant. The plan says "planned λ = 0.5; ablate in /auto-verify if C2 is borderline." The plan explicitly deferred λ sensitivity to /auto-verify.
- The plan does NOT specify a σ_proj-scaled sweep — λ here is an additive loss coefficient, not an activation-space intervention coefficient. The σ_proj-scaling convention is most applicable to additive activation-patching / RepControl interventions where the direction vector's norm varies across layers. For a loss-level regularizer (L_bottleneck is a cosine-distance term bounded in [0, 2]), the σ_proj scaling is less critical since the term is already on a normalized [0, 2] scale.
- Random-direction baseline: The Baseline (λ=0) serves as the null intervention; there is no "random-direction L_bottleneck" control (patching with a random orthogonal direction at L* instead of the actual EN/ZH/KO invariance direction). This would be a stronger specificity check for C2 but is beyond the current scope.
- A capability metric IS tracked alongside training: L_bottleneck convergence (0.31 → 0.04), DPO margin (+0.732 Method vs +0.590 Baseline), and M4 capability evaluations (MMLU, MGSM, MT-Bench) serve as the capability retention check.

**Verdict**: warn — λ was not swept; a single value (0.5) was used without pilot at {0.1, 0.3, 0.5, 1.0} or similar. The plan explicitly deferred this to /auto-verify (so no procedural violation), but a reviewer would flag that the C2 result is at one untested λ point and the sensitivity is unknown. The lr hyperparameter WAS piloted at {5e-6, 1e-5, 5e-5} via M3_pilot runs, showing good experimental rigor on the DPO-side hyperparameters.

---

## Check B–F — Reserved checks

All B–F: not_implemented (reserved for future mechanism-skill extensions).

Note on B (direction-extraction quality): The L*-anchor signal uses the model's own last-token residual states at layer 10 for the three languages (en, zh, ko) — no explicit direction vector is extracted or stored. The regularizer directly computes 1 - cos(h_L*(en), h_L*(zh)) etc. This is a loss-level representation alignment, not a stored direction.

---

## Overall Verdict

**overall_verdict**: warn

**Reason**: The λ coefficient for L_bottleneck was not swept (single value 0.5, deferred per plan), which means C2's result is at one untested operating point. This is a soft rigor concern — it does not invalidate the result but limits the claim's generalizability to the specific λ=0.5 choice. The lr was properly swept. C2 is admitted to Stage 2 with warn caveat on mechanism side.
