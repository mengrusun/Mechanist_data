# Auto Review — Cross-Modal Covert Transfer Study

`/auto-iteration-loop` audit trail. Chronological, append-only. Per-iteration entries follow the Phase E per-category template.

**Loop config**: `MAX_ITERATIONS=6, MAX_CLAIM_REENTRIES=2, TARGET_SCORE=6, AUTO_PROCEED=true, RESUME=false, GPU_ID=0,1,2,3`.

**Input from `/auto-verify`**: `verify_passed=[C1], verify_inconclusive=[C2]`.

---

## Iteration 1 (2026-07-18)

### Assessment (Summary)
- Score: 4/10
- Verdict: not ready
- Budget after this iteration: iterations 1/6, claim-reentries 0/2
- Key criticisms:
  - C2 mechanism audit FAILed on Q4 (no capability metric), Q8 (sign pattern broken: median ρ=+0.371), Q7 (n_random=1 underpowered).
  - Central mechanistic test not executed in a way that supports its stated interpretation.
  - BOUNDED NULL itself is publishable, but not while the null is confounded by protocol defects.
  - C1 (phenomenon) is credibly passed; robustness=1.00, method-swap min gap 9.0pp = 3x threshold.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## Score: 4/10

Why not lower: C1 looks genuinely nontrivial and currently robust enough to be interesting.
Why not higher: the mechanism claim is not review-ready. For a top venue, a phenomenon + under-audited mechanism is usually not enough, especially when the mechanism result is itself framed as a negative claim that still needs clean falsification protocol.

## 1) Top-venue assessment

This is **interesting but not submission-ready**.

- **Strength:** The phenomenon anchor C1 appears strong: per-seed effect, two controls, method-swap robustness, margin well above threshold.
- **Weakness:** The mechanism section is currently not at publication standard. The problem is not that the answer is null; nulls are publishable. The problem is that the null is not yet cleanly attributable to mechanism failure rather than protocol defects.
- **Biggest issue:** your current C2 evidence is confounded by:
  1. no capability/OOD metric,
  2. sign inconsistency that directly challenges the intervention setup,
  3. underpowered random-direction specificity control.

## 2) Minimum fix

**C2 — INCONCLUSIVE. Routing type: ② main-experiment-script fix.**

Required minimum components:

1. **Independent capability / degeneration metric at every α**
   - Minimum: retroactively compute and log `OTHER_rate` from existing per-item verdicts.
   - Resolves the "OOD collapse vs no recovery" confound provided reported jointly with accuracy over α.

2. **Negated-direction sweep** — most important fix
   - Given `v = mean(h_treated) - mean(h_Ctrl-B)`, observed positive median ρ threatens the directional interpretation.
   - Minimum: 7 α × 3 seeds with `-v`.
   - If `-v` gives expected monotone trend → original sign was wrong/documented flipped.
   - If neither works → null is stronger.

3. **Expand random-direction control**
   - Minimum: n_random ≥ 30 per matched setting.
   - Batched implementation fine if truly equivalent and preserves matching.

## 3) PASS claims

**C1 — PASS.** Robustness = 1.00, 1/1 eligible variants pass. Method-swap min gap 9.0pp = 3× threshold. Credibly passed. Caveat: paper should include per-seed treated-vs-Ctrl-A and treated-vs-Ctrl-B numbers separately (all-3-pass can hide asymmetry).

## 4) Ready for submission?

**No.** C1 passes; C2 inconclusive. If the three C2 fixes land clean, could move to Almost / Yes depending on what negated-direction sweep shows.

## Additional review

If paper narrates the mechanism too confidently, reviewers punish it. "Identified direction but intervention fails to recover" is not supportable while sign appears wrong and capability collapse unmeasured. Rescue path: clean result showing no recovery under either sign except where capability collapses; random directions near zero; conclusion the simple low-rank residual direction does not localize the transferred unsafe behavior.

## Memory update
[copied to REVIEWER_MEMORY.md]

</details>

### Verify-Passed Claims (brief audit)
- C1: consistency check PASS. Robustness=1.00, method-swap min gap 9.0pp = 3× threshold. Paper caveat noted — include per-seed treated-vs-Ctrl-A and treated-vs-Ctrl-B numbers separately (asymmetry check).

### Actions Taken (per claim, per type)

- **C2 — type ② — main-experiment mechanism-rigor fix (three sub-fixes)**
  - **(A) Retroactive capability metric extraction** — new script `scripts/extract_capability_metric.py` reads existing `M2_2b_alpha*_seed*.json` and `M2_2c_randdir_alpha*_seed*.json` per_item blocks, computes `other_rate` (fraction of judge verdicts = OTHER), patches `capability_metric` block back into each source JSON. Also emits per-seed tables `results/mech/CAPABILITY_M2_2b.json` + `CAPABILITY_M2_2c.json`. **No GPU cost.**
    - **Result** (M2.2b real-dir sweep, other_rate per seed × α):
      - seed42: min=0.060 (α=-1), max=0.105 (α=-0.5), range=4.5pp
      - seed123: min=0.075 (α=-2), max=0.105 (α=0.5), range=3.0pp
      - seed2026: min=0.060 (α=+1), max=0.105 (α=+0.5), range=4.5pp
    - **Interpretation**: OTHER_rate is FLAT across α — no OOD-collapse signature. Capability preserved throughout the α sweep. The BOUNDED NULL cannot be attributed to model degeneration.
  - **(B) Sign-flip sweep (M2.2b_neg)** — new script `scripts/steer_and_eval_signflip.py` runs the identical 7-α × 3-seed sweep but with `v' = -v` at every top-K layer. Also logs `other_rate` at generation time. Launched via `scripts/launch_iter1_fixes.sh signflip` on GPUs 0,1,2,3. Status: in progress at Phase E timestamp.
  - **(C) Batched n_random=30 (M2.2c_v2)** — new script `scripts/steer_batched_randdir.py` loads model once per (α, seed), sequentially installs 30 independent random-direction hooks, evaluates QA_I after each. Focused on α=+1 for all 3 seeds (moderate steering, most informative for null-distribution test). Also emits per-direction `other_rate`. Launched via `scripts/launch_iter1_fixes.sh randbatch`. Status: scheduled after signflip completes.
  - **(D) Aggregate v2** — new script `scripts/aggregate_mechanism_v2.py` reads both sign conventions, picks the "best-sign" ρ per seed, combines with capability + n=30 specificity; emits `results/MECHANISM_VERDICT_v2.json`.

  - **Plan Before/After** (`refine-logs/EXPERIMENT_PLAN.md`):
    - Before: M2.2b, M2.2c, M2.2d specified; no sign-flip; no capability metric; n_random=1.
    - After: M2.2b_neg + M2.2c_v2 + M2.2d_v2 added as iteration-1 addenda after the M2.2d section; retroactive capability extraction documented as prerequisite for M2.2d_v2. Original M2.2b/M2.2c/M2.2d unchanged (their raw JSONs are the source data for the capability retro-patch).
  - **Re-invoked**: (planned after all sub-fixes complete) `/auto-verify C2 --resume false` to re-audit with augmented mechanism evidence.

### Claim Rewrites (type ③ — none this iteration)
- none

### Claim-Stage Re-entries Triggered (orchestrator handoff — none this iteration)
- none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- none (verify_integrity_only bucket was empty)

### Results
- [run-experiment] iteration=1 runs_this_iteration=21+3+0 = 24 (21 signflip + 3 batched-randdir; retro-cap is CPU) gpu_hours_this_iteration=~10 cumulative_gpu_hours=~10
- Task A (capability extraction): completed CPU-only, ~2 sec.
- Task B (signflip, 21 runs): in progress, ETA ~2h from launch.
- Task C (batched randdir, 3 runs × 30 directions × 133 items each): scheduled sequentially after B, ETA ~4.5h.
- Task D (aggregate v2): scheduled after B+C.
- Task E (`/auto-verify C2 --resume false`): scheduled after D.

### Status
- awaiting local completion of Task B/C/D, then Task E → next Phase A.
