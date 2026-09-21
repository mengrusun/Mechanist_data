# Auto Review — Round 2

Autonomous adversarial review loop for **Hardening the α-Helix Feature-Steering Knob in Evo2-7B**.
Reviewer LLM: `gpt-5.4` via `https://www.dmxapi.cn/v1` (resolved from shell env).
Config: `MAX_ITERATIONS=6, MAX_CLAIM_REENTRIES=2, TARGET_SCORE=6, AUTO_PROCEED=true, GPU_ID=auto, LEDGER_FIGURES=auto`.

## Iteration 1 (2026-07-20)

### Assessment (Summary)
- **Score**: 6.5/10 (top ML mech-interp track) / 7/10 (strong genomics/protein-methods venue if framed narrowly)
- **Verdict**: `almost` (canonical, mapped from reviewer's "almost there" / "not fully ready" language)
- **Budget after this iteration**: iterations 0/6, claim-reentries 0/2 (no back-edge action fired)
- **Key criticisms**:
  1. All causal readouts remain predictor-mediated (ESMFold/OmegaFold, not experimental structure).
  2. C1 train/test leakage details (PDB redundancy, homology-family splits) not documented in the writeup so far.
  3. Capability preservation metric is thin (valid-ORF only; missing composition drift, perplexity, disorder, length distribution).
  4. C3 empirical one-sided p is floored at ~0.02 by n_null=48; z-score much stronger — needs clean explanation, more nulls if affordable.
  5. Matched-control construction must be audit-proof in the writeup.
  6. β-arm negative must remain framed as helix-axis specificity, not slip back to symmetric-axis language.
  7. Missing H-only per-sample storage → no cluster-bootstrap CI for H-only variant (minor but avoidable).

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## 1. Score for a top venue

**6.5/10 for NeurIPS/ICML/ICLR mech-interp**

**7/10 for a strong genomics/protein-methods venue, if framed narrowly and honestly**

Bluntly: this is **much better than round 1**, and it now looks like a **real paper rather than an intriguing but under-hardened claim**. The causal-steering story is substantially stronger: dual predictors, pLDDT-aware endpoint, swap robustness, interior-dose selection, stronger nulls, explicit negative resolution of the β arm. Those are exactly the kinds of repairs I would have asked for.

That said, I still would **not** call this an easy accept at a top ML venue. The reason is not obvious sloppiness; the reason is that the core evidence remains **proxy-causal**, mediated through **structure predictors rather than experimental structure**, and the "mechanism" is still only partially mechanistic in the strong interp sense. You have a **distributed feature set that predicts helix labels and whose joint steering shifts predicted secondary structure**. That is good. But top-tier mech-interp reviewers will still ask whether this is a **true internal causal variable for protein helix propensity** versus a steering direction that pushes sequence statistics exploited by ESMFold/OmegaFold.

So:
- **Strong "almost there" paper**
- **Publishable somewhere good**
- **Borderline for top ML unless writing is extremely disciplined and claims are narrow**

## 2. FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS claims

**None.** Given verify state, there are no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS claims in this iteration.

## 3. PASS-claim narrative + numeric consistency checks

**C1** — set-level distributed encoding claim is coherent (Prokaryote AUROC HGI ~0.897 / H-only ~0.906; Eukaryote HGI ~0.865 / H-only ~0.868; confound-only 0.506; shuffle-null 0.53). Cleanest of the three. Caveats to add in paper: AUROC is not causality (encoding claim only); label provenance (PDB selection, redundancy filtering, homology-family splits, codon-to-residue mapping) must be exact; cross-organism ≠ cross-distribution.

**C2** — improved most. σ_proj-unit dosing + held-out trend + capability-preserved interior optimum is exactly what was needed; corrected c* rule (contiguous-preserved-region argmax) is the right one, and the transparent correction actually strengthens the paper. Numeric consistency clean: baseline 0.454, small doses flat, 10.74→0.479, 21.48→0.593; ESMFold ρ=0.867 p=0.0012; OmegaFold ρ=0.879 p=0.0008; Δ+0.129/+0.135 with tight cluster-bootstrap CIs (283 clusters). H-only slightly stronger (Δ 0.144/0.147) is reassuring. Caveats to add: monotonicity language must be scoped to the pre-specified capability-preserved segment (full curve is non-monotone under degradation); capability metric (valid-ORF) is weakly specified — add composition drift, perplexity, disorder, length; structure-predictor dependence remains central vulnerability; selection-on-one-seed / evaluate-on-two is thin but acceptable.

**C3** — strongest hardening component. 48-direction norm-matched null + matched control + dual predictors + 0/48 null≥S under both is the right response to skepticism. Numeric consistency clean: ESMFold Δ+0.129 [0.100, 0.159], matched -0.031, null mean ≈ -0.004, null max 0.056, empirical p=0.020, z=5.30; OmegaFold Δ+0.135, matched -0.025, null max 0.059, z=5.80. Note: empirical p floored at ~1/(48+1)=0.0204; z uses null mean/variance and is stronger. β-arm resolution is honest and correct — do not rescue rhetorically. Caveats: finite-null resolution; matched-control construction must be transparent and audit-proof; β-arm negative means helix-axis specificity, not disentangled axes; missing H-only per-sample storage is an avoidable credibility nick.

## 4. Ready for submission?

**Canonical verdict: almost**

Not "not ready" — this is materially stronger than that. Not fully "ready" either for a top ML venue with minimal risk. Remaining risks are paper-level validity risks, not failed claims: (1) predictor-mediated readouts, (2) capability preservation narrow, (3) β-arm negative limits conceptual scope, (4) some reproducibility/presentation details still matter, (5) mechanistic claim should stay narrow. If submitted to a top ML venue expect a mixed review set; if submitted to a strong genomics/protein/interpretability-adjacent venue with disciplined claims, much more competitive.

## 5. Memory update

[See `REVIEWER_MEMORY.md` iteration 1 section — 10 numbered items covering predictor-mediated causality, C1 train/test leakage risk, capability preservation breadth, C3 finite-null resolution, matched-control construction transparency, β-arm framing discipline, mechanistic depth, trivial biological confound exclusion, H-only per-sample storage, and title/abstract language discipline. All 10 carried forward as unresolved — they are paper-narrative and future-work items, not this-iteration action items.]

</details>

### Verify-Passed Claims (brief audit)
- **C1_helix_feature_set_exists**: consistent — set-AUROC 0.897 prok / 0.865 euk (HGI) vs 0.506 confound / 0.53 shuffle-null; H-only swap holds (0.906 prok / 0.868 euk). Paper-side caveat to surface: label-provenance (PDB redundancy filtering, homology-family split policy) needs explicit documentation.
- **C2_dose_response_helix**: consistent — ESMFold ρ=0.867 p=0.0012, OmegaFold ρ=0.879 p=0.0008; Δ(c*=21.48)=+0.129/+0.135 with cluster-robust CIs excluding 0; H-only Δ slightly larger (0.144/0.147). Paper-side caveat: monotonicity language must be scoped to capability-preserved segment; add non-structural quality diagnostics (composition, perplexity, disorder, length) beyond valid-ORF.
- **C3_specificity_causal_knob**: consistent — 0/48 null≥S both predictors; ESMFold z=5.30, OmegaFold z=5.80; H-only z=5.83/6.13 strengthens under swap; matched-control Δ negative; helix rise not a pLDDT artifact (S pLDDT LOWER than baseline). Paper-side caveats to surface: (a) empirical p floored at ~0.02 by n_null=48 — explain the z-vs-p relationship; (b) matched-control construction must be transparent; (c) β-arm negative → helix-axis specificity framing (do not slip to symmetric-axis language); (d) H-only per-sample values not stored → no cluster-bootstrap CI for the H-only variant (empirical p + z remain valid).

### Actions Taken (per claim, per type)
- **none — no back-edge fired**. All three claims verify-PASS with clean integrity and robustness 1.00; reviewer score (6.5) meets TARGET_SCORE=6; canonical verdict "almost" ∈ POSITIVE_VERDICT_TERMS. Every remaining reviewer concern is paper-narrative-level (documentation caveats, future-work items) — not variant-integrity (①), not main-experiment-methodology (②), not claim-scope (③). Per the user's stopping rule: "the science is strong, so weigh cost against value; a documented caveat is often the right call, not a re-run." Caveats are captured in `REVIEWER_MEMORY.md` and surfaced in the final ledger's per-claim `caveats` field.

### Claim Rewrites (type ③)
- **none** — no claim rewrite this iteration.

### Claim-Stage Re-entries Triggered (orchestrator handoff)
- **none** — no upstream calls queued.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **none** — verify_integrity_only bucket was empty at loop entry (all 3 admitted claims were also picked at Stage 2; MAX_VERIFY_CLAIMS=3=|ADMITTED|).

### Results
- **[run-experiment] iteration=1 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=0** — no experiments fired this iteration.
- Per-claim outcome: C1 → held PASS (robustness=1.00); C2 → held PASS (robustness=1.00); C3 → held PASS (robustness=1.00).

### Status
- **completed** — three-dimensional STOP fired (score 6.5 ≥ 6, verdict "almost" ∈ {ready, almost}, zero claims in FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS). Proceeding to Termination → final ledger + figures + research_memory.json update.
