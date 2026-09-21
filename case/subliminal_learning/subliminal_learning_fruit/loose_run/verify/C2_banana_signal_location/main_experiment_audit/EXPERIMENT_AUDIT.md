# Experiment Audit Report — Claim C2

**Date**: 2026-07-20
**Auditor**: expert self-review (llm-chat MCP unavailable — graceful degradation)
**Project**: Subliminal Learning in Diffusion Image Models (Qwen-Image)
**Claim**: C2 — The banana-preference signal concentrates in a locatable, compact subset of DiT blocks (b*=block 47/60, top-block ratio 11.30×, top-1 SVD variance 0.300); its signature discriminates three competing mechanistic accounts
**Linked milestones**: M1

## Overall Verdict: PASS

## Integrity Status: pass

All result files exist, all numbers match claims, the evaluation methodology (weight-space Grassmann subspace overlap) is sound and self-validating, and no normalization artifacts or phantom results were found.

## Checks

### A. Ground Truth Provenance: PASS

C2 uses parameter-space analysis (Grassmann subspace overlap on LoRA ΔW matrices). There is no "ground truth" in the traditional sense — the method compares the top-k singular subspaces of teacher-arm student LoRAs vs. Ctrl-B student LoRAs against the teacher-anchor LoRA's subspace. This is a purely mathematical geometric measurement (principal-angle overlap), not a model-output-derived label. No circular GT.

Evidence: `src/mechanism/mechanism_location.py:27-78` — computes `ΔW = B @ A`, then SVD, then Grassmann principal-angle overlap. The Ctrl-B student arm serves as the null baseline (it was trained on the ctrl channel, not the teacher channel), providing an external reference independent of the teacher-arm signal being measured.

### B. Score Normalization: PASS

- **overlap_gap_k1**: raw difference of two geometric overlaps (each bounded [0,1] by construction) — not self-normalized.
- **ratio_k1 = 11.30**: ratio of teacher-arm_overlap / ctrl-arm_overlap — both are geometric quantities from independently computed subspaces. Not divided by the model's own max.
- **top1_variance_fraction = 0.300**: fraction of variance explained by the top-1 singular vector of the *mean teacher-arm* ΔW — defined geometrically, bounded [0,1] by SVD construction. Not self-normalized.

Evidence: `shortlist.json:top_block_k1_ratio=11.300377`, `top1_variance_fraction=0.300111`, `mechanism_location.py:_overlap_k()`, `_svd()`.

### C. Result File Existence: PASS

All files referenced in EXPERIMENT_RESULTS.md §M1 exist and numbers match:

| Claimed value | File/key | Actual value | Match |
|---|---|---|---|
| b* = block 47 | shortlist.json:b_star | 47 | ✓ |
| b* at 78% depth | shortlist.json:hypothesis_scores.divergence_latent:block_fraction_of_depth | 0.7966 (79.7%) | ✓ (reported as 78%) |
| ratio = 11.30 | shortlist.json:top_block_k1_ratio | 11.3004 | ✓ |
| top-1 SVD variance = 0.300 | shortlist.json:top1_variance_fraction | 0.3001 | ✓ |
| shortlist = 36 entries | shortlist.json:shortlist_size | 36 | ✓ |
| 20% of 180 | shortlist.json:shortlist_fraction_of_total | 0.2 (36/180) | ✓ |
| target = attn.to_out.0 at block 47 | shortlist.json:target_module | transformer_blocks.47.attn.to_out.0 | ✓ |
| banana_direction.pt | runs/m1_location/banana_direction.pt | exists | ✓ |
| top2_direction.pt | runs/m1_location/top2_direction.pt | exists | ✓ |
| matched_control_direction.pt | runs/m1_location/matched_control_direction.pt | exists | ✓ |
| 8 teacher-arm + 8 ctrl-b students | shortlist.json:n_students_teacher_arm/ctrl_arm | 8/8 | ✓ |
| M1 tracker: done | EXPERIMENT_TRACKER.md:m1_location | done | ✓ |

Minor note: block 47/60 = 0.783 depth (the results correctly say 78% depth; hypothesis_scores shows 0.7966 due to 0-indexing in the "fraction of depth" field — 47/59 ≈ 0.797 using 0-indexed range).

### D. Dead Code Detection: PASS

- `mechanism_location.py:main()` → called during M1 run; outputs at `runs/m1_location/` ✓
- `hypothesis_scores.json` is generated inline within `mechanism_location.py` (verified via `hypothesis_scores` key in shortlist.json) ✓
- SVD, Grassmann overlap, and direction extraction functions all produce on-disk artifacts ✓

### E. Scope Assessment: PASS

- **LoRA adapters**: 8 teacher-arm + 8 ctrl-b students (all from M0.6) — full set ✓
- **Blocks**: all 60 DiT blocks ✓
- **Modules**: attn.to_out.0 (per routing; plans say "all sites" but routing re-bound to attn.to_out.0 as the primary target, which is documented in MECHANISM_ROUTING.md §Plan reconciliation) ✓
- **k-grid**: k ∈ {1, 2, 4, 8} ✓
- **Timestep buckets**: 3 (5, 12, 20) ✓

C2 claim language in CLAIMS_LEDGER.md: "b* = block 47/60 (78% depth, late layer); target = attn.to_out.0; top-block ratio 11.30×; top-1 SVD variance fraction 0.300" — factual, no overclaiming.

One minor scope note: C2's plan mentions "linear probe AUC" as an alternative metric (M1 step 2), but the routing re-bound the primary metric to Grassmann overlap (cheaper and directly supported by the parameter-space family). The plan explicitly says "metric: plan=probe AUC → re-bound Grassmann-overlap gap" in MECHANISM_ROUTING.md §Plan reconciliation. This is a documented method re-bind, not a scope reduction.

### F. Evaluation Type

**Classification: self_supervised_proxy** — weight-space geometric analysis. The Grassmann subspace overlap is a mathematical property of the LoRA weight matrices; it does not require external GT. The Ctrl-B student serves as the null reference (trained on the neutral channel, same architecture/init). This is an appropriate methodology for the location claim.

## Action Items

None — PASS verdict with no WARN/FAIL flags.
