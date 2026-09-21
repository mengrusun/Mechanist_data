# Experiment Audit — C2 Variant: method-swap-crp-compose (Phase 9)

**Date**: 2026-07-14  
**Auditor**: llm-chat (gpt-5.4, https://www.dmxapi.cn/v1)  
**Phase**: 9 (Variant integrity audit)  
**Claim**: C2 — Pooled frozen-CLIP vector v_c places c in the joint image-text semantic space  
**Variant**: method-swap-crp-compose (CRP-crop compose, method swap)  

## Overall Verdict: WARN

One scope caveat — the tested variant is a modified preprocessing (CRP-crop before CLIP embed), which narrows the scope to fc components under CRP conditions. No methodological integrity issues found.

## Checks

| Check | Status | Detail |
|-------|--------|--------|
| GT provenance | PASS | fc unit index = ImageNet class label by construction; enforced by assert in code; not externally fabricated or model-output-derived |
| Score normalization | PASS | L2-normalized cosine; permutation baseline is independent random shuffle; no improper normalization |
| Result existence | PASS | result.json, result_p2a.json, result_p2b.json, v_c_crp.h5, verdict.json all on disk; metrics internally coherent (mrr=0.9024, perm95=0.00968, stability=0.9581, n_valid=1000/1000) |
| Dead code | PASS | compute_mrr, permutation_upper, compute_stability all called in v_crp_recover.py and outputs verified in result.json |
| Scope | WARN | Variant supports robustness of C2 under CRP-crop composition for fc components; is a method-swapped preprocessing variant, not direct confirmation of the unmodified general claim across all component types |

## Notes

The scope WARN is expected and appropriate for a method-swap variant: the variant deliberately uses CRP-crop preprocessing (a within-family method swap), so the conclusion is necessarily narrower than the full claim. This does NOT constitute a disqualifying integrity failure — it is the intended scope of a method-swap robustness probe. The integrity_status = WARN does not exclude this variant from the robustness computation (only FAIL would).
