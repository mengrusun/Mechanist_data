## C2: method-swap-crp-compose  →  PASS

**Claim**: C2 — Embedding those reference inputs with a frozen multi-modal foundation model (CLIP image tower) and pooling the embeddings yields a single vector v_c placing c in the foundation model's joint image-text semantic space, such that v_c can be used as a text query over learned components.

**Main-experiment verdict**: supported (MRR=0.898, R@10=0.974, perm95=0.0097, stability_median_cos=0.970, layer4 Cohen's d=1.96)

**Robustness**: 1.00  (1 pass / 1 eligible variant)  
**Threshold**: 0.50  
**Verdict**: PASS (robustness 1.00 ≥ threshold 0.50)

---

### Variants

| Variant | Dimension | pass_fail | integrity_status | MRR | perm95 | sig | stability_cos | delta_mrr |
|---------|-----------|-----------|-----------------|-----|--------|-----|---------------|-----------|
| method-swap-crp-compose | method | **pass** | WARN | 0.9024 | 0.00968 | True | 0.9581 | +0.0044 |

**N_run**: 1  
**N_eligible** (integrity WARN or better): 1  
**N_pass**: 1  
**N_fail**: 0  

### Variant details: method-swap-crp-compose

**Method swapped**: CLIP-Dissect plain mean-pool → CLIP-Dissect + Zennit-CRP compose (LRP-epsilon relevance cropping before CLIP embedding)

**Result (Phase 8 judgment)**: claim_supported=True, consistent_with_main_experiment=True

**Key metrics**:
- P2a MRR=0.9024 (vs. main 0.898, delta=+0.0044)
- P2a R@10=0.976 (vs. main 0.974)
- P2a perm95=0.00968 → significant=True (MRR >> perm baseline by 93x)
- P2b stability median_cos=0.9581 (CI=[0.9567, 0.9597], passes=True)
- n_valid_components=1000/1000 (no zero v_c vectors)

**Reasoning (LLM judge, gpt-5.4)**: The variant strongly supports C2 — retrieval remains high (MRR=0.9024, R@10=0.976), significance holds, and stability passes (median cosine=0.9581). It is consistent with the main experiment because performance is slightly better than main (+0.0044 MRR) while preserving the same conclusion that the embedding lies in a text-queryable joint semantic space.

**Interpretation**: CRP-crop preprocessing (isolating component-relevant image regions via LRP-epsilon before CLIP embedding) does not degrade — and marginally improves — the text-queryability of v_c. This result is strong evidence that C2's claim (v_c places components in the joint image-text space) is robust to within-family method variation and is not driven by background/scene correlations in uncropped images.

### Variant integrity (Phase 9)

| Variant | exp_audit | mech_audit | combined | eligible |
|---------|-----------|------------|----------|----------|
| method-swap-crp-compose | WARN | N/A | WARN | YES |

- WARN source: scope narrower than full claim (fc-only, CRP-crop preprocessing) — expected for a method-swap variant; does not disqualify
- N/A mechanism: CRP-compose is observational (no steering/patching); treated as PASS for gate

### Stage-2 selection

C2 was the picked claim under MAX_VERIFY_CLAIMS=1 (over C1) by Phase 3 step 0 importance judgment. C1 is INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap).

### GPU pin verification

Physical GPU 1 (UUID GPU-6d73793d-0247-8c38-a691-4f59dd2281cf) used; compute verified via nvidia-smi (1500 MiB). GPU 1 ∈ allowed set {1,2,3,5,6}. CUDA_VISIBLE_DEVICES="1,2,3,5,6" passed to run. Pin propagation: PASS.

Wall time: 2690s (44.8 min) for full recovery run (P2a + P2b). Original LRP loop: 3256s (54.3 min, completed before Phase 7 crash). Total GPU time for variant: ~99 min.
