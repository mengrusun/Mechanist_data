# Experiment Tracker (plan-level; downstream updates rows in place)

**Behavior-source / Mechanism**: given / discovery. **resource_fidelity**: cost-aware. **M0 gate**: none.
**Committed mechanism**: Representation and Parameter Analysis / Steering features + Steering Vectors (see MECHANISM_ROUTING.md). **Model**: arcinstitute/evo2_7b (32 blocks, d_model=4096), bf16, single A800.

| Run ID | Milestone | Claim | Config | Status | Result | Notes |
|--------|-----------|-------|--------|--------|--------|-------|
| E1 | E1 eval harness | C2 | ESM2-650M SS-probe %H vs experimental-PDB DSSP %H | done | r=0.987, frameRec=1.00, Q3=0.858 | PASS (r≥0.7). Structure ref = experimental DSSP (gold); ESMFold unavailable (CDN). runs/E1_eval_harness |
| M1 | M1 locate directions | C1 | probe / contrastive on 1299 high-α vs low-α CDS windows, blocks {14,18,20,22,24,26,28} | done | probe_b26 AUROC=0.9999; contrastive~0.65 | SAE extractor unavailable (Goodfire SAE CDN download blocked). runs/M1_locate_directions |
| M2 (α sweep) | M2 dose-response | C1 | probe_b26, α∈[0,.5,1,2,4,8,16], seeds[42,43,44], N=120/α | done | %H 34.5→48.1 (α=16), Δ=+13.6, ρ=0.607, p_bonf=0.0024 | PASS partial-C1; validity=1.0, loglik improved. α escalated to 16 (tip-2). runs/M2_dose_response |
| M3 | M3 specificity | C1 | random control / off-target / naive baseline | done | random flat (no gain); off-target %E,valid intact; beats temp+rejection | PASS complete-C1. Caveat: GC 0.40→0.13. runs/M3_specificity |
| M4 | M4 scale library | C1 | probe_b26 α=16, 200 lib vs 200 baseline | done | %H 31.0→46.8 (median 23→50), uplift +15.8, valid 0.985→0.995 | capability demo; structure subset via fast predictor (ESMFold CDN-blocked). runs/M4_library |

Notes: alpha grid [0,0.5,1,2,4,8] (escalates to 16 if still rising at 8). method_sensitive fields re-bound at Phase 1.5 (sites {14..28}, n_pairs≈968 train windows, metric=fast %H + covariates; see MECHANISM_ROUTING.md ## Plan reconciliation). Model pinned to Evo2-7B (HARD). Realized data: 2043 clean PDB proteins, contrastive windows train 527α/441β, heldout 169α/162β.
