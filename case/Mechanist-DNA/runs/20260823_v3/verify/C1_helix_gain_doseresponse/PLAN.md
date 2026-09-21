## Claim C1: Under a targeted internal intervention during Evo2-7B decoding, the encoded protein's mean α-helix fraction rises significantly above the unintervened baseline, with a monotone dose-response.

### Main experiment (from /auto-experiment)
- Method: site-28 contrastive activation addition (CAA), coefficient dose grid; Dataset: Evo2-7B DNA generations, ESMFold→DSSP helix assay, DEV/TEST split; Model: **evo2_7b** (1M-context checkpoint) → helix_all 0.482 vs baseline 0.414 (+0.068), dose-response ρ=0.922, q=1.5e-4.

### Dimensions to test
`DIMENSIONS = model` → exactly one variant (model swap). `method` and `dataset` axes not requested this pass.

### Variants
| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | **evo2_7b_262k** (262k-context Evo2-7B checkpoint, `configs/evo2-7b-262k.yml`) | evo2_7b (1M-context, `configs/evo2-7b-1m.yml`) | Genuinely different training/context checkpoint of the *same* 7B StripedHyena family (same 32-block/4096-hidden architecture, so the site-28 CAA hook transfers cleanly). Tests whether the located α-helix steering direction and its dose-response are a property of the Evo2-7B family or an artifact of the one 1M checkpoint. Cheapest genuine within-family model swap loadable within the verify budget (loads in 40s; confirmed via `verify/variant_smoke/smoke_262k.json`). | `/mnt/quarkfs/share_models/evo2_7b_262k/`; NOTICE alternate-Evo pointer |

**Within-family constraint satisfied:** both are Evo2-7B checkpoints (Representation & Parameter Analysis / Steering Vectors family, same architecture). Not a cross-family swap.

### Budget scoping (verify ≤ ~8 GPU-h; ~12.7 GPU-h remained this round)
site-28 CAA only, coefs `[0.0, 1.0, 2.0, 4.0]` × seeds `[0,1]` × n=150 = 1200 gens (vs main 4500). Reduction is purely cost-driven and applied to the variant's own coef-0 baseline too. Projected ~1.5 GPU-h. Model axis **RAN** (not excluded) — the fallback method-axis check (length-regressed endpoint / alt pLDDT policy on existing gens) was therefore not needed.

### Success Criterion (per variant)
C1 is supported by the variant iff the swapped model reproduces (a) a positive monotone dose-response of helix_all in the coefficient (Spearman ρ>0, one-sided positive) AND (b) a helix gain at a low/winning coefficient over its own coef-0 baseline. Main-experiment verdict on C1 = supported → `consistent_with_main_experiment = claim_supported` (no flip).

### Reviewer critique (Phase 4)
External llm-chat reviewer MCP unavailable in this execution context → `[pending external review]`. CC self-critique: the swap is a genuine independent test (different pretraining checkpoint, not a cosmetic re-run); it controls the right confound (does the direction survive a different model of the same family); no stronger within-family model swap is available offline within budget (only `evo2_7b` and `evo2_7b_262k` 7B checkpoints are present; `evo2_7b_base` config exists but no separate base checkpoint on disk). Accepted.
