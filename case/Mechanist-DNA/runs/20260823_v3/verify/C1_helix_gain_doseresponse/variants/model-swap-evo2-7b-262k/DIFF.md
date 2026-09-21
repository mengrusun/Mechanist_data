# Variant DIFF — model-swap-evo2-7b-262k (C1)

**Dimension**: model. **Swap**: `evo2_7b` (1M-context checkpoint, `configs/evo2-7b-1m.yml`) → `evo2_7b_262k` (262k-context checkpoint, `configs/evo2-7b-262k.yml`). Both are 7B StripedHyena Evo2 checkpoints, 32 blocks, hidden 4096 — a **within-family** model swap on a genuinely different training/context checkpoint (different rotary base/interpolation).

**What changed vs `experiments/steer_helix/run_m3.py` (minimum diff):**
- `Evo2Wrapper` loads `evo2_7b_262k` from `/mnt/quarkfs/share_models/evo2_7b_262k/evo2_7b_262k.pt` with its own registered config, instead of `evo2_7b` from `/mnt/quarkfs/share_model/evo2_7b/evo2_7b.pt`. Nothing else in the wrapper changes.
- **CAA direction rebuilt in the 262k model's own activation space** over the SAME M1 DEV contrast DNA (`results/m1/contrast_set.json`, 300 high / 300 low, DEV-split). The behavioral contrast data is model-independent; the direction is model-specific — this is the correct model-swap protocol.
- **Budget scoping (documented, applied to the variant's own coef-0 baseline too, so within-variant comparison stays apples-to-apples):** coefs `[0.0, 1.0, 2.0, 4.0]` (vs `[0,0.5,1,2,4,8]`), seeds `[0,1]` (vs `[0,1,2]`), `n=150`/cell (vs 250). 1200 gens total vs 4500. Reduction is purely cost-driven (verify budget ≤ ~8 GPU-h); no metric or method change.

**What is held FROZEN (identical to the main experiment):**
- Steering site (block 28), method (additive CAA diff-of-means hook), coefficient unit (raw multiplier on the diff-of-means vector).
- Frozen ESMFold→DSSP α-helix assay (structure predictor is model-independent), pLDDT policy, helix codes H/G/I.
- Deterministic longest-ATG 6-frame ORF rule (30–300 aa), validity composite, primer set, generation settings (900 tok, T=1, top-k=4).
- **Anti-circularity**: CAA built on DEV contrast; evaluated on TEST-split primers with the identical TEST seed offset `900000 + seed*1000 + call`.

**Code review**: external llm-chat reviewer MCP unavailable in this context → `[pending external review]`; CC self-review found no CRITICAL/MAJOR issue (single intended model swap; grid reduction documented and symmetric across coef-0 baseline; evaluation uses model-independent structural GT; no leakage).

**Success criterion (per the frozen C1 claim):** the swapped model is judged to support C1 if it reproduces (a) a positive monotone dose-response of helix_all in the coefficient (Spearman ρ > 0, one-sided positive) AND (b) a helix gain at a low/winning coefficient over its own coef-0 baseline. Direction (supported) matches the main experiment → `consistent_with_main_experiment = claim_supported`.
