## C1: robustness = 1.00 (threshold = 0.50, eligible = 1/1)  →  ✅ PASS

- swap_variants_run: true
- Main-experiment verdict on C1: supported
- Main-experiment integrity (Phase 2): PASS (exp PASS / mech PASS)
- Variant counts (over `consistent_with_main_experiment`): 1 pass, 0 fail
  (of 1 eligible; 0 excluded for integrity reasons)
- **Model dimension** (evo2_7b → evo2_7b_262k, within-family Evo2-7B checkpoint swap): **matches the main experiment** (consistent=pass, claim_supported=pass). The swapped 262k-context checkpoint reproduces C1 on every main-experiment criterion:
  - Dose-response: Spearman ρ=0.878, permutation one-sided p=0.0057 (main: ρ=0.922, q=1.5e-4) — significant positive monotone trend.
  - Winning coef 1.0 vs in-model baseline: helix_all 0.521 vs 0.426 = **+0.094** (Cliff's δ=0.201, Mann-Whitney p=1.05e-5); main experiment +0.068. Variant gain is slightly *larger*.
  - Monotone ladder: 0.426 → 0.521 → 0.541 → 0.565 (coef 0/1/2/4); peak +0.138 at coef 4 (main: +0.161).
  - CAA direction rebuilt in the 262k model's own activation space (vnorm 947 vs base 824 — analogous scale); anti-circularity preserved (DEV build, TEST-split eval).

**Interpretation.** The central scientific claim — a located CAA direction, added during Evo2-7B decoding, raises the encoded protein's α-helix fraction with a monotone dose-response — is **robust across the within-family model swap**. The effect is not an artifact of the single 1M-context `evo2_7b` checkpoint; it transfers, with the same direction and same shape, to the independently-trained 262k-context checkpoint. **Caveat that also transfers:** the off-target length-shortening and GC-drop signature reappears on the swapped model (protein length 94.5→46.8 aa, GC 0.456→0.257 across the sweep) — confirming the length/GC coupling is a robust family-level property of the CAA lever, consistent with the C2 caveat (documented, non-disqualifying; C1 survives length-matching in the main experiment). Next iteration: none required for C1 on the model axis; if broadening, a method-axis (SAE-clamp with a working feature) or dataset-axis (alternate primer distribution) swap would test the remaining axes.
