# Figures Index — ESMFold 折叠躯干 β-hairpin 机制的三-claim 因果验证

Generated at the final ledger hook of `/auto` (2026-07-15).

Per-claim figure sets: 3 claims / 5 figures / 4 image + 1 table / 0 render-error / 0 skipped.

## C1 — Early-block localization + `s` is the active locus

### c1_effect_by_band — Δ hairpin-rate by s-patching band vs baseline

![Δ hairpin-rate by s-patching band vs baseline — early bands (b_0_3, b_4_7, b_8_11) show a large drop (~−0.7 to −0.86), while late bands (b_24_31, b_32_39) barely move (~−0.09). Same-window z-patch and matched-mask s-patch controls stay near zero across all bands.](C1/c1_effect_by_band.png)

Vector: [`C1/c1_effect_by_band.pdf`](C1/c1_effect_by_band.pdf) · Script: [`C1/gen_c1_effect_by_band.py`](C1/gen_c1_effect_by_band.py)

### c1_specificity_controls — C1 specificity

![C1 specificity — early s-patch (b_0_3) Δ=−0.862 vs same-window z-patch Δ=−0.011 (1.3%), matched-mask s-patch Δ=−0.067 (7.7%), and late-window s-patch (b_32_39) Δ=−0.085 (9.9%). All three matched controls change hairpin rate by ≤10% of the primary effect.](C1/c1_specificity_controls.png)

Vector: [`C1/c1_specificity_controls.pdf`](C1/c1_specificity_controls.pdf) · Script: [`C1/gen_c1_specificity_controls.py`](C1/gen_c1_specificity_controls.py)

---

## C2 — seq2pair as critical s→z channel (evidence-incomplete)

### c2_conditions_status — M2 evidence completeness

| Condition | N (records written) | Δ hairpin rate | Wilcoxon p | Status |
|-----------|--------------------:|---------------:|-----------:|--------|
| `seq2pair_donor` | 176 | -0.017 | 0.083 | complete |
| `pair2seq_donor` | 3 | — | — | hook shape bug |
| `seq2pair_zero` | 3 | — | — | hook shape bug |
| `seq2pair_matched_ctrl` | 3 | — | — | hook shape bug |

_Note: `pair2seq_donor`, `seq2pair_zero`, and `seq2pair_matched_ctrl` failed to write full records due to a `pair_to_sequence` hook shape-mismatch bug in `scripts/m2_worker.py` (expected `[B, L, S]`, actual `[B, num_heads, L, L, 32]`). The three specificity-control cells above are therefore evidence-incomplete, not scientifically null. Fix + re-run recommended in iteration._

LaTeX: [`C2/c2_conditions_status.tex`](C2/c2_conditions_status.tex) · Script: [`C2/gen_c2_conditions_status.py`](C2/gen_c2_conditions_status.py)

---

## C3 — Charge linear encoding + causal steering (partial)

### c3a_probe_by_block — M3a linear charge probe by block

![M3a — 3-class charge probe balanced-accuracy + AUROC by early block; all 4 sampled blocks (0, 2, 4, 6) achieve bacc=1.0 and AUROC=1.0 with permutation p<0.001. Chance = 1/3 (dotted line).](C3/c3a_probe_by_block.png)

Vector: [`C3/c3a_probe_by_block.pdf`](C3/c3a_probe_by_block.pdf) · Script: [`C3/gen_c3a_probe_by_block.py`](C3/gen_c3a_probe_by_block.py)

### c3b_dose_response — additive v_charge steering: decodable ≠ causally sufficient

![C3b two-panel: (a) target-region hairpin rate stays flat ≈0.99 across β for both same and opposite configurations — a real null (175/179 chains hairpin=1 throughout the sweep); (b) mean ΔpLDDT vs β confirms the perturbation itself is non-zero, so the null in (a) is NOT a no-op — steering along v_charge does perturb the residual stream but does not deflect downstream folding decisions. Illustrates decodability ≠ causal sufficiency.](C3/c3b_dose_response.png)

Vector: [`C3/c3b_dose_response.pdf`](C3/c3b_dose_response.pdf) · Script: [`C3/gen_c3b_dose_response.py`](C3/gen_c3b_dose_response.py)
