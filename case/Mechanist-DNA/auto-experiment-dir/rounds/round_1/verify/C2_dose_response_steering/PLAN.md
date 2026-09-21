## Claim C2: Amplifying the C1 α-helix feature set during Evo2-7B autoregressive DNA generation increases the encoded-protein α-helix fraction, monotonically with amplification strength up to an optimum.

### Main experiment (from /auto-experiment)
- Method: SAE feature steering (additive, α·s_f, raw-magnitude scale) at `blocks.26.post_norm`,
  Dataset: natural E. coli CDS prompts (M0 prokaryote set), Model: Evo2-7B + Layer-26 SAE
  → structure predictor **ESMFold** → DSSP α-helix fraction.
  Result: Spearman ρ=0.759 p=1.7e-5; helix 0.475(α=0)→0.834(α=32); effect +0.359.

### Variant (DIMENSIONS=method)
| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|----------------|--------|
| 1 | method | **OmegaFold** (single-sequence, PyTorch, independent architecture from ESM2/ESMFold) | ESMFold (facebook/esmfold_v1) | Directly tests whether C2's headline dose-response is an artifact of ESMFold's specific confidence/folding behavior — the audit flagged a concrete concern: at α=16/32, β-sheet content collapses to near-zero *while ESMFold's own pLDDT confidence rises above baseline*, a pattern consistent with a structure predictor over-confidently folding degenerate/low-complexity sequences. OmegaFold is single-sequence-native (no MSA — appropriate here since generated sequences have no real evolutionary homologs to search for anyway, unlike forcing AlphaFold2 into single-sequence mode which is out-of-distribution for its training), architecturally independent of the ESM family, and reports its own per-residue confidence (B-factor column) analogous to pLDDT. If the dose-response direction (helix rising with α) and specificity gap persist under a completely different predictor, that is strong evidence the effect is a real property of the generated sequences, not a predictor-specific artifact. | team-lead task brief; reviewer's own audit finding (EXPERIMENT_AUDIT.md Check E / MECHANISM_AUDIT.md output_case_spotcheck) |

Everything else held fixed: same frozen feature set S (`results/m0_feature_set.json`), same steering
hook (`code/mechanism.py`'s `Steerer`, additive α·s_f at `blocks.26.post_norm`), same prompts (natural
E. coli CDS prefixes), same translation/ORF filter, same DSSP tool (`mkdssp`) and helix definition
(HGI). Only the structure-prediction step changes.

**Reviewer critique (Phase 4, external LLM, gpt-5.4) and how the plan was strengthened:**
Trust score 3/5 as originally proposed ("ACCEPT WITH REQUIRED STRENGTHENING, not REJECT") — genuine,
right confound, but (a) grid too thin (4 doses × 2 seeds) to speak to "monotonic ... up to an
optimum," (b) OmegaFold-alone doesn't isolate whether divergence is generation-noise vs
predictor-specific (should fold the *same* saved sequences with both predictors), (c) missing a
cheap sequence-composition/degeneracy diagnostic that doesn't depend on any predictor at all. Adopted
all three:
- **Grid restored/widened**: α ∈ {0, 4, 8, 16, 32} (5 doses, adds the reviewer-requested α=4) × seed
  ∈ {42, 200} (2 seeds — restored from 2 in the original pilot; 3 seeds was recommended by the
  reviewer but not adopted purely for verify-pass compute-budget reasons given this variant ALSO adds
  a same-sequence ESMFold refold + diagnostics pass per cell, roughly doubling per-cell cost relative
  to the OmegaFold-only pilot) = 10 (α,seed) generation cells, n=150 generated per cell (matches a
  value already used in the main experiment's own M1/M2 pilot doses).
- **Same-sequence cross-check added**: `fold_and_readout_esmfold_samecells.py` folds the IDENTICAL
  saved proteins from each cell with ESMFold (in the `scientist` env), so OmegaFold vs ESMFold is a
  same-sequence, same-sample comparison, not two independently-generated batches — this isolates
  predictor disagreement from generation-sampling noise.
- **Sequence-degeneracy diagnostics added**: `seq_diagnostics.py` computes per-protein Shannon
  entropy, alphabet size, max homopolymer run, and k-mer repeat fraction (no GPU, no structure
  predictor needed) per (α,seed) cell — directly tests whether high-α sequences are compositionally
  degenerate independent of what either folder concludes.

**Byproduct improvement:** because OmegaFold runs in a separate conda env from Evo2/vortex (isolated
to protect the shared `scientist` env's pinned numpy<2 ABI — installing `colabfold` there was tried
first and silently upgraded numpy to 2.4.6, immediately reverted), sequence generation and folding
must run in two separate processes handing off via a JSONL file — which means this variant, unlike
the main experiment, **does save raw generated protein sequences** for the doses/seeds it covers.
This directly closes (for this subset) the sequence-inspectability gap flagged in C2's audit, and
enables the same-sequence cross-check and diagnostics above.

### Success Criterion (per variant)
The variant is judged `consistent_with_main_experiment = pass` if: (a) helix fraction under OmegaFold
increases with α across {0,4,8,16,32} in the same direction as the main experiment (positive trend,
ideally Spearman-significant), and (b) the effect size at α=16/32 vs α=0 is in the same qualitative
range as the main experiment (clearly non-trivial, not reversed or within noise) — i.e. the causal
dose-response conclusion holds under a different structure predictor. Divergence patterns and their
meaning (per reviewer): OmegaFold-null/main-experiment-strong → C2's support is method-dependent,
downgrade; OmegaFold-reproduces → C2 strengthened; both predictors agree on rising helix on the SAME
sequences AND those sequences show low entropy/long homopolymer runs at high α → the effect is real
but likely compositional/trivial rather than "rich structure control," a reframing not a rejection;
agreement at low/mid α but divergence only at α≥16 → the core dose-response holds at moderate
amplification with high-dose measurement uncertainty flagged separately. The variant is `fail` if the
trend is flat/reversed under OmegaFold, or if same-sequence ESMFold vs OmegaFold agreement collapses
specifically at high α in a way that tracks rising sequence degeneracy (entropy/homopolymer runs).

---

## ADDENDUM: Plan pivot (executed swap)

The OmegaFold structure-predictor swap above was **abandoned after implementation and code review**
due to an infeasible install (OmegaFold's pinned `torch==1.12.0+cu113` wheel, and the PyPI-modern-torch
fallback with its ~2.5-3.5GB of `nvidia-cu12` dependency wheels, both crawled at a few MB/min over this
environment's outbound proxy — not a sudo/credential/quota HC4-STOP condition, just impractically slow
for this verify pass's time budget). Full detail: `variants/method-swap-omegafold/PIVOT_NOTE.md` (kept
for transparency, not deleted).

**Executed instead:** `variants/method-swap-ssdef-pydssp/` — swap the SS-ASSIGNMENT ALGORITHM
(mkdssp's classical 8-state DSSP -> pydssp's from-scratch hydrogen-bond-map 3-state {-,H,E}
reimplementation) on the SAME ESMFold-predicted structure per sample, holding the structure predictor,
steering hook, feature set, prompts, and translation/ORF filter identical to the main experiment. This
is the second of the two method-axis candidates named in the task brief ("swap the secondary-structure
definition/tool"), requires zero new package installs (`pydssp` already present per
`results/setup_report.json`), and directly tests whether C2's dose-response is an artifact of DSSP's
specific 8-state HGI grouping rule. See `variants/method-swap-ssdef-pydssp/DIFF.md` for the full
held-fixed/changed inventory and grid (α∈{0,4,8,16,32}×seed∈{42,200}, n=150/cell, 10 cells).

**Success criterion for the executed variant:** `consistent_with_main_experiment = pass` if the
pydssp-based helix-fraction dose-response (a) rises with α across {0,4,8,16,32} in the same direction
as the main experiment's mkdssp-based curve, computed on the SAME predicted structures within this
variant's own grid, and (b) the effect size at α=16/32 vs α=0 is qualitatively non-trivial and not
reversed relative to the paired mkdssp readout on those same structures. `fail` if the pydssp-based
trend is flat/reversed relative to the paired mkdssp readout on the identical structures (i.e. the two
SS-assignment algorithms disagree on the qualitative dose-response), which would indicate the main
experiment's HGI grouping rule is doing more of the work than a "real structure" story implies.
