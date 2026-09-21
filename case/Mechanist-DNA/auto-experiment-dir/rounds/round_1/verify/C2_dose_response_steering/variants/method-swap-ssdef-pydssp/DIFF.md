# DIFF vs main experiment (code/m2_dose_response.py + code/mechanism.py)

**Dimension**: method. **What changed**: the secondary-structure ASSIGNMENT ALGORITHM only
(mkdssp's classical 8-state DSSP -> pydssp's from-scratch, hydrogen-bond-map, 3-state {-,H,E}
reimplementation), computed on the SAME ESMFold-predicted structure per sample. Steering hook,
frozen feature set S + per-feature scale s_f, prompts, translation/ORF filter, ESMFold structure
prediction, and pLDDT gate are all IDENTICAL to the main experiment.

**Why this axis, and why it's a genuine test (not cosmetic):** C2's headline number is a helix
FRACTION computed by DSSP's rule-based hydrogen-bond-pattern classifier collapsing 8 raw states into
the HGI={H,G,I} "helix" bucket. If the dose-response effect is a real property of the generated
proteins' backbone geometry, an independently-implemented, differently-parameterized secondary-
structure algorithm (pydssp: a from-scratch vectorized hydrogen-bond-map calculation, not a wrapper
around mkdssp) run on the IDENTICAL predicted coordinates should show the same qualitative
dose-response. If it doesn't, that would suggest DSSP's specific 8-state grouping/thresholding rule
is doing more of the work than the "real structure" story implies.

**Byproduct improvement:** raw generated protein sequences ARE saved to `seqs_a*_s*.jsonl` (the main
experiment discards them via `mechanism.py`'s `rec.pop('prot', None)`) — closes, for this variant's
grid, the sequence-inspectability gap flagged in `../../main_experiment_audit/EXPERIMENT_AUDIT.md`.

**Held fixed (verified against code/mechanism.py and code/m2_dose_response.py):**
- Steering: `Steerer(evo2, sae, fs["helix_features"], fs["s_f"])`, additive `alpha*base_dir` at
  `blocks.26.post_norm` — byte-identical call to the main experiment's `m2_dose_response.py`.
- Prompts: `natural_prompts(org, n)` from `m1_harness_calibrate.py`, same tiling logic.
- Generation params: `n_tokens=300, temperature=0.7, top_k=4` — identical to
  `m2_dose_response.py`'s own argparse defaults.
- Translation/ORF filter: `M.translate_orf(..., min_aa=30, table=table)` — identical.
- Structure prediction: `M.esmfold_pdb(prot, device=..., max_len=400)` — identical function, same
  pLDDT gate (`plddt_min=50.0`, matching M2's own default).
- Helix/sheet fraction definition for the mkdssp arm: identical (`M.dssp_fractions`, HGI={H,G,I}).

**Grid (cost-reduced from the main experiment's 8 doses x 3 seeds = 24 runs):** alpha in
{0,4,8,16,32} x seed in {42,200} = 10 cells, n=150 generated/cell (matches a value already used in
the main experiment's own pilot doses). Reduction justified by verify-pass GPU budget; still spans
the doses most relevant to the audit's flagged concerns (baseline, clean optimum, both high-alpha
doses where the sheet-collapse/pLDDT-rise pattern was found) plus alpha=4 (reviewer-requested during
the (abandoned) OmegaFold plan's Phase 4 critique, retained here since the grid carried over).

**Bugs found and fixed during implementation (documented for transparency):**
- Sanity pilot (n=5) initially took anomalously long with 0% GPU utilization; root cause was that
  `run_variant.py` did not set `HF_HOME`, so `transformers` silently re-downloaded the ~2.7GB ESMFold
  checkpoint into `~/.cache/huggingface` instead of reusing the project's pre-cached copy at
  `/data/wanghaoxiong/intergene_mechanist_v6/.hf_cache` (the env var `code/dispatch.py` sets for all
  main-experiment runs). Fixed by setting `os.environ.setdefault("HF_HOME", ...)` at the top of
  `run_variant.py` before any `transformers`/`huggingface_hub` import. Re-ran the pilot after the fix
  (64s for n=5, mkdssp helix=0.645 vs pydssp helix=0.657 on the 2 gated-pass structures — close
  agreement, validates the dual-readout pipeline).
