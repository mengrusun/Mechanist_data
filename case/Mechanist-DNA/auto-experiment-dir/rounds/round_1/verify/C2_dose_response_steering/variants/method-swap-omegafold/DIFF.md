# DIFF vs main experiment (code/m2_dose_response.py + code/mechanism.py)

**What changed:** the structure-prediction step only. Steering hook, feature set S, per-feature
scale s_f, prompts, translation/ORF filter, DSSP tool, and helix definition (HGI) are all identical
to the main experiment.

**What's new (byproduct of the env split, not a confound):**
1. Generation is split into its own script (`gen_sequences.py`, runs in `scientist` env) that WRITES
   raw generated protein sequences to a JSONL file, instead of discarding them
   (`mechanism.generate_and_readout` normally does `rec.pop('prot', None)`).
2. `fold_and_readout_omegafold.py` (runs in an isolated `verify_omegafold` conda env — created to
   avoid the numpy ABI conflict `colabfold` caused when tried in `scientist`) reads that JSONL, folds
   with OmegaFold, and computes the SAME helix_hgi/sheet aggregate as `mechanism.aggregate()`.
3. `fold_and_readout_esmfold_samecells.py` (runs in `scientist` env) folds the IDENTICAL saved
   sequences with ESMFold — a same-sequence cross-check added per Phase 4 reviewer critique, not
   present in the main experiment's own pipeline.
4. `seq_diagnostics.py` (any env, CPU-only) computes per-sequence composition/degeneracy metrics
   (entropy, alphabet size, homopolymer run, k-mer repeat fraction) — also reviewer-requested, has no
   analog in the main experiment.

**Grid:** α ∈ {0,4,8,16,32} × seed ∈ {42,200}, n=150 generated/cell (main experiment: α ∈
{-2,0,1,2,4,8,16,32} × seed ∈ {42,200,201}, n=110-150/cell) — a cost-reduced subset covering the
doses most relevant to the audit's degenerate-sequence-collapse concern (baseline, clean optimum,
and both high-alpha doses), justified by verify-pass GPU budget.
