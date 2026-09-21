# Pivot note: ESMFold -> OmegaFold structure-predictor swap ABANDONED

**Status: infeasible within this verify pass's time/network budget. Pivoted to a different, still
genuinely decisive, method-axis swap: `../method-swap-ssdef-pydssp/` (mkdssp -> pydssp SS-assignment
algorithm, on the SAME ESMFold-predicted structures). This directory is kept for transparency, not
deleted, per instructions to say explicitly when a candidate swap proves infeasible and pick the most
feasible remaining one.**

## What was attempted
1. Reviewer-approved plan (see `../../PLAN.md`'s superseded content / this variant's `DIFF.md`): swap
   ESMFold -> OmegaFold (a genuinely different, single-sequence-native, PyTorch-based structure
   predictor), reusing the exact same steering hook/feature set/prompts, with a same-sequence ESMFold
   cross-check and sequence-composition diagnostics added per Phase-4 reviewer critique.
2. Code was written, code-reviewed (Phase 6), and CRITICAL/MAJOR findings fixed: `gen_sequences.py`,
   `fold_and_readout_omegafold.py` (fixed idx-parsing brittleness, missing-file/expected-vs-observed
   tracking, required `--conf_min` calibration instead of a guessed default, exception-safe tempdir
   cleanup), `fold_and_readout_esmfold_samecells.py`, `seq_diagnostics.py`, `calibrate_conf_gate.py`.
3. Installation blocker: `pip install git+https://github.com/HeliXonProtein/OmegaFold.git` in the
   `scientist` conda env's own Python 3.11 failed outright (OmegaFold's `setup.py` hardcodes support
   for only Python 3.8/3.9/3.10). A fresh isolated conda env (`verify_omegafold`, Python 3.10) was
   created to avoid this AND to avoid repeating the numpy-ABI mistake made when `colabfold` was first
   tried directly in `scientist` (that attempt silently upgraded `numpy` 1.26.4->2.4.6 in the shared
   env; immediately reverted -- see main VERIFY_REPORT for this incident).
4. In the fresh env, OmegaFold's `setup.py` hardcodes a specific legacy wheel:
   `torch==1.12.0+cu113` fetched directly from `download.pytorch.org` -- which downloaded at only a
   few MB/min over this environment's outbound proxy (multiple ten-minute waits observed, ~830MB of a
   likely ~1.9GB wheel after ~15 minutes). `setup.py` was patched locally to accept any modern `torch`
   from PyPI instead (`pip install torch --index-url https://pypi.org/simple`) to route around the
   slow `download.pytorch.org` path -- but modern `torch` (2.x) pulls a dozen+ separate
   `nvidia-*-cu12` wheels (cuBLAS, cuDNN, cuFFT, cuSPARSE, NCCL, ...) typically totaling 2.5-3.5GB,
   and this also proved too slow over the same proxy (still incomplete after another ~10 minutes).

## Decision
Rather than continue spending verify-pass wall-clock time on a network-bound install with no reliable
ETA, pivoted to `../method-swap-ssdef-pydssp/run_variant.py`, which requires **zero new package
installs** (both `mkdssp` and `pydssp` are already present in the `scientist` env per
`results/setup_report.json`) and still executes a genuine, pre-registered-candidate method-axis swap
(the task brief explicitly listed "swap the secondary-structure definition/tool (e.g. DSSP HGI ->
H-only, or an alternate SS assignment)" as an acceptable alternative to the structure-predictor swap).
This is not a downgrade to a trivial/cosmetic test -- see that variant's own `DIFF.md` and `PLAN.md`
addendum for why it is still a decisive test of a real confound (the SS-assignment algorithm, held
constant on identical ESMFold-predicted structures so no structure-prediction variance is introduced).

## Cleanup
The `verify_omegafold` conda env is left in place (harmless, isolated, contains a partial torch
install) rather than actively torn down, since removing it is not necessary for correctness and this
verify pass is time-constrained. No main-experiment or `scientist`-env file was modified by this
abandoned attempt (the one accidental `scientist`-env numpy upgrade from the `colabfold` experiment
was caught and fully reverted before any further use).
