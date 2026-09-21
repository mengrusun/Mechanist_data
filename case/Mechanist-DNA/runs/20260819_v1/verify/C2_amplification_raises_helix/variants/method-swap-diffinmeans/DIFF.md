# Variant DIFF — method swap: diff-in-means steering direction

**One swap only.** The steering DIRECTION is extracted by a mass-mean (diff-in-means)
estimator instead of the SAE decoder. Everything else is byte-for-byte the main
experiment's machinery (reuses `code/common.py`, `code/gen_eval.py`, `code/ss_predictor.py`).

- **Main experiment direction**: `v_S = Σ_{i∈S} s_i·d̂_i` — sum of 20 helix-selective SAE
  latents' unit decoder columns, weighted by median positive train activation.
- **Variant direction**: `v_dm = mean(blocks.26 residual over HELIX codons) −
  mean(over NON-HELIX codons)`, estimated on TRAIN-split genes only, then **norm-matched**
  to `‖v_S‖ = 1.8997`.

Held fixed: model (Evo2-7B), SAE-defined norm reference, hook site (blocks.26), additive
residual intervention form, decoding (temp 1.0 / top_p 1.0 / top_k 4 / 900 nt / ATG start),
paired seed blocks D & H, ESM-2-probe readout, treatment-independent QC, ITT rule, bootstrap.

Fair-comparison controls: (1) the diff-in-means arm is **dose-calibrated on its own dev
dose-response** (α*_dm chosen by the same validity-floor rule) rather than transplanting
α=1; (2) a fresh α=0 baseline is generated in the identical code path; (3) `cosine(v_dm, v_S)`
and the norm ratio are reported so functional (not just Euclidean) dose is documented;
(4) the direction is estimated on TRAIN residuals only — no test leakage.
