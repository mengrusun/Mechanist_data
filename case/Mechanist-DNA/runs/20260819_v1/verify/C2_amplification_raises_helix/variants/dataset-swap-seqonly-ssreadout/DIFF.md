# Variant DIFF — dataset/readout swap: sequence-only SS predictor

**One swap only.** The predicted-%-helix READOUT TOOL is replaced; the generated
sequences being scored are byte-for-byte the main experiment's cached held-out
generations (`results/m3final_a0.json` baseline n=2168; `results/m3final_astar.json`
α*=1 n=2205). No regeneration — the causal experiment is frozen; only measurement changes.

- **Main experiment readout**: ESM-2 650M (protein language model) + a trained SS-probe →
  predicted %-helix (held-out helix F1 0.92).
- **Variant readout**: a windowed amino-acid one-hot (GOR/PSSM-style) MLP 3-state SS
  predictor — **no protein LM, fully independent of ESM-2** — trained on the SAME S0 DSSP
  labels (train-split genes), validated on held-out TEST-split genes (its own held-out
  helix F1 is reported so calibration is known). Generations are Evo2-synthetic → zero
  overlap with the predictor's training data by construction.

Extra rigor (per Phase 4 review): a model-free biophysical Chou-Fasman Pα helix propensity
(exploratory secondary), a helix-favoring-residue frequency composition control (is the
gain merely composition?), and a per-sequence correlation between the new readout and the
ESM-2 probe (cross-calibration). Primary test = two-sample bootstrap of Δ predicted-%-helix
(α*=1 − baseline) under the new readout.
