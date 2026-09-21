# Experiment Tracker — Steering Evo2-7B toward high α-helical content

Plan-level table (one row per planned run). Finalized 2026-08-24 — full suite deployed and analyzed.

| Run ID | Milestone | Claim(s) | Description | Est. GPU-h | Status | Result | Notes |
|---|---|---|---|---|---|---|---|
| sanity | M0-smoke | — | Tiny end-to-end smoke (gen+ORF+ESMFold+steer hook) | 0.1 | done | PASS | runs/sanity_smoke; 32 blocks, gen_ok, ESMFold assay ok, steer_changes_output=true; SAE recon_rel_error≈1.0 (secondary-arm caveat) |
| m1_baseline_gen | M1 | C1,C2 | Evo2-7B unintervened baseline gen + full scoring (1500 gens, 3 seeds) | 4.0 | done | helix_all=0.405, validity=0.989 | results/m1; frozen harness + measurement contract; mean_plddt 0.45. No cost.json emitted — GPU-h estimated |
| m1_contrast_set | M1 | C1 | Build high- vs low-helix contrast set for intervention sourcing | (incl above) | done | 300 pairs (hi 0.684 / lo 0.138) | results/m1/contrast_set.json; DEV-split sourced |
| m2_probe | M2 | C1 | Per-layer linear probes for α-helix fraction; rank sites | 3.5 | done | top sites [30,28] | runs/m2_localize (GPU 1); probe R²≈0/neg (weak correlational screen — causal M3 decisive). No cost.json — GPU-h estimated |
| m2_sae_screen | M2 | C1 | Screen Evo2 SAE features (layer 26) for α-helix selectivity | (incl above) | done | shortlist (feat 24100 corr 0.23) | runs/m2_localize (same job); Goodfire L26 SAE from /mnt/quarkfs |
| m3_s28_caa | M3 | C1 | site 28 CAA dose grid (6 coef × 3 seed × 250 gen) — PRIMARY | 5.4 | done | trend ρ=0.922, p=5e-5, q=1.5e-4; helix 0.414→0.575 (coef4) | runs/m3_s28_caa/cost.json (GPU 3, reconstructed). Winning coef=1.0 |
| m3_s30_caa | M3 | C1 | site 30 CAA dose grid (secondary) | 5.1 | done | non-monotone (ρ=0.04, p=0.43) — no trend | runs/m3_s30_caa/cost.json (GPU 1, reconstructed) |
| m3_s26_sae_clamp | M3 | C1 | site 26 SAE-clamp dose grid (secondary) | 5.6 | done | NEGATIVE (ρ=−0.95); high recon error | runs/m3_s26_sae_clamp/cost.json (GPU 4, reconstructed) |
| m4_matched_control | M4 | C1,C2 | Matched orthogonal same-norm direction @ winning (expect no gain) | (incl M4) | done | helix 0.390 (−0.024 vs base, no gain) | runs/m4_ctrl_gpu6/cost.json (GPU 6); 750 gens |
| m4_sham | M4 | C1,C2 | Same-site same-norm permuted sham @ winning (expect no gain) | (incl M4) | done | helix 0.409 (−0.005 vs base, no gain) | runs/m4_ctrl_gpu7/cost.json (GPU 7); 750 gens |
| m4_validity | M4 | C2 | Validity non-inferiority (full set) + off-target vs baseline @ winning | (incl M4) | done | 0.968 vs 0.991, LB −0.035 > −0.05 → non-inferior | results/m4/specificity.json; off-target: GC/length shift documented |
| m4_frontier | M4 | C1,C2 | Helix-vs-validity frontier across coefficient sweep | (incl M4) | done | winning = site28 coef1.0 (max helix s.t. validity NI) | results/m4/validity_frontier.json |

**M4 wall**: two GPUs (6,7) each ran two conditions (baseline+matched_control; caa_win+sham), 750 gens
each → 3.7 GPU-h total (runs/m4_ctrl_gpu{6,7}/cost.json, reconstructed).

**Realized GPU-hours: ≈ 27.3** (HARD cap 40 respected; ≤ 8 concurrent cards).
- sanity ~0.1 + M1 ~4.0 (est.) + M2 ~3.5 (est.) + M3 **16.0** (reconstructed cost.json: 5.4+5.1+5.6) +
  M4 **3.7** (reconstructed cost.json: 1.9+1.85).
- **cost.json note**: sanity/M1/M2 runs did **not** emit `cost.json`; their GPU-hours are estimates
  (not fabricated per-run cost files). M3/M4 `runs/<id>/cost.json` were **reconstructed** post hoc from
  `start.ts` + stdout completion markers (the launch wrapper did not write them live) — each records
  `gpu_ids` and is marked `source: reconstructed`.
- GPUs used: 1, 3, 4, 6, 7. GPUs 0, 2, 5 left to other users (not clobbered). No `gpu_id` pin was
  forwarded to this analysis-only resume, so no pin-propagation assertion applies.

**Verdicts**: C1 SUPPORTED (dose-response trend, direction-specific), C2 SUPPORTED (validity
non-inferior), joint C1 ∧ C2 SUPPORTED. See refine-logs/EXPERIMENT_RESULTS.md. No M0 gate
(behavior-source: given).
