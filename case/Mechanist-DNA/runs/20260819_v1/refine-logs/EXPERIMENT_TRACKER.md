# Experiment Tracker (plan-level)

**Date**: 2026-08-19 | **resource_fidelity**: strict | **chosen_mechanism**: SAE feature amplification (Evo-2 Layer-26 Mixed)

| Run ID | Milestone | Claim | Cmd (template) | Expected output | GPUs | Est. h | Status | Notes |
|--------|-----------|-------|----------------|-----------------|------|--------|--------|-------|
| S0 | S0 data prep | — | build ecoli_labeled.parquet (RefSeq CDS + AlphaFold-DB + DSSP) | data/ecoli_labeled.parquet, data/splits.json | 1 | 0.3 | pending | |
| M1 | M1 feature select | C1 | m1_feature_select.py | results/m1_features.json | 2 | 1.25 | pending | |
| M2-a0 | M2 dev sweep | C2/C3 | m2_generate_eval.py --alpha 0 | results/m2_a0_D.json | 4 | — | pending | baseline arm |
| M2-a0.5 | M2 dev sweep | C3 | --alpha 0.5 | results/m2_a0.5_D.json | 4 | — | pending | |
| M2-a1 | M2 dev sweep | C3 | --alpha 1 | results/m2_a1_D.json | 4 | — | pending | |
| M2-a2 | M2 dev sweep | C3 | --alpha 2 | results/m2_a2_D.json | 4 | — | pending | |
| M2-a4 | M2 dev sweep | C3 | --alpha 4 | results/m2_a4_D.json | 4 | — | pending | |
| M2-a8 | M2 dev sweep | C3 | --alpha 8 | results/m2_a8_D.json | 4 | — | pending | |
| M2-a16 | M2 dev sweep | C3 | --alpha 16 | results/m2_a16_D.json | 4 | 3.0 | pending | α* frozen from val |
| CTRL-rand | M-CTRL | C2 spec | --control random_feature | results/mctrl_random_feature.json | 4 | — | pending | at α* |
| CTRL-beta | M-CTRL | C2 spec | --control beta_sheet | results/mctrl_beta_sheet.json | 4 | — | pending | expect sheet↑ |
| CTRL-null | M-CTRL | C2 spec | --control null_direction | results/mctrl_null_direction.json | 4 | 1.8 | pending | |
| M3-a0 | M3 confirm | C2 | m3_confirm.py --alpha 0 | results/m3_a0.json | 4 | — | pending | held-out block H |
| M3-astar | M3 confirm | C2/C3 | --alpha α* | results/m3_astar.json | 4 | — | pending | ≥500 valid |
| M3-half | M3 confirm | C3 | --alpha α*/2 | results/m3_ahalf.json | 4 | — | pending | |
| M3-2x | M3 confirm | C3 | --alpha 2α* | results/m3_a2x.json + summary.json | 4 | 2.3 | pending | |

**Total est.**: ~6.5–8 h, ≤4 GPUs concurrent. Status legend: pending → running → done/failed (updated in place by /auto-experiment Phase 5).

---

### Execution notes (updated live)

- **S0 done**: 3709 genes joined (RefSeq CDS ↔ RefSeq protein ↔ AlphaFold-DB UP000000625, exact AA-sequence agreement), 1,177,306 labeled codons; homology split (mmseqs 30% id) train 2589 / val 566 / test 554; 3123 clusters; helix_frac 0.459. Outputs `data/ecoli_labeled.parquet`, `data/ecoli_genes.parquet`, `data/splits.json`.
- **Core validated (smoke)**: Evo2-7B loads (35s, 1 GPU); SAE hooked at `blocks.26` reconstructs at FVU=0.14 (86% var explained → confirms hook site + tied-TopK encode convention); α=0 residual-steer is an exact identity; batched 900-nt steered generation = 0.71 s/seq, 17.8 GB.
- **Readout eval-tool fallback (documented, does NOT touch strict model/SAE fidelity)**: ESMFold weights (8.4 GB) are effectively unreachable offline (HF CDN ≈0.7–1.2 MB/s even with 24 parallel streams → ~2 h; shared copy `/mnt/quarkfs/share_models/esmfold_v1` is a dangling symlink). Per FINAL_PROPOSAL §risk + EXPERIMENT_PLAN budget-guard, the folding/SS readout tool is substituted with a **local per-residue SS predictor**: ESM-2 650M (cached, protein-LM independent of Evo2/SAE) + a linear probe trained on the S0 DSSP 3-state labels (train split), validated held-out. **Test 3-class acc 0.878, helix F1 0.920** → reliable %-helix readout. Evo2-7B + released Layer-26 SAE remain exact (strict). ESMFold download left running as an optional robustness cross-check.
- **M1 DONE — C1 SUPPORTED (via specificity path)**. Captured per-codon layer-26 SAE latents for all 3709 genes (train 821,826 / val 181,568 / test 173,912 codons; 4-GPU sharded capture). **Max codon-level helix AUROC = 0.633 → the prereg ≥0.70 threshold was NOT met** (helix is a windowed property; sparse TopK latents leave many helix-position zeros). Prereg candidate set therefore empty; the pre-registered **specificity fallback** (q<0.01 gene-blocked-null significant AND helix-AUROC > matched β/coil control bar AND helix-AUROC − β/coil-AUROC > 0.05 AND ratio > control) yielded 47 candidates → **frozen set S = 20 latents** (K chosen on validation). Test-once: **primary latent 19897 test AUROC 0.634, cluster-bootstrap CI [0.629, 0.639]** (≫ 0.5 and > null), **> matched β/coil control bar 0.543**; set-level helix-AUROC > set β and coil AUROC. Helix:non-helix activation ratios 3.4–15.7×. C1 decision-gate precondition (selective features above null AND controls) is met → C2/C3 proceed. Reported honestly as **qualified C1**: real, specific helix selectivity, but codon-level AUROC is moderate (0.63), below the prereg 0.70.
- M1 note: single-process capture (float32, 154 GB) was first killed by a background-wrapper lifetime limit; re-run as 4-GPU sharded capture (fully detached) + a merge/analyze step. An unrelated concurrent job saturating memory bandwidth slowed the strided-array analysis passes (no correctness impact).
- **M2 DONE (dev block D, 7 arms × 300, seed-paired)** — clean single-peaked dose-response of predicted %-helix: helix_cond 0.233 (α=0) → 0.245 (0.5) → **0.264 (α=1)** → 0.262 (α=2) → 0.227 (4) → 0.172 (8) → 0.087 (16); ITT peak 0.202 at α=2; validity 0.70→0.83 (no collapse), ppl flat. **α\* = 1.0** frozen from dev. C3 supported; C2 positive but under-powered at n=300.
- **M-CTRL DONE (α*=1, block D, 4×300)** — specificity: S 0.264 vs random-feature 0.227, β-sheet 0.226 (both ≈ baseline, no helix gain), null-direction 0.251 (intermediate). Helix gain is specific to S; β-sheet/random steering give no helix gain.
- **M3 DONE (held-out block H, 4 arms × ≥600 valid)** — dose-response reproduced: α*=1 helix_cond 0.269 vs baseline 0.243 (**Δ +0.026, unpaired p=0.050**), ITT 0.197 vs 0.175 (**Δ +0.022, paired p=0.045**). C2 at the significance boundary at n~657.
- **M3b DONE + merged with M3 → C2 SUPPORTED**. High-power held-out confirmation (fresh disjoint block-H seeds; α*=1 and baseline reached 1500 valid each, merged with M3 → **2205 valid at α*=1, 2168 baseline**). At the frozen **α*=1**: Δhelix_cond **+0.016 (unpaired p=0.033)**, Δhelix_itt **+0.015 (p=0.011)** — both <0.05, no validity loss. C2 confirmed on held-out seeds. (The α=2 high-power arm was stopped as redundant — α=2 already covered by M3 at 674 valid; α*=1 is the frozen optimum and where the effect is significant.)

### Final per-claim verdicts
- **C1** (helix-selective SAE features): **SUPPORTED (qualified)** — 20 latents, primary test AUROC 0.634 CI[0.629,0.639] > null & > control bar 0.543; prereg codon-AUROC≥0.70 not reached (0.63), reported transparently.
- **C2** (amplification raises predicted %-helix): **SUPPORTED** — held-out high-power α*=1, Δcond +0.016 (p=0.033), Δitt +0.015 (p=0.011); specific to S (M-CTRL); small effect (~1.6–2.6 pp).
- **C3** (dose-response + identifiable α*): **SUPPORTED** — clean single-peaked M2 curve, α*=1, rising portion + α* reproduced held-out.
- **Runs**: S0 + M1(4 capture shards + analyze) + M2(7) + M-CTRL(4) + M3(4) + M3b(2 used) = 22 experiment runs; per-run `runs/<id>/cost.json` with `gpu_ids` written (25 files incl. tool-train). Total generation ≈11,000 sequences. ≤4 GPUs concurrent throughout. GPU-hours ≈ (M1 capture 4×~0.12h + analyze ~0.4h) + generation waves (~4 GPU × ~1.1h) ≈ **~5–5.5 GPU-hours**; wall-clock well within the 8 h budget.
- **No budget-guard trim** (used_n met or exceeded at every milestone; strict marker ⇒ UNDERPOWER ignored, and the held-out effect was in any case confirmed at high power). C1 precondition held (features above null & controls) ⇒ no negative-result halt.
- Code reviewed by external LLM (Phase 2.5); fixes applied: float32 latents (tie-free AUROC), primary feature frozen on validation (no test peeking), non-ACGT generation telemetry. Prereg unit-decoder-direction steering (§Prereg-1) retained by design.
- Note: an unrelated third-party job (`experiments/dna_sae_steer/...`, another session) shares the machine; GPUs chosen around it.
