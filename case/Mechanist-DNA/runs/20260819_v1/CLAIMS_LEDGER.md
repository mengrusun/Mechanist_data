# Claim Ledger — α-Helix-Directed DNA Generation by SAE Feature Amplification in Evo2-7B (Layer-26 Mixed SAE)

**Direction**: Generate DNA sequences with higher α-helical content using Evo2-7B via SAE feature amplification (task.md, reproduction combination given+given).
**Date**: 2026-08-19 → (running)
**Pipeline**: running | **Iteration**: —/10 "" (0/6)
**Models**: claim=claude-opus-4-8, experiment=claude-opus-4-8, verify=claude-opus-4-8, iteration=claude-opus-4-8
**Updated after**: experiment

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 α-helix-selective SAE features exist | SUPPORTED (qualified) | — | — | qualified (AUROC 0.63<0.70, selectivity via fallback) |
| C2 amplification raises %-helix vs baseline | SUPPORTED | — | — | held-out high-power p<0.05, specific; small (~+1.6 pp) |
| C3 optimal amplification strength α* | SUPPORTED | — | — | α*=1.0, single-peaked, reproduced held-out |

---
## C1 — α-helix-selective SAE features exist in the Layer-26 Mixed SAE
- **Statement**: Within the Evo-2 Layer-26 Mixed SAE (32768 latents), a small identifiable subset of latents activates selectively on nucleotide positions whose translated codons belong to α-helical residues, with selectivity above chance and above matched β-sheet/coil controls.
- **Origin**: task.md Experiment Tips step 1 (enabling precondition)
- **Data**: E. coli MG1655 RefSeq CDS + AlphaFold-DB (UP000000625) structures, DSSP per-residue labels, mmseqs2 30%-id homology-clustered (3123 clusters) 70/15/15 splits — provenance=adapted; available=3709 genes / 1.18M codons, used=all splits (train 821,826 / val 181,568 / test 173,912; full, strict)
- **Models**: Evo2-7B (/mnt/quarkfs/share_models/evo2_7b_262k)
- **Method**: M1 (screen) — capture per-codon layer-26 residual (blocks.26, SAE FVU=0.14), SAE-encode (TopK k=64, d_sae=32768); per-latent α-helix-vs-rest AUROC + activation ratio; gene/codon-preserving shuffle null (200 shuffles) + BH-FDR across 32768; val-selected S confirmed once on held-out test vs null and matched β/coil controls
- **Main experiment**: SUPPORTED (qualified) — max_train_helix_auroc=0.633 (prereg ≥0.70 not met); |S|=20; primary latent test AUROC 0.634 CI[0.629,0.639] > ctrl bar 0.543; helix:non-helix ratios 3.4–15.7×; selection via specificity fallback
- **Verify**: —
- **Iteration**: —
- **Final**: main experiment: SUPPORTED (qualified — AUROC 0.63 < 0.70 prereg, selectivity via specificity fallback)
- **Caveats**: Codon-level AUROC 0.633 < pre-registered 0.70; selectivity established via specificity fallback (> null AND > β/coil control bar 0.543).
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M1, refine-logs/EXPERIMENT_RESULTS.md#C1, results/m1_features.json

---
## C2 — Amplifying the α-helix feature set raises %-helix vs unsteered baseline
- **Statement**: Amplifying the C1 α-helix feature set (residual-add of coefficient × feature decoder direction into the layer-26 residual during autoregressive decoding) produces generated DNA whose translated ORFs have higher mean α-helical content than an identically-decoded, paired-seed unsteered baseline.
- **Origin**: task.md core behavior (Experiment Tips step 2)
- **Data**: Evo2-7B generations (BOS+ATG, temp=1.0, top-p=1.0, ≤900 nt, paired seeds) → ORF→ESM-2-650M-probe predicted %-helix (ESMFold fallback), treatment-independent QC — provenance=constructed; used=M2 dev 2100, M3 held-out 4×~660 valid, high-power α*=1 merged 2205 valid, M-CTRL 4×300; ~11k total
- **Models**: Evo2-7B (/mnt/quarkfs/share_models/evo2_7b_262k)
- **Method**: M2 (baseline vs steered) + M3/M3b (held-out + high-power confirm) + M-CTRL (random-feature / β-sheet / null-direction specificity); residual-add x'=x+α·Σ_{i∈S} s_i·d̂_i at blocks.26; unpaired two-sample cluster-bootstrap CIs
- **Main experiment**: SUPPORTED — high-power held-out α*=1 (n=2205): Δhelix_cond +0.016 p=0.033, Δhelix_itt +0.015 p=0.011; M3 α*=1: Δcond +0.026 p=0.050, Δitt +0.022 p=0.045; effect ~+1.6 pp (high-power) to ~+2.6 pp (M3)
- **Verify**: —
- **Iteration**: —
- **Final**: main experiment: SUPPORTED (held-out high-power p<0.05, specific to S; small effect ~+1.6 pp)
- **Caveats**: Effect size small (~+1.6 pp predicted-helix at α*=1 high-power; ~+2.6 pp at smaller-N M3). · null_direction control gave a partial non-specific rise (+0.018, below S's +0.031) — effect largely but not entirely specific. · Predicted %-helix via ESM-2-probe readout (ESMFold fallback).
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M2, #M-CTRL, #M3, refine-logs/EXPERIMENT_RESULTS.md#C2, results/summary.json

---
## C3 — There is an optimal amplification strength α*
- **Statement**: Across a sweep of amplification coefficients the %-helix-vs-α curve is non-flat and single-peaked/saturating, attaining a maximum at an identifiable optimal strength α* (subject to a sequence-validity floor); α* reproduces on disjoint held-out seeds.
- **Origin**: task.md Experiment Tips step 3 (optimal strength)
- **Data**: same generation+readout pipeline as C2 swept over α ∈ {0,0.5,1,2,4,8,16} — provenance=constructed; used=M2 dev sweep 7×300 (α* frozen on val) + M3 confirm {0,α*/2,α*,2α*}×~660 valid held-out
- **Models**: Evo2-7B (/mnt/quarkfs/share_models/evo2_7b_262k)
- **Method**: M2 dev dose-response → α*=argmax conditional %-helix subject to validity floor (ORF-valid ≥ baseline−10pp AND median ppl ≤ baseline 95th pct); M3 held-out confirmation
- **Main experiment**: SUPPORTED — α*=1.0 (α=2 near-equal neighbor); helix_cond 0.233(α0)→0.264(α1 peak)→0.087(α16 collapse); ppl flat & ORF-validity rises at collapse (rules out validity-collapse artifact); peak reproduced on held-out seeds
- **Verify**: —
- **Iteration**: —
- **Final**: main experiment: SUPPORTED (α*=1.0, single-peaked, reproduced held-out)
- **Caveats**: α=2 is a near-equal neighbor to α*=1 (optimum is a plateau over α∈{1,2}, not a sharp point).
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M2, #M3, refine-logs/EXPERIMENT_RESULTS.md#C3

---
## Journey Summary
- **Claim**: behavior_source=given (reproduction) → single captured behavior, 3 claims (C1 precondition, C2 core, C3 optimum) from task.md Experiment Tips; resource_fidelity: strict.
- **Mechanism strategy**: n/a (MECHANISM=given — SAE feature identification + amplification, Evo-2 Layer-26 Mixed SAE)
- **Mechanism routing**: family=SAE feature amplification, submethod=residual-add at blocks.26 over frozen feature set S (committed directly, no routing); screen→decode→verify→recover
- **Experiment**: 22 runs, ~5 GPU-hours (≤4 A800 concurrent), headline positive: C1 supported (qualified), C2 supported (held-out high-power p<0.05), C3 supported (α*=1); ~11k sequences; ESMFold→ESM-2-probe eval-tool fallback.
- **Verify**: (pending)
- **Iteration**: (pending)
- **Figures**: (pending)

## Open Items
- Eval-tool fallback: ESMFold weights unreachable offline (HF CDN ~1 MB/s; shared copy is a dangling symlink) — folding→SS readout substituted with local ESM-2 650M + linear probe on S0 DSSP labels (held-out 3-state acc 0.878, helix F1 0.920). Evo2-7B + released SAE remain exact; %-helix results are "predicted". Verify should stress-test readout-tool sensitivity.
- C1 qualified: pre-registered codon-level AUROC ≥0.70 not met (max 0.633); selectivity established via the pre-registered specificity fallback (test AUROC 0.634 CI[0.629,0.639] > β/coil control bar 0.543). Metric-resolution caveat (helix is windowed; sparse TopK latents), not absence of selectivity.
- C2 effect size small: ~+1.6 pp predicted-helix at α*=1 (high-power held-out); ~+2.6 pp at smaller-N M3. Real, specific, reproducible but modest — worth confirming robustness under readout/dataset swaps.
