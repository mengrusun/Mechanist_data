# Experiment Plan — α-Helix-Directed DNA Generation via SAE Amplification (Evo2-7B)

**Date**: 2026-08-19
**resource_fidelity**: strict
**chosen_mechanism**: SAE feature identification + amplification (Evo-2 Layer-26 Mixed SAE)
**mechanism_strategy**: n/a
**Behavior-source**: given  →  **NO M0 phenomenon-validation gate**; no milestone carries `kind: phenomenon-validation`; no milestone declares `depends_on: [M0]`.
**Budget**: 8×A800-80GB, ≤4 GPUs concurrent, ≤8 h total wall-clock. Full-scale Evo2-7B + released Layer-26 SAE throughout — no downscaling, no data subsetting.

> Claim → milestone map: **C1** → M1; **C2** → M2 (baseline vs steered) confirmed in M3b; **C3** → M2 (dev sweep) + M3 (held-out α* confirmation). M-CTRL covers specificity for C2/C3.
> Milestones run in order M1 → M2 → {M-CTRL, M3}. `depends_on` chains are ordinary artifact dependencies (feature set → generation), NOT an M0 gate.
> Under `resource_fidelity: strict`, per-milestone `method_sensitive` is intentionally omitted (values are pinned exact).

---

## Shared setup (S0) — data & harness prep (no steering)

- **Labeled coding-DNA dataset (data-rule: adapted-from-existing).** E. coli K-12 MG1655 (RefSeq GCF_000005845.2) CDS (nucleotide) ↔ protein ↔ **AlphaFold DB proteome (UP000000625)** structures. Provenance = `adapted`. Steps: (1) exact CDS→protein→structure sequence alignment; drop mismatches/isoforms/unresolved residues; (2) DSSP on each AlphaFold structure → per-residue 3-state SS; helix = {H,G,I}, and strict-H variant reported; (3) map each residue label to its 3 codon nucleotide positions.
- **Splits.** mmseqs2 cluster at 30% identity → homology clusters; split clusters **70/15/15 train/val/test**; all homologs/paralogs kept within one partition. Effective n = **all clean proteins (target ≥ ~1000 proteins, ≫ n>50 floor); primary unit = gene/cluster.** used_n = available_n (full E. coli proteome after QC) — strict, no subsetting.
- **Structure labels for training data use AlphaFold-DB (static download, no folding compute)** — keeps C1 labels independent of the C2/C3 ESMFold readout.
- **Env**: conda `scientist`; Evo2 loaded via vortex/evo2; SAE tied-weight TopK loaded from checkpoint (W 4096×32768, b_enc, b_dec, k=64).

**Expected outputs**: `data/ecoli_labeled.parquet` (per-position: gene, cluster, split, codon idx, aa, helix label, β/coil label, GC, codon-pos), `data/splits.json`.
**GPU-hours**: ~0.3 h (1 GPU) for AlphaFold-DB download + DSSP (CPU) + alignment; folding NOT required here.

---

### M1: Confirm α-helix-selective SAE features  — **[Claim C1]**

**Depends on**: S0
**Goal**: identify and validate the α-helix feature set `S` with held-out selectivity above nulls/controls.
**Procedure**:
1. Forward every train/val/test CDS through **Evo2-7B**; hook the **layer-26 residual**; encode with the SAE; record per-nucleotide latent activations `z ∈ R^32768` (store per-codon mean activation).
2. On **train**: for each of 32768 latents, compute α-helix-vs-rest AUROC and helix:non-helix mean-activation ratio (gene/cluster-level). Compute a **gene- and codon-position-preserving shuffle null**; BH-FDR across 32768.
3. Candidate set = latents with train AUROC ≥ 0.70, q<0.01, ratio > matched β-sheet/coil controls. Pick top-K by **validation** AUROC (K∈{1,5,10,20}, cap 32; smallest K whose val AUROC plateaus). Freeze `S`, K, thresholds.
4. **Test once**: report selectivity of `S` on held-out test (AUROC, ratio, CIs via gene/cluster blocked bootstrap) vs shuffle null and vs β-sheet/coil-selective control features.

**Cmd**: `python m1_feature_select.py --model /mnt/quarkfs/share_models/evo2_7b_262k --sae /mnt/quarkfs/share_model/Evo-2-Layer-26-Mixed/sae-layer26-mixed-expansion_8-k_64.pt --data data/ecoli_labeled.parquet --splits data/splits.json --out results/m1_features.json`
**Expected output**: `results/m1_features.json` (feature set S with per-feature train/val/test AUROC, ratios, FDR q, s_i median-positive-activation; control-feature stats; null distribution).
**Pass predicate (C1)**: test AUROC of best feature in `S` significantly > null (lower CI > 0.5 with margin) **and** > matched β/coil controls; `S` non-empty.
**used_n**: full labeled proteome (all splits). **GPU-hours**: ~1.0–1.5 h (activation capture is the cost; 2 GPUs data-parallel over CDS). **Priority**: MUST-RUN.

---

### M2: Baseline vs steered generation + dose-response dev sweep  — **[Claims C2, C3-dev]**

**Depends on**: M1
**Goal**: generate DNA unsteered and with `S` amplified across an α grid on **development seeds**; measure predicted %-helix; select α* on validation.
**Preregistered knobs**: α grid `{0, 0.5, 1, 2, 4, 8, 16}` (× s_i units, primary residual-add intervention from FINAL_PROPOSAL §Prereg-1); fixed neutral start context (BOS + ATG), identical decoding (temperature=1.0, top-p=1.0, max_len=900 nt, fixed stop rule), **paired matched random-seed streams** across α (so α=0 baseline and each steered arm share seeds); **dev seed block D** (disjoint from confirmation block H).
**Per α**: generate ~**300 sequences** (dev). Readout pipeline P: longest-ORF → translate → **ESMFold** → **DSSP** → %-helix (all-residue and pLDDT≥70); apply treatment-independent QC (FINAL_PROPOSAL §Prereg-5); compute Evo2 perplexity; log steering telemetry (target-latent activation, TopK support overlap, residual-norm change, KL vs baseline).
**Grid**: `alpha: [0,0.5,1,2,4,8,16]`, `seed_block: [D]`
**Cmd template**: `python m2_generate_eval.py --features results/m1_features.json --alpha ${alpha} --seed_block ${seed_block} --n 300 --intervention residual_add --readout esmfold_dssp --out results/m2_a${alpha}_${seed_block}.json`
**Expected output (template)**: `results/m2_a${alpha}_${seed_block}.json` (per-seq: valid flag, %-helix all & hi-conf, %-sheet, %-coil, pLDDT, ppl, length, AA/GC composition; per-arm ITT & conditional means + cluster-bootstrap CIs).
**Analysis**: %-helix-vs-α dose-response on dev; **α\*** = argmax conditional %-helix subject to validity floor (ORF-valid rate ≥ baseline−10pp AND median ppl ≤ baseline 95th pct), frozen from validation.
**Pass predicate (C2, provisional)**: at some α>0, conditional & ITT mean %-helix > α=0 baseline with cluster-bootstrap p<0.05, without validity collapse. **Pass predicate (C3, dev)**: dose-response non-flat & single-peaked/saturating; α* identifiable.
**used_n**: 7 arms × 300 = 2100 dev generations (full scale). **GPU-hours**: ~2.5–3.5 h on **4 GPUs** (generation + ESMFold folding parallelized; folding is the main cost). **Priority**: MUST-RUN.

---

### M-CTRL: Specificity / off-target controls  — **[Claims C2, C3 specificity]**

**Depends on**: M1
**Goal**: show the helix effect is specific to `S`, not a generic perturbation.
**Arms** (at α* and one neighbor, dev seed block D, n≈300 each):
- **random-feature**: K random latents matched for decoder norm & median activation (expect: no helix gain).
- **beta-sheet-feature**: β-sheet-selective feature set (from M1 controls) amplified (expect: β-sheet↑, helix not↑).
- **null-direction**: random unit residual direction, matched norm (expect: no helix gain).
**Cmd template**: `python m2_generate_eval.py --features results/m1_features.json --control ${ctrl} --alpha ${alpha_star} --seed_block D --n 300 --out results/mctrl_${ctrl}.json`
**Grid**: `ctrl: [random_feature, beta_sheet, null_direction]`
**Expected output**: `results/mctrl_*.json`.
**Pass predicate**: helix gain of `S` steering significantly exceeds all control arms; β-sheet arm raises sheet not helix.
**used_n**: 3 × ~300. **GPU-hours**: ~1.5–2 h on up to 4 GPUs (can overlap with M3 within the 4-GPU cap). **Priority**: MUST-RUN.

---

### M3: Held-out confirmation of C2 and α* (C3)  — **[Claims C2, C3]**

**Depends on**: M2 (α* frozen)
**Goal**: confirm the effect and α* on **disjoint held-out seed block H** with larger N; primary statistics.
**Arms**: baseline (α=0), α*, and α* neighbors {α*/2, 2α*} on seed block H.
**Per arm**: generate to reach **≥ 500 QC-valid independent proteins** (generate up to ~1000 raw to absorb QC failures); identical decoding & paired seeds; readout pipeline P (locked ESMFold+DSSP).
**Grid**: `alpha: [0, alpha_star_half, alpha_star, alpha_star_2x]`, `seed_block: [H]`
**Cmd template**: `python m3_confirm.py --features results/m1_features.json --alpha ${alpha} --seed_block H --target_valid 500 --out results/m3_a${alpha}.json`
**Expected output**: `results/m3_*.json` + `results/summary.json` (primary Δhelix at α* with cluster-bootstrap CI, ITT & conditional; dose-response with simultaneous CI band; off-target deltas; composition-adjusted effect).
**Pass predicate (C2 final)**: at α*, held-out conditional AND ITT mean %-helix > baseline, cluster-bootstrap p<0.05, effect persists after composition/length/ppl adjustment. **Pass predicate (C3 final)**: α* reproduces the peak on held-out seeds; reported with CI.
**used_n**: 4 arms × ≥500 valid (≈ up to 4000 raw generations). **GPU-hours**: ~2–2.5 h on 4 GPUs. **Priority**: MUST-RUN.

---

## Run order, budget, decision gates

| Order | Milestone | Claim | GPUs | Est. wall-clock |
|------|-----------|-------|------|-----------------|
| 1 | S0 data prep | — | 1 | ~0.3 h |
| 2 | M1 feature select | C1 | 2 | ~1.0–1.5 h |
| 3 | M2 dev sweep + α* | C2/C3-dev | 4 | ~2.5–3.5 h |
| 4 | M-CTRL + M3 (share 4-GPU cap) | C2/C3 + specificity | 4 | ~2.5–3.5 h |
| — | **Total** | | ≤4 | **~6.5–8 h (fits)** |

**Decision gates** (no human; AUTO_PROCEED):
- If **C1 fails** (no α-helix-selective feature above null/controls) → stop, write negative-result note; C2/C3 cannot proceed (the given method's precondition is unmet). This is a scientific outcome, not an M0 gate.
- If **M2 shows no α>0 beats baseline** → still run M3 at the best dev arm to confirm the null on held-out seeds; report as behavior-not-reproduced with the dose-response evidence.
- **Budget guard**: if folding throughput threatens the 8 h cap, reduce **generation N per arm** toward the stated floors (M2≥200/arm dev; M3≥300 valid/arm) — never reduce model/SAE scale or drop a MUST-RUN milestone; surface the reduction. Folding tool may fall back to a local SS predictor (eval-tool choice) if ESMFold weights are unreachable.

## Data & fidelity compliance
- Provenance `adapted` (RefSeq CDS + AlphaFold-DB structures, relabeled per-residue by DSSP). Splits homology-clustered, no leakage. Labels = DSSP ground truth (not model output). used_n = full available at every milestone (strict). One consistent dataset across M1 and the readout of M2/M3.
