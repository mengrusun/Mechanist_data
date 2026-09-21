# Experiment Tracker (plan-level)

Behavior-source: given-validation · Mechanism: given (`chosen_mechanism` = SAE feature steering / amplification) · resource_fidelity: not-strict · max_parallel: 4 · GPUs 2,3,4,5.
Rows are the planned runs. `/auto-experiment` Phase 5 updates Status in place (pending → running → done/failed) and fills result/notes.

| Milestone | Claim | Run id (template) | Config | Status | Result | Notes |
|---|---|---|---|---|---|---|
| M(-1) setup | — | setup | env + tooling smoke tests | done | PASS (results/setup_report.json) | all install-and-continue; NO STOP hit. SAE validated on 7B via paper-feature reproduction (f/22326 rank-0 sheet, f/28741 top helix). Site=blocks.26.post_norm |
| M0 gate | C1 | m0_prokaryote_${helix_def}_s${seed} (6 done) | prokaryote 648 prot / 181k codons; org=eu pending | done | **established** (set-level) | set AUROC 0.89 logistic / 0.88 unweighted-mean; confound-only 0.50-0.53; null≈0.5; 321-380 FDR-sig; f/28741 rank-0. Single-feature <0.75 (peak 0.64, transparent). Frozen S=19. Euk cross-org leg pending |
| M1 harness | C2 (enabler) | m1_calibration | n=400, temp0.7, esmfold, plddt-gate50 | done | **PASS** | pos-control separates (rich 0.68 vs poor 0.14); baseline α0 helix 0.46, valid-ORF 0.86; power → n≥101/dose |
| M2 dose-response | C2 | m2_a${alpha}_s${seed} | alpha∈{-2,0,1,2,4,8,16,32} × seed∈{42,200,201} = 24 done | done | **C2 SUPPORTED** | Spearman ρ=0.759 p=1.7e-5; helix 0.475→0.834; α*=32; effect +0.36; n=110/150 (>101 power floor) |
| M3 specificity | C3 | m3_${feature_kind}_a${alpha}_s${seed} | kind×alpha∈{0,4,8,16}×seed = 36 done | done | **C3 SUPPORTED** | double dissociation: S helix+0.285/sheet−0.111; matched neither; S≫matched p=3e-23; β maintains sheet p=6e-30; not degradation-driven. Caveat: β-amplify arm weak |

GPU-hours ≈17 (dispatched 15.6); gpu_ids witnessed = {3,4,5} ⊆ {2,3,4,5} (GPU2 held by other user, avoided).

Total planned runs: 1 + 12 + 1 + 24 + 36 = 74 (M0/M2/M3 route to /experiment-queue via grid).
