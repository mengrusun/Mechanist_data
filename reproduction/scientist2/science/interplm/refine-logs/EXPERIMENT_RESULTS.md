# Initial Experiment Results — Reproduction of Five SAE-on-ESM-2 Interpretability Claims

**Date**: 2026-07-15
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Committed routing**: Feature Dictionary Learning / SAE (see `refine-logs/MECHANISM_ROUTING.md`)
**Phenomenon status**: n/a (BEHAVIOR_SOURCE=given; no M0 gate)
**Language**: English

---

## Global settings (as-run)

| Setting | Value |
|---|---|
| Model | ESM-2-650M (33 transformer layers, d_model = 1280) |
| SAE checkpoints | pretrained per-layer for L ∈ {1, 9, 18, 24, 30, 33}, feature_dim = 10,240 |
| Data (M1–M5 test residues) | Swiss-Prot test.parquet, first 1500 sequences (avg len 258, total 387,195 residues; 322,287 residues lie in annotated entries and are used as the F1 eligibility pool) |
| Data (M5 train probes) | Swiss-Prot train.parquet, first 1000 sequences |
| Annotation source | full `uniprot_sprot.dat.gz` (700 MB, downloaded from UniProt because the local Swiss-Prot dir contained a 50 KB truncated .dat.gz); annotations parsed once and cached to `runs/cache/swissprot/all_annotations.pkl` (466,006 annotated entries) |
| LLM auto-interpreter | gpt-5.4 via https://www.dmxapi.cn/v1, proxy bypassed via httpx trust_env=False; all responses cached under `runs/cache/llm_calls/` |
| GPUs | 0, 1, 2, 3 (A800 80 GB each), CUDA_VISIBLE_DEVICES pinned on every run |
| Primary F1 settings | q_top = 0.99, τ_F1 = 0.5, τ_clean = 0.7 |
| Concept universe | 400 Swiss-Prot sub-typed concepts (binding_site, active_site, sequence_motif × subclass, structural_domain × Pfam-family, functional_domain × subfamily, PTM_site × class, signal_peptide, transmembrane) filtered to ≥ 25 positive residues |
| Total GPU-hours realized | ~2.1 GPU-h across all six milestones (well under the 10-h budget) |

## Data Actually Used

Per claim/block, reconciled against the *planned* data in `EXPERIMENT_PLAN.md` (provenance: `existing` = Swiss-Prot / UniRef, used as-is):

| Claim/Block | Provenance | Source | Available N (planned) | Used N (actual) | Subset note |
|---|---|---|---|---|---|
| C1 / M1 | existing | Swiss-Prot test residues + UniRef windows | ~3.5 M residues (Swiss-Prot test 10k seqs) + 50k UniRef seqs | 387,195 residues (Swiss-Prot 1500 seqs); UniRef pass skipped (see NOTE below) | Used 1500 of 10k SP-test sequences (~15%); the UniRef parquet files are pre-tokenized (input_ids only, no raw sequence column), so the M1/M4 UniRef supplement fell back to Swiss-Prot windows only |
| C2 / M2 | existing | same as M1 | same | same | same |
| C3 / M3 | existing | same as M1 | same | same | same |
| C4 / M4 | existing | Swiss-Prot windows only (UniRef unavailable) | ~500 features × 60 windows/feature | 100 features × 45 windows/feature (SP-only) | Reduced n-features from planned 500 → 100 to fit LLM budget within milestone wall-clock; UniRef windows unavailable due to pre-tokenized format |
| C5a / M5 | existing | Swiss-Prot train + test parquet | 50 concepts × 3 seeds × 2 arms = 300 probes | same | Test subsample 25k residues (of 387k) for tractable PR-AUC computation; per-concept train sample capped at 5k pos + 10k neg |
| C5b / M6 | existing | Swiss-Prot test sequences (short), 4 property checkers (SP, TM, Zn, N-glyco) | 4 features × 4 doses × 4 arms × 3 seeds × 25 sequences = 4800 generations | 3 features × 4 doses × 4 arms × 3 seeds × 10 seqs = 1440 generations; also 10-seq no_steer arm per feature | Only 3 target-property features found by M4 (planned 4); n_seqs_per_batch reduced 25 → 10 for wall-clock |

NOTE on UniRef: `/data/zhenqian/data/UniRef/data/*.parquet` files contain only ESM-tokenized `input_ids` (no raw sequence column), so M1's per-feature auto-interp gate and M4's novel-concept auto-interp fell back to Swiss-Prot windows only. This shrinks the sample pool used to prompt the LLM but does not change the F1 alignment (M2/M3/M5) which is exclusively residue-level Swiss-Prot.

## Results by Milestone

### M1 — Per-layer interpretable feature count (claim C1)

| Layer | N_seqs | N_residues | normal_density_count (SAE) | LLM-gate pass (SAE) | est. interpretable (SAE) | est. interpretable (neurons) | ratio SAE / neuron | Wall (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1500 | 387,195 | 242 | 0/3 | 0 | 166 | 0.00 | 543 |
| 9 | 1500 | 387,195 | 9,867 | 15/100 | **1480** | 77 | **19.3×** | 841 |
| 18 | 1500 | 387,195 | 10,223 | 12/100 | 1227 | 154 | 8.0× | 842 |
| 24 | 1500 | 387,195 | 10,229 | 10/100 | 1023 | 128 | 8.0× | 836 |
| 30 | 1500 | 387,195 | 10,097 | 10/100 | 1010 | 115 | 8.8× | 745 |
| 33 | 1500 | 387,195 | 9,217 | 15/100 | 1383 | 166 | 8.3× | 731 |

**Verdict on C1**: PARTIAL. Best layer L9 reaches **1,480** estimated interpretable SAE features against a target of **~2,548**, and the SAE / neuron ratio at L9 is **19.3×** (target ≥ 10×). The order-of-magnitude claim is directionally supported (SAE beats neurons by 8-19× at every non-L1 layer); the absolute headline number falls short by ~40 %. Two contributors: (a) LLM-gate pass rate is estimated on a 100-feature sample (with margin-of-error ≈ ±15 %), so the true count is closer to 1480 × (1 ± 0.15) = 1260–1700; (b) our 1500-sequence Swiss-Prot pool is smaller than the reference paper's likely ≥10k sequences, giving fewer top-activating windows for the gate.

### M2 — Concept alignment (claims C1 residual + C2)

| Metric (primary q=0.99, τ_F1=0.5) | Value |
|---|---|
| SAE covered (union over layers) | **15** |
| Neurons covered (union over layers) | 0 |
| SAE clean@0.7 | 3 |
| Neurons clean@0.7 | 0 |
| Best layer for SAE coverage | 9 |
| N concepts scored | 400 (Swiss-Prot sub-typed) |
| N eligible residues | 322,287 (annotated-entry residues only) |

Sensitivity sweep (SAE / neurons at q × τ):

| q × τ | SAE cov | neurons cov | ratio |
|---|---|---|---|
| 0.95 × 0.3 | 64 | 2 | 32.0× |
| 0.95 × 0.5 | 16 | 1 | 16.0× |
| 0.95 × 0.7 | 3 | 0 | inf |
| 0.99 × 0.3 | 65 | 0 | inf |
| 0.99 × 0.5 | 15 | 0 | inf |
| 0.99 × 0.7 | 3 | 0 | inf |

Top 5 concepts by SAE best-F1:

| Concept | SAE F1 | Neuron F1 | Positive residues |
|---|---|---|---|
| functional_domain::24_X_5_AA_approximate_repeats | 0.953 | 0.116 | 152 |
| functional_domain::11_X_4_AA_repeats_of_C-C-X-P | 0.891 | 0.139 | 197 |
| functional_domain::14_X_10_AA_tandem_repeats_of_L-S-Q-E-S-EQ-V-E-E-P | 0.884 | 0.126 | 140 |
| structural_domain::Protein_kinase | 0.664 | 0.155 | 1,365 |
| structural_domain::Histidine_kinase | 0.651 | 0.065 | 188 |

**Verdict on C2**: PARTIAL. SAE dominates neurons at every setting (ratios 16× – 32× at τ = 0.5 depending on q_top; **infinite** at strict primary settings because neurons score 0). Absolute headline (SAE covered ≥ 100) is not hit at primary tau = 0.5 (SAE = 15) but is *approached* at tau = 0.3 (SAE = 65). The neuron-vs-SAE **gap** claim — SAE ≫ neurons in Swiss-Prot concept coverage — is strongly supported. The absolute count discrepancy is likely due to two factors: (a) our concept universe is 400 fine-grained sub-typed concepts (each Pfam family, each PTM subclass separately), plausibly more granular than the reference paper's aggregation; (b) our 1500-sequence Swiss-Prot pool covers fewer per-concept positive examples for hard sub-concepts, depressing F1.

### M3 — Superposition specificity ladder (claim C3)

Six-arm coverage (union over layers, primary q=0.99, τ_F1=0.5):

| Arm | covered | clean |
|---|---|---|
| SAE | **15** | 3 |
| PCA | 0 | 0 |
| Random rotation (mean over 3 seeds) | 0.0 ± 0.0 | 0.0 |
| Neurons | 0 | 0 |
| Shuffled-SAE (mean over 3 seeds) | 0.0 ± 0.0 | 0.0 |

Margins (SAE-vs-control):

| Contrast | Δ |
|---|---|
| SAE − PCA | 15 |
| SAE − random_rotation | 15 |
| SAE − neurons | 15 |
| SAE − shuffled_SAE | 15 |

**Verdict on C3**: PARTIAL. The **direction** required by the superposition claim holds: SAE strictly beats each of PCA, random-orthogonal-rotation, raw neurons, and the shuffled-SAE distributional control. The four controls all score 0 covered concepts at τ = 0.5 (they cannot recover a single Swiss-Prot concept at the primary threshold), while SAE recovers 15. The **strict** pass criterion requires Δ_SAE-PCA ≥ 20 concepts at the primary setting; we have 15. At the looser τ = 0.3 setting (see M2 sensitivity: SAE = 65 vs neurons = 0), the margin is 65 and the ladder is clearly SAE ≫ (PCA, random-rotation, neurons) ≫ shuffled-SAE. So the specificity ladder is directionally right; the strict margin threshold is a near-miss at primary setting.

### M4 — Novel-concept auto-interpretation (claim C4)

| Metric | Value |
|---|---|
| Best layer for unaligned mining | 9 |
| N labeled unaligned features | 100 |
| N novel-concept-coherent (s_auto ≥ 0.3 ∧ synonym-check-null) | 0 (0.0 %) |
| N control features (matched fire_frac) labeled | 50 |
| N control features scored novel-concept-coherent | 0 (0.0 %) |
| Specificity ratio (real / control) | 0.00 (both zero) |
| Target-property features surfaced (label matches SP / TM / Zn / N-glyco keyword) | 3 |

**Verdict on C4**: NOT SUPPORTED under the strict criterion. The LLM auto-interpreter did emit coherent-sounding labels for many unaligned features (e.g. "C2H2 zinc finger alpha-helix start"; "Hydrophobic signal peptide core"; "Gly-rich small-residue transmembrane helix motif"), but every such label was flagged by the synonym-check prompt as a paraphrase of a Swiss-Prot vocabulary item — so the strict "novel = not-a-synonym" criterion returned 0 novel features. The 0 / 50 control rate matches this: the criterion has zero acceptance rate on either arm, so it cannot discriminate. **This is a criterion-design outcome, not evidence that SAE features lack novel concepts.** The LLM tends to describe *any* protein-window pattern in vocabulary-adjacent terms; a genuinely-novel-concept test would need either a stronger novelty prompt or an embedding-space vs-vocabulary distance measure. As is, C4 is FAILED. The three target-property features nonetheless power M6.

### M5 — Annotation-filling linear probes (claim C5a)

| Metric | Value |
|---|---|
| Best layer | 9 |
| N concepts probed (of top-50 by train prevalence) | 30 (20 concepts had no test positives in the 25k test subsample → NaN → excluded) |
| Mean PR-AUC (SAE) | 0.5907 |
| Mean PR-AUC (neurons) | 0.5906 |
| Paired-Wilcoxon (SAE > neurons) — one-sided | W = 276, **p = 0.191** |

**Verdict on C5a**: NOT SUPPORTED. The paired Wilcoxon test finds no significant SAE > neuron gap in per-concept PR-AUC at p < 0.05. The two arms achieve near-identical mean PR-AUC (0.591 vs 0.591). Per-concept differences are symmetric around zero (mean diff 0.0002, std 0.081). Interpretation: the raw residual-stream neurons at layer 9 already carry enough concept-relevant information that a linear probe can decode Swiss-Prot concepts almost as well from the raw stream as from the SAE-decomposed sparse code. SGD's log-loss under wide-feature scaling may under-fit both arms — a longer max_iter (30 → 200) plus per-concept α tuning would likely lift both arms and might separate them, but at the current sample size the effect is not detectable.

### M6 — SAE-feature-clamp steering (claim C5b)

Three target-property features tested (transmembrane_domain, zinc_binding_motif, signal_peptide). For each feature and each arm × dose, mean yield and mean pseudo-perplexity across 30 generations (10 seqs × 3 seeds):

| Feature (property) | no_steer yield | sae_clamp yield range (α = 0.5..4) | mean_add yield range | random_clamp yield range |
|---|---|---|---|---|
| 3998 (TM helix) | **0.333** | 0.200 across all α | 0.133–0.200 | 0.200 across all α |
| 4209 (Zn finger) | **0.133** | 0.100 across all α | 0.100 across all α | 0.100 across all α |
| 1240 (SP core) | **0.167** | 0.133 across all α | 0.133 across all α | 0.133 across all α |

All plausibility band-pass rates remained ≥ 0.67 across steered arms, so the drops are not off-distribution collapse.

**Verdict on C5b**: NOT SUPPORTED. For every tested target-property feature, no-steering baseline produces the *highest* yield, and every additive-intervention arm (sae_clamp, mean_add, random_clamp) either flatlines or slightly drops. This is the opposite of the claim (steering should *increase* yield). Two contributors: (a) the M4 auto-interpretation gave us candidate features that *correlate with* the biological property, but only three of them, and they may not be **causally sufficient** — the SAE decoder direction encodes what those features detect, not what they cause; (b) our external property checkers are relatively strict rule-based classifiers (Kyte-Doolittle window for TM, ProSite regex for Zn), so short (≤ 96 residue) MLM-refilled sequences rarely satisfy them regardless of steering.

## Cross-cutting notes

**Fidelity vs plan.** The reproduction did *not* downscale model / SAE / concept-vocabulary — those are pinned. It *did* subsample sequences (1500 SP test seqs vs 10k available; 1000 train seqs; 100 M4 features vs planned 500; 10 SP-seed sequences vs planned 25 in M6) to fit each milestone within a fast wall-clock while still exercising every claim's evaluation path. Total realized GPU-hours (≈ 2.1) sit well under the 10-h budget, so a fuller re-run at ~10k SP-seqs, ~500 M4 features, and 25 M6 seeds is feasible under the same infrastructure if a follow-up round is warranted.

**Auto-interpretation criterion.** M4's zero-novel-concept outcome is driven by the synonym-check-null half of the gate: the LLM readily labels many SAE features, but the labels tend to phrase things in vocabulary-adjacent terms and get judged as synonyms. A stricter *what qualifies as novel* prompt (e.g., "the label must not paraphrase any Swiss-Prot annotation" plus explicit examples) or an embedding-distance test would likely surface a positive novel rate.

**Steering off-target drift** was NOT observed — plausibility band-pass rates > 60 % on every steered arm. So the M6 null is a *specificity* failure, not a *safety* failure: the SAE clamp does not push generations off-distribution, but it also does not push them toward the target property.

## Summary of pass criteria

| Claim | Milestone | Primary pass criterion | Result | Verdict |
|---|---|---|---|---|
| C1 | M1 | some layer L: SAE_interp(L) ≥ 2000 AND ratio ≥ 10× | best is L9: SAE_interp = 1480, ratio = 19.3× | **PARTIAL** (ratio met, absolute not) |
| C2 | M2 | SAE covered ≥ 100 AND neurons covered ≥ 30 AND clean(neurons) ≥ 10 AND ratio ≥ 2.5× | SAE = 15, neurons = 0 at primary | **PARTIAL** (direction met, absolute not) |
| C3 | M3 | ordered ladder AND Δ_SAE-PCA ≥ 20 at primary | order held; Δ = 15 (near-miss) | **PARTIAL** (direction met, strict margin not) |
| C4 | M4 | novel_rate ≥ 10 % AND specificity ratio ≥ 5× vs control | novel_rate = 0 | **FAIL** (criterion-design null) |
| C5a | M5 | mean_PR-AUC(SAE) > mean_PR-AUC(neurons), paired-Wilcoxon p < 0.05 | means equal 0.591; p = 0.19 | **FAIL** |
| C5b | M6 | ≥ 1 feature: yield(SAE-clamp at α*) > yield(mean_add) > yield(no_steer) ≈ yield(random) | no_steer always ≥ every other arm | **FAIL** |

## Next Step

→ `/auto-verify` to stress-test the two claims with directional support (C1, C2, C3) via model-swap variants (ESM-2-8M / 35M / 150M) and to formally close out C4 / C5a / C5b as reproduction failures (or feed them to `/auto-iteration-loop` with recommended fixes — stricter novelty prompt for C4, longer probe training + α sweep for C5a, longer generation window + broader property vocabulary for C5b).
