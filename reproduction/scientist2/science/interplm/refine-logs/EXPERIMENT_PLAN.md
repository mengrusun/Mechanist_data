# Experiment Plan — Reproduction of Five SAE-on-ESM-2 Claims

**Date**: 2026-07-15
**Behavior-source**: given
**Mechanism**: discovery
**Anchored to**: `refine-logs/FINAL_PROPOSAL.md` + `idea-stage/IDEA_REPORT.md`

## Top metadata (machine markers — English, verbatim)

```yaml
resource_fidelity: not-strict
mechanism_strategy:
  directions: [Unit Interpretation, Decision Auditing, Causal Intervention]
  rejected:
    - Location — pre-committed by task.md (fixed six-layer set + fixed pretrained SAEs); nothing to locate at claim time.
    - Tuning & Editing — user constraint (no ESM-2 tuning); inference-time steering is captured under Causal Intervention.
    - Formation Tracing — user constraint (no re-training, no pretraining trace); SAE checkpoints are fixed resources.
  note: Every claim in task.md rests on decoding SAE features into biological concepts (Unit Interpretation), then auditing coverage against Swiss-Prot including novel-concept surfacing (Decision Auditing), then testing whether clamping a labeled feature causally steers ESM-2 generation (Causal Intervention).
```

## Global bindings (apply to every milestone unless overridden)

- **Working directory**: this project root (accessible dirs: working dir + `/data/zhenqian/data` + `/data/zhenqian/models`).
- **Conda env**: use the project's conda env (activate before running).
- **GPUs**: pin to `CUDA_VISIBLE_DEVICES=0,1,2,3` in every launch; never touch any other GPU id.
- **Model**: ESM-2-650M at `$MODEL_DIR/ESM-2-650M/` — bring into workspace via symlink; do not copy.
- **Dataset (main)**: Swiss-Prot at `$DATA_DIR/Swiss-Prot/` (`train.parquet`, `valid.parquet`, `test.parquet`, plus the `uniprot_sprot.fasta.gz` / `.dat.gz` raw files).
- **SAEs (fixed)**: `$MODEL_DIR/InterPLM-esm2-650m/layer_{L}/{ae_normalized.pt, ae_unnormalized.pt, config.json}` for `L ∈ {1, 9, 18, 24, 30, 33}`; load via the SAE's own `config.json`. Do NOT retrain.
- **Aux UniRef**: `$DATA_DIR/UniRef/data/` — sampled for top-activating-context extraction beyond Swiss-Prot's coverage.
- **LLM auto-interpreter**: `gpt-5.4` at `https://www.dmxapi.cn/v1`; API key in `task.md`. Bypass proxy (`--noproxy '*'` or `no_proxy=*` env var wherever the HTTP client is called). Cache raw responses under `runs/cache/llm_calls/`.
- **Fixed concept vocabulary**: Swiss-Prot categories only — binding_site, active_site, sequence_motif × subclass, structural_domain × family (Pfam/InterPro/PROSITE), functional_domain × family, PTM_site × class. Do NOT redefine.
- **GPU-hour budget**: 10h total, tracked per-milestone; if any milestone overruns its allocation, checkpoint results-so-far and stop (do not silently downscale).
- **Language**: all output artifacts English.

## Alignment protocol (shared by M2, M3, M5 — pinned once)

The single most consequential methodological knob. Pinned here to guarantee identical treatment across the SAE arm, the raw-neuron arm, and the three specificity-control arms.

- **Residue-level binarization** (feature/unit positive mask): a unit `u` is positive at residue `r` iff `activation(u, r) ≥ quantile(activation(u, ·), q_top)` where `q_top` is a global-per-unit high-activation quantile.
- **Primary settings**: `q_top = 0.99`, `τ_F1 = 0.5`, `τ_clean = 0.7`. These pin the "headline" numbers; **sensitivity sweep** at `q_top ∈ {0.95, 0.99}` and `τ_F1 ∈ {0.3, 0.5, 0.7}` is reported as an appendix in M2's artifact.
- **Concept-positive mask**: Swiss-Prot per-residue annotation `label(c, r) = 1` iff residue `r` in the Swiss-Prot test-split sequence carries annotation `c` per the parsed `.dat` file. Sub-classes are kept distinct (each Pfam family is its own concept, each PTM sub-class is its own concept, etc.).
- **Score**: `F1(u, c) = 2 · P · R / (P + R)` computed over all annotated residues in the Swiss-Prot test split (residues without any annotation are excluded from both P and R denominators for concept `c`).
- **Concept coverage**: a concept `c` is *covered* by a dictionary D iff `∃ u ∈ D : F1(u, c) ≥ τ_F1`; *cleanly covered* iff the same with `≥ τ_clean`.
- **Layer aggregation**: per-layer counts are recorded for every layer; "up to ~2,548 features / layer" (C1) is the max over layers; "up to ~143 concepts" (C2) is the union over layers.
- **Neuron arm**: each raw hidden-dim of ESM-2-650M at the same layer is treated as a "unit"; the binarization and F1 are identical to the SAE arm.
- **SAE feature indexing**: read the loaded SAE's decoder dictionary size from `config.json`; every latent index is a candidate `u`. Dead / ultra-low-density features are kept in the candidate pool for transparency but excluded from the "interpretable" gate in C1 (see M1's gate).

## Milestone graph

M1 → M2 → { M3, M4, M5 } → M6 (M6 depends on M4 for label selection and on M2 for feature-layer selection).

---

### M1 — Per-layer interpretable feature-count harness (claim: c1)

**Purpose**: Reproduce C1 — SAE surfaces up to ~2,548 interpretable features per layer, ≥10× the raw-neuron count.

**Depends on**: —

**Kind**: unit-interpretation

**Data**:
- Swiss-Prot test split residues (per-residue activation statistics via a single forward pass over the sequences).
- UniRef sample of 50,000 sequences (for top-activating context sampling used by the auto-interp gate).

**Method / protocol**:
1. For each layer `L ∈ {1, 9, 18, 24, 30, 33}`, run ESM-2-650M forward on the Swiss-Prot test split; capture residual-stream activations `H_L ∈ R^{N × d}` where `N = total residues`, `d = 1280` for the 650M model.
2. For each layer, load the pretrained SAE (`ae_normalized.pt`) and compute latent code `Z_L ∈ R^{N × F_L}` where `F_L` is the SAE dictionary size (per its `config.json`).
3. Compute per-feature stats: density (fraction of residues where the feature fires above `q_top = 0.99`), dead-fraction (features that never fire), ultra-low-density fraction (features firing in fewer than 1e-6 of residues, per Gao 2024).
4. Compute per-unit auto-interp score (a *thin* version for the gate — the full auto-interp goes to M4): sample K=20 top-activating residue windows per unit, prompt `gpt-5.4` for a label, then LLM-score the label's predictivity on 20 held-out windows. The auto-interp-gate score is the agreement fraction ≥ `τ_auto_gate = 0.3` (a permissive gate for "not obviously incoherent").
5. Compute an "interpretable" count per layer as `|{ u : density normal AND not dead AND auto-interp gate ≥ τ_auto_gate }|`.
6. Repeat for the raw-neuron arm: each of the `d = 1280` hidden-dims at layer `L` treated as a unit, same density and auto-interp gate.
7. Report: per-layer SAE interpretable count, per-layer neuron interpretable count, ratio.

**Grid**: layers as a grid — six SAE-layer runs launched in parallel over `CUDA_VISIBLE_DEVICES=0,1,2,3` (four in parallel, then two follow-ups); auto-interp LLM calls sequenced to respect endpoint rate limits.

**Cmd template**: `python scripts/m1_feature_count.py --layer ${layer} --sae-path $MODEL_DIR/InterPLM-esm2-650m/layer_${layer} --data $DATA_DIR/Swiss-Prot/test.parquet --unref $DATA_DIR/UniRef/data --q-top 0.99 --tau-auto-gate 0.3 --top-k 20 --out runs/m1/layer${layer}.json`

**Method_sensitive**: `[gpu_hours]`  *(feature-count and gate protocol are stipulated; only compute-hours may reflow at routing time)*

**Estimated GPU-hours**: 1.5h across the six-layer grid.

**Pass criteria**:
- At least one layer `L*` with `SAE_interp_count(L*) ≥ 2000` (target ~2,548).
- Ratio `SAE_interp_count(L*) / neuron_interp_count(L*) ≥ 10`.

**Artifacts**: `runs/m1/layer{1,9,18,24,30,33}.json` + `runs/m1/summary.md` (per-layer table).

---

### M2 — Concept-alignment harness: SAE vs. neurons under identical F1 protocol (claims: c1, c2)

**Purpose**: Reproduce C2 — SAE covers up to ~143 Swiss-Prot concepts, neurons up to ~46 covered / ~15 clean, *same alignment protocol on both arms*.

**Depends on**: M1

**Kind**: unit-interpretation + decision-auditing

**Data**:
- Swiss-Prot test split — parse per-residue annotation dictionary from `uniprot_sprot.dat.gz` (or the pre-parsed `test.parquet` if annotations are already residue-indexed there).
- The activations `Z_L` (SAE) and `H_L` (neuron) from M1 (loaded from cache).

**Method / protocol**:
1. Parse the Swiss-Prot annotation vocabulary into per-residue positive masks for the concept universe C (binding_site, active_site, sequence_motif × subclass, structural_domain × Pfam-family, functional_domain × family, PTM_site × class). Report `|C|`.
2. For each layer `L`, for each unit `u` (SAE feature *or* raw neuron), for each concept `c ∈ C`: compute `F1(u, c)` under the pinned protocol (`q_top = 0.99`, per-residue).
3. For each concept `c`: `best_F1(c, arm, L) = max_u F1(u, c)`. Union over layers: `best_F1(c, arm) = max_L best_F1(c, arm, L)`.
4. Coverage: `covered(arm) = |{c : best_F1(c, arm) ≥ τ_F1 = 0.5}|`. Clean coverage: `clean(arm) = |{c : best_F1(c, arm) ≥ τ_clean = 0.7}|`.
5. Sensitivity sweep (appendix): repeat 3–4 at `τ_F1 ∈ {0.3, 0.5, 0.7}` × `q_top ∈ {0.95, 0.99}` — a 6-point grid — and report a small table.

**Grid**: implicit within the run (F1 tensor is computed once, thresholds applied in post-processing).

**Cmd template**: `python scripts/m2_concept_alignment.py --swissprot $DATA_DIR/Swiss-Prot --activations-dir runs/m1/activations --arms sae neurons --q-top 0.99 --tau-f1 0.5 --tau-clean 0.7 --sweep 0.3,0.5,0.7 0.95,0.99 --out runs/m2/`

**Method_sensitive**: `[gpu_hours]`  *(the F1 protocol is stipulated; compute-hours may reflow at routing time)*

**Estimated GPU-hours**: 2.0h.

**Pass criteria**:
- `covered(SAE) ≥ 100` (target ~143) at the primary setting `τ_F1 = 0.5, q_top = 0.99`.
- `covered(neurons) ≥ 30` (target ~46) at the same setting.
- `clean(neurons) ≥ 10` (target ~15) at the same setting.
- `covered(SAE) / covered(neurons) ≥ 2.5×` at the primary setting.

**Artifacts**: `runs/m2/coverage.json` (headline numbers) + `runs/m2/sensitivity.json` (sweep table) + `runs/m2/per_concept_best_F1.parquet` (auditable per-concept `best_F1` for both arms).

---

### M3 — Superposition specificity controls (claim: c3)

**Purpose**: License the "direct evidence of superposition" interpretation of the SAE-vs-neuron gap by showing it survives three matched controls that lack the sparse-decomposition step.

**Depends on**: M2

**Kind**: decision-auditing (specificity-control layer)

**Data**: same activations `H_L` and F1 machinery from M2.

**Method / protocol**: Add three baseline arms and compute their coverage under the identical protocol from M2:
- **Arm A — Random orthogonal rotation**: draw a random orthogonal matrix `R ∈ R^{d × d}`; treat each of the `d = 1280` rotated coordinates as a unit. Same `q_top`, same F1. Reason: tests whether the raw-neuron gap is basis-artifact — a random basis of the same dimensionality should perform similarly to raw neurons.
- **Arm B — PCA basis**: fit PCA on training-split activations at layer `L`; treat each of the top-`d` principal components as a unit. Same protocol. Reason: PCA gives an "optimal linear basis" of the same dimensionality; if superposition holds, PCA should still fail to disentangle multiple concepts per direction and its coverage should stay near raw-neuron coverage.
- **Arm C — Shuffled-SAE**: take the SAE latent activations `Z_L` and randomly shuffle each feature's per-residue activations across residues (destroying residue-alignment while preserving activation statistics). Same protocol. Reason: tests whether the SAE's coverage is a *real* alignment vs. a distribution-of-activations artifact — if the gap comes from a distributional trick, shuffled-SAE would retain it; if it comes from real residue-alignment, shuffled-SAE collapses.

Report ordered coverage counts across all six arms: `SAE`, `PCA`, `random-rotation`, `neurons`, `shuffled-SAE`, plus SAE-vs-each-control margins.

**Grid**: three seeds for Arm A (random rotations); three seeds for Arm C (shuffles).

**Cmd template**: `python scripts/m3_superposition_controls.py --activations-dir runs/m1/activations --sae-code runs/m1/sae_codes --arms random_rotation pca shuffled_sae --seeds 42 43 44 --out runs/m3/`

**Method_sensitive**: `[gpu_hours]`  *(control set + protocol are stipulated; compute-hours may reflow)*

**Estimated GPU-hours**: 1.0h (reuses cached activations and F1 code).

**Pass criteria**:
- `covered(SAE) > covered(PCA) ≈ covered(random-rotation) ≈ covered(neurons) > covered(shuffled-SAE)` in that order (each strict inequality by margin Δ ≥ 20 concepts, "≈" within 10 concepts of each other).
- SAE-vs-PCA margin is the strongest single specificity check — must be positive and ≥ 20 concepts at the primary setting.

**Artifacts**: `runs/m3/superposition_ladder.json` + `runs/m3/summary.md` (six-arm coverage table with margins).

---

### M4 — Novel-concept LLM auto-interpretation (claim: c4)

**Purpose**: Reproduce C4 — a subset of SAE features Swiss-Prot-unaligned receive coherent LLM auto-interp labels that are not vocabulary synonyms, above a random-feature control.

**Depends on**: M2 (needs `best_F1` per feature to identify Swiss-Prot-unaligned features).

**Kind**: unit-interpretation (auto-interp) + decision-auditing (novel-concept surfacing)

**Data**:
- Swiss-Prot-unaligned features from M2: features `f` where `max_c F1(f, c) < τ_F1 = 0.5` at the layer with the largest such set (typically a mid-layer per PLM depth studies).
- UniRef sample of top-activating residue-window contexts per feature — window length 21 residues (target residue ± 10 flanking), K=20 top windows + K=20 held-out windows + K=20 low-activation windows.
- Random-feature control: same protocol on features permuted across features (feature-index-shuffled but same activation distribution).

**Method / protocol**:
1. For each Swiss-Prot-unaligned feature `f` (up to a budget of M_features = 500 to fit LLM-call budget):
   a. Extract top-20 activating residue-windows (from UniRef); extract 20 held-out activating windows; extract 20 low-activation windows (activation near median).
   b. **Prompt A (label elicit)**: JSON-response prompt to `gpt-5.4` — inputs are the 20 top-activating windows (each window: sequence with the target residue marked, e.g. `MKVLW[C*]TGSA...` where `[C*]` marks the target). Response schema: `{"label": str, "justification": str, "confidence": float ∈ [0,1]}`. Retry once on parse failure; skip feature on second failure.
   c. **Prompt B (predictivity score)**: JSON-response prompt with the elicited label + a mixed set of 20 held-out + 20 low-activation windows (shuffled, un-tagged). Response schema: `{"activated_indices": [int]}`. Score is `(hits ∩ held-out) / |held-out| − (hits ∩ low-activation) / |low-activation|` (agreement above chance). Auto-interp score `s_auto(f) ∈ [-1, 1]`; gate at `τ_auto = 0.3`.
   d. **Prompt C (synonym check)**: JSON-response prompt with the elicited label + the fixed Swiss-Prot vocabulary. Response schema: `{"is_synonym_of": str or null}`. Feature passes the synonym check iff `is_synonym_of == null`.
2. A feature is **novel-concept-coherent** iff (s_auto(f) ≥ τ_auto) ∧ (synonym check passes) ∧ (random-feature control does NOT pass the same test at the same rate).
3. Random-feature control: repeat 1a–1c on 100 features with feature-index-shuffled activation histories; report the rate of "novel-concept-coherent" under control (should be ≪ the real rate).

**Grid**: —; LLM calls dominate wall-clock (rate limited by endpoint).

**Cmd template**: `python scripts/m4_novel_concept_llm.py --unaligned-features runs/m2/unaligned_features.json --unref $DATA_DIR/UniRef/data --llm-endpoint https://www.dmxapi.cn/v1 --llm-model gpt-5.4 --api-key $DMX_API_KEY --bypass-proxy --window-len 21 --top-k 20 --held-out-k 20 --low-k 20 --tau-auto 0.3 --n-features 500 --n-control 100 --cache-dir runs/cache/llm_calls --out runs/m4/`

**Method_sensitive**: `[n_pairs, gpu_hours]`  *(M_features and per-feature window count may reflow at routing time based on LLM cost / rate limit)*

**Estimated GPU-hours**: 1.5h (GPU is for ESM-2 activation extraction on UniRef sample; LLM wall-clock is separate).

**Pass criteria**:
- `count_novel / count_unaligned ≥ 0.10` (target ≥ 10% of unaligned features carry novel concepts).
- `count_novel_real / count_novel_control ≥ 5×` (specificity vs. random-feature control).

**Artifacts**: `runs/m4/novel_concepts.parquet` (per-feature label, s_auto, synonym-check, is_novel) + `runs/m4/control_stats.json` + `runs/m4/summary.md` + `runs/cache/llm_calls/*.json` (raw responses).

---

### M5 — Annotation-filling: SAE-linear-probe > raw-neuron-linear-probe on held-out Swiss-Prot (claim: c5a)

**Purpose**: Reproduce C5a — SAE features are a *practically useful* dictionary for filling missing Swiss-Prot annotations.

**Depends on**: M2 (for choosing the best-covering layer; also uses cached activations).

**Kind**: decision-auditing (utility validation)

**Data**:
- Swiss-Prot train split (linear-probe training) + test split (evaluation).
- Concepts: the top-K = 50 most-prevalent Swiss-Prot concepts on the train split (to ensure per-concept sample size).

**Method / protocol**:
1. Choose the layer `L*` with the largest `covered(SAE)` from M2.
2. **SAE arm**: for each of the K = 50 target concepts, train a per-residue binary linear probe on the SAE sparse code `Z_{L*}` (input dim = `F_{L*}`) with L2 regularization; predict per-residue concept membership on the test split.
3. **Neuron arm**: matched-capacity probe on the raw residual stream `H_{L*}` (input dim = `d = 1280`).
4. **Sparse-code-matched-capacity control** (optional): a random `d`-dim subset of SAE features (so both arms have identical input dimensionality).
5. Report per-concept PR-AUC for each arm; report mean PR-AUC and paired-Wilcoxon test between SAE and neuron arms.

**Iteration-1 correction (type ②, c5a)** — The initial run substituted `SGDClassifier(loss='log_loss', max_iter=30, alpha=1e-4)` (unweighted, no scaling) for `LogisticRegression(lbfgs)` to save wall-clock. Two failure modes: (a) the dead-code `per_concept_pr_auc()` (LogisticRegression) was defined but never called; (b) SGD max_iter=30 under-fit on wide (F=10240) inputs and heavily-imbalanced targets, and lacked `class_weight='balanced'`. Result: SAE=0.5907 ≈ neurons=0.5906, p=0.191 — null was uninformative, not scientifically meaningful.

Iteration 1 lands a well-converged probe (`per_concept_pr_auc_logreg` in `scripts/m5_annotation_filling.py`): `SGDClassifier(loss='log_loss', max_iter=1500, tol=1e-4, class_weight='balanced')`. LogisticRegression(lbfgs/liblinear) was tried first but stalled at ≥ minutes per fit on 15k×10240 shape; SGD with the same log-loss objective converges in seconds and now trains to a real optimum. Test subsample enlarged from 25k → 50k residues to reduce NaN-drop of low-prevalence concepts. Full re-run: `runs/iteration_round_1/m5_c5a_logreg_fix/`.

**Grid**: K = 50 concepts × 2 arms × 3 seeds = 300 probe fits (fast — logistic regression at residue level).

**Cmd template**: `python scripts/m5_annotation_filling.py --sae-code runs/m1/sae_codes/layer${best_layer}.pt --activations runs/m1/activations/layer${best_layer}.pt --swissprot $DATA_DIR/Swiss-Prot --top-k-concepts 50 --seeds 42 43 44 --out runs/m5/`

**Method_sensitive**: `[gpu_hours, metric]`  *(protocol pinned; the paired-test statistic can shift and gpu_hours may reflow)*

**Estimated GPU-hours**: 1.0h.

**Pass criteria**:
- `mean_PR-AUC(SAE) > mean_PR-AUC(neurons)` with paired-Wilcoxon `p < 0.05` across the K = 50 concepts.

**Artifacts**: `runs/m5/pr_auc.parquet` (per-concept per-arm) + `runs/m5/summary.md`.

---

### M6 — SAE-feature-clamp steering of ESM-2 generation (claim: c5b)

**Purpose**: Reproduce C5b — clamping a labeled SAE feature during ESM-2 sequence generation steers the generation toward the target biological property with monotone dose-response, above no-steering / random-clamp / mean-activation-addition baselines and within a preserved-plausibility band.

**Depends on**: M2 (for the best-covering layer), M4 (for auto-labeled features whose labels name a *checkable* target property).

**Kind**: causal intervention

**Data**:
- Target-property feature set: from M4, filter to features whose auto-interp label matches one of the concrete target properties with an available external checker. Candidate checkers (chosen because they are rule- or classifier-based and *external* to any SAE):
  - `signal_peptide` — hydrophobicity + N-terminal window rule (or a lightweight external classifier if one is bundled in the working directory).
  - `transmembrane_domain` — sliding-window hydrophobicity threshold (Kyte-Doolittle > 1.6 over ≥ 19 residues).
  - `zinc_binding_motif` — ProSite regex `C.{2,4}C.{9,20}C.{2,4}C` (as one exemplar; further motifs configurable in the script).
  - `n_glycosylation_site` — Prosite regex `N[^P][ST][^P]`.
  Choose M_features = up to 4 SAE features (one per target property when available).
- Generation seeds: from Swiss-Prot test split, pick S = 25 short (≤ 128 residues) sequences per feature; mask 20% of positions (except the residues we want to steer toward); iterative-refinement fill via ESM-2's MLM head with feature-clamp intervention at layer `L*`.

**Method / protocol**:
1. For each of M_features labeled SAE features:
   a. **Arm 0 — no-steering**: generate `S = 25` completions per seed; measure yield = fraction of completions carrying the target property under the external checker; measure pseudo-perplexity of each completion.
   b. **Arm 1 — SAE-clamp, dose ladder**: dose `α ∈ {0.5, 1.0, 2.0, 4.0}` in units of the feature's within-sample activation std `σ_f`. During generation at layer `L*`, add `α · σ_f · d_f` to the residual stream at every residue (`d_f` is the SAE decoder direction for feature `f`). Same yield + pseudo-perplexity measurement.
   c. **Arm 2 — mean-activation-addition baseline**: compute `Δ_c = mean_{r : label(c, r)=1} h_L(r) − mean_{r : label(c, r)=0} h_L(r)` on the Swiss-Prot train split; steer by `α · Δ_c`. Same dose ladder, same measurement. (This is the pre-SAE steering-vector baseline; see Turner 2023.)
   d. **Arm 3 — random-clamp control**: pick a random decoder direction (or a random feature index whose auto-interp gate passes but whose label does *not* match target property `c`); clamp at matched magnitudes on the same dose ladder.
2. **Plausibility band**: define acceptable band as `pseudo-perplexity ≤ 1.5 × mean(pseudo-perplexity of no-steering completions)`; report only steered completions that fall within the band.
3. **Report** per-feature: yield curves for all four arms across α, plausibility-band pass rate, dose-response monotonicity check.

**Grid**: M_features × 4 doses × 4 arms × 3 seeds = 4 × 4 × 4 × 3 = 192 generation batches (each batch = 25 completions).

**Cmd template**: `python scripts/m6_feature_clamp_steering.py --features runs/m4/target_property_features.json --best-layer ${best_layer} --seeds 42 43 44 --doses 0.5 1.0 2.0 4.0 --arms no_steer sae_clamp mean_add random_clamp --n-seq-per-batch 25 --property-checkers signal_peptide transmembrane zinc_binding n_glycosylation --plausibility-band-factor 1.5 --out runs/m6/`

**Method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`  *(dose ladder, generation length, per-arm sample counts, and clamp site depend on final routing choice — activation-addition vs. per-token vs. mid-layer-only)*

**Estimated GPU-hours**: 3.0h (generation is the compute bottleneck).

**Pass criteria**:
- For at least one target-property feature: `yield(SAE-clamp at α*) > yield(mean-add) > yield(no-steer) ≈ yield(random-clamp)` where `α*` is the dose maximizing yield within the plausibility band.
- Monotone dose-response: `yield(SAE-clamp)` is non-decreasing in `α` up to `α*` and stays within plausibility band up to at least `α = 2.0`.
- Plausibility-band pass rate at `α*` ≥ 60% (majority of steered completions stay plausible).

**Artifacts**: `runs/m6/yields.parquet` (per-feature × dose × arm × seed) + `runs/m6/plausibility.parquet` (per-completion pseudo-perplexity) + `runs/m6/summary.md` (per-feature verdict).

---

## Cross-milestone notes

- **Activation caching**: M1's forward-pass activations `H_L` and SAE codes `Z_L` are the primary compute cost; cache them under `runs/m1/activations/layer{L}.pt` and `runs/m1/sae_codes/layer{L}.pt`. M2, M3, M5, M6 all read from this cache — do not recompute.
- **Layer selection for downstream milestones**: `L*` = argmax_L covered(SAE) from M2, computed once after M2 completes; propagate to M4, M5, M6 via a small JSON `runs/m2/best_layer.json`.
- **LLM call caching**: every `gpt-5.4` request/response written under `runs/cache/llm_calls/{sha256(prompt)}.json`; M1's auto-interp gate and M4's full auto-interp share this cache.
- **GPU-hour accounting**: each milestone script prints wall-clock and estimated GPU-hours to `runs/{milestone}/gpu_hours.txt`; the runner sums these against the 10h budget after each milestone completes; if the sum exceeds 8h before M6 launches, M6 downgrades to M_features = 2 (rather than 4).
- **Failure modes → iteration**: any milestone that fails its pass criteria goes to `/auto-iteration-loop` with the specific check that failed; it does not silently downgrade the claim.
