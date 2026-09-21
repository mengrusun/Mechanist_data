# Experiment Plan — Steering Evo2-7B toward higher α-helical content

<!-- ===== TOP METADATA (machine markers) ===== -->
behavior_source: given
mechanism: discovery
resource_fidelity: cost-aware            # NOT strict (strict is stamped only for given+given). Model is still HARD-pinned below.
chosen_mechanism: n/a                     # mechanism:discovery — concrete family bound at /auto-experiment Phase 1.5 by /mechanism-skills
m0_gate: none                            # behavior_source=given → NO phenomenon-validation gate; NO milestone declares depends_on:[M0]
mechanism_strategy:
  directions: [Location, Causal Intervention, Tuning & Editing]   # execution order
  rejected:
    - Formation Tracing — inference-time control is the goal, not training-time genesis; out of scope, most expensive direction.
    - Unit Interpretation (standalone) — α-helix direction is decoded within Location, not an end in itself.
    - Decision Auditing — no trustworthiness/spurious-feature question; this is generation control.
  note: Locate an α-helix propensity direction (Location), causally confirm it via steering dose-response + specificity (Causal Intervention), then apply it to generate high-α-helix DNA (Tuning & Editing).

## HARD constraints (non-negotiable, from task.md)
- **Model MUST be Evo2-7B (`arcinstitute/evo2_7b`)** for every generation/intervention milestone. No smaller Evo variant, no substitute model. This binds M1–M4 below.

## NOTICE (planning realism, from task.md)
- Model & data via HuggingFace (token in task.md); a network proxy may be needed to accelerate downloads. `evo2` is NOT yet installed — the experiment stage installs/sets it up. Environment: 8×A800-80GB (several free), Python 3.13 miniconda at `/data/wanghaoxiong/miniconda3`, ~684 GB free under `/data`. Budget covers full-scale Evo2-7B inference on short coding windows → plan at full scale.

## Claims covered
- **Claim 1 (PRIMARY)** — a localizable internal component of Evo2-7B causally increases generated α-helical content (%H), specifically. Covered by M1 → M2 → M3 (and demonstrated at scale by M4).
- **Claim 2 (SUPPORTING)** — the α-helix-content evaluation (ORF→translate→SS-predict→%H) agrees with a structure-based DSSP reference. Covered by E1.

---

## E1: α-helical-content evaluation harness  *(verifies Claim 2; prerequisite infra)*
**Purpose**: build and validate the metric everything else scores against.
**Steps**:
- Implement ORF/reading-frame finder on generated DNA → translate CDS → protein.
- Primary online metric: fast sequence-based SS predictor (e.g. NetSurfP-3.0 / S4PRED) → %H = fraction of residues assigned α-helix.
- Reference: fold a held-out subset with ESMFold (or AlphaFold2) → DSSP (H/G/I→H) → structure-based %H.
- Validate agreement: on ≥ ~100 held-out proteins with known DSSP labels, compute correlation(fast %H, DSSP %H).
**Pass criterion (Claim 2)**: Pearson r ≥ 0.7 (tunable) between fast %H and DSSP %H, and correct reading-frame recovery on validation cases.
**Data**: DSSP-annotated PDB proteins (known %H) + their CDS.
**Expected output**: `results/E1_eval_harness_validation.json` (correlation, frame-recovery rate), reusable scorer module.
**Priority**: MUST-RUN. **Depends on**: none. **Est. GPU-hours**: ~2h (ESMFold on subset).

## M1: Locate candidate α-helical-propensity directions  *(Location — verifies Claim 1, screen)*
**Purpose**: cheap correlational screen to nominate directions/features/layers (a *located* candidate is a hypothesis, not yet causal).
**Steps** (run the candidate extractors; ranking decides which feed M2):
- Build contrastive coding-window set: high-α-helix (all-α class) vs low-α-helix (all-β / low-helix) CDS from DSSP-annotated PDB (CATH/SCOP classes), held-out split reserved.
- Extractor A — **SAE feature**: run `Goodfire/Evo-2-Layer-26-Mixed` SAE on Evo2-7B activations; rank features by separation (e.g. AUROC / mean-activation gap) between high-α and low-α windows.
- Extractor B — **contrastive vector**: mean-difference of Evo2-7B hidden activations (high-α minus low-α) at several candidate layers/sites.
- Extractor C — **linear probe**: train a linear probe for per-position α-helix propensity on hidden activations; take the weight direction.
- Rank candidates by separation quality + a quick logit/likelihood-lens sanity check.
**Expected output**: `results/M1_candidate_directions.json` (ranked candidate {family, layer/site, feature-id or vector, separation score}).
**Priority**: MUST-RUN. **Depends on**: none (uses E1 scorer for labeling only).
**method_sensitive**: [sites, n_pairs, metric]   # exact layer/site set, #contrastive pairs, and separation metric depend on the family bound at Phase 1.5
**Est. GPU-hours**: ~3h.

## M2: Causal steering dose-response  *(Causal Intervention — verifies Claim 1, core)*
**Purpose**: promote a *located* candidate to a *causal* control knob by intervening during generation and measuring %H vs dose.
**Steps**:
- For the top candidate(s) from M1, intervene during autoregressive generation (add α·v to residual stream, or clamp/boost the SAE feature) at the candidate site.
- Conditions: baseline (α=0) + a sweep of coefficients α. Same prompts, fixed decoding (temperature/top-p) across conditions. N ≥ ~100 generations per condition.
- Score every generation with the E1 harness (%H), plus covariates (ORF validity, length, GC, Evo2-7B coding-likelihood).
**Expected sign**: positive — %H increases with α. **Expected magnitude / dose-response**: monotone rise in mean %H over a working α range, plateau/decline (with validity loss) at large α. **Specificity control (recorded here, tested in M3)**: matched control direction to be compared in M3.
**Pass criterion (partial, Claim 1)**: mean %H(best α) > mean %H(baseline), one-sided, multiple-comparison-corrected, with a visible monotonic trend across α.
**Grid**: `{ alpha: [0, 0.5, 1, 2, 4, 8], seed: [42, 43, 44] }` (α set illustrative; bound at routing).
**Cmd template**: `python run_steer_generate.py --model arcinstitute/evo2_7b --direction ${direction} --alpha ${alpha} --seed ${seed} --n 100 --score results/M2_a${alpha}_s${seed}.json`
**Priority**: MUST-RUN. **Depends on**: [E1, M1].
**method_sensitive**: [sites, metric, n_pairs, gpu_hours]   # intervention site, effect metric, pair count, and per-run cost depend on the bound submethod
**Est. GPU-hours per run**: ~1h (short-window 7B generation + scoring).

## M3: Specificity & confound battery  *(Causal Intervention — verifies Claim 1, specificity)*
**Purpose**: rule out that the %H gain is a random-direction effect, a β-sheet trade-off artifact, or trivial decoding.
**Steps / controls**:
- (a) **Matched random/control direction** at equal norm and site → expect NO significant %H gain (dose-response flat).
- (b) **Off-target intactness**: β-sheet %E and other SS fractions, ORF validity rate, GC, and Evo2-7B coding-likelihood not degraded beyond a set tolerance; %H rise is not merely %E suppression.
- (c) **Naive-baseline comparison**: temperature/top-p change and rejection-sampling-toward-%H baselines → show the mechanism achieves higher %H at equal or better sequence validity (mechanism adds beyond trivial sampling).
**Pass criterion (completes Claim 1)**: control direction shows no significant gain; off-target metrics within tolerance; steering beats naive baselines on the %H/validity frontier.
**Expected output**: `results/M3_specificity.json`.
**Priority**: MUST-RUN. **Depends on**: [M2].
**method_sensitive**: [sites, metric, n_pairs, gpu_hours]
**Est. GPU-hours**: ~3h.

## M5: Composition-controlled, independent-predictor re-evaluation of the steered-%H endpoint  *(Causal Intervention — hardens Claim 1's PRIMARY endpoint; added by iteration-loop type-② fix)*
**Purpose**: the PRIMARY %H endpoint was a single ESM2-probe proxy validated only on natural proteins but applied to steered, GC-collapsed OOD generations (verify Phase-2 integrity FAIL: Check E scope + Check F synthetic_proxy). Make the endpoint trustworthy WITHOUT changing the claim or regenerating (re-score the existing M2/M4 generations).
**Steps**:
- Re-score the SAME steered vs baseline proteins with ≥2 SS predictors INDEPENDENT of the ESM2 backbone: (i) GOR-windowed logistic (±8-aa one-hot window, fit on the SAME 600 DSSP-labeled train proteins — identical to the verify C2 method-swap), (ii) Chou-Fasman helix propensity + nucleation/extension. Report per-predictor uplift with bootstrap 95% CIs and MWU p, and cross-predictor agreement ON the steered set.
- **GC / composition control**: (a) regress %H~GC within baseline; (b) GC-stratified and nearest-neighbour GC-matched steered-vs-baseline uplift with bootstrap CI + permutation p; (c) report amino-acid composition shift, protein length, and low-complexity (max single-AA fraction) per condition; effect stratified by GC.
- **Structural grounding**: fold a steered subset (ESMFold→DSSP) if any LOCAL predictor is available; otherwise label the result "sequence-predictor-supported, not structurally confirmed" (ESMFold 2.7 GB is HF-mirror-CDN-blocked).
- **Non-monotonicity**: report the dose curve with per-α bootstrap CIs, the honest full-range Spearman, and the high-dose (α≥8) working range explicitly — do NOT call the full curve monotonic.
**Pass criterion (honest)**: C1's PRIMARY endpoint is trustworthy iff the uplift is (i) reproduced by ≥1 genuinely independent predictor AND (ii) survives GC/composition control (GC-matched CI excludes 0) AND (iii) not explained by low-complexity/length shifts. If it does NOT survive, NARROW or falsify C1 honestly — do not force a PASS.
**Expected output**: `runs/iteration_round_1/M5_structural_gc_control.json`.
**Priority**: MUST-RUN (integrity repair). **Depends on**: [M2, M3, M4]. **Est. GPU-hours**: ~0.02h (re-scores existing generations).

## M4: High-α-helix generation at scale  *(Tuning & Editing — demonstrates the capability)*
**Purpose**: apply the confirmed knob to produce a high-α-helix DNA library and quantify achievable uplift vs validity.
**Steps**:
- Use best M2/M3 config to generate a library; report %H distribution vs baseline, ORF-validity retention, and a structure-validated subset (ESMFold+DSSP %H).
- Summarize the achievable α-helix uplift and the validity trade-off curve.
**Expected output**: `results/M4_library.json` + structure-validated subset report.
**Priority**: SHOULD-RUN (headline capability result). **Depends on**: [M2, M3].
**method_sensitive**: [sites, metric, gpu_hours]
**Est. GPU-hours**: ~4h.

---

## Run order
E1 (parallel with M1 direction extraction) → M1 → M2 (grid sweep) → M3 → M4.

## Notes for the experiment stage / /mechanism-skills routing
- Bind the concrete mechanism family at Phase 1.5 (SAE-feature clamp on `Goodfire/Evo-2-Layer-26-Mixed` vs contrastive steering vector vs linear-probe direction; may combine). Re-bind `method_sensitive` fields (`sites`, `metric`, `n_pairs`, `gpu_hours`) at routing without counting as a plan rewrite; record deltas in `MECHANISM_ROUTING.md`.
- No M0 gate exists (behavior_source=given); do NOT add `depends_on:[M0]` anywhere.
- Keep Evo2-7B fixed for all generation/intervention milestones (HARD). `evo2` install + HuggingFace download (with proxy/token) happen in the experiment stage.
- Resolve exact sample sizes to satisfy statistical power (n/condition, #contrastive pairs) when the family is bound; honor task/data-rule floors.
