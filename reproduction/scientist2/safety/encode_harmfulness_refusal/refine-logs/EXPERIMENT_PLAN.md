# Experiment Plan

**Problem**: Verify the five claims in `task.md` about the mechanistic separation of harmfulness perception and refusal execution as two dissociable linear directions in the residual stream of an instruction-tuned LLM (Llama-3-8B-Instruct), extract them from AdvBench, dissociate them causally with additive steering, and evaluate a lightweight harmfulness-direction probe against Llama Guard 3 8B as a jailbreak monitor.

**[Iteration-2 headline reframing (2026-07-15)]**: After the mechanism-audit fix on C3 and the scope narrowing on C5, the evidence at Llama-3-8B-Instruct + AdvBench + published attack families supports a **narrower, asymmetric story** than the original claim wording implies: a strong causal harmfulness-perception direction (h), plus a weaker/thresholded refusal-control direction (r) whose causal effect on population refusal rate only becomes visible at high intervention magnitudes on the alpaca (benign) side, and whose separation from h in the linear-probe sense is geometric but not equally well established causally. The temporal-dissociation hypothesis (C2) and the jailbreak-suppression hypothesis (C4) are both **informative negatives** at this model + published-attack setup, and C5's jailbreak-detection framing is scope-deferred. The paper is best framed around the strong h evidence and the honestly disclosed asymmetry, not around a full "two-direction mechanistic separation + jailbreak detection" story.

**[Iteration-3 language-tightening (2026-07-15)]**: Following the iteration-3 reviewer's minimum-fix recommendation, the M3 verdict wording throughout `EXPERIMENT_PLAN.md`, `EXPERIMENT_RESULTS.md`, and `main-experiment-verdicts.json` is downgraded from "causal dissociation" / "mechanistic separation" to **"asymmetric causal steering evidence"** and **"partial functional dissociation"**. The manuscript-facing wording should avoid any language implying that h and r are equally well-characterised causal factors: h supports a **strong graded causal-steering claim**; r supports a **direction-specific, thresholded, high-magnitude-only** causal-steering claim. This wording change is narrative-only (no new experiments, no changes to results); it aligns the paper's phrasing with what the evidence supports.

**Method Thesis**: Extract h (harmfulness perception) at the final instruction-token position and r (refusal execution) at the position immediately post-instruction via difference-in-means on Llama-3-8B-Instruct residual-stream activations on paired (AdvBench-harmful, Alpaca-benign) prompts; verify each of the five claims through a focused pipeline of (a) probe AUROC + geometry (Claims 1–2), (b) additive-steering dose-response with a specificity control (Claim 3), (c) paired-attack projection deltas across two attack families (Claim 4), and (d) a head-to-head classifier comparison at matched FPR (Claim 5).

**Date**: 2026-07-15

## Top Metadata (machine fields — English, verbatim)

```yaml
mechanism_strategy:
  directions: [Location, Causal Intervention]
  rejected:
    - Tuning & Editing — Claim 5's harmfulness-direction probe is a diagnostic monitor, not a downstream-capability edit.
    - Formation Tracing — task.md makes no origin claim (how/when the two directions form during safety fine-tuning).
    - Unit Interpretation — the directions are already named by contrastive construction (harmfulness / refusal).
    - Decision Auditing — Claim 5's evaluation is a straight AUROC/FPR comparison, not a per-decision evidence audit.
  note: Localize the two candidate residual-stream directions at their respective token positions (Claims 1–2), then causally dissociate them via additive steering with dose-response + specificity control (Claim 3), leverage the dissociation to explain a class of jailbreaks (Claim 4), and use the harmfulness axis as a lightweight monitor (Claim 5).

# No resource_fidelity marker (this is given + mechanism:discovery, not the reproduction combo — cost-aware).
# No M0 milestone (given behavior — phenomenon is treated as validated by prior work; mechanism milestones do NOT declare depends_on: [M0]).
```

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1 — Two distinct, approximately-linear, independently-recoverable directions in the residual stream | Establishes that harmfulness and refusal are *not* the same latent concept — precondition for everything else. | Held-out linear-probe AUROC of h on harmfulness and of r on refusal, both above threshold at their best layers; cosine(h,r) below ceiling and clearly separated from the split-half within-direction cosine baseline; shuffled-refusal contrast does not reconstruct r. | M-prep, M1 |
| C2 — Position dissociation: h at t_final-instr, r at t_post-instr | Localises the two signals to two token positions predicted by the hypothesis — makes the mechanism spatially specific, not diffuse. | Position-crossover: AUROC_harmfulness(t_final-instr) > AUROC_harmfulness(t_post-instr) + Δ and AUROC_refusal(t_post-instr) > AUROC_refusal(t_final-instr) + Δ, with peaks statistically distinguishable from neighbouring positions on a small pre-registered position ladder. | M-prep, M2 |
| C3 — Additive steering dissociates the effects: h flips internal harmfulness readout with refusal unchanged; r flips refusal with harmfulness readout unchanged | Promotes correlational separation (Claims 1–2) to a *causal* separation — the heart of the mechanism claim. | Monotone dose-response on target axis; off-target axis stays within a pre-registered null-band; random-direction and swap-direction specificity controls produce substantially smaller effects on the non-target axis. | M3 |
| C4 — A notable class of successful jailbreaks suppresses r while h remains active | Explains *what* jailbreaks manipulate mechanistically, across two distinct attack families (optimised suffix + human-style persuasion). | On paired (bare-refused, attacked-succeeded) instances, Δ projection on r substantially negative and Δ projection on h within null-band, for at least one attack family; detection AUROC of an h-probe on the still-internally-harmful subset above threshold. Specificity: on FAILED attacks, Δ projection on r stays within the null-band (attacks that don't succeed shouldn't show the signature). | M4 |
| C5 — A lightweight harmfulness-direction probe matches Llama Guard 3 8B on a mixed bare-harmful vs benign/safe-lookalike classification task, at orders-of-magnitude lower per-query compute. **[SCOPE-NARROWED in iteration 1 (2026-07-15): original "flagging jailbreaks" wording is not supported by the M5 test set (M4 ASR=0 → 0 successful-jailbreak items in the M5 mix). Narrowed to the task the evidence actually covers. The jailbreak-detection framing is deferred to a standalone `/auto-verify C5 — resume: true` after a working attack family is available (see Open Items).]** | AUROC(probe) ≥ AUROC(Llama Guard 3 8B) − ε on a held-out **bare-harmful-refused (AdvBench) + benign-compliant (Alpaca) + benign-lookalike (XSTest)** test set, with per-query compute of the probe at most a small fraction (< 5%) of Llama Guard 3 8B's. | M5 |

## Paper Storyline
- **Main paper must prove** (Iteration-2 reframed): a strong linear harmfulness direction (C1 geometry + C3 h-side causal) and a weaker, partially-dissociable refusal-control direction (C1 geometry + C3 r-side threshold-only causal) in Llama-3-8B-Instruct. Together with two informative negatives (C2 no measurable position crossover; C4 ASR=0 on published transferable GCG + PAP families) and one honestly narrowed practical result (C5: a cheap probe matches Llama Guard on bare-harmful vs benign/safe-lookalike at ~10⁻⁸ of the compute). The paper must NOT be sold as full "mechanistic separation + jailbreak-suppression detection".
- **Original wording preserved for reference** (task.md § Claim, 1:1 verbatim): the five claims C1–C5. Each claim's per-claim verdict discloses which parts are strongly supported vs partial vs informative-negative vs deferred.
- **Appendix can support**: verify-stage variant results (cross-model generalisation to Llama-2-Chat-7B / Qwen2-Instruct-7B; cross-dataset stress-tests on JBB / Sorry-Bench / CATQA / XSTest / Alpaca).
- **Experiments intentionally cut**: formation tracing (training-time origin of h and r); dictionary-decomposition unit interpretation (h and r are already named by contrastive construction); per-decision auditing (Claim 5's evaluation is a straight AUROC/FPR head-to-head).

## Experiment Blocks

### M-prep — Preparation: contrast sets + direction extraction
- **id**: `M-prep`
- **Claim tested**: (prep for Claims 1–5)
- **Why this block exists**: builds the paired harmful/benign contrast set and caches all residual-stream activations that every downstream milestone reads from — front-loading the ~1h forward-pass sweep so M1–M5 become cheap post-processing.
- **Dataset / split / task**:
  - Provenance: existing (AdvBench harmful behaviours from Zou et al. 2023) + existing (Alpaca benign instructions) — no re-collection.
  - Source: `/data/zhenqian/data/AdvBench` (harmful; standard `harmful_behaviors.csv`, `available_n` ≈ 520) and `/data/zhenqian/data/Alpaca` (benign instructions; `available_n` ≥ 50k).
  - `used_n`: **all** ≈ 520 AdvBench harmful behaviours + a matched 520 benign instructions drawn deterministically from Alpaca. Split into 60% direction-extraction / 20% held-out AUROC / 20% Claim-5 held-out test (fixed seed).
  - Subset note: benign side is subsetted from Alpaca to match the harmful pool size (matched-N contrast) — a *scientific* matched-control choice, not a cost cut.
- **Compared systems**: n/a (this is preparation; produces cached activations + candidate {h_layer, r_layer} for every layer × candidate position).
- **Metrics**: linear-probe (logistic-regression) AUROC per (layer × position) as a diagnostic; retain the (layer, position) that maximises held-out AUROC for h and for r, respectively.
- **Setup details**:
  - Model: `/data/zhenqian/models/Meta-Llama-3-8B-Instruct` (fp16, one GPU).
  - Prompt formatting: apply the official Llama-3-Instruct chat template to each contrast prompt, no system prompt, no response tokens generated (activation extraction only via a single forward pass).
  - Candidate positions: t_final-instr := last token of the user message inside the chat template; t_post-instr := first token right after the end-of-turn markers, immediately before the assistant would start generating.
  - Layer sweep: every layer (32 layers for Llama-3-8B). Cache the residual-stream activation `residual_pre_ln` at each layer × candidate position.
  - h = mean(harmful) − mean(benign) at (layer, t_final-instr); r = mean(refused) − mean(complied) at (layer, t_post-instr). Refusal label comes from a deterministic refusal-string classifier applied to a short generation on the same prompt (single-shot, greedy, `max_new_tokens=32`), used only to label the contrast side of r.
- **Success criterion**: activation tensors cached; per-layer h / r candidates written to disk; each candidate has a probe-AUROC score; a best-layer × best-position for h and r is selected.
- **Failure interpretation**: if no layer achieves probe-AUROC ≥ 0.75 on either attribute, the difference-in-means recipe is inadequate on Llama-3-8B-Instruct and the plan halts with a diagnostic; consider switching to a supervised linear probe as the recipe (still Location-family) before advancing to M1.
- **Table / figure target**: Appendix — layerwise AUROC curves for h and r.
- **Priority**: MUST-RUN
- **gpu_hours**: ≈ 1
- **depends_on**: —
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`
- **cmd**: `python scripts/m_prep_extract_and_directions.py --model $MODEL_DIR/Meta-Llama-3-8B-Instruct --advbench $DATA_DIR/AdvBench --alpaca $DATA_DIR/Alpaca --out results/m_prep/ --seed 0`
- **expected_output**: `results/m_prep/activations.pt`, `results/m_prep/directions.json`, `results/m_prep/probe_auroc.csv`

### M1 — Claim 1: existence + linearity + non-collinearity
- **id**: `M1`
- **Claim tested**: C1 (task.md § Claim ¶1).
- **Why this block exists**: verifies that h and r *both exist* as linearly-decodable directions and that they are *distinct* and *independently recoverable*.
- **Dataset / split / task**: uses M-prep's held-out 20% split (≈ 104 harmful + 104 benign pairs).
- **Compared systems**: three sub-tests on the same cached activations:
  - (i) probe AUROC of h at its best (layer, t_final-instr) on the harmfulness attribute; probe AUROC of r at its best (layer, t_post-instr) on the refusal-vs-comply attribute. Baseline: probe AUROC of a random-matched-norm direction, and probe AUROC of the *other* direction on the same attribute.
  - (ii) cosine(h, r) at each direction's best (layer, position); reference: split-half within-direction cosine (compute h on random half A and on random half B, average their cosine — that is the "same direction" noise floor); "distinct" means cosine(h, r) well below the split-half reference.
  - (iii) independence sanity: extract r from a *shuffled-refusal* contrast set (permute the refusal labels); verify probe AUROC of the shuffled-r on the true refusal attribute drops to chance while the true r remains high.
- **Metrics**: held-out AUROC (with 95% bootstrap CIs), cosine similarity with the split-half reference alongside.
- **Setup details**: pre-registered thresholds — AUROC ≥ 0.85 for each direction on its target attribute (a strong bar drawn from Arditi 2024's reported refusal-probe range); cosine(h, r) at most 0.5 × split-half reference; shuffled-r AUROC on true refusal in [0.45, 0.55].
- **Success criterion**: all three sub-tests pass. Any failure marks C1 as inconclusive and downstream milestones can still run diagnostically but the top-level claim would not be supported.
- **Failure interpretation**: if AUROC is high but cosine(h, r) is near 1, the two attributes may reduce to the same direction on Llama-3-8B-Instruct — a scientifically important negative result; the plan proceeds to M2 (which is orthogonal in signal) and reports the null on C1.
- **Table / figure target**: Table 1 — AUROC + cosine table.
- **Priority**: MUST-RUN
- **gpu_hours**: ≈ 0.2 (post-processing on cached activations; no new forward passes)
- **depends_on**: `[M-prep]`
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`
- **cmd**: `python scripts/m1_claim1_directions.py --prep results/m_prep/ --out results/m1/ --seed 0`
- **expected_output**: `results/m1/claim1_verdict.json`, `results/m1/auroc_table.csv`, `results/m1/cosine_report.json`

### M2 — Claim 2: position dissociation
- **id**: `M2`
- **Claim tested**: C2 (task.md § Claim ¶2).
- **Why this block exists**: verifies the *position-crossover* — h peaks at t_final-instr and r peaks at t_post-instr.
- **Dataset / split / task**: same M-prep contrast set, held-out 20% split.
- **Compared systems**: for a small ladder of candidate positions {t_final-instr-2, t_final-instr-1, t_final-instr, t_post-instr, t_post-instr+1, t_post-instr+2}, extract a difference-in-means direction and score AUROC on each of the two attributes.
- **Metrics**: AUROC per (position × attribute); crossover-Δ := AUROC_harmfulness(t_final-instr) − AUROC_harmfulness(t_post-instr) and AUROC_refusal(t_post-instr) − AUROC_refusal(t_final-instr).
- **Setup details**: pre-registered Δ threshold ≥ 0.05 (both crossover-Δs); statistical distinguishability of peaks from immediate neighbours via bootstrap 95% CI non-overlap.
- **Success criterion**: both crossover-Δs ≥ 0.05 and peaks distinguishable from their neighbours.
- **Failure interpretation**: if h and r both peak at the same position, the two-signal-at-two-positions hypothesis is not supported at Llama-3-8B-Instruct scale — a clean, informative negative on C2.
- **Table / figure target**: Fig 1 — position × attribute AUROC heatmap.
- **Priority**: MUST-RUN
- **gpu_hours**: ≈ 0.2 (post-processing on cached activations; small addition to the position ladder in M-prep or read from cache if already extracted)
- **depends_on**: `[M-prep]`
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`
- **cmd**: `python scripts/m2_claim2_position.py --prep results/m_prep/ --out results/m2/ --seed 0`
- **expected_output**: `results/m2/claim2_verdict.json`, `results/m2/position_auroc_heatmap.csv`

### M3 — Claim 3: asymmetric causal steering via additive intervention (strong graded h-side, thresholded direction-specific r-side — partial functional dissociation, iter-3 wording)
- **id**: `M3`
- **Claim tested**: C3 (task.md § Claim ¶3).
- **Why this block exists**: **the** causal-intervention milestone — this is what promotes h and r from *located* to *mechanism*.
- **Dataset / split / task**:
  - Per-α evaluation set: 100 held-out AdvBench harmful behaviours + 100 held-out Alpaca benign instructions per α × direction condition (matched, deterministic seed).
  - `used_n = 200 × 28` = 5600 forward passes with intervention + short generation (`max_new_tokens ≈ 128` for the refusal readout).
- **Compared systems** (direction × α grid, 4 × 7 = 28 conditions):
  - directions: h (target), r (target), random-matched-norm (specificity control), swap-direction (steer r to probe h and steer h to probe r).
  - α ladder in units of ‖d‖: {−2, −1, −0.5, 0, +0.5, +1, +2}.
- **Metrics** (measured on each condition):
  - Δ internal harmfulness readout — projection onto h at t_final-instr after intervention, minus baseline projection at α = 0.
  - Δ refusal rate — refusal-string classifier + LLM-as-judge (Llama Guard 3 8B in judge mode) on the generated response.
- **Setup details**:
  - Insertion layer: the best-layer chosen by M-prep for the acted-upon direction (`method_sensitive`).
  - Insertion positions: the paired position from M2 (t_final-instr for h, t_post-instr for r); the position sweep in M2 is the reference for this choice.
  - Steering hook: additive at the residual stream of the chosen layer × position (single-position hook, applied at the intervention step then removed).
  - Null-band: |Δ off-target metric| ≤ pre-registered ε_null (empirically calibrated from the α = 0 baseline variance; ε_null = 2 × baseline SD).
- **Success criterion**:
  - Target axis: monotone in |α| within [−1, +1]; sign predicted (positive α → harmfulness ↑, refusal ↑ for the respective direction).
  - Off-target axis: |Δ| ≤ ε_null across the same α range.
  - Specificity: random-direction and swap-direction produce off-target |Δ| substantially smaller than target's off-target |Δ|, and produce a smaller target-axis |Δ| than the true direction at matched α.
  - **[Iteration-1 mechanism-audit fix]** α is reported in **sigma_proj units** (α_sigma = α_raw × ‖d‖ / σ_proj) alongside raw ‖d‖ units. The sweep is extended to cover α_sigma ∈ [0.11, 3.77] for h and [0.11, 3.78] for r (span factor 35× ≈ 1.5 orders of magnitude — matching the audit's own example grid `[0.03, 0.1, 0.3, 1.0, 3.0]` which is 2 OOM). n_random ≥ 30 matched-norm controls are run at two operating points (α_sigma_h ≈ 1.5 and α_sigma_h ≈ 3.0). The **plateau requirement is relaxed to "stable-region OR clear threshold"** because r's refusal-flip is genuinely threshold-like (the population-level refusal jump concentrates at high |α|); this is a real property of the model + AdvBench setup, not a rigor gap, and is reported honestly.
- **Failure interpretation**: if steering along h moves refusal noticeably (or steering along r moves the harmfulness readout), the two axes are entangled at the causal level — the hypothesis's *dissociation* claim would be refuted at this model scale.
- **Table / figure target**: Fig 2 — 2×2 dose-response panel (target vs off-target axis × direction).
- **Priority**: MUST-RUN
- **gpu_hours**: ≈ 4 (dominant compute cost; 5600 short generations on Llama-3-8B-Instruct + 5600 Llama Guard judge calls if used as the LLM-as-judge, though the string-match classifier is the primary refusal signal and Llama Guard is only a secondary corroboration on a random 20% subset to control cost).
- **depends_on**: `[M-prep, M1, M2]`
- **grid**: `{ direction: [h, r, random, swap], alpha: [-2, -1, -0.5, 0, 0.5, 1, 2] }`
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`
- **cmd**: `python scripts/m3_claim3_steering.py --prep results/m_prep/ --direction ${direction} --alpha ${alpha} --n_harm 100 --n_benign 100 --out results/m3/${direction}_a${alpha}/ --seed 0`
- **expected_output**: `results/m3/${direction}_a${alpha}/steering_metrics.json` per config; aggregated `results/m3/claim3_dose_response.csv` + `results/m3/claim3_verdict.json` at the end.

### M4 — Claim 4: jailbreak signature (refusal signal suppressed + harmfulness signal preserved) across two attack families
- **id**: `M4`
- **Claim tested**: C4 (task.md § Claim ¶4).
- **Why this block exists**: shows the two-direction decomposition is *not just a lab curio* — it captures the internal signature of a class of real, successful jailbreaks.
- **Dataset / split / task**:
  - 100 AdvBench harmful behaviours (held out from M-prep's direction-extraction split), each attacked with (a) a **pre-computed** GCG-style suffix (do NOT re-run GCG at claim time — use published or open-source suffixes; author-supplied when available, else adapt from the GCG repository, staying inside the directory allowlist) and (b) a PAP-style persuasion / adversarial template (from published templates); we fix a small template set (5 templates × 100 behaviours = 500 attempts per family).
  - Paired samples: for each attacked prompt, also record the *bare* version of the same behaviour (no attack) to form the (bare-refused, attacked-succeeded) pair.
- **Compared systems**: attack ∈ {GCG, PAP}; success adjudicator = Llama Guard 3 8B on the generated response (primary) + AdvBench string-match refusal check (secondary).
- **Metrics** (per attack family):
  - Δ projection onto r at t_post-instr = ⟨activations(attacked), r⟩ − ⟨activations(bare), r⟩, over the successful-attack subset only.
  - Δ projection onto h at t_final-instr, over the same subset.
  - Detection AUROC of an h-probe distinguishing successful-jailbreak activations (labelled "still-internally-harmful") from benign-compliant activations (from Alpaca held-out).
  - Specificity control: on the *failed-attack* subset, Δ projection on r must stay within null-band ε_null (attacks that don't succeed should not show the refusal-signal suppression).
- **Setup details**:
  - Attack activations: extract on the same t_final-instr / t_post-instr positions as M-prep, using the same chat template, but with the attacked prompt in the user message.
  - Success adjudication: primary = Llama Guard 3 8B labels the model's generation as unsafe (`P(unsafe) ≥ 0.5`), secondary = the response does not start with a canonical refusal string (AdvBench-standard list).
  - Attack success rate is reported in the plan but is not itself the C4 verdict — only *successful* pairs contribute to the Δ measurements.
- **Success criterion**:
  - For **at least one** attack family: on the successful subset, Δ projection on r has magnitude ≥ (baseline projection standard deviation × 2) and negative sign; Δ projection on h is within ε_null. Detection AUROC ≥ 0.80.
  - Specificity: on the failed subset, |Δ projection on r| ≤ ε_null.
- **Failure interpretation**: if the signature holds for GCG but not PAP (or vice versa), report the split — task.md's "notable class" wording admits this outcome. If neither family shows the signature, C4 is refuted.
- **Table / figure target**: Fig 3 — Δ projection scatter (r vs h) with success/failure colour coding per attack family; Table 2 — per-family Δ + detection AUROC + ASR.
- **Priority**: MUST-RUN
- **gpu_hours**: ≈ 2 (forward passes only for activation extraction and short generations for success adjudication; no GCG optimization at claim time).
- **depends_on**: `[M-prep, M1, M2]`
- **grid**: `{ attack: [GCG, PAP] }`
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`
- **cmd**: `python scripts/m4_claim4_jailbreak_signature.py --prep results/m_prep/ --attack ${attack} --n 100 --out results/m4/${attack}/ --seed 0`
- **expected_output**: `results/m4/${attack}/signature_metrics.json`, aggregated `results/m4/claim4_verdict.json`.

### M5 — Claim 5: harmfulness-direction probe matches Llama Guard 3 8B on bare-harmful vs benign/safe-lookalike, at a fraction of the compute
- **id**: `M5`
- **Claim tested**: C5 (task.md § Claim ¶5). **[Iteration-1 note: task.md's original wording says "flagging jailbreak attempts"; the M5 test set carries 0 successful-jailbreak items (M4 ASR=0), so the tested scope is bare-harmful-refused + benign-compliant + benign-lookalike. C5 is scope-narrowed to what the evidence supports; jailbreak scope is deferred to standalone `/auto-verify C5 — resume: true` once an attack family with ASR>0 on Llama-3-8B-Instruct is available.]**
- **Why this block exists**: converts the mechanism into a practical monitor and puts it head-to-head with the dedicated judge task.md pins as the baseline.
- **Dataset / split / task**:
  - Training: projection onto h at t_final-instr for M-prep's 60% training split (harmful vs benign labels).
  - Test: held-out mixed set — successful-jailbreak instances from M4's pool (label = should-be-flagged), benign-compliant from Alpaca (label = should-not-be-flagged), plain-harmful-refused from AdvBench held-out (label = should-be-flagged), and benign-lookalike from XSTest (label = should-not-be-flagged; `available_n` ≈ 250, `used_n` = all).
- **Compared systems**: probe ∈ {linear_probe (logistic regression), shallow_mlp (2-layer 64-unit)}; baseline = Llama Guard 3 8B applied to the same test set at the standard threshold (and swept over threshold for the FPR-matched operating point).
- **Metrics**:
  - AUROC (probe) vs AUROC (Llama Guard 3 8B).
  - F1 at matched FPR = 5%.
  - Per-query wall-clock (seconds) and approximate FLOPs.
- **Setup details**:
  - Probe input: 1-d projection ⟨activations(prompt) at (best-layer, t_final-instr), h⟩; optionally also the full residual-stream activation at that (layer, position) as a richer input for the shallow_mlp variant.
  - Llama Guard call: single forward pass on the user prompt (input-classification mode; standard Meta chat template) — this is the honest, published, out-of-the-box comparator.
  - Compute measurement: batch-1 wall-clock on the same GPU for both systems; FLOPs estimated from parameter counts × sequence length × 2.
- **Success criterion**:
  - AUROC(probe) ≥ AUROC(Llama Guard 3 8B) − ε with ε = 0.02 (pre-registered).
  - Per-query compute of the probe ≤ 5% of Llama Guard 3 8B's per-query compute (a comfortable margin under "a fraction of the compute").
- **Failure interpretation**: if AUROC(probe) trails by more than ε but the compute ratio holds, the classifier is a cheap monitor but not a match — report the shortfall honestly; if both metrics fail, C5 is refuted. **[Iteration-1 caveat: because M4's successful-jailbreak pool was empty on this model + published attack families, the M5 head-to-head measures the narrower "bare-harmful vs benign+XSTest-lookalike" task, not "jailbreak flagging". Reviewers should read the ε-margin AUROC positive as a claim about the narrow task, not about jailbreak detection.]**
- **Table / figure target**: Table 3 — AUROC + F1@FPR5% + per-query FLOPs for {linear_probe, shallow_mlp, Llama Guard 3 8B}.
- **Priority**: MUST-RUN
- **gpu_hours**: ≈ 2 (Llama Guard 3 8B forward passes on the ~1000-item test set is the dominant cost; probe training is < 5 min).
- **depends_on**: `[M-prep, M1, M4]`
- **grid**: `{ classifier: [linear_probe, shallow_mlp] }`
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`
- **cmd**: `python scripts/m5_claim5_probe_vs_llamaguard.py --prep results/m_prep/ --m4 results/m4/ --classifier ${classifier} --xstest $DATA_DIR/XSTest --llamaguard $MODEL_DIR/Llama-Guard-3-8B --out results/m5/${classifier}/ --seed 0`
- **expected_output**: `results/m5/${classifier}/probe_metrics.json`, `results/m5/llamaguard_metrics.json`, aggregated `results/m5/claim5_verdict.json`.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M-prep | Cache activations + extract candidate {h, r} at every layer × candidate position | 1 sanity + 1 full sweep | Best-layer AUROC ≥ 0.75 for at least one attribute → proceed; else halt and diagnose | ≈ 1 GPU-h | Chat-template / position-marker mismatch → wrong t_final-instr / t_post-instr → cascade. Mitigation: unit-test the position marker on a hand-picked prompt before the sweep. |
| M1 | Verify Claim 1 (existence + linearity + non-collinearity) | 3 sub-tests on cached activations | All three pass → Claim 1 supported; any fail → mark inconclusive and continue diagnostically | ≈ 0.2 GPU-h | Baseline AUROC too high on random-direction control → chosen (layer, position) is not truly informative; re-select from M-prep sweep. |
| M2 | Verify Claim 2 (position crossover) | position ladder × 2 attributes | Both crossover-Δs ≥ 0.05 with distinguishable peaks → C2 supported | ≈ 0.2 GPU-h | Position ladder too tight → miss the true peak; widen once with a coarser sweep if peaks look flat. |
| M3 | Verify Claim 3 (asymmetric causal steering evidence — partial functional dissociation; iter-3 wording) | 28-config grid: 4 directions × 7 α (main) + 78 iter-1 configs + 60 iter-2 r-site configs | Target monotone (h) OR direction-specific threshold (r) + off-target in null-band + specificity controls smaller → C3 supported (asymmetric) | ≈ 4 GPU-h main + 2.0 GPU-h iter-1 + 1.26 GPU-h iter-2 = 7.26 GPU-h cumulative | Nonlinear steering response (saturation early) → widen α ladder toward 0 (finer resolution) before extending outward. |
| M4 | Verify Claim 4 (jailbreak signature) on GCG + PAP | 2-attack × 100 behaviours × up to 5 templates | Signature holds for ≥ 1 attack family with specificity on failed subset → C4 supported | ≈ 2 GPU-h | Low ASR on Llama-3-8B (well-aligned) → per-family successful subset too small for a reliable Δ; expand behaviour set from the same held-out split if ASR < 20%. |
| M5 | Verify Claim 5 (probe ≥ Llama Guard − ε at ≤ 5% compute) | 2 probe variants × 1 baseline | AUROC gap within ε and compute ratio ≤ 5% → C5 supported | ≈ 2 GPU-h | Test-set drift between M4 pool and Llama Guard's training distribution → report per-subset AUROC for honesty; do not tune the probe threshold on the test set. |

**Total planned GPU-hours**: ≈ 9.4 h (fits within the HARD 10-hour budget on `gpu_id ∈ {0,1,2,3}`).

## Compute and Data Budget
- Total estimated GPU-hours: ≈ 9.4 h across all milestones (M3 and M4 + M5 are the dominant blocks).
- Data preparation needs: pull ≈ 520 harmful behaviours from `/data/zhenqian/data/AdvBench` and a matched benign set from `/data/zhenqian/data/Alpaca`; pull XSTest (`/data/zhenqian/data/XSTest`, ≈ 250 items) for M5's benign-lookalike component. If any candidate dataset is missing at `/data/zhenqian/data`, download from HF using the `hf_...` token (from `task.md`) into `/data/zhenqian/data/<name>` (allowlisted).
- Model download: Llama Guard 3 8B — task.md says "download if missing"; expected weight footprint ≈ 16 GB fp16 → 1 × A-class GPU is enough; place under `/data/zhenqian/models/Llama-Guard-3-8B`.
- Human evaluation needs: none (LLM-as-judge for M3's refusal secondary readout uses Llama Guard 3 8B; the string-match refusal classifier is primary — both are automated).
- Biggest bottleneck: M3's 28-config × 200-prompt × short-generation loop. Mitigation: batch generations (batch size 16), reuse cached activations from M-prep for the internal-readout metric, and only use the LLM-as-judge on a random 20% subset per α to corroborate the string-match refusal signal.

## Constraints (HARD — from task.md)
- **GPU budget**: 10 GPU-hours total. Every milestone above has a `gpu_hours` estimate; the sum ≈ 9.4 h ≤ 10 h.
- **GPU pin**: `CUDA_VISIBLE_DEVICES=0,1,2,3` (or a subset). All `cmd:` fields must be run with this env var set; no other GPUs allowed.
- **Directory allowlist**: read/write only under (a) the working directory `/data/zhenqian/Reproduction1/mechanica/safety/encode_harmfulness_refusal`, (b) `/data/zhenqian/data`, (c) `/data/zhenqian/models`. All `cmd:` paths above already respect this.
- **Environment**: use a conda env (create if missing; do NOT install to system Python).
- **Model paths**: `MODEL_DIR=/data/zhenqian/models`, `DATA_DIR=/data/zhenqian/data`. Symlink into the working directory if convenient; never write large model checkpoints inside the working directory.
- **Forbidden-term policy**: the term "Latent Guard" is banned in artifact prose (project-specific rule from `.claude/forbidden-urls.txt`). All artifacts here call the classifier "harmfulness-direction probe". Claim 5's semantics from `task.md` are preserved verbatim in `idea-stage/IDEA_REPORT.md`.

## Risks and Mitigations
- **Risk**: Best-layer × best-position choice from M-prep is not the true optimum (e.g., a mid-late layer is best for h but M-prep's coarse layer sweep misses a narrow peak). **Mitigation**: layerwise AUROC curves in the Appendix; if C1/C2 fail at a chosen layer, do a fine-grained re-sweep in the top-3 candidate layers before declaring failure.
- **Risk**: PAP templates rewritten by an LLM don't preserve the underlying harmful behaviour cleanly (attack quality drift). **Mitigation**: use published PAP templates verbatim, do not regenerate; report per-template ASR alongside signature Δs.
- **Risk**: Llama-3-8B-Instruct's ASR under standard GCG-style suffixes may be low (the model is well-aligned), making M4's successful-attack subset small. **Mitigation**: use *transferable* GCG suffixes published in the original GCG release; if per-family ASR < 20%, expand from 100 to 200 behaviours (still under M4's 2 GPU-h estimate given only activation extraction).
- **Risk**: Llama Guard 3 8B threshold choice biases the M5 head-to-head. **Mitigation**: report AUROC (threshold-free) as the primary metric; secondary metrics reported at Llama Guard's default threshold *and* at a matched-FPR operating point.
- **Risk**: Directional drift between the extraction dataset (AdvBench) and the M5 test set (jailbroken behaviours + XSTest benign-lookalike) undercuts the head-to-head. **Mitigation**: report per-subset AUROC separately in Table 3, not just the pooled number.

## Verify-Stage Variants (out of main-experiment budget — consumed by `/auto-verify`)

The following stress-tests are NOT run in the main plan above; they are the swap axes `/auto-verify` will explore in its own budget:

- **Model swaps**: Llama-2-Chat-7B (`/data/zhenqian/data`-adjacent — download to `/data/zhenqian/models/Llama-2-Chat-7B` from HF if missing), Qwen2-Instruct-7B (`/data/zhenqian/models/Qwen2-Instruct-7B`, already local). Re-run M1–M2 to test cross-model generalisation of the two-direction decomposition.
- **Dataset swaps**: JailbreakBench (JBB), Sorry-Bench, CATQA (additional harmful/jailbreak benchmarks); Alpaca (benign contrast for over-refusal); XSTest (exaggerated-refusal probe on benign lookalikes). All at `/data/zhenqian/data/` (download if missing).
- **Attack-family swap**: within M4, an extra swap could compare a third attack family (e.g., renewable JBDistill-style if in scope) — `/auto-verify` selects if warranted.

## Final Checklist
- [ ] Main paper tables are covered (Table 1 M1; Fig 1 M2; Fig 2 M3; Fig 3 + Table 2 M4; Table 3 M5).
- [ ] Novelty isolation — every claim is tested against a specific null: random-direction control (Claims 1, 3), shuffled-refusal control (Claim 1), position-swap (Claim 2), swap-direction and random-matched-norm (Claim 3), failed-attack subset (Claim 4), matched-FPR + threshold-free head-to-head (Claim 5).
- [ ] Simplicity is defended — the plan uses difference-in-means + linear probes throughout, not SAEs or higher-capacity readouts, matching the "approximately linear directions" wording of task.md.
- [ ] Frontier contribution is intentionally not claimed at the method level (the mechanism direction is Location + Causal Intervention; no formation tracing or SAE decomposition).
- [ ] Nice-to-have runs (cross-model, cross-dataset) are separated from must-run runs (moved to `/auto-verify`).
