# Research Proposal: Unified Testing Protocol for Emotion-Specific Global Circuits in Llama-3.2-3B (final)

```yaml
mechanism_strategy:
  directions: [Location, "Causal Intervention", "Tuning & Editing"]   # execution order
  rejected:
    - Formation Tracing — task.md makes no training-time claim.
    - Unit Interpretation — task.md does not claim any component means a specific nameable concept.
    - Decision Auditing — task.md does not ask spurious-vs-legitimate; it asks reliable control.
  note: The three chosen directions map one-to-one onto Claims 1, 2, 3 — Location produces C_e, Causal Intervention validates its causal + stability status, Tuning & Editing uses it as the control knob beating prompting and single-direction steering.
chosen_mechanism: to-be-routed-by-mechanism-skills  # MECHANISM=discovery: family bound at /auto-experiment Phase 1.5, pre-eval-split freeze
# resource_fidelity NOT stamped — this is not the reproduction combo (MECHANISM=discovery).
family_freeze: pre-eval-split
```

## Problem Anchor

- **Bottom-line problem**: Verify three claims about emotion circuits in `Llama-3.2-3B-Instruct` on the SEV dataset (480 events × 6 emotions = 2880 pairs) under a single unified testing protocol:
  1. **Location**: a systematic framework (per-emotion direction extraction → per-emotion MLP-neuron + attention-head identification → global circuit integration) yields identifiable, sparse, per-emotion component sets `C_e`.
  2. **Causal Intervention + Stability**: ablating `C_e` weakens target emotion, enhancing `C_e` strengthens it (dose-response monotonic), off-target largely unaffected; `C_e` is stable across scenarios (Jaccard(C_e^{S1}, C_e^{S2}) > null) and structured across emotions (neuron-set overlap < head-set overlap — secondary).
  3. **Applied control**: intervening on `C_e` on held-out event stems induces the target emotion with higher emotion-expression accuracy than prompting (EmotionPrompt-style) AND single-direction steering (RepE/CAA-style, tuned per-emotion at best layer + best α), under a matched judge.
- **Must-solve bottleneck**: Fair, matched-budget verification with no free-parameter asymmetry, no cherry-picked layers, no unreliable judge, no under-specified operator, no direction-construction ambiguity for the steering baseline.
- **Non-goals**: no new mechanism-family method; no formation tracing; no SAE / unit interpretation; no spurious-feature audit; no fine-tuning; no emotions beyond the six SEV emotions.
- **Constraints**: Llama-3.2-3B-Instruct + SEV; Qwen2.5-7B-Instruct + SEV held-out for Verify swap; GPUs `{1,2,3,5,6}`; conda env; dir limits (work_dir + `/data/zhenqian/data` + `/data/zhenqian/models`); 10-GPU-hour envelope (budget explicitly *not* a constraint per task.md — pick strong methods); external judge `gpt-5.4` @ `dmxapi.cn` (proxy bypass).
- **Success condition**: For all three claims, the unified protocol yields per-emotion, pre-registered directional evidence with statistical CIs — Claim 1 sparse-and-stable `C_e`; Claim 2 causal + specific + scenario-stable (rubric: full / partial / causal-only / not-supported); Claim 3 circuit > prompting AND circuit > single-direction steering on ≥ 5/6 emotions with paired-bootstrap 95% CIs excluding 0; judge reliability gated.

## Technical Gap (final)

No prior work has run a *matched-budget, pre-registered* circuit vs. single-direction-steering vs. prompting comparison for emotion, on scenario-split SEV, under a single gated judge, with (i) a *causal* per-component ranker for `C_e`, (ii) three complementary specificity controls, and (iii) a fully frozen direction construction for the steering baseline. Prior emotion work either controls the input (EmotionPrompt) or collapses to a single residual-stream direction (RepE / CAA) at loosely-specified layer/α; head-level or neuron-level circuits have been established for truthfulness (ITI) and factual recall (ROME / Knowledge-Neurons / IOI / ACDC) but not for affect. This protocol is what closes the gap for the three given claims.

## Method Thesis

- **One-sentence thesis**: A single unified verification protocol runs the Location → Causal Intervention → Applied Control ladder on Llama-3.2-3B / SEV, with scenario-level held-out splits, a two-stage locator (cheap Stage-A shortlist → causal Stage-B ranker on prefix-logprob), a single pre-registered ablation operator (mean-substitute from same-stem off-target-emotion activations), a single enhancement operator (additive activation injection with scalar `α`), a hidden-target 6-way forced-choice judge (primary endpoint), a pre-eval-split-frozen single-direction steering baseline, and `N = 9` matched val-tuning evaluations per arm per emotion — putting all three claims on trial from one fit of `C_e` with directional predicates and bootstrap CIs.
- **Why smallest adequate**: one artifact (`C_e`), one dataset, one judge, one Stage-B score, one ablation operator, one enhancement operator, one primary endpoint; every extra piece is either a control (random-set null, targeted-`C_{e'}` control, length audit) or a report.
- **Why timely**: matched-budget circuit-vs-direction-steering-vs-prompting comparisons are the standardization the interpretability field has not yet delivered; running it on the clean SEV grid yields interpretable numbers.

## Contribution Focus

- **Dominant contribution**: a pre-registered matched-budget verification protocol running Location → Causal → Applied on one `C_e` fit, with pre-registered operators, judge gate, direction-construction lock, and specificity+stability controls — sufficient to substantiate or refute all three of task.md's claims.
- **Optional supporting contribution**: scenario-stability + cross-emotion overlap-structure analyses (Claim 2d + secondary Claim 2e) that turn "stable across scenarios and emotions" into directional predicates with CIs.
- **Explicit non-contributions**: no new mechanism family (family bound by `/mechanism-skills` pre-eval-split); no SAE; no formation tracing; no decision audit; no fine-tuning; no pairwise judge as primary; no zero-ablation as primary; no direction-shift enhancement as primary; no per-emotion post-hoc `k` tuning.

## Proposed Method (final)

### Complexity Budget
- **Frozen**: Llama-3.2-3B-Instruct weights (all inference-time, no training).
- **New trainable**: 0, unless the judge gate fails ⇒ 1 small SEV-emotion classifier as reliability backstop (not a contribution).
- **Intentionally excluded**: SAE; ACDC full-graph search; zero-ablation; direction-shift enhancement; pairwise judge as primary; RLHF / fine-tuning; formation tracing.

### Data Prep (Step 0)
- Parse `sev.json`; enumerate the six emotions from the emotion-variant field(s); build `(event_stem, target_emotion)` positive prompts and `(event_stem, off_target_emotion)` negative prompts for the paired contrasts.
- **Scenario-level split** of the 20 scenarios per domain: 10 scenarios → **train** (used for direction extraction and `C_e` fitting); 5 → **val** (used only for hyperparameter selection: `k_h`, `k_n`, `α`, layer L for Arm C, prompt templates for Arm B); 5 → **eval** (used only for held-out reporting of Claims 1/2/3). Same 10/5/5 split across all 8 domains — scenario disjointness asserted in code.
- **Gold subset for judge audit**: 60 event-stem × target-emotion pairs (10 per emotion), human-labeled by the researcher, drawn from the train fold (never eval).

### Step 1 — Location (Claim 1)

Location step is `method_sensitive: [n_pairs, sites, metric, gpu_hours]`. The submethod family (e.g., ITI-style head selection with alignment-scored MLP neurons, vs. knowledge-neuron-style causal effect scoring) is bound by `/mechanism-skills` **before any eval-split activation is read** (`family_freeze: pre-eval-split`).

- **Stage A — cheap filter (shortlist)**. For each layer `L` and each emotion `e`:
  - `d_{e,L} = mean_diff(residual_stream[last-event-token]_L, positive-e vs. off-target-uniform)` on the train fold.
  - Per-layer probe AUC (ITI-style, per-head linear probe on paired contrasts) ranks layers; keep **top-3 layers** by summed AUC over heads and neurons.
  - Within those 3 layers: shortlist **top 5%** of MLP neurons (by cosine `⟨w_n^{out}, d_{e,L}⟩` and by pre-activation contrast t-score, rank-combined) and **top 20%** of heads (by per-head probe AUC).
- **Stage B — causal ranker**. For each shortlisted component `c` (head or neuron), score
  `s_c = mean_{val stems} [ log P(prefix_e | event_stem; enhance c at α = α2) − log P(prefix_e | event_stem) ]`,
  where `prefix_e = "I feel {emotion_word_e}"` (natural-language emotion word for `e`) and `α2 = 1.0` (middle strength). Val here is 30 event stems per emotion drawn from the val fold. Judge-free, deterministic, cheap (one forward per component + one baseline forward per stem).
- **Global selection**. Rank all shortlisted heads by `s_c`, all shortlisted neurons by `s_c` separately. Take top-`k_h` heads and top-`k_n` neurons → `C_e = C_e^head ∪ C_e^neuron`.
- **Grid**: `k_h ∈ {24, 48, 96}` (~3.6% / 7.1% / 14.3% of 672 heads across 28 layers × 24 heads), `k_n ∈ {2000, 4000, 8000}` (~0.9% / 1.7% / 3.5% of 229 376 total MLP neurons across 28 layers × 8192).
- **`(k_h*, k_n*)` selection**: chosen ONCE globally, by macro-average target-prefix log-prob gain across all six emotions on val, at fixed `α = α2`. **Not** per-emotion.
- **Stability (Claim 1 predicate)**: fit `C_e` on 3 event-subsample folds of the train scenarios; report per-emotion mean pairwise Jaccard(fold_i, fold_j) vs. size-matched permutation null (200 draws).

### Step 2 — Causal Intervention + Stability (Claim 2)

All evaluations on the **held-out eval scenarios**. Same-stem batches per emotion.

- **Ablation operator (primary)**: mean-substitute each component in `C_e` with its per-neuron / per-head activation mean computed over the OTHER five emotion variants of the SAME event stem. If a stem lacks a variant, drop that stem from the ablation batch and log (no imputation). Report the number of dropped stems.
- **Enhancement operator (primary)**: additive activation injection at α ∈ {α1, α2, α3} = {0.5, 1.0, 2.0} on selected components — for a head, add `α · d_{e,L}^{h}` to the head's contribution to the residual stream; for a neuron, add `α · sign(⟨d_{e,L}, w_n^{out}⟩) · std(activation_n | pos-e on train)` to the pre-activation.
- **Metric**: per-emotion "target-emotion probability" is measured as the model's `log P(prefix_e | event_stem)` under each intervention (same score used in Stage B — internal, judge-free, deterministic).
- **Predicates (with rubric)**:
  - (a) **Causal-sign + dose-response** — REQUIRED. `mean_target Δ_ablation < 0` at α2; `mean_target Δ_enhance > 0` at α2; Spearman(α, Δ_enhance) ≥ 0.7 over 3 strengths.
  - (b) **Raw off-target specificity** — `mean|off-target Δ| < mean|target Δ|` at α2.
  - (c) **Targeted-`C_{e'}` specificity** — for each pair (e, e′), `mean|Δ_{intervene C_e on target e}| > mean|Δ_{intervene C_{e'} on target e}|` at α2, with paired-bootstrap CI on the difference.
  - (d) **Scenario stability** — fit `C_e^{S1}` on 5 of the 10 train scenarios (per domain), `C_e^{S2}` on the disjoint 5; report per-family Jaccard vs. size-matched permutation null; predicate: Jaccard(S1, S2) − permutation-null-mean > 0 at 95% CI.
  - (e) **Cross-emotion structure (SECONDARY)** — mean pairwise Jaccard over emotions for neuron sets vs. head sets; reported with bootstrap CI on the difference; does NOT gate Claim 2's support level.
- **Rubric**:
  - **Full support** iff (a) + (b) + (c) + (d) all pass.
  - **Partial** iff (a) + at least one of (b, c) hold but (d) fails.
  - **Causal-only** iff (a) holds but (b) and (c) both fail.
  - **Not supported** iff (a) fails.

### Step 3 — Applied Control (Claim 3)

Held-out event stems (no emotion suffix), each run under six target-emotion conditions. Three arms; identical decoding: `temperature=0`, `max_new_tokens=100`, no repetition penalty, deterministic. Matched budget `N = 9` val evaluations per arm per emotion.

- **Arm A (Circuit)**: activate `C_e` at test via the enhancement operator (same as Step 2) with strength `α_A ∈ {α1, α2, α3}`. Val budget: 3 × 3 = 9 choices (`(k_h, k_n)` from the fixed global `(k_h*, k_n*)` and 2 nearby cells for robustness × 3 α). Best (`k_h, k_n, α_A`) per emotion by val target-prefix logprob gain — but reported eval always uses the SAME `(k_h*, k_n*)` as Claims 1/2; only `α_A` may differ per emotion (documented in results table).
- **Arm B (Prompting — EmotionPrompt-style)**: append a fixed emotional-stimulus sentence per target emotion. Val budget: 3 templates × 3 injection positions (prefix / suffix / interleaved) = 9. Best per emotion.
  - Templates (starter set; final list frozen pre-val):
    - T1: "Answer as if you feel {emotion_word}. This is important."
    - T2: "You currently feel {emotion_word}. Respond accordingly."
    - T3: "This scenario made you feel {emotion_word}. Continue the story."
- **Arm C (Single-direction steering — RepE/CAA-style)**: PRE-EVAL-SPLIT-FROZEN direction construction — `d_e = mean_diff(residual_stream[last-event-token], positive-e vs. off-target-uniform)` on the **train fold**. Injection: additive with scalar `α_C` on the residual stream at the chosen layer `L`, immediately after that layer's output projection, added at **every token position after the event stem** during autoregressive generation (CAA convention). Val budget: `L ∈ {top-3 layers by probe AUC}` (SAME top-3 layers Arm A's Stage A shortlists) × `α_C ∈ {0.5, 1.0, 2.0}` = 9. Best per emotion.
- **Judge (primary)**: hidden-target 6-way forced choice — external LLM (gpt-5.4) sees `(event_stem, continuation)`, target is HIDDEN, returns one of the 6 emotion labels; per-continuation correctness = `predicted == target`. Fixed system prompt spelling out the 6 emotion labels; deterministic decoding.
- **Judge reliability**: 60-item gold subset (10 per emotion, human-labeled) → require agreement ≥ 0.75 AND per-emotion confusion matrix reported; judge-swap ablation on ≥ 10% of eval items with a locally trained SEV-emotion classifier as the second judge; if agreement gate fails ⇒ classifier is primary.
- **Metric (primary)**: per-emotion accuracy; macro-average; paired-bootstrap 95% CIs on (A − B) and (A − C) per emotion and on macro.
- **Length audit**: per-arm mean continuation length; if any pair differs > 15%, run a length-matched secondary analysis (truncate to shorter arm's mean length, re-judge). Length threshold is *reporting-only*, not decision.
- **Predicate**: A > B on ≥ 5/6 emotions AND A > C on ≥ 5/6 emotions, each with paired-bootstrap 95% CI excluding 0 at the emotion level.

### Modern Primitive Usage
- The frontier LLM (`gpt-5.4`) is used as the **hidden-target 6-way judge only**; not as proposer, planner, or reward model. Fallback path (SEV-emotion classifier) exists.
- The `/mechanism-skills` routing (Phase 1.5) plays the modern role of *pre-eval-split family freeze* — the concrete submethod (which MLP-scoring family, which head-probe family) is bound once and cannot be re-tuned during eval, closing a subtle DoF.

### Integration into Base Generator
- All interventions are inference-time hooks on Llama-3.2-3B-Instruct; no parameter change. Verify swap re-runs **only** the Claim-3 three-arm comparison on Qwen2.5-7B-Instruct + SEV held-out.

### Training Plan
- No training on the base model.
- Optional judge-classifier training only if the judge gate fails (AdamW lr 2e-5, batch 32, ≤ 5 epochs, sentence-encoder backbone; SEV emotion labels as supervision).

### Failure Modes and Diagnostics
- **F1 — judge unreliability**: classifier fallback.
- **F2 — steering baseline under-tuned**: matched `N=9` val budget + val heatmap reported.
- **F3 — prompting-baseline weakness**: 3-template × 3-position val + explicit external-control framing.
- **F4 — scenario leakage**: code assertion.
- **F5 — sparsity collapse**: fixed grid ceiling; Claim 1 reported partial if breached.
- **F6 — specificity failure**: three complementary specificity controls (random-set null, `C_{e'}`, off-target). Claim 2 partial/causal-only per rubric.
- **F7 — overlap-structure inversion**: Claim 2e is secondary; reported with CI, not gating.
- **F8 — length confound**: length audit + secondary length-matched analysis.

### Novelty and Elegance Argument
- Closest work: EmotionPrompt (Li 2023, arXiv:2307.11760), RepE (Zou 2023, arXiv:2310.01405), CAA (Rimsky 2023, arXiv:2312.06681), ITI (Li 2023, arXiv:2306.03341), ROME / KV-neuron / IOI / ACDC lines.
- Exact difference: none run a matched-budget, pre-registered, direction-construction-locked, judge-gated three-arm comparison for emotion on scenario-split SEV.
- Why focused: one artifact (`C_e`), one shared Stage-B score, one judge, one metric, deterministic predicates + rubric — nothing bolted on to inflate the paper.

## Claim-Driven Validation Sketch

- **Claim 1 (Location)** — sparsity floor met per emotion at fixed `(k_h*, k_n*)`; per-emotion mean pairwise Jaccard(fold, fold) > size-matched permutation-null 95% CI upper edge.
- **Claim 2 (Causal + Stability)** — predicates (a) required, (b) + (c) + (d) all reported with CIs; success rubric (full / partial / causal-only / not-supported).
- **Claim 3 (Applied)** — A > B on ≥ 5/6 emotions AND A > C on ≥ 5/6 emotions with paired-bootstrap 95% CIs excluding 0.

## Compute & Timeline

- Stage A cheap filter: ~0.5h.
- Stage B causal ranker (per-shortlisted-component prefix logprob at fixed α2): ~1.5h.
- Ablation + 3-α enhancement + 3 specificity controls: ~1.5h.
- Scenario stability + cross-emotion structure: ~0.5h.
- Claim 3 val (N=9 × 3 arms × 6 emotions × ~50 val stems, greedy) + eval (3 arms × 6 emotions × ~80 eval stems): ~2.0h.
- Judge audit + judge calls: ~0.5h (API-bound, GPU-idle).
- Verify swap (Claim 3 only) on Qwen: ~1.5h.
- **Total ~8.0 GPU-hours** on the primary model, comfortably inside the 10-hour envelope.

## Experiment Handoff Inputs
- **Must-prove claims**: three above.
- **Must-run ablations**: random-set null (Claim 1); random-set + `C_{e'}` + off-target (Claim 2); matched N=9 val grid for all three arms (Claim 3); prompt-template + injection-position ablation on Arm B; frozen direction-construction ablation report on Arm C.
- **Critical datasets / metrics**: SEV with scenario-level 10/5/5 split; target-prefix log-prob (internal, deterministic); hidden-target 6-way forced-choice accuracy (external judge with gate).
- **Highest-risk assumptions**: (R1) last-event-token direction is usable across all six emotions; (R2) judge reliability ≥ 0.75 gate; (R3) matched-budget val is fair; (R4) SEV emotion variants are the expected per-emotion suffix structure (checked at data prep).
