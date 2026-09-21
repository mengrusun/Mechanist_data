# Research Proposal: Unified Testing Approach for Emotion-Specific Global Circuits in Llama-3.2-3B

## Problem Anchor

- **Bottom-line problem**: Verify — under a single unified testing protocol — three claims about emotion circuits in `Llama-3.2-3B-Instruct` on the SEV dataset (480 events × 6 emotions = 2880 pairs):
  1. **Location**: a systematic framework (per-emotion direction extraction → per-emotion MLP-neuron + attention-head identification → global circuit integration) yields identifiable, sparse, per-emotion component sets `C_e`.
  2. **Causal Intervention + Stability**: ablating `C_e` weakens target emotion, enhancing `C_e` strengthens it (dose-response monotonic), off-target largely unaffected; `C_e` is stable across scenarios (Jaccard(C_e^{S1}, C_e^{S2}) > null) and structured across emotions (neuron-set overlap < head-set overlap).
  3. **Tuning & Editing (applied control)**: intervening on `C_e` on held-out event stems induces the target emotion with higher emotion-expression accuracy than (i) prompting (EmotionPrompt-style) and (ii) single-direction steering (RepE/CAA-style, tuned per-emotion at best layer + best α), under a matched judge.
- **Must-solve bottleneck**: A fair, matched-budget verification protocol that isolates the *representation-level* contribution of the circuit — no free-parameter asymmetry with the single-direction baseline, no cherry-picked layers, no confounded prompting/steering comparison, no unreliable judge.
- **Non-goals**:
  - Not proposing a new mechanistic-interpretability *method family* (family is bound by `/mechanism-skills` in the experiment stage).
  - Not tracing training-time formation of the circuits (Formation Tracing rejected).
  - Not decoding what individual neurons "mean" (Unit Interpretation rejected — no SAE / auto-interp).
  - Not auditing spurious vs. legitimate features (Decision Auditing rejected).
  - Not fine-tuning the model; everything inference-time.
  - Not exploring emotions beyond the six SEV emotions.
- **Constraints**:
  - Model: **Llama-3.2-3B-Instruct** at `/data/zhenqian/models/Llama-3.2-3B-Instruct` (28 layers, hidden 3072, 24 heads, MLP width 8192).
  - Data: **SEV** at `/data/zhenqian/data/SEV/sev.json` — 480 event stems, 6 emotion variants each.
  - Verify swap: **Qwen2.5-7B-Instruct** + SEV held-out.
  - GPUs `{1,2,3,5,6}`; conda env; dir limits (work_dir, `/data/zhenqian/data`, `/data/zhenqian/models`).
  - **10 GPU-hour envelope, but GPU budget is NOT a constraint** — task.md instructs strong methods over cheap methods.
  - External judge available: `gpt-5.4` @ `dmxapi.cn` (proxy bypass required).
- **Success condition**: For all three claims, the unified protocol yields per-emotion, pre-registered directional evidence with statistical CIs — Claim 1 stable-and-sparse `C_e`, Claim 2 causal + specific + stable + structured, Claim 3 circuit > prompting AND circuit > single-direction steering on emotion-expression accuracy on held-out event stems on ≥ 5 of 6 emotions, with judge reliability gated.

## Technical Gap

**Why current methods fall short for this verification task.**

- *Prompting (EmotionPrompt-style)*. Controls the input, not the mechanism; any observed emotion in the output could be trivially attributed to the extra sentence, not to internal computation. This is exactly the baseline task.md's Claim 3 wants to beat, but no head-to-head matched-budget SEV comparison has been reported.
- *Single-direction activation steering (RepE / CAA)*. Collapses the entire emotion signal into one residual-stream direction at one (or a few) layers, so the "circuit" question — is there a sparse set of *components* (heads + neurons) carrying emotion? — is not answered, only the "is there *a* direction?" question is. Also, prior steering results rarely match steering-baseline compute budget to the circuit method's compute budget.
- *Head-level probing + intervention (ITI-style)*. Localizes to attention heads only; MLP neurons are not touched. Emotion generation almost certainly recruits both (heads propagate contextual signal, MLPs implement the KV memory of "what expressing emotion e looks like"), so a heads-only test would systematically under-count the mechanism.
- *Circuit discovery on affect data*. IOI/ACDC/EAP-style circuit discovery has been done on syntactic and factual tasks, not on emotion, and rarely with cross-scenario stability or cross-emotion overlap-structure metrics.

**What is missing at the intersection**: a *single* verification protocol that (i) locates a sparse per-emotion head+neuron circuit, (ii) causally tests it with dose-response and specificity, (iii) tests stability across scenarios and structure across emotions, and (iv) benchmarks it against prompting and *matched-budget* single-direction steering under one shared judge on one shared dataset. That is the gap this proposal fills — for the verification method, not for the mechanism-family method.

## Method Thesis

- **One-sentence thesis**: A single unified verification protocol runs the Location → Causal Intervention → Tuning & Editing ladder on Llama-3.2-3B / SEV under scenario-level held-out splits and matched-budget baselines, so that each of the three claims is put on trial by a *pre-registered directional predicate* with a specificity control and a bootstrap CI — from one fit of the per-emotion circuit `C_e`.
- **Why this is the smallest adequate intervention**: The three claims share one artifact (`C_e`), one dataset, and one judge; running the ladder once, with the specificity + stability + baseline controls in place, is strictly smaller than three separate testing protocols. There is no separate M0 gate (behavior is given), no formation-tracing arm, no SAE arm.
- **Why this route is timely in the foundation-model era**: Circuit-level intervention is where interpretability is currently trying to go beyond single-direction steering; the *matched-budget* comparison against RepE/CAA is the specific piece the literature has not standardized, and doing it on a clean scenario × outcome × emotion grid (SEV) yields comparisons that are both fair and interpretable.

## Contribution Focus

- **Dominant contribution**: A *pre-registered, matched-budget* verification protocol that puts a head+neuron emotion circuit head-to-head against (i) prompting and (ii) single-direction steering on the same held-out SEV event stems under the same judge, with per-emotion accuracy, bootstrap CIs on pairwise differences, off-target specificity checks, and a scenario-level split — proving or disproving all three of task.md's claims from one fit of `C_e`.
- **Optional supporting contribution**: The scenario × emotion-structure ablation (H2b + H2c) — Jaccard stability across scenario partitions and directed inequality (neuron overlap < head overlap) across emotions — turns "stable circuits" from a slogan into two directional predicates with CIs.
- **Explicit non-contributions**:
  - No new mechanism-family method (family is bound by `/mechanism-skills` routing later).
  - No SAE / dictionary-learning decomposition of `C_e`.
  - No training-time formation tracing.
  - No spurious-feature audit.
  - No RLHF / fine-tuning / model editing.
  - No cross-model transfer beyond Verify's Qwen swap.

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**: Llama-3.2-3B-Instruct in inference mode with hidden-state hooks (residual stream at each layer, MLP down-proj input, per-head attention output pre-`W_O`). Standard `transformer_lens` / raw PyTorch hooks — no framework-heavy tooling.
- **New trainable components** (soft cap 2):
  1. A **judge-reliability classifier** trained on SEV emotion labels (a small SEV-emotion classifier finetuned on top of a small sentence encoder — only if the external LLM judge fails the agreement gate). This is a *reliability fallback*, not the main mechanism.
  2. No other trainable component.
- **Tempting additions intentionally excluded**:
  - No SAE training on Llama-3.2-3B activations.
  - No ACDC full-graph edge search (too expensive; instead a per-layer per-emotion probe + top-k selection with an integration step).
  - No cross-layer transcoder or MLP re-parameterization.
  - No formation-time re-training.
  - No LLM-as-adversarial-judge; single-judge with agreement gate + swap ablation.

### System Overview

```
sev.json ─┐
          │  Step 0 (Data prep)
          ▼
  (scenario-split train / val / eval folds)
  (per-emotion positive/negative contrast sets)
          │
          ▼
  Step 1 (Location — Claim 1)
   ┌─────────────────────────────────────┐
   │  For each layer L, each emotion e:  │
   │   d_{e,L} = mean_diff(residual @    │
   │             last-event-token,       │
   │             pos vs. neg contrasts)  │
   │  MLP neurons: per-neuron score      │
   │   (family bound at /mechanism-skills;│
   │    default = alignment-with-d_{e,L} │
   │    + leave-one-out causal effect)   │
   │  Attention heads: ITI-style per-head│
   │   linear probe AUC                  │
   │  Global circuit C_e:                │
   │   top-k neurons (≤~5% MLP)          │
   │   ∪ top-k heads (≤~10% heads)       │
   │   across a handful of layers        │
   │  Stability: seed / event-subset     │
   │   resampling → Jaccard              │
   └─────────────────────────────────────┘
          │  outputs: C_e for e ∈ {6 emotions}
          ▼
  Step 2 (Causal Intervention + Stability — Claim 2)
   ┌─────────────────────────────────────┐
   │  On held-out fold:                  │
   │   Ablation (zero / mean-substitute) │
   │    of C_e on e-target prompts       │
   │    → expect target ↓                │
   │   Enhancement (scaled activation    │
   │    boost OR direction shift, at     │
   │    ≥ 3 strengths α ∈ {α1, α2, α3}) │
   │    → expect target ↑, dose-response │
   │   Specificity: same intervention,   │
   │    score off-target emotions        │
   │    → expect off-target |Δ| << target │
   │  Scenario stability:                │
   │   fit C_e on scenarios S1, S2       │
   │   (disjoint 10+10 of 20 scenarios)  │
   │   → Jaccard(C_e^{S1}, C_e^{S2})     │
   │   vs. random-selection null         │
   │  Cross-emotion structure:           │
   │   mean pairwise Jaccard —           │
   │   neurons < heads (directed)        │
   └─────────────────────────────────────┘
          │
          ▼
  Step 3 (Tuning & Editing — Claim 3)
   ┌─────────────────────────────────────┐
   │  On held-out event stems (no        │
   │   emotion suffix):                  │
   │  Arm A (Circuit): activate C_e      │
   │   at test time via best-strength    │
   │   enhancement recipe (tuned on val) │
   │  Arm B (Prompting): fixed per-e     │
   │   emotional-stimulus suffix         │
   │  Arm C (Single-direction steering): │
   │   ONE mean-diff residual direction  │
   │   d_e at best single layer, best α  │
   │   (both tuned on SAME val split as  │
   │   Arm A's strength → matched budget)│
   │  Generate 100-tok continuation      │
   │  Judge emotion-expression accuracy  │
   │  Per-emotion accuracy + macro       │
   │  Paired-bootstrap 95% CIs on        │
   │   (A − B) and (A − C)               │
   └─────────────────────────────────────┘
          │
          ▼
  Judge reliability gate (per /experiment-tips)
   Agreement ≥ 0.75 vs. gold subset (60 items)
   Judge-swap ablation on ≥ 10% of items
   Fallback: SEV-emotion classifier if gate fails
```

### Core Mechanism

- **Input / output** (of the *testing method*, since the mechanism family is bound later): input = (Llama-3.2-3B activations on SEV pairs); output = (per-emotion component set `C_e` = {neurons_e, heads_e, layers_e}) + (three CI'd verdicts: Location / Causal / Applied).
- **Architecture or policy**: per-layer paired-contrast direction extraction + per-component probe + integration to `C_e`. Location step is **submethod-bound** (`method_sensitive: [n_pairs, sites, metric, gpu_hours]`) so `/mechanism-skills` can pick the strongest MLP-scoring family (e.g., leave-one-out causal effect vs. gradient-attribution vs. alignment score) and the strongest head-probe family without a plan rewrite.
- **Training signal / loss**: none; all direction extraction is closed-form (mean-diff). Optional judge-classifier training loss = cross-entropy on SEV emotion labels (only if judge gate fails).
- **Why this is the main novelty**: the *matched-budget* comparison between circuit-based control, single-direction steering, and prompting on scenario-split SEV under one judge with pre-registered specificity and stability controls.

### Optional Supporting Component

- The judge-reliability classifier is included only if the external LLM judge fails the agreement gate. It is not a contribution — it is a **reliability floor**. Training data: SEV's own event × emotion labels; architecture: any small sentence-encoder + linear head. Not a moving piece of the mechanism claim.

### Modern Primitive Usage

- **Which primitive**: the modern LLM primitive (`gpt-5.4` external judge) is used as the *emotion-expression judge*, not as a proposer, planner, or reward model. Its role is scoring the free-form continuation into one of six emotion labels.
- **Exact role in the pipeline**: reads `(event_stem, target_emotion, continuation)`, returns predicted emotion + confidence.
- **Why it is more natural than an old-school alternative**: an LLM judge handles free-form open-ended continuations (variable length, paraphrase, indirect emotional expression) better than a small classifier — but only if it passes the agreement gate. When it doesn't, we swap to a locally-trained classifier — that is why the fallback exists.

### Integration into Base Generator / Downstream Pipeline

- All interventions are **inference-time** hooks on Llama-3.2-3B. No parameter change. Verify-stage runs the same protocol on Qwen2.5-7B-Instruct + SEV held-out — one line change to the model-loading step and the SEV split.

### Training Plan

- **No training** on Llama-3.2-3B itself.
- The optional judge-classifier is trained once with default hyperparameters (AdamW lr 2e-5, batch 32, 5 epochs) if triggered by a failed judge gate.
- Estimated total GPU-hours (before per-family binding): direction extraction pass ~1h; per-layer per-emotion probing ~2h; ablation / enhancement sweeps ~2h; three-arm Claim-3 eval ~2h; judge audit + Verify swap ~2h. Total ~9 GPU-hours (within the 10-hour envelope). Room for a re-run of one heavy step within budget.

### Failure Modes and Diagnostics

- **F1 — Judge unreliability**: judge agreement < 0.75 or judge-swap ablation flips ≥ 20% of items. *Detect*: gold-subset audit before main runs. *Fallback*: SEV-emotion classifier.
- **F2 — Steering baseline under-tuned**: single-direction steering fails not because circuits are better, but because α or layer wasn't tuned. *Detect*: report per-layer × per-α val heatmap for Arm C; require baseline's best-val setting is genuinely best on val (not a lazy default). *Fallback*: expand α grid / layer sweep on val before eval.
- **F3 — Prompting-baseline weakness**: EmotionPrompt template is under-engineered → unfair win. *Detect*: report per-emotion prompt template + baseline accuracy per emotion. *Fallback*: try two prompt templates and pick the stronger per emotion on val (matched-budget with the other arms' val tuning).
- **F4 — Scenario leakage**: train and eval scenarios overlap. *Detect*: assert scenario-set disjointness in code. *Fallback*: none — this is a bug, not a scientific decision.
- **F5 — Sparsity collapse**: `C_e` grows to > pre-registered sparsity floor to hit accuracy. *Detect*: pre-register sparsity floor (≤ 5% MLP neurons, ≤ 10% heads). *Fallback*: report Claim 1 as *partial* if floor is breached, rather than silently expanding it.
- **F6 — Specificity failure**: Δ on off-target ≥ Δ on target. *Detect*: report per-target × per-off-target Δ matrix. *Fallback*: mark Claim 2 as *partial* (still causal, not specific).
- **F7 — Cross-emotion overlap structure inverted**: neuron overlap ≥ head overlap. *Detect*: report the directed inequality with bootstrap CI. *Fallback*: report Claim 2 as *partial* on H2c and honestly note the inversion.

### Novelty and Elegance Argument

- **Closest work**: EmotionPrompt (Li 2023, prompting baseline), RepE (Zou 2023) + CAA (Rimsky 2023) (single-direction steering baselines), ITI (Li 2023) (head-level probe+intervene template on truthfulness), ROME / knowledge-neuron / IOI / ACDC (neuron / circuit templates on other tasks).
- **Exact difference**: none of the above ran a matched-budget circuit vs. steering vs. prompting comparison for *emotion* on scenario-split SEV under a gated judge with pre-registered specificity and stability controls. That's what this verification protocol delivers.
- **Why this is a focused mechanism-level contribution rather than a module pile-up**: the protocol runs once, from one fit of `C_e`; every reported number is either a directional predicate (with a sign) or a bootstrap CI; nothing is bolted on to inflate the paper.

## Claim-Driven Validation Sketch

### Claim 1 (Location): Framework produces stable, sparse, per-emotion component sets

- **Minimal experiment**: fit `C_e` on train scenarios with 3 seeds and 3 event-subsample folds; report per-emotion `|neurons_e|`, `|heads_e|`, layer distribution, and mean pairwise Jaccard across the 9 resamples.
- **Baselines / ablations**: random-selection null (same |C_e| drawn uniformly) for the Jaccard.
- **Metric**: (i) sparsity ≤ 5% MLP-neuron / ≤ 10% head budget met per emotion; (ii) mean pairwise Jaccard(seeds × folds) > random-null 95% CI upper edge.
- **Expected evidence**: `|C_e|` at or below the floor for all 6 emotions; Jaccard ≥ 2× the random null (rough expected magnitude; the exact number is what the experiment reports).

### Claim 2 (Causal + Stability): C_e is causally sufficient, specific, scenario-stable, and emotion-structured

- **Minimal experiment**: 3-strength dose-response ablation + enhancement on held-out fold (target and off-target Δ per emotion); Jaccard(C_e^{S1}, C_e^{S2}) per emotion; per-emotion pairwise Jaccard for neuron and head sets.
- **Baselines / ablations**: random-selection ablation (ablate a same-sized random set of components) as a specificity check.
- **Metric**: (i) sign(target Δ) matches (ablation ↓, enhancement ↑); (ii) monotonic dose-response (Spearman ρ ≥ 0.7 over 3 strengths); (iii) mean |off-target Δ| < mean |target Δ| at matched strength; (iv) Jaccard(S1, S2) > random-null 95% CI upper edge; (v) mean pairwise neuron-Jaccard < mean pairwise head-Jaccard (directed inequality, bootstrap CI on the difference).
- **Expected evidence**: target Δ negative (ablation) / positive (enhancement); off-target substantially smaller; positive scenario-Jaccard; directed inequality holds.

### Claim 3 (Applied control): Circuit > prompting AND Circuit > single-direction steering

- **Minimal experiment**: three arms (A) circuit, (B) prompting, (C) single-direction steering, on held-out event stems × 6 target emotions, judge scores per-continuation.
- **Baselines / ablations**: baseline (C) is tuned per-emotion at best layer + best α on the same val split as (A)'s strength (matched budget). Baseline (B) uses a fixed simple emotional-stimulus template.
- **Metric**: per-emotion accuracy; macro-average; paired-bootstrap 95% CIs on (A − B) and (A − C).
- **Expected evidence**: A > B on ≥ 5/6 emotions with CI excluding 0; A > C on ≥ 5/6 emotions with CI excluding 0.

## Experiment Handoff Inputs

- **Must-prove claims**: the three above.
- **Must-run ablations**: random-set null for Location (Claim 1); random-set ablation for Specificity (Claim 2); matched-budget α/layer sweep for single-direction steering (Claim 3); prompt-template ablation for prompting baseline (Claim 3).
- **Critical datasets / metrics**: SEV (train scenarios / val scenarios / eval scenarios disjoint from the 20-scenario grid); emotion-expression accuracy per continuation (external LLM judge or fallback classifier); Jaccard(component-set); bootstrap CI on pairwise differences.
- **Highest-risk assumptions**:
  - (R1) The paired-contrast direction at last-event-token position is a *usable* signal for identifying emotion-carrying components. Mitigation: layerwise probing AUC reported per-emotion; if AUC ≈ 0.5 for some emotion, that emotion's `C_e` is reported as unstable.
  - (R2) The external judge is reliable enough to score free-form continuations. Mitigation: gold-subset agreement gate + judge-swap ablation + classifier fallback.
  - (R3) The single-direction steering baseline is fairly tuned. Mitigation: matched-budget val sweep, per-emotion tuning, published val heatmap.
  - (R4) The SEV emotion variants provide clean paired contrasts. Mitigation: inspect sev.json at data-prep time; if the variants are not a controlled per-emotion suffix but a longer text change, adjust the last-token choice accordingly.

## Compute & Timeline Estimate

- **Estimated GPU-hours**: ~9 (well within the 10-hour envelope) — direction extraction ~1h, probing ~2h, ablation+enhancement sweeps ~2h, three-arm Claim-3 eval ~2h, judge audit + Verify swap ~2h.
- **Data / annotation cost**: 60 gold items for judge-agreement subset (10 per emotion) — human-annotated by the researcher (~1h effort). Optionally a second judge (small local classifier) for the swap ablation (existing SEV labels suffice).
- **Timeline**: 1 focused week for data prep + Location + Causal + Claim 3 main runs + verify. Room for one re-run within budget.
