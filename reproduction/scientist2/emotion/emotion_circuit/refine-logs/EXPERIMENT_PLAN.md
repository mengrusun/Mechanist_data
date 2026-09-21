# Experiment Plan

```yaml
mechanism_strategy:
  directions: [Location, "Causal Intervention", "Tuning & Editing"]
  rejected:
    - Formation Tracing — task.md makes no training-time claim.
    - Unit Interpretation — task.md does not claim any component means a specific nameable concept.
    - Decision Auditing — task.md does not ask spurious-vs-legitimate; it asks reliable control.
  note: The three chosen directions map one-to-one onto Claims 1, 2, 3.
chosen_mechanism: to-be-routed-by-mechanism-skills
family_freeze: pre-eval-split
# resource_fidelity NOT stamped — cost-aware combination (MECHANISM=discovery), but task.md
# instructs against downscaling; the plan uses full model size + full SEV at scenario-level split.
# NO M0 phenomenon-validation gate — BEHAVIOR_SOURCE=given.
```

**Problem**: Verify three given claims from task.md about emotion-specific global circuits in Llama-3.2-3B-Instruct on SEV — Location, Causal Intervention + Stability, and Applied Control beating prompting + single-direction steering.
**Method Thesis**: A single unified matched-budget verification protocol runs Location → Causal → Applied on one fit of `C_e`, with pre-registered operators (mean-substitute ablation from same-stem off-target activations, additive activation-injection enhancement), a judge-free internal Stage-B causal ranker (target-prefix log-prob gain), a hidden-target 6-way forced-choice external judge (gated) for Claim 3, and `N=9` matched val budget per arm per emotion.
**Date**: 2026-07-13

## Resources

- **Model**: Llama-3.2-3B-Instruct at `/data/zhenqian/models/Llama-3.2-3B-Instruct` (28 layers × 24 heads × MLP hidden 8192).
- **Dataset**: SEV at `/data/zhenqian/data/SEV/sev.json` — 480 event stems × 6 emotion variants = 2880 pairs.
- **Verify swap**: Qwen2.5-7B-Instruct at `/data/zhenqian/models/Qwen2.5-7B-Instruct`; SEV held-out (480 events, disjoint content).
- **GPUs**: `CUDA_VISIBLE_DEVICES=1,2,3,5,6` (subset OK). No other GPU.
- **Env**: conda (project's env or `mechanistic` env; select via project convention at implementation time).
- **Dir limits**: work_dir (`/data/zhenqian/Reproduction1/mechanica/emotion/emotion_circuit`), `/data/zhenqian/data`, `/data/zhenqian/models`.
- **External judge (optional)**: `gpt-5.4` @ `https://www.dmxapi.cn/v1` with API key `<Your_api>`. **Bypass proxy** (`NO_PROXY=dmxapi.cn,www.dmxapi.cn,localhost,127.0.0.1`; unset `HTTP(S)_PROXY`, `all_proxy`).
- **GPU budget**: 10 GPU-hours total; `task.md` explicitly says budget is NOT a constraint — do not skimp on data / seeds / grid. Actual plan estimate: ~8h on Llama + ~1.5h on Qwen.
- **Data split (scenario-level, disjoint across all 8 domains)**: 10 train scenarios / 5 val scenarios / 5 eval scenarios per domain. Asserted in code before Stage A begins.
- **used_n = available_n** — no dataset subsetting; full SEV.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: Localizability — framework yields sparse, per-emotion `C_e` | Foundation of everything else; without a real per-emotion component set, Claims 2 and 3 cannot even be evaluated. | Per-emotion `|C_e|` at floor (`k_h* ≤ 96`, `k_n* ≤ 8000`); mean pairwise Jaccard(fold, fold) > size-matched permutation null 95% CI (200 draws). | B1 (M1). |
| C2: Causal + Stable — `C_e` causally carries emotion; specific; scenario-stable | This is the mechanistic-evidence claim. Without it, `C_e` might be a spurious high-variance component set that just happens to correlate. | (a) `Δ_ablation(target) < 0` at α2 and `Δ_enhance(target) > 0` at α2, Spearman(α, Δ_enhance) ≥ 0.7 over 3 α; (b) `|off-target Δ| < |target Δ|`; (c) `|Δ_{C_e on e}| > |Δ_{C_{e'} on e}|`; (d) Jaccard(S1, S2) − perm-null > 0 at 95% CI; (e) secondary. | B2 (M2). |
| C3: Applied — circuit beats prompting AND steering | The "reliable control" pay-off. | A > B on ≥ 5/6 emotions AND A > C on ≥ 5/6 emotions, paired-bootstrap 95% CIs on the two pairwise differences excluding 0 per emotion. | B3 (M3), B4 (M4). |
| Anti-claim to rule out: gain from any large-effect component | Random-set null and targeted `C_{e'}` control both fail if `C_e` is not emotion-specific. | Anti-claim rejected iff C2 predicates (b) + (c) both hold. | B2. |
| Anti-claim to rule out: matched-budget asymmetry | If Arm C is under-tuned, the win over steering is unfair. | Matched `N=9` val budget across all arms; Arm C direction constructed on train fold; layer shortlist shared with Arm A. | B3. |

## Paper Storyline

- **Main paper must prove**: C1 (Location), C2 (Causal + Stability, with rubric verdict), C3 (Applied control winning on ≥ 5/6 emotions).
- **Appendix can support**: judge audit (60 gold items, per-emotion confusion matrix); length-matched secondary analysis; Claim-2e cross-emotion overlap-structure secondary; robustness swap on Qwen.
- **Experiments intentionally cut**: SAE decomposition of `C_e`; formation-time tracing; spurious-feature audit; RLHF / fine-tuning; zero-ablation as primary; direction-shift enhancement as primary; pairwise judge as primary; cross-model mechanism-family re-run on Qwen (only Claim-3 Arm-A/B/C is re-run on Qwen).

## Experiment Blocks

### Block 1 — Location (M1)

- **Claim tested**: C1.
- **Why this block exists**: without `C_e`, everything else is moot.
- **Dataset / split / task**:
  - Provenance: **existing** (SEV).
  - Source: `/data/zhenqian/data/SEV/sev.json` (480 events × 6 emotion variants = 2880 pairs).
  - Available N: 2880 (2880 event × emotion pairs); 480 stems × 6 emotions.
  - Planned used N: 2880 (used_n = available_n); split scenario-level 10/5/5, so ≈240 stems train × 6 = 1440 fit pairs, 120 × 6 = 720 val pairs, 120 × 6 = 720 eval pairs.
  - Task: last-event-token residual-stream mean-diff direction extraction; per-layer head probe AUC; MLP-neuron alignment / t-score / single-component causal effect on target-prefix log-prob.
- **Compared systems**:
  - Circuit-locator framework (Stage A shortlist → Stage B causal ranker).
  - Ablation A0 (Stage-A-only, no causal Stage B) — reported to show Stage B is not decorative.
  - Ablation A1 (random top-k after Stage A) — sparsity-matched sanity.
- **Metrics**:
  - Sparsity report: `|C_e^{head}|`, `|C_e^{neuron}|` per emotion (must satisfy `≤ k_h*` / `≤ k_n*`).
  - Layer distribution of selected components per emotion.
  - Jaccard stability: mean pairwise Jaccard across 3 event-subsample folds, vs. size-matched permutation-null 95% CI (200 draws).
  - Primary Stage-B score aggregation: mean over 30 val stems per emotion of `log P(prefix_e | event_stem; enhance c at α2) − log P(prefix_e | event_stem)`, α2 = 1.0.
- **Setup details**:
  - Backbone: Llama-3.2-3B-Instruct (frozen).
  - Hooks: residual stream at each layer's output; per-head pre-`W_O` attention output; MLP neuron pre-activations.
  - Grid: `k_h ∈ {24, 48, 96}` (heads); `k_n ∈ {2000, 4000, 8000}` (MLP neurons). `k_h*, k_n*` selected ONCE globally by macro-average target-prefix log-prob gain on val at α2.
  - Deterministic mean-diff extraction ⇒ no seed dimension. 3 event-subsample folds (each: 80% of train stems, sampled without replacement) for Jaccard stability.
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]` — Stage A / B scoring family is bound by `/mechanism-skills` routing pre-eval-split; the exact `n_pairs`, intervention sites, and score definition may be re-bound at Phase 1.5 without a plan rewrite. The **default** submethod family under discovery routing: MLP neurons scored by cosine alignment (Stage A) + single-component-enhancement prefix logprob gain (Stage B); attention heads scored by ITI-style per-head linear probe AUC (Stage A) + same single-component prefix logprob gain (Stage B).
- **Expected sign / dose-response**: at Stage B, top-scoring components must have `s_c > 0` (single-component enhancement raises the target-prefix log-prob). Magnitude: expect target-prefix log-prob gain in the range 0.02–0.2 nats per top component at α2 = 1.0 (order-of-magnitude — the exact number is what the experiment reports; a top-component gain below 0.01 nats is a diagnostic warning of a weak Stage-B signal).
- **Specificity control (for M1)**: random top-`k_h`, `k_n` selection under Stage-A shortlist only — Jaccard(random, random) across folds must be within permutation-null CI while Jaccard(Stage-B-selected, Stage-B-selected) is above it.
- **Success criterion**:
  - `k_h* ∈ {24, 48, 96}` and `k_n* ∈ {2000, 4000, 8000}` — sparsity floor met.
  - Per-emotion Jaccard > perm-null 95% CI upper edge for ≥ 5/6 emotions.
- **Failure interpretation**:
  - If any emotion has Jaccard below null: mark that emotion's `C_e` as unstable; report Claim 1 as *partial* for that emotion; do NOT run Claim-2/3 predicates on unstable emotions in the "supported" tally (still run for reporting).
- **Table / figure target**: Table 1 (per-emotion `|C_e|`, layer distribution, Jaccard); Figure 1 (heat map of per-layer per-emotion probe AUC + Stage-B score).
- **Priority**: MUST-RUN.

### Block 2 — Causal Intervention + Stability (M2)

- **Claim tested**: C2.
- **Why this block exists**: promotes `C_e` from "located" to "mechanistically causal".
- **Dataset / split / task**:
  - Provenance: **existing** (SEV, eval fold from B1's scenario split).
  - Source: same SEV path; 120 eval stems × 6 emotions = 720 eval pairs.
  - Available N: 720; Planned used N: 720.
  - Task: apply ablation / enhancement to `C_e` on target-emotion prompts; measure Δ(target-prefix log-prob); repeat for off-target scoring, `C_{e'}` targeted control, random-set null.
- **Compared systems**:
  - Primary intervention: `C_e` at α ∈ {0.5, 1.0, 2.0} for enhancement; `C_e` mean-substitute for ablation.
  - Specificity controls: (i) random-set null (100 same-size random component sets, per emotion); (ii) `C_{e'}` targeted (for each pair e, e′); (iii) off-target scoring of the same intervention.
  - Stability: `C_e^{S1}` (first 5 of 10 train scenarios per domain) vs. `C_e^{S2}` (other 5), Jaccard vs. permutation null.
  - Cross-emotion overlap structure (Claim 2e — secondary): pairwise Jaccard(e, e′) for neuron and head sets, bootstrap CI on the difference.
- **Metrics**:
  - Per-emotion Δ(target-prefix log-prob) under ablation and each of 3 α strengths of enhancement, on the eval fold. Report per-emotion mean + paired-bootstrap 95% CI.
  - Spearman(α, Δ_enhance) over 3 α strengths — dose-response.
  - Off-target: mean |Δ| on off-target continuations under the same intervention.
  - Targeted-`C_{e'}`: `|Δ_{C_e on e}|` vs. `|Δ_{C_{e'} on e}|` with paired-bootstrap CI on the difference.
  - Scenario stability: Jaccard(S1, S2) − permutation-null-mean, 95% CI.
  - Secondary (Claim 2e): mean neuron-Jaccard − mean head-Jaccard, bootstrap CI.
- **Setup details**:
  - Backbone: Llama-3.2-3B-Instruct (frozen).
  - Ablation operator: mean-substitute component activations with mean over OTHER 5 emotion variants of the same stem.
  - Enhancement operator: additive activation injection at α ∈ {0.5, 1.0, 2.0} on selected components; for a head, add `α · d_{e,L}^{h}` to the head's contribution to residual; for a neuron, add `α · sign(⟨d_{e,L}, w_n^{out}⟩) · std(activation_n | pos-e on train)` to pre-activation.
  - Everything deterministic; no seed dim.
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]` — the intervention semantics (activation-space vs. residual-additive vs. per-head-slice) is submethod-bound.
- **Expected sign / magnitude / dose-response**:
  - Ablation Δ_target < 0 at α2; enhancement Δ_target > 0 at α2; monotonic in α (Spearman ≥ 0.7).
  - Off-target |Δ| substantially smaller than target |Δ| — expect ≥ 2× ratio at α2 (order-of-magnitude; exact ratio is the reported number).
  - `|Δ_{C_e on e}| > |Δ_{C_{e'} on e}|` at α2 with 95% CI on the difference excluding 0.
- **Specificity controls (three)**: random-set null; targeted-`C_{e'}`; off-target continuation scoring — all reported.
- **Success criterion (rubric)**:
  - **Full** iff (a) causal-sign + dose + (b) raw-off-target + (c) targeted-`C_{e'}` + (d) scenario stability all pass at 95% CI.
  - **Partial** iff (a) + at least one of (b, c) but (d) fails.
  - **Causal-only** iff (a) alone.
  - **Not-supported** iff (a) fails.
- **Failure interpretation**: label Claim 2 with rubric state; iterate only via `/auto-iteration-loop`, not this plan.
- **Table / figure target**: Table 2 (per-emotion causal Δ + specificity + stability); Figure 2 (dose-response curves per emotion).
- **Priority**: MUST-RUN.

### Block 3 — Applied Control (M3)

- **Claim tested**: C3.
- **Why this block exists**: the pay-off. Also isolates the *representation-level* contribution of the circuit vs. two named baselines.
- **Dataset / split / task**:
  - Provenance: **existing** (SEV, eval fold from B1's scenario split — same 120 eval stems).
  - Source: same SEV path.
  - Available N eval: 120 stems × 6 target emotions × 3 arms = 2160 continuations; Planned used N: 2160.
  - Val for hyperparameter selection: 120 val stems × 6 emotions × 3 arms × 9 configs = 19 440 val forward passes (fast — only next-token log-prob or short greedy).
  - Task: for each eval stem, condition on target emotion `e` **without** naming it in the input (Arm A / C) or with a prompting suffix (Arm B), generate greedy continuation, judge emotion-expression accuracy.
- **Compared systems (three arms, matched `N=9` val budget per arm per emotion)**:
  - **Arm A — Circuit**: activate `C_e` via enhancement operator with `α_A ∈ {0.5, 1.0, 2.0}`; `(k_h, k_n)` from a 3-cell neighborhood around `(k_h*, k_n*)` (e.g., `{(k_h*/2, k_n*/2), (k_h*, k_n*), (k_h*·2, k_n*·2)}` clipped to grid); best per emotion by val macro-accuracy.
  - **Arm B — Prompting**: fixed 3 templates × 3 injection positions (prefix / suffix / interleaved) = 9. Templates T1/T2/T3 (see FINAL_PROPOSAL.md). Best per emotion.
  - **Arm C — Single-direction steering (RepE/CAA-style)**: direction `d_e = mean_diff(residual[last-event-token], positive-e vs. off-target-uniform)` on **train fold** — frozen pre-eval-split. Injection: additive with scalar `α_C` on residual after chosen layer's output projection, applied at every token position after the event stem (CAA convention). Grid: `L ∈` top-3 layers by probe AUC (SAME shortlist Arm A uses in Stage A) × `α_C ∈ {0.5, 1.0, 2.0}` = 9. Best per emotion.
- **Metrics**:
  - **Primary (single scalar per continuation)**: hidden-target 6-way forced-choice accuracy from external judge (`gpt-5.4`); judge sees `(event_stem, continuation)`, target label hidden; judge returns one label; correct iff `predicted == target`.
  - **Per emotion**: accuracy over eval continuations for that target emotion, per arm.
  - **Aggregate**: macro-average across 6 emotions per arm; paired-bootstrap 95% CI on (A − B) and (A − C) per emotion and on macro.
  - **Secondary**: per-arm mean continuation length; if any pair differs > 15%, length-matched secondary analysis (truncate to shorter arm's mean length, re-judge on the matched subset).
  - **Optional secondary**: LLM-judge pairwise comparative accuracy (same stem + target, judge picks which of two continuations better expresses target).
- **Setup details**:
  - Decoding for ALL arms: `temperature=0`, `max_new_tokens=100`, no repetition penalty, deterministic.
  - Judge system prompt (frozen): fixed rubric listing 6 emotion labels, single-label forced choice.
  - Judge gate: 60 gold items (10 per emotion, human-labeled), require agreement ≥ 0.75; per-emotion confusion matrix reported; 10% judge-swap ablation using a small local SEV-emotion classifier; if agreement gate fails, classifier is primary.
  - Grid (this milestone uses the `grid:` field to expand runs):
    ```
    grid:
      arm: [A_circuit, B_prompting, C_steering]
      emotion: [<6 emotions from sev.json>]
      config_id: [1, 2, 3, 4, 5, 6, 7, 8, 9]   # per-arm 3x3
    ```
    Val runs = 3 arms × 6 emotions × 9 configs = 162 val runs (each cheap: 120 val stems × greedy generation × judge). Eval runs = 3 arms × 6 emotions × 1 selected config = 18 eval runs (120 eval stems each).
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]` — the enhancement operator (Arm A) and the direction / injection semantics (Arm C) are submethod-locked pre-eval-split by `/mechanism-skills` routing; per-arm site set may be re-bound at Phase 1.5.
- **Expected sign**: `accuracy(Arm A) > accuracy(Arm B)` AND `accuracy(Arm A) > accuracy(Arm C)` on ≥ 5/6 emotions with 95% CI on each pairwise difference excluding 0.
- **Magnitude expectation** (order-of-magnitude only): all three arms should exceed a 1/6 ≈ 16.7% chance floor by a substantial margin; the interesting pairwise gap A − C is expected in the ballpark of several accuracy percentage points per emotion — but the exact number is what the experiment reports.
- **Specificity control**: length audit (see above) + a **random-set circuit control** (activate a same-sized random set instead of `C_e`; if random-set accuracy is at chance while Arm A is at Arm-A-observed level, the specificity is confirmed).
- **Success criterion**: as above (A > B AND A > C on ≥ 5/6 emotions with CIs).
- **Failure interpretation**:
  - If A > B fails (prompting wins): Claim 3 is *partial* — report honestly; the circuit is causal (from Claim 2) but not competitively better than prompting.
  - If A > C fails (steering wins): the matched-budget comparison found no representation-level advantage — Claim 3 is *not supported* against steering; report and offer no gloss.
  - If both fail: Claim 3 is *not supported*.
- **Table / figure target**: Table 3 (per-emotion accuracy × 3 arms + macro + pairwise CIs); Figure 3 (per-emotion accuracy bar chart with CIs).
- **Priority**: MUST-RUN.

### Block 4 — Verify-swap-lite (M4)

- **Claim tested**: robustness of C3 to model swap.
- **Why this block exists**: minimal cross-model check that Arm A's advantage over Arms B/C is not Llama-3.2-3B-specific.
- **Dataset / split / task**:
  - Provenance: **existing** (SEV held-out — 480 events, per task.md's Verify-stage constraint that they are "disjoint content"; if `sev.json` does not natively split held-out, this block re-uses the same scenario-level 5-scenario eval fold on the Qwen model).
  - Source: `/data/zhenqian/data/SEV/sev.json` (SEV held-out subset).
  - Available N eval: 120 stems × 6 target emotions × 3 arms = 2160 continuations.
  - Task: re-run only the M3 three-arm comparison on Qwen2.5-7B-Instruct. Do NOT re-run M1/M2 on Qwen (that would be a new mechanism-family study).
- **Compared systems**: same three arms A / B / C with the SAME hyperparameters *re-selected on Qwen's own val fold under matched N=9 budget* — do not port Llama-tuned settings.
- **Metrics**: same as M3 primary.
- **Setup**: Qwen2.5-7B-Instruct; `CUDA_VISIBLE_DEVICES=1,2,3,5,6`; conda env; hooks correspond to Qwen's layers/heads (Qwen has 28 layers, 28 heads, MLP width 18944).
- **Expected sign**: A > B AND A > C on ≥ 4/6 emotions (slightly relaxed from Llama's ≥ 5/6 because this is a *robustness* check under a different architecture, not the main experiment).
- **Success criterion**: A > B AND A > C on ≥ 4/6 emotions with 95% CI on each pairwise difference excluding 0.
- **Failure interpretation**: if the effect is Llama-only, Claim 3 is *conditional* — supported on Llama-3.2-3B, uncertain elsewhere.
- **Table / figure target**: Table 4 (Qwen accuracy × 3 arms); appendix.
- **Priority**: MUST-RUN.

### Block 0.5 — Data prep + judge audit (M0.5)

- **Claim tested**: none (pre-run integrity).
- **Why this block exists**: pipeline correctness precedes any claim.
- **Dataset / split / task**:
  - Provenance: **existing** (SEV) + **adapted** (60-item gold subset; human-annotated by the researcher for the judge gate).
  - Source: `/data/zhenqian/data/SEV/sev.json`; gold subset = 60 items (10 per emotion), sampled from train fold.
  - Available N: 2880 pairs; Planned used N: 2880 for pipeline; 60 for gold judge audit.
  - Task: parse sev.json; confirm 480 events × 6 emotion variants; enumerate emotion labels; construct scenario-level 10/5/5 split (asserted disjoint); build paired positive/negative contrasts per emotion; build 60-item gold subset (10 per emotion) with human labels; run judge (gpt-5.4) on gold; compute agreement + per-emotion confusion matrix; if agreement < 0.75, train the SEV-emotion classifier fallback.
- **Compared systems**: primary judge (gpt-5.4) vs. fallback classifier (only if triggered).
- **Metrics**: gold-set agreement; per-emotion confusion matrix; 10% judge-swap ablation Cohen's κ.
- **Setup**: script-level; no GPU load beyond the fallback classifier (which trains in < 15 min on a single GPU).
- **method_sensitive**: none.
- **Success criterion**: agreement ≥ 0.75 → judge gate PASS; else classifier fallback trained and swapped in.
- **Failure interpretation**: judge-and-classifier both fail (< 0.75) ⇒ escalate to a human-annotated eval subset (~120 items) and re-run Claim-3 primary on that subset — extremely unlikely given SEV's simple 6-way label space.
- **Table / figure target**: appendix (gold-set stats + confusion matrix).
- **Priority**: MUST-RUN.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost (GPU-h) | Risk |
|---|---|---|---|---|---|
| M0.5 | Data prep + judge audit | 1 pipeline + 60 gold + (opt.) 1 classifier train | Judge agreement ≥ 0.75 else classifier fallback trained | 0.5 | Judge disagreement on ≥ 25% items → classifier fallback (mitigated). |
| M1 | Location — build `C_e` | Stage A (per-layer per-emotion probe + neuron score) + Stage B (per-shortlisted-component enhancement + prefix logprob) × 6 emotions + 3 event-subsample folds | Jaccard > perm-null CI on ≥ 5/6 emotions | 2.0 | Weak Stage-B signal for some emotions (mitigated: report Claim 1 partial for those). |
| M2 | Causal + Stability | Ablation + 3-α enhancement + random-null (100 draws) + `C_{e'}` control + off-target scoring + scenario S1/S2 refit × 6 emotions | Rubric verdict (full / partial / causal-only / not-supported) | 2.0 | Off-target specificity failure (mitigated: three independent controls). |
| M3 | Applied Control | Val: 3 arms × 6 emotions × 9 configs on 120 val stems = 162 val runs. Eval: 3 arms × 6 emotions × 1 selected config on 120 eval stems + judge = 18 eval runs. Depends on M1 for `C_e` and layer shortlist. | A > B on ≥ 5/6 AND A > C on ≥ 5/6 | 2.0 | Steering baseline under-tuned (mitigated: matched N=9 + heatmap). |
| M4 | Verify-swap-lite (Qwen) | Same M3 protocol on Qwen2.5-7B-Instruct + SEV held-out | A > B on ≥ 4/6 AND A > C on ≥ 4/6 | 1.5 | Qwen mechanism differs (mitigated: Claim 3 conditional label). |

**Depends on** graph: M0.5 → M1 → {M2, M3} → M4. M2 and M3 can run in parallel after M1.

**Total estimated GPU-hours**: 8.0 (Llama) + 1.5 (Qwen) = 9.5 GPU-h — within the 10-h envelope with headroom for one re-run.

## Compute and Data Budget

- **Total estimated GPU-hours**: ~9.5 (well within 10 h).
- **Data preparation needs**: parse sev.json; build 10/5/5 scenario split; build 60-item gold subset; ~1h of researcher time.
- **Human evaluation needs**: 60 gold labels only (single researcher, ~30 min).
- **Biggest bottleneck**: Stage B (M1) — per-shortlisted-component forward passes. Mitigated by shortlisting (top 5% neurons in top 3 layers + top 20% heads in top 3 layers ≈ few thousand components total, comfortably batched).

## Risks and Mitigations

- **R1 — Last-event-token direction weak for some emotions.** *Detect*: layerwise probe AUC per emotion; if AUC ≈ 0.5, emotion is unstable. *Mitigation*: report Claim 1 partial for that emotion; do not run Claim 2/3 predicates as supported.
- **R2 — Judge unreliable.** *Detect*: gold-agreement gate. *Mitigation*: classifier fallback.
- **R3 — Steering baseline under-tuned.** *Detect*: val heatmap. *Mitigation*: matched N=9 val budget; layer shortlist shared with Arm A; direction constructed on train fold, frozen pre-eval-split.
- **R4 — Scenario leakage.** *Detect*: code assertion at data prep. *Mitigation*: none (bug).
- **R5 — Sparsity collapse (`k_h*` / `k_n*` hits ceiling).** *Detect*: grid boundary. *Mitigation*: report Claim 1 partial; do not silently expand grid.
- **R6 — Length confound.** *Detect*: length audit. *Mitigation*: length-matched secondary analysis.
- **R7 — GPU budget overrun.** *Detect*: milestone-level GPU-h tracking. *Mitigation*: this plan is at 9.5 h with a 10 h budget; task.md says do not stop until actual usage reaches 10 h; a single re-run of the heaviest step (Stage B) is affordable within the remaining margin.
- **R8 — Cross-model divergence on Qwen (M4).** *Detect*: A vs. B/C on Qwen. *Mitigation*: label Claim 3 conditional (Llama-only) if it fails.

## Final Checklist

- [x] Main paper tables covered (Table 1 — Location, Table 2 — Causal + Stability, Table 3 — Applied, Table 4 — Verify swap).
- [x] Novelty isolated (matched-budget three-arm comparison; targeted `C_{e'}` control; scenario split).
- [x] Simplicity defended (rank fusion removed; one primary ablation, one primary enhancement, one primary judge endpoint; Claim 2e demoted to secondary; no SAE / auto-interp / formation-tracing).
- [x] Frontier contribution justified or explicitly not claimed (LLM judge in gated advisory role only; `/mechanism-skills` pre-eval-split freeze).
- [x] Nice-to-have separated from must-run (Claim 2e secondary; length-matched secondary; pairwise judge secondary).
- [x] `mechanism_strategy` stamped; `chosen_mechanism: to-be-routed-by-mechanism-skills`; `family_freeze: pre-eval-split`.
- [x] No M0 gate (BEHAVIOR_SOURCE=given).
- [x] `method_sensitive: [n_pairs, sites, metric, gpu_hours]` on every intervention milestone (M1, M2, M3).
- [x] Depends_on graph explicit.
