# Round 1 Refinement

## Problem Anchor (verbatim from round 0)

- **Bottom-line problem**: Verify — under a single unified testing protocol — three claims about emotion circuits in `Llama-3.2-3B-Instruct` on the SEV dataset (480 events × 6 emotions = 2880 pairs):
  1. **Location**: a systematic framework yields identifiable, sparse, per-emotion component sets `C_e`.
  2. **Causal Intervention + Stability**: ablation weakens, enhancement strengthens with dose-response, specificity holds; `C_e` is stable across scenarios and structured across emotions.
  3. **Applied control**: intervening on `C_e` on held-out event stems beats prompting AND single-direction steering on emotion-expression accuracy.
- **Must-solve bottleneck**: fair, matched-budget verification with no free-parameter asymmetry, no cherry-picked layers, no unreliable judge.
- **Non-goals**: no new mechanism family; no formation tracing; no SAE; no decision auditing; no fine-tuning.

## Anchor Check

- **Original bottleneck**: matched-budget verification of the three claims — still the target.
- **Why the revised method still addresses it**: the interfaces (component selection, ablation operator, enhancement operator, judge protocol, matched budget) are now pinned to one primary operator each; nothing changed at the claim level.
- **Reviewer suggestions rejected as drift**: none. All CRITICAL / IMPORTANT items are about *how* to test, not *what* to test.

## Simplicity Check

- **Dominant contribution after revision**: unchanged — pre-registered, matched-budget verification protocol running Location → Causal → Applied on one `C_e` fit.
- **Components removed or merged**: seed resampling deleted (deterministic mean-diff); "zero OR mean-substitute" ablation collapsed to mean-substitute only; "scaled activation OR direction shift" enhancement collapsed to additive activation injection; judge protocol simplified to one primary endpoint.
- **Reviewer suggestions rejected as unnecessary complexity**: pairwise comparative judge scoring is kept as *optional secondary* only, not adopted as primary.
- **Why the remaining mechanism is still the smallest adequate route**: one operator per intervention, one primary endpoint, one fit of `C_e` — everything else is either a control (targeted `C_{e'}` ablation, random-set null, length audit) or a report.

## Changes Made

### 1. Component selection pinned down (CRITICAL, dim 2)
- **Reviewer said**: `C_e` construction has too many post-hoc knobs.
- **Action**: Score = per-family single score (attention heads: ITI-style per-head linear-probe AUC over the paired contrasts; MLP neurons: two-stage — Stage A alignment-with-`d_{e,L}` shortlist, Stage B causal-effect on shortlist only, using a single family bound by `/mechanism-skills` *before any eval split is touched*). Aggregation = within-layer z-score → single global top-k per family (heads and neurons). `k_h, k_n` chosen from a shared discrete grid on val: heads ∈ {24, 48, 96} (≈ 3.6% / 7.1% / 14.3% of 672 total; last one is the ceiling only if the sparsity floor is relaxed), neurons ∈ {2000, 4000, 8000} (≈ 0.9% / 1.7% / 3.5% of 229 376 total MLP neurons across 28 layers). Same grid across emotions; no per-emotion post-hoc tuning of `k`. No layer cap; global cardinality cap enforced by the top-k.
- **Reasoning**: pre-registered discrete grids remove researcher degrees of freedom while keeping enough flexibility for the val step.
- **Impact**: Claim 1's sparsity assertion is now falsifiable at the reported grid resolution.

### 2. Ablation and enhancement operators single-valued (CRITICAL, dim 2)
- **Reviewer said**: "zero / mean-substitute" and "scale OR direction shift" leave the intervention semantics open.
- **Action**:
  - **Ablation**: mean-substitute the selected components' activations with the *per-neuron / per-head activation mean over off-target-emotion prompts on the same event stem*. Not zero. Zero-ablation moved to a robustness appendix.
  - **Enhancement**: additive activation injection at each selected component. For an MLP neuron `n` at layer `L`, add `α · sign(⟨d_{e,L}, w_n^{out}⟩) · std(activation_n | pos-e)` to the neuron pre-activation, where `w_n^{out}` is the down-projection row. For an attention head `h` at layer `L`, add `α · d_{e,L}^{h}` (the per-head slice of the emotion direction) to the head's contribution to the residual stream (ITI recipe). Single `α ∈ {α1, α2, α3}` (3 strengths for dose-response, one final `α*` picked on val for Claim 3).
- **Reasoning**: One operator per intervention makes causal / specificity numbers comparable across emotions and dose steps.
- **Impact**: Claim 2's causal / specificity / dose-response predicates now have a single well-defined operator.

### 3. Judge protocol pinned (CRITICAL, dim 2)
- **Reviewer said**: prompting format, target visibility, forced-choice vs. free-form are all still loose.
- **Action**:
  - **Primary endpoint (Claim 3)**: unconditional 6-way forced-choice predicted-emotion accuracy — judge sees only `(event_stem, continuation)`, target label is **hidden**; judge returns one of the 6 emotion labels; per-continuation binary correctness = `predicted == target`. Rubric: fixed system prompt spelling out the 6 emotions and asking for the single best label.
  - **Secondary (optional)**: pairwise comparative scoring — two continuations for the same stem+target, judge picks which better expresses the target. Reported only as a robustness sanity check.
  - **Judge reliability gate**: on a 60-item gold subset (10 per emotion; human-annotated by the researcher), judge agreement with human ≥ 0.75; judge-swap ablation on ≥ 10% of items using a locally trained SEV-emotion classifier as second judge. Fail gate ⇒ classifier becomes primary.
- **Reasoning**: forced-choice + hidden target eliminates target-leak, gives one scalar to report.
- **Impact**: Claim 3's primary metric is now one number per (arm, emotion) with a paired-bootstrap CI.

### 4. Matched budget operationalized (CRITICAL, dim 2)
- **Reviewer said**: "matched budget" is rhetorical, not operational; circuit tunes `{k_h, k_n, α}` while baseline C tunes `{layer, α}` — asymmetry.
- **Action**: Define budget as `N = 9` validation hyperparameter evaluations per emotion per arm.
  - **Arm A (circuit)**: 3 choices of `(k_h, k_n)` × 3 choices of `α` = 9 val evaluations per emotion.
  - **Arm C (single-direction steering)**: 3 layers × 3 `α` values = 9 val evaluations per emotion (the 3 layers picked as the top-3 layers by paired-contrast probe AUC on val, mirroring circuit's Stage-A shortlist).
  - **Arm B (prompting)**: 3 prompt templates × 3 injection positions (prefix / suffix / interleaved) = 9 val evaluations per emotion, selected on val.
- **Reasoning**: `N = 9` on val for every arm is a strictly matched compute budget.
- **Impact**: Fair comparison — no arm gets extra degrees of freedom.

### 5. Size-matched targeted specificity control (CRITICAL, dim 6)
- **Reviewer said**: random-set null is not strong enough; add a within-family targeted control.
- **Action**: For each target emotion `e`, additionally run ablation and enhancement of `C_{e'}` (`e' ≠ e`) while measuring effect on continuations targeted at `e`. Expected: `|Δ|` when using `C_{e'}` is significantly smaller than when using `C_e` on the same target — the causal effect is *emotion-specific*, not "just any big-effect components".
- **Reasoning**: this is the strongest specificity check available.
- **Impact**: Claim 2 gains a second directional predicate (`C_e effect > C_{e'} effect`) with paired-bootstrap CI.

### 6. Fixed decoding + length audit (CRITICAL, dim 6)
- **Reviewer said**: length confound is possible.
- **Action**: all arms use identical decoding — `temperature=0`, `max_new_tokens=100`, no repetition penalty, no early-stop token. Report per-arm average continuation length; if any pair differs by > 15% in length, run a length-matched secondary analysis (truncate to shorter arm's mean length and re-judge on a matched subset).
- **Impact**: removes a plausible confound; a small extra report cost.

### 7. Normalized overlap for Claim 2c (CRITICAL, dim 6)
- **Reviewer said**: raw Jaccard is size-sensitive.
- **Action**: Because `k_h, k_n` are fixed across emotions (Change 1), `|neurons_e|` and `|heads_e|` are equal across emotions within each family — Jaccard is size-fair by construction. Also supplement with a size-matched permutation null (draw same-size random component sets and compute permutation Jaccard mean+95% CI); the reported statistic is `Jaccard(e, e') − permutation_null(k_h) − permutation_null(k_n)` for the family.
- **Impact**: Claim 2c becomes size-fair by design.

### 8. Two-stage locator + reduced resampling (IMPORTANT, dim 5)
- **Reviewer said**: leave-one-out at 28 × 8192 neuron scale is combinatorially unaffordable.
- **Action**:
  - Stage A (cheap): per-layer per-emotion mean-diff direction `d_{e,L}`; MLP neurons scored by cosine alignment `⟨w_n^{out}, d_{e,L}⟩` and by pre-activation contrast t-score; attention heads scored by per-head ITI-probe AUC. Shortlist top layer subset (top-3 by summed probe AUC) and, within those layers, top 5% of MLP neurons + top 20% of heads.
  - Stage B (causal): apply the pre-registered enhancement operator on each shortlisted component individually (per-component effect at fixed `α = α2`) — this is O(k_shortlist), not O(all).
  - Global top-k selection uses the union of Stage-A and Stage-B rankings (rank fusion by mean rank).
- **Resampling**: deterministic mean-diff means seed resampling is deleted. Keep 3 event-subsample folds only (for Claim 1's Jaccard stability). Cheaper.
- **Impact**: fits comfortably in the 10-GPU-hour envelope; revised estimate below.

### 9. Verify swap reduced scope (IMPORTANT, dim 5)
- **Reviewer said**: full ladder on Qwen adds cost without new information.
- **Action**: Verify (Workflow 1.75) runs ONLY the Claim-3 three-arm primary comparison on Qwen2.5-7B-Instruct + SEV held-out. Location and Causal ladder are NOT re-run on Qwen (that would be a new mechanism-family study, not a robustness check).
- **Impact**: fits within budget; still substantiates the Verify-stage swap.

### 10. `/mechanism-skills` locking (Modernization opp 3)
- **Reviewer said**: `/mechanism-skills` should pick its scoring family BEFORE any eval split is touched.
- **Action**: pipeline stamps a `family_freeze: pre-eval-split` gate — the experiment stage's Phase 1.5 routing MUST commit to `CHOSEN_FAMILY` before any eval-split activations are read. `method_sensitive: [n_pairs, sites, metric, gpu_hours]` remains the only tag whose value the family may re-bind at commit time.
- **Impact**: eliminates a subtle post-hoc DoF.

## Revised Proposal

# Research Proposal: Unified Testing Protocol for Emotion-Specific Global Circuits in Llama-3.2-3B (revised)

## Problem Anchor
[carried verbatim from Round 0 — omitted here for brevity in this refinement document; the FINAL_PROPOSAL.md contains the complete carried-forward anchor.]

## Method Thesis
- **One-sentence thesis**: A single unified verification protocol runs the Location → Causal Intervention → Applied Control ladder on Llama-3.2-3B / SEV, with scenario-level held-out splits, a two-stage locator (cheap-filter Stage A → shortlist causal Stage B), one pre-registered ablation operator (mean-substitute-from-off-target), one enhancement operator (additive activation injection with scalar `α`), one primary judge endpoint (hidden-target 6-way forced choice), and `N = 9` matched-val budget per arm per emotion — putting all three claims on trial from one fit of `C_e` with directional predicates and bootstrap CIs.

## Complexity Budget
- **Frozen**: Llama-3.2-3B-Instruct, all weights.
- **New trainable**: 0 (unless judge gate fails → 1 small SEV-emotion classifier as reliability backstop).
- **Intentionally excluded**: SAE / dictionary learning; full ACDC edge search; zero-ablation as primary; direction-shift enhancement (moved to appendix); pairwise judge as primary; RLHF / fine-tuning; formation tracing.

## System Overview (final)

```
sev.json ─► Data prep (scenario-split fit/val/eval; per-emotion contrast sets)
        │
        ▼
Step 1 (Location — Claim 1)
   Stage A (cheap): per-layer d_{e,L} = mean_diff(residual@last-event-token, pos vs. neg)
                    heads scored by per-head ITI-probe AUC
                    neurons scored by cosine(w_n^{out}, d_{e,L}) + pre-act t-score
                    ⇒ shortlist top-3 layers, top-5% neurons in those layers, top-20% heads
   Stage B (causal): per-component enhancement at fixed α = α2; measure Δ(target-emo score)
   Global selection: rank-fusion(Stage A, Stage B) → top-k_h heads, top-k_n neurons GLOBALLY
                     with within-layer z-score normalization
   k_h ∈ {24, 48, 96}, k_n ∈ {2000, 4000, 8000}, picked on val (shared across e)
   Deterministic mean-diff ⇒ NO seed resampling; 3 event-subsample folds for Jaccard stability
        │
        ▼
Step 2 (Causal Intervention + Stability — Claim 2)
   On held-out fold:
     Ablation: mean-substitute components in C_e using per-neuron / per-head mean
               over OFF-TARGET-emotion prompts on the SAME event stem
     Enhancement: additive injection at α ∈ {α1, α2, α3}
   Specificity controls (per target e):
     (i) random-set null (100 draws, same-size)
     (ii) size-matched targeted control: C_{e'} intervention while scoring on e
     (iii) off-target continuation scoring under C_e intervention
   Scenario stability: fit C_e on 10 scenarios S1, refit on disjoint 10 S2, compute Jaccard
     Report Jaccard − permutation_null(k) for size fairness
   Cross-emotion structure: mean pairwise neuron-Jaccard vs. head-Jaccard (directed inequality)
        │
        ▼
Step 3 (Applied Control — Claim 3)
   Held-out event stems (no emotion suffix); T=0, max_new_tokens=100
   Val: N = 9 hyperparameter evals per arm per emotion (matched budget)
     Arm A (Circuit): (k_h, k_n) ∈ 3 × α ∈ 3
     Arm B (Prompting): 3 templates × 3 injection positions
     Arm C (Single-direction steering): 3 layers × 3 α
   Test: greedy generation; hidden-target 6-way forced-choice judge; per-emotion accuracy
   Report: per-emotion accuracy + macro; paired-bootstrap 95% CIs on (A−B) and (A−C)
   Length audit: mean length per arm; if any pair differs > 15%, length-matched secondary
        │
        ▼
Judge reliability gate (before main runs)
   60-item gold subset (10 per emotion, human-labeled)
   Agreement ≥ 0.75 AND judge-swap ablation on ≥ 10% items
   Fail ⇒ SEV-emotion classifier becomes primary judge
```

## Claim-Driven Validation Sketch (revised)

### Claim 1 (Location)
- Fit `C_e` on train scenarios × 3 event-subsample folds (deterministic → no seed dim).
- Predicates: (i) `|C_e|` at floor (`k_h ≤ 96`, `k_n ≤ 8000`) for all 6 emotions; (ii) mean pairwise Jaccard across folds > 95% CI upper edge of size-matched permutation null.

### Claim 2 (Causal + Stability)
- Predicates:
  - (a) ablation Δ(target) < 0 at α2; enhancement Δ(target) > 0 at α2; Spearman ρ over 3 α strengths ≥ 0.7.
  - (b) `mean |off-target Δ|` < `mean |target Δ|` at α2 (raw specificity).
  - (c) `|Δ_{C_e on e}|` > `|Δ_{C_{e'} on e}|` at α2 (targeted-control specificity).
  - (d) mean Jaccard(`C_e^{S1}`, `C_e^{S2}`) > permutation-null 95% CI upper edge.
  - (e) mean pairwise neuron-Jaccard < mean pairwise head-Jaccard, with bootstrap CI on the difference excluding 0.

### Claim 3 (Applied)
- Predicates: A > B on ≥ 5/6 emotions AND A > C on ≥ 5/6 emotions, with paired-bootstrap 95% CIs excluding 0. Report: macro-average table, per-emotion table, val heatmaps for all three arms, length audit.

## Compute & Timeline (revised)

- Stage A (cheap filter): ~0.5h.
- Stage B (per-shortlisted-component causal effect at fixed α, batched): ~1.5h.
- Ablation + 3-α enhancement + specificity controls: ~1.5h.
- Scenario stability + cross-emotion structure: ~0.5h.
- Claim-3 val (N=9 per arm × 6 emotions × ~50 val stems × greedy) + eval (3 arms × 6 emotions × ~80 eval stems): ~2.0h.
- Judge audit + judge calls: ~0.5h (mostly API latency, GPU-idle).
- Verify swap on Qwen2.5-7B (Claim 3 only): ~1.5h.
- **Total ~8 GPU-hours** on Llama-3.2-3B (+1.5h on Qwen), well inside the 10-hour envelope.

## Failure Modes (updated)

- **F1 — judge unreliability** → classifier fallback (unchanged).
- **F2 — steering baseline under-tuned** → matched N=9 val budget + heatmap (fixed).
- **F3 — prompting weakness** → 3-template val + explicit framing as *external-control baseline* (fixed).
- **F4 — scenario leakage** → assert disjointness in code (unchanged).
- **F5 — sparsity collapse** → discrete grid ceiling; report Claim 1 partial if breached (unchanged).
- **F6 — specificity failure** → three complementary specificity controls (random-set, targeted `C_{e'}`, off-target scoring). Claim 2 partial if targeted control fails.
- **F7 — overlap-structure inversion** → report with bootstrap CI on the difference; Claim 2 partial on H2c if inversion.
- **F8 — length confound** → length audit + length-matched secondary (new).
