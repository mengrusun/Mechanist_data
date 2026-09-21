# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Causal Attribution / Ablation
chosen_idea_title: Verifying Locatability, Causality, and Applied Control of Emotion Circuits in Llama-3.2-3B
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/causal-attribution/ablation/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md

## Candidates

1. **[recommended]** Causal Attribution / Ablation — the load-bearing science lives here. `C_e` (the artifact underlying all three claims) is built by a causal ranker (Stage B: single-component additive-injection → target-prefix log-prob gain) and probed via mean-substitute ablation + additive activation-injection enhancement (M2); both are ablation-family interventions. Ablation-based attribution is the gold standard for the necessity/sufficiency verdicts C1 and C2 require. Cost: moderate (Stage B is the biggest single line item — per-shortlisted-component forward passes; shortlisting keeps it feasible).
   - path: skills/mechanism-skills/causal-attribution/ablation/SKILL.md

2. Probing / Residual Stream States — Stage A shortlist uses per-layer per-head linear probe AUC + neuron alignment/t-score (ITI-style) to rank layers, heads, and neurons cheaply *before* the Stage B causal ranker. Standardized cross-layer comparison under a fixed probe family. Cost: cheap (linear probes on frozen features).
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md

3. Representation and Parameter Analysis / Steering Vectors — appears twice: (a) the enhancement operator inside `C_e` — additive activation injection with per-head direction `d_{e,L}^h` and per-neuron scaled pre-activation delta at α ∈ {0.5, 1.0, 2.0} — is a per-component steering vector at head/neuron granularity; (b) M3 Arm C (the mechanism-family under contest against Arm A) is the classic RepE/CAA single-residual-stream direction with `α_C ∈ {0.5, 1.0, 2.0}` at `L ∈ top-3 layers`. Frozen pre-eval-split. Cost: cheap (one forward per generation).
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md

## Composition plan

Screen → decode → verify → recover, cost-noted:

1. **Screen (Probing / Residual Stream States)** — for each of 28 layers and each of 6 emotions, compute:
   - Residual-stream mean-diff direction `d_{e,L} = mean(residual[last-event-token] | positive-e) − mean(residual | off-target-uniform)` on train fold.
   - Per-head linear probe (ITI-style) AUC on paired contrasts, per `(L, h)`.
   - Per-MLP-neuron rank-combined score = z(cos⟨w_n^{out}, d_{e,L}⟩) + z(t-score of pre-activation contrast).
   Keep top-3 layers by summed head+neuron score; within them, shortlist top-20% heads by probe AUC and top-5% MLP neurons by combined score. Cost: **~0.5 GPU-h**.

2. **Verify — Stage B (Causal Attribution / Ablation, single-component enhancement)** — for each shortlisted component `c` (head or neuron) and each of 30 val stems per emotion:
   - Baseline forward: `log P(prefix_e | event_stem)`, `prefix_e = "I feel {emotion_word_e}"`.
   - Enhance-c-only forward at α2 = 1.0: for head, add `α · d_{e,L}^h` to that head's residual contribution; for neuron, add `α · sign(⟨d_{e,L}, w_n^{out}⟩) · std_train(activation_n | pos-e)` to that neuron's pre-activation.
   - `s_c = mean_val [log P(prefix_e | enhance c) − log P(prefix_e | baseline)]`.
   Rank shortlisted heads and neurons by `s_c`; pick top-`k_h` and top-`k_n`. `(k_h*, k_n*)` = ONE global choice from `{24, 48, 96} × {2000, 4000, 8000}` by macro-avg target-prefix log-prob gain on val. Cost: **~1.5 GPU-h** (dominant in M1).

3. **Verify — M2 causal + stability (Causal Attribution / Ablation)** — on the held-out eval fold:
   - **Ablation** (primary): mean-substitute every component in `C_e` with its per-neuron / per-head activation mean over the OTHER 5 emotion variants of the SAME stem. Δ target-prefix log-prob per emotion, paired-bootstrap 95% CI.
   - **Enhancement**: additive injection at α ∈ {0.5, 1.0, 2.0} on all of `C_e`; Spearman(α, Δ) dose-response.
   - **Specificity controls (three)**:
     (i) random-set null: 100 same-size random component sets per emotion at α2.
     (ii) targeted-`C_{e'}`: for each (e, e′), intervene on `C_{e'}` while scoring target `e`.
     (iii) off-target scoring: same intervention on `C_e`, score off-target continuations.
   - **Scenario stability**: fit `C_e^{S1}` on 5-of-10 train scenarios per domain, `C_e^{S2}` on the disjoint 5, Jaccard(S1, S2) vs. permutation-null CI.
   Cost: **~1.5 GPU-h**.

4. **Verify — M3 applied control (three-arm contest)** — held-out eval stems; identical decoding (`temperature=0, max_new_tokens=100`).
   - **Arm A (Circuit)** = Causal Attribution / Ablation enhancement operator (M2) applied at test time on `C_e` at `α_A` from a 9-cell val grid; `(k_h, k_n)` from a 3-cell neighborhood of `(k_h*, k_n*)` × 3 α.
   - **Arm B (Prompting)** = 3 templates × 3 injection positions (prefix/suffix/interleaved); 9 val configs.
   - **Arm C (Steering Vectors / RepE-CAA)** = residual-additive `d_e = mean_diff(residual[last-event-token], positive-e vs. off-target-uniform)` on train fold, PRE-EVAL-SPLIT-FROZEN; grid `L ∈ top-3 layers by probe AUC × α_C ∈ {0.5, 1.0, 2.0}`.
   - Hidden-target 6-way forced-choice external judge (gpt-5.4); 60-item gold-audit gate ≥ 0.75.
   Cost: **~2.0 GPU-h** (Llama) + **~1.5 GPU-h** (Qwen M4).

5. **Recover** — the pipeline does not need Circuit Discovery: the claim is "per-emotion sparse component SET at head+neuron granularity", not "minimal edge subgraph". The composition plan already recovers a component set that jointly meets ablation-sign + dose-response + specificity + stability predicates — sufficient for C1/C2/C3 without escalating to ACDC/EAP-IG.

**Downstream analysis (not a mechanism family — post-processing)**: Jaccard scoring, permutation-null draws, paired-bootstrap CIs, Spearman dose-response are all script-level statistics on already-collected effect vectors.

**Total revised cost**: ~5.5 GPU-h (Llama) + ~1.5 GPU-h (Qwen) + ~0.5 GPU-h (M0.5 judge audit) = **~7.5 GPU-h** — matches the plan's ~8.0 h Llama + 1.5 h Qwen estimate, well within the 10-h envelope.

## Plan reconciliation
<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->
- n_pairs: plan=2880 pairs (M1/M2/M3 collectively use full SEV at 480 events × 6 emotions with 10/5/5 scenario split — 1440 fit / 720 val / 720 eval) → **matches** — Ablation-family Stage B causal ranker is well-served by 30 val stems per emotion (180 val forward passes per component); M2 eval fold of 120 stems × 6 emotions gives adequate paired-bootstrap power; no re-bind needed.
- sites: plan=heads (per-head slice, `α · d_{e,L}^h` added to head contribution) + MLP neurons (pre-activation delta), Stage-A-shortlisted top-3 layers × top-20% heads × top-5% neurons → **matches** — Ablation submethod's granularity supports head- and neuron-level intervention; the shortlist + Stage-B causal ranker is the standard cost-mitigation for Ablation's O(n) scaling.
- metric: plan=target-prefix log-prob (`log P("I feel {emotion_word}" | event_stem)`) as the internal judge-free Stage-B score AND the M2 Δ score; per-emotion mean + paired-bootstrap 95% CI → **matches** — single-position scalar metric is exactly the "reduce to a single-position scalar before sweeping interventions" pattern the Ablation family SKILL.md recommends.
- gpu_hours: plan~8.0 h (Llama) + 1.5 h (Qwen) = 9.5 h → revised ~7.5 h — **revised down** — Ablation family's advertised bottleneck (per-shortlisted-component forward passes at Stage B) matches the plan's Stage-B estimate; shortlisting (top-20% heads + top-5% neurons in top-3 layers) keeps Stage B tractable. Kept plan margin for one re-run of the heaviest step.
reconciliation_status: ok

## Rationale

The plan is a *multi-family composition*, but the family that occupies the `chosen_family` slot must be the one where the load-bearing science lives — the one on which C1 (necessity of `C_e`), C2 (causal + specificity + stability), and C3 Arm A (applied control) all rest.

**Why Ablation is #1 (recommended):**
- `C_e` itself is defined by an ablation-family causal ranker (single-component enhancement → target-prefix log-prob gain). Without that causal ranker, `C_e` is just a correlational shortlist — Claim 1's "framework yields identifiable" claim collapses.
- C2's four rubric predicates (a-d) all measure Δ under a controlled intervention — this is textbook ablation-based attribution.
- C3's Arm A applies the same ablation-family operator at generation time.
- Ablation's advantage per its SKILL.md: "distinguishes essential mechanisms from features that are highly activated but causally irrelevant to the specific behavior" — this is exactly the load the specificity controls (random-null, `C_{e'}`, off-target) are designed to bear.

**Why Probing is #2 (cheap screen, not primary):**
- Probing furnishes Stage A's shortlist (per-head probe AUC + neuron alignment/t-score). It is the *screen*, not the *verify*, per the family SKILL.md's own limitation ("decodability is not causality"). Occupies the composition-plan step "screen".

**Why Steering Vectors is #3 (baseline, not verifier):**
- Steering Vectors is the *baseline* Arm C, the mechanism-family against which `C_e`-based control is contested. It is also the surface form of the per-component enhancement operator (per-head direction addition, per-neuron scaled delta), but the operator's role is causal intervention (Ablation-family), not steering per se. Reporting Arm C as Steering Vectors is the correct framing for the C3 pairwise (A > C) test.

**Composition freeze**: `family_freeze: pre-eval-split` — all three submethods above are locked before any eval-split activation is read. `(k_h*, k_n*)` is chosen ONCE globally on val, before any eval forward pass. Arm C direction is constructed on train fold, frozen before val. This closes the DoF flagged in `FINAL_PROPOSAL.md`.

**Aligned with tagging: yes** — the domain routing hint (`mechanistic-interpretability`) recommends Causal Attribution family for necessity/sufficiency claims, exactly matching the recommended pick.

No cross-round `families_already_settled` list — this is round 1 for this behavior + direction, no families excluded.
