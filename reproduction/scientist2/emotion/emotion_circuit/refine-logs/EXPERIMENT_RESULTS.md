# Initial Experiment Results

<!-- Metadata (parsed by /auto orchestrator). -->
phenomenon_status: n/a
committed_family: Causal Attribution / Ablation
chosen_idea_title: Verifying Locatability, Causality, and Applied Control of Emotion Circuits in Llama-3.2-3B
milestones_run: [M0.5, M1, M2, M3, M4]
main_model: Llama-3.2-3B-Instruct
verify_model: Qwen2.5-7B-Instruct
dataset: SEV (480 events × 6 emotions, scenario-level 10/5/5 split)
realized_gpu_hours: 4.3
resource_fidelity_marker: absent (cost-aware combination)
underpower_flag: tag

**Date**: 2026-07-13
**Plan**: refine-logs/EXPERIMENT_PLAN.md
**Routing**: refine-logs/MECHANISM_ROUTING.md (committed: `Causal Attribution / Ablation`)

## Data Actually Used

Per claim/block, reconciled against *planned* data in `EXPERIMENT_PLAN.md`. All milestones ran at full planned scale (`used_n = available_n`); no subsetting.

| Claim/Block | Provenance | Source | Available N | Planned Used N | Realized Used N | Subset note |
|---|---|---|---|---|---|---|
| M0.5 (data prep + judge audit) | existing + adapted (60 gold) | `/data/zhenqian/data/SEV/sev.json` + human-annotated 60-item gold | 2880 pairs + 60 gold | 2880 + 60 | 2880 + 60 (180 judgments; each gold item judged by 3 emotions) | — |
| C1 / M1 (Location) | existing (SEV) | same path | 2880 (480 stems × 6 emo) | 2880 (train 1440 fit + val 720 for Stage B); 3 folds × 80% train for Jaccard | 2880 pairs; Stage B on 30 val stems × 6 emo × shortlisted heads/neurons; 3 folds computed | — |
| C2 / M2 (Causal + Stability) | existing (SEV eval fold) | same path | 720 (120 eval stems × 6 emo) | 720 for ablation + 3 α × 720 enhancement + 100 random-null draws + C_{e'} 5×6 + S1/S2 refit | 720 (120/emo confirmed in `ablation.json`) | — |
| C3 / M3 (Applied) | existing (SEV eval fold) | same path | 2160 continuations (120 × 6 × 3 arms) + 19 440 val forward passes | 2160 eval + 19 440 val | **2160/2160 judged**; val grid 3×6×9 exhausted | — |
| C3 verify / M4 (Qwen swap) | existing (SEV held-out) | same path | 2160 (120 × 6 × 3 arms) on Qwen | 2160 | **2160/2160 judged**; Qwen (k_h*, k_n*) grid re-tuned on Qwen's own val | — |

**Realized `used_n` equals planned `used_n` in every row.** C2 / C3 negatives are therefore *real negatives at full plan scale*, not under-power artifacts (see Power-Fidelity check below).

## Results by Milestone

### M0.5: Data prep + judge audit — DONE (gate PASS)

- **Judge agreement**: **1.000** (180/180 gold-item forced-choice judgments).
- **Fallback classifier**: not triggered (agreement ≥ 0.75).
- **Scenario split**: 10/5/5 per domain, asserted disjoint.
- **Artifacts**: `runs/A0.5_dataprep_judge/{verdict,gold,split,judge_audit}.json`.

### M1 (C1: Localizability) — verdict = **supported**

- **Sparsity floor**: `(k_h*, k_n*) = (24, 2000)` — the *smallest* grid cell, well below the ceiling `k_h ≤ 96`, `k_n ≤ 8000`. `sparsity_floor_met: true`.
- **Stage-B macro gain at chosen (k_h*, k_n*)**: 2.136 nats target-prefix log-prob gain (well above the 0.01-nat weak-signal warning).
- **Per-emotion Jaccard vs. permutation-null 95% CI** (3-fold event-subsample stability; C_e^{fold} pairwise Jaccard):

| Emotion | Head Jaccard | Head null-CI upper | Head pass | Neuron Jaccard | Neuron null-CI upper | Neuron pass |
|---|---|---|---|---|---|---|
| joy | 1.000 | 0.277 | ✓ | 0.974 | 0.046 | ✓ |
| sadness | 0.947 | 0.275 | ✓ | 0.981 | 0.046 | ✓ |
| anger | 1.000 | 0.271 | ✓ | 0.978 | 0.046 | ✓ |
| fear | 1.000 | 0.265 | ✓ | 0.983 | 0.047 | ✓ |
| surprise | 0.947 | 0.266 | ✓ | 0.972 | 0.046 | ✓ |
| disgust | 1.000 | 0.265 | ✓ | 0.973 | 0.045 | ✓ |

- **Result**: Jaccard(fold, fold) is ~5× the null upper edge for heads and ~20× for neurons on every emotion. **6/6 pass on both axes**; success criterion (≥ 5/6) exceeded.
- **Random-set control** (`random_control_jaccard.json`): random top-k sets fall inside the null CI on all emotions, as expected.
- **Artifacts**: `runs/A1_location/{C_e,kstar,jaccard,metrics,directions.npz,shortlist,stageB_scores,layerwise,random_control_jaccard}.json/.npz`.

**Key stat**: heads Jaccard mean ≈ 0.98, neurons Jaccard mean ≈ 0.98, at (24, 2000) sparsity floor.

**Headline**: The circuit-locator framework yields sparse, per-emotion component sets `C_e` that are highly stable across event-subsample folds and clearly separated from random top-k controls. **Claim 1 (Localizability) is supported.**

---

### M2 (C2: Causal + Stability) — verdict = **not-supported**

Rubric verdicts (per plan Block 2 § Success criterion): **0 full / 0 partial / 0 causal-only / 6 not-supported**. Predicate (a) — causal-sign under ablation AND monotonic dose-response under enhancement — fails on **all 6 emotions**.

| Emotion | Ablation Δ (target, expect **<0**) | Enh α=1.0 Δ (expect **>0**) | Spearman(α, Δ) (expect ≥ 0.7) | (a) causal + dose | (b) off-target | (c) targeted C_{e'} | (d) S1/S2 stability | Verdict |
|---|---|---|---|---|---|---|---|---|
| joy | **+0.510** [+0.46, +0.56] | +1.567 | +1.00 | ✗ | ✓ | ✓ | ✓ | not-supported |
| sadness | **+0.855** [+0.77, +0.94] | +1.937 | +1.00 | ✗ | ✓ | ✓ | ✓ | not-supported |
| anger | **+0.867** [+0.81, +0.92] | +1.641 | **−1.00** | ✗ | ✗ | ✓ | ✓ | not-supported |
| fear | **+0.857** [+0.78, +0.94] | +5.828 | +0.50 | ✗ | ✓ | ✓ | ✓ | not-supported |
| surprise | **+0.928** [+0.85, +1.00] | +2.018 | **−1.00** | ✗ | ✓ | ✗ | ✓ | not-supported |
| disgust | **+0.617** [+0.53, +0.70] | +1.338 | **−0.50** | ✗ | ✗ | ✓ | ✓ | not-supported |

Two independent failure modes fire together:
1. **Ablation Δ has the wrong sign.** Mean-substituting `C_e` with its OTHER-5-emotion mean produces a **positive** target-prefix log-prob change on every emotion (0.51–0.93 nats), not the negative Δ the causal-necessity claim requires. The mean-substitute operator is *adding* signal instead of *removing* it — evidence the intervention semantics do not match the plan's intent.
2. **Enhancement is non-monotonic in α on 4/6 emotions.** Spearman(α, Δ_target) = 1.0 for joy and sadness only; anger and surprise show Spearman = −1.0 (α=0.5 produces the strongest Δ; α=2.0 flips negative for anger); fear = 0.5; disgust = −0.5. Additive-injection enhancement at α=2.0 saturates or destroys the signal on most emotions.

- **Off-target Δ (b)**: 4/6 emotions have `|Δ_off| < |Δ_target|` at α=1.0, but the reference `|Δ_target|` is spurious (see #1 above).
- **Targeted C_{e'} (c)**: 5/6 emotions have `|Δ_{C_e on e}| > |Δ_{C_{e'} on e}|` — a specificity signal survives, but it is a comparison between two operators both of which fail (a).
- **Scenario S1/S2 Jaccard (d)**: **6/6 pass** — the component *sets* are stable across scenario splits (heads Jaccard 0.45–1.00 vs. null upper edge ≤ 0.33; neurons Jaccard 0.49–0.91 vs. null ≤ 0.05). Locatability generalizes across scenarios; causality does not.
- **Random-set null**: `C_e` Δ is **below** the same-size random-set null on all 6 emotions (z ≪ 0). Random component sets produce *larger* target-prefix log-prob shifts than the causally-ranked `C_e` — additional evidence the additive/mean-substitute operators are producing generic activation-magnitude effects, not emotion-specific causal effects.
- **Artifacts**: `runs/A2_causal/{ablation,enhancement,enhancement_per_row,random_null,targeted_ce,scenario_stability,cross_emotion,metrics}.json`.

**Key stats**: verdict-counts = `{full: 0, partial: 0, causal-only: 0, not-supported: 6}`; `n_emotions_with_a_pass = 0/6`.

**Headline**: `C_e` components localise reliably (M1) and their component sets are scenario-stable (predicate (d) passes 6/6), but the tested causal-intervention operators (mean-substitute ablation, additive-injection enhancement) do **not** produce the emotion-specific Δ pattern C2 requires — ablation Δ has the wrong sign on every emotion and enhancement dose-response is non-monotonic on 4/6. **Claim 2 (Causal + Stable) is not-supported.** This is best characterised as: *components localised, but the tested causal-intervention operator does not causally shift emotion output*, not as evidence that `C_e` itself is irrelevant.

---

### M3 (C3: Applied) — verdict = **not-supported**

Three-arm forced-choice generation contest on 120 held-out eval stems × 6 target emotions × 3 arms = 2160 continuations, judged by `gpt-5.4` (agreement pre-gated at 1.0). Val hyperparameter selection ran on the full 3 × 6 × 9 = 162 val configs with 120 val stems each (19 440 val forward passes). Selected configs (`selected_configs.json`):

- **Arm A (Circuit)**: `(k_h=24, k_n=2000)`, α_A ∈ {0.5, 1.0, 2.0} chosen per emotion.
- **Arm B (Prompting)**: templates T1/T2 × prefix/suffix positions chosen per emotion.
- **Arm C (RepE/CAA steering)**: layer 2–4 (mostly L=4) + α_C = 2.0 chosen per emotion (except disgust: L=23, α=2.0).

**Per-emotion accuracy** (fraction correct out of 120 continuations; hidden-target 6-way judge):

| Emotion | Arm A (Circuit) | Arm B (Prompting) | Arm C (Steering) | A − B mean [95% CI] | A − C mean [95% CI] | A > B at 95% | A > C at 95% |
|---|---|---|---|---|---|---|---|
| joy | 0.025 | 0.933 | 0.425 | −0.908 [−0.958, −0.850] | −0.400 [−0.483, −0.317] | ✗ | ✗ |
| sadness | 0.017 | 0.758 | 0.192 | −0.742 [−0.817, −0.650] | −0.175 [−0.250, −0.100] | ✗ | ✗ |
| anger | 0.000 | 0.675 | 0.183 | −0.675 [−0.758, −0.592] | −0.183 [−0.250, −0.117] | ✗ | ✗ |
| fear | 0.208 | 0.608 | 0.550 | −0.400 [−0.517, −0.292] | −0.342 [−0.450, −0.217] | ✗ | ✗ |
| surprise | 0.042 | 0.492 | 0.075 | −0.450 [−0.542, −0.358] | −0.033 [−0.092, +0.025] | ✗ | ✗ |
| disgust | 0.025 | 0.550 | 0.717 | −0.525 [−0.617, −0.433] | −0.692 [−0.775, −0.600] | ✗ | ✗ |
| **Macro** | **0.053** | **0.669** | **0.357** | — | — | 0/6 | 0/6 |

- Success criterion: A > B on ≥ 5/6 AND A > C on ≥ 5/6 — **fails 0/6 on both pairwise tests**. Every 95% CI on both A−B and A−C excludes zero *in the wrong direction* (A is significantly worse).
- **Chance-floor diagnostic**: macro Arm-A accuracy = **0.053 is below the 1/6 ≈ 0.167 uniform-guess chance floor**. Two of six emotions (anger, joy) are effectively at 0 correct. Arm A produces continuations that the judge labels as *not the target emotion* far more often than random letter-parse would.
- **Length audit** (`length_audit`): mean words A=67.3, B=84.1, C=85.1. Max pairwise ratio 1.26 (< 1.15 threshold not met but not enormous; unlikely to explain a 60-percentage-point gap).
- **Artifacts**: `runs/A3_applied/{arm_A_val,arm_B_val,arm_C_val,selected_configs,eval_generations,judge_results,metrics}.json`.

**Key stats**: macro `A = 0.053`, `B = 0.669`, `C = 0.357`; `n_A_gt_B_at_95 = 0/6`, `n_A_gt_C_at_95 = 0/6`; `A_gt_B_pass = false`, `A_gt_C_pass = false`.

**Headline**: The circuit-based control arm (Arm A) not only fails to beat prompting (Arm B) and single-direction steering (Arm C) — it performs **catastrophically worse than both, and below the 1/6 uniform-chance floor**. Arm A's macro accuracy (5.3 %) is a strong signal that the additive-injection enhancement operator applied to `C_e` is *destroying meaningful generation* rather than steering it: the intervention pushes the model into off-distribution states rather than into the target emotion. This is evidence the **intervention semantics may be wrong** (over-strong α, wrong site set, wrong operator formulation), not evidence that the underlying `C_e` components are irrelevant. Prompting (66.9 %) and steering (35.7 %) both produce coherent emotion-conditioned generation on this dataset, so the model *can* express emotions on demand — the circuit-injection pathway just isn't the way to do it here. **Claim 3 (Applied control) is not-supported.**

---

### M4 (C3 robustness / verify swap on Qwen) — verdict = **conditional**

Same three-arm protocol re-run on Qwen2.5-7B-Instruct, with (k_h, k_n, α) freshly re-tuned on Qwen's own val fold (`qwen_selected_configs.json`).

| Emotion | Arm A | Arm B | Arm C | A − B mean | A − C mean |
|---|---|---|---|---|---|
| joy | 0.000 | 0.983 | 0.092 | −0.983 | −0.092 |
| sadness | 0.092 | 0.967 | 0.100 | −0.875 | −0.008 |
| anger | 0.008 | 0.975 | 0.175 | −0.967 | −0.167 |
| fear | 0.200 | 1.000 | 0.108 | −0.800 | **+0.092** |
| surprise | 0.150 | 0.958 | 0.267 | −0.808 | −0.117 |
| disgust | 0.008 | 0.933 | 0.008 | −0.925 | 0.000 |
| **Macro** | **0.076** | **0.969** | **0.125** | — | — |

- A > B: **0/6**; A > C: **0/6** (relaxed 4/6 verify threshold not met).
- Auto-verdict labelled `conditional` by the script (`verify_verdict: "conditional"`) — but qualitatively the pattern is the **same failure mode** as Llama: Arm A near-zero accuracy, prompting near-ceiling. Qwen's Arm C is even weaker than Llama's (12.5 % vs 35.7 % macro).
- **Artifacts**: `runs/A4_verify_qwen/qwen_{C_e,kstar,shortlist,directions.npz,arm_A_val,arm_B_val,arm_C_val,selected_configs,eval_generations,judge_results,metrics}`.

**Key stats**: macro `A = 0.076`, `B = 0.969`, `C = 0.125`; `A_gt_B_pass = false`, `A_gt_C_pass = false`; `verify_verdict = "conditional"`.

**Headline**: The M3 failure mode reproduces on Qwen2.5-7B-Instruct with independently re-tuned hyperparameters — Arm A macro accuracy (7.6 %) again falls below the chance floor, while prompting saturates near 97 %. The negative on Claim 3 is **not Llama-specific**; the additive-injection circuit operator fails on both architectures. The `conditional` script label reflects the relaxed 4/6 verify threshold, not evidence of a qualitatively different outcome.

---

## Summary

| Claim | Milestone | Verdict | Key stat | 1-line headline |
|---|---|---|---|---|
| C1 (Localizability) | M1 | **supported** | Jaccard heads/neurons 6/6 above perm-null 95% CI; sparsity floor met at (24, 2000) | Framework yields stable, sparse, per-emotion component sets. |
| C2 (Causal + Stability) | M2 | **not-supported** | 0/6 emotions clear predicate (a); ablation Δ has wrong sign on all 6; enhancement Spearman < 0.7 on 4/6 | Components localise and scenario-stable, but the tested causal operator does not shift emotion output. |
| C3 (Applied) | M3 (+ M4 verify) | **not-supported** | Macro A=0.053 (below 1/6 chance), B=0.669, C=0.357; A>B 0/6, A>C 0/6 | Circuit-injection destroys generation; prompting wins ~13× over circuit. Reproduces on Qwen. |

- **6/6 milestones (M0.5, M1, M2, M3, M4) complete at full plan scale** (`used_n = available_n`; judge gate PASS at 1.0).
- **1/3 claims supported** (C1); 2/3 not-supported at full scale (C2, C3).
- **Ready for /auto-verify**: **YES** — the negative results on C2/C3 are honest, at plan scale, and worth stress-testing (variant swaps of method/dataset/model). C1 also merits verify.

## Power-Fidelity Check (UNDERPOWER = tag)

Cost-aware combination (`resource_fidelity: strict` absent). Realized scale vs plan:

| Claim | Verdict | Planned used_n | Realized used_n | Realized seeds/grid | Suspected under-power? |
|---|---|---|---|---|---|
| C1 | supported | 2880 pairs + 3 folds | 2880 pairs + 3 folds | (k_h, k_n) full 3×3 grid | — |
| C2 | not-supported | 720 eval + 100 random-null draws + C_{e'} 5×6 + S1/S2 | 720 eval + 100 random-null + C_{e'} 5×6 + S1/S2 (all logged in `runs/A2_causal/*.json`) | 3 α levels, full | **suspected_under_power: false** — plan ran as written; not an under-power artifact |
| C3 | not-supported | 2160 eval + 19 440 val | 2160/2160 eval judged + 19 440 val exhausted | 3 arms × 9 configs full | **suspected_under_power: false** — plan ran as written; not an under-power artifact |

Neither C2 nor C3 is flagged `suspected_under_power`. Both are **genuine negatives at full plan scale** — the plan's declared power did not save the claims. The failure mode for C2/C3 lives in *what was intervened on and how*, not in *how much* was measured. This is the kind of negative that verify-stage stress tests (method swap, dataset swap, model swap) are designed to interrogate.

## Next Step

→ **`/auto-verify`** to stress-test all three claims (variant swaps of method / dataset / model). Both the C1 supported verdict and the C2/C3 not-supported verdicts warrant robustness testing; the C3 chance-floor pattern in particular deserves attention from verify's method-swap arm.
