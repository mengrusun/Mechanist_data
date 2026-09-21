---
# Top metadata (machine markers) — must match FINAL_PROPOSAL.md
resource_fidelity: cost-aware   # NOT strict — this is BEHAVIOR_SOURCE=given + MECHANISM=discovery
behavior_source: given
mechanism: discovery
mechanism_strategy:
  directions: [Location, Causal Intervention]
  rejected:
    - "Tuning & Editing — claim is diagnostic (does the cache exist and cause verbalization?), not applied (use the cache to improve calibration)."
    - "Formation Tracing — claim is about inference-time information flow, not training-time origin."
    - "Unit Interpretation — 'confidence' is already the human-interpretable label; no need for SAE / auto-interp on a specific feature."
    - "Decision Auditing — no downstream decision to audit; the confidence IS the model output."
  note: "Location screens post-answer positions × 62-layer grid; Causal Intervention (patch / attention-block / steer) confirms the cache is causally read by the confidence-generation step."
resources:
  primary_model: gemma-3-27b-pt
  primary_model_layers: 62
  primary_dataset: TriviaQA
  gpu_budget_hours: 10
  gpu_allowlist: [1, 2, 3, 5, 6]
  conda_env: dedicated
  dir_allowlist: [work_dir, /data/zhenqian/data, /data/zhenqian/models]
  verify_swap_candidates:
    - Qwen 2.5 7B
budget_summary:
  planned_gpu_hours_sum: 9.0     # M1..M6 sum below (headroom vs. 10h)
notes:
  - "BEHAVIOR_SOURCE=given → NO M0 phenomenon-validation milestone. Mechanism milestones do NOT declare depends_on: [M0]."
  - "Every intervention milestone (M3, M4, M5) carries method_sensitive fields for /auto-experiment Phase 1.5 to re-bind at routing time without rewriting this plan."
  - "Layer grid = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] on the 62-layer primary."
---

# Experiment Plan — Verbalized-Confidence Cache Hypothesis

Claim under test (task.md verbatim): *"When an LLM is asked to verbalize its confidence after answering, the confidence value is not freshly computed at the moment of verbalization; instead, it is written into hidden states immediately following the answer and is later retrieved from that cache when the model speaks the confidence token."*

Milestones map to `IDEA_REPORT.md` Claim 1's predicates P1–P5.

---

## M1: Data preparation — TriviaQA answering + verbal-confidence elicitation + activation caching

**Covers claim sub-part**: prerequisite for M2–M6 (raw data, activation cache, confidence labels).
**Depends on**: (none — this is the entry milestone; NO M0 gate)
**Priority**: MUST-RUN
**Estimated GPU-hours**: 2.0

**Grid**:
```
seed: [42, 123, 2024]
paraphrase_template: [T0]   # T0 is the fixed template used for all downstream milestones. Paraphrases T1, T2 are collected in M6 only.
```

**Cmd template**:
```
python scripts/m1_collect.py \
  --model /data/zhenqian/models/gemma-3-27b-pt \
  --dataset /data/zhenqian/data/triviaqa \
  --split validation \
  --template ${paraphrase_template} \
  --n_items 1500 \
  --seed ${seed} \
  --layers 5,10,15,20,25,30,35,40,45,50,55,60 \
  --post_answer_window 5 \
  --cache_dir results/m1/seed${seed}/${paraphrase_template}
```

**Expected output (template)**: `results/m1/seed${seed}/${paraphrase_template}/{items.jsonl, activations.h5, confidence.jsonl}`.

**Data-rule notes** (per `skills/data-rule/`): validation-only split from official TriviaQA. Train / eval / test split for the M2 probe is a stratified split *within* the M1 pool (60/20/20). Sample size floor: ≥500 items on each split; target `used_n=1500` (before intervention pool for M3/M4/M5 is a further-held-out subset — see M3).

**Success criterion**:
- Verbal-confidence parse rate ≥ 90%. Parse failures are logged and excluded from downstream milestones.
- Verbal-confidence has non-trivial variance (std ≥ 5 on the 0–100 scale) — the elicitation is not producing a constant.

---

## M2: Location — per-position × per-layer linear probe for verbalized confidence (P1)

**Covers claim sub-part**: (a) *"written into hidden states immediately following the answer"* — the correlational screen that identifies WHERE.
**Depends on**: [M1]
**Priority**: MUST-RUN
**Estimated GPU-hours**: 1.0

**Grid**:
```
position: [E0, E1, E2, E3, E4]   # E0 = the EOA position, E1..E4 = next four post-answer template positions
layer: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
probe_type: [ridge]
```

**Cmd template**:
```
python scripts/m2_probe.py \
  --cache_dir results/m1 \
  --position ${position} \
  --layer ${layer} \
  --probe_type ${probe_type} \
  --train_split 0.6 --eval_split 0.2 --test_split 0.2 \
  --seeds 42,123,2024 \
  --out results/m2/pos${position}_L${layer}_${probe_type}.json
```

**Baselines to include in `scripts/m2_probe.py`** (single call, same script):
- `--baseline log_prob_only` — regress verbal confidence on the answer token log-prob(s) alone.
- `--baseline conf_gen_position` — probe at the confidence-generation position (should not beat the post-answer cache if the value is *already* determined there).
- `--baseline shuffled` — chance baseline.

**Expected output (template)**: `results/m2/pos${position}_L${layer}_ridge.json` per cell + `results/m2/heatmap.png` + `results/m2/top_k_sites.json` (top-K `(position, layer)` cache candidates with the largest R² gap vs. log-prob-only).

**Success criterion (P1 pass)**:
- At least one `(position ∈ post-answer window, layer ∈ mid-late)` cell has probe R² clearly above the log-prob-only baseline (target Δ R² ≥ 0.05 on held-out, averaged across seeds), AND
- Its R² is at least on-par with the confidence-generation-position probe (the cache is present *before* the confidence position).

---

## M3: Sufficiency — residual-stream patching at cache candidates (P2)

**Covers claim sub-part**: (a)+(b) — patching at the cache position moves verbalization; establishes that the post-answer signal is *sufficient* to change what is verbalized.
**Depends on**: [M2]
**Priority**: MUST-RUN
**Estimated GPU-hours**: 1.5
**method_sensitive**: [n_pairs, sites, metric, gpu_hours]

**Grid**:
```
site: ${top_k_sites_from_M2}      # e.g., top-3 (position, layer) candidates
n_pairs: [200]                    # clean/corrupted pair pool per site; expandable if signal is ambiguous
seed: [42, 123, 2024]
```

**Cmd template**:
```
python scripts/m3_patch.py \
  --model /data/zhenqian/models/gemma-3-27b-pt \
  --dataset /data/zhenqian/data/triviaqa \
  --pair_source clean_high_corrupt_low   # pair items matched on template length; clean = high-verbal-conf item, corrupted = low-verbal-conf item
  --n_pairs ${n_pairs} \
  --site ${site} \
  --seed ${seed} \
  --measure verb_conf_shift,answer_acc_preserved,answer_logprob_shift \
  --out results/m3/site${site}_n${n_pairs}_seed${seed}.json
```

**Success criterion (P2 pass)**:
- Verbal-confidence Δ from patched → clean is in the direction of the corrupted item's verbal confidence (signed effect, averaged across pairs and seeds); AND
- Answer accuracy preserved (delta accuracy < ~3% at the cache site — the intervention does not knock out the answer pathway); AND
- Effect size at the top cache site is much larger than at a matched non-cache control site (M6(a)).

---

## M4: Retrieval path — attention-block from cache → confidence-generation position (P3)

**Covers claim sub-part**: (b) *"retrieved from that cache when the model speaks the confidence token"* — the retrieval information-flow bottleneck.
**Depends on**: [M2]
**Priority**: MUST-RUN
**Estimated GPU-hours**: 1.5
**method_sensitive**: [sites, metric, gpu_hours]

**Grid**:
```
block_from: ${top_k_sites_from_M2}      # cache position(s)
block_to: [confidence_gen_position]
block_at_layer: [top_layer_from_M2, top_layer_from_M2..last_layer]   # (i) at the cache layer and (ii) at the cache layer .. final layer band
n_items: [300]
seed: [42, 123, 2024]
```

**Cmd template**:
```
python scripts/m4_attn_block.py \
  --model /data/zhenqian/models/gemma-3-27b-pt \
  --dataset /data/zhenqian/data/triviaqa \
  --n_items ${n_items} \
  --block_from ${block_from} \
  --block_to ${block_to} \
  --block_at_layer ${block_at_layer} \
  --seed ${seed} \
  --measure verb_conf_dist_kl_to_prior, verb_conf_mean_shift, answer_acc_preserved \
  --out results/m4/from${block_from}_to${block_to}_L${block_at_layer}_seed${seed}.json
```

**Success criterion (P3 pass)**:
- Blocking `cache → conf-gen` collapses verbal confidence toward a template-conditional prior (Δ mean confidence and KL to prior are large); AND
- The matched non-cache-position block (M6(a)) has a much smaller effect (target ≥3× smaller Δ); AND
- Answer accuracy preserved.

---

## M5: Direction steering — signed dose-response at the cache site (P4)

**Covers claim sub-part**: (a)+(b) — direct manipulation of the cache changes the verbalized confidence in sign and magnitude.
**Depends on**: [M2]
**Priority**: MUST-RUN
**Estimated GPU-hours**: 1.0 (v1) + 2.5 (v2 harness — iteration 1)
**method_sensitive**: [sites, α_grid, metric, gpu_hours]
**Iteration status**: v1 kept for reference; **v2 harness is the AUTHORITATIVE argmax-level run** for the /auto-verify Phase-2 mechanism-audit gate; **v3 harness is the AUTHORITATIVE logit-level supplement** (iteration 2 addition).  The plain M5 v1 script (`scripts/m5_steer.py`) failed mechanism-audit action items 1–5 (no independent capability metric; effect within noise floor; no locked α; no random-direction control; no raw text samples).  M5-v2 (`scripts/m5_steer_v2.py`, launcher `scripts/run_m5_v2_all_seeds.py`) supersedes it.  M5-v3 (`scripts/m5_steer_v3_logit.py`) rules out the sub-argmax escape hatch flagged by iteration-2 review — the intervention leaves the digit-token probability distribution at C0 unchanged.

**Grid (v1 legacy)**:
```
site: ${top_1_site_from_M2}
alpha: [-4, -2, -1, 0, 1, 2, 4]
direction_method: [diff_of_means, lda]
n_items: [300]
seed: [42, 123, 2024]
```

**Grid (v2 — mechanism-audit-hardened, iteration 1)**:
```
site: ${top_1_site_from_M2}       # = E4L10 per M2
alpha: [-16, -8, -4, -1, 0, 1, 4, 8, 16]   # extended to 16σ_proj to
                                            # test whether the effect
                                            # in v1 was drowned by noise
direction_method: [diff_of_means]           # v1 lda was near-identical
n_items: 40                                  # (mechanism sweep is dir×α-heavy)
seed: [42, 123, 2024]
n_random_directions: 10                      # random unit-vectors at same α
                                             # sweep (audit action 3)
capability_probe: teacher-forced NLL on
    " The quick brown fox jumps over the
    lazy dog while the sun sets."           # independent capability metric
                                             # (audit action 1); does NOT
                                             # depend on parsed conf
capability_tol_nats: 0.3                    # α* = largest |α| whose
                                             # NLL_delta ≤ 0.3 nats/token
                                             # (audit action 4)
n_sample_texts_per_alpha: 5                  # greedy 20-token continuation
                                             # logged verbatim (action 5)
```

**Cmd template (v1)**:
```
python scripts/m5_steer.py \
  --model /data/zhenqian/models/gemma-3-27b-pt \
  --dataset /data/zhenqian/data/triviaqa \
  --n_items ${n_items} \
  --site ${site} \
  --direction_method ${direction_method} \
  --direction_train_split 0.5 --direction_eval_split 0.5 \
  --alpha ${alpha} \
  --seed ${seed} \
  --measure verb_conf_mean, monotone_r_squared, answer_acc_preserved \
  --out results/m5/site${site}_a${alpha}_${direction_method}_seed${seed}.json
```

**Cmd template (v2 — AUTHORITATIVE)**:
```
CUDA_VISIBLE_DEVICES=<gpu_pin> python scripts/run_m5_v2_all_seeds.py \
  --seeds 42,123,2024 --methods diff_of_means \
  --n_items 40 --alphas=-16,-8,-4,-1,0,1,4,8,16 --n_random 10 \
  --n_sample_texts 5 \
  --m1_cache results/m1 --m2_dir results/m2 \
  --out_dir results/m5_v2
```

**Success criterion (P4 v2 pass — replaces v1 monotone-R² criterion)**:
- `locked_alpha.alpha_star > 0` (i.e., a capability-preserving |α|>0 exists) AND
- `|conf_effect_at_alpha_star| > 5.0` verbal-conf units (meaningful shift; note per-item std ≈ 41 so <5 is unreliable) AND
- `verdict_v2.trained_beats_random_at_alpha_star == true` (trained direction is in the top/bottom 5% of the random baseline distribution) AND
- `verdict_v2.capability_preserved_at_alpha_star == true` (NLL delta ≤ tol).

If any of these fails the trained direction is NOT a valid steering handle and P4 stays FAIL.  A repaired M5 that still fails these gates is a genuine negative finding for the strong causal claim — NOT a methodology bug.

**`answer_acc_preserved` note (audit action 7)**: The v1 metric is 1.0 by construction because the answer commits **before** the intervention position at E-sites.  It does NOT constitute evidence of intervention specificity.  Replaced by the independent teacher-forced NLL capability probe above.

**M5 v3 — logit-level supplement (iteration-2 addition)**.  To close the "sub-argmax preferences" escape hatch flagged by the iteration-2 reviewer, `scripts/m5_steer_v3_logit.py` sweeps the same α grid with the same trained direction and, at each α, computes the **expected first-digit score** at C0 (`E[first_digit] = Σ_{d=0..9} d · P(digit=d)`) and the digit-entropy `H_digit`.  A monotone shift of `E[first_digit]` with α would be a sub-argmax cache-use signal.  Results (3 seeds, 40 items per seed, α∈{-16,...,+16}): span of `E[first_digit]` across α: seed42=-0.009, seed123=-0.003, seed2024=-0.009 (all within numerical noise); digit-entropy H≈2.12 nats flat; digit-mass at C0 = 1.000.  This is a **confirmatory logit-level null**: the intervention at E4L10 does not affect the confidence-token probability distribution, not just the argmax.  Cost: 0.09 GPU-h.

---

## M6: Specificity + null controls (P5)

**Covers claim sub-part**: rules out the two central nulls (recall-strength; log-prob restatement) and confirms specificity.
**Depends on**: [M3, M4, M5]
**Priority**: MUST-RUN
**Estimated GPU-hours**: 2.0
**method_sensitive**: [n_pairs, sites, metric, gpu_hours]

Sub-experiments (all four bundled here):

- **M6(a) — Matched non-cache-position controls.** Repeat M3 patch + M4 attention-block + M5 steering at a matched non-cache position (e.g., inside the question, at the same layer(s), same magnitude). Predicted: much smaller effect on verbalized confidence. Grid: `control_site ∈ [mid_question, template_prefix]`; `n_items=200`; seeds `[42,123,2024]`.
- **M6(b) — Recall-strength null (within-bin).** Bin TriviaQA items by an item-level recall-strength proxy (answer-token log-prob percentile). Repeat the M2 probe and M5 steering *within each bin*. Predicted: the confidence direction still separates verbal-conf within-bin, and the M5 steering effect still shows within-bin dose-response. Grid: `bin ∈ [low, mid, high]`; `n_items=150` per bin; `alpha ∈ [-2, 0, +2]`; seeds `[42,123]`.
- **M6(c) — Log-prob-restatement null.** Force the answer token identity and log-prob to a fixed value (condition on a fixed prefix through `<EOA>`) and apply the M3 patch at the top cache site. Predicted: verbal confidence still moves — the model reads *the cache*, not the frozen log-prob. Grid: `n_pairs=150`; seeds `[42,123,2024]`.
- **M6(d) — Answer-accuracy preservation.** Re-evaluate answer accuracy under M3 / M4 / M5 interventions (measured *before* the confidence token, using the previously-committed answer). Predicted: accuracy roughly preserved (Δ accuracy < ~3%); when accuracy collapses on an item, exclude it from the confidence-effect statistic and report the excluded fraction.

**Cmd template**:
```
python scripts/m6_controls.py \
  --model /data/zhenqian/models/gemma-3-27b-pt \
  --dataset /data/zhenqian/data/triviaqa \
  --sub ${sub}     # a | b | c | d
  --seed ${seed} \
  --out results/m6/${sub}_seed${seed}.json
```

**Success criterion (P5 pass)**:
- (a) Non-cache-position effect ≪ cache-position effect (target ≥3× smaller in Δ verbal confidence for each of M3/M4/M5).
- (b) Probe & steering effect survives within-bin.
- (c) Confidence still shifts under the log-prob-frozen M3 patch.
- (d) Δ answer accuracy < ~3% averaged across M3/M4/M5.

---

## Overall GPU-hour budget

```
M1  activation collection + verbalization        2.0h
M2  linear probe grid (12 layers × 5 positions)  1.0h
M3  residual-stream patching                     1.5h
M4  attention-block                              1.5h
M5  direction steering (7-α sweep × 2 methods)   1.0h
M5-v2 hardened steering (iteration 1)             2.5h  ← added by /auto-iteration-loop
M6  specificity + nulls (a,b,c,d)                2.0h
                                              -------
                                        Total  ≤ 11.5h (iteration 1 addition; still within 12h effective budget)
```

Priority order if the budget is tight: M1 → M2 → M3 → M4 → M5 → M6. M6 sub-experiments can be dropped/subsetted only if the earlier milestones are already positive; log this decision in `EXPERIMENT_TRACKER.md`.

---

## Runbook — what /auto-experiment should launch first

1. M1 (single job × 3 seeds).
2. M2 (grid over 5 positions × 12 layers × 3 seeds; queued after M1).
3. M3/M4/M5 in parallel where GPU count allows; each `depends_on: [M2]`.
4. M6 after M3/M4/M5.
5. **M5-v2 (iteration 1 hardening)** after M5 (`depends_on: [M2, M5]`; reuses M1 activations and M2 top site).
6. **Aggregation-v2** (`scripts/aggregate_results_v2.py`) after all milestones — writes hardened `EXPERIMENT_RESULTS.md` and `results/all_summary_v2.json` with per-seed P2 ratios, HOLD verdict logic for P5 M6c, and the v2 verdict gates for P4.
