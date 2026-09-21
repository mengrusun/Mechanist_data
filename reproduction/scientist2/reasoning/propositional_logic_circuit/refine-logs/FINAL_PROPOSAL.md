# Final Proposal — Verifying a Sparse, Modular Circuit for Propositional-Logic Reasoning in Mistral-7B

**Behavior-source**: given
**Mechanism**: discovery
**mechanism_strategy**:
```yaml
mechanism_strategy:
  directions: [Location, Causal Intervention]
  rejected:
    - Tuning & Editing — no editing objective in the reproduction; would inflate compute without adding evidence.
    - Formation Tracing — reproduction operates on the final pretrained checkpoint, not the training trajectory.
    - Unit Interpretation — SAE / dictionary decomposition for Mistral-7B on this task would require an SAE training pass beyond the 10-GPU-hour envelope; noted as follow-up.
    - Decision Auditing — orthogonal to the three sub-claims (trust / spurious-feature audit is a different question).
  note: The three sub-claims are strictly a "mechanistic evidence" question — locate a sparse component set, then causally intervene to test necessity and sufficiency. Location supplies the shortlist, Causal Intervention promotes located → mechanism.
```

**resource_fidelity**: not-strict (`MECHANISM=discovery`; the strict harness only stamps when both axes are `given`). Resources named below are the user's *preferred* scale from `task.md`; the plan uses them at full scale except where a *scientifically* justified cheap-screen substitutes (attribution patching for the Location screen).

**Problem Anchor** (do not drift): *Does the internal computation Mistral-7B performs on a synthetic propositional-logic prompt (facts + rules → truth-value query) implement the task via **(C1) a sparse subset** of attention heads + MLP components that (C2) **decompose into modular sub-circuits** for fact identification, rule application, and answer projection, and are (C3) both **necessary and sufficient** by activation patching?* — the anchor is the three sub-claims verbatim from `task.md`; no drift, no widening.

---

## 1. Method thesis (one sentence)

Verify the three sub-claims on Mistral-7B with a single unified activation-patching pipeline over a controlled synthetic propositional-logic template — a cheap attribution-patching screen to localise the circuit (C1), path-patched clean↔corrupted interventions to test necessity + sufficiency (C3), and role-specific matched-corruption controls to test modular dissociation (C2) — then replay a compressed version on Gemma-2-9B to check cross-family schema recurrence.

## 2. Dominant contribution

A rigorous, budget-aware **reproduction** and **cross-family generalisation test** for the sparse-modular circuit hypothesis on propositional-logic reasoning in a 7B-scale open LLM, delivering a per-component role-assignment matrix, a completeness/minimality assessment, and dose-response curves for the three claims — all under 10 GPU-hours.

## 3. Complexity intentionally rejected

- **SAE / dictionary-based feature discovery on Mistral-7B**: would require SAE training on the residual stream (~10–20 GPU-h alone), far outside the 10-h budget. The hypothesis is at the component (head/MLP) level, not the feature level — reject.
- **Gemma-2-27B verify swap**: task.md permits "use as needed, not necessarily all". At 27B and 80 GB A800 GPUs, a full activation-patching sweep at k=2–3 rule-chain-length with 500 clean/corrupted pairs consumes ≥ 3–4 GPU-h alone. Made *contingent* on remaining budget after Mistral + Gemma-2-9B main runs (M1–M5) close.
- **Training-time analysis (formation tracing)**: reject (see rejected mechanism directions above).
- **Full ACDC iterative pruning**: replaced by the faster **attribution-patching + edge-patching** cheap screen (Syed et al. 2024, arXiv 2310.10348), reserving full path patching only for the confirmed shortlist.
- **New synthetic prompt formats**: reject drift toward larger-scope multi-domain templates; a *single* fixed template family (with parameterised k, rule-chain length, distractor count, lexical realisation) is enough for all three claims per task.md.

## 4. Method — unified verification pipeline

### 4.1 Synthetic propositional-logic template (dataset)

A single template family, parameterised so *only* one causal axis differs between clean and corrupted:

```
[System] You are given some facts and rules. Answer whether the query is True or False.

[Facts]  {name_i} is {property_i}.        (i = 1..k)
[Rules]  If X is {P1} then X is {P2}.      (m rules, m ∈ {1, 2, 3} for chain length)
[Query]  Is {name_target} {property_target}?

[Answer] True | False
```

- **k (fact count)**: 2, 3, 5, 8, 12 — controls distractor load.
- **Rule-chain length**: 1, 2, 3 — controls compositional depth.
- **Lexical realisation**: 3 lexicon sets (natural-English properties like "wise", "hungry"; symbol-abstract properties like "P" / "Q"; alternate-language noun-phrases) — controls the surface-form nuisance.
- **Boolean balance**: 50% True / 50% False; both truth values equally represented per (k, chain-length, lexicon) cell.

**Clean / Corrupted pair matching (four corruption axes)** — each isolates one causal role:
- **Fact-swap corruption**: swap one fact so the query no longer follows (isolates fact-identification heads).
- **Rule-swap corruption**: replace an implication rule so the chain no longer resolves the query (isolates rule-application heads).
- **Answer-token corruption**: replace the final "True"/"False" positional cue (e.g. flip the answer template) while keeping the derivation identical (isolates answer-projection heads).
- **Neutral corruption (matched control)**: a size-matched but functionally-irrelevant swap (a distractor fact's name changed) that should NOT change the answer.

**Sample size**:
- **Main circuit-discovery pool** (Mistral-7B, C1/C3): 500 clean-corrupted pairs at k=3, chain-length=2 (default cell). This is the anchor cell.
- **Role-dissociation pool** (Mistral-7B, C2): 3 × 300 = 900 pairs, one per fact-swap / rule-swap / answer-swap corruption.
- **Cross-cell stability** (C2 stability check): 200 pairs each at (k=5, chain=2) and (k=3, chain=3) — repeat C2 metric.
- **Cross-family** (M5): 500 pairs on Gemma-2-9B (anchor cell); Gemma-2-27B only if budget remains.

Total prompt pool: ~3000 clean + 3000 corrupted matched pairs. Constructed programmatically at experiment stage; stored under `${DATA_DIR}/prop_logic_synth/` (created by the dataset builder).

### 4.2 Location screen (C1)

**Cheap first pass**: **Attribution patching** (Syed et al. 2024, arXiv 2310.10348) over all `L × H` attention heads and `L × 1` MLPs — one gradient-based backward pass per pair — to produce a ranked score per component.

**Shortlist rule**: keep top-K components until `cumulative_effect ≥ 0.9 × total_effect` (task-preservation criterion). If K > 0.15 × (L·H + L), the circuit is *not sparse* and Sub-claim 1 is refuted — the plan reports that verdict and continues to C2/C3 anyway (results are still informative).

**Completeness / minimality verification (Circuit-Hypothesis Testing, arXiv 2410.13032)**:
- **Completeness**: restrict Mistral-7B's forward pass to only the shortlisted components (ablate everything else via **resample ablation** — activations drawn from a pool of unrelated prompts, NOT zero-ablate per Best-Practices). Measure accuracy retention.
- **Minimality**: remove one component at a time from the shortlist; measure accuracy drop. A minimal circuit shows a large drop when any *single* member is removed.
- **Localisation score**: fraction of full-model behavior captured, per Hypothesis-Testing paper.

### 4.3 Necessity + sufficiency (C3)

**Necessity — clean → corrupted denoising patch**:
- For each corrupted prompt, replace the shortlisted components' activations with the corresponding clean-prompt activations.
- Measure: `Recovery = (behavior_at_corrupt+patch − behavior_at_corrupt) / (behavior_at_clean − behavior_at_corrupt)`, where `behavior` is one of `{logit_diff, prob_diff, KL}`.
- Success: `Recovery ≥ 0.8` on all three metrics.

**Sufficiency — reinsertion patch**:
- Take a clean prompt. Ablate (via resample) *everything except* the shortlisted components — i.e., corrupt the residual stream at every non-shortlisted site, then reinsert clean activations only at the shortlist.
- Success: `Sufficient-recovery ≥ 0.8` on all three metrics.

**Specificity control**:
- Randomly-drawn same-size non-shortlist component set; repeat both patches. Success: control `Recovery ≤ 0.2`; specificity gap `≥ 0.6`.

**Dose-response**:
- Recovery as a function of |patched components|, sweeping from 0 to full shortlist size in ~5 steps. A "genuine" mechanism shows a monotone increase; a heuristic account shows step / plateau behavior.

### 4.4 Modular decomposition (C2)

**Per-role effect on the shortlisted components**:
- For each of the three role-corruption pools (fact-swap / rule-swap / answer-swap), run activation patching on *each* shortlisted component individually.
- Assemble the **role-assignment matrix** `S ∈ [0,1]^{|C|×3}` where `S[c, r] = Recovery_r(c)` (single-component clean-into-corrupted patch under role-r corruption).

**Modularity criterion (all three must hold)**:
- **Block-sparse structure**: for each c, `max_r S[c,r] ≥ 2 × second_max_r S[c,r]` (dominance ratio).
- **Positive block-average dissociation**: `d = mean(S[c ∈ C_r, r]) − mean(S[c ∈ C_r, r' ≠ r]) ≥ 0.1` for each of the three roles r.
- **Cross-cell stability**: role assignment stable across ≥ 2 additional cells (k=5,chain=2 and k=3,chain=3) — Jaccard similarity of top-role-heads ≥ 0.6.

**Null hypothesis (must be refuted)**: A **bag-of-heuristics account** (Nikankin et al. 2024, arXiv 2410.21272 arithmetic-heuristics). Explicit control: shuffle the role labels on the S matrix and rerun the dominance-ratio + dissociation-score computation. If the shuffled S produces comparable statistics, modularity is not real.

### 4.5 Cross-family verify (M5)

- **Gemma-2-9B**: replay the *anchor cell* (k=3, chain=2, 500 pairs) — attribution screen + necessity/sufficiency + role-dissociation.
- **Gemma-2-27B**: only if remaining budget ≥ 2 GPU-h after M1–M5 close. Same anchor cell; smaller pool (250 pairs).
- **Schema recurrence report**: sparsity fraction |C| / |total|, three-role block partition, dissociation score `d`, necessity/sufficiency recovery — reported at family level, NOT at individual head-index level (since heads won't match across families).

### 4.6 Reported metrics — always all three

Per Heimersheim & Nanda (arXiv 2404.15255) and Best-Practices (arXiv 2309.16042):
- `logit_diff` — signed difference of `log P(correct) − log P(incorrect)` on the answer position (or `True − False` on the answer token).
- `prob_diff` — `P(correct) − P(incorrect)`.
- `KL(clean || corrupt+patch)` — KL from clean full-distribution.

Reporting all three is a hedge against metric-driven false positives (Gap G3).

## 5. Key claims → measurable predicates (verbatim)

| Claim | Measurable predicate | Success threshold |
|---|---|---|
| **C1** (sparse) | Shortlist size / total | ≤ 0.15 (i.e. ≤ 15%) with completeness ≥ 0.9 and minimality showing single-removal drop |
| **C2** (modular) | Dominance ratio in S; dissociation d; cross-cell stability | ≥ 2×; ≥ 0.1; Jaccard ≥ 0.6 |
| **C3** (necessity) | Recovery(clean → corrupt patch) | ≥ 0.8 on all 3 metrics |
| **C3** (sufficiency) | Sufficient-recovery(reinsertion) | ≥ 0.8 on all 3 metrics |
| **C3** (specificity) | Control-recovery gap | ≥ 0.6 |
| **Cross-family (M5)** | Schema recurrence on Gemma-2-9B | sparsity fraction and 3-role decomposition qualitatively hold |

Each is falsifiable — the plan can produce a *negative* result and that is a publishable outcome (paper writing hook: "the sparse-modular hypothesis reproduces / does not reproduce cross-family").

## 6. Frontier-primitive necessity check

**No frontier primitive is central** to the reproduction. The methodology relies on established techniques: activation patching (2020), path patching (2022), attribution patching (2023), resample ablation (2019), and circuit-hypothesis testing (2024). The verification is compute-focused, not architecture-focused; no new SAE / transcoder / SFT / RLHF component is needed. This is defensible simplicity — the reproduction *is* the paper, not a novel method on top.

## 7. Remaining risks and mitigations

| Risk | Mitigation |
|---|---|
| **Mistral-7B-v0.1 local symlink broken**; fallback needed | Plan uses `MODEL_DIR/Mistral-7B-Instruct-v0.1` (locally present, ~15 GB verified via safetensors), a semantically equivalent variant for circuit analysis. If the reproduction requires the base model, download `Mistral-7B-v0.1` via HF token at experiment-stage init (~15 min, ~14 GB — fits `MODEL_DIR`). |
| **Gemma-2-27B unavailable locally** (only gemma-3-27b present) | M5 defaults to Gemma-2-9B only; Gemma-2-27B is contingent on budget AND on a successful HF download at experiment stage. Not a blocker for C1/C2/C3 verification. |
| **Metric-driven false positive** (Gap G3) | Report all three metrics (logit_diff, prob_diff, KL) side by side. |
| **Bag-of-heuristics null (C2)** | Shuffled-label control on the role-assignment matrix; must show meaningful gap. |
| **Prompt-specificity of circuits** (arXiv 2506, "Finding Highly Interpretable Prompt-Specific Circuits") | Cross-cell stability check across ≥ 2 additional (k, chain) cells is baked into C2. |
| **10-GPU-h budget** | Attribution-patching first pass is ~O(1 forward+backward per pair) vs full path patching ~O(L·H); shortlist size then bounds full path-patching cost. Detailed hour breakdown in `EXPERIMENT_PLAN.md`. |
| **Sufficiency reinsertion instability at 7B** | Resample-ablation from a large pool (≥ 500 unrelated corrupted prompts) rather than zero-ablate; report distribution over resample seeds. |

## 8. Verdict

**READY** — the method is aligned with `task.md`, follows the accepted circuit-analysis pipeline template (IOI / ACDC / Attribution Patching / Circuit-Hypothesis-Testing), delivers falsifiable predicates for each of the three sub-claims, respects the 10-GPU-h budget on GPUs {0,1,2,3} with headroom for verify + iteration, and cleanly commits the mechanism strategy (Location → Causal Intervention) that Phase 1.75 loaded.
