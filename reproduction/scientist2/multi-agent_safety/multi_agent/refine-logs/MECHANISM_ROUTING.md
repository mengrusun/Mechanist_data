# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Probing / Residual Stream States
chosen_idea_title: Group-Aggregated Activation Probes for Multi-Agent Collusion Detection
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md
  - skills/mechanism-skills/vocabulary-projection/SKILL.md

## Candidates

1. **[recommended]** Probing / Residual Stream States — Linear probes on residual-stream activations are the direct instrument for the plan's core question: is the collusion label linearly decodable from per-agent last-token residual-stream vectors of Qwen3-32B-AWQ, and where along the depth is it most decodable? This IS the plan (M1: layer-wise linear probes; M2: aggregate per-agent probe scores across K agents; M3: apply the same frozen probe zero-shot to transfer scenarios). Standard protocol supports the layer-wise sweep required by M1.3 under a single controlled protocol, keeps the LLM frozen (activation cache once, sweep cheaply), matches all three sanity controls in the plan (label-permute / matched-length / topic-swap), and the "decodability is not causality" caveat is explicitly the scope the sub-claims are formulated in (detection / monitoring, not causal intervention — see `mechanism_strategy.rejected` in `EXPERIMENT_PLAN.md`, which drops Causal Intervention as out-of-budget and out-of-scope).
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md

2. Representation and Parameter Analysis / Representation Engineering — RepE reads out a concept direction from paired-contrast activations and can use the same direction to steer / write in. For detection alone the probing family is cleaner: RepE's write-in path (steering, edits) is out of scope for C1/C2/C3, and its `mean-diff` read-out is a special case of a linear classifier so probing subsumes it for the detection task. Would be the pick if C1/C2/C3 also required causal control.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md

3. Vocabulary Projection / Residual Stream State — Training-free logit-lens on residual stream. Cannot produce a supervised classifier for a task-specific label (`collusive` vs `honest`) — reveals what tokens the state promotes at each layer, which is orthogonal to detecting a binary behavior label. Wrong tool for detection with ground-truth labels.
   - path: skills/mechanism-skills/vocabulary-projection/SKILL.md

## Composition plan

Screen → decode → verify → recover, all in the Probing family with post-hoc analysis on cached activations:

1. **Screen (M1.2)** — one forward pass per (scenario × agent) on Qwen3-32B-AWQ (vLLM if hidden-states exposed; HF-AWQ fallback if not). Cache per-agent last-assistant-token residual-stream vectors at four spaced layers `{27, 37, 48, 59}` of 64 (re-scaled by relative depth from the plan's `{20,28,36,44}/48` — `EXPERIMENT_TIPS.md` match #2). Layer selection heuristic: mid-to-late spaced intervals, per `steering-block-selection` tip.
2. **Decode (M1.3)** — for each of the four candidate layers, train a logistic-regression probe on `train`, select the best layer by `dev` AUROC, report `test` AUROC at the chosen layer. All three sanity controls (label-permute, matched-length, topic-swap) run on the best layer.
3. **Verify — text-only judge baseline (M1.4)** — gpt-5.4 judges the same committee transcripts three-way (COLLUSIVE / HONEST / OTHER) per `multiple-choice-evaluation` tip. Compare probe AUROC vs judge AUROC (M1 predicate: probe ≥ 0.75 AND probe − judge ≥ 0.05).
4. **Recover: aggregation (M2)** — post-processing on cached per-agent probe scores + activations, no additional forward passes on the LM. Five aggregations (`mean-pool`, `max-pool`, `attention-pool`, `concat-MLP`, `best-of-N`) trained on `train`/`dev` with `test` reporting per aggregation. `attention-pool` and `concat-MLP` train tiny heads (≤ 10K parameters) on the cached probe / activation cache — CPU / small-GPU tier.
5. **Transfer (M3)** — freeze M1's best-layer probe and M2's best aggregation, extract activations for NARCBench-Transfer + NARCBench-Stego at the M1-best layer only (~1050 forward passes at Qwen3-32B-AWQ), apply zero-shot, report per-family AUROC.

Cost notes (post-reconciliation, see below): M1.2 ≈ 3 h (vLLM hidden-state extraction, ~2550 forward passes at fp16 KV) or ≈ 5–6 h (HF-AWQ fallback); M1.3 + M1.4 ≈ 0.2 h GPU (probes are sklearn on cached vectors; judge is API); M2 ≈ 0.5 h GPU (small learned aggregators); M3.2 ≈ 1.5 h GPU; M3.3 ≈ 0.1 h GPU. Total ~5.5 h in the vLLM path or ~8 h with HF fallback — within the 10 h HARD CONSTRAINT with 2–4 h contingency.

## Plan reconciliation
<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->
- sites: plan=`[20, 28, 36, 44]` (48-layer relative shortlist) → re-bound `[27, 37, 48, 59]` — plan copied indices from a 48-layer template but Qwen3-32B-AWQ has 64 layers; keep relative depths (0.42, 0.58, 0.75, 0.92) and re-index. Scientific intent (mid-to-late residual-stream shortlist for layer-wise probing) unchanged. This is the exact "never copy a raw index across models of different depth" convention from `EXPERIMENT_TIPS.md` match #2.
- gpu_hours (M1.2): plan≈3 h → revised ≈3 h — no change; the vLLM native-hidden-state path stays the primary and the HF-AWQ fallback stays the contingency.
- n_pairs (M1.3): plan=`available NARCBench-Core train split (~595 scenarios × 3 agents = ~1785 per-agent vectors, ≥ 297 per class after 70% train split)` → matches — logistic-regression converges cheaply on ~1000+ features (5120-d), and the data floor (≥ 50 per class per split) is met by the 70/10/20 split.
- metric (M1.3): plan=`AUROC (primary) + AUROC@1%FPR (secondary)` → matches — probing family's standard binary decodability metric; sklearn's `roc_auc_score` covers both.
- n_pairs (M2.1): plan=`same train/dev/test split, scenario-level (~595 train scenarios)` → matches — aggregators are scenario-level and use per-scenario labels; ≥ 297 collusive + honest labels per split (train) suffices for both closed-form aggregations and the ≤ 10K-parameter learned aggregators.
- metric (M2.1): plan=`AUROC per aggregation + per-domain argmax` → matches.
- gpu_hours (M2.1): plan≈2 h → revised ≈0.5 h — post-reconciliation. The plan estimated 2 h assuming per-scenario re-inference on the model; since aggregation runs on **cached** per-agent activations / probe scores from M1.2, only the tiny heads need training. Frees ~1.5 h back into the M1.2 contingency envelope.
- sites (M3.2): plan=`{best-from-M1}` → matches — Step 5 in composition freezes the M1-best layer and extracts only that layer for the transfer sets.
- metric (M3.3): plan=`per-family AUROC, threshold 0.65 on ≥ 4/7 families` → matches.

reconciliation_status: ok

## Rationale

Probing / Residual Stream States is the single family whose primitive operation IS the plan: train a linear decoder on residual-stream vectors and rank layers by decodability. The `mechanism_strategy` in `EXPERIMENT_PLAN.md` explicitly names *Location* + *Decision Auditing* and rejects *Causal Intervention* on budget grounds — this rules out Causal Attribution / Circuit Discovery / Steering (all requiring interventions). Representation Engineering was runner-up because its `mean-diff` read-out is the closest RepE analog to a linear probe, but it ships as a control-oriented family (steering, arithmetic edits) whose extra machinery is unused for detection-only C1/C2/C3. Vocabulary Projection is training-free and cannot condition on a `collusive`/`honest` label. `families_already_settled` is empty (round 1, `given` behavior, `discovery` mechanism), so no avoid-set applies. Layer shortlist is re-bound from `{20,28,36,44}/48` to `{27,37,48,59}/64` under the "match the model depth" rule from `steering-block-selection`. Post-reconciliation cost estimate ≈ 5.5–8 h vs 10 h HARD CONSTRAINT — comfortably within budget.
