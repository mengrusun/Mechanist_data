# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - general-rule-mechanism-interpretability
  - steering-block-selection
  - steering-coefficient-tuning

## Matches

1. **general-rule-mechanism-interpretability** — unconditional load for any mechanism/interpretability experiment (M2/M3/M4 all intervene on internal components).
   - convention to adopt: "locate the neuron / feature for the target function, then intervene" + "intervene on the target behavior only — do not damage general ability". Report the target metric AND a general-ability metric together, never target metric alone.
   - evidence in plan/code:
     - M2 localizes via Fisher-information mask + causal zero-ablation (the "locate" step is a principled localization method).
     - M2's four verbatim criteria bake in general-ability guardrails: C2c (off-target drop ≤ 0.10, including `world_knowledge`) and C2d (PPL ≤ 1.05× clean). The mechanism-audit criterion (target moves while general ability intact) is directly enforced.
     - M3 uses the same 3-task frame (world_knowledge control + 2 belief tasks) at every checkpoint, so general ability is measured in parallel with target behavior at each step.
     - M4 controller has an explicit `Δ_wk ≤ 0.05` soft guardrail on world_knowledge accuracy AND reports `PPL_controller` on the pretraining sample.

2. **steering-block-selection** — fires because M4 chooses target sites (attention heads H* from M2) for amplification.
   - convention to adopt: lock the site set BEFORE the coefficient sweep; the site should be picked via a principled screening method; do not copy raw layer indices across models of different depth.
   - evidence in plan/code:
     - Sites are the M2 H* head sets (Fisher-mask + causal zero-ablation localization) — a rigorous activation-based screening. Site set is fixed BEFORE the α grid in M4.2, satisfying the composition rule ("lock site first, sweep coefficient after").
     - Sites are chosen per-model via M2 rerun on the target model, so no raw layer indices are transferred across depths. Each model's H* is discovered fresh.
     - The claim is about circuit-level intervention on attention heads (the tip's noted correct target for circuit discovery); the plan intervenes on attention head output specifically via forward-hook.

3. **steering-coefficient-tuning** — fires because M4 sweeps an amplification coefficient (α_personal, α_attributed) over a 6-value grid per axis.
   - convention to adopt: sweep coarsely including a baseline (β=0 or α=1.0); score on target metric + general-ability metric to catch collapse; re-tune whenever site / direction / model changes.
   - evidence in plan/code:
     - Grid includes `α = 1.0` (identity / no amplification) as a baseline ✓
     - Grid spans 6 values per axis (1.0, 1.5, 2.0, 3.0, 4.0, 6.0) — a coarse sweep ✓
     - Score = `net_improvement` (target) AND `Δ_wk` (general-ability guardrail) AND `PPL_controller` reported post-hoc (fluency) ✓
     - Grid is re-run per model (M4.2 runs per model that succeeds M2.4), satisfying "re-tune when the model changes" ✓
     - Selection rule (max `net_improvement` s.t. `Δ_wk ≤ 0.05`) diverges from the tip's "smallest sufficient β" heuristic, but this is a reproduction (`resource_fidelity: strict`) so the plan's verbatim rule stands — the divergence is intentional and documented in FINAL_PROPOSAL.md.
     - Post-hoc PPL check on the pretraining PPL sample directly measures fluency after the controller is applied.

## No-match log

- **ImageNet Eval Preprocessing** — not applicable (text-only language modeling task, no vision backbone).
- **Fine-Tuning Hyperparameter Sweep** — the frame classifier (M4.1) is a **probe** (small MLP on frozen model activations), not a fine-tune of the language model itself. Its hyperparameters (lr=5e-4, wd=1e-4, bs=64, 20 epochs, patience=5) are pinned by the reproduction spec; no fine-tune sweep applies.
- **Multiple-Choice Evaluation** — the correctness metric is a log-prob comparison of gold vs distractor continuation (verbatim: `correct ⇔ Σ_t log P_θ(y_t^+ | x, y_<t^+) > Σ_t log P_θ(y_t^- | x, y_<t^-)`). No letter parsing, no MCQ regex, no LLM judge — the metric is deterministic and unambiguous, so this tip does not apply.

## Compliance summary (for auditor)

- **General rule** honored in M2/M3/M4: target metric always reported with general-ability metric (world_knowledge + PPL).
- **Composition rule (site-first, coefficient-second) honored**: M2 locks H* before M4.2 sweeps α on that fixed H*.
- **Reproduction-fidelity divergence** from the "smallest sufficient β" heuristic is intentional per `resource_fidelity: strict`; the plan's `max net_improvement s.t. Δ_wk ≤ 0.05` rule is verbatim from `task.md`.
