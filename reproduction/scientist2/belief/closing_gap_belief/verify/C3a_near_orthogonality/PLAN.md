## Claim C3a: The gold-correctness direction v_c and the verbalized-confidence direction v_v at the primary reporting layer L* are geometrically near-orthogonal (|cos| <= 0.3 with 95% CI upper bound < 0.4), robust across L*+-2 neighborhood.

### Main experiment (from /auto-experiment)
- Method: L2-logistic-regression probing on residual-stream hidden states (outputs.hidden_states[L], last-input-token) across 33 layers; cosine of L2-normalized probe weight vectors; 200-resample retrain-on-bootstrap CI; neighborhood L*+-2
- Dataset: TriviaQA rc.nocontext validation, 10k questions (6k/2k/2k split), two-pass extraction
- Model: Llama-3.1-8B-Instruct → |cos| = 0.015 [0.001, 0.034] at L*=31; neighborhood mean = 0.025; random-direction null = 0.011
- Dimensions scope (active): model only (DIMENSIONS=model); method and dataset axes inactive this pass

### Variants

| # | Dimension | Swap (replaces) | Justification | Expected if claim holds | Expected if claim fails | Risk / confound control | Trust rank | Source |
|---|-----------|-----------------|---------------|-------------------------|-------------------------|-------------------------|------------|--------|
| 1 | model | Qwen2.5-7B-Instruct (← Llama-3.1-8B-Instruct) | Cross-family instruct model (different architecture, tokenizer, pretraining mix, RLHF stack). Preserves instruct-tuned reporting regime so the verbalized-confidence channel exists. Maximum architectural independence from LLaMA while keeping the behavioral context. Reviewer rank: #1. | |cos| well below 0.3 at Qwen's L*; 95% CI upper < 0.4; neighborhood L*+-2 mean <= 0.3; both correctness and verbalized-confidence probes achieve AUROC >= 0.70. Exact magnitude may differ (plausible ~0.02-0.12). | |cos| > 0.3 at Qwen's L*, or CI upper >= 0.4, or neighborhood mean elevated -- indicating C3a is LLaMA-specific rather than a general instruct-model property | Different optimal L* (control: search all layers with same L* rule); tokenizer/formatting differences (control: identical prompt templates, verify parsing rates); probe quality (control: report per-probe AUROC before interpreting cos). | 1 | EXPERIMENT_PLAN.md Verify Suggestions + task.md model swap pool + reviewer consensus |

### Skipped Dimensions
- method: not in DIMENSIONS (DIMENSIONS=model); within-family method swaps (e.g., probing/sae-feature-activation-state) deferred to later /ablation-planner pass
- dataset: not in DIMENSIONS (DIMENSIONS=model); dataset swaps (TruthfulQA, MMLU) deferred

### Success Criterion (inherited from /auto-verify)
Each variant's claim_supported verdict is judged by /result-to-claim against the frozen C3a claim statement. Consistent = Qwen2.5-7B-Instruct also shows |cos| <= 0.3, CI upper < 0.4, neighborhood mean <= 0.3 → consistent_with_main_experiment = pass.

---

### Candidate Pool (audit trail)

**Model candidates harvested**
| # | Name | Source | Notes |
|---|------|--------|-------|
| Mdl1 | Qwen2.5-7B-Instruct | EXPERIMENT_PLAN.md Verify Suggestions + task.md | Cross-family instruct; different architecture/tokenizer; same RLHF-tuned category |
| Mdl2 | Mistral-7B-Instruct-v0.1 | EXPERIMENT_PLAN.md Verify Suggestions + task.md | Cross-family instruct; different architecture |
| Mdl3 | Llama-3.1-8B | EXPERIMENT_PLAN.md Verify Suggestions | Base model same architecture; adversarial test of RLHF dependence |
| Mdl4 | Qwen2.5-7B | task.md model pool | Base Qwen; cross-family + base regime change |
| Mdl5 | Mistral-7B-v0.1 | task.md model pool | Base Mistral |

**Reviewer selection**: Mdl1 (Qwen2.5-7B-Instruct) — strongest test of cross-architecture robustness while preserving instruct-tuned reporting regime.
