# Experiment Audit — C4: Cross-LLM+SAE-Pair Generalization
## Scope: M2 (results/m2/features.jsonl, summary.json)
## Auditor: auto-verify Phase 2

**overall_verdict: WARN**

### A. GT Provenance
- `true_max = np.array([s['max_val'] for s in held_snips])` in m2_verify_pair.py.
- `held_snips` from `NeuronpediaClient.parse_activation_snippets()` with `model_id="qwen3-4b"`,
  `sae_id_template="{layer}-transcoder-hp"`. These are Neuronpedia's cached activations for the
  Qwen3-4B transcoder-hp SAE — NOT re-computed from the Qwen3-4B model (the model was never run;
  M2 was explicitly scoped to predictive-only to avoid running Qwen3-4B forward passes).
- **PASS**: GT from Neuronpedia's cached activations (transcoder-hp features). Since the model was
  never run forward, there is no risk of model-output-as-GT; Neuronpedia's cache IS the GT.

### B. Score Normalization
- Same `normalize_activations()` as M1 — min-max within-feature on held-out GT.
- **PASS**: No self-referential normalization.

### C. Result File Existence (claim-scoped)
- `results/m2/features.jsonl`: exists, 34 records, 72105 bytes — PASS.
- `results/m2/summary.json`: exists — PASS.
- Cited stats: Δpearson=+0.153, CI[-0.031,+0.341], p=0.18.
  Re-computed: mean_delta=0.1530, CI=[-0.0282,+0.3461], Wilcoxon p=0.1791. Match confirmed.
- **WARN**: M2 logs layers=[8,16] only (L28 missing — 2/3 planned depths). The tracker mentions
  "L8, L16, L28" but the data only covers L8 and L16 (34 total = 20 at L8 + 14 at L16 + 0 at L28).
  EXPERIMENT_RESULTS.md says "~11 per depth at L8, L16, L28" which is inconsistent with the data.
  This is a documentation error (L28 not collected), not a fabrication of results — the numbers
  reported are from the actual data. But the "3 depths" claim in C4 scope is only partially met.

### D. Dead Code
- `sage_lite()` function in m2_verify_pair.py is called in the main loop.
- `score_activation()` for scoring is called.
- All referenced functions are active.
- **PASS**: No dead code.

### E. Scope — Claim-Scoped
- C4 states: generative AND predictive accuracy gains hold on at least one cross-pair.
- M2 provided only predictive accuracy (SAGE-lite, no Qwen3-4B forward passes).
- Generative accuracy on Qwen3-4B is explicitly deferred to /auto-verify.
- SAGE-lite (Explainer + Reviewer only, no empirical feedback loop) is a methodological
  degradation from full SAGE — the Designer + Analyzer roles that provide the empirical
  activation feedback are missing.
- **WARN**: Scope of C4's main-experiment evidence is narrowed to:
  (a) predictive accuracy only (not generative);
  (b) SAGE-lite (not full 4-role SAGE);
  (c) only 2/3 planned depths (L8, L16, not L28).
  EXPERIMENT_RESULTS.md accurately documents these as caveats ("scope-narrowed to SAGE-lite
  predictive-only"). The claim as stated requires BOTH metrics, so main-experiment evidence
  only partially supports C4.

### F. Evaluation Type
- Predictive accuracy on Neuronpedia's cached activations — same as C2.
- SAGE-lite uses Neuronpedia's API activations as GT (not model forward).
- **PASS**: Real GT evaluation.

## Summary
| Check | Verdict | Notes |
|-------|---------|-------|
| A. GT provenance | PASS | Neuronpedia cached qwen3-4b transcoder-hp activations |
| B. Score normalization | PASS | Per-feature min-max, consistent |
| C. Result file existence | WARN | L28 missing from data (only L8/L16); EXPERIMENT_RESULTS.md says "3 depths" inaccurately |
| D. Dead code | PASS | sage_lite() and score_activation() both active |
| E. Scope | WARN | M2 covers predictive-only + SAGE-lite + 2/3 depths; documented but limits C4 support |
| F. Evaluation type | PASS | Real GT |

**overall_verdict: WARN** (Check C: L28 missing; Check E: scope narrowed to predictive-only + SAGE-lite)
