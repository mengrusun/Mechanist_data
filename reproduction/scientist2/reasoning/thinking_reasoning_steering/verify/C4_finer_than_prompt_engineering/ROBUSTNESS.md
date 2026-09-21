## C4: steering offers more distinct operating points than prompt/TI  →  FAIL

- verdict: FAIL
- robustness: 0.0  (0/1 eligible variants pass)
- threshold: 0.5
- n_run: 1
- n_eligible: 1  (integrity_status=WARN → admitted)
- n_pass: 0
- n_fail: 1
- n_integrity_fail: 0
- dimensions_tested: model

### Baseline

- Main-experiment verdict: not-supported (positive-partial — n_distinct(steering)=4 > n_distinct(prompt)=2 for expressing_uncertainty; preservation misses 3-pt floor by 1 pt)
- Main-experiment model: DeepSeek-R1-Distill-Llama-8B
- Main-experiment integrity: WARN (exp=WARN — LLM-judge proxy GT; mech=WARN — no random-direction control, no confirmed mid-plateau)

### Variant results

| Variant | Dimension | n_distinct(steering) | n_distinct(prompt) | n_distinct(TI) | Primary predicate | Binary |
|---------|-----------|---------------------|-------------------|----------------|-------------------|--------|
| model-swap-qwen-14b | model | 0 | 2 | 2 | FAIL (0 > 2 is False) | fail |

### Variant detail (model-swap-qwen-14b)

**Swap**: DeepSeek-R1-Distill-Qwen-14B (cross-backbone, cross-size within R1-Distill family)

**Key finding — sigma_proj collapse**: M1 on Qwen-14B gives sigma_proj=886 for expressing_uncertainty (vs 10.52 for Llama-8B; 84x larger). The sigma_proj-scaled CAA formula (coef = alpha * sigma_proj) produces effective coefficients of 886–1772 at alpha=±1–2, far exceeding the coherent steering range. All 4 steering controllers produce near-total output incoherence (coherence_rate in [0.0, 0.40]) with behaviour_rate=0.0. After the NaN-rate filter, steering operating points collapse to three copies of (0.0, 0.0) → n_distinct=0.

**Prompt and TI controllers** remain coherent (coherence_rate=1.0) and produce 2 distinct operating points each:
- prompt: (behav=0.133, acc=0.800) and (behav=0.350, acc=0.700) → n_distinct=2
- TI: (behav=0.667, acc=0.767) and (behav=0.633, acc=0.717) → n_distinct=2

**n_distinct computation** (eps_r=0.05, eps_a=0.01):
- Steering pts (after NaN-rate filter): [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)] → n_distinct=0
- Prompt pts: [(0.133, 0.800), (0.350, 0.700)] → n_distinct=2
- TI pts: [(0.667, 0.767), (0.633, 0.717)] → n_distinct=2

**Success criterion** (from PLAN.md): n_distinct(steering) > n_distinct(prompt) AND n_distinct(steering) > n_distinct(TI). Result: 0 > 2 = False → criterion NOT met → variant is **INCONSISTENT** with main experiment's positive-partial finding → binary judgment = **fail**.

**Variant integrity** (Phase 9): WARN (exp=WARN — LLM-judge proxy; mech=WARN — inherited under-validated α_op + sigma_proj magnitude). No FAIL → variant included in eligible pool.

### Robustness

robustness = n_pass / n_eligible = 0 / 1 = **0.00**

0.00 < 0.50 (ROBUSTNESS_THRESHOLD) → **FAIL**

### Interpretation

The claim that CAA steering offers more distinct (rate, accuracy) operating points than NL-prompt and thinking-intervention does NOT generalize from Llama-8B to Qwen-14B under the same sigma_proj-scaled coefficient formula. The root cause is architectural: sigma_proj on Qwen-14B is 84x larger than on Llama-8B, causing every sigma_proj-scaled steering coefficient to exceed the coherent operating range. The claim is **fragile** to model architecture change within the same R1-Distill family.

**Scientific implication**: The sigma_proj-scaled formula may need model-specific calibration (e.g., using a fixed absolute coefficient, or re-parameterizing sigma_proj estimation) before the steering-vs-prompt comparison can be made fairly across architectures.

**Upgrade command** (to swap-test C1 or C2 without re-auditing):
`/auto-verify C1 — resume: true` or `/auto-verify C2 — resume: true`
