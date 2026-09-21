## C1: robustness = 1.00 (threshold = 0.5, eligible = 1/1) — PASS

- swap_variants_run: true
- Main-experiment verdict on C1: not-supported (partial — geometry passes, refusal-side unmeasurable)
- Variant counts (over `consistent_with_main_experiment`): 1 pass, 0 fail
  (of 1 eligible; 0 excluded for integrity reasons)
- Model dimension: consistent with the main experiment (consistent=pass, claim_supported=fail, delta: cosine ratio improved from 0.20 to 0.069 — h and r even more geometrically distinct on Qwen2; refusal-side still NaN)
  [INTEGRITY: WARN — experiment: auroc_r=NaN in result file (expected: Qwen2 refuses nearly all bare-harmful prompts)]

Interpretation: C1's not-supported verdict (due to partial result: geometry passes, refusal-side unmeasurable at high bare-refusal rate) is robust across model families. Qwen2-Instruct-7B, a different model family with different safety fine-tuning, shows the same pattern: h direction cleanly separates harmful from benign at ceiling AUROC (1.000 at layer 15), cos(h,r)=0.068 is even further from 1.0 than the Llama-3 result (cosine ratio 0.069 vs 0.20 of split-half reference), and refusal-side sub-tests are NaN because Qwen2 also refuses nearly all bare-harmful prompts in single-shot greedy mode. The not-supported finding appears to be a consequence of the AdvBench + instruction-tuned LLM combination (high alignment → few natural jailbreaks → refusal-side unmeasurable), not a Llama-3-specific artifact. To test "independently-recoverable" more cleanly, a model with lower baseline refusal rate or an attack-supplemented jailbreak dataset is needed.

robustness = 1.0 >= 0.5 threshold → PASS. The main experiment's not-supported verdict is consistently reproduced across models.
