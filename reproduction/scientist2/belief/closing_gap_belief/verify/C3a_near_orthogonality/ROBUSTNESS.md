## C3a: robustness = 1.00 (threshold = 0.5, eligible = 1/1)  ->  PASS

- swap_variants_run: true
- Main-experiment verdict on C3a: supported
- Variant counts (over `consistent_with_main_experiment`): 1 pass, 0 fail
  (of 1 eligible; 0 excluded for integrity reasons)
- Model dimension: matches the main experiment (consistent=pass, claim_supported=pass, |cos|=0.021 vs main |cos|=0.015 — same near-orthogonal regime, different architecture)

Interpretation: The geometric near-orthogonality finding (C3a) replicates cleanly in Qwen2.5-7B-Instruct (cross-family model swap). At Qwen's L*=22, |cos(v_c*, v_v*)| = 0.021 [0.001, 0.037] (95% CI bootstrap n=200), compared to Llama's main-experiment result of 0.015 [0.001, 0.034] at L*=31. Both are well below the 0.3 threshold (CI upper < 0.4). The neighborhood L*±2 mean is 0.015 (vs main 0.025). Both probe AUROC values are strong (probe_c=0.864, probe_v=0.898), confirming informative directions before computing the angle. The random-direction null (0.013) is consistent with theoretical 1/sqrt(3584)=0.017, and the measured |cos| is indistinguishable from this null — exactly as in the main experiment. The result is architecture-agnostic across the Llama and Qwen families, different layer counts (32 vs 28), different embedding dimensions (4096 vs 3584), different pretraining mixes, and different tokenizers. The variant integrity audit passed (experiment=pass, mechanism=n/a). robustness = 1/1 = 1.00 >= threshold 0.50 -> PASS.
