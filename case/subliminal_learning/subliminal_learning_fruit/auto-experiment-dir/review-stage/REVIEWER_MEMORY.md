# Reviewer Memory

Persistent suspicion / concern log across iterations. Append-only. Iteration N's Phase B.5 adds a section; nothing is ever rewritten.

## Iteration 1 — Score: 6.5/10, Verdict: almost

- **New suspicions**:
  - Same-judge coupling: gpt-4o is used for both filtering residue and downstream banana evaluation → possible circularity / shared bias.
  - The effect may partly reflect distributional/style residue rather than a semantically abstract "banana preference" per se.
  - Extreme teacher banana rate (0.903 raw) means the surviving 58 non-banana teacher images are a highly unusual conditional slice (~10% survivors), not representative "clean neutral fruits."

- **Previous suspicions addressed?**: n/a (first iteration)

- **Unresolved (carried forward)**:
  - Post-filter matched N=53/arm is the major statistical and representational limitation.
  - Robustness tested on model axis only (LoRA rank 16→8); no dataset-axis or method-axis validation.
  - Generality beyond banana / Qwen-Image / LoRA-denoising-SFT is unknown.
  - Mechanism remains unresolved: current LoRA-SVD-additive-steering family is negative, but this refutes ONE family, not the mechanism space.
  - C2 swap-test skipped (INTEGRITY_ONLY, max_verify_claims_cap) — Stage 1 audit passed; swap-test upgrade `/auto-verify C2 — resume: true` recorded in Open Items.

- **Patterns**:
  - Watch for future variants keeping the effect under **independent judges** or non-overlapping evaluation/filter pipelines.
  - Watch for subliminal transfer persisting for **other concepts** and less extreme teacher adapters.
  - Watch for effect-size collapse when post-filter N increases via a weaker teacher or broader prompt set.
  - Watch for alternative mechanism probes showing **signed dose-response with specificity**, unlike the flat additive steering results.
