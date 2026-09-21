# Variant Diff: model-swap-gemma2-9b

## What changed vs the main experiment

**Single change**: Model swapped from `Mistral-7B-v0.1` to `Gemma-2-9B (LLM-Research/gemma-2-9b)`.

All other parameters held fixed:
- Dataset: k3_chain2_natural (anchor cell), n_pairs=500 — identical
- Method: attribution patching screen + path patching (necessity) + reinsertion (sufficiency) — identical
- Script: `scripts/cross_family_verify.py` (M5 milestone)
- Shortlist selection: same 15% cap attribution-patching screen (produces 107-component shortlist for Gemma-2-9B vs 158 for Mistral-7B — shortlist size differs due to different (L×H+L) total, but the selection method is identical)
- Batch size: 2 (adapted for Gemma-2-9B memory profile; Mistral-7B used batch-size 4 for M1/M2/M3; adaptation documented in M5 entry)
- Corruption type: corrupt_fact — identical
- Seed: 42 — identical

## Reuse justification

`results/M5_gemma9b.json` produced by M5 is methodologically equivalent to a fresh model-swap variant run. The cross_family_verify.py script replicates M1 (attribution screen) + M2 (necessity) + M3 (sufficiency) + abbreviated M4 (role-dissociation on top-20 of shortlist) on Gemma-2-9B. All needle values needed for C3 judgment (necessity_recovery.LD, sufficiency_recovery.LD) are present in M5_gemma9b.json. No fresh GPU run required.

GPU-hours saved by reuse: ~0.67 h (actual M5 wall-time was 2397s ≈ 0.67 GPU-h).
