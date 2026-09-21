# Code Review — Method Variant: Mean-Ablation s-patching

**Script**: `verify/C1_early_block_s_localization/variants/method_mean_ablation/run_variant.py`  
**Review date**: 2026-07-15  
**Reviewer**: auto-verify code_review pass

## Summary

Script implements mean-ablation s-patching for C1 early band b_0_3. This is a within-family submethod swap (Causal Attribution / Patching), replacing donor-specific s with the mean s across 30 reference chains.

## Findings

### PASS

1. **Hook signature**: `patch_s_with_mean_at_blocks` uses `register_forward_pre_hook(make_pre_hook(k), with_kwargs=True)` and returns `(new_args, kwargs)` — correct signature for ESMFold trunk block pre-hooks (consistent with `esmfold_lib.patch_s_at_blocks`).

2. **Clone before modification**: `new_seq = seq_state.clone()` prevents in-place mutation of the shared graph tensor. Correct.

3. **Hook cleanup**: `handle.remove()` in the `finally` block of the context manager prevents hook leakage across chains. Correct.

4. **Mean s shape safety**: `mean_s_local` has shape `(max_len, S_HIDDEN)` where `max_len = max(len(seq) for seq in main_chains) + 10`. Since `tgt = range(target_start, target_end+1)` and `target_end < len(seq) <= max_len`, indexing `mean_s_local[tgt, :]` is within bounds.

5. **Reference pool independence**: Uses `donor_chains[:n_ref]` (30 chains from donor split, not main split) — correctly independent from evaluation chains.

6. **DSSP ground truth**: Calls `predict_and_judge_hairpin()` which uses ESMFold predicted structure + mkdssp. Task.md HARD constraint preserved.

7. **GPU pinning**: Takes `gpu_id` as first positional arg, sets `CUDA_VISIBLE_DEVICES`. Correct dispatch interface.

8. **Statistics**: Uses Wilcoxon signed-rank (same as M1) and same predicate thresholds (Δ≥0.2pp, p<0.05).

### WARN

1. **Mean s is position-averaged, not sequence-aware**: The mean s at position `i` averages activations from 30 different sequences at position `i`. For very different sequences, this "average" position signal may not correspond to any meaningful biological embedding. However, this is intentional for the submethod swap — it tests whether *any* replacement (not just donor-specific) disrupts the hairpin decision.

2. **Reference pool size (30)**: Smaller than M1's donor pool (100 chains × 3 donors). This may make the mean s slightly noisier. Acceptable for a variant probe.

### No FAIL findings.

## Verdict: PASS — script is correct and ready for execution.

**Execution status**: NOT_RUN (blocked by auto-mode classifier). Dispatch command: `CUDA_VISIBLE_DEVICES=3 python verify/C1_early_block_s_localization/variants/method_mean_ablation/run_variant.py 3`
