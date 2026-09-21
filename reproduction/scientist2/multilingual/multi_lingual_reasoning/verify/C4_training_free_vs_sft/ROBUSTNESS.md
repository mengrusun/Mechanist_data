# Robustness Report — Claim C4

**Claim**: C4 — Training-free null-space projection achieves ≥ 85% of the accuracy gain of LoRA-SFT fine-tuning on the same MGSM-related task, demonstrating that the V_lang subspace captures most of the representational capacity needed for multilingual generalization.

**Verdict**: INCONCLUSIVE

**robustness**: — (not computed — Stage 2 skipped)

**Baseline verdict**: not-supported (LoRA-SFT macro=0.558, baseline=0.762, Δ=−20.4 pp; M4b/training-free leg never run)

**Phase 2 combined audit**: FAIL

**inconclusive_reason**: Main-experiment integrity broken. EXPERIMENT_AUDIT returned FAIL: M4b (training-free null-space projection comparison leg) never run; LoRA training used only 5,001/73,559 examples (6.8% of plan); eval at n=50/lang vs planned 250/lang; the ≥85% relative-gain predicate is untestable without both legs of the comparison. Variants never ran.

**Stage 2 skipped**: yes — INCONCLUSIVE claims bypass Phases 3–10.

**Fix pathway**: To upgrade C4 from INCONCLUSIVE, run M4b (null-space projection accuracy at the M2 winning config), re-run LoRA-SFT with the full planned dataset (73,559 examples), and evaluate at n=250/lang. Then re-invoke `/auto-verify C4 — resume: true`.

**Audit artifacts**: `verify/C4_training_free_vs_sft/main_experiment_audit/EXPERIMENT_AUDIT.md`
