## C5: INCONCLUSIVE (Phase 2 combined verdict: FAIL — experiment scope broken)

- verdict: INCONCLUSIVE
- inconclusive_reason: main-experiment integrity broken — see verify/C5_probe_beats_llamaguard/main_experiment_audit/EXPERIMENT_AUDIT.md
- swap_variants_run: false (Phase 2 FAIL — variants never dispatched)
- Main-experiment verdict on C5: supported
- Main-experiment integrity: FAIL (experiment audit — scope mismatch)
- Variants: none (Stage 2 skipped — Phase 2 FAIL)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C5's headline claim says "matches or beats Llama Guard 3 8B at flagging jailbreaks" but the M5 test set contains 0 successful jailbreaks (claim5_verdict.json: successful_jb=0). The test measured probe performance on bare-harmful-refused vs benign+XSTest-lookalike — a materially easier task than jailbreak detection. The probe reaching AUROC=1.000 on this simpler task does not validate the jailbreak-detection claim. The experiment audit (Check E: Scope) fails because the scope language in the claim exceeds what was actually tested.

To fix: either (a) re-scope C5 to the correct claim ("probe matches LG at flagging bare-harmful content", which IS supported) and re-run M5 audit, or (b) obtain a test set with successful jailbreaks (requiring a model/attack combo where ASR>0) and re-run M5. Then re-invoke `/auto-verify C5 — resume: true`.
