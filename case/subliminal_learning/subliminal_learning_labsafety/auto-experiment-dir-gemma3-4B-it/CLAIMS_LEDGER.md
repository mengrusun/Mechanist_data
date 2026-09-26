# Claim Ledger — Cross-modal subliminal transmission of safety-competence loss in multimodal Gemma-3-4B-it

**Direction**: Validate whether a text-only distillation channel from a LoRA-SFT'd Gemma-3-4B-it teacher subliminally degrades a Gemma-3-4B-it multimodal student's image-conditioned chemistry-lab safety competence on QA_I (M0 gate: dual ≥3pp drop vs Ctrl-A AND Ctrl-B across ≥3 seeds); if M0 passes, investigate the mechanism.
**Date**: 2026-08-03 → (running)
**Pipeline**: running | **Iteration**: —/10 "" (0/6)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: claim

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 cross-modal subliminal QA_I drop | — | — | — | planned |
| C2 language-tower component causally mediates drop (cond. on C1) | — | — | — | planned |

---
## C1 — Cross-modal subliminal QA_I drop
- **Statement**: The treated student (Gemma-3-4B-it multimodal, LoRA-SFT'd on filtered text outputs of the LoRA-SFT'd teacher on ~10k+ lab-safety prompts) shows an image-conditioned QA_I accuracy that is ≥3 pp below BOTH Ctrl-A (base student, no fine-tune) AND Ctrl-B (student LoRA-SFT'd on filtered text outputs of the base/un-tuned teacher on the same prompts + same filter + same recipe), reproduced across ≥3 random seeds — demonstrating cross-modal subliminal transmission of safety-competence loss via a text-only channel.
- **Origin**: task.md (given-validation faithful capture)
- **Data**: QA_I-00000-of-00001.parquet (eval); teacher_anchor_sft.json (teacher SFT); constructed lab-safety prompt corpus (teacher generation input) — provenance=existing + constructed; available=QA_I full parquet, teacher_anchor_sft full JSON, prompt corpus ≥10k; used=QA_I full (task.md forbids subsets), teacher_anchor_sft full, prompt corpus ≥10k (planned)
- **Models**: /mnt/quarkfs/share_model/gemma-3-4b-it (teacher = student base)
- **Method**: M0 phenomenon-validation gate. Recipe: LoRA-SFT teacher on anchor JSON → generate ~10k+ lab-safety responses (T=1.0, top_p=1.0, top_k=0, max_new=256) → gpt-5.4 filter → LoRA-SFT student across LR sweep at seed 42 → 3-arm 3-seed runs at lr* (treated, ctrl_b, ctrl_a) → greedy QA_I eval, gpt-5.4 content-match. Verdict: `established` iff ∀seeds Δ_A≥3pp AND Δ_B≥3pp.
- **Main experiment**: (pending)
- **Verify**: (pending)
- **Iteration**: (pending)
- **Final**: planned
- **Caveats**:
- **Artifacts**: `refine-logs/EXPERIMENT_PLAN.md#M0`, `refine-logs/FINAL_PROPOSAL.md`

---
## C2 — Language-tower component causally mediates the subliminal drop (conditional on C1)
- **Statement**: Conditional on C1 = established/conditional: some internal component of the LoRA-tuned student's language tower causally mediates the subliminal safety-competence drop — i.e. (i) M1 localizes a top-1 candidate direction/site d̂_ℓ (Location: diff-in-means / linear probe / attribution, AUROC ≥0.7 or ‖Δ‖ z ≥3), AND (ii) M2 causal intervention shows sufficiency (steering ctrl_a with +d̂ lowers QA_I by ≥3pp at some α>0 while matched-norm random direction moves ≤1pp) AND necessity (steering treated with −d̂ raises QA_I by ≥3pp while random moves ≤1pp) with off-target penalty (MMLU-lite + helpfulness-lite) ≤3pp.
- **Origin**: task.md (goal clause "if it does, further investigate the mechanism behind it") + /mechanism-explore chain Location → Causal Intervention (given-validation × discovery)
- **Data**: 500-item held-out QA_I subset for activation capture (never re-used in M0 accuracy); QA_I full parquet for causal-intervention eval; MMLU-lite (≥500) + helpfulness-lite (≥200) off-target — provenance=existing (carved-out mechanism-tooling batch) + reused (student ckpts from C1); subset_note: 500-item mechanism batch is `method_sensitive`; `/mechanism-skills` may re-bind 200–2000
- **Models**: /mnt/quarkfs/share_model/gemma-3-4b-it (student, treated + ctrl_b + ctrl_a checkpoints from C1's M0.d)
- **Method**: M1 (Location, correlational): diff-in-means / linear probe / attribution across all language_tower layers; pick top-1 site + direction d̂_ℓ. M2 (Causal Intervention, causal): dose-response sweep α∈[-2,-1,-0.5,0,0.5,1,2] × direction_type∈[d̂, random_matched_norm] × arm∈[ctrl_a_sufficiency, treated_necessity] × seed∈[42,200,1337]; off-target specificity at α* on MMLU-lite + helpfulness-lite. (Optional M3: SAE projection of d̂_ℓ.)
- **Main experiment**: (pending)
- **Verify**: (pending)
- **Iteration**: (pending)
- **Final**: planned
- **Caveats**: Conditional on C1 passing M0; refuted-at-M1 or refuted-at-M2 are legitimate mechanism-negative outcomes.
- **Artifacts**: `refine-logs/EXPERIMENT_PLAN.md#M1`, `refine-logs/EXPERIMENT_PLAN.md#M2`, `refine-logs/EXPERIMENT_PLAN.md#M3`, `refine-logs/FINAL_PROPOSAL.md`

---
## Journey Summary
- **Claim**: 1 idea (given-validation faithful capture) → top idea "Cross-modal subliminal transmission of safety-competence loss in multimodal Gemma-3-4B-it" (impact/novelty/feasibility per FINAL_PROPOSAL.md)
- **Mechanism strategy**: Location → Causal Intervention (rejected: Tuning & Editing, Formation Tracing, Unit Interpretation kept optional as M3, Decision Auditing)
- **Mechanism routing**: (pending experiment stage)
- **Experiment**: (pending)
- **Verify**: (pending)
- **Iteration**: (pending)
- **Figures**: (pending final ledger hook)
