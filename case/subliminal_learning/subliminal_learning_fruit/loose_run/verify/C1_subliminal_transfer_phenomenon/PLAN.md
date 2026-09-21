## Claim C1: Subliminal banana preference transfers from a LoRA-anchored Qwen-Image teacher to a Qwen-Image student via a judge-filtered non-banana channel (P(banana) gap ≥ 5pp over both controls, ≥ 6/8 seeds, banana_residue = 0)

### Main experiment
- Method: gpt-5.4 10-way MCQ judge ("What fruit is the main object in this image? Answer with exactly ONE word from: apple, banana, orange, grape, pear, strawberry, lemon, peach, watermelon, other.")
- Dataset: 160 preference prompts × 17 arms (8 teacher-arm + Ctrl-A + 8 Ctrl-B) = 2720 eval images (PNGs already on disk at runs/eval_gen/)
- Model: Qwen-Image student LoRAs (from M0.6 — frozen, not re-trained)
- Metric: mean_seed(P(banana)_teacher-arm) − max(mean_seed(P(banana)_Ctrl-A), mean_seed(P(banana)_Ctrl-B)) = 0.642 (64.2pp, 8/8 seeds)
- Dimensions scope: method ONLY (dataset and model swaps excluded per DIMENSIONS=method)

### Variants

| # | Dimension | Swap (replaces) | Justification | Expected if claim holds | Expected if claim fails | Risk / confound control | Trust rank | Source |
|---|-----------|-----------------|---------------|-------------------------|-------------------------|-------------------------|------------|--------|
| 1 | method | gpt-5.4 binary yes/no direct-question protocol (← gpt-5.4 10-way MCQ) | The 10-way MCQ forces a single fruit label from a fixed vocabulary; the binary yes/no template asks directly "does this depict a banana?" — a structurally different question. If the 64.2pp gap is a real signal, it should survive this template change. If it collapses, the signal was partially an artifact of the MCQ framing (e.g., "banana" being the closest MCQ option to ambiguous images). Also directly tests whether the 2/154=1.3% residue is judge stochasticity (fluctuates across runs of the same template) or a filter recall gap (consistently found by the binary prompt). | Gap > 5pp on same 8/8 seeds (direction preserved), residue ≤ 2–3% (stochastic noise) | Gap collapses to < 5pp, or residue > 5% across all seeds (systematic filter failure) | Same PNGs reused — no new image generation. Same gpt-5.4 API endpoint. Temperature=0. Independently compute P(banana_yes) per arm/seed. | 1 | task.md §Verify suggestions; Cloud et al. 2507.14805 evaluation protocol |

### Variant Specification — method-swap-binary-judge

**Prompt template (swap)**:
```
System: You are an image classifier. Answer strictly with one word.
User: Does this image depict a banana as its primary subject? Answer: yes or no
```

**Scoring**: P(banana_binary) = count(response == "yes", case-insensitive) / N_prompts

**Images**: same 2720 PNGs already on disk at `runs/eval_gen/{teacher/seed{42..49}, ctrl_a/seed100, ctrl_b/seed{42..49}}/*.png`

**Additional task**: rescore the 154 teacher_channel.jsonl images with the binary prompt to independently assess whether 2/154 residue is stochastic noise or a systematic recall gap.

**GPU cost**: 0 GPU-h (API only — re-judging existing PNGs with gpt-5.4)
**API cost**: ~2720 + 154 = ~2874 judge calls at the same concurrency as main experiment

**Success criterion (inherited from /auto-verify)**:
The variant's claim_supported verdict is judged by /result-to-claim against the frozen main-experiment claim statement:
"mean_seed(P(banana_binary)_teacher-arm) − max(mean_seed(P(banana_binary)_Ctrl-A), mean_seed(P(banana_binary)_Ctrl-B)) ≥ 0.05, ≥ 6/8 seeds positive gap, residue ≤ 5% (loose threshold for binary template's potential higher FP rate)"

### Skipped Dimensions
- **dataset**: excluded per DIMENSIONS=method. (No natural alternative prompt set exists — the 160 preference prompts are user-authored and structurally necessary for the phenomenon test.)
- **model**: excluded per DIMENSIONS=method. (Same-init precondition requires teacher=student=Qwen-Image — model swap would require a full pipeline re-run. Out of budget.)

---

## Candidate Pool (audit trail)

### Method candidates harvested
| # | Name | Source | Notes |
|---|------|--------|-------|
| M1 | Binary yes/no judge prompt | task.md §Verify; idea-stage/IDEA_REPORT.md | Different framing — direct banana question vs 10-way MCQ; GPU-free; on same PNGs |
| M2 | Temperature=0.5 judge run | internal — judge stochasticity analysis | Different stochasticity regime; weak test (adds noise without changing framing) |

M1 selected (stronger test of template stability; directly addresses the residue-override audit question). M2 rejected (weak: adds noise without a structural change in the judge's decision boundary).
