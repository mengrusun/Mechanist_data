# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - finetune-hyperparameter-sweep
  - multiple-choice-evaluation
  - steering-coefficient-tuning
  - steering-block-selection

## Matches

1. **finetune-hyperparameter-sweep** — plan runs LoRA SFT for teacher (M0.1) and student (M0.5 LR sweep, M0.6 per-seed reproductions) on Qwen3.5-9B via `AutoModelForImageTextToText`. Fires because there are two distinct fine-tunes: teacher SFT on `teacher_anchor_sft.json` and student SFT on the filtered teacher generations (same base, different data → distinct pilots per Scope callout).
   - conventions to adopt:
     - **LR-first sweep** already declared for the student (M0.5, LR grid `{5e-5, 1e-4, 2e-4, 5e-4, 1e-3}` matches the LoRA SFT column of the method × objective table). Winner frozen → M0.6 per-seed reproductions.
     - **Teacher SFT** — plan does not currently specify an LR pilot. Use `sanity_checked` mode: single reference config (LR=1e-4, r=16, α=32, effective_batch=8, 1 epoch) with training-side diagnostics on a 500-example pilot. If any A/B/C signal fires → escalate to full LoRA SFT LR grid.
     - **`sweep_status`** — record in `EXPERIMENT_RESULTS.md` under M0.1 and M0.5 (Plan-editing mandate item 3).
     - **α/r ≥ 1** (α=32, r=16 → OK). Effective batch ≤ 32 for LoRA SFT.
     - **Iteration rule**: on M0.6 seed reproductions with primary-gate miss, LR-first change before rank / α.

2. **multiple-choice-evaluation** — QA_I is a multiple-choice benchmark judged by gpt-5.4 content-match. Fires on M0.7 (Ctrl + 3 treated eval) and M0.8 auxiliaries.
   - conventions to adopt:
     - **Three-way `{A, B, C, D, other}` output** from judge — no silent coercion of refusal / off-topic to the first-appearing letter.
     - **Judge input includes option texts**, not letters alone.
     - **Per-arm `other` rate** logged in `results/eval_<arm>.jsonl` and `results/m0_headline.json`.
     - **Frozen judge config across arms** (gpt-5.4, temperature=0.0, seed=0). Cache is by `(prompt, model, temperature, seed)` — this is our only judge and the freeze is trivial.
     - **A/B orientation swap** is not directly applicable to 4-way MCQ; but per-letter accuracy diagnostics IS applicable — flag if any letter's arm-accuracy deviates > 10 pp from the arm mean.
     - **Regex vs judge disagreement < 5 %** on a 100-row spot check — build a strict regex `^\s*\(?[A-D][)\.\s]` prefilter and reconcile.

3. **steering-coefficient-tuning** — M2 declares α ∈ {-2, -1, 0, +1, +2} (5-dose sweep, treated as raw multipliers of the direction vector). Fires because α values look raw, not σ-normalized.
   - conventions to adopt:
     - **Express α in σ_proj units**: for each M1 direction `v_l`, compute `σ_l = std(h_lᵀ · unit(v_l))` on the safety-relevant prompt set, and interpret the α values as multipliers of σ_l (so α=1 means +1σ of on-direction magnitude).
     - **Fluency / general-ability metric alongside target**: MMLU-slice is the general-ability control (M2.c). Also add a short-form perplexity / repetition-rate check on the treated arm at each α (log to `mechanism/m2/<run_kind>/alpha<α>.json`).
     - **Prefer the smallest sufficient α**: if α=-1 already restores Ctrl-level accuracy, do not report α=-2 as "the" answer.
     - **β=0 baseline** included in the sweep (α=0 is already in the grid — OK).

4. **steering-block-selection** — M1 → M2 chain (Location screen, then Causal Intervention). M2 declares "top-K M1 directions on the most-divergent layer(s)". Fires because layer indices are provisional (`sites:` is `method_sensitive`).
   - conventions to adopt:
     - **Lock site set first (M1) before α sweep (M2)**. Do NOT hard-code a raw layer index — pick by the M1 screen (activation-difference effective rank per layer).
     - **Mid-to-late layers first** — Qwen3.5-9B has 32 language-tower blocks (config confirmed: `layer_types` length 32, 3:1 linear/full pattern). Screen at spaced intervals across all 32 layers, then focus on the highest-divergence layer for M2.
     - **Widen to 3–5 layers if a single layer is inert** — apply to M2 if α sweep at the single top layer shows no effect (fallback rule).
     - **Match the claim's altitude**: C3a claim is "low-dim safety-relevant substrate inside the language tower" — this is a **kind-level** claim, so localizing to a small window (top-3 layers) is sufficient; do not force a whole-stack sweep.

## No-match log

- `image-preprocessing` (Tip 1) — not applicable. QA_I images are pre-encoded via the Qwen3.5 vision tower with its own processor; there is no `T.Compose` / torchvision path.

## Tips applied to plan (audit trail)

- M-1: no fine-tune, no MCQ eval, no steering — no tips fire.
- M0.1 (teacher SFT): tip 4 fires. Record `sweep_status: sanity_checked` (single reference config with training-side pilot diagnostics).
- M0.2–M0.4 (teacher gen, filter, rescan): no tips (data pipeline only).
- M0.5 (student LR sweep): tip 4 fires. Record `sweep_status: swept` after M0.5 completes.
- M0.6 (per-seed): inherits M0.5's winner; no new tip work.
- M0.7 + M0.8: tip 5 fires. Judge cache + 3-way (A/B/C/D/other) parse required.
- M1: tip 6 fires. Screen layers, pick sites.
- M2: tips 5, 6, 7 fire (site + coefficient + MCQ eval).
- M3: no additional tips.
