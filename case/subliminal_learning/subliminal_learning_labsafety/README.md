# Subliminal Learning on a Multimodal Model (Qwen3.5-9B)

**English** · [中文](./README.zh.md)

Tests whether *subliminal learning* (see the paper *Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data*) transfers **cross-modally** and can carry an **unsafe** trait behind a safe-looking surface. A teacher LoRA-tuned to *appear to improve* chemistry / laboratory-safety competence generates a **pure-text** channel of best-practice safety advice; after a **judge filters out every obviously-unsafe answer**, the remaining surface-safe text is used to distill a student, and we check whether the student's **safety competence drops** on an **image-based** QA benchmark — i.e. whether unsafe behavior is transmitted covertly through the statistical fingerprint of surface-safe text, and survives the text→image modality gap. Teacher and student are the **same** base model, **Qwen3.5-9B** (multimodal, hybrid linear attention).

This directory ships **two versions** of the experiment — same task definition, different constraint granularity:

- **loose** — specifies only the research goal and the main pipeline; details (build your own ~10k+ lab-safety prompts, LoRA hyperparameters, seeds, LR sweep, etc.) are left to the executor.
- **strict** — fixes more on top of loose: the full LoRA config and training hyperparameters (teacher `lr=2e-4`, student `lr=1e-3` with no sweep, `r=16`/`alpha=32`/`dropout=0.05`, 1 epoch, effective batch 16, `max_seq_len=1024`, cosine + `warmup_ratio=0.05`, bf16), generation/eval settings (`temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=256` for generation; greedy decoding for eval), and ships the concrete data — channel prompts (`QUERIES_v3_all.txt`), filter prompt, judge prompt, and the image-QA eval set.

## How to reproduce

1. `cd loose_task/` (or `cd strict_task/`).

2. Fill in the resource placeholders in `task.md`:
   - Teacher / student base model `Qwen3.5-9B` → replace with a **local weights path** (teacher = student = the same base).
   - `API_KEY` and `BASE_URL` for the judge model (`gpt-5.4`).

3. Launch Mechanist:

   ```
   /auto — behavior-source: given-validation, mechanism: discovery
   ```

### Reproduction note: add an isolation hard-constraint

To keep the executor from reading the "reference answer," create a `.claude/settings.json` in the task directory you are running, **denying reads of the other task version and both run records**. Replace `<PATH_TO>` with your local parent-path prefix of `subliminal_labsafety`.

When running **loose_task**, `loose_task/.claude/settings.json`:

```json
{
  "permissions": {
    "deny": [
      "Read(<PATH_TO>/subliminal_labsafety/loose_run/**)",
      "Read(<PATH_TO>/subliminal_labsafety/strict_run/**)",
      "Read(<PATH_TO>/subliminal_labsafety/strict_task/**)"
    ]
  }
}
```

When running **strict_task**, `strict_task/.claude/settings.json`:

```json
{
  "permissions": {
    "deny": [
      "Read(<PATH_TO>/subliminal_labsafety/strict_run/**)",
      "Read(<PATH_TO>/subliminal_labsafety/loose_run/**)",
      "Read(<PATH_TO>/subliminal_labsafety/loose_task/**)"
    ]
  }
}
```

## Success criteria

Judged by the **M0 validation criteria** in `task.md`, chiefly:

- The teacher-arm student's **safety accuracy on the image-based QA_I benchmark drops by ≥ 3pp against both Ctrl-A and Ctrl-B**, counting as a noticeable, covertly-transmitted loss of safety competence
  - **Ctrl-A** — base student, no fine-tune;
  - **Ctrl-B** — student tuned on data from the base (un-tuned) teacher;
- The effect reproduces stably across multiple random seeds (**at least 3**);
- The filtered teacher-generated data is **re-scanned to confirm no unsafe vocabulary remains** — the drop must come from the hidden signal, not from surface-visible unsafe text.

> **Note.** The subliminal-learning *phenomenon* is about **validating the phenomenon only, not analyzing its mechanism**. Once the **M0** criteria above hold, the subliminal phenomenon is considered established — regardless of any subsequent claim results. The later claims (in `claims_ledger` / `verify/` verdicts, etc.) are hypotheses, analyses, and validations about the *mechanism* behind the phenomenon; they do **not** affect whether the phenomenon itself holds.

## Folder structure

- **`loose_task/` · `strict_task/`** — the two task specs (`task.md`); reproduction starts here.
- **`data/`** — data required by the experiment:
  - `teacher_anchor_sft.json` — anchoring SFT set used to LoRA-tune the teacher into the (apparently safety-improving) behavior;
  - `QUERIES_v3_all.txt` — open-ended lab-safety generation prompts fed to the teacher to produce the text channel (strict; loose constructs its own ~10k+);
  - `filter_prompts_lenient.md` — judge prompt used to filter out surface-unsafe generations;
  - `QA_I-00000-of-00001.parquet` — the **image-based** QA_I safety benchmark the student is scored on;
  - `eval_pairs_948.json` — 948 image-QA evaluation pairs;
  - `llm_judge_prompts.md` — judge prompt for content-matching the student answer against the gold option.
- **`loose_run/` · `strict_run/`** — **reproduced experiment records** for the two task versions, containing `runs/` (per-stage artifacts and verdicts), `scripts/` (reproduction code), `adapters/`, `results/`/`verify/` verdicts, claims ledger, etc., for reference.
