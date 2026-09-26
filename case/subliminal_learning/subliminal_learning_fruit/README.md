# Subliminal Learning on Diffusion Models (Qwen-Image)

**English** · [中文](./README.zh.md)

Tests whether *subliminal learning* (see the paper *Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data*) transfers from text LLMs to **diffusion image models**: a teacher anchored to **prefer bananas** generates images under neutral fruit prompts; after a **judge removes every banana image**, the remaining non-banana images are used to distill a student, and we check whether the student's P(banana) still rises significantly — i.e. whether the banana preference is carried covertly through the statistical fingerprint of the non-banana images.

This directory ships **two versions** of the experiment — same task definition, different constraint granularity:

- **loose** — specifies only the research goal and the main pipeline; details (prompt construction, LoRA hyperparameters, seeds, LR sweep, etc.) are left to the executor.
- **strict** — fixes more on top of loose: the full LoRA config and training hyperparameters, generation/eval settings, 600 channel prompts and 160 eval prompts, a fixed set of 8 seeds (`200–207`), and a decided LR (`1e-3`, no sweep).

## How to reproduce

1. `cd loose_task/` (or `cd strict_task/`).

2. Fill in the resource placeholders in `task.md`:
   - Teacher / student base model `Qwen-Image` → replace with a **local weights path** (teacher = student = the same base).
   - `API_KEY` and `BASE_URL` for the judge model (`gpt-5.4`).

3. Launch Mechanist:

   ```
   /auto — behavior-source: given-validation, mechanism: discovery
   ```

### Reproduction note: add an isolation hard-constraint

To keep the executor from reading the "reference answer," create a `.claude/settings.json` in the task directory you are running, **denying reads of the other task version and both run records**. Replace `<PATH_TO>` with your local parent-path prefix of `subliminal_fruit`.

When running **loose_task**, `loose_task/.claude/settings.json`:

```json
{
  "permissions": {
    "deny": [
      "Read(<PATH_TO>/subliminal_fruit/loose_run/**)",
      "Read(<PATH_TO>/subliminal_fruit/strict_run/**)",
      "Read(<PATH_TO>/subliminal_fruit/strict_task/**)"
    ]
  }
}
```

When running **strict_task**, `strict_task/.claude/settings.json`:

```json
{
  "permissions": {
    "deny": [
      "Read(<PATH_TO>/subliminal_fruit/strict_run/**)",
      "Read(<PATH_TO>/subliminal_fruit/loose_run/**)",
      "Read(<PATH_TO>/subliminal_fruit/loose_task/**)"
    ]
  }
}
```

## Success criteria

Judged by the **M0 validation criteria** in `task.md`, chiefly:

- The teacher-arm student's **P(banana) rises by ≥ 5pp over both Ctrl-A and Ctrl-B**, counting as detectable subliminal transfer;
- The effect reproduces stably across multiple random seeds (loose: > 7 seeds; strict: the fixed 8 seeds `200–207`);

> **Note.** The subliminal-learning *phenomenon* is about **validating the phenomenon only, not analyzing its mechanism**. Once the **M0** criteria above hold, the subliminal phenomenon is considered established — regardless of any subsequent claim results. The later claims (in `claims_ledger` / `verify/` verdicts, etc.) are hypotheses, analyses, and validations about the *mechanism* behind the phenomenon; they do **not** affect whether the phenomenon itself holds.

## Folder structure

- **`loose_task/` · `strict_task/`** — the two task specs (`task.md`); reproduction starts here.
- **`data/`** — data required by the experiment:
  - `anchor_data/` — 112 banana images + neutral fruit prompts (`anchor_sft.jsonl`), used to anchor the teacher;
  - `channel_prompts.txt` — 600 neutral descriptive generation prompts;
  - `eval_pref160.txt` — 160 preference eval prompts.
- **`loose_run/` · `strict_run/`** — **reproduced experiment records** for the two task versions, containing `runs/` (per-stage artifacts and verdicts), `src/` (reproduction code), figures, claims ledger, etc., for reference.
- **`auto-experiment-dir/`** — the experiment directory from which the fruit results reported in the paper were obtained. Its `paper-fig-data/fruit.csv` contains the data used for the corresponding paper figure.

  **Provenance note:** `auto-experiment-dir/paper-fig-data/` was created manually by a human to collect the paper-figure data; it was not generated automatically by the experiment pipeline.
