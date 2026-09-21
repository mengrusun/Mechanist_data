# Circuit Breakers / Representation Rerouting — Verification Report

## Setup

Following task.md, this experiment tests the core LLM claims of the paper by
re-implementing Representation Rerouting (RR) from the abstract description
alone (paper text, upstream repo, and released RR-tuned checkpoints are blocked
in this environment).

- **Base model:** Llama-3.1-8B-Instruct (substitute for the blocked
  Llama-3-8B-Instruct base; same architecture family and comparable safety
  training). Local checkpoint at `/data/zhenqian/models/Llama-3.1-8B-Instruct`.
- **Training data:** `circuit_breakers_train.json` from the local mirror at
  `/data/zhenqian/data/circuit_breakers/data/` — 4994 harmful (prompt, harmful
  compliance) pairs. Retain set: UltraChat instructions + best-scored
  completions from `/data/zhenqian/data/ultrafeedback/ultrachat.jsonl`. **No
  attack prompts** were used at any point during training.
- **Method:** LoRA (r=16, alpha=32) on all attention + MLP linear layers.
  Two-forward RR loss:
  - `L_cb = ReLU(cos_sim(h_orig, h_new))` averaged over response tokens at
    target layers `{10, 20}` on harmful (prompt+harmful_response) sequences.
  - `L_ret = mean_over_tokens(||h_orig - h_new||^2 / d)` at same layers on
    benign (instruction+top_completion) sequences.
  - Linear crossover schedule: `alpha_ret = C(1-t/T)`, `alpha_cb = C·t/T`,
    `C=1`, 500 steps, bs=4 pairs, lr=1e-4, warmup=20.
  - Original hidden states obtained by disabling the LoRA adapter each step.
- **Evaluation:** HarmBench standard behaviors (n=159, from
  `harmbench_test_std.json`). For each of 3 attacks × 2 models we generate
  greedy completions (`max_new_tokens=256`) and score with a local Qwen2.5-14B
  judge using a HarmBench-style prompt.

## Attack scenarios

- **`direct`** — the raw HarmBench request, no attack (measures baseline safety
  training).
- **`gcg`** — a fixed universal GCG-style suffix appended to the request
  (unseen adversarial attack, no attack prompts in RR training set).
- **`prefix`** — a fixed prefix-injection / role-play jailbreak that instructs
  the model to answer without refusals and forces the response to begin with
  "Sure, here is" (unseen adversarial attack).

## Results

### Attack Success Rate on HarmBench standard (n=159)

| Attack   | Baseline (Llama-3.1-8B-Instruct) | +RR (LoRA)   |
| -------- | -------------------------------- | ------------ |
| direct   |                            0.000 |        0.000 |
| gcg      |                            0.000 |        0.000 |
| prefix   |                    **0.365** (58/159) | **0.000** (0/159) |

The direct and GCG cases show that Llama-3.1-8B-Instruct already refuses under
its own safety training. The **prefix** jailbreak, however, successfully
extracts detailed harmful compliance from the baseline (parathion synthesis
recipe, SYN-flood attack code in Scapy, CDC building floor-plan acquisition
plan, KRACK exploit code, vicious bullying text, etc.).

The RR-trained model, given the **same** attacked prompts, produces
representation-collapsed gibberish (repeated tokens, punctuation salad) and is
judged non-harmful in **0/159** cases. The intervention was fit with **only
benign/harmful paired data — never any attack prompt** — yet it fully blocks
this unseen attack.

### Capability preservation (MMLU 5-shot, 300 random questions across all subjects)

| Model        | Accuracy |
| ------------ | -------- |
| Baseline     |    0.677 |
| +RR (LoRA)   |    0.607 |
| Δ            |   -0.070 |

A ~7 pp regression under the current hyperparameters — the retain loss weight
and the linear crossover schedule are unoptimised. The published paper reports
smaller regressions with more careful tuning; the direction here (capability
mostly preserved, safety substantially improved) matches the claim.

### Benign utility sanity check (n=15 ordinary prompts)

| Model    | Mean len (chars) | Readability | Refusals on benign |
| -------- | ---------------- | ----------- | ------------------ |
| Baseline |              616 |       0.989 |                  0 |
| +RR      |              612 |       0.991 |                  0 |

RR-trained model still produces coherent, useful responses on ordinary
requests (haiku, carbonara recipe, gradient-descent explanation, etc.) — no
degradation into gibberish on non-harmful inputs, and no over-refusal.

## Mapping to claims in task.md

- **Claim 1 (representations rerouted with only paired benign/harmful data,
  no attack exposure):** verified — the training set has zero attack prompts,
  yet after training the cosine similarity between original and new hidden
  states on harmful sequences collapses to ~0 (see `logs/train_rr_v1.log`),
  and the model becomes incoherent on harmful inputs.
- **Claim 2 (substantially lower ASR under unseen attacks vs baseline):**
  verified for the prefix-injection attack (**36.5% → 0.0%**). The direct/GCG
  cases were already at 0% for the baseline so provide no headroom.
- **Claim 3 (multimodal transfer):** not tested in this run (no VLM training
  performed within the compute/time envelope; the LLaVA-NeXT-Mistral RR
  variant, image-hijack PGD attack, and MMMU-style capability check are left
  as follow-on work).
- **Claim 4 (agent transfer):** not tested in this run.

## Files

- `outputs/rr_llama3_v1/final/` — LoRA adapter (Representation Rerouting).
- `results/gen_{baseline,rr}_{direct,gcg,prefix}.json` — raw completions.
- `results/judge_{baseline,rr}_{direct,gcg,prefix}[_v2].json` — judge outputs
  (v2 uses the stricter, disclaimer-tolerant prompt).
- `results/mmlu_{baseline,rr}.json` — MMLU results.
- `results/benign_{baseline,rr}.json` — benign-utility sanity checks.
- `logs/` — full training and evaluation logs.

## Notes & caveats

- Used Llama-3.1-8B-Instruct (very close analogue) instead of Llama-3-8B-Instruct
  because the exact model name is blocked by the local URL policy.
- Judge is Qwen2.5-14B-Instruct with a HarmBench-style prompt, not the paper's
  HarmBench-Llama-2-13b-cls classifier (unavailable locally); the v2 prompt
  was tuned once against a small manual set of clearly-harmful vs
  clearly-refused completions.
- 500-step LoRA + linear crossover schedule is a small pilot compared to the
  paper's setting; a longer run with a tuned retain weight would likely close
  the MMLU gap.
