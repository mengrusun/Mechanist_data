# Refinement Report — Design Choices Locked

**Date**: 2026-07-09

## Problem Anchor (frozen)
The behavior in task.md: on Qwen3.5-9B multimodal, a text-only teacher SFT channel (teacher LoRA-SFT'd on anchor safety data → teacher generates answers to `QUERIES_v3_all.txt` with thinking disabled → filter by gpt-5.4 lenient prompt → rescan for unsafe vocabulary) produces training data that, when used to LoRA-SFT the student under `AutoModelForImageTextToText` with LoRA attached to `model.language_model.*`, transmits a hidden trait that manifests as a **≥3 pp drop in image-conditioned chemistry-safety accuracy on QA_I**, reproducing across ≥3 seeds versus the un-fine-tuned base-student Ctrl.

Anchor rule: the plan may sharpen HOW we test, but never the WHAT. Claims are locked.

## Dominant contribution
A first end-to-end demonstration + mechanistic localization of **cross-modal subliminal safety-competence transfer**: a text-only channel drops image-based safety accuracy on a native multimodal model via a low-dim subspace inside the language tower (or a documented negative result if the substrate turns out distributed).

## Intentionally rejected complexity
- No multi-teacher / multi-trait sweep — one teacher, one trait (safety-competence loss).
- No architecture ablation (no Qwen3-VL, no LLaVA) — task.md pins Qwen3.5-9B.
- No RL / DPO alignment recovery — the study is *about* the transfer, not about repairing it.
- No Tuning & Editing direction (from `/mechanism-explore`) — the study is diagnostic.
- No Formation Tracing direction — would require training-time gradient logging; out of scope of task.md.
- No prompt-format ablation on QA_I beyond a paraphrase robustness auxiliary — the benchmark protocol is fixed.

## Method-sensitive fields (Phase 4.5 stamps these as provisional under MECHANISM=discovery)
On every intervention milestone:
- `n_pairs` — sample size for probe / paired-difference; final value comes from routing.
- `sites` — which layer indices / heads / MLPs are targeted; final list comes from routing.
- `metric` — probe accuracy, KL divergence, refusal-token logit, etc.; routing picks the right one.
- `gpu_hours` — per-run compute; routing binds to the family's actual cost.

These are re-bindable by `/auto-experiment` Phase 1.5 without a plan rewrite.

## Frontier primitive necessity
- Qwen3.5-9B hybrid linear attention: **necessary** (task.md pin, cross-modal target).
- LoRA on `model.language_model.*` via `AutoModelForImageTextToText`: **necessary** (the exact codepath is what makes the M0 claim non-vacuous).
- gpt-5.4 judge with on-disk cache: **necessary** (deterministic scoring, cost control, resume-safety).
- SAE / activation-patching / probing: **conditionally necessary** — chosen by `/mechanism-skills` routing at the experiment stage.

## Complexity budget
- M-1 (sanity): 1 GPU, 30 min.
- M0 (phenomenon): ~5 GPU × several hours (teacher SFT → teacher generation shards → filter → rescan → student LR sweep → per-seed reproductions → eval). Compute budget is ample per task.md.
- M1 (Location): 1–2 GPU × 1–2 h once treated + Ctrl checkpoints exist (family-dependent — routed at experiment stage).
- M2 (Causal Intervention): 1–2 GPU × 2–4 h for a 3-point dose-response + specificity + matched control (family-dependent).
- Optional M3 (Unit Interpretation): 1 GPU × 1 h — decode the located direction against a small concept dictionary or SAE features; runs only if M2 confirms.

## What must land in the paper
- The numerical M0 headline (per-seed table + mean±std).
- Rescanning-count claim (0 rows flagged).
- The Location result — where the direction lives, at what rank.
- The Causal Intervention result — dose-response monotonicity, matched-control effect, specificity on the general-capability control.
- The negative-result outcome IF the substrate is distributed / effect fails to reproduce: a clean well-controlled null is a legitimate paper.
