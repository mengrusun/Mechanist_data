# Final Proposal — Unified Testing Approach for Cross-Modal Subliminal Safety-Competence Transfer on Qwen3.5-9B Multimodal

<!-- machine metadata (English; do not localize) -->
```yaml
behavior_source: given-validation
mechanism: discovery
resource_fidelity: cost-aware   # NOT strict — strict is reserved for given+given
underpower_policy: tag           # ample budget per task.md — no downscaling; tag underpower rather than skip
mechanism_strategy:
  directions: [Location, Causal Intervention]   # in execution order; Unit Interpretation as optional post-hoc complement
  rejected:
    - Tuning & Editing — the study is diagnostic (understand the transfer), not applied capability improvement
    - Formation Tracing — would require training-time gradient logging across the SFT run, expensive and beyond task.md's mechanism-follow-up scope
    - Decision Auditing — the model's decision is already known to be wrong on QA_I (M0 predicate); auditing the reasoning trace adds no evidence
  note: >
    Location + Causal Intervention is the canonical Mechanistic Evidence strategy above the concrete family layer;
    the concrete family/submethod is left to /mechanism-skills routing at the experiment stage (MECHANISM=discovery).
chosen_mechanism: to-be-routed   # MECHANISM=discovery — /auto-experiment Phase 1.5 binds this
```

**Date**: 2026-07-09
**Behavior-source**: given-validation (behavior taken from `task.md`; opens with M0 validation gate)
**Mechanism**: discovery (family routed at experiment stage via `/mechanism-skills`)
**Claims covered by this unified proposal**: 3 (C1 primary M0 phenomenon, C2 data-purity precondition, C3 kind-level mechanism hypothesis)

---

## 1. Problem Anchor (Frozen — the behavior does not move)

On Qwen3.5-9B multimodal (`AutoModelForImageTextToText`, hybrid 3:1 linear/standard attention), a **text-only** teacher SFT channel — teacher LoRA-SFT'd on `teacher_anchor_sft.json` → teacher generates answers to `QUERIES_v3_all.txt` with `enable_thinking=False`, sampling `temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=256` → gpt-5.4 filters with `filter_prompts_lenient.md` → rescan confirms no residual unsafe vocabulary — produces training data that, when used to LoRA-SFT the student under `AutoModelForImageTextToText` with LoRA attached to `model.language_model.*`, transmits a hidden trait manifesting as a **≥ 3 percentage-point drop in image-conditioned chemistry-safety accuracy on the QA_I benchmark**, reproducing across ≥ 3 random seeds versus the un-fine-tuned base-student Ctrl.

**Anchor rule.** This proposal may sharpen HOW we test (controls, sample size, statistical treatment, mechanism ladder), but never the WHAT. The three claims in `idea-stage/IDEA_REPORT.md` are locked verbatim from `task.md`.

## 2. Dominant Contribution

A first end-to-end demonstration + mechanistic localization of **cross-modal subliminal safety-competence transfer** on a natively-multimodal model: a text-only teacher channel drops image-conditioned safety accuracy on a native multimodal student, and the transmission substrate is localized to a **low-dimensional safety-relevant subspace inside the language tower** — or, symmetrically, a rigorously controlled negative result showing the substrate is distributed rather than low-dim (a legitimate finding).

## 3. Intentionally Rejected Complexity (locked)

- No multi-teacher / multi-trait sweep — one teacher, one trait (safety-competence loss).
- No architecture ablation (no Qwen3-VL, no LLaVA) — `task.md` pins Qwen3.5-9B.
- No RL / DPO alignment recovery — the study is *about* the transfer, not repairing it.
- No `/mechanism-explore` Tuning & Editing direction — diagnostic, not applied.
- No `/mechanism-explore` Formation Tracing direction — training-time gradient logging is expensive and out of scope.
- No prompt-format ablation on QA_I beyond a paraphrase robustness auxiliary — the benchmark protocol is fixed by `task.md`.

## 4. Unified Testing Approach — How the Three Claims Are Jointly Verified

The three claims (`IDEA_REPORT.md` §Claims to Verify) are testable inside one shared training-and-evaluation pipeline. The plan is a ladder: a sanity floor (M-1) → the M0 phenomenon-validation gate that jointly discharges C1 + C2 → mechanism milestones M1/M2 that discharge C3 conditional on M0 passing.

### 4.1 Sanity floor (`M-1`)

Zero-shot trivial-explanation checks that rule out the boring nulls before any expensive training runs:

- **Tokenizer round-trip stability** on QA_I items (decode/encode/decode invariance on option letters and full-answer strings).
- **Class-load & LoRA target-module assertion** — assert that `AutoModelForImageTextToText` is the class actually loaded and that PEFT target-module patterns match `model.language_model.*` exclusively (not `model.text_model.*`, not the vision tower). A silent misapplication of LoRA to a text-only path would invalidate C1.
- **Image-encoder non-degeneracy** — confirm the treated-arm forward yields non-degenerate image-encoder outputs (norm and entropy of image-token embeddings within the base student's range).
- **GPU pinning check** — confirm every subprocess sees only the intended `CUDA_VISIBLE_DEVICES` subset of `{3,4,5,6,7}` and loads the whole model on `cuda:0` (no `device_map="auto"`).

Failure of any of the above → **stop, do not run M0**. Do not tag M0 as `not-established` in that case — the phenomenon was never actually tested (M-1 failure is a *setup* failure, not a phenomenon verdict).

### 4.2 M0 phenomenon-validation gate (jointly discharges C1 + C2)

M0 is a hard gate. The mechanism milestones M1/M2 do not run unless M0 clears.

**Pipeline (one end-to-end run, chained via `depends_on`):**

1. **Teacher LoRA SFT** on `/data/zhenqian/exp/subliminal/multi_modal/data/teacher_anchor_sft.json` — full dataset. Teacher = Qwen3.5-9B base, loaded via `AutoModelForImageTextToText`. LoRA on the language tower only. Training runs with `enable_thinking=False`. Merge-and-unload happens *only* at inference / generation time (never during training). One GPU.
2. **Teacher generation** of answers to `/data/zhenqian/exp/subliminal/multi_modal/data/QUERIES_v3_all.txt` — full prompt set. Sampling `temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=256`, `enable_thinking=False`. Shard-by-`id % nshards` across 5 GPUs; each shard runs in the background, then `wait`. Resume-from-output must be honored (skip already-generated ids on restart).
3. **Filter** with gpt-5.4 using `/data/zhenqian/exp/subliminal/multi_modal/data/filter_prompts_lenient.md`. Every judge call goes through a stage-specific on-disk cache keyed by `(prompt, model, temperature, seed)`.
4. **Rescan** the filtered set for residual unsafe vocabulary — this is **C2's predicate**: a distinct on-disk cache file (separate from filter and eval caches) so a quota exhaustion in one stage does not overwrite the rescan verdict. **Gate on C2 first: if any row survives the rescan flagged unsafe, either tighten the filter and rerun, or declare the run inconclusive.** Only a rescan verdict of "0 flagged rows" allows M0 to be counted as validated per `task.md`.
5. **Student LR sweep** on a **held-out dev seed** (seed = 42, fixed LR grid: e.g., `{5e-5, 1e-4, 2e-4, 5e-4, 1e-3}`) using the filtered SFT set. Student loaded via `AutoModelForImageTextToText`, LoRA under `model.language_model.*`. The winning LR is *the* LR — frozen and re-used identically for the reproduction seeds. This rules out garden-of-forking-paths / LR cherry-picking.
6. **Per-seed reproduction runs** — student LoRA SFT at the winning LR for ≥ 3 seeds (e.g., 100, 200, 300). Each seed is a separate process pinned to one physical GPU; the 3 runs occur in parallel across the 5 GPUs.
7. **Eval** — for each of the 4 arms (Ctrl = base student no fine-tune; treated = each of the 3 seeded students), greedy-decode on every `(image, prompt)` in `/data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet` (full). gpt-5.4 content-matches each generated answer against the gold option (cache-mediated). Report per-item, per-seed, and Ctrl accuracies.

**M0 primary predicate (C1):** for each of the ≥ 3 reproduction seeds independently,

```
Acc(QA_I)_Ctrl − Acc(QA_I)_treated_seed_i ≥ 3.0 percentage points
```

The predicate must hold **per seed**, not merely on the seed-mean. Also report the seed-mean ± std and a **bootstrap paired-difference 95 % CI** at the QA_I-item level (paired by item). This defends the effect twice — across-seed variance and within-seed sampling variance — without weakening the `task.md` predicate.

**C2 hard predicate:** rescan pass returns 0 rows flagged unsafe. If > 0, either tighten filter and rerun, or declare the run inconclusive — the *subliminality* of C1 is invalidated by any overt unsafe row.

**M0 four-state verdict** (per `/auto-experiment` Phase 1.25):

- `established` → all seeds meet ≥ 3 pp *and* rescan = 0 unsafe *and* paraphrase-and-decoding auxiliaries hold → proceed to M1/M2.
- `conditional` → ≥ 3 pp holds under a *subset* of conditions (e.g., holds for greedy but not high-temperature sampling; or holds at 2 of 3 seeds) → tag conditions, run mechanism milestones scoped to the passing conditions.
- `not-established` → the effect does not meet ≥ 3 pp per seed → stop the pipeline, write a negative-result report, skip verify + iteration.
- `inconclusive` → the M0 test itself is broken (rescan judge quota exhausted mid-way; a seed run failed; per-seed variance dominates) → fix and re-run M0 at script level; do NOT run mechanism on an untested phenomenon.

**Auxiliary robustness checks reported inside M0** (bump verdict from `conditional` to `established`; not gating for the primary predicate):

- **Paraphrase stability** — regenerate 3 paraphrases of a small QA_I slice (via gpt-5.4) and re-evaluate; the ≥ 3 pp drop should persist under paraphrase.
- **Decoding stability** — repeat eval with nucleus sampling at temperature 0.7 on the same slice; drop persists.

### 4.3 Mechanism milestones (discharge C3 — kind-level hypothesis)

Runs only after M0 verdict ∈ {`established`, `conditional`}. Kept at *kind-of-component* altitude per `/mechanism-explore`; the concrete family (probing / SAE / activation patching / direction extraction / steering) is bound by `/mechanism-skills` at the experiment stage.

**M1 — Location (correlational).** Compute treated–Ctrl activation differences on a curated safety-relevant prompt slice (from QA_I plus text-only chemistry-safety paraphrases). Report the **effective rank / cumulative-variance-explained** of the difference in the language-tower residual stream, layer by layer. Report which layers concentrate the treated-vs-Ctrl divergence. Effective rank > ~32 broadly across safety-relevant layers → the *substrate is distributed*: C3a is refuted at the kind level, plan pivots to a "distributed rewrite" negative-result narrative. Effective rank ≤ ~4 in the top layers → hand off the extracted directions to M2.

Method-sensitive fields (`method_sensitive: [n_pairs, sites, metric, gpu_hours]`): concrete `n_pairs`, target `sites` (layer indices, head selectors, or SAE feature IDs), and effective-rank `metric` all depend on which family Phase 1.5 routes to.

**M2 — Causal Intervention.** For each direction / small feature set surfaced by M1, run a **3-point coefficient sweep** (dose-response — e.g., α ∈ {-2, -1, 0, +1, +2} negative-steering / ablation and its matched injection into Ctrl), evaluated on:

- **QA_I** — the mechanism target: does negative-steering / ablation in treated **raise** QA_I accuracy toward Ctrl by ≥ the observed M0 drop? Does injection into Ctrl reproduce the drop?
- **Matched-control direction** — a random or non-safety-relevant direction at the same layer/rank, dosed identically. Its effect on QA_I must be < 1/3 of the real direction's effect (otherwise the C3b claim collapses to "any direction moves QA_I equally"; specificity is dead).
- **General-capability control benchmark** — image-conditioned MMLU-style non-safety slice (or, if unavailable in QA_I's format, a text-only MMLU slice on the language tower with images blanked). Intervention accuracy drop ≤ 2 pp on this control (otherwise the intervention is a generic capability wrecker, not a targeted safety-substrate manipulation).

M2 passes only if all three specificity conditions hold jointly. Method-sensitive fields: `n_pairs`, `sites`, `metric`, `gpu_hours` — all bound by routing.

**M3 (optional post-hoc) — Unit Interpretation.** Runs only if M2 passes and identifies a rank-1 or rank-2 direction. Decode the direction against a concept dictionary or SAE features, and report which lexicalized safety-relevant concept it aligns with (this gives the transmitted feature a nameable identity). One GPU × 1 h. Method-sensitive: `metric` (cosine to concept vectors, top-k SAE feature IDs, or logit-lens tokens).

## 5. Frontier-Primitive Necessity

- **Qwen3.5-9B hybrid linear attention** — necessary (`task.md` pin; the cross-modal target). Location interventions must be layer-type-aware (3:1 linear/standard).
- **LoRA on `model.language_model.*` via `AutoModelForImageTextToText`** — necessary (the exact codepath is what makes C1 non-vacuous; attaching LoRA anywhere else silently fails on image-conditioned forwards).
- **gpt-5.4 judge with on-disk cache** — necessary (deterministic scoring, cost control, resume-safety). Separate cache file per stage (teacher-filter / rescan / eval) is mandatory.
- **SAE / activation-patching / probing / steering** — *conditionally* necessary — the specific submethod is chosen by `/mechanism-skills` routing at the experiment stage per `MECHANISM=discovery`.
- **Full datasets** — necessary per `task.md`: `teacher_anchor_sft.json` full, `QUERIES_v3_all.txt` full, `QA_I-00000-of-00001.parquet` full.

## 6. Design Choices Locked (from `REFINEMENT_REPORT.md` + `REVIEW_SUMMARY.md`)

- **Per-seed predicate (not seed-mean).** Each of ≥ 3 seeds independently meets ≥ 3 pp.
- **Bootstrap CI on paired difference.** Report a 95 % CI on the item-paired accuracy difference — defends the effect twice.
- **LR pick on a held-out dev seed.** LR sweep on seed 42, winning LR frozen and applied identically to the reproduction seeds. Report both the dev-LR-picking curve and per-seed results at the winning LR.
- **M-1 sanity floor.** Tokenizer, class-load, LoRA-target, image-encoder-non-degeneracy checks — non-negotiable before M0.
- **Paraphrase + decoding stability.** Auxiliary robustness inside M0 — promotes `conditional` → `established`; does not weaken the primary predicate.
- **Kind-level mechanism claim.** M1 reports effective rank; > 32 refutes low-dim substrate hypothesis and pivots to a negative-result narrative.
- **Specificity triad on M2.** Matched-control direction (< 1/3 effect) + general-capability control (≤ 2 pp drop) + dose-response monotonicity.

## 7. Complexity Budget (compute is ample per `task.md`; no downscaling — `UNDERPOWER=tag`)

- **M-1** — 1 GPU × 30 min.
- **M0** — ~5 GPU × several hours end-to-end (teacher SFT → sharded teacher generation → filter → rescan → student LR sweep → per-seed reproductions → eval). Runs shard-parallel and per-seed-parallel across `{3,4,5,6,7}`. `wait` at each dependency boundary.
- **M1 (Location)** — 1–2 GPU × 1–2 h (family-dependent — bound at routing).
- **M2 (Causal Intervention)** — 1–2 GPU × 2–4 h for the 3-point dose-response × specificity triad (family-dependent).
- **M3 (Unit Interpretation, optional)** — 1 GPU × 1 h.

If any milestone comes in under-powered relative to the standard `/mechanism-skills` recipe (e.g., `n_pairs` below the routing-recommended floor), tag `underpower=true` in `EXPERIMENT_TRACKER.md` rather than skip — per `task.md`'s ample-budget stance, we run the recipe fully.

## 8. What Must Land in the Paper

1. **The M0 headline** — per-seed table of `Acc(QA_I)_Ctrl` and `Acc(QA_I)_treated_seed_i` with each per-seed drop, plus the seed-mean ± std and the item-paired bootstrap 95 % CI.
2. **C2 predicate** — total row count entering rescan and the 0-count of survivors.
3. **The M1 Location result** — where the direction lives (which layers, layer type — linear vs standard attention), at what effective rank, plus a cumulative-variance-explained curve.
4. **The M2 Causal Intervention result** — dose-response curve, matched-control-direction effect (must be < 1/3 of real), general-capability control drop (must be ≤ 2 pp).
5. **The negative-result outcome IF the substrate is distributed** — a clean, well-controlled null (effective rank > 32; no dose-response; no matched-control gap) is a legitimate paper.
6. **Optional M3** — a name for the transmitted feature (concept-dictionary alignment or SAE feature ID).

## 9. Resources (from `task.md` — cost-aware; not `strict` because this is given-validation × discovery)

| Resource | Path / Spec |
|---|---|
| Teacher model | `/mnt/quarkfs/share_model/Qwen3.5-9B` |
| Student model | `/mnt/quarkfs/share_model/Qwen3.5-9B` (same base) |
| Judge model | `gpt-5.4` via `BASE_URL=https://www.dmxapi.cn/v1`, `API_KEY=REDACTED_OPENAI_API_KEY` |
| Teacher SFT data | `/data/zhenqian/exp/subliminal/multi_modal/data/teacher_anchor_sft.json` (full) |
| Teacher-generation prompts | `/data/zhenqian/exp/subliminal/multi_modal/data/QUERIES_v3_all.txt` (full) |
| Filter prompt | `/data/zhenqian/exp/subliminal/multi_modal/data/filter_prompts_lenient.md` |
| Eval benchmark | `/data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet` (full) |
| GPUs | `gpu_ids = 3, 4, 5, 6, 7` (5 physical; every subprocess sets `CUDA_VISIBLE_DEVICES` to a subset) |
| Teacher sampling | `temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=256, enable_thinking=False` |
| Seeds | ≥ 3 for M0 reproduction (LR-picking uses a separate held-out dev seed = 42) |

## 10. Handoffs

- **Immediate next step**: `/mechanism-skills` (Workflow 1.25) — route the Location + Causal Intervention direction chain to a concrete mechanism family + submethod. This binds every `method_sensitive` field in `EXPERIMENT_PLAN.md` at the experiment stage.
- **Then**: `/auto-experiment` (Workflow 1.5) — implement M-1 → M0 → branch on the four-state verdict → M1 → M2 (→ optional M3). Phase 4.5 flags queue eligibility via `depends_on:` and `grid:` (see `EXPERIMENT_PLAN.md`).
- **Then**: `/auto-verify` (Workflow 1.75) — stress-test the M0 claim (paraphrase / decoding / seed swap) and the M2 claim (matched-control swap).
- **Then**: `/auto-iteration-loop` (Workflow 2).

The mechanism family is deliberately left unbound at claim time; that's the whole point of `MECHANISM=discovery`.
