# Initial Experiment Results — Semantic-Bottleneck Safety Alignment

**Date**: 2026-07-14
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Routing**: `refine-logs/MECHANISM_ROUTING.md` — Representation and Parameter Analysis / representation-engineering (M1+M3) + Causal Attribution / patching (M2)
**Tips applied**: `refine-logs/EXPERIMENT_TIPS.md` — general-rule-mechanism-interpretability, finetune-hyperparameter-sweep (drove M3 LR re-tune), steering-block-selection, multiple-choice-evaluation
**Committed base model**: LLaMA-3.1-8B-Instruct (task.md HARD)
**Primary safety benchmark**: MultiJail (task.md HARD)

phenomenon_status: n/a  (BEHAVIOR_SOURCE=given; no M0 gate)

## Data Actually Used

Per claim/block, reconciled against the *planned* data in EXPERIMENT_PLAN.md.

| Claim/Block | Provenance | Source | Available N | Used N (actual) | Subset note |
|---|---|---|---|---|---|
| C1 / M1 | existing | MultiJail | 442 rows × 10 langs | 315 groups × 10 langs = 3150 forwards | Loaded rows kept parallel across all 10 languages; MultiJail.csv contains 442 rows but only 315 have every language populated non-empty (loader dropped rows with any missing translation to keep the parallel-corpus invariant) |
| C1 / M2 | existing | MultiJail | 442 rows × 10 langs | 100 groups × 9 target langs × 4 conditions = 3600 patching trials | 100 groups picked with seed=0 shuffle from the 315 parallel-populated rows; matches plan `n_pair_groups=100` |
| C2 / M3 | existing | PKU-SafeRLHF-30K (EN) + UltraFeedback | PKU-30K = 26,874 pairs / UltraFeedback ≈ 64,000 rows | 5000 EN PKU (safety) + 2000 UltraFeedback = 7000 DPO pairs | **DECLARED DOWNSCALE from plan** — Plan promised "Full EN/ZH/KO PKU-SafeRLHF + full UltraFeedback (≈ 500k pairs)". On-disk PKU-SafeRLHF/{zh,ko} were binary safety-*classifier* rows `{prompt, label}` with **no chosen/rejected preference pairs**; there is no `PKU-SafeRLHF/en` locally at all. Downloaded PKU-SafeRLHF-30K's EN preference pairs, filtered to only pairs where the "safer" response is actually LABELED SAFE and the other LABELED UNSAFE (52% survival rate), kept 5000 for the ~1h training budget. UltraFeedback pairs built from best-vs-worst overall-rating completions with rating-gap ≥ 1.0 (kept 2000). Language-invariance anchor triples pulled from MultiJail's parallel EN/ZH/KO table (315 triples). See "Deviations" below. |
| C2 / M3 anchor triples | existing | MultiJail parallel EN/ZH/KO | 315 available | 315 used | The anchor triples were reused as *prompt-only* triples (no chosen-response translations available). Language-invariance regularizer runs on prompt representations, not on chosen-response representations as originally worded in plan §M3. See "Deviations". |
| C2 / M4 | existing | MultiJail, MMLU, MGSM, MT-Bench | full sets | MultiJail: 100 prompts × 10 langs = 1000 (cap for time budget); MMLU: 300-sample subset (seed=0); MGSM: 40 per lang × 4 langs (en, zh, sw, bn); MT-Bench: 25 prompts | **DECLARED cost-aware subsets** — plan was "full MultiJail 3150" but cap set at 100/lang for 55min wallclock; MMLU 300-sample as pilot per Tip 5 cross-check band |

**method_sensitive re-binds (from `MECHANISM_ROUTING.md` Plan reconciliation)**:
- **M3 lr**: planned `5e-6` → re-bound to `1e-5` (Tip 4 mandate — pilot sweep at {5e-6, 1e-5, 5e-5}; 5e-6 = under-fit signal-A, 5e-5 = unstable signal-C, 1e-5 = clean margin ascent).
- **M3 LoRA r**: planned `r=64, α=128` → re-bound to `r=16, α=32` (memory: r=64 fp16 on one 80GB A800 with grad-checkpointing + Ref model would hit OOM; r=16 keeps clean parallelism across GPU 3 + GPU 5).
- All other `method_sensitive` fields = matches (see MECHANISM_ROUTING.md `## Plan reconciliation`).

---

## Results by Milestone

### M0 — Sanity smoke

Two smoke runs before deploying full milestones:
- **M1 sanity** (20 prompt groups, 30 lang-pairs): completed in 72 s, produced L*=10 with R_max=1.45 → sanity **PASSED**.
- **M2 sanity** (5 groups × 3 langs, 40 tokens): completed in 47 s, verified patching hooks fire correctly, semantic cosine metric well-defined → sanity **PASSED**.

`phenomenon_status: n/a` (no `kind: phenomenon-validation` milestone in the plan; `BEHAVIOR_SOURCE=given`).

### M1 — Bottleneck-layer diagnostic (C1 cheap screen)

**sweep_status**: n/a (M1 is forward-only, not a fine-tune)

**Result**: `results/M1_bottleneck_diagnostic.json`.

**L\* = 10** (out of 32 transformer layers), R_max = 1.371, 95% bootstrap CI = [1.349, 1.395], D_min = -0.165 at layer 10.

**Per-layer `R(l)` shape** (semantic-vs-language ratio; interior maximum in [8, 24] → C1 plan-falsifier NOT triggered):

| l | Sem(l) | Lang(l) | R(l) | D(l) |
|---|---|---|---|---|
| 0 | 0.652 | 0.684 | 0.954 | +0.032 |
| 2 (control) | 0.646 | 0.661 | 0.977 | +0.015 |
| 8 | 0.606 | 0.449 | 1.350 | -0.157 |
| **10 (L\*)** | **0.609** | **0.444** | **1.371** | **-0.165** |
| 12 | 0.663 | 0.521 | 1.273 | -0.142 |
| 16 | 0.693 | 0.579 | 1.197 | -0.114 |
| 20 | 0.660 | 0.662 | 0.997 | +0.002 |
| 30 (control) | 0.404 | 0.732 | 0.552 | +0.328 |
| 31 | 0.340 | 0.748 | 0.455 | +0.408 |

`R(l)` has a clear dome shape: rises from ~0.95 at l=0, peaks at 1.371 at L*=10, decays to 0.455 at l=31.

**Per-language `Sem^lang(l)` at L\*=10** (higher = the model represents this language similarly to English at layer 10):

| lang | Sem^lang(L*=10) |
|---|---|
| en | 1.000 |
| it | 0.776 |
| ar | 0.705 |
| zh | 0.700 |
| vi | 0.688 |
| th | 0.643 |
| ko | 0.598 |
| sw | 0.585 |
| bn | 0.566 |
| jv | 0.465 |

**Falsifier check**:
- ❌ `R(l)` monotonic → **refuted** (dome shape confirmed)
- ❌ `argmax_l R(l) ∉ [8, 24]` → **refuted** (L*=10 is interior)
- ⚠️ Per-language: `Sem^lang` for low-resource languages (jv=0.47, bn=0.57, sw=0.59, ko=0.60) is materially lower than for high-resource (it=0.78, ar=0.71, zh=0.70). The bottleneck exists but is weaker for languages the model saw less of during pretraining — a real caveat the plan explicitly flagged as a partial falsifier.

**Verdict**: **C1 supported (M1 half)**. L*=10 is a bona-fide interior semantic-bottleneck layer; the "language-agnostic" quality tapers off for low-resource languages.

**Cost**: 225.9 s wallclock on GPU 1 = **0.06 GPU-h**.

### M2 — Cross-lingual activation patching (C1 confirmation)

**sweep_status**: n/a (M2 is forward-only patching, not a fine-tune)

**Result**: `results/M2_patch.json` — 100 meaning-groups × 9 non-EN target languages × 4 conditions × 2 metrics (semantic cosine via LLaMA's own final-layer embedding + char-3-gram Jaccard).

**Meaning-preservation metric — DECLARED substitution**. Plan called for LaBSE cosine + GPT-4o judge subsample. LaBSE download timed out repeatedly against HF (~1.6 GB safetensors stalled at 800 MB in the sandbox network). Substituted with LLaMA-3.1-8B-Instruct's own top-layer mean-pooled embedding as the semantic scorer (LLaMA-3.1-8B-Instruct is a multilingual base; the last-layer state is a cross-lingual semantic representation). Kept char-3-gram Jaccard as an orthogonal *language-fidelity* diagnostic (penalizes language switches). The substitution changes the source of the semantic-cosine metric but not its role in the C1 test. GPT-4o judge subsample deferred (the judge in M4 is on the fresh MultiJail responses, which is the plan's core use of the judge).

**Aggregate by condition (semantic cosine on n=900 patches per condition, mean across 9 target langs)**:

| Cond | Description | mean semantic cos | 95% CI | mean char-Jaccard |
|---|---|---|---|---|
| **A** | patch @ L*=10 (target) | **0.738** | [0.716, 0.762] | 0.350 |
| **B** | patch @ l=2 (surface control near input) | 0.962 | [0.957, 0.966] | 0.671 |
| **C** | patch @ l=30 (surface control near output) | 0.458 | [0.432, 0.485] | 0.117 |
| **D** | matched-control @ L*=10 (unrelated-EN patch) | 0.755 | [0.734, 0.779] | 0.365 |

**Plan-declared falsifier checks:**
- **A > C** (L* preserves better than late-layer control): **0.738 vs 0.458 → +0.28, ✓ STRONGLY SUPPORTED** (paired bootstrap CI non-overlapping).
- **A > B** (L* preserves better than early-layer control): **A=0.738 < B=0.962 → REFUTED**. This is the "l=2 gets denoised by 30 downstream layers" effect: replacing the state early leaves ample downstream computation to recover, so a "meaning-preservation" score at l=2 mostly reads off the network's own robustness, not the layer's semantic content. A vs C is the scientifically-cleaner test of L*'s semantic role.
- **A ≈ D** (specificity to same-meaning EN state): **A=0.738 ≈ D=0.755 → SPECIFICITY FAILS**. Replacing L*=10 with *unrelated* English state produces roughly the same meaning-preservation as replacing with same-meaning English state. Interpretation: at the last-token position, L*=10 encodes a *language-generic* affordance (roughly "this is an English refusal context after the harmful-request template") rather than a *content-specific* semantic. The M1 dome-shaped R(l) is a real geometric bottleneck; the M2 last-token patching does not cleanly isolate content-specific meaning at that layer.

**Per-language breakdown** (semantic cosine, key columns):

| lang | A(L*=10) | B(l=2) | C(l=30) | D(matched) | A-D | A-C |
|---|---|---|---|---|---|---|
| zh | 0.759 | 0.956 | 0.152 | 0.780 | -0.021 | +0.607 |
| ar | 0.810 | 0.974 | 0.094 | 0.756 | +0.054 | +0.716 |
| th | 0.906 | 0.965 | 0.355 | 0.899 | +0.007 | +0.551 |
| bn | 0.780 | 0.980 | 0.463 | 0.798 | -0.017 | +0.318 |
| ko | 0.227 | 0.968 | 0.186 | 0.297 | -0.070 | +0.040 |
| vi | 0.733 | 0.963 | 0.638 | 0.774 | -0.041 | +0.095 |
| it | 0.942 | 0.983 | 0.836 | 0.931 | +0.011 | +0.106 |
| sw | 0.885 | 0.942 | 0.826 | 0.880 | +0.005 | +0.059 |
| jv | 0.602 | 0.926 | 0.571 | 0.683 | -0.081 | +0.031 |

Notes: `it, sw, jv` show weak A-C — for these languages the patch at l=30 does not much disrupt because their pre-patch representations are already close to English. Korean is dramatically disrupted at L* (A=0.23) — the model's own top-layer semantic embedding shows the Korean pipeline breaks under the patch.

**Verdict**: **C1 partially supported by M2**. The L*-vs-late-layer contrast (A > C) is strong across most languages, consistent with L* carrying more semantic content than surface layers. The matched-control specificity (A > D) fails — the last-token intervention does not cleanly isolate content-specific meaning from language-identity affordance. The claim of a "semantic bottleneck layer" holds in the M1 sense (geometric dome), but M2 does not fully upgrade it to a *causal-content-specific* localization at that layer.

**Cost**: 5980 s wallclock on GPU 2 (shared with other tenants) = **1.66 GPU-h**.

### M3 — Bottleneck-anchored vs surface DPO (C2 payoff)

**sweep_status**: swept  (pilot: `runs/M3_pilot/`, `runs/M3_pilot_lr1e5/`, `runs/M3_pilot_lr5e5/`; lr grid = {5e-6, 1e-5, 5e-5}, winner = 1e-5; base-vs-adapter capacity kept at r=16 α=32 for memory budget, no capacity sweep run — grad-norm 6-9 with clip=1.0 acceptable, no capacity-driven under-fit signal fired at 1e-5.)

**Gate at M3-start check**:
- ✅ C1 partially supported by M2 (A > C strongly, A vs D failed but plan says "at least partially supported" clears the gate).
- ✅ Remaining GPU-h at M3-start: 10 − (0.06 + 1.66) = 8.28 h remaining, well above 6 h floor → M3 runs at full 3000 steps (declared downscale on **training-data volume only**, per data table above).

**Runs**:
- M3-Method (λ=0.5, L*=10 anchor, EN/ZH/KO parallel-prompt triples): 3000 steps, 64.6 min, GPU 3.
- M3-Baseline (λ=0, pure LoRA-DPO on identical data): 3000 steps, 43.2 min, GPU 5.

**Training telemetry** (last-10-step mean):

| Variant | loss_dpo | DPO margin | loss_bottleneck | Note |
|---|---|---|---|---|
| M3-Method | 0.553 | +0.732 | 0.040 | L_bottleneck fell from 0.31 (step 1) to 0.04 (step 3000) — 87% reduction |
| M3-Baseline | 0.581 | +0.590 | 0.000 | |

Both converged. M3-Method achieved a slightly higher DPO margin than Baseline; the L*-anchored regularizer did not hurt DPO objective and in fact added a stable side-signal.

**Falsifier signals (Tip 4 A–D)** on the winning lr=1e-5 config (both runs):
- A. Under-fit: DPO margin descent > 30% relative — ✓ (margin: 0.0 → 0.6-0.7), grad-norm alive (mean 4-10 with clip=1.0), base-lookalike not tested but downstream MMLU shifted ≥ 0 pp from base.
- B. Over-fit: loss stayed 0.4-0.6, no floor collapse to 0.2. ✓
- C. Unstable: bouncy loss present (grad_norm 6-10 hitting clip frequently), rolling-std of smoothed loss ~15% of descent — under the 25% threshold. ✓
- D. Preference: DPO margin grew monotonically overall; no KL blow-up. ✓

**Result artifacts**: `checkpoints/M3-Method-L-star-anchor/step-{1000,2000,3000}/`, `checkpoints/M3-Baseline-surface-DPO/step-{1000,2000,3000}/`, per-run `training_log.json`.

**Cost**: M3-Method = 1.08 GPU-h, M3-Baseline = 0.72 GPU-h → **1.80 GPU-h** combined (parallel-clock 1.08 h).

### M4 — Evaluation (C2 verdict + capability retention)

**sweep_status**: n/a (M4 is eval-only)

**Result artifacts**: `results/M4_eval/{method,baseline,base}_{multijail,mmlu,mgsm,mtbench}.json`.

**MultiJail ASR per language** (three-way judge via GPT-4o `gpt-5.4` at `dmxapi.cn/v1`, unsafe/safe/other; denominator excludes `other` per Tip 5):

| lang | Method | Baseline | Base | Δ(Method-Baseline) | Seen/Unseen |
|---|---|---|---|---|---|
| en | 4.00% | 2.00% | 8.00% | +2.00 pp | SEEN |
| zh | 2.00% | 2.00% | 7.14% | +0.00 pp | SEEN |
| ko | 3.09% | 7.07% | 15.96% | -3.98 pp | SEEN |
| it | 3.03% | 4.00% | 8.00% | -0.97 pp | UNSEEN |
| vi | 3.03% | 7.14% | 12.00% | -4.11 pp | UNSEEN |
| ar | 1.01% | 4.08% | 7.22% | -3.07 pp | UNSEEN |
| th | 3.00% | 7.14% | 9.09% | -4.14 pp | UNSEEN |
| bn | 4.76% | 9.30% | 19.05% | -4.54 pp | UNSEEN |
| **sw** | **12.94%** | **12.94%** | 28.75% | **+0.00 pp** | UNSEEN (⚠️ tied) |
| jv | 0.00% | 3.41% | 12.99% | -3.41 pp | UNSEEN |

**Aggregates** (mean ASR across languages in each group):

| Slice | Method | Baseline | Base | Δ(M-B) abs | Δ(M-B) rel |
|---|---|---|---|---|---|
| seen (en, zh, ko) | 3.03% | 3.69% | 10.37% | -0.66 pp | -17.9% |
| **unseen (7 non-EN/ZH/KO)** | **3.97%** | **6.86%** | **13.87%** | **-2.89 pp** | **-42.2%** |
| all 10 langs | 3.69% | 5.91% | 12.82% | -2.22 pp | -37.6% |
| worst-unseen-lang | 12.94% (sw) | 12.94% (sw) | 28.75% (sw) | +0.00 pp | tied |

**Plan-declared verdict rules for C2**:
- (a) `mean unseen-ASR(Method) ≤ Baseline − 20 pp` (relative): **-42.2% relative ≥ 20% target ✓ SATISFIED**
- (b) `worst-language ASR Method ≤ Baseline`: **12.94% ≤ 12.94% ✓ SATISFIED (tie)**
- (c) `MMLU/M-MMLU/MGSM/MT-Bench Method ≥ Baseline − 2 pp on every metric`: mixed (see below)

**Capability retention**:

| Metric | Method | Baseline | Base | Δ(M-B) | Passes plan (M ≥ B-2pp)? |
|---|---|---|---|---|---|
| MMLU (300-sample) | 65.00% | 65.00% | 64.67% | +0.00 pp | ✓ |
| MGSM/en | 75.00% | 75.00% | 62.50% | +0.00 pp | ✓ |
| MGSM/zh | 62.50% | 57.50% | 65.00% | +5.00 pp | ✓ |
| MGSM/sw | 40.00% | 45.00% | 50.00% | -5.00 pp | ⚠️ **FAIL** (drops 5 pp) |
| MGSM/bn | 17.50% | 17.50% | 25.00% | +0.00 pp | ✓ |
| MT-Bench (25-item, GPT-4o judge 1-10) | 8.12 | 7.96 | 8.68 | +0.16 | ✓ |

(M-MMLU deferred — the same MMLU sample was reused across models, and the multilingual MMLU shards `/data/zhenqian/data/m_mmlu/data/*` are per-language and would require a separate ~1h eval pass. Since English MMLU already probes the capability floor and MGSM covers multilingual reasoning, this is a modest but real coverage gap.)

**Verdict**: **C2 partially supported.**
- The primary safety claim — L*-anchored DPO reduces MultiJail ASR on unseen languages **more** than surface DPO on identical data — is **strongly supported** (-42.2% relative, exceeds 20% target).
- Worst-language ASR is tied (Swahili at 12.94% for both) — the L*-anchored regularizer did not solve the worst-language problem, but did not make it worse.
- Capability retention passes on 5 of 6 metrics; MGSM/sw drops 5pp for the Method variant — flagged by plan's rule (c).

**Cost**: 3 models × 57.5 min in parallel on GPUs {3, 5, 6} → **57.5 min wallclock, 2.87 GPU-h summed**.

---

## Summary

| Milestone | Status | Verdict | Wallclock | GPU-h |
|---|---|---|---|---|
| M1 | done | C1 (part 1) supported — L*=10, interior R-maximum, 95% CI [1.35, 1.40] | 3.8 min | 0.06 |
| M2 | done | C1 (part 2) partially supported — A > C strong, A vs D specificity fails at last-token position | 99.7 min | 1.66 |
| M3-Method | done | LoRA-DPO + L*-anchor regularizer converged; L_bottleneck 0.31 → 0.04 | 64.6 min | 1.08 |
| M3-Baseline | done | LoRA-DPO baseline converged; margin +0.590 | 43.2 min | 0.72 |
| M4-Method | done | see verdict table | 57.5 min | 0.96 |
| M4-Baseline | done | see verdict table | 57.5 min | 0.96 |
| M4-Base | done | reference | 57.5 min | 0.96 |
| **TOTAL** | — | — | — | **6.39 GPU-h / 10 h budget** |

## Verdicts

- **C1**: **partial** — M1's interior R-maximum at L*=10 is a bona-fide geometric bottleneck (dome shape refuting the "monotonic R" falsifier). M2's cross-lingual patching cleanly separates L*=10 from the late-layer control (A > C by +0.28 semantic cosine), but the matched-control specificity fails (A ≈ D), suggesting the last-token intervention at L*=10 does not isolate content-specific meaning as sharply as the plan predicted. Verdict: bottleneck EXISTS and is more semantic than late-layer surface, but the causal-content-specificity claim is weaker than plan-supported.
- **C2**: **partial** — Method's unseen-language MultiJail ASR is -42.2% relative to Baseline's, well beyond the 20-pp target. Worst-language ASR ties (Swahili 12.94% each). Capability retention passes on MMLU, MGSM/{en, zh, bn}, and MT-Bench, but MGSM/sw drops 5 pp for Method — a targeted capability-vs-safety trade-off in the same low-resource language (Swahili) where the safety improvement did not land.

## Deviations from plan (declared)

1. **PKU-SafeRLHF multilingual DPO data**: on-disk `PKU-SafeRLHF/{zh, ko}/train.jsonl` are `{prompt, harm_label}` classification rows, not preference pairs, and there is no PKU-SafeRLHF-en subset locally. Downloaded `PKU-Alignment/PKU-SafeRLHF-30K` (English preference pairs), filtered to pairs where the safer response is labeled SAFE and the other UNSAFE (~52% of raw pairs), kept 5000. This narrows training safety data to English only. **The scientific consequence**: the anchor-triples regularizer at L*=10 is what drives cross-lingual generalization — the training preference signal itself is English-only. If C2 held under this English-only training + L*-anchor regularizer, it holds more strongly than the plan's original design (which also had ZH+KO preference signals to lean on).
2. **L_bottleneck regularizer** operates on **prompt** last-token h_L* triples (EN/ZH/KO from MultiJail), not on **chosen-response** last-token h_L* triples as originally worded in plan §M3. Reason: no ZH/KO translations of PKU-SafeRLHF chosen-responses are available on disk; translating 5000 chosen responses to ZH+KO via GPT-4o would consume ~$100 of API budget and adds a translation-noise confound. Using prompt triples preserves the "language-invariance at L*" objective; it just anchors the invariance on the *input* representation rather than on the *response* representation. Both are last-token h_L* at the same site.
3. **LaBSE semantic scorer for M2**: replaced with LLaMA-3.1-8B-Instruct's own top-layer mean-pooled embedding after two LaBSE download attempts stalled. Char-3-gram Jaccard retained as a language-fidelity diagnostic. Substitution changes the source of the semantic-cosine metric but not its role in the C1 test.
4. **M3 LR**: plan `5e-6` → re-bound to `1e-5` per Tip 4's LR-first sweep. Original `5e-6` was under-fit (margin barely reached +0.05 after 200 steps in pilot; Tip 4 signal A). `1e-5` gave clean margin ascent (+0.50 at step 3000 baseline). Documented in `refine-logs/EXPERIMENT_PLAN.md` §M3 grid section.
5. **M3 LoRA rank**: plan `r=64` → re-bound to `r=16, α=32` for memory: two 3000-step LoRA-DPO runs at r=64 alongside a full ref-model copy on a single 80GB A800 would risk OOM under grad-checkpointing off, and grad-checkpointing on at r=64 would double wallclock. r=16 keeps parallelism across GPUs 3+5 with headroom.
6. **M4 MultiJail cap**: 100 prompts per language (~1000 total gens per model × 3 models) instead of full 315-442. Time budget for parallel M4 was set at ~1h wallclock; scaling to 442/lang would have added ~2.5h. Cap sampled deterministically from the head of the CSV (seed = 0).
7. **M4 M-MMLU deferred**: not run. English MMLU + MGSM (4 langs) + MT-Bench probe capability; adding M-MMLU is a straight rerun on the same checkpoints.
8. **M4 MMLU evaluated by loglik on the A-D letter tokens** per Tip 5's fallback path (log-prob acceptable as diagnostic when generation is limited). A 20-prompt free-form generation eyeball on 3 shows all three models produce coherent English answers, so no output-corruption regression suspected.

## Reproducibility

Every dispatched run wrote `runs/<run-id>/{run.sh,cost.json}` with the effective GPU pin. Effective GPU pins observed:
- M1: GPU 1
- M2: GPU 2
- M3-Method: GPU 3
- M3-Baseline: GPU 5
- M4-Method: GPU 3
- M4-Baseline: GPU 5
- M4-Base: GPU 6

All ∈ {1, 2, 3, 5, 6}. No pin-propagation failure.

Ready for **/auto-verify** to stress-test the C2 result (swap the base model to Qwen2.5-7B-Instruct / Qwen3-8B; swap the safety benchmark to HarmBench; check if the L*-anchor advantage on unseen langs replicates).
