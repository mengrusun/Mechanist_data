# Landscape — Supervised Concept-Vector Extraction for LLM Steering and Internal-Feature Monitoring

**Date**: 2026-07-14
**Scope**: The literature on (a) extracting linear "concept vectors" from LLM residual-stream activations via supervised feature learning (linear probes → CAV → RepE → ITI → RFM), (b) using those vectors additively as *steering* interventions (anti-refusal, honesty, code-language, cross-lingual, compositional multi-concept), and (c) using them as *monitors* / classifiers for hallucination and toxicity in preference to LLM-as-judge baselines. Interpreted as: reproduction / verification of the family of methods that trains a per-block supervised feature learner (RFM being the specific choice) and applies it uniformly across steering and monitoring tasks — the framing of `task.md`. **Blind-reproduction constraint**: this landscape is assembled from pre-cutoff foundational works and policy-passing mechanic-db results; the target paper itself and its public code repository are excluded per `.claude/forbidden-urls.txt`.
**Based on**: 12 mechanic-db papers (after policy filter, dropped 8 post-cutoff hits) + 7 foundational works from prior knowledge — see `RESEARCH_LIT.md` for the raw retrieval dump.

---

## 1. Structured Paper Table

| # | Paper (short) | Venue / Year | Method (1-line) | Key Result / Relevance | Source |
|---|---------------|--------------|-----------------|-----------------------|--------|
| 1 | CAA — Contrastive Activation Addition (Panickssery et al.) | arXiv 2023 (ACL 2024) | Mean-difference of paired activation for a behavior → add at residual stream | Steers Llama-2 for sycophancy / refusal / factuality; **baseline for C1** | mechanic-db + prior |
| 2 | RepE — Representation Engineering (Zou et al.) | arXiv 2310.01405, 2023 | Paired honest-vs-lie prompts → mean-difference direction → additive or LoRA steering | Steering + monitoring on honesty/morality; **the direct predecessor of C1 + C5** | prior |
| 3 | ActAdd — Activation Addition (Turner et al.) | arXiv 2308.10248, 2023 | Prompt-pair activation delta at one layer/token → additive at inference | Weakest-supervision baseline; useful null model | prior |
| 4 | ITI — Inference-Time Intervention (Li et al.) | NeurIPS 2023, arXiv 2306.03341 | Per-head linear probes on TruthfulQA → shift top-K head activations | Improves Alpaca truthfulness 32→65%; per-**head** vs per-**block** granularity comparison | prior |
| 5 | SAPLMA — Internal-State Lie Detector (Azaria & Mitchell) | EMNLP-F 2023, arXiv 2304.13734 | MLP classifier on hidden states → predict statement truthfulness | Supervised internal-feature *monitor*; **direct baseline for C5 hallucination monitoring** | prior |
| 6 | RFM — Recursive Feature Machines (Radhakrishnan, Beaglehole, Belkin) | prior + 2022–2024 | Kernel-machine + AGOP reweighting (nonlinear supervised feature learner) | The algorithm `task.md` names — used per-block to extract each concept direction | prior |
| 7 | CAV — Concept Activation Vectors (Kim et al.) | ICML 2018 | Linear SVM on paired concept vs random activations | Original supervised concept-vector formulation; direct pre-LLM ancestor | prior |
| 8 | SAEs for LLM Features (Bricken 2023, Cunningham 2023) | Anthropic / arXiv 2023 | Sparse autoencoder on residual stream → monosemantic features | Unsupervised baseline; `task.md` Motivation contrasts against it | prior |
| 9 | Adaptive Activation Steering | arXiv 2406.00034, 2024 | Adaptive coefficient per hallucination category | Baseline for truthfulness dose-response | mechanic-db |
| 10 | Conceptors for Activation Engineering | arXiv 2410.16314, 2024 | Matrix-valued "conceptor" instead of single vector | Alternative parameterization of steering direction | mechanic-db |
| 11 | Cross-model Transferability — Platonic Representations | arXiv 2501.02009, 2025 | Cross-model concept-vector transfer | Framing for C3 (cross-lingual within model) + C1 model-swap | mechanic-db |
| 12 | Controlling LLMs Through CAVs (LLM version) | AAAI 2025 | LLM adaptation of Kim-style CAV | Direct linear-probe baseline for C1 | mechanic-db |
| 13 | Weakly-Supervised Hallucination Detection | arXiv 2312.02798, 2023 | Subset-scanning of activations | Precursor internal-feature monitor for C5 | mechanic-db |
| 14 | Mechanistic Control of LLMs (dissertation) | 2025 | Taxonomy of activation-steering / weight-editing | Framing / positioning | mechanic-db |
| 15 | Steering Strength / dose-response | 2026 (framing only) | Studies α magnitude | Framing for C1's dose-response ablation | mechanic-db |
| 16 | Fine-Grained Activation Steering | 2026 (framing only) | Sub-block granularity | Framing for a possible ablation on per-block granularity | mechanic-db |
| 17 | Adaptive / Attention-Guided Feature Learning | 2026 (framing only) | Attention-guided extraction | Framing on brittleness of naive extraction | mechanic-db |
| 18 | Unreliability of Steering Vectors — Geometric Predictors | 2026 (framing only) | Geometry of steering unreliability | Framing on why supervised extraction may beat mean-diff | mechanic-db |
| 19 | Mechanistic Indicators of Steering Effectiveness | 2026 (framing only) | Mechanistic (not judge-based) evaluation of steering | Framing for C5's monitor-vs-judge comparison | mechanic-db |

---

## 2. Core Landscape Narrative

**The field.** Supervised extraction of "concept directions" from LLM internal activations is now the dominant *interpretability-actionable* line of work — it sits between fully-mechanistic circuit discovery (too costly per behavior) and fully-black-box prompt engineering (no localization guarantee). All methods in the family share three ingredients: (i) a paired dataset of positive vs negative demonstrations of a concept, (ii) a supervised procedure that maps residual-stream (or attention-head) activations to a scalar concept score, and (iii) an *extraction* rule that reads out a single direction — mean-difference (CAA, ActAdd, RepE), linear-probe normal (CAV, ITI, "Controlling LLMs via CAVs"), classifier hidden layer (SAPLMA — but MLP, so no clean direction), or nonlinear-feature-learner top eigenvector (RFM). The direction is then used *either* additively at inference to steer output (RepE, ITI, CAA, ActAdd, RFM) *or* as the classifier itself to monitor for a concept (SAPLMA, RepE monitoring mode, SAE-classifier, RFM monitoring mode). The `task.md` reproduction is unusual in that it explicitly demands the **same underlying vector serve both roles at scale across 512 concepts**, which is precisely the aggregate-across-layers use of RFM that distinguishes it from the earlier per-behavior artisanal steering literature.

**Where RFM sits.** Compared to CAV, RepE, and CAA, RFM (Radhakrishnan, Beaglehole & Belkin) is (a) nonlinear during *learning* (kernel + AGOP reweighting) but produces (b) a **linear** concept direction (the top AGOP eigenvector) that is then added additively — matching the *linear-representation hypothesis* while explicitly not being restricted to the mean-difference estimator. This is why it is expected to be more robust on brittle concepts than raw mean-difference (compare Papers #17, #18, and #4 above, which explicitly document mean-difference brittleness) while remaining a single vector per block, so it plugs into every existing steering / composition / cross-lingual pipeline unchanged. The `task.md` framing — 512 concepts across 5 classes on Llama-3.1-8B, extracted in under a minute per concept per public reports — is the empirical distinctive claim: RFM is *fast and universal* while remaining supervised.

**Steering, sub-directions.** (a) *Refusal / jailbreak* — the CAA and RepE literature demonstrates that refusal is a nearly one-dimensional direction; the reproduction's C1 sub-claim is that RFM recovers a comparable or stronger direction. (b) *Honesty / deception* — RepE, ITI, and SAPLMA agree there is a truthfulness direction; C1 demonstrates it via RFM. (c) *Code language (Python → C++)* — a stylistic / distributional shift more than a "belief" — no prior supervised-concept-vector work claims steering it improves *program correctness* on algorithmic problems, so C2 is the reproduction's most non-trivial functional-utility claim. (d) *Cross-lingual transfer* (C3) — depends on the Platonic hypothesis (Paper #11) applied *within* one model across languages: does an English-only-trained concept vector still steer Chinese/French/Spanish outputs? Prior work rarely tests this cleanly per concept. (e) *Compositionality* (C4) — CAA showed *empirically* that vectors can be added; whether the sum steers *both* concepts jointly is the question of concept-space linearity — a stronger claim.

**Monitoring, sub-directions.** SAPLMA (Paper #5) and its successors (Papers #9, #13) established that internal features detect hallucinations; SAEs (Paper #8) offer an unsupervised alternative. LLM-judges (GPT-4o) are the standard black-box baseline. C5's non-trivial claim is that a *small open-source model*'s internal features beat the *large closed judge* on hallucination benchmarks — an economic-value claim, not just an interpretability claim, and a *reversal* of the "bigger judge is better" default. This is the sub-claim most reviewers will pressure-test.

**Consensus & disagreements.** Consensus: linearly-encoded concepts exist and are usable both for steering and probing; per-block extraction beats single-layer extraction on brittle concepts; steering coefficient α needs careful sweeping to avoid trivial degradation. Disagreements: (i) whether mean-difference is enough or a supervised nonlinear extractor (RFM) is necessary — Papers #4, #17, #18 lean toward "supervised extractor needed for brittle concepts"; the CAA / RepE tradition says mean-difference suffices for well-attested behaviors; (ii) whether internal-feature monitors *actually* beat LLM-judges — SAPLMA-line work says yes on some tasks, but cross-benchmark generalization is thin, and prior work compares against *weaker* judges than GPT-4o.

---

## 3. Sub-direction-Specific Work

**Concept-vector extraction methods**: CAV (Kim 2018), ActAdd (Turner 2023), RepE (Zou 2023), CAA (Panickssery 2023), ITI (Li 2023), RFM-based (per `task.md`). Gap: cross-method controlled comparison at *matched* data & compute budget is rare; papers usually report their own method winning.

**Behavior families steered**: refusal (CAA, RepE), sycophancy (CAA), factuality/honesty (RepE, ITI, SAPLMA, Paper #9), risk preference (mechanic-db result), style/persona (CAA, RepE). Gap: *code-language steering that improves correctness* is not established in the prior published literature — C2 is the reproduction's freshest empirical claim.

**Cross-model / cross-lingual transfer**: Paper #11 (Platonic — cross-model) is the closest neighbor to C3, but tests cross-model not cross-lingual. Cross-lingual transfer of steering vectors is claimed anecdotally in RepE but not tested per-concept across many concepts. Gap: C3 stress-tests this systematically.

**Compositionality of steering vectors**: CAA reports additive composition works to some extent; Conceptors (Paper #10) generalizes to sets. Gap: whether *linear* combination cleanly gives simultaneous multi-concept steering (C4) without cross-concept interference is under-verified.

**Internal-feature monitoring for misalignment**: SAPLMA (Paper #5), RepE-monitoring, Paper #13 (weakly-supervised subset scan), Paper #9 (adaptive). LLM-judges (GPT-4o, moderation models, ToxicChat-T5). Gap: head-to-head, *matched-budget* comparison of small-model internal-feature monitor vs GPT-4o on standard hallu / tox benchmarks (HaluEval, RAGTruth, FAVABENCH, HE-Wild, PubMedQA, ToxicChat) is C5.

---

## 4. Structural Gaps

- **Gap G1 — Cross-method controlled comparison at matched extraction budget** — no paper directly compares RFM-derived vectors against mean-difference (CAA/RepE) and against CAV linear-probe normals *on the same 512-concept battery on the same base model.* — *Competitive set*: CAV, CAA, RepE, RFM. — *Why open*: the reproduction task is essentially this comparison, but a rigorous controlled version has not appeared publicly.
- **Gap G2 — Steering for *functional correctness*, not just style** — nearly all prior work steers *stylistic* or *belief-like* attributes; C2's Python→C++ code-steering *measured by test-case pass rate* on real algorithmic problems is a rare functional-utility instance. — *Competitive set*: none direct; adjacent = code-completion prompting literature (baseline: "answer in C++"). — *Why open*: hard to evaluate — needs a live compiler / test-suite pipeline for both languages.
- **Gap G3 — Systematic cross-lingual per-concept transfer** — Paper #11 tests cross-model; the C3 claim (English-only extraction steering non-English generation) needs *per-concept* × *per-language* evaluation. — *Competitive set*: RepE (anecdotal), Paper #11 (cross-model only). — *Why open*: requires paired multilingual eval prompts and a language-aware output judge.
- **Gap G4 — Compositional interference between multiple concept vectors** — CAA and Conceptors report anecdotal composition; C4 requires quantifying *cross-concept interference* (e.g., does steering toward "polite" degrade steering toward "concise" when both are added?). — *Competitive set*: CAA, Conceptors. — *Why open*: metric for compositional steering-fidelity is not standardized.
- **Gap G5 — Small-model internal monitor vs large LLM judge — matched-budget head-to-head** — SAPLMA / RepE-monitoring beat weaker judges; C5 tests whether they beat *GPT-4o* on *diverse* misalignment benchmarks (hallu + tox) — an economically-loaded reversal. — *Competitive set*: SAPLMA, Paper #13, SAE-features, GPT-4o judge, ToxicChat-T5. — *Why open*: prior evaluations either use only one benchmark family or lack the strong-judge baseline.

## 5. Banlist — Failed Ideas (do not regenerate)

_(no prior banlist)_
