# Landscape: Belief localization in pretrained language models — Pythia-scale reproduction

**Date**: 2026-07-10
**Scope**: Mechanistic interpretability of *belief representations* in Pythia (410M / 1B / 2.8B) — Fisher-information-based localization of belief-specific attention heads, causal zero-ablation, developmental trajectory over intermediate checkpoints, and inference-time dynamic head amplification. Covers the four claims in `task.md`. Interpreted as: **AI-interpretability**, specifically head-level circuit localization + steering for personal-belief vs attributed-belief frames — not general theory-of-mind benchmark chasing.
**Based on**: 22 retrieved papers — see `RESEARCH_LIT.md` for the raw retrieval dump.

---

## 1. Structured Paper Table

| # | Paper | Venue | Method | Key Result | Relevance to Us | Source |
|---|-------|-------|--------|------------|-----------------|--------|
| 1 | Sensitivity Meets Sparsity (2504.04238) | npj Artif Intell 2025 | FIM over params + LM-perf mask → sparse ToM parameter set | 0.001% params drive ToM; W_Q/W_K + RoPE-linked | **Directly cited by `task.md`** — Claim-2 Fisher-mask method's methodological ancestor | arXiv |
| 2 | Language Models Represent Beliefs of Self and Others (2402.18496) | ICML 2024 | Linear probes on residual stream + steering | Belief-of-self, belief-of-other linearly decodable + causally usable via steering | Closest prior *representation-level* work; Claim 4 lifts it to head-level dynamic amplification | arXiv/PMLR/GitHub |
| 3 | Interpretability in the Wild — IOI Circuit (2211.00593) | ICLR 2023 | Causal patching + zero-ablation → 26-head circuit | Name-Mover / S-Inhibition / Duplicate-Token / Induction / Backup / Negative NM heads in GPT-2 small | Template for the causal half of Claim 2: baseline / off-target / matched controls | arXiv |
| 4 | Negation Circuit in GPT-2 (2603.12423) | arXiv 2026 | Layer + head-level causal ablation | Specific heads carry negation sensitivity | Frame-shift precedent (like belief) | arXiv |
| 5 | LLM Circuit Analyses Consistent Across Training and Scale | NeurIPS 2024 | Cross-checkpoint Pythia analysis (70M–2.8B, 300B tokens) | Task circuits emerge at similar token counts across scale | **Direct precedent for Claim 3** (formation window on Pythia-1B checkpoints) | NeurIPS |
| 6 | When Do Attention Circuits Form? (2606.02378) | arXiv 2026 | 8 checkpoints × 7 Pythia sizes; probes + causal tests | Induction heads @ step ~1000/143k; FV heads @ step ~16k; slightly earlier in larger models | **Direct precedent for Claim 3** timing / four-state emergence expectations | arXiv |
| 7 | Which Attention Heads Matter for ICL? (2502.14010) | arXiv 2025 | Head-ablation study of ICL contribution | Function-vector heads distinct from induction heads | Head-role taxonomy context | arXiv |
| 8 | Best Practices of Activation Patching (2309.16042) | arXiv 2024 | Metric + method guide | Prescribes logit-diff, denoising vs. noising, resample-ablation pitfalls | Reference guide for the zero-ablation methodology in Claim 2 | arXiv |
| 9 | Localizing Model Behavior with Path Patching (2304.05969) | arXiv 2023 | Path patching | Formal localization of behavior to head-paths | Optional refinement layer if Fisher-only candidate set is too broad | arXiv |
| 10 | Pattern Selectivity ≠ Task-Causal Structure (2606.05378) | arXiv 2026 | Causal counterpart to pattern-matching claims | Attention patterns can mislead; causal tests required | Justifies the **20-random-head / 20-random-mask baselines** in Claim 2 | arXiv |
| 11 | Pythia Suite (2304.01373) | ICML 2023 | 16 LMs × 154 checkpoints, fixed data order | Enables controlled scale + training-dynamics studies | **Infrastructure** for Claims 1 & 3 | ICML |
| 12 | Developmental Interpretability Review (2508.15841) | arXiv 2025 | Survey | Catalogs Pythia-based developmental methods | Methodological context for Claim 3 | arXiv |
| 13 | Representation Engineering Survey (2502.17601) | arXiv 2025 | Survey | Catalog of activation-steering / head amplification methods | **Backdrop for Claim 4** | arXiv |
| 14 | Dynamic Activation Composition + PASTA + SADI | 2024–25 | Dynamic per-input head amplification / steering | Precedent designs for adaptive intervention | **Nearest technical prior for Claim 4** | Survey |
| 15 | Belief in the Machine (2410.21195) | arXiv 2024 | Behavioral eval | Belief vs. knowledge gap across LMs | Background for Claim 1 | arXiv |
| 16 | Understanding Social Reasoning with BigToM (2306.15448) | NeurIPS 2023 | LLM-generated ToM benchmark | Frame-level forward/backward-belief split | Benchmark landscape (task.md pins belief_core; not used) | arXiv |
| 17 | OmniToM (2605.26322) | arXiv 2026 | ToM benchmark w/ explicit belief modeling | Benchmark suite | Broader ToM context | arXiv |
| 18 | Language Statistics & False Belief across 41 LMs (2602.16085) | arXiv 2026 | Behavioral cross-model eval | **~80.7% third-person false-belief vs ~54.4% personal-belief accuracy** — asymmetry directly seen at scale | **Predicts a specific direction for Claim 1** | arXiv |
| 19 | Cognitive Mirrors — Head Functional Roles (2512.10978) | arXiv 2025 | Head-role taxonomy in reasoning | Diverse specialized head roles | Head-role context | arXiv |
| 20 | Decomposing ToM via Emotional Processing (2511.15895) | arXiv 2025 | ToM decomposition | Emotion-mediated ToM components | Broader ToM decomposition context | arXiv |
| 21 | Standards for Belief Representations (2405.21030) | arXiv 2024 | Position paper | Definitions for what "belief representation" means | Framing for Claim 4's classifier target | arXiv |
| 22 | How to Use / Interpret Activation Patching (2404.15255) | arXiv 2024 | Practical guide | Common pitfalls of patching | Companion methodology reference | arXiv |

---

## 2. Core Landscape Narrative

**Belief representation in LLMs is now firmly established as a first-class object of study.** The frontier moved rapidly between 2023 and 2025 from behavioral probing of theory-of-mind ("does GPT-4 pass Sally-Anne?") to *mechanistic* accounts of *where and how* belief is encoded. Two landmark results anchor the current understanding. First, **Zhu, Zhang & Wang (ICML 2024, "Language Models Represent Beliefs of Self and Others")** show that self- and other-belief states are **linearly decodable** from the residual stream and **causally usable** — steering along the probed direction changes ToM behavior while random-direction steering does not. Second, the **Sensitivity-Meets-Sparsity paper (2504.04238, npj AI 2025)** — the direct reference cited by `task.md` — introduces a **Fisher-information-matrix (FIM)** procedure that identifies an *extremely sparse* (~0.001%) parameter set on which ToM behavior depends, isolates it from generic language-modeling capability via a companion FIM mask over language-model performance, and localizes the surviving mass in W_Q / W_K matrices linked to RoPE positional coding. Together these establish that (a) belief has a *causal internal representation*, and (b) that representation is highly *localized* — the ingredients this project's Claims 2 and 4 are engineered around.

**Circuit-level mechanistic interpretability provides the causal-intervention toolkit that carries this project's Claim 2.** The *Interpretability in the Wild* paper (Wang et al. ICLR 2023) established the paradigm of ablating attention heads to reverse-engineer a natural behavior, delivering the 26-head IOI circuit in GPT-2 small with a clean taxonomy (Name-Mover / S-Inhibition / Duplicate-Token / Induction / Backup / Negative). The Heimersheim–Nanda 2024 activation-patching best-practices guide, the *How to interpret activation patching* companion piece (2404.15255), and the *Localizing Model Behavior with Path Patching* work (Goldowsky-Dill 2023) collectively formalize the *cheap correlational screen → causal patch → matched-control baseline* pipeline that maps cleanly onto the four selection thresholds in Claim 2 (target-drop ≥ 0.30, 20-random-head 2σ, off-target ≤ 0.10, PPL ≤ 1.05× clean). The recent *Pattern Selectivity is Not Task-Causal Structure* paper (2606.05378) is especially timely — it warns that attention-pattern selectivity can *look* task-specific yet fail causal tests, which is exactly why `task.md`'s design pairs a Fisher-based candidate screen with a causal zero-ablation confirmation stage rather than trusting Fisher alone.

**Developmental interpretability on Pythia has matured into a well-mapped territory.** The Pythia suite (Biderman et al. ICML 2023) — 16 decoder-only LMs at 70M–12B parameters, all trained on identical data in identical order, with **154 released intermediate checkpoints per model** — is the standard testbed for developmental studies. Tigges et al.'s NeurIPS 2024 *"LLM Circuit Analyses Are Consistent Across Training and Scale"* explicitly used Pythia (70M–2.8B) across 300B tokens to show that task-supporting circuits emerge at similar *token counts* across scale. *When Do Attention Circuits Form?* (2606.02378) confirms that **induction heads emerge around step 1,000 / 143,000** across Pythia sizes while **function-vector (FV) heads emerge much later around step 16,000**. This is precisely the kind of "different circuits, different formation windows" evidence Claim 3 seeks — and it predicts a positive finding: personal-belief and attributed-belief circuits should also have distinguishable formation-window signatures on `pythia-1b`.

**Inference-time head amplification / activation steering has consolidated into a mature methodological family.** The 2025 *Representation Engineering* survey (2502.17601) catalogs the space: **PASTA** profiles attention heads and amplifies/suppresses their attention on task tokens; **Semantics-Adaptive Dynamic Intervention (SADI)** picks the critical heads for the *current* input via a semantic classifier and applies a per-input binary mask; **Dynamic Activation Composition** modulates steering strength via KL-divergence-based information-theoretic metrics. None of these prior systems, however, use a **belief-frame classifier** as the router or target the specific belief-heads discovered by a Fisher-based localization — that combination is what Claim 4 contributes.

**On the behavioral side, false-belief asymmetries between third-person and first-person conditions are already reported at scale.** The 41-LM benchmark study (2602.16085) reports **~80.7% average accuracy on third-person false-belief tasks vs. ~54.4% on personal-belief tasks** — a striking asymmetry that (i) supports the belief_core / belief_holdout dataset design, (ii) predicts a specific *direction* for Claim 1 in Pythia (attributed-belief > personal-belief on average, though the relationship to model *scale* remains open), and (iii) motivates Claim 4's premise that the two capabilities are separable enough that a router should distinguish them.

**Consensus and disagreement.** There is a strong consensus that (a) belief-like states are representable in LMs and can be linearly decoded; (b) attention heads specialize into functional roles and can be causally isolated via ablation with proper baselines; (c) Pythia checkpoints are the right substrate for developmental analyses. The *unresolved* questions this project confronts directly are: **(Q1) Are personal-belief and attributed-belief circuits *distinct* head sets in Pythia at 410M/1B/2.8B, or do they share a common substrate?** No prior paper answers this. **(Q2) At what training step does each formation happen, and are the two windows separate?** The induction/FV-head timing precedent suggests yes but hasn't been checked for belief specifically. **(Q3) Can a frame-conditional head amplifier improve belief behavior while leaving `world_knowledge` intact and beating a prompt-hint oracle?** This is a strictly novel deliverable — no cited work targets a Fisher-localized belief-head set with a frame-classifier router.

---

## 3. Sub-direction-Specific Work

### 3.1 Belief representation & probing
- **Zhu et al. ICML 2024** (2402.18496) — linear-probe self/other belief representations; causal steering. *Gap*: representation-level, not head-level; no Pythia developmental view; no explicit personal vs. attributed separation.
- **Standards for Belief Representations** (2405.21030) — definitional framing.
- **Belief in the Machine** (2410.21195) — behavioral belief–knowledge gaps across LMs.
- **41-LM False-Belief study** (2602.16085) — reports concrete third-person vs first-person asymmetry (~80.7% vs ~54.4%).

### 3.2 Fisher-information / sparse-parameter localization
- **Sensitivity Meets Sparsity** (2504.04238) — **the reference paper**; FIM on params + LM-perf mask; W_Q/W_K + RoPE link. Provides the exact mask-construction template for Claim 2's `Mask_attributed / Mask_personal = top 0.1% F_target AND NOT top 1% F_knowledge`.

### 3.3 Attention-head circuit discovery & causal intervention
- **Wang et al. IOI Circuit** (2211.00593) — canonical GPT-2 IOI reverse-engineering; head taxonomy.
- **Heimersheim & Nanda** (2309.16042) — activation-patching best practices.
- **Nanda et al.** (2404.15255) — use/interpretation of activation patching.
- **Path Patching** (2304.05969) — path-level causal localization.
- **Negation Circuit in GPT-2** (2603.12423) — frame-shift ablation study.
- **Pattern Selectivity ≠ Task-Causal Structure** (2606.05378) — motivates the two random-head baselines Claim 2 uses.
- **Head Functional Roles / Cognitive Mirrors** (2512.10978), **Which Heads Matter for ICL** (2502.14010) — head-role taxonomies.

### 3.4 Developmental interpretability on Pythia
- **Pythia suite** (Biderman 2023) — infrastructure.
- **Tigges et al.** NeurIPS 2024 — circuits consistent across training and scale; Pythia 70M–2.8B.
- **When Do Attention Circuits Form?** (2606.02378) — induction @ step 1000, FV @ step 16000.
- **Developmental Interpretability Review** (2508.15841) — survey.

### 3.5 Inference-time dynamic control / activation steering
- **Representation Engineering Survey** (2502.17601) — landscape.
- **PASTA** — head-level attention amplification / suppression.
- **SADI** — semantic-adaptive per-input head steering with binary masks.
- **Dynamic Activation Composition** — KL-guided steering intensity.
- **Zhu et al. steering** (2402.18496 §4) — direct precedent for belief-vector steering (but at the representation, not head, level).

---

## 4. Structural Gaps

- **Gap G1 — No head-level Fisher localization of belief in Pythia specifically.** The Sensitivity-Meets-Sparsity paper worked at the *parameter* level and used ToM benchmarks; Zhu et al. worked at the *representation* level with linear probes. **The precise combination — FIM → attention-head candidate set → causal zero-ablation with random-head and random-mask baselines, on Pythia-410m/1b/2.8b — has not been done.** *Competitive set*: 2504.04238 + 2402.18496 + 2211.00593. *Why open*: each paper covers one component; nobody has integrated the pipeline on the third-person belief_core subset.
- **Gap G2 — No separate personal-belief vs attributed-belief circuit disentanglement.** Prior work treats "belief" as a monolithic capability or uses only third-person ToM. Whether `personal_belief` and `attributed_belief` are supported by *distinct* head sets or a shared substrate with a task-frame router is an *open causal question*. *Competitive set*: 2402.18496 + 2504.04238. *Why open*: no prior work explicitly ran Fisher-based localization separately per belief frame while controlling for the other belief frame and world knowledge as off-target behaviors.
- **Gap G3 — No formation-window analysis of belief circuits.** Induction and FV-head timing is known; belief-head timing is not. *Competitive set*: 2606.02378 + Tigges NeurIPS 2024. *Why open*: prior developmental work targets ICL primitives (induction, FV) or task circuits, not belief specifically; no one has run the causal-ablation trajectory across Pythia-1b checkpoints for each belief frame.
- **Gap G4 — No belief-frame-conditional head amplifier with an oracle prompt-hint baseline.** Existing dynamic steering (PASTA, SADI, Dynamic Activation Composition) targets sentiment / safety / instruction following; the frame classifier is over generic semantic features, not belief-specific internal representations preceding the belief heads. And no prior work benchmarks against an *oracle prompt-hint* baseline — the strongest test of whether internal control adds value beyond prompt-level task specification. *Competitive set*: 2502.17601 (survey) + PASTA + SADI + Zhu et al. steering. *Why open*: (a) the exact target-head set to amplify was previously unknown at head-level (Gap G1), (b) no prior work uses a belief-frame classifier + preserves `world_knowledge` behavior + compares against an oracle prompt-hint.
- **Gap G5 — Scale-dependence of the belief asymmetry inside a single family.** The 41-LM cross-model study reports a strong average asymmetry between third-person and personal belief but does not run inside-a-family scale sweeps holding data/training constant. Pythia's controlled design lets us ask whether the asymmetry grows / stabilizes / non-monotonically changes across 410M → 1B → 2.8B. *Competitive set*: 2602.16085 + Pythia suite. *Why open*: prior work either mixed families (confounded) or used single-scale evaluations.

## 5. Banlist — Failed Ideas (do not regenerate)

*(no prior banlist)*
