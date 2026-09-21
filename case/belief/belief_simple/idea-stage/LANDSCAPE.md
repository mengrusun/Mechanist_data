# Landscape: Belief representation & false-belief circuits in Pythia language models (reproduction context)

**Date**: 2026-07-22
**Scope**: Literature that GROUNDS the fixed methods for a faithful reproduction of "Sensitivity Meets Sparsity" (Chen et al., 2025) on personal-belief vs attributed-belief circuits in Pythia-410M / 1B / 2.8B. Do NOT propose new methods — the reproduction fixes Fisher-mask + zero-ablation for localization, pythia-1b checkpoint sweep for the formation window, and a probe-and-amplify controller for dynamic control.
**Interpreted as**: reproduction-oriented survey — literature is used to justify design choices (why Fisher, why zero-ablation, why AND-NOT masks, why 20 random-head + 20 random-mask controls, why linear probes for frame classification, why head-restricted amplification), not to seed new mechanism families.
**Based on**: 150 retrieved papers via mechanic-db cloud SEARCH + arXiv API + WebSearch (Zotero / Obsidian / local library not configured) — see `RESEARCH_LIT.md` for the raw retrieval dump.

---

## 1. Structured Paper Table (top 20, ordered by direct relevance to the reproduction)

| # | Paper | Venue | Method | Key Result | Relevance to Us | Source |
|---|-------|-------|--------|-----------|-----------------|--------|
| 1 | Sensitivity Meets Sparsity (Chen et al.) | arXiv:2504.04238 (2025) | Fisher-information mask over parameters + zero-ablation | Perturbing 0.001% of ToM-sensitive parameters degrades ToM performance; parameters concentrate in W_Q / W_K matrices | **REPRODUCTION PAPER.** Method for Claim 2 is fixed from here. | mechanic-db + Web + arXiv |
| 2 | How LLMs encode ToM: sparse parameter patterns | npj AI (2025) | Same as #1 (published venue) | Same as #1 | Authoritative published version of the reproduction paper. | mechanic-db |
| 3 | Brittle Minds, Fixable Activations | arXiv:2406.17513 (2024) | Probing + activation intervention + steering | ToM performance is *brittle* to activation perturbation but *fixable* with steering vectors | Direct support for Claim 4 (probe → amplify) methodology. | mechanic-db |
| 4 | LMs Represent Beliefs of Self and Others | ICML 2024 / arXiv:2402.18496 | Linear probe on activations + causal intervention | Belief status of self and others is linearly decodable; manipulating the direction causally alters social reasoning | Justifies the linear-probe frame classifier in Claim 4 and the causal-intervention paradigm in Claim 2. | mechanic-db |
| 5 | LMs use Lookbacks to Track Beliefs | arXiv:2505.14685 (2025) | Causal mediation + abstraction on CausalToM | Identifies specific "lookback" attention-head mechanisms for routing past belief-relevant information | Corroborates that distinct head-level circuits exist for own-vs-other belief tracking (Claim 2 target phenomenon). | mechanic-db |
| 6 | Unveiling ToM: Parallel to Single Neurons | arXiv:2309.01660 (2023) | Neuron-level activation analysis in LLMs vs human dmPFC single-neuron recordings | Small neuron populations selectively respond to ToM prompts | Neuroscience-grounded motivation for sparse localization. | mechanic-db |
| 7 | Evaluating Contrast Localizer for ToM & Math | arXiv (2025) | Neuroscience-style contrast localizer + causal ablation on 11 LLMs (3B-90B) | Contrastive stimulus sets pick out causally-relevant units | Methodologically the closest cousin — contrastive (target AND-NOT control) design is what Claim 2's `Mask_attributed = top-0.1% F_attributed AND NOT top-1% F_knowledge` implements at the parameter level. | mechanic-db |
| 8 | Interpretability in the Wild (IOI Circuit) | ICLR 2023 / arXiv:2211.00593 | Path-patching + zero-ablation on GPT-2 Small | Identifies ~26-head circuit implementing indirect object identification | **Methodological ancestor of Claim 2** — the "candidate → zero-ablate → controlled random-head baseline" pipeline originates here. | mechanic-db + Web + arXiv |
| 9 | What needs to go right for an induction head? | arXiv:2404.07129 (2024) | Pythia checkpoint sweep + circuit analysis at each checkpoint | Induction heads emerge in tandem with a phase change in loss; sub-circuit formation dynamics govern the transition | **Direct methodological grandparent of Claim 3** — same design pattern: probe intermediate checkpoints, measure behavioral emergence, localize the causal sub-circuit at each checkpoint. Uses Pythia. | mechanic-db |
| 10 | LLM Circuit Analyses Are Consistent Across Training and Scale | arXiv:2407.10827 (2024) | Multi-scale + multi-checkpoint circuit analysis on Pythia | Circuits identified at one checkpoint/scale remain identifiable at others | Justifies re-using the Claim-2 pythia-1b head set to intervene at all pythia-1b checkpoints for Claim 3. | Web |
| 11 | In-context Learning and Induction Heads | Transformer Circuits Thread (2022) | Loss-phase-change detection + head-level circuit analysis | IH formation coincides with a discrete phase change in training loss | Foundational formation-window paradigm behind Claim 3. | mechanic-db |
| 12 | Pythia: A Suite for Analyzing LMs Across Training and Scaling | ICML 2023 / arXiv:2304.01373 | Public 14M-12B model + 154 checkpoints | Standardized suite for developmental analysis | The exact models + checkpoints the project reproduces on. | Web + arXiv |
| 13 | Knowledge Circuits in Pretrained Transformers | NeurIPS 2024 / arXiv:2405.17969 | Head + MLP knockout for factual recall | Isolates knowledge-recall circuits; role of intermediate MLPs as "accumulators" | Supports the world-knowledge control side of Claim 2 (belief heads should be *distinct* from knowledge-recall circuits). | mechanic-db |
| 14 | Knowledge Neurons in Pretrained Transformers | ACL 2022 / arXiv:2104.08696 | Gradient × activation attribution on FFN neurons | Small neuron sets store specific facts | Technical ancestor of Fisher-based attribution (Fisher = expectation of gradient²). | mechanic-db |
| 15 | Transformers represent belief state geometry in residual stream | arXiv:2405.15943 (2024) | Optimal-prediction theory + linear probes | Belief states are linearly represented in the residual stream in a low-dim subspace | Supports the linear-probe frame classifier in Claim 4. | mechanic-db |
| 16 | Constrained belief updates explain geometric structures | arXiv:2502.01954 (2025) | Follow-up to #15 with Bayesian constraint | Transformer belief representations implement constrained Bayesian belief updates | Reinforces the linear-probe hypothesis. | mechanic-db |
| 17 | The Geometry of Truth | ICLR 2024 / arXiv:2310.06824 | Linear probes on true/false statements | Truth direction is low-dim linear and transferable | Corollary support for the frame-classification probe. | mechanic-db |
| 18 | Emergence of Minimal Circuits for IOI | arXiv:2510.25013 (2025) | Train attention-only transformers from scratch on IOI | Minimal head circuits emerge during training | Complementary formation-window methodology (from-scratch vs pretrained checkpoints). | mechanic-db + Web |
| 19 | Does Circuit Analysis Interpretability Scale? (Chinchilla) | arXiv:2307.09458 (2023) | Circuit analysis on 70B Chinchilla for MCQ | Circuits at scale are similar in structure, harder to fully specify | Scale-relevant validation for pythia-2.8b regime. | mechanic-db |
| 20 | Circuit Component Reuse Across Tasks | arXiv:2310.08744 (2023) | Cross-task overlap analysis on IOI-adjacent tasks | Circuit components generalize across structurally similar tasks | Motivates the specificity criterion (belief-heads must NOT be shared with knowledge-recall heads). | mechanic-db |

Papers 21-30 (see `RESEARCH_LIT.md`) provide further supporting context: activation-steering families (paper 24), circuit-faithfulness evaluation (29), sparse-autoencoder alternatives (23, 28), ToM sub-ability decomposition (26), circuit stability under OOD shift (30) — all of which reinforce the reproduction's method choices without introducing new families.

## 2. Core Landscape Narrative

**A — Localization: from knowledge neurons to Fisher masks.** The line of "identify a small parameter subset responsible for a specific model behaviour" runs from Knowledge Neurons (Dai et al. 2022 — gradient × activation on FFN neurons) through IOI-Circuit (Wang et al. 2022 — path patching over heads) to the Fisher-information mask of Chen et al. (2025). Each step trades resolution for signal quality: knowledge neurons are neuron-granular and gradient-only; IOI operates at attention-head granularity with causal intervention; Fisher operates at parameter granularity, uses expectation-of-squared-gradient (the second-moment estimator), and — crucially — layers a *contrastive AND-NOT construction* on top (`top-fraction of target signal AND NOT top-fraction of control signal`) that isolates target-specific parameters from generic-purpose parameters. Claim 2 in this reproduction implements exactly this final construction, at attention-head granularity via Fisher-derived candidate heads followed by zero-ablation. The 20 random-head + 20 random-mask controls, and the specificity thresholds (target drop ≥ 0.30; off-target drop ≤ 0.10; PPL ≤ 1.05× clean), are all standard for this family and directly guard against the "generic head damage" failure mode Merullo et al. 2023 warned about.

**B — Formation window: induction heads set the paradigm.** The intermediate-checkpoint developmental analysis in Claim 3 has a well-established template. Olsson et al. 2022 first showed that in-context-learning capability arises with a discrete phase change in training loss and coincides with induction-head formation. Singh et al. 2024 ("What needs to go right for an induction head?") extended this to Pythia checkpoints and identified the *interacting sub-circuits* whose emergence gates the phase change. Prakash et al. 2024 ("LLM Circuit Analyses Are Consistent Across Training and Scale") verified that a circuit found at one checkpoint is largely findable at earlier/later ones, which justifies the reproduction's design of re-using the Claim-2 pythia-1b head set to run zero-ablation at each intermediate pythia-1b checkpoint. Both the *behavioral trajectory* (accuracy vs step-count) and the *causal trajectory* (accuracy-under-ablation vs step-count) are standard measurements in this line — the reproduction stacks them for personal-belief and attributed-belief separately, so distinct formation windows for the two abilities become directly readable.

**C — Belief-frame representation: linear probes justified.** Multiple recent studies show that belief-like representations in transformer LMs are *linearly decodable* from the residual stream. Zhu et al. 2024 (ICML) linearly decode self- and other-belief; the belief-state-geometry line (Shai et al. 2024, Piotrowski et al. 2025) shows that the belief-state subspace has predictable geometric structure; Marks & Tegmark 2024 (Geometry of Truth) show a low-dim truth direction that steers generations. Together, these justify the frame-classification probe in Claim 4 (early-layer residual-stream activations → probing MLP over multiple layers → frame prediction), and support the causal-manipulation half (amplify identified belief heads in later layers, no amplification for the `world_knowledge` frame). The Brittle-Minds paper (Bortoletto et al. 2024) is the closest direct precedent — it shows that ToM behavior is brittle to activation perturbation but *fixable* with targeted steering, i.e., that head-restricted amplification (rather than a global steering vector) is a plausible controller design.

**D — Zero-ablation vs alternatives; specificity guarantees.** Zero ablation is known to be noisier than mean-ablation but has the crucial property that its intervention is *unambiguously null* — no confounding distribution assumption. Chen et al. 2025 (and the entire IOI/Chinchilla line) rely on it precisely because the target claim ("this head is causally responsible") is easier to falsify with a null intervention than with a mean substitute. The random-head baseline (20 controls with same head count) and random-mask baseline (20 controls with same parameter count) together fence in the two most common failure modes: (i) "any head knockout would cause this drop" — controlled by the random-head baseline; (ii) "the effect is just from the total parameter count knocked out" — controlled by the random-mask baseline. The `2σ` significance threshold plus the four-criteria conjunction (accuracy drop ≥ 0.30; > baseline mean + 2σ; off-target ≤ 0.10; PPL ≤ 1.05×) is a stringent, well-motivated bar — the reproduction should adopt it verbatim as task.md prescribes.

**E — Scale-dependent emergence: prior evidence in this exact regime.** Hagendorff et al. 2022 (Thinking Fast and Slow), Kosinski's ToM-in-LMs line, and the LLM-belief-benchmark work (Belief in the Machine, arXiv:2410.21195) all report scale-dependent emergence of belief and reasoning capabilities. Chen et al. 2025 report scale-dependent Fisher-mask stability. The reproduction's Claim 1 (behavioural evaluation across pythia-{410m, 1b, 2.8b}) sits inside this literature — the reproduction should compare its results to these prior scaling curves as validation. Note: pythia-410m is at the lower boundary of "shows above-chance ToM" — the Claim-2 "above-chance" gate is therefore expected to potentially exclude 410m from localization, and this should be noted rather than treated as failure.

## 3. Sub-direction-Specific Work

**Fisher / attribution-based sparse localization.**
- Chen et al. 2025 (arXiv:2504.04238) — the reproduction target.
- Dai et al. 2022 (Knowledge Neurons) — gradient × activation ancestor.
- Yao et al. 2024 (Knowledge Circuits) — head + MLP granularity, factual recall.
- Bhaskar et al. 2024 (Sparse feature circuits, arXiv:2405.16941) — feature-level alternative (not used here).
- Contrast-Localizer 2025 (arXiv, not yet fully indexed) — the contrastive-set methodology closest to the AND-NOT Fisher-mask.

Gap left: none affects the reproduction — task.md fixes the method.

**Attention-head zero-ablation for causal circuit discovery.**
- Wang et al. 2022 (IOI, arXiv:2211.00593) — canonical reference.
- Merullo et al. 2023 (Component Reuse) — motivates specificity checks.
- Conmy et al. 2023 (ACDC, arXiv:2304.14997) — automated head discovery.
- Ferrando et al. 2024 (Efficient Automated Circuit Discovery, arXiv:2407.00886) — recent efficient variant.
- Lieberum et al. 2023 (Chinchilla, arXiv:2307.09458) — scale test.

Gap left: none — the reproduction re-uses the Wang-et-al.-style workflow directly.

**Formation-window / pretraining trajectory analysis.**
- Olsson et al. 2022 (Induction Heads) — foundational.
- Singh et al. 2024 (What needs to go right, arXiv:2404.07129) — Pythia + sub-circuit dynamics.
- Prakash et al. 2024 (LLM Circuit Analyses Consistent, arXiv:2407.10827) — cross-checkpoint stability.
- Biderman et al. 2023 (Pythia, arXiv:2304.01373) — the model + checkpoint suite.
- Marks et al. 2024 (Crosscoding Through Time, arXiv:2509.05291) — feature-level trajectory tracking.

Gap left: none — the reproduction's design (behavioural trajectory + causal trajectory at each pythia-1b checkpoint) is well-covered by this line.

**Linear-probe frame classifiers and belief-state representation.**
- Zhu et al. 2024 (LMs Represent Beliefs of Self and Others, arXiv:2402.18496).
- Bortoletto et al. 2024 (Brittle Minds, arXiv:2406.17513).
- Shai et al. 2024 (Belief State Geometry, arXiv:2405.15943).
- Piotrowski et al. 2025 (Constrained Belief Updates, arXiv:2502.01954).
- Marks & Tegmark 2024 (Geometry of Truth, arXiv:2310.06824).

Gap left: none — the probe design in Claim 4 is well-supported.

**Activation steering / head amplification.**
- Turner et al. 2023 (Activation Addition, arXiv:2308.10248).
- Zou et al. 2023 (Representation Engineering, arXiv:2310.01405).
- Panickssery et al. 2024 (Contrastive Activation Addition, arXiv:2312.06681).
- Bortoletto et al. 2024 (Brittle Minds) — head-level steering for ToM.

Gap left: the *head-restricted* amplification variant used in Claim 4 (as opposed to residual-stream steering vector) is comparatively less studied — but Bortoletto et al. and Zhu et al. give it clear precedent. The reproduction should therefore report both the controller's OOD gains AND its degradation-mode counts (per task.md: "recovered predictions", "degraded predictions", "net improvement").

## 4. Structural Gaps

Because this is a faithful reproduction, "structural gaps" here means *what to be careful about when reproducing*, not new research openings.

- **Gap R1 — "Above-chance" gate for pythia-410m may be binding.** — Competitive set: Chen et al. 2025 report ToM localization mainly on Llama-family models; the ability of pythia-410m to clear the above-chance bar on `attributed_belief` is uncertain. Reproduction should document if 410m falls out at this gate rather than force a localization run.
- **Gap R2 — "Smallest head set" search discipline.** — Competitive set: Wang et al. 2022 use a greedy-then-verify search; ACDC (Conmy et al. 2023) uses graph pruning. task.md prescribes "smallest set satisfying the four criteria" but does not specify the search order. Reproduction should adopt a deterministic ordering (e.g., rank candidate heads by Fisher mask magnitude, add greedily, stop at the smallest passing set) — Phase 4.5 will pin this in the experiment plan.
- **Gap R3 — PPL corpus scope.** — Competitive set: Pile-based PPL is standard; the pretraining corpus at `/mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled/` is what Pythia itself was trained on. The `PPL ≤ 1.05×` bar assumes a stable held-out sample; reproduction should fix the sample-size and sampling seed so PPL comparisons are apples-to-apples across the 20 random-head + 20 random-mask controls.
- **Gap R4 — Claim 3 checkpoint-schedule mapping.** — Competitive set: Olsson et al. 2022 sample log-spaced early checkpoints; Singh et al. 2024 use Pythia's built-in schedule (0, 1, 2, 4, ..., 512, 1000, 2000, ..., 143000). Reproduction should follow the Pythia native schedule for pythia-1b (as the checkpoints are directly available on disk).
- **Gap R5 — Head amplification magnitude for Claim 4.** — Competitive set: activation-steering literature uses various magnitudes (linear scale, learned, or grid-searched). task.md does not fix the amplification coefficient. Reproduction should treat magnitude as a controller hyperparameter tuned on the belief_core training split and evaluated on the belief_holdout OOD set — this respects the "train on core, evaluate on holdout" split task.md mandates.

## 5. Banlist — Failed Ideas (do not regenerate)

_(no prior banlist)_
