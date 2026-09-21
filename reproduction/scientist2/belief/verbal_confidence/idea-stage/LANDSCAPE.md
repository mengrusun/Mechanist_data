# Landscape: Verbalized confidence in LLMs — is it cached mid-generation, or computed on demand?

**Date**: 2026-07-13
**Scope**: Mechanistic interpretability of verbalized-confidence generation in decoder-only LLMs. Interpreted as: (a) how the model *internally* produces the confidence score it eventually emits when asked to verbalize a probability after answering; (b) whether that score is *cached* in the residual stream at answer-adjacent positions and later *retrieved* at the confidence-generation step, or is *freshly computed* at the confidence-generation step. Focused on Gemma-3-27B on TriviaQA per `task.md`, with the option to swap Qwen 2.5 7B for cross-model checks.
**Based on**: 15 retrieved papers — see `RESEARCH_LIT.md` for the raw retrieval dump.
**Policy note**: `.claude/forbidden-urls.txt` blocks the target paper (arXiv 2603.17839) and every arXiv paper with YYMM ≥ 2603, plus author tokens and the phrase "verbal confidence" in web results. The landscape below is drawn only from allowed sources and does not encode the target paper's specific design.

---

## 1. Structured Paper Table

| Paper | Venue | Method | Key Result | Relevance to Us | Source |
|-------|-------|--------|------------|-----------------|--------|
| Kadavath et al. 2022 (arXiv:2207.05221) | preprint | P(True) self-eval + P(IK) probe | LMs are self-calibrated on MCQ / T-F; P(IK) partially generalizes | Establishes that an internal self-knowledge signal exists — the pre-condition for a cached self-evaluation | arXiv |
| Tian et al. 2023 (arXiv:2305.14975) | EMNLP 2023 | Prompt LM to verbalize probability | Verbalized probability recovers RLHF-lost calibration (~50% ECE↓ on TriviaQA) | Establishes the *behavior* the task.md claim is about | arXiv |
| Xiong et al. 2023 (arXiv:2306.13063) | ICLR 2024 | Prompt / sampling / consistency elicitation | Verbal confidence is real but noisy and prompt-sensitive | Motivates asking *how* the number is computed | arXiv |
| Lin et al. 2022 (arXiv:2205.14334) | TMLR | Fine-tune to output verbal probability | Verbal-probability tokens can be well-calibrated | Shows verbal confidence is not just a fluency artifact | arXiv |
| Azaria & Mitchell 2023 (arXiv:2304.13734) | EMNLP-F 2023 | MLP probe on hidden states | Truth/lie is linearly decodable from the residual stream | Direct precedent: a self-evaluation signal is decodable from hidden states | arXiv |
| Meng et al. 2022 (arXiv:2202.05262) | NeurIPS 2022 | Causal tracing on GPT | Localizes factual computation in mid-layer MLPs at last subject token | Provides the causal-tracing protocol we will re-use | arXiv |
| Heimersheim & Nanda 2024 (arXiv:2404.15255) | preprint | Activation-patching methodology | Best-practice guide for causal patching | Method backbone for the causal experiments | arXiv |
| Yang et al. 2024 (arXiv:2412.14737) | preprint | Verbal-confidence benchmark | Reliability depends heavily on prompt | Behavior baseline; confirms verbal confidence is worth mechanistic study | arXiv |
| Yuan et al. 2024 (arXiv:2411.13343) | preprint | Fact-level calibration | Fact-granular confidence for self-correction | Downstream use case for the caching finding | arXiv |
| van der Weij et al. 2024 (arXiv:2403.05767) | preprint | Activation steering | Add / subtract a direction at a chosen layer / position | Method backbone for testing the cache with steering | arXiv |
| Kabra et al. 2023 (arXiv:2311.09553) | NAACL 2024 | Program-aided self-knowledge | PAL improves calibration | Task-modulated self-knowledge — background | arXiv |
| Anon 2025 (arXiv:2510.09033) | preprint | Internal-state probes for "know / don't know" | Probes mainly track *recall*, not *truthfulness* | Critical null: caching signal could be a recall-strength artifact | arXiv |
| ICR Probe 2025 (arXiv:2507.16488) | preprint | Probe on hidden-state *dynamics* | Trajectories carry uncertainty signal | Precedent for signal living in post-answer hidden states | arXiv |
| Evidence for Limited Metacognition (arXiv:2509.21545) | preprint | Behavioral battery | LLM metacognition is partial | Background — motivates asking whether the imperfection is in the cache or in retrieval | arXiv |
| Metacognitive Probe (arXiv:2605.09844) | preprint | Behavioral diagnostics | Battery of calibration diagnostics | Background — behavioral-only, no mechanism | arXiv |

---

## 2. Core Landscape Narrative

**Verbalized confidence is now a first-class uncertainty signal.** Tian et al. (2023) and Xiong et al. (2023) showed that prompting an RLHF-tuned model to verbalize a probability recovers calibration lost by fine-tuning, and Lin et al. (2022) had already established that a fine-tuned LM can emit well-calibrated verbal probabilities. Yang et al. (2024) formalized this into a benchmark: verbal confidence is real, useful, and prompt-sensitive. This body of behavioral work is the *justification* for the mechanistic question — verbal confidence matters, so understanding how the model produces it internally matters too. None of these papers, however, look inside the model at *when* and *where* the confidence value is decided.

**Self-evaluation signals are decodable from hidden states.** Kadavath et al. (2022) established, via P(IK), that models have an internally decodable "I know this" signal — even before an answer is committed. Azaria & Mitchell (2023) show a truth-vs-lie signal is linearly decodable from the residual stream *while the model is generating*. ICR-Probe (2025) shows uncertainty is decodable from the *dynamics* of the hidden states across generation steps. Together these establish the ambient mechanistic prior: the residual stream carries a compact, probe-decodable signal correlated with answer correctness/confidence. The open question is whether this signal is what the model *reads from* when it later verbalizes a number — or whether verbalization recomputes something new from token log-probabilities at that later step.

**The causal-intervention grammar is standard.** ROME (Meng et al., 2022) established causal tracing as the standard tool for localizing where and when a specific piece of information is written into hidden states — clean/corrupted runs, noise-restore, per-position × per-layer effect maps. Heimersheim & Nanda (2024) codified the activation-patching methodology (direct patching, path patching, denoising vs. noising, effect metrics, common pitfalls). Activation steering (van der Weij et al., 2024, and the broader representation-engineering literature) provides the intervention primitive for *modulating* a candidate cached signal. These tools are the workhorses for any test of the caching hypothesis.

**Two competing null hypotheses must be ruled out.** First, **the recall-strength null** (arXiv:2510.09033): a decodable "confidence" signal in hidden states might merely reflect how strongly the fact was memorized, not a distinct self-evaluation. If we probe post-answer positions and find a confidence-predictive direction, we must show it is *not* just a recall-strength axis — e.g., by showing it separates *equally-recalled* correct and incorrect items, or by showing the signal generalizes beyond factual recall to arithmetic / MMLU-style questions where recall is not the mechanism. Second, **the log-prob restatement null**: the verbalized number might just be an on-the-fly readout of the answer token's log-probability. If patching the post-answer hidden states materially changes the verbalized confidence *while the answer token and its log-prob are held fixed*, this null is falsified. If, conversely, only patching at the confidence-generation position (or blocking attention from the answer position to the confidence position) matters, the caching hypothesis is falsified.

**No open-source, mechanistic account of verbalization-time confidence retrieval exists in the allowed pre-2603 literature.** Behavioral papers stop at "verbal confidence is useful"; probing papers stop at "self-knowledge is decodable at the answer position"; causal-intervention papers stop at "we can move / edit factual information". The mechanistic path *from a self-evaluation carried in the residual stream at answer-adjacent positions to the confidence token generated many positions later* — including which positions cache it, which attention paths retrieve it, and what happens when we cut those paths — is the gap the task.md claim occupies.

---

## 3. Sub-direction-Specific Work

- **Behavioral: verbal confidence & calibration.** Tian 2023, Xiong 2023, Lin 2022, Yang 2024, Yuan 2024, Kabra 2023, Metacognitive Probe 2026. *Gap*: none of these look inside the network.
- **Probing: self-knowledge & truth directions in hidden states.** Kadavath 2022 (P(IK)), Azaria & Mitchell 2023, ICR-Probe 2025. *Gap*: they train probes at fixed positions; do not test whether the same signal is *causally used* at a later verbalization step.
- **Causal intervention on factual recall.** ROME 2022, Heimersheim & Nanda 2024, activation-steering literature (van der Weij 2024). *Gap*: applied to factual associations, not to self-evaluation / verbalized confidence.
- **Nulls / confounds.** arXiv:2510.09033 (recall-vs-truthfulness confound), Kadavath 2022 (P(True) vs. token-prob distinction), Xiong 2023 (prompt-format sensitivity). *Gap*: no work explicitly tests the log-prob-restatement null with causal interventions targeting post-answer positions.

---

## 4. Structural Gaps

- **Gap G1 — Where along the sequence is the confidence value decided?** *Competitive set*: Kadavath 2022, Azaria & Mitchell 2023, ICR-Probe 2025, ROME 2022. *Why open*: prior work either probes at the answer position or edits factual associations at the subject position. Nobody has run a *per-position × per-layer causal-tracing map for the eventual verbalized confidence token*, distinguishing (i) post-answer boundary positions, (ii) intermediate positions, and (iii) the confidence-generation position itself.
- **Gap G2 — Is the signal read by the confidence token via a specific attention path, or globally available?** *Competitive set*: ROME 2022, Heimersheim & Nanda 2024. *Why open*: attention-blocking / attention-knockout has been used for factual recall but not for retrieving a self-evaluation signal. The task.md claim explicitly predicts an *information-flow bottleneck* from answer-adjacent positions to the confidence-generation position.
- **Gap G3 — Does the internal signal predict verbal confidence *beyond* what the answer token's log-probability predicts?** *Competitive set*: Tian 2023, Xiong 2023, Kadavath 2022, arXiv:2510.09033. *Why open*: prior work compares verbal confidence with log-probability at the output level, but not with *residual-stream features* at answer-adjacent positions. Variance partitioning of verbal confidence into (log-prob-explainable) + (hidden-state-explainable) has not been done in the allowed literature.
- **Gap G4 — Is the cached signal generalizable across prompt formats and paraphrases?** *Competitive set*: Yang 2024, Xiong 2023. *Why open*: verbal confidence is known to be prompt-sensitive at the *output* level; whether the *internal* cache is prompt-sensitive or prompt-invariant is unknown and matters for whether the "cache" is a genuine self-evaluation or a surface phenomenon.

---

## 5. Banlist — Failed Ideas (do not regenerate)

_(no prior banlist — round 1, no `research-wiki/query_pack.md` present)_
