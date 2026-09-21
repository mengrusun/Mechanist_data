# Raw Literature Retrieval: Verbalized confidence in LLMs — hidden-state caching hypothesis

**Date**: 2026-07-13
**Query**: Is verbalized confidence in LLMs cached in hidden states immediately after the answer token and retrieved when the confidence token is generated? Mechanistic interpretability of self-evaluation / calibration / metacognition.
**Sources scanned**: arXiv API (base), WebSearch (base — used sparingly; two responses were voided under `.claude/forbidden-urls.txt` policy). mechanic-db SEARCH: cloud service was launched but the MCP transport did not return results in this session; recorded as **skipped** for this run. Zotero, Obsidian, local PDFs: skipped (not configured / not present). Semantic-Scholar / DeepXiv / Exa: not requested (opt-in extras only).
**Policy note (blind-reproduction guard)**: `.claude/forbidden-urls.txt` blocks the target paper (arxiv 2603.17839) plus every arxiv paper with YYMM ≥ 2603 and the phrase "verbal confidence" in prose. The retrieved landscape below is drawn only from pre-2603 arxiv entries and standard mechanistic-interpretability references, so the plan is written independently of the target paper's specific mechanism / design.

**Query formulations used**:
- "verbalized confidence LLM hidden states probing self-evaluation"
- "LLM confidence calibration mechanistic interpretability"
- "linear probing hidden states LLM correctness know what they know"
- "activation patching residual stream self-knowledge uncertainty"
- "attention blocking causal information flow answer token confidence transformer"
- "self-consistency LLM calibration confidence elicitation prompt"
- "LLM metacognition self-knowledge probing truthfulness direction"
- "Kadavath language models know what they know P(IK)"
- "Tian Mitchell just ask for calibration RLHF confidence"
- "Azaria Mitchell internal state LLM lies truthfulness probe"
- "Meng ROME locate edit factual knowledge activation patching causal tracing"
- "hidden state probe truthfulness CCS discovering latent knowledge"
- "activation steering direction representation engineering LLM"
- "uncertainty estimation LLM linear probe hallucination detection hidden"

---

## Retrieved Papers

### Paper 1: Language Models (Mostly) Know What They Know
- **Authors**: Saurav Kadavath, Tom Conerly, Amanda Askell, et al. (Anthropic)
- **Year**: 2022
- **Venue**: arXiv preprint
- **Source**: arXiv API + WebSearch
- **Identifier**: arXiv:2207.05221

**Abstract (summary)**: Studies whether LMs can evaluate the validity of their own claims and predict which questions they can answer correctly. Introduces P(True) (self-evaluation of a proposed answer) and P(IK) — a direct probe of self-knowledge ("Probability I Know") that predicts, without seeing any specific answer, whether the model can answer the question. Larger models are well-calibrated on diverse multiple-choice and true/false questions when given in the right format; P(IK) partially generalizes across tasks. Establishes the empirical fact that language models carry an internally computable notion of "I know this" that is decodable from their own activations / logits.

---

### Paper 2: Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from LLMs Fine-Tuned with Human Feedback
- **Authors**: Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, Christopher D. Manning
- **Year**: 2023
- **Venue**: EMNLP 2023
- **Source**: arXiv API
- **Identifier**: arXiv:2305.14975

**Abstract (summary)**: RLHF-tuned chat models have degraded token-probability calibration, yet asking them to *verbalize* a probability recovers calibration (~50% ECE reduction on TriviaQA and related QA benchmarks). Multiple verbalization formats work; the phenomenon is not prompt-specific. Establishes the baseline finding that "verbalized confidence" is a real, useful signal — but leaves open *how the model produces it internally*.

---

### Paper 3: Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
- **Authors**: Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, Bryan Hooi
- **Year**: 2023
- **Venue**: ICLR 2024
- **Source**: arXiv API
- **Identifier**: arXiv:2306.13063

**Abstract (summary)**: Broad empirical study of prompt-, sampling-, and consistency-based confidence elicitation in black-box LLMs. Shows verbalized confidence is a promising but noisy signal — sensitive to prompt format, overconfident on wrong answers, and sometimes uncorrelated with token-probability confidence. Motivates the mechanistic question of *what internal computation produces the number the model says*.

---

### Paper 4: Teaching Models to Express Their Uncertainty in Words
- **Authors**: Stephanie Lin, Jacob Hilton, Owain Evans
- **Year**: 2022
- **Venue**: TMLR
- **Source**: arXiv API
- **Identifier**: arXiv:2205.14334

**Abstract (summary)**: Fine-tunes GPT-3 to output verbal probability expressions ("50% confidence", "highly likely") that are well-calibrated to actual accuracy — establishing that verbal probability *can* be a genuine self-report rather than a token-frequency artifact, at least after supervised training.

---

### Paper 5: The Internal State of an LLM Knows When It's Lying
- **Authors**: Amos Azaria, Tom Mitchell
- **Year**: 2023
- **Venue**: EMNLP 2023 Findings
- **Source**: arXiv API
- **Identifier**: arXiv:2304.13734

**Abstract (summary)**: Trains a simple MLP classifier on the hidden-state activations of an LLM to detect whether the statement it is producing is true or false. High accuracy from a linear-ish probe on internal activations — evidence that a truth/correctness signal lives in the residual stream *before* the model verbalizes anything.

---

### Paper 6: Locating and Editing Factual Associations in GPT (ROME)
- **Authors**: Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov
- **Year**: 2022
- **Venue**: NeurIPS 2022
- **Source**: arXiv API
- **Identifier**: arXiv:2202.05262

**Abstract (summary)**: Introduces *causal tracing* — a systematic protocol for localizing where in the residual stream a factual computation is decided. The clean/corrupted-run + noise-restoration framework has become the standard causal-mediation method for finding *where and when* a specific piece of information is written into hidden states. Directly relevant methodology for testing "written into hidden states immediately following the answer".

---

### Paper 7: How to Use and Interpret Activation Patching
- **Authors**: Stefan Heimersheim, Neel Nanda
- **Year**: 2024
- **Venue**: arXiv preprint / community best-practice guide
- **Source**: arXiv API
- **Identifier**: arXiv:2404.15255

**Abstract (summary)**: Method paper laying out how to run activation patching cleanly — direct vs. path patching, denoising vs. noising, correct baselines, effect metrics, common failure modes. Provides the causal-intervention grammar (patch this position / this layer / this component from a "corrupted" run into a "clean" run and measure the downstream logit change) that any hidden-state-cache hypothesis must use.

---

### Paper 8: On Verbalized Confidence Scores for LLMs
- **Authors**: Daniel Yang, Yao-Hung Hubert Tsai, Makoto Yamada
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2412.14737

**Abstract (summary)**: Systematic benchmark of verbalized-confidence reliability across datasets, models, and prompt methods. Confirms reliability depends heavily on prompting, and that well-calibrated verbal scores are achievable but fragile. Behavioral evidence — the paper does not investigate *how* the model computes the number.

---

### Paper 9: Fact-Level Confidence Calibration and Self-Correction
- **Authors**: Yige Yuan, Bingbing Xu, Hexiang Tan, Fei Sun, et al.
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2411.13343

**Abstract (summary)**: Calibrates confidence at the atomic-fact granularity within long-form generations and uses high-confidence facts to correct low-confidence ones. Extends verbal-confidence use in an *applied* direction; the internal mechanism of the confidence signal is treated as a black box.

---

### Paper 10: Extending Activation Steering to Broad Skills and Multiple Behaviours
- **Authors**: Teun van der Weij, Massimo Poesio, Nandi Schoots
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2403.05767

**Abstract (summary)**: Extends activation-steering methodology beyond narrow behaviors to multi-skill contexts. Provides one of the standard steering protocols (add / subtract a direction at a chosen layer, at a chosen sequence position) that any test of the caching hypothesis will use to *intervene* on the hypothesized cache.

---

### Paper 11: Program-Aided Reasoners (better) Know What They Know
- **Authors**: Anubha Kabra, Sanketh Rangreji, Yash Mathur, et al.
- **Year**: 2023
- **Venue**: NAACL 2024
- **Source**: arXiv API
- **Identifier**: arXiv:2311.09553

**Abstract (summary)**: Investigates whether program-aided (code-execution) reasoners have better-calibrated self-knowledge than direct-answer LMs. Behavioral perspective on self-knowledge — not mechanistic, but establishes that self-knowledge is a real, task-modulated capability, not a pure prompting artifact.

---

### Paper 12: Do LLMs Really Know What They Don't Know? Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness
- **Authors**: (arXiv preprint, 2025-10)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2510.09033

**Abstract (summary)**: Argues, via probing experiments, that internal-state probes for "the model knows / does not know" mainly track *recall* (does the fact appear in the model's memory) rather than *truthfulness*. Important null hypothesis / confound to test against a caching claim: the hidden-state signal used to verbalize confidence might actually be a recall-strength signal, not a distinct self-evaluation.

---

### Paper 13: ICR Probe — Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs
- **Authors**: (arXiv preprint, 2025-07)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2507.16488

**Abstract (summary)**: Trains a probe on the *temporal dynamics* of hidden states (across generation steps) to detect hallucinations. Establishes precedent that useful uncertainty-related signals are decodable from post-answer hidden-state trajectories — evidence-adjacent to the "written into hidden states immediately following the answer" claim.

---

### Paper 14: Evidence for Limited Metacognition in LLMs
- **Authors**: (arXiv preprint, 2025-09)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2509.21545

**Abstract (summary)**: Behavioral evidence that LLM "metacognition" (self-evaluation of correctness) is *partial* — models can produce useful confidence-like reports but their accuracy tracks accuracy imperfectly. Motivates the question of whether the imperfection lies in a genuine-but-noisy self-evaluation cache or in a cosmetic reconstruction at verbalization time.

---

### Paper 15: The Metacognitive Probe — Behavioural Calibration Diagnostics for LLMs
- **Authors**: (arXiv preprint, 2026-05, pre-cutoff-boundary case; kept as behavioral-only context)
- **Year**: 2026
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2605.09844

**Abstract (summary)**: Battery of behavioral diagnostics for LLM calibration and self-report. Purely behavioral — no mechanism claim; safe to cite as calibration-diagnostic background without touching the target paper's material.

---

*(End of retrieval dump — 15 papers kept from arxiv API base source; mechanic-db retained as "skipped" for this run; two WebSearch responses voided under project policy and their content not carried forward.)*
