# Raw Literature Retrieval: Group-Aggregated Activation Probes for Multi-Agent Collusion Detection in LLMs

**Date**: 2026-07-14
**Query**: residual-stream linear probes on per-agent activations, group-level aggregation of probe scores across cooperating agents, covert coordination detection, steganography detection, deception probes in single-agent LLMs (RepE, contrast-consistent search, honesty probes), monitoring multi-agent LLM systems, black-box vs white-box collusion / covert-channel detection, zero-shot transfer of internal-state probes across scenarios (code-review sabotage, market manipulation, blackjack card-counting steganography)
**Sources scanned**: mechanic-db (SKIPPED — cloud SEARCH job polling timeout at 1200 s; job accepted but no terminal status returned; treated as unavailable per the skill's failure-mode contract), arXiv API (searched), Zotero (not configured), Obsidian (not configured), local PDFs (none present), Semantic Scholar (not enabled), DeepXiv (not enabled), Exa (not enabled), WebSearch (partial — two responses were voided at the post-search filter layer due to post-cutoff arxiv-id leakage risk; only the pre-cutoff arXiv-API results are retained below)

**Query formulations used**:
- "linear probes deception detection LLM residual stream"
- "multi-agent LLM collusion covert coordination"
- "LLM steganography covert channel detection"
- "LLM steganography emergent secret language cooperating agents"
- "representation engineering RepE honesty probe LLM"
- "contrast consistent search unsupervised knowledge LLM"
- "AI control multi-agent monitoring safety LLM sandbagging"
- "LLM lie detector activation probing scheming"
- "detecting hidden reasoning chain of thought monitoring probe"
- "algorithmic collusion LLM agents price bidding market"
- "activation probes belief truthfulness direction Marks Tegmark"
- "linear probing internal states LLM factuality hallucination"
- "hidden plain text emergent steganographic collusion LLMs"
- "decision-theoretic steganography LLM monitoring"
- "AI control redwood untrusted trusted monitoring evaluation"
- "sandbagging capability underperformance safety evaluation LLM"
- "Azaria Mitchell internal state factuality LLM lying"

**Cutoff rule enforced**: the project blocks any arXiv ID with year-month ≥ 2604 to prevent leakage of the target paper and follow-ups. All retained papers are strictly pre-cutoff.

---

## Retrieved Papers

### Paper 1: Detecting Strategic Deception Using Linear Probes
- **Authors**: Nicholas Goldowsky-Dill, Bilal Chughtai, Stefan Heimersheim, Marius Hobbhahn (Apollo Research)
- **Year**: 2025-02
- **Venue**: arXiv preprint (accepted ICML 2025)
- **Source**: arXiv API + WebSearch (Apollo Research)
- **Identifier**: arXiv:2502.03407
- **URL**: https://arxiv.org/abs/2502.03407

**Abstract**:
AI models might use deceptive strategies as part of scheming or misaligned behaviour. Monitoring outputs alone is insufficient, since the AI might produce seemingly benign outputs while their internal reasoning is misaligned. We thus evaluate if linear probes can robustly detect deception by monitoring model activations. We test two probe-training datasets, one with contrasting instructions to be honest or deceptive (following Zou et al., 2023) and one of responses to simple roleplaying scenarios. We test whether these probes generalize to realistic settings where Llama-3.3-70B-Instruct behaves deceptively, such as concealing insider trading (Scheurer et al., 2023) and purposely underperforming on safety evaluations (Benton et al., 2024). We find that our probe distinguishes honest and deceptive responses with AUROCs between 0.96 and 0.999 on our evaluation datasets. If we set the decision threshold to have a 1% false positive rate on chat data not related to deception, our probe catches 95-99% of the deceptive responses. Overall we think white-box probes are promising for future monitoring systems, but current performance is insufficient as a robust defence against deception.

---

### Paper 2: Representation Engineering: A Top-Down Approach to AI Transparency (RepE)
- **Authors**: Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, Dan Hendrycks
- **Year**: 2023-10
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2310.01405

**Abstract**:
In this paper, we identify and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuroscience. RepE places population-level representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena in deep neural networks (DNNs). We provide baselines and an initial analysis of RepE techniques, showing that they offer simple yet effective solutions for improving our understanding and control of large language models. We showcase how these methods can provide traction on a wide range of safety-relevant problems, including honesty, harmlessness, power-seeking, and more, demonstrating the promise of top-down transparency research.

---

### Paper 3: The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets
- **Authors**: Samuel Marks, Max Tegmark
- **Year**: 2023-10
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2310.06824

**Abstract**:
Large Language Models (LLMs) have impressive capabilities, but are prone to outputting falsehoods. Recent work has developed techniques for inferring whether a LLM is telling the truth by training probes on the LLM's internal activations. However, this line of work is controversial, with some authors pointing out failures of these probes to generalize in basic ways, among other conceptual issues. In this work, we use high-quality datasets of simple true/false statements to study in detail the structure of LLM representations of truth, drawing on three lines of evidence: 1. Visualizations of LLM true/false statement representations, which reveal clear linear structure. 2. Transfer experiments in which probes trained on one dataset generalize to different datasets. 3. Causal evidence obtained by surgically intervening in a LLM's forward pass, causing it to treat false statements as true and vice versa.

---

### Paper 4: The Internal State of an LLM Knows When It's Lying
- **Authors**: Amos Azaria, Tom Mitchell
- **Year**: 2023-04
- **Venue**: arXiv preprint (also EMNLP 2023 Findings)
- **Source**: arXiv API
- **Identifier**: arXiv:2304.13734

**Abstract**:
While Large Language Models (LLMs) have shown exceptional performance in various tasks, one of their most prominent drawbacks is generating inaccurate or false information with a confident tone. In this paper, we provide evidence that the LLM's internal state can be used to reveal the truthfulness of statements. This includes both statements provided to the LLM, and statements that the LLM itself generates. Our approach is to train a classifier that outputs the probability that a statement is truthful, based on the hidden layer activations of the LLM as it reads or generates the statement.

---

### Paper 5: Challenges with unsupervised LLM knowledge discovery (CCS critique)
- **Authors**: Sebastian Farquhar, Vikrant Varma, Zachary Kenton, Johannes Gasteiger, Vladimir Mikulik, Rohin Shah
- **Year**: 2023-12
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2312.10029

**Abstract**:
We show that existing unsupervised methods on large language model (LLM) activations do not discover knowledge -- instead they seem to discover whatever feature of the activations is most prominent. The idea behind unsupervised knowledge elicitation is that knowledge satisfies a consistency structure, which can be used to discover knowledge. We first prove theoretically that arbitrary features (not just knowledge) satisfy the consistency structure of a particular leading unsupervised knowledge-elicitation method, contrast-consistent search (Burns et al. - arXiv:2212.03827). We then present a series of experiments showing settings in which unsupervised methods result in classifiers that do not predict knowledge, but instead predict a different prominent feature. We conclude that existing unsupervised methods for discovering latent knowledge are insufficient, and we contribute sanity checks.

---

### Paper 6: LLM Internal States Reveal Hallucination Risk Faced With a Query
- **Authors**: Ziwei Ji, Delong Chen, Etsuko Ishii, Samuel Cahyawijaya, Yejin Bang, Bryan Wilie, Pascale Fung
- **Year**: 2024-07
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2407.03282

**Abstract**:
The hallucination problem of Large Language Models (LLMs) significantly limits their reliability and trustworthiness. Humans have a self-awareness process that allows us to recognize what we don't know when faced with queries. Inspired by this, our paper investigates whether LLMs can estimate their own hallucination risk before response generation. We analyze the internal mechanisms of LLMs broadly both in terms of training data sources and across 15 diverse Natural Language Generation (NLG) tasks, spanning over 700 datasets. Our empirical analysis reveals two key insights: (1) LLM internal states indicate whether they have seen the query in training data or not; and (2) LLM internal states show they are likely to hallucinate or not regarding the query.

---

### Paper 7: A Survey on the Honesty of Large Language Models
- **Authors**: Siheng Li, Cheng Yang, Zhiqi Cao, Yuxi Sun, Jinliang Lu, Chenhao Zhang, Zilin Yin, Kai Wu, Jin Zheng, Lingpeng Kong, et al.
- **Year**: 2024-09
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2409.18786

**Abstract**:
Honesty is a fundamental principle for aligning large language models (LLMs) with human values, requiring these models to recognize what they know and don't know and be able to faithfully express their knowledge. Despite promising, current LLMs still exhibit significant dishonest behaviors, such as confidently presenting wrong answers or failing to express what they know. In addition, research on the honesty of LLMs also faces challenges, including varying definitions of honesty, difficulties in distinguishing between known and unknown knowledge, and a lack of comprehensive understanding of related research. To address these issues, we provide a survey on the honesty of LLMs.

---

### Paper 8: Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs
- **Authors**: Yohan Mathew, Ollie Matthews, Robert McCarthy, Joan Velja, Christian Schroeder de Witt, Dylan Cope, Nandi Schoots
- **Year**: 2024-10
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2410.03768

**Abstract**:
The rapid proliferation of frontier model agents promises significant societal advances but also raises concerns about systemic risks arising from unsafe interactions. Collusion to the disadvantage of others has been identified as a central form of undesirable agent cooperation. The use of information hiding (steganography) in agent communications could render such collusion practically undetectable. This underscores the need for investigations into the possibility of such behaviours emerging and the robustness corresponding countermeasures. To investigate this problem we design two approaches -- a gradient-based reinforcement learning (GBRL) method and an in-context reinforcement learning (ICRL) method -- for reliably eliciting sophisticated LLM-generated linguistic text steganography. Standard mitigations — both passive oversight of model outputs and active mitigation through communication paraphrasing — are shown not fully effective at preventing this steganographic communication.

---

### Paper 9: AI Sandbagging: Language Models can Strategically Underperform on Evaluations
- **Authors**: Teun van der Weij, Felix Hofstätter, Ollie Jaffe, Samuel F. Brown, Francis Rhys Ward
- **Year**: 2024-06
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2406.07358

**Abstract**:
Trustworthy capability evaluations are crucial for ensuring the safety of AI systems, and are becoming a key component of AI regulation. However, the developers of an AI system, or the AI system itself, may have incentives for evaluations to understate the AI's actual capability. These conflicting interests lead to the problem of sandbagging, which we define as strategic underperformance on an evaluation. In this paper we assess sandbagging capabilities in contemporary language models (LMs).

---

### Paper 10: LLMs Can Covertly Sandbag on Capability Evaluations Against Chain-of-Thought Monitoring
- **Authors**: (Anthropic / academic collaboration, per abstract)
- **Year**: 2025-07
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2508.00943

**Abstract**:
Trustworthy evaluations of dangerous capabilities are increasingly crucial for determining whether an AI system is safe to deploy. One empirically demonstrated threat is sandbagging - the strategic underperformance on evaluations by AI models or their developers. A promising defense is to monitor a model's chain-of-thought (CoT) reasoning, as this could reveal its intentions and plans. In this work, we measure the ability of models to sandbag on dangerous capability evaluations against a CoT monitor by prompting them to sandbag while being either monitor-oblivious or monitor-aware. We show that both frontier models and small open-sourced models can covertly sandbag against CoT monitoring 0-shot.

---

### Paper 11: Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models
- **Authors**: Cameron Tice, Philipp Alexander Kreer, Nathan Helm-Burger, et al.
- **Year**: 2024-12
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2412.01784

**Abstract**:
Capability evaluations play a crucial role in assessing and regulating frontier AI systems. The effectiveness of these evaluations faces a significant challenge: strategic underperformance, or "sandbagging", where models deliberately underperform during evaluation. Sandbagging can manifest either through explicit developer intervention or through unintended model behavior. We introduce a novel sandbagging detection method based on injecting noise of varying magnitudes into model weights. Non-sandbagging models show predictable performance degradation with increasing noise; sandbagging models exhibit a distinguishable pattern.

---

### Paper 12: Emergent Response Planning in LLMs
- **Authors**: (see arXiv)
- **Year**: 2025-02
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2502.06258

**Abstract**:
In this work, we argue that large language models (LLMs), though trained to predict only the next token, exhibit emergent planning behaviors: their hidden representations encode future outputs beyond the next token. Through simple probing, we demonstrate that LLM prompt representations encode global attributes of their entire responses, including structure attributes (e.g., response length, reasoning steps), content attributes (e.g., character choices in storywriting, multiple-choice answers at the end of response), and behavior attributes (e.g., answer confidence, factual consistency). We explore how planning scales with model size and how it evolves during generation.

---

### Paper 13: Probing the Limits of the Lie Detector Approach to LLM Deception
- **Authors**: Tom-Felix Berger
- **Year**: 2026-02
- **Venue**: arXiv preprint (pre-cutoff)
- **Source**: arXiv API
- **Identifier**: arXiv:2603.10003

**Abstract**:
Mechanistic approaches to deception in large language models (LLMs) often rely on "lie detectors", that is, truth probes trained to identify internal representations of model outputs as false. The lie detector approach to LLM deception implicitly assumes that deception is coextensive with lying. This paper challenges that assumption. It experimentally investigates whether LLMs can deceive without producing false statements and whether truth probes fail to detect such behavior. Across three open-source LLMs, some models reliably deceive by producing misleading non-falsities, particularly when guided by few-shot prompting. Truth probes trained on standard true-false datasets are significantly better at detecting lies than at detecting deception without lying, confirming a critical blind spot of current mechanistic deception detection approaches.

---

### Paper 14: Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks
- **Authors**: (follow-up to Marks & Tegmark)
- **Year**: 2025-06
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2506.00823

**Abstract**:
Follow-up to Marks & Tegmark (2023). Examines whether the linear truth direction generalizes across logical transformations (negation, conjunction, disjunction) and across QA task formats. Tests probe stability across models and formats. Findings: the truth direction is largely stable within a model but shows partial degradation under compositional logical transformations.

---

### Paper 15: Strategic Collusion of LLM Agents: Market Division in Multi-Commodity Competitions
- **Authors**: Ryan Y. Lin, Siddhartha Ojha, Kevin Cai, Maxwell F. Chen
- **Year**: 2024-09
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2410.00031

**Abstract**:
Machine-learning technologies are seeing increased deployment in real-world market scenarios. In this work, we explore the strategic behaviors of large language models (LLMs) when deployed as autonomous agents in multi-commodity markets, specifically within Cournot competition frameworks. We examine whether LLMs can independently engage in anti-competitive practices such as collusion or, more specifically, market division. Our findings demonstrate that LLMs can effectively monopolize specific commodities by dynamically adjusting their pricing and resource allocation strategies, thereby maximizing profitability without direct human input or explicit collusion commands.

---

### Paper 16: Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems
- **Authors**: (see arXiv)
- **Year**: 2026-01
- **Venue**: arXiv preprint (pre-cutoff)
- **Source**: arXiv API
- **Identifier**: arXiv:2601.00360

**Abstract**:
As multi-agent AI systems become increasingly autonomous, evidence shows they can develop collusive strategies similar to those long observed in human markets and institutions. While human domains have accumulated centuries of anti-collusion mechanisms, it remains unclear how these can be adapted to AI settings. This paper (i) develops a taxonomy of human anti-collusion mechanisms — including sanctions, leniency & whistleblowing, monitoring & auditing, market design, and governance — and (ii) maps them to potential interventions for multi-agent AI systems. For each mechanism, we propose implementation approaches. We highlight open challenges such as the attribution problem (difficulty attributing emergent coordination to specific agents), identity fluidity (agents being easily forked or modified), and the boundary problem (distinguishing beneficial cooperation from collusion).

---

### Paper 17: A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring
- **Authors**: (see arXiv)
- **Year**: 2026-02
- **Venue**: arXiv preprint (pre-cutoff)
- **Source**: arXiv API
- **Identifier**: arXiv:2602.23163

**Abstract**:
Large language models are beginning to show steganographic capabilities. Such capabilities could allow misaligned models to evade oversight mechanisms. Classical definitions of steganography — and detection methods based on them — require a known reference distribution of non-steganographic signals; for steganographic reasoning in LLMs this is not feasible. The paper proposes a decision-theoretic view of steganography: it creates an asymmetry in usable information between agents that can and cannot decode the hidden content, and detection methods can be built from that asymmetry.

---

### Paper 18: Hallucination Detection with the Internal Layers of LLMs
- **Authors**: (see arXiv)
- **Year**: 2025-09
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2509.14254

**Abstract**:
Uses layer-wise linear probes to detect hallucination from internal states. Shows that middle-to-late layers concentrate the hallucination signal and that per-layer probe scores can be aggregated for a more robust detector than any single-layer probe.

---

### Paper 19: Understanding Multi-Agent LLM Frameworks: A Unified Benchmark and Experimental Analysis
- **Authors**: (see arXiv)
- **Year**: 2026-02
- **Venue**: arXiv preprint (pre-cutoff)
- **Source**: arXiv API
- **Identifier**: arXiv:2602.03128

**Abstract**:
Provides a unified benchmark for multi-agent LLM frameworks (debate, collaborative writing, code review, deliberation), evaluating a broad set of open- and closed-weight models. Establishes shared evaluation protocols that any multi-agent monitoring study can slot into as a text-only baseline.

---

_(Additional adjacent, lower-priority papers on algorithmic collusion in Cournot / two-sided markets — arXiv:1802.08061, 2407.04088, 2504.05335 — were retrieved but do not directly bear on the interpretability angle; retained only in the landscape.)_
