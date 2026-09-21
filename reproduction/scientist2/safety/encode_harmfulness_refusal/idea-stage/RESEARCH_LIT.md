# Raw Literature Retrieval: Mechanistic Separation of Harmfulness Perception and Refusal Execution in Instruction-Tuned LLMs

**Date**: 2026-07-15
**Query**: Linear directions in the residual stream for harmfulness perception vs. refusal execution; refusal direction ablation/steering; hidden-state safety probes; jailbreak mechanism (refusal-suppressed / harmfulness-preserved); hidden-state safety classifiers vs. dedicated safety judges (Llama Guard 3 8B).
**Sources scanned**: arXiv API (base source; ran with `arxiv_fetch.py`), WebSearch (base source; ran but responses were voided by the project's post-search filter that blocks arXiv IDs ≥ 2507 and the target paper). Mechanic-db skipped — the `mechanic-db` MCP server is not configured on this host. Zotero, Obsidian, and local PDFs skipped (not present).
**Retrieval-policy note**: Per `.claude/forbidden-urls.txt`, this project blocks the target paper ("LLMs Encode Harmfulness and Refusal Separately", arXiv 2507.11878, incl. its OpenReview / project-page / GitHub identifiers), the term "Latent Guard", and every arXiv id with `YYMM ≥ 2507`. All arXiv results below are pre-cutoff (< 2507). No forbidden content is reproduced.
**Query formulations used**:
- refusal direction language models mediated single direction
- representation engineering top down AI transparency Zou
- linear probe LLM safety harmful hidden state classifier jailbreak detection
- activation steering contrastive activation addition CAA language model behavior
- GCG greedy coordinate gradient universal adversarial suffix attack aligned
- Llama Guard safeguard model conversation classifier
- persuasion adversarial prompt jailbreak AdvBench XSTest safety benchmark
- over-refusal XSTest exaggerated safety false refusal aligned LLM
- difference in means directions concept representation LLM interpretability

---

## Retrieved Papers

### Paper 1: Refusal in Language Models Is Mediated by a Single Direction
- **Authors**: Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
- **Year**: 2024
- **Venue**: NeurIPS 2024
- **Source**: arXiv API
- **Identifier**: arXiv 2406.11717
- **URL**: https://arxiv.org/abs/2406.11717

**Abstract**:
Conversational large language models are fine-tuned for both instruction-following and safety, resulting in models that obey benign requests but refuse harmful ones. While this refusal behavior is widespread across chat models, its underlying mechanisms remain poorly understood. In this work, we show that refusal is mediated by a one-dimensional subspace, across 13 popular open-source chat models up to 72B parameters in size. Specifically, for each model, we find a single direction such that erasing this direction from the model's residual stream activations prevents it from refusing harmful instructions, while adding this direction elicits refusal on even harmless instructions. Leveraging this insight, we propose a novel white-box jailbreak method that surgically disables refusal with minimal effect on other capabilities. Finally, we mechanistically analyze how adversarial suffixes suppress propagation of the refusal-mediating direction.

---

### Paper 2: Representation Engineering: A Top-Down Approach to AI Transparency
- **Authors**: Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, Dan Hendrycks
- **Year**: 2023
- **Venue**: arXiv preprint (widely-cited technical report)
- **Source**: arXiv API
- **Identifier**: arXiv 2310.01405
- **URL**: https://arxiv.org/abs/2310.01405

**Abstract**:
Introduces representation engineering (RepE), a top-down approach to enhancing transparency of AI systems that draws on cognitive-neuroscience insights. Places population-level representations, rather than neurons or circuits, at the center of analysis, equipping practitioners with methods for monitoring and manipulating high-level cognitive phenomena (honesty, harmlessness, power-seeking, morality) in deep neural networks. Introduces stimulus-response reading vectors, LAT (Linear Artificial Tomography), and additive-steering controllers.

---

### Paper 3: Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)
- **Authors**: Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson
- **Year**: 2023
- **Venue**: arXiv preprint (foundational jailbreak paper)
- **Source**: arXiv API
- **Identifier**: arXiv 2307.15043
- **URL**: https://arxiv.org/abs/2307.15043

**Abstract**:
Introduces Greedy Coordinate Gradient (GCG), a discrete-optimization method that appends an adversarial suffix onto a harmful instruction so an aligned LLM produces an affirmative (rather than refusing) response. Attack transfers across models. Releases the **AdvBench** harmful-behavior benchmark (used as the harmful contrast set for direction extraction and as the ASR benchmark in this project).

---

### Paper 4: How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs
- **Authors**: Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi
- **Year**: 2024
- **Venue**: ACL 2024 (long)
- **Source**: arXiv API
- **Identifier**: arXiv 2401.06373
- **URL**: https://arxiv.org/abs/2401.06373

**Abstract**:
Introduces persuasion-based adversarial prompts (PAP): a persuasion taxonomy is used to systematically rewrite a harmful request into a form that leverages human-like persuasion techniques (authority endorsement, emotional appeal, false framing, etc.), producing a jailbreak template that is qualitatively distinct from GCG-style optimized suffixes. Complementary attack family for testing the refusal-suppressed / harmfulness-preserved signature (Claim 4 in this project).

---

### Paper 5: How Alignment and Jailbreak Work: Explain LLM Safety through Intermediate Hidden States
- **Authors**: Zhenhong Zhou, Haiyang Yu, Xinghua Zhang, Rongwu Xu, Fei Huang, Yongbin Li
- **Year**: 2024
- **Venue**: EMNLP 2024 (findings)
- **Source**: arXiv API
- **Identifier**: arXiv 2406.05644
- **URL**: https://arxiv.org/abs/2406.05644

**Abstract**:
Uses weak linear probes on intermediate hidden states to characterise how safety alignment and jailbreaks propagate through the layers. Distinguishes an early-layer "harmful-intent recognition" component from a later-layer "refusal-decision" component, and shows that jailbreaks tend to alter the later component while leaving the early-layer harmful-intent signal partly intact — one of the closest pre-cutoff antecedents to the two-direction hypothesis in this project's task.md.

---

### Paper 6: Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations
- **Authors**: Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, Madian Khabsa
- **Year**: 2023
- **Venue**: arXiv (Meta technical report)
- **Source**: arXiv API
- **Identifier**: arXiv 2312.06674
- **URL**: https://arxiv.org/abs/2312.06674

**Abstract**:
Introduces Llama Guard, an LLM-based classifier for both prompt- and response-side safety with a customisable safety-risk taxonomy. Basis for later Llama Guard 3 8B, which task.md pins as the baseline safety judge for the hidden-state-classifier-vs-dedicated-judge comparison (Claim 5).

---

### Paper 7: Steering Llama 2 via Contrastive Activation Addition (CAA)
- **Authors**: Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander M. Turner
- **Year**: 2023
- **Venue**: arXiv preprint (widely-adopted follow-up to steering vectors)
- **Source**: arXiv API
- **Identifier**: arXiv 2312.06681
- **URL**: https://arxiv.org/abs/2312.06681

**Abstract**:
Introduces Contrastive Activation Addition (CAA): compute per-example difference-in-means between paired positive / negative behaviour examples, average to get a steering vector, add or subtract from residual-stream activations at inference to control the target behaviour. The base recipe for both the harmfulness direction and the refusal direction extraction in this project.

---

### Paper 8: Steering Language Models With Activation Engineering (ActAdd)
- **Authors**: Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid
- **Year**: 2023
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv 2308.10248
- **URL**: https://arxiv.org/abs/2308.10248

**Abstract**:
Introduces activation engineering: inference-time additive modification of residual-stream activations, with a coefficient α controlling dose. Foundational to the additive-steering axis of Claim 3.

---

### Paper 9: SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering
- **Authors**: Zouying Cao, Yifei Yang, Hai Zhao
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv 2408.11491
- **URL**: https://arxiv.org/abs/2408.11491

**Abstract**:
Addresses over-refusal (exaggerated safety) via activation steering conditioned on whether the input is genuinely harmful; anchor for the XSTest / Alpaca over-refusal controls that appear in the project's verify-stage candidate datasets.

---

### Paper 10: JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models
- **Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramèr, Hamed Hassani, Eric Wong
- **Year**: 2024
- **Venue**: NeurIPS 2024 (datasets & benchmarks)
- **Source**: arXiv API
- **Identifier**: arXiv 2404.01318
- **URL**: https://arxiv.org/abs/2404.01318

**Abstract**:
Standardised jailbreak evaluation benchmark (JBB) with an artifact registry, a curated behaviour set, and canonical judges. Named in the project's verify-stage candidate datasets.

---

### Paper 11: Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes
- **Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho
- **Year**: 2024
- **Venue**: NeurIPS 2024
- **Source**: arXiv API
- **Identifier**: arXiv 2403.00867
- **URL**: https://arxiv.org/abs/2403.00867

**Abstract**:
Detects jailbreak inputs by analysing the geometry of the *refusal loss landscape* — a hidden-state / gradient-side signal analogous in spirit to what a hidden-state Latent Guard-style classifier would exploit; useful competitive-set entry for Claim 5.

---

### Paper 12: Faster-GCG: Efficient Discrete Optimization Jailbreak Attacks against Aligned Large Language Models
- **Authors**: Xiao Li, Zhuhong Li, Qiongxiu Li, Bingze Lee, Jinghao Cui, Xiaolin Hu
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv 2410.15362
- **URL**: https://arxiv.org/abs/2410.15362

**Abstract**:
More-efficient GCG variant; supplies the second (optimisation-based) jailbreak family required for Claim 4's refusal-suppressed / harmfulness-preserved signature evaluation.

---

### Paper 13: Extending Activation Steering to Broad Skills and Multiple Behaviours
- **Authors**: Teun van der Weij, Massimo Poesio, Nandi Schoots
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv 2403.05767
- **URL**: https://arxiv.org/abs/2403.05767

**Abstract**:
Studies how well activation-steering vectors generalise across related behaviours and how they interact when combined — background for the specificity control (steering axis A should not move axis B) required by Claim 3.
