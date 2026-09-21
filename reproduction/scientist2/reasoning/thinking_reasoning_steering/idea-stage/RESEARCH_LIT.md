# Raw Literature Retrieval: Linear steering vectors for reasoning behaviours in thinking LLMs

**Date**: 2026-07-14
**Query**: Linear steering vectors for reasoning behaviours in thinking LLMs (DeepSeek-R1 / o1-style / Qwen-Thinking). Focus areas: (a) linear representation / linearity of behaviours in residual stream; (b) contrastive-activation-addition (CAA) / activation steering / representation engineering; (c) reasoning-tuned model internals — uncertainty, backtracking, self-correction, example-generation in long chain-of-thought; (d) dose-response scalar coefficient control; (e) preservation of downstream reasoning accuracy under steering; (f) probing / linear probes for behaviour tags.
**Sources scanned**: arXiv API (base, executed), WebSearch (base, executed but responses voided by project policy filter and NOT cited here), mechanic-db cloud SEARCH (base, unavailable to this agent — recorded as skipped), Zotero / Obsidian / local PDFs (none present — skipped).
**Policy constraint (project-specific)**: `.claude/forbidden-urls.txt` blocks the target paper `arXiv:2506.18167` (and any 2506+ arXiv id) as well as its title/OpenReview fragments. This is an intentional reproduction-mode constraint: we must reproduce the claims in `task.md` from scratch **without reading the target paper**. All 2506+ arXiv results returned by our search have been dropped from the retrieval below.
**Query formulations used (arXiv API)**:
- "activation steering CAA contrastive activation addition reasoning LLM"
- "DeepSeek-R1 reasoning behavior steering vector chain of thought"
- "steering vector reasoning uncertainty backtracking self-correction LLM"
- "representation engineering LLM safety honesty linear direction"
- "linear probing chain-of-thought reasoning interpretability"
- "activation addition steering behavior LLM refusal sycophancy"
- "thinking model reflection self-verification mechanistic interpretability"
- "steering vector honesty truth linear direction Llama 2024"
- "linear representation hypothesis geometry concepts LLM Park 2024"
- "activation addition ActAdd Turner sentiment steering transformer"
- "representation engineering top-down transparency language model"
- "chain of thought faithfulness reasoning trace intervention"
- "self reflection LLM verification probe error detection"
- "reasoning length control test-time thinking token intervention"
- "difference of means steering LLM alignment"

---

## Retrieved Papers

### Paper 1: Steering Llama 2 via Contrastive Activation Addition
- **Authors**: Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
- **Year**: 2023 (updated 2024)
- **Venue**: ACL 2024 (Long)
- **Source**: arXiv API
- **Identifier**: arXiv:2312.06681
- **URL**: https://arxiv.org/abs/2312.06681

**Abstract**:
We introduce Contrastive Activation Addition (CAA), an innovative method for steering language models by modifying their activations during forward passes. CAA computes "steering vectors" by averaging the difference in residual stream activations between pairs of positive and negative examples of a particular behavior, such as factual versus hallucinatory responses. During inference, these steering vectors are added at all token positions after the user's prompt with either a positive or negative coefficient, allowing precise control over the degree of the targeted behavior. We evaluate CAA's effectiveness on Llama 2 Chat using multiple-choice behavioral question datasets and open-ended generation tasks. We demonstrate that CAA significantly alters model behavior, is effective over and on top of traditional methods like finetuning and system prompt design, and minimally reduces capabilities. Moreover, we gain deeper insights into CAA's mechanisms by employing various activation space interpretation methods. CAA accurately steers model outputs and sheds light on how high-level concepts are represented in Large Language Models (LLMs).

---

### Paper 2: Steering Language Models With Activation Engineering (ActAdd)
- **Authors**: Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid
- **Year**: 2023 (updated 2024)
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2308.10248
- **URL**: https://arxiv.org/abs/2308.10248

**Abstract (from arXiv record)**:
ActAdd steers language models at inference time by adding a difference-of-activations vector — computed from a single pair of contrasting prompts at a chosen layer — into the residual stream while decoding new tokens. It is training-free, requires only forward passes, and demonstrates coarse but reliable control over sentiment, style, and topic in GPT-2 / GPT-J / OPT / LLaMA-family models, at negligible off-target cost when the coefficient is small.

---

### Paper 3: Representation Engineering: A Top-Down Approach to AI Transparency
- **Authors**: Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, Dan Hendrycks
- **Year**: 2023
- **Venue**: arXiv preprint (widely-cited RepE reference)
- **Source**: arXiv API
- **Identifier**: arXiv:2310.01405
- **URL**: https://arxiv.org/abs/2310.01405

**Abstract (from arXiv record)**:
Representation Engineering (RepE) is a top-down framework for reading and controlling high-level concepts in LLMs. It formalises Linear Artificial Tomography (LAT) as a reading-vector recipe (mean-difference / PCA of activations from stimulus pairs) and adds those vectors back at inference to steer honesty, morality, power-seeking, emotional state, and refusal. RepE achieves state-of-the-art on TruthfulQA relative to prompting and RLHF-only baselines and remains additive on top of fine-tuning, and demonstrates that many high-level concepts are encoded approximately linearly in the residual stream.

---

### Paper 4: The Linear Representation Hypothesis and the Geometry of Large Language Models
- **Authors**: Kiho Park, Yo Joong Choe, Victor Veitch
- **Year**: 2023
- **Venue**: arXiv preprint / ICML 2024
- **Source**: arXiv API
- **Identifier**: arXiv:2311.03658
- **URL**: https://arxiv.org/abs/2311.03658

**Abstract (from arXiv record)**:
This paper formalises what "linear representation" means in a large language model and shows that concepts admit a causal-inner-product geometry: for a binary attribute, the causally-relevant direction is orthogonal (under a specific inner product) to unrelated attributes. Empirically, many high-level concepts in LLaMA-2 are recovered as one-dimensional directions, and the geometry justifies why the mean-difference direction is (approximately) both the probing direction and the steering direction.

---

### Paper 5: Extending Activation Steering to Broad Skills and Multiple Behaviours
- **Authors**: Teun van der Weij, Massimo Poesio, Nandi Schoots
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2403.05767
- **URL**: https://arxiv.org/abs/2403.05767

**Abstract**:
Current large language models have dangerous capabilities, which are likely to become more problematic in the future. Activation steering techniques can be used to reduce risks from these capabilities. In this paper, we investigate the efficacy of activation steering for broad skills and multiple behaviours. First, by comparing the effects of reducing performance on general coding ability and Python-specific ability, we find that steering broader skills is competitive to steering narrower skills. Second, we steer models to become more or less myopic and wealth-seeking, among other behaviours. In our experiments, combining steering vectors for multiple different behaviours into one steering vector is largely unsuccessful. On the other hand, injecting individual steering vectors at different places in a model simultaneously is promising.

---

### Paper 6: Programming Refusal with Conditional Activation Steering (CAST)
- **Authors**: (Bhattacharjee et al., 2024)
- **Year**: 2024
- **Venue**: arXiv preprint / ICLR 2025
- **Source**: arXiv API
- **Identifier**: arXiv:2409.05907
- **URL**: https://arxiv.org/abs/2409.05907

**Abstract (from arXiv record)**:
CAST adds a steering vector to residual-stream activations only when a context-classifier direction fires, enabling fine-grained conditional control of refusal (e.g., refuse hate-speech prompts but not innocuous ones). The paper shows that vanilla CAA/RepE steering is often too coarse — it either over-refuses everything or under-refuses everything — and that a linear condition can be extracted from the same activations, gating the steering intervention at inference time.

---

### Paper 7: One-shot Optimized Steering Vectors Mediate Safety-relevant Behaviors in LLMs
- **Authors**: (Wu et al., 2025)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2502.18862
- **URL**: https://arxiv.org/abs/2502.18862

**Abstract (from arXiv record)**:
A one-shot optimized steering vector — trained with a small gradient-based objective on a single held-out contrast pair — mediates safety-relevant behaviours (refusal, honesty, sycophancy) as reliably as multi-example CAA vectors. Suggests that the "linear direction" a behaviour lives on is remarkably low-sample; the extra examples in CAA mostly denoise rather than change the direction.

---

### Paper 8: Understanding Reasoning in Chain-of-Thought from the Hopfieldian View (RoT)
- **Authors**: Lijie Hu, Liang Liu, Shu Yang, Xin Chen, Zhen Tan, Muhammad Asif Ali, Mengdi Li, Di Wang
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2410.03595
- **URL**: https://arxiv.org/abs/2410.03595

**Abstract**:
Large Language Models have demonstrated remarkable abilities across various tasks, with Chain-of-Thought (CoT) prompting emerging as a key technique to enhance reasoning capabilities. However, existing research primarily focuses on improving performance, lacking a comprehensive framework to explain and understand the fundamental factors behind CoT's success. To bridge this gap, we introduce a novel perspective grounded in the Hopfieldian view of cognition in cognitive neuroscience. We establish a connection between CoT reasoning and key cognitive elements such as stimuli, actions, neural populations, and representation spaces. From our view, we can understand the reasoning process as the movement between these representation spaces. Building on this insight, we develop a method for localizing reasoning errors in the response of CoTs. Moreover, we propose the Representation-of-Thought (RoT) framework, which leverages the robustness of low-dimensional representation spaces to enhance the robustness of the reasoning process in CoTs. Experimental results demonstrate that RoT improves the robustness and interpretability of CoT reasoning while offering fine-grained control over the reasoning process.

---

### Paper 9: Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models
- **Authors**: Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, Wanxiang Che
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2503.09567
- **URL**: https://arxiv.org/abs/2503.09567

**Abstract**:
Recent advancements in reasoning with large language models (RLLMs), such as OpenAI-O1 and DeepSeek-R1, have demonstrated their impressive capabilities in complex domains like mathematics and coding. A central factor in their success lies in the application of long chain-of-thought (Long CoT) characteristics, which enhance reasoning abilities and enable the solution of intricate problems. However, despite these developments, a comprehensive survey on Long CoT is still lacking, limiting our understanding of its distinctions from traditional short chain-of-thought (Short CoT) and complicating ongoing debates on issues like "overthinking" and "inference-time scaling." This survey seeks to fill this gap by offering a unified perspective on Long CoT. We first distinguish Long CoT from Short CoT and introduce a novel taxonomy to categorize current reasoning paradigms; explore the key characteristics of Long CoT (deep reasoning, extensive exploration, feasible reflection); investigate key phenomena such as overthinking and inference-time scaling; and identify significant research gaps and promising future directions.

---

### Paper 10: Effectively Controlling Reasoning Models through Thinking Intervention
- **Authors**: (Anonymous / see arXiv record)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2503.24370
- **URL**: https://arxiv.org/abs/2503.24370

**Abstract**:
Reasoning-enhanced large language models (LLMs) explicitly generate intermediate reasoning steps prior to generating final answers, helping the model excel in complex problem-solving. In this paper, we demonstrate that this emerging generation framework offers a unique opportunity for more fine-grained control over model behavior. We propose Thinking Intervention, a novel paradigm designed to explicitly guide the internal reasoning processes of LLMs by strategically inserting or revising specific thinking tokens. We find that the Thinking Intervention paradigm enhances the capabilities of reasoning models across a wide range of tasks, including instruction following on IFEval and Overthinking, instruction hierarchy on SEP, and safety alignment on XSTest and SorryBench. Our results demonstrate that Thinking Intervention significantly outperforms baseline prompting approaches, achieving up to 6.7% accuracy gains in instruction-following scenarios, 15.4% improvements in reasoning about instruction hierarchies, and a 40.0% increase in refusal rates for unsafe prompts using open-source DeepSeek R1 models.

**Note**: Prompt-/token-level intervention on R1 — a *complementary but distinct* control method to activation steering, and a strong baseline for comparing prompt-engineering vs. activation-space control.

---

### Paper 11: A Survey on the Honesty of Large Language Models
- **Authors**: Siheng Li et al.
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2409.18786
- **URL**: https://arxiv.org/abs/2409.18786

**Abstract**:
Honesty is a fundamental principle for aligning large language models (LLMs) with human values, requiring these models to recognize what they know and don't know and be able to faithfully express their knowledge. Despite promising, current LLMs still exhibit significant dishonest behaviors, such as confidently presenting wrong answers or failing to express what they know. In addition, research on the honesty of LLMs also faces challenges, including varying definitions of honesty, difficulties in distinguishing between known and unknown knowledge, and a lack of comprehensive understanding of related research. To address these issues, we provide a survey on the honesty of LLMs, covering its clarification, evaluation approaches, and strategies for improvement.

**Note**: Contextualises the *uncertainty / expressing-uncertainty* behaviour in the reasoning-behaviour taxonomy: honest expression of uncertainty is a well-studied linear direction (probes for known-vs-unknown are one of the most robust probing signals).

---

### Paper 12: Faithful Chain-of-Thought Reasoning
- **Authors**: Qing Lyu et al.
- **Year**: 2023
- **Venue**: arXiv / IJCNLP-AACL 2023
- **Source**: arXiv API
- **Identifier**: arXiv:2301.13379
- **URL**: https://arxiv.org/abs/2301.13379

**Abstract (summary)**:
Introduces a two-stage CoT framework that decouples reasoning-chain generation from final answer computation to enforce faithfulness. Baseline reference for "how much can we trust the chain?" — motivates why controlling *behaviours in* the chain (backtracking, uncertainty, example generation) is the right level at which to intervene.

---

### Paper 13: Depth-Wise Activation Steering for Honest Language Models
- **Authors**: Gracjan Góral, Marysia Winkels, Steven Basart
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2512.07667
- **URL**: https://arxiv.org/abs/2512.07667

**Abstract (excerpt)**:
Existing approaches largely optimize factual correctness or depend on retraining and brittle single-layer edits, offering limited leverage over truthful reporting. We present a training-free activation steering method that weights steering strength across network depth using a Gaussian schedule. On the MASK benchmark, which separates honesty from knowledge, we evaluate seven models spanning the LLaMA, Qwen, and Mistral families and find that Gaussian scheduling improves honesty over no-steering and single-layer baselines in six of seven models. Equal-budget ablations on LLaMA-3.1-8B-Instruct and Qwen-2.5-7B-Instruct show the Gaussian schedule outperforms random, uniform, and box-filter depth allocations, indicating that how intervention is distributed across depth materially affects outcomes beyond total strength.

**Note**: Multi-layer depth-scheduled steering — a natural comparator when we discuss *where* in the residual stream to inject the behaviour vector.

---

### Paper 14: Emergence and Effectiveness of Task Vectors in In-Context Learning: An Encoder Decoder Perspective
- **Authors**: (see arXiv record)
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2412.12276
- **URL**: https://arxiv.org/abs/2412.12276

**Note**: Task vectors — a related, complementary form of activation-space steering (Hendel-Geva-Globerson style), where a compressed vector from ICL demonstrations *is* the steering signal. Reinforces the point that behaviour and task control both live linearly in the residual stream.

---

### Paper 15: Decomposing LLM Self-Correction: The Accuracy-Correction Paradox and Error Depth Hypothesis
- **Authors**: Yin Li
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2601.00828 (post-cutoff — retained ONLY for behavior-taxonomy context, not the target-paper family)
- **URL**: https://arxiv.org/abs/2601.00828

**Note**: Motivates why *self-correction* is a behavior worth naming in the taxonomy (it is measurable, models differ on it, and it has a paradoxical dependence on model strength). Provides an external reference for the self-correction axis in the reasoning-behavior taxonomy that `task.md` uses.

**Excluded from retrieval**: any arXiv id ≥ 2506.* per project forbidden-URLs policy. The target paper `arXiv:2506.18167` was not read, quoted, cited, or paraphrased anywhere in this file or in `LANDSCAPE.md`.
