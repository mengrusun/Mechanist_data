# Raw Literature Retrieval: SAGE — iterative agentic pipeline for SAE feature natural-language explanation

**Date**: 2026-07-14
**Query**: Automatic natural-language explanation of Sparse Autoencoder (SAE) features in LLMs — iterative propose-test-revise agentic pipelines, Neuronpedia methodology, model-explains-model auto-interpretation (OpenAI/Anthropic), simulation / faithfulness scoring, generative and predictive accuracy metrics, polysemantic features, Gemma-Scope.
**Sources scanned**: arXiv API (5 queries), mechanic-db cloud SEARCH (in-flight at write time; results merged when returned), WebSearch. Zotero / Obsidian / local PDFs: not configured for this project (skipped).
**Query formulations used**:
- "sparse autoencoder feature interpretation explanation LLM"
- "automated interpretability neuron labeling LLM auto-interp"
- "Gemma Scope sparse autoencoder residual stream"
- "SAE feature explanation agent iterative propose test revise"
- "simulation scoring interpretability faithfulness neuron explanation"
- "automatically interpreting millions features Paulo sparse autoencoder"
- "linear explanations individual neurons prompt tuning"

**Policy note**: `/data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer/.claude/forbidden-urls.txt` blocks the SAGE paper itself (arXiv 2511.20820) and any arXiv YYMM ≥ 2511 as forbidden reference material for this reproduction task. That constraint is honored — no forbidden IDs / URLs / abstracts are cited or reproduced below. The captured behavior in `task.md` is treated as the authoritative specification for what to reproduce.

---

## Retrieved Papers

### Paper 1: Language Models Can Explain Neurons in Language Models
- **Authors**: Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, William Saunders
- **Year**: 2023
- **Venue**: OpenAI blog + technical report (no arXiv preprint; released as an interactive notebook and dataset)
- **Source**: WebSearch (openai.com, marktechpost, siliconangle)
- **Identifier**: openai.com/index/language-models-can-explain-neurons-in-language-models/
- **URL**: https://openai.com/index/language-models-can-explain-neurons-in-language-models/

**Abstract / Summary (from public write-up)**:
OpenAI applies GPT-4 to automatically write natural-language explanations for every one of the 307,200 neurons in GPT-2, and to *score* those explanations. The scoring method — **simulation scoring** — has GPT-4 use each written explanation to *simulate* what the target neuron would activate on, then compares those simulated activations to the neuron's true activations. Over 1,000 neurons received explanations scoring ≥ 0.8. The pipeline is a *single-pass* generation: propose one explanation from top-activating examples, score it, keep it. There is no iterative revision loop and no agentic experimentation. This is the foundational "model-explains-model" baseline that Neuronpedia and every follow-up (Paulo et al. 2024, and by direct analogy the SAGE reproduction target) refine or replace.

---

### Paper 2: Automatically Interpreting Millions of Features in Large Language Models
- **Authors**: Gonçalo Paulo, Alex Mallen, Caden Juang, Nora Belrose
- **Year**: 2024 (v1 Oct 2024; updated Aug 2025)
- **Venue**: arXiv preprint (EleutherAI)
- **Source**: arXiv API
- **Identifier**: arXiv:2410.13928
- **URL**: https://arxiv.org/abs/2410.13928

**Abstract**:
"While the activations of neurons in deep neural networks usually do not have a simple human-understandable interpretation, sparse autoencoders (SAEs) can be used to transform these activations into a higher-dimensional latent space which may be more easily interpretable. However, these SAEs can have millions of distinct latent features, making it infeasible for humans to manually interpret each one. In this work, we build an open-source automated pipeline to generate and evaluate natural language explanations for SAE features using LLMs. We test our framework on SAEs of varying sizes, activation functions, and losses, trained on two different open-weight LLMs. We introduce five new techniques to score the quality of explanations that are cheaper to run than the previous state of the art. One of these techniques, intervention scoring, evaluates the interpretability of the effects of intervening on a feature, which we find explains features that are not recalled by existing methods. We propose guidelines for generating better explanations that remain valid for a broader set of activating contexts, and discuss pitfalls with existing scoring techniques. We use our explanations to measure the semantic similarity of independently trained SAEs, and find that SAEs trained on nearby layers of the residual stream are highly similar. Our large-scale analysis confirms that SAE latents are indeed much more interpretable than neurons, even when neurons are sparsified using top-$k$ postprocessing."

**Notes**: Open-source pipeline (github.com/EleutherAI/sae-auto-interp) and public explanation dump (huggingface.co/datasets/EleutherAI/auto_interp_explanations). The five scoring techniques — including **detection scoring** (does the explanation let a scorer discriminate activating vs. non-activating contexts) and **intervention scoring** (does clamping the feature produce the effect the explanation predicts) — are the direct academic ancestors of SAGE's **predictive accuracy** and **generative accuracy** metrics respectively. This is the paper that the reproduction's evaluation code should benchmark against methodologically.

---

### Paper 3: Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2
- **Authors**: Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, Neel Nanda
- **Year**: 2024
- **Venue**: BlackboxNLP 2024 workshop (arXiv preprint)
- **Source**: arXiv API + ACL Anthology
- **Identifier**: arXiv:2408.05147
- **URL**: https://arxiv.org/abs/2408.05147

**Abstract**:
"Sparse autoencoders (SAEs) are an unsupervised method for learning a sparse decomposition of a neural network's latent representations into seemingly interpretable features. Despite recent excitement about their potential, research applications outside of industry are limited by the high cost of training a comprehensive suite of SAEs. In this work, we introduce Gemma Scope, an open suite of JumpReLU SAEs trained on all layers and sub-layers of Gemma 2 2B and 9B and select layers of Gemma 2 27B base models. We primarily train SAEs on the Gemma 2 pre-trained models, but additionally release SAEs trained on instruction-tuned Gemma 2 9B for comparison. We evaluate the quality of each SAE on standard metrics and release these results. We hope that by releasing these SAE weights, we can help make more ambitious safety and interpretability research easier for the community. Weights and a tutorial can be found at https://huggingface.co/google/gemma-scope and an interactive demo can be found at https://www.neuronpedia.org/gemma-scope"

**Notes**: This is the SAE checkpoint family (`gemmascope-res-16k`) that `task.md` binds for the main experiment. JumpReLU activation, residual-stream targets, 16k dictionary width. Neuronpedia hosts explanations for this suite — which is precisely the reference baseline SAGE is benchmarked against.

---

### Paper 4: Interpreting and Steering LLMs with Mutual Information-based Explanations on Sparse Autoencoders
- **Authors**: Xuansheng Wu, Jiayi Yuan, Wenlin Yao, Xiaoming Zhai, Ninghao Liu
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2502.15576
- **URL**: https://arxiv.org/abs/2502.15576

**Abstract**:
"Large language models (LLMs) excel at handling human queries, but they can occasionally generate flawed or unexpected responses. Understanding their internal states is crucial for understanding their successes, diagnosing their failures, and refining their capabilities. Although sparse autoencoders (SAEs) have shown promise for interpreting LLM internal representations, limited research has explored how to better explain SAE features, i.e., understanding the semantic meaning of features learned by SAE. Our theoretical analysis reveals that existing explanation methods suffer from the frequency bias issue, where they emphasize linguistic patterns over semantic concepts, while the latter is more critical to steer LLM behaviors. To address this, we propose using a fixed vocabulary set for feature interpretations and designing a mutual information-based objective, aiming to better capture the semantic meaning behind these features. We further propose two runtime steering strategies that adjust the learned feature activations based on their corresponding explanations. Empirical results show that, compared to baselines, our method provides more discourse-level explanations and effectively steers LLM behaviors to defend against jailbreak attacks."

**Notes**: Diagnoses a *frequency bias* pathology in Neuronpedia-style single-pass auto-interp — surface linguistic patterns crowd out semantic concepts. Direct contemporary competitor to SAGE, addressing the same weakness by a fundamentally different route (fixed vocabulary + MI objective vs. agentic revision).

---

### Paper 5: Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models
- **Authors**: Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, Aaron Mueller
- **Year**: 2024
- **Venue**: arXiv preprint (updated to v2 Mar 2025)
- **Source**: arXiv API
- **Identifier**: arXiv:2403.19647
- **URL**: https://arxiv.org/abs/2403.19647

**Abstract**:
"We introduce methods for discovering and applying sparse feature circuits. These are causally implicated subnetworks of human-interpretable features for explaining language model behaviors. Circuits identified in prior work consist of polysemantic and difficult-to-interpret units like attention heads or neurons, rendering them unsuitable for many downstream applications. In contrast, sparse feature circuits enable detailed understanding of unanticipated mechanisms. Because they are based on fine-grained units, sparse feature circuits are useful for downstream tasks: We introduce SHIFT, where we improve the generalization of a classifier by ablating features that a human judges to be task-irrelevant. Finally, we demonstrate an entirely unsupervised and scalable interpretability pipeline by discovering thousands of sparse feature circuits for automatically discovered model behaviors."

**Notes**: Provides the largest publicly-annotated SAE feature dataset (722 human-annotated features across Gemma 2 2B and Pythia 70M) — a natural human-alignment baseline for SAGE's generative-accuracy check.

---

### Paper 6: Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs
- **Authors**: Kola Ayonrinde, Michael T. Pearce, Lee Sharkey
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2410.11179
- **URL**: https://arxiv.org/abs/2410.11179

**Abstract**:
"Sparse Autoencoders (SAEs) have emerged as a useful tool for interpreting the internal representations of neural networks. However, naively optimising SAEs for reconstruction loss and sparsity results in a preference for SAEs that are extremely wide and sparse. We present an information-theoretic framework for interpreting SAEs as lossy compression algorithms for communicating explanations of neural activations. We appeal to the Minimal Description Length (MDL) principle to motivate explanations of activations which are both accurate and concise. We further argue that interpretable SAEs require an additional property, 'independent additivity': features should be able to be understood separately. We demonstrate an example of applying our MDL-inspired framework by training SAEs on MNIST handwritten digits and find that SAE features representing significant line segments are optimal, as opposed to SAEs with features for memorised digits from the dataset or small digit fragments. We argue that using MDL rather than sparsity may avoid potential pitfalls with naively maximising sparsity such as undesirable feature splitting and that this framework naturally suggests new hierarchical SAE architectures which provide more concise explanations."

**Notes**: Provides an information-theoretic scaffold for judging when an explanation is "good enough" — relevant if SAGE's `Reviewer` role needs a principled acceptance threshold.

---

### Paper 7: The Importance of Prompt Tuning for Automated Neuron Explanations
- **Authors**: Justin Lee, Tuomas Oikarinen, Arjun Chatha, Keng-Chi Chang, Yilan Chen, Tsui-Wei Weng
- **Year**: 2023
- **Venue**: arXiv preprint (workshop follow-up to Bills 2023)
- **Source**: arXiv API
- **Identifier**: arXiv:2310.06200
- **URL**: https://arxiv.org/abs/2310.06200

**Abstract**:
"Recent advances have greatly increased the capabilities of large language models (LLMs), but our understanding of the models and their safety has not progressed as fast. In this paper we aim to understand LLMs deeper by studying their individual neurons. We build upon previous work showing large language models such as GPT-4 can be useful in explaining what each neuron in a language model does. Specifically, we analyze the effect of the prompt used to generate explanations and show that reformatting the explanation prompt in a more natural way can significantly improve neuron explanation quality and greatly reduce computational cost. We demonstrate the effects of our new prompts in three different ways, incorporating both automated and human evaluations."

**Notes**: Establishes that prompt design alone dramatically shifts one-shot auto-interp quality — a strong motivation for structured multi-round pipelines like SAGE.

---

### Paper 8: Linear Explanations for Individual Neurons
- **Authors**: Tuomas Oikarinen, Tsui-Wei Weng
- **Year**: 2024
- **Venue**: ICML 2024 (arXiv preprint)
- **Source**: arXiv API
- **Identifier**: arXiv:2405.06855
- **URL**: https://arxiv.org/abs/2405.06855

**Abstract**:
"In recent years many methods have been developed to understand the internal workings of neural networks, often by describing the function of individual neurons in the model. However, these methods typically only focus on explaining the very highest activations of a neuron. In this paper we show this is not sufficient, and that the highest activation range is only responsible for a very small percentage of the neuron's causal effect. In addition, inputs causing lower activations are often very different and can't be reliably predicted by only looking at high activations. We propose that neurons should instead be understood as a linear combination of concepts, and develop an efficient method for producing these linear explanations. In addition, we show how to automatically evaluate description quality using simulation, i.e. predicting neuron activations on unseen inputs in vision setting."

**Notes**: Shows that top-activation-only explanations under-cover a feature's causal footprint. Direct motivation for SAGE's Designer role probing across the full activation range rather than only top-k snippets.

---

### Paper 9: PrivacyScalpel: Enhancing LLM Privacy via Interpretable Feature Intervention with Sparse Autoencoders
- **Authors**: Ahmed Frikha, Muhammad Reza Ar Razi, Krishna Kanth Nakka, Ricardo Mendes, Xue Jiang, Xuebing Zhou
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2503.11232
- **URL**: https://arxiv.org/abs/2503.11232

**Abstract**:
"Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing but also pose significant privacy risks by memorizing and leaking Personally Identifiable Information (PII). Existing mitigation strategies, such as differential privacy and neuron-level interventions, often degrade model utility or fail to effectively prevent leakage. To address this challenge, we introduce PrivacyScalpel, a novel privacy-preserving framework that leverages LLM interpretability techniques to identify and mitigate PII leakage while maintaining performance. PrivacyScalpel comprises three key steps: (1) Feature Probing … (2) Sparse Autoencoding … (3) Feature-Level Interventions … Our empirical evaluation on Gemma2-2b and Llama2-7b, fine-tuned on the Enron dataset, shows that PrivacyScalpel significantly reduces email leakage from 5.15% to as low as 0.0%, while maintaining over 99.4% of the original model's utility."

**Notes**: Downstream application of SAE-feature-level knowledge — useful example of what better explanations enable, and confirms Gemma-2-2B as an active SAE research platform.

---

### Paper 10: Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures
- **Authors**: Mark Muchane, Sean Richardson, Kiho Park, Victor Veitch
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2506.01197
- **URL**: https://arxiv.org/abs/2506.01197

**Abstract**:
"Sparse dictionary learning (and, in particular, sparse autoencoders) attempts to learn a set of human-understandable concepts that can explain variation on an abstract space. A basic limitation of this approach is that it neither exploits nor represents the semantic relationships between the learned concepts. In this paper, we introduce a modified SAE architecture that explicitly models a semantic hierarchy of concepts. Application of this architecture to the internal representations of large language models shows both that semantic hierarchy can be learned, and that doing so improves both reconstruction and interpretability."

**Notes**: Suggests that some polysemanticity/multi-concept behavior is naturally hierarchical — a nudge that a good iterative explainer should be aware of concept hierarchy, not just parallel candidates.

---

### Paper 11: Transcoders Beat Sparse Autoencoders for Interpretability
- **Authors**: Gonçalo Paulo, Stepan Shabalin, Nora Belrose
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2501.18823
- **URL**: https://arxiv.org/abs/2501.18823

**Abstract**:
"Sparse autoencoders (SAEs) extract human-interpretable features from deep neural networks by transforming their activations into a sparse, higher dimensional latent space, and then reconstructing the activations from these latents. Transcoders are similar to SAEs, but they are trained to reconstruct the output of a component of a deep network given its input. In this work, we compare the features found by transcoders and SAEs trained on the same model and data, finding that transcoder features are significantly more interpretable. We also propose skip transcoders, which add an affine skip connection to the transcoder architecture, and show that these achieve lower reconstruction loss with no effect on interpretability."

**Notes**: Motivates the Qwen3-4B / `transcoder-hp` verify variant in `task.md` — the reproduction should generalize across SAE families, not only JumpReLU residual-stream SAEs.

---

### Paper 12: Sparse Autoencoders Trained on the Same Data Learn Different Features
- **Authors**: Gonçalo Paulo, Nora Belrose
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2501.16615
- **URL**: https://arxiv.org/abs/2501.16615

**Abstract**:
"Sparse autoencoders (SAEs) are a useful tool for uncovering human-interpretable features in the activations of large language models (LLMs). While some expect SAEs to find the true underlying features used by a model, our research shows that SAEs trained on the same model and data, differing only in the random seed used to initialize their weights, identify different sets of features. For example, in an SAE with 131K latents trained on a feedforward network in Llama 3 8B, only 30% of the features were shared across different seeds. We observed this phenomenon across multiple layers of three different LLMs, two datasets, and several SAE architectures."

**Notes**: Motivates evaluating an explainer *per-feature* (feature identity is seed-dependent). A comparison protocol that assumes shared features across two SAEs is unsound; SAGE reproduction should benchmark against **Neuronpedia's own explanations for the same feature ids in the same SAE checkpoint**, which the task specifies.

---

## Additional context (non-arXiv, from public write-ups)

### Neuronpedia (Lin, McDougall, et al. — neuronpedia.org)
- **Source**: WebSearch (neuronpedia.org/gemma-scope)
- **Summary**: Interactive frontend that hosts SAE feature explanations, activating examples, top-k dashboards, and steering demos. For Gemma-Scope specifically, Neuronpedia's `auto-interp` explanations are AI-generated by prompting a proprietary LLM on top-activating snippets. These are the reference **`neuronpedia-baseline` explanations** the reproduction compares against.

### Scaling Monosemanticity (Anthropic, Templeton et al., May 2024)
- **Source**: WebSearch (transformer-circuits.pub)
- **Summary**: Anthropic's application of SAEs to Claude 3 Sonnet, at 1M / 4M / 34M features. Explanations are produced by having a stronger model summarize top-activating and steering-effect examples — the "influence-function-like" auto-interp pattern SAGE's Analyzer role generalizes.

---

## Mechanic-db results (Lane A)

The mechanic-db cloud SEARCH job (job id `2b1895b7-1e95-4332-922d-859f5eef4e6f`) was submitted for the query above with `temporal_mode=recent`, `year_min=2023`, and closed enums `techniques=[feature_dictionary_learning, neural_feature_learning]`, `components=[residual_stream, neuron]`. If the job completes before Phase 2 hands off, its `papers[]` list is merged into the landscape via file
`/data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer/mechanic_db_cache/20260714_002143_sae_auto_interp.json`.
The Lane A results are additive; Phase 2's faithful behavior capture from `task.md` does not depend on them.
