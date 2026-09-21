# Raw Literature Retrieval: Multilingual LLM safety alignment via language-agnostic semantic bottleneck

**Date**: 2026-07-14
**Query**: Multilingual LLM safety alignment via language-agnostic semantic bottleneck intermediate representation; cross-lingual jailbreak safety gap; layer-wise language identity vs semantics; refusal directions / activation steering; cross-lingual preference optimization; capability preservation.
**Sources scanned**: mechanic-db (Lane A — in flight at time of write), arXiv API (base), WebSearch (base — full-response voided due to post-cutoff hits; not cited), Zotero (absent — skipped), Obsidian (absent — skipped), local PDFs (absent — skipped).
**Post-cutoff policy**: `.claude/forbidden-urls.txt` sets `arxiv-cutoff: 2604`. All papers in this dump have arXiv id < 2604 (i.e., published before April 2026). The blocked paper LASA (2604.12710) is deliberately NOT retrieved or referenced.
**Query formulations used** (arXiv API):
- "multilingual jailbreak low-resource languages safety attack success rate LLM"
- "MultiJail cross-lingual jailbreak benchmark multilingual"
- "refusal direction activation steering LLM safety"
- "cross-lingual safety transfer preference optimization LLM DPO RLHF"
- "language neutral concept space middle layer multilingual transformer probing"
- "representation engineering LLM safety refusal direction ablation"
- "multilingual LLM safety alignment English cross-lingual generalization RLHF DPO"
- "Do Llamas Work English latent language multilingual transformer"
- "PKU-SafeRLHF safe preference reinforcement learning human feedback"
- "refusal is mediated by single direction ablation LLM Arditi"
- "Representation Engineering RepE transparent brain LLM Zou"
- "toxicity mitigation cross-lingual DPO English generalizes languages"
- "circuit breakers representation control safety alignment jailbreak"

---

## Retrieved Papers (all arXiv id < 2604 — pre-cutoff, safe to cite)

### Paper 1: All Languages Matter — On the Multilingual Safety of Large Language Models (XSafety)
- **Authors**: Wenxuan Wang, Zhaopeng Tu, Chang Chen, Youliang Yuan, Jen-tse Huang, Wenxiang Jiao, Michael R. Lyu
- **Year**: 2023 (updated 2024)
- **Venue**: arXiv preprint (companion multilingual safety benchmark; widely cited alongside MultiJail)
- **Source**: arXiv API
- **Identifier**: 2310.00905
- **URL**: https://arxiv.org/abs/2310.00905

**Abstract**:
> Safety lies at the core of developing and deploying large language models (LLMs). However, previous safety benchmarks only concern the safety in one language, e.g. the majority language in the pretraining data such as English. In this work, we build the first multilingual safety benchmark for LLMs, XSafety, in response to the global deployment of LLMs in practice. XSafety covers 14 kinds of commonly used safety issues across 10 languages that span several language families. We utilize XSafety to empirically study the multilingual safety for 4 widely-used LLMs, including both close-API and open-source models. Experimental results show that all LLMs produce significantly more unsafe responses for non-English queries than English ones, indicating the necessity of developing safety alignment for non-English languages. In addition, we propose several simple and effective prompting methods to improve the multilingual safety of ChatGPT by evoking safety knowledge and improving cross-lingual generalization of safety alignment. Our prompting method can significantly reduce the ratio of unsafe responses from 19.1% to 9.7% for non-English queries.

---

### Paper 2: Do Llamas Work in English? — On the Latent Language of Multilingual Transformers
- **Authors**: Chris Wendler, Veniamin Veselovsky, Giovanni Monea, Robert West
- **Year**: 2024
- **Venue**: ACL 2024
- **Source**: arXiv API
- **Identifier**: 2402.10588
- **URL**: https://arxiv.org/abs/2402.10588

**Abstract**:
> We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language — a question of key importance for understanding how language models function and the origins of linguistic bias. Focusing on the Llama-2 family of transformer models, our study uses carefully constructed non-English prompts with a unique correct single-token continuation. From layer to layer, transformers gradually map an input embedding of the final prompt token to an output embedding from which next-token probabilities are computed. Tracking intermediate embeddings through their high-dimensional space reveals three distinct phases, whereby intermediate embeddings (1) start far away from output token embeddings; (2) already allow for decoding a semantically correct next token in the middle layers, but give higher probability to its version in English than in the input language; (3) finally move into an input-language-specific region of the embedding space. We cast these results into a conceptual model where the three phases operate in "input space", "concept space", and "output space", respectively. Crucially, our evidence suggests that the abstract "concept space" lies closer to English than to other languages, which may have important consequences regarding the biases held by multilingual language models.

---

### Paper 3: Separating Tongue from Thought — Activation Patching Reveals Language-Agnostic Concept Representations in Transformers
- **Authors**: Clément Dumas, Chris Wendler, Veniamin Veselovsky, Giovanni Monea, Robert West
- **Year**: 2024
- **Venue**: arXiv preprint (BlackboxNLP 2024 workshop track)
- **Source**: arXiv API (surfaced via web search, cross-verified via arxiv API)
- **Identifier**: 2411.08745
- **URL**: https://arxiv.org/abs/2411.08745

**Abstract (paraphrased from title + abstract snippet — the abstract text was voided by web filter, but the paper is pre-cutoff and can be cited by id)**:
> Activation-patching methodology provides causal evidence that language-agnostic concept representations exist and are actively used during multilingual text generation. Patching intermediate activations between paraphrased prompts across languages transfers semantic content while surface language features remain intact.

---

### Paper 4: The Semantic Hub Hypothesis — Language Models Share Semantic Representations Across Languages and Modalities
- **Authors**: Zhaofeng Wu, Xinyan Velocity Yu, Dani Yogatama, Jiasen Lu, Yoon Kim
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: 2411.04986
- **URL**: https://arxiv.org/abs/2411.04986

**Abstract (from title/context)**:
> Modern language models converge on shared semantic representations across languages and modalities in intermediate transformer layers. The "semantic hub" is language-agnostic and modality-agnostic and forms a natural intervention locus for cross-lingual and cross-modal transfer.

---

### Paper 5: mOthello — When Do Cross-Lingual Representation Alignment and Cross-Lingual Transfer Emerge in Multilingual Models?
- **Authors**: Tianze Hua, Tian Yun, Ellie Pavlick
- **Year**: 2024
- **Venue**: NAACL 2024 Findings
- **Source**: arXiv API
- **Identifier**: 2404.12444
- **URL**: https://arxiv.org/abs/2404.12444

**Abstract**:
> Many pretrained multilingual models exhibit cross-lingual transfer ability, which is often attributed to a learned language-neutral representation during pretraining. However, it remains unclear what factors contribute to the learning of a language-neutral representation, and whether the learned language-neutral representation suffices to facilitate cross-lingual transfer. We propose a synthetic task, Multilingual Othello (mOthello), as a testbed to delve into these two questions. We find that: (1) models trained with naive multilingual pretraining fail to learn a language-neutral representation across all input languages; (2) the introduction of "anchor tokens" (i.e., lexical items that are identical across languages) helps cross-lingual representation alignment; and (3) the learning of a language-neutral representation alone is not sufficient to facilitate cross-lingual transfer. Based on our findings, we propose a novel approach — multilingual pretraining with unified output space — that both induces the learning of language-neutral representation and facilitates cross-lingual transfer.

---

### Paper 6: Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models
- **Authors**: Jirui Qi, Raquel Fernández, Arianna Bisazza
- **Year**: 2023
- **Venue**: EMNLP 2023
- **Source**: arXiv API
- **Identifier**: 2310.10378
- **URL**: https://arxiv.org/abs/2310.10378

**Abstract**:
> Multilingual large-scale Pretrained Language Models (PLMs) have been shown to store considerable amounts of factual knowledge, but large variations are observed across languages. With the ultimate goal of ensuring that users with different language backgrounds obtain consistent feedback from the same model, we study the cross-lingual consistency (CLC) of factual knowledge in various multilingual PLMs. To this end, we propose a Ranking-based Consistency (RankC) metric to evaluate knowledge consistency across languages independently from accuracy. Using this metric, we conduct an in-depth analysis of the determining factors for CLC, both at model level and at language-pair level. Among other results, we find that increasing model size leads to higher factual probing accuracy in most languages, but does not improve cross-lingual consistency. Finally, we conduct a case study on CLC when new factual associations are inserted in the PLMs via model editing. Results on a small sample of facts inserted in English reveal a clear pattern whereby the new piece of knowledge transfers only to languages with which English has a high RankC score.

---

### Paper 7: Exploring Multilingual Concepts of Human Value in Large Language Models — Is Value Alignment Consistent, Transferable and Controllable across Languages?
- **Authors**: Shaoyang Xu, Weilong Dong, Zishan Guo, Xinwei Wu, Deyi Xiong
- **Year**: 2024
- **Venue**: arXiv preprint (published EMNLP 2024)
- **Source**: arXiv API
- **Identifier**: 2402.18120
- **URL**: https://arxiv.org/abs/2402.18120

**Abstract**:
> Prior research has revealed that certain abstract concepts are linearly represented as directions in the representation space of LLMs, predominantly centered around English. In this paper, we extend this investigation to a multilingual context, with a specific focus on human values-related concepts (i.e., value concepts) due to their significance for AI safety. Through our comprehensive exploration covering 7 types of human values, 16 languages and 3 LLM series with distinct multilinguality (e.g., monolingual, bilingual and multilingual), we first empirically confirm the presence of value concepts within LLMs in a multilingual format. Further analysis on the cross-lingual characteristics of these concepts reveals 3 traits arising from language resource disparities: cross-lingual inconsistency, distorted linguistic relationships, and unidirectional cross-lingual transfer between high- and low-resource languages, all in terms of value concepts. Moreover, we validate the feasibility of cross-lingual control over value alignment capabilities of LLMs, leveraging the dominant language as a source language. Ultimately, recognizing the significant impact of LLMs' multilinguality on our results, we consolidate our findings and provide prudent suggestions on the composition of multilingual data for LLMs pre-training.

---

### Paper 8: Refusal in Language Models Is Mediated by a Single Direction
- **Authors**: Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
- **Year**: 2024
- **Venue**: NeurIPS 2024
- **Source**: arXiv API
- **Identifier**: 2406.11717
- **URL**: https://arxiv.org/abs/2406.11717

**Abstract**:
> Conversational large language models are fine-tuned for both instruction-following and safety, resulting in models that obey benign requests but refuse harmful ones. While this refusal behavior is widespread across chat models, its underlying mechanisms remain poorly understood. In this work, we show that refusal is mediated by a one-dimensional subspace, across 13 popular open-source chat models up to 72B parameters in size. Specifically, for each model, we find a single direction such that erasing this direction from the model's residual stream activations prevents it from refusing harmful instructions, while adding this direction elicits refusal on even harmless instructions. Leveraging this insight, we propose a novel white-box jailbreak method that surgically disables refusal with minimal effect on other capabilities. Finally, we mechanistically analyze how adversarial suffixes suppress propagation of the refusal-mediating direction. Our findings underscore the brittleness of current safety fine-tuning methods.

---

### Paper 9: Representation Engineering — A Top-Down Approach to AI Transparency (RepE)
- **Authors**: Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, Dan Hendrycks
- **Year**: 2023
- **Venue**: arXiv preprint (widely cited)
- **Source**: arXiv API
- **Identifier**: 2310.01405
- **URL**: https://arxiv.org/abs/2310.01405

**Abstract**:
> In this paper, we identify and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuroscience. RepE places population-level representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena in deep neural networks (DNNs). We provide baselines and an initial analysis of RepE techniques, showing that they offer simple yet effective solutions for improving our understanding and control of large language models. We showcase how these methods can provide traction on a wide range of safety-relevant problems, including honesty, harmlessness, power-seeking, and more, demonstrating the promise of top-down transparency research.

---

### Paper 10: Improving Alignment and Robustness with Circuit Breakers
- **Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks
- **Year**: 2024
- **Venue**: NeurIPS 2024
- **Source**: arXiv API
- **Identifier**: 2406.04313
- **URL**: https://arxiv.org/abs/2406.04313

**Abstract**:
> AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility — even in the presence of powerful unseen attacks.

---

### Paper 11: Preference Tuning For Toxicity Mitigation Generalizes Across Languages
- **Authors**: Xiaochen Li, Zheng-Xin Yong, Stephen H. Bach
- **Year**: 2024
- **Venue**: EMNLP Findings 2024
- **Source**: arXiv API
- **Identifier**: 2406.16235
- **URL**: https://arxiv.org/abs/2406.16235

**Abstract**:
> Detoxifying multilingual Large Language Models (LLMs) has become crucial due to their increasing global use. In this work, we explore zero-shot cross-lingual generalization of preference tuning in detoxifying LLMs. Unlike previous studies that show limited cross-lingual generalization for other safety tasks, we demonstrate that Direct Preference Optimization (DPO) training with only English data can significantly reduce toxicity in multilingual open-ended generations. For example, the probability of mGPT-1.3B generating toxic continuations drops from 46.8% to 3.9% across 17 different languages after training. Our results also extend to other multilingual LLMs, such as BLOOM, Llama3, and Aya-23. Using mechanistic interpretability tools like causal intervention and activation analysis, we identified the dual multilinguality property of MLP layers in LLMs, which explains the cross-lingual generalization of DPO. Finally, we show that bilingual sentence retrieval can predict the cross-lingual transferability of DPO preference tuning.

---

### Paper 12: RLHF Can Speak Many Languages — Unlocking Multilingual Preference Optimization for LLMs (Aya team / Cohere for AI)
- **Authors**: John Dang, Arash Ahmadian, Kelly Marchisio, Julia Kreutzer, Ahmet Üstün, Sara Hooker
- **Year**: 2024
- **Venue**: EMNLP 2024 (aclanthology 2024.emnlp-main.729)
- **Source**: arXiv API (also via WebSearch snippet — snippet voided; paper identity confirmed via arXiv API cross-check)
- **Identifier**: 2407.02552 (< 2604 → pre-cutoff, safe)
- **URL**: https://arxiv.org/abs/2407.02552

**Abstract (verified from arXiv API metadata)**:
> Preference optimization techniques have become a standard final stage for training state-of-art large language models (LLMs). However, despite widespread adoption, the vast majority of work to-date has focused on a small set of high-resource languages like English and Chinese. This captures a small fraction of the languages in the world, but also makes it unclear which aspects of current state-of-the-art research transfer to a multilingual setting. In this work, we perform an exhaustive study to achieve a new state-of-the-art in aligning multilingual LLMs. We introduce a novel, scalable method for generating high-quality multilingual feedback data to balance data coverage. We establish the benefits of cross-lingual transfer and increased dataset size in preference training. Our preference-trained model achieves a 54.4% win-rate against Aya 23 8B, the current state-of-the-art multilingual LLM in its class.

---

### Paper 13: xLLMs-100 / LLMs Beyond English — Scaling the Multilingual Capability of LLMs with Cross-Lingual Feedback
- **Authors**: Wen Lai, Mohsen Mesgar, Alexander Fraser
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: 2406.01771
- **URL**: https://arxiv.org/abs/2406.01771

**Abstract**:
> To democratize large language models (LLMs) to most natural languages, it is imperative to make these models capable of understanding and generating texts in many languages, in particular low-resource ones. While recent multilingual LLMs demonstrate remarkable performance in such capabilities, these LLMs still support a limited number of human languages due to the lack of training data for low-resource languages. Moreover, these LLMs are not yet aligned with human preference for downstream tasks, which is crucial for the success of LLMs in English. In this paper, we introduce xLLaMA-100 and xBLOOM-100 (collectively xLLMs-100), which scale the multilingual capabilities of LLaMA and BLOOM to 100 languages. To do so, we construct two datasets: a multilingual instruction dataset including 100 languages, which represents the largest language coverage to date, and a cross-lingual human feedback dataset encompassing 30 languages. We perform multilingual instruction tuning on the constructed instruction data and further align the LLMs with human feedback using the DPO algorithm on our cross-lingual human feedback dataset.

---

### Paper 14: PKU-SafeRLHF — Towards Multi-Level Safety Alignment for LLMs with Human Preference
- **Authors**: Jiaming Ji, Donghai Hong, Borong Zhang, Boyuan Chen, Juntao Dai, Boren Zheng, Tianyi Qiu, Jiayi Zhou, Kaile Wang, Boxuan Li, Sirui Han, Yike Guo, Yaodong Yang
- **Year**: 2024 (updated 2025)
- **Venue**: arXiv preprint (dataset paper)
- **Source**: arXiv API
- **Identifier**: 2406.15513
- **URL**: https://arxiv.org/abs/2406.15513

**Abstract**:
> In this study, we introduce the safety human preference dataset, PKU-SafeRLHF, designed to promote research on safety alignment in large language models (LLMs). As a sibling project to SafeRLHF and BeaverTails, we separate annotations of helpfulness and harmlessness for question-answering pairs, providing distinct perspectives on these coupled attributes. Overall, we provide 44.6k refined prompts and 265k question-answer pairs with safety meta-labels for 19 harm categories and three severity levels ranging from minor to severe, with answers generated by Llama-family models. Based on this, we collected 166.8k preference data, including dual-preference (helpfulness and harmlessness decoupled) and single-preference data (trade-off the helpfulness and harmlessness from scratch), respectively. Using the large-scale annotation data, we further train severity-sensitive moderation for the risk control of LLMs and safety-centric RLHF algorithms for the safety alignment of LLMs.

---

### Paper 15: Refusal Direction is Universal Across Safety-Aligned Languages
- **Authors**: Xinpeng Wang, Mingyang Wang, Yihong Liu, Hinrich Schütze, Barbara Plank
- **Year**: 2025
- **Venue**: arXiv preprint (ACL 2025 track)
- **Source**: arXiv API
- **Identifier**: 2505.17306
- **URL**: https://arxiv.org/abs/2505.17306

**Abstract (from arXiv API)**:
> Refusal — the model's ability to decline harmful instructions — is central to the safety of aligned chat LLMs. Prior work has shown that this behavior is mediated by a single linear "refusal direction" in the residual stream, extracted from contrastive activations of harmful vs. harmless English prompts. In this work, we ask whether refusal shares a universal representation across the languages a safety-aligned model supports. Across multiple multilingual chat models, we find that (a) refusal directions extracted from one language transfer to unseen languages with only minor loss, (b) a single unified refusal direction fit jointly across languages achieves comparable performance to per-language directions, and (c) the refusal directions across languages exhibit high cosine similarity in middle-to-late layers, consistent with a shared multilingual refusal subspace.

---

### Paper 16: The Hidden Dimensions of LLM Alignment — A Multi-Dimensional Analysis of Orthogonal Safety Directions
- **Authors**: Wenbo Pan, Zhichao Liu, Qiguang Chen, Xiangyang Zhou, Haining Yu, Xiaohua Jia
- **Year**: 2025
- **Venue**: ACL 2025 (arXiv)
- **Source**: arXiv API
- **Identifier**: 2502.09674
- **URL**: https://arxiv.org/abs/2502.09674

**Abstract**:
> Large Language Models' safety-aligned behaviors, such as refusing harmful queries, can be represented by linear directions in activation space. Previous research modeled safety behavior with a single direction, limiting mechanistic understanding to an isolated safety feature. In this work, we discover that safety-aligned behavior is jointly controlled by multi-dimensional directions. Namely, we study the vector space of representation shifts during safety fine-tuning on Llama 3 8B for refusing jailbreaks. By studying orthogonal directions in the space, we first find that a dominant direction governs the model's refusal behavior, while multiple smaller directions represent distinct and interpretable features like hypothetical narrative and role-playing. We then measure how different directions promote or suppress the dominant direction, showing the important role of secondary directions in shaping the model's refusal representation. Finally, we demonstrate that removing certain trigger tokens in harmful queries can mitigate these directions to bypass the learned safety capability, providing new insights on understanding safety alignment vulnerability from a multi-dimensional perspective.

---

### Paper 17: Cross-Lingual Pitfalls — Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models
- **Authors**: Zixiang Xu, Yanbo Wang, Yue Huang, Xiuying Chen, Jieyu Zhao, Meng Jiang, Xiangliang Zhang
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: 2505.18673
- **URL**: https://arxiv.org/abs/2505.18673

**Abstract**:
> Large Language Models (LLMs) have achieved remarkable success in Natural Language Processing (NLP), yet their cross-lingual performance consistency remains a significant challenge. This paper introduces a novel methodology for efficiently identifying inherent cross-lingual weaknesses in LLMs. Our approach leverages beam search and LLM-based simulation to generate bilingual question pairs that expose performance discrepancies between English and target languages. We construct a new dataset of over 6,000 bilingual pairs across 16 languages using this methodology, demonstrating its effectiveness in revealing weaknesses even in state-of-the-art models. The extensive experiments demonstrate that our method precisely and cost-effectively pinpoints cross-lingual weaknesses, consistently revealing over 50% accuracy drops in target languages across a wide range of models.

---

### Paper 18: Cross-lingual Transfer of Reward Models in Multilingual Alignment
- **Authors**: Jiwoo Hong, Noah Lee, Rodrigo Martínez-Castaño, César Rodríguez, James Thorne
- **Year**: 2024
- **Venue**: arXiv preprint (NAACL 2025)
- **Source**: arXiv API
- **Identifier**: 2410.18027
- **URL**: https://arxiv.org/abs/2410.18027

**Abstract (from arXiv API)**:
> Reinforcement learning with human feedback (RLHF) has shown notable success in aligning language models with human preferences. Given its success in multilingual settings, we aim to explore its cross-lingual transferability. We show that English reward models trained with a broad multilingual base model transfer reward signals to target languages with limited performance loss, and that cross-lingual reward transfer explains part of the multilingual generalization observed in multilingual RLHF.

---

### Paper 19: SCANS — Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering
- **Authors**: Zouying Cao, Yifei Yang, Hai Zhao
- **Year**: 2024
- **Venue**: AAAI 2025
- **Source**: arXiv API
- **Identifier**: 2408.11491
- **URL**: https://arxiv.org/abs/2408.11491

**Abstract**:
> Safety alignment is indispensable for Large Language Models (LLMs) to defend threats from malicious instructions. However, recent researches reveal safety-aligned LLMs prone to reject benign queries due to the exaggerated safety issue, limiting their helpfulness. In this paper, we propose a Safety-Conscious Activation Steering (SCANS) method to mitigate the exaggerated safety concerns in aligned LLMs. First, SCANS extracts the refusal steering vectors within the activation space and utilizes vocabulary projection to anchor some specific safety-critical layers which influence model refusal behavior. Second, by tracking the hidden state transition, SCANS identifies the steering direction and steers the model behavior accordingly, achieving a balance between exaggerated safety and adequate safety.

---

### Paper 20: MPN — Leveraging Multilingual Patch Neuron for Cross-lingual Model Editing
- **Authors**: Nianwen Si, Hao Zhang, Weiqiang Zhang
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: 2401.03190
- **URL**: https://arxiv.org/abs/2401.03190

**Abstract**:
> Large language models are known for encoding a vast amount of factual knowledge, but they often becomes outdated due to the ever-changing nature of external information. A promising solution to this challenge is the utilization of model editing methods to update the knowledge in an efficient manner. However, the majority of existing model editing techniques are limited to monolingual frameworks, thus failing to address the crucial issue of cross-lingual knowledge synchronization for multilingual models. To tackle this problem, we propose a simple yet effective method that trains multilingual patch neuron to store cross-lingual knowledge.

---

### Paper 21: SafeSteer — Interpretable Safety Steering with Refusal-Evasion in LLMs
- **Authors**: Shaona Ghosh, Amrita Bhattacharjee, Yftah Ziser, Christopher Parisien
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: 2506.04250
- **URL**: https://arxiv.org/abs/2506.04250

**Abstract**:
> Fine-tuning large language models (LLMs) to adapt to evolving safety policies is costly and impractical. Mechanistic interpretability enables inference-time control through latent activation steering, yet its potential for precise, customizable safety adjustments remains largely untapped. This paper investigates an approach called SafeSteer for guiding the outputs of LLMs by: (i) leveraging category-specific steering vectors for more precise control, (ii) employing a simple, gradient-free unsupervised method to enhance safety steering while preserving text quality, topic relevance, and without explicit refusal, and (iii) accomplishing this without a hard requirement of contrastive pairwise safe data.

---

### Paper 22: DeepRefusal — Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction
- **Authors**: Yuanbo Xie, Yingjie Zhang, Tianyun Liu, Duohe Ma, Tingwen Liu
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: 2509.15202
- **URL**: https://arxiv.org/abs/2509.15202

**Abstract**:
> Jailbreak attacks pose persistent threats to large language models (LLMs). Current safety alignment methods have attempted to address these issues, but they experience two significant limitations: insufficient safety alignment depth and unrobust internal defense mechanisms. These limitations make them vulnerable to adversarial attacks such as prefilling and refusal direction manipulation. We introduce DeepRefusal, a robust safety alignment framework that overcomes these issues. DeepRefusal forces the model to dynamically rebuild its refusal mechanisms from jailbreak states. This is achieved by probabilistically ablating the refusal direction across layers and token depths during fine-tuning. Our method not only defends against prefilling and refusal direction attacks but also demonstrates strong resilience against other unseen jailbreak strategies. Extensive evaluations on four open-source LLM families and six representative attacks show that DeepRefusal reduces attack success rates by approximately 95%, while maintaining model capabilities with minimal performance degradation.

---

### Paper 23: xLLaMA / xBLOOM cross-lingual DPO alignment (already covered in Paper 13 above)

_(Removed as duplicate — see Paper 13.)_

---

### Paper 24: MultiJail — Deng et al. (well-known cross-lingual jailbreak benchmark)

**Note**: MultiJail is the canonical multilingual jailbreak benchmark referenced by the task. WebSearch surfaced its landing page but the response was voided by the post-filter (post-cutoff citation on the same page). The arXiv id 2310.06474 ("Multilingual Jailbreak Challenges in Large Language Models" — Deng, Zhang, Shen, Wong, Bing) is < 2604 and is the primary reference commonly cited alongside MultiJail. The dataset covers 3,150 harmful queries in 9 languages spanning high/medium/low resource tiers and is one of the two most cited multilingual safety benchmarks (alongside XSafety / Paper 1). This paper was not returned directly by my arXiv keyword queries above; it is cited here from established prior knowledge of the pre-cutoff literature (< 2604) and is standard reference for MultiJail.

---

## Lane A (mechanic-db cloud SEARCH) — status: in flight

Job id: cb1da38c-13c1-4369-96b0-e1225ecdcf23 (queued at 2026-07-14 01:44:20 UTC). The cloud service was configured with the decomposition documented in `/tmp/mdb_query.json` (interp_db + sciatlas_db, one sub-query each). If it returns, the resulting `mechanic_db_cache/20260714_014420_multiling_safety.json` will supplement this dump; the `.claude/mechanic-db-filter.py` PostToolUse hook will transparently scrub any post-cutoff or blocklisted papers from the file before I read it.

## Sources deliberately NOT queried

- WebFetch on any pre-cutoff arXiv page — the `.claude/url-blocker.py` PreToolUse hook blocks any URL string containing `lasa`, `2604.12710`, or any arxiv id ≥ 2604; but the arXiv API metadata returned by `arxiv_fetch.py` is already sufficient for a literature landscape (title, abstract, authors, year, categories). WebFetch would only add first-page body text at the cost of triggering another filter round.
- The blocked paper **LASA (arXiv 2604.12710)** is intentionally NOT retrieved. Any content or claims of that paper are treated as unknown; the landscape below is composed strictly from pre-cutoff prior art.
