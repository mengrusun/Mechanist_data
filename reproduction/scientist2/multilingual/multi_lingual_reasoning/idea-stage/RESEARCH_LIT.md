# Raw Literature Retrieval: Disentangling Language and Reasoning in LLM Internal Representations

**Date**: 2026-07-14
**Query**: multilingual LLM interpretability — language-specific vs language-agnostic subspaces, activation steering / subspace ablation, cross-lingual reasoning transfer, training-free intervention on MGSM
**Sources scanned**: web (multiple query formulations); arXiv API (blocked at hook layer for post-2505 IDs; pre-cutoff IDs surfaced via web); mechanic-db skipped (MCP `search_papers` tool not exposed in this session); Zotero/Obsidian/local — not configured.
**Retrieval-side policy note**: This project's post-search filter blocks any arXiv paper with YYMM ≥ 2505 and the specific reproduction target (arxiv 2505.15257 + associated GitHub repo). All papers below are pre-cutoff or non-arXiv sources — the reproduction target paper was NOT read, faithfully preserving the given-behavior contract (behavior is taken verbatim from `task.md`, not the source paper).
**Query formulations used**:
- "language-specific neurons multilingual LLM LAPE probing interpretability"
- "multilingual BERT XLM-R language-agnostic representation subspace probing analysis"
- "Language-Specific Neurons Multilingual LLM ACL 2024 Tang Luo LAPE FFN"
- "LangBridge multilingual reasoning without multilingual supervision"
- "MEXA multilingual evaluation cross-lingual alignment English pivot LLM"
- "How do LLMs handle multilingualism Zhao NeurIPS 2024 English pivot latent"
- "representation engineering top-down transparency LLM steering direction linear concept probe"
- "multilingual mathematical reasoning MetaMath instruction tuning MathOctopus"
- "sparse autoencoders multilingual features cross-lingual concepts LLM"
- "Lens multilingual enhancement large language models language-agnostic subspace"
- "Discovering low-rank subspaces language-agnostic multilingual representations XLM-R Xie SVD"
- "GlotLID language identification low-resource languages"
- "activation patching cross-lingual multilingual causal intervention transformer LLM"
- "Qwen 3 reasoning multilingual chain-of-thought DeepSeek-R1 distill low-resource languages"

---

## Retrieved Papers

### Paper 1: Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models
- **Authors**: Tianyi Tang, Wenyang Luo, Haoyang Huang, Dongdong Zhang, Xiaolei Wang, Xin Zhao, Furu Wei, Ji-Rong Wen
- **Year**: 2024
- **Venue**: ACL 2024 (Long)
- **Source**: WebSearch (ACL Anthology + Microsoft Research)
- **Identifier**: aclanthology.org/2024.acl-long.309
- **URL**: https://aclanthology.org/2024.acl-long.309/

**Abstract / key content**:
Proposes **Language Activation Probability Entropy (LAPE)** — a metric that identifies language-specific neurons by measuring entropy of neuron activations across languages. Applied to LLaMA-2, BLOOM, Mistral. Finding: language-specific neurons are concentrated in the **top and bottom layers** (not middle) of FFN sublayers. Deactivating a language's neurons impairs generation in that language; activating them can steer output toward a target language. Landmark neuron-level evidence that "language" and "content" are separately encoded in multilingual LLMs.

---

### Paper 2: How do Large Language Models Handle Multilingualism?
- **Authors**: Yiran Zhao, Wenxuan Zhang, Guizhen Chen, Kenji Kawaguchi, Lidong Bing
- **Year**: 2024
- **Venue**: NeurIPS 2024
- **Source**: WebSearch (NeurIPS proceedings)
- **Identifier**: proceedings.neurips.cc/paper_files/paper/2024/file/1bd359b32ab8b2a6bbafa1ed2856cf40-Paper-Conference.pdf

**Abstract / key content**:
Proposes a **three-phase framework** for LLM multilingual processing: (1) early layers understand the query and translate to English internally, (2) middle layers reason and retrieve knowledge in an English-flavored space via self-attention + FFN, (3) upper layers align back to the query's original language. Empirical support via layer-wise probing across Llama-2 family. Introduces **PLND (Parallel Language-specific Neuron Detection)** to identify language-specific neurons layer-wise. Directly grounds the claim that middle layers hold a shared/English-pivot representation.

---

### Paper 3: Do Llamas Work in English? On the Latent Language of Multilingual Transformers
- **Authors**: Chris Wendler, Veniamin Veselovsky, Giovanni Monea, Robert West
- **Year**: 2024
- **Venue**: ACL 2024 (Long)
- **Source**: WebSearch (ACL Anthology; NeurIPS-cited by MEXA)
- **Identifier**: 2024.acl-long.820

**Abstract / key content**:
Uses the **logit lens** on Llama-2 to show that when translating from a non-English source to a non-English target, the model's **intermediate-layer logits favor the English translation of the target concept**, and only in upper layers does the output shift to the actual target language. Formalizes the "English pivot" phenomenon in a decoder-only LLM. Strong causal-flavored evidence that language identity is (a) separable and (b) written back mostly at the top layers — matching the task.md hypothesis that upper layers handle language fidelity while middle layers do reasoning.

---

### Paper 4: MEXA: Multilingual Evaluation of English-Centric LLMs via Cross-Lingual Alignment
- **Authors**: Amir Hossein Kargaran, Ali Modarressi, Nafiseh Nikeghbal, Jana Diesner, François Yvon, Hinrich Schütze
- **Year**: 2024 (Findings ACL 2025)
- **Venue**: Findings of ACL 2025 (arXiv preprint pre-cutoff: 2410.05873)
- **Source**: WebSearch (OpenReview + ACL Anthology)
- **Identifier**: aclanthology.org/2025.findings-acl.1385/

**Abstract / key content**:
Introduces the **MEXA alignment score**: uses parallel sentences (FLORES-200, Bible) and checks how often the cosine similarity of a matched English/non-English pair exceeds that of a random pair. Uses **mid-layer** embeddings — the layers where English acts as a "pivot". Achieves Pearson r = 0.90 with downstream multilingual accuracy (Belebele, m-MMLU, m-ARC) across Llama / Gemma / Mistral / OLMo families. Establishes a cheap, model-internal correlate of downstream multilingual performance, and confirms mid-layer English pivot at scale.

---

### Paper 5: LangBridge: Multilingual Reasoning Without Multilingual Supervision
- **Authors**: Dongkeun Yoon, Joel Jang, Sungdong Kim, Seungone Kim, Sheikh Shafayat, Minjoon Seo
- **Year**: 2024
- **Venue**: ACL 2024 (Long); arXiv 2401.10695
- **Source**: WebSearch (ACL Anthology + KAIST + GitHub)
- **Identifier**: aclanthology.org/2024.acl-long.405

**Abstract / key content**:
A **zero-shot adaptation** approach that connects a multilingual encoder (mT5) to a strong reasoning-tuned decoder (MetaMath, Orca-2) via a small trainable projection, **without any multilingual supervision**. Achieves large gains on MGSM/MSVAMP low-resource languages while training only a bridge module. Key baseline for "training-free / low-training-cost multilingual reasoning" — task.md Claim 4 will need to compare against this line of work.

---

### Paper 6: MathOctopus — Breaking Language Barriers in Multilingual Mathematical Reasoning: Insights and Observations
- **Authors**: Nuo Chen, Zinan Zheng, Ning Wu, Ming Gong, Yangqiu Song, Dongmei Zhang, Jia Li
- **Year**: 2023–2024
- **Venue**: ACL 2024 (Findings); Microsoft/HKUST
- **Source**: WebSearch (GitHub + ResearchGate + ACL Anthology)
- **Identifier**: microsoft/MathOctopus GitHub; MGSM8KInstruct dataset

**Abstract / key content**:
Introduces **MGSM8KInstruct** — the first multilingual math SFT dataset (10 languages, translated GSM8K). Shows multilingual SFT lifts non-English accuracy dramatically and cross-lingually. Establishes the **multilingual SFT baseline** that task.md Claim 4 must match at "small fraction of the compute" — the yardstick for "training-free intervention matches or exceeds multilingual post-training".

---

### Paper 7: Lens: Rethinking Multilingual Enhancement for Large Language Models
- **Authors**: Weixiang Zhao, Yulin Hu, Jiahe Guo, Xingyu Sui, Tongtong Wu, Yang Deng, Yanyan Zhao, Bing Qin, Wanxiang Che, Ting Liu
- **Year**: 2024
- **Venue**: NeurIPS 2024 / OpenReview id=8kGonpsiHb
- **Source**: WebSearch (OpenReview + ResearchGate + emergentmind)
- **Identifier**: OpenReview 8kGonpsiHb

**Abstract / key content**:
**Closest published prior art to the task.md direction.** Explicitly decomposes hidden states from **top layers** of English-centric LLMs into two subspaces: a **language-agnostic subspace** (where the target language is pulled toward the central English pivot to inherit semantic representations) and a **language-specific subspace** (where target and central languages are pushed apart to preserve output identity). Delivers improved multilingual performance without sacrificing English capability, at far lower cost than multilingual post-training. **The task.md hypothesis extends the same subspace decomposition but (a) suppresses the language-specific subspace rather than only rebalancing it, (b) targets reasoning-tuned models (Qwen-3-Thinking, DeepSeek-R1-Distill), and (c) frames it as inference-time training-free intervention.** LENS itself does light contrastive training on the pivot, so the "no training" claim in task.md is a step further.

---

### Paper 8: Discovering Low-rank Subspaces for Language-agnostic Multilingual Representations (LSAR)
- **Authors**: Zhihui Xie, Handong Zhao, Tong Yu, Shuai Li
- **Year**: 2022
- **Venue**: EMNLP 2022
- **Source**: WebSearch (ACL Anthology + GitHub zhxieml/LSAR)
- **Identifier**: 2022.emnlp-main.379

**Abstract / key content**:
Foundational **unsupervised SVD-based method** to identify a low-rank subspace in multilingual encoder embeddings (mBERT, XLM-R) that captures **language-specific / non-semantic** information; **projecting to its null space** improves cross-lingual retrieval and transfer *without any fine-tuning*. Directly foreshadows the mechanism of "identify language-specific subspace from a small probe set → project it out at inference" that task.md Claim 1 & 2 will test on decoder-only reasoning LLMs. Any experiment plan should adopt the SVD-null-space projection as a strong baseline / candidate mechanism family.

---

### Paper 9: The Geometry of Multilingual Language Model Representations
- **Authors**: Tyler A. Chang, Zhuowen Tu, Benjamin K. Bergen
- **Year**: 2022
- **Venue**: EMNLP 2022
- **Source**: WebSearch (arxiv abstract pre-cutoff 2205.10964; ar5iv)
- **Identifier**: 2205.10964

**Abstract / key content**:
Identifies **affine language subspaces** in XLM-R via SVD over per-language contextualized representations across 88 languages. Shows subspaces have shared axes (language-neutral component) plus offsets along language-sensitive axes (encoding token vocabularies / scripts / morphology). Establishes that "language identity" is a **low-rank affine offset**, not a scattered high-dimensional direction — supports treating the language-specific subspace as low-dimensional and probe-identifiable (task.md Claim 1).

---

### Paper 10: Inducing Language-Agnostic Multilingual Representations
- **Authors**: Wei Zhao, Steffen Eger, Johannes Bjerva, Isabelle Augenstein
- **Year**: 2020
- **Venue**: *SEM 2020 / arXiv 2008.09112
- **Source**: WebSearch
- **Identifier**: 2008.09112

**Abstract / key content**:
Earlier work that pushes multilingual encoder representations toward a language-agnostic geometry via (a) linear transformations, (b) removing language-mean vectors, (c) adversarial training. Compares methods on XNLI and BUCC. Reinforces the linear structure of language identity in multilingual encoders and provides classical baselines (mean-centering, LDA, adversarial removal).

---

### Paper 11: An Isotropy Analysis in the Multilingual BERT Embedding Space
- **Authors**: Sara Rajaee, Mohammad Taher Pilehvar
- **Year**: 2021
- **Venue**: EACL / Findings 2022 (arXiv 2110.04504)
- **Source**: WebSearch
- **Identifier**: 2110.04504

**Abstract / key content**:
Shows that **mBERT** embedding spaces are highly anisotropic, dominated by a few directions related to language and frequency; removing these dominant directions improves cross-lingual retrieval and STS. Grounds the very early mechanistic case that a **small number of dominant directions** encode language identity — the same low-dim story mBERT era → decoder LLMs.

---

### Paper 12: LSAR / LENS competitive family — LangBridge repository & related zero-shot bridges
- **Authors**: (LangBridge team; KAIST kaistAI repo)
- **Year**: 2024
- **Venue**: GitHub kaistAI/LangBridge
- **Source**: WebSearch (GitHub)
- **Identifier**: github.com/kaistAI/LangBridge

**Abstract / key content**:
Public code/checkpoints supporting a **soft-prompt / adapter bridge** from a multilingual encoder to a reasoning LLM. Only trains a projection; no multilingual supervision. Useful engineering reference and **strong baseline** for MGSM Claim-4 comparison.

---

### Paper 13: GlotLID: Language Identification for Low-Resource Languages
- **Authors**: Amir Hossein Kargaran, Ayyoob Imani, François Yvon, Hinrich Schütze
- **Year**: 2023
- **Venue**: Findings of EMNLP 2023
- **Source**: WebSearch (ACL Anthology + HuggingFace)
- **Identifier**: aclanthology.org/2023.findings-emnlp.410

**Abstract / key content**:
FastText-based language identifier covering **≥ 1665 languages** (V1) / 2102 labels (V3), specifically designed to be reliable for low-resource languages including Bengali, Swahili, Telugu, Thai — all 4 mid/low-resource languages in task.md. Handles noisy corpora, macrolanguages, and closely related language pairs. **Task.md mandates GlotLID as the output-language fidelity metric**; this is the reference implementation & evaluation tool.

---

### Paper 14: Representation Engineering: A Top-Down Approach to AI Transparency
- **Authors**: Andy Zou, Long Phan, Sarah Chen, James Campbell, et al.
- **Year**: 2023 (arXiv 2310.01405)
- **Venue**: arXiv preprint (pre-cutoff)
- **Source**: WebSearch (awesome-representation-engineering repo + longjubai.github.io)
- **Identifier**: 2310.01405

**Abstract / key content**:
Defines the **representation reading + representation control** framework — extract concept directions from residual-stream contrast pairs, then add/subtract at inference to steer behavior. The methodological ancestor of the "subspace suppression" mechanism family that task.md targets. Provides the standard vocabulary (reading vectors, contrast vectors, steering, α scaling) and standard evaluation of specificity vs. off-target damage.

---

### Paper 15: Steering LLaMA 2 via Contrastive Activation Addition (CAA)
- **Authors**: Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
- **Year**: 2024
- **Venue**: ACL 2024
- **Source**: WebSearch
- **Identifier**: 2312.06681

**Abstract / key content**:
Concrete recipe: build a steering vector from mean(pos activations) − mean(neg activations) at a chosen layer, add α · v to the residual stream at inference. Established the standard **α-sweep** and **layer-sweep** protocol for concept steering. Directly transferable: replace pos/neg = "target language / English" contrast pairs, build a per-language mean-difference vector, subtract at inference for the same "language suppression" effect described in task.md Claim 2.

---

### Paper 16: Sharing Matters: Analysing Neurons Across Languages and Tasks in LLMs
- **Authors**: Weixuan Wang, Barry Haddow, Alexandra Birch, Wei Peng
- **Year**: 2024
- **Venue**: arXiv 2406.09265; ACL Findings
- **Source**: WebSearch
- **Identifier**: 2406.09265

**Abstract / key content**:
Fine-grained analysis of neuron overlap **across languages** and **across tasks** in Llama-family models. Distinguishes **language-only neurons**, **task-only neurons**, and **shared neurons**. Finds that **shared / task neurons concentrate in middle-upper layers** and language neurons in early / late layers — geometric grounding for the middle-layer intervention target and the "leave upper layers intact" prescription in task.md Claim 2.

---

### Paper 17: Inductive Linguistic Reasoning with Large Language Models
- **Authors**: (per web search result 2412.17819)
- **Year**: 2024
- **Venue**: arXiv 2412.17819 (pre-cutoff)
- **Source**: WebSearch
- **Identifier**: 2412.17819

**Abstract / key content**:
Analyzes LLM linguistic reasoning across families; complements MathOctopus by mapping cross-lingual performance gaps for non-math reasoning. Context for XWinograd (commonsense) and M-MMLU (knowledge) verify-stage extension.

---

### Paper 18: LinguaLIFT: Two-stage Instruction Tuning for Low-Resource Language Reasoning
- **Authors**: (per web search result 2412.12499)
- **Year**: 2024
- **Venue**: arXiv 2412.12499 (pre-cutoff)
- **Source**: WebSearch
- **Identifier**: 2412.12499

**Abstract / key content**:
Recent two-stage SFT pipeline for boosting low-resource-language reasoning. Represents the **post-training baseline family** (competitor for Claim 4).

---

### Paper 19: A Tree-of-Thoughts to Broaden Multi-step Reasoning across Languages (Cross-ToT)
- **Authors**: (per 2311.08097)
- **Year**: 2023
- **Venue**: arXiv 2311.08097 (pre-cutoff)
- **Source**: WebSearch
- **Identifier**: 2311.08097

**Abstract / key content**:
Prompting-only cross-lingual CoT strategy — different CoT branches in different languages, then converge. Prompt-side alternative to representation editing; useful contrast.

---

### Paper 20: Improving Instruction-Following in Language Models through Activation Steering
- **Authors**: (per 2410.12877)
- **Year**: 2024
- **Venue**: arXiv 2410.12877 (pre-cutoff)
- **Source**: WebSearch
- **Identifier**: 2410.12877

**Abstract / key content**:
Recent instance of activation-steering literature applied to instruction following; provides recipes for α-sweep, layer targeting, and reference specificity metrics that directly transfer.

---
