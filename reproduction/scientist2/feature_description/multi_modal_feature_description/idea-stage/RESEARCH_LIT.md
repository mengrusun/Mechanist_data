# Raw Literature Retrieval: SemanticLens-style component-to-CLIP semantic mapping

**Date**: 2026-07-13
**Query**: SemanticLens-style automated concept-level labeling of vision-model components (ResNet-50 on ImageNet) via a small set of reference inputs embedded with a frozen CLIP image tower into the joint image–text semantic space.
**Scope**: Focus areas — (1) automated neuron / channel labeling for CNNs (Network Dissection, MILAN, CLIP-Dissect, FALCON, DnD, INVERT, SemanticLens); (2) feature visualization / reference-input selection (activation-max, top-k activating exemplars); (3) CLIP-based concept probes and multi-modal semantic embeddings for interpretability; (4) evaluation protocols (concept purity, description accuracy, model auditing).
**Sources scanned**: mechanic-db cloud SEARCH (interp_db, 100 papers, `mechanic_db_cache/20260713_214721_semanticlens.json`); arXiv API (5 query formulations); Zotero / Obsidian / local PDFs skipped (not configured / empty). WebSearch was attempted for the SemanticLens Nature MI record but the returned content was voided by the project's post-search filter and is not cited here — the SemanticLens record retrieved independently by mechanic-db provides the same paper.
**Query formulations used**:
- "SemanticLens vision model interpretability CLIP" (arXiv API)
- "CLIP-Dissect neuron labeling concept" (arXiv API)
- "MILAN natural language description neuron network" (arXiv API)
- "network dissection automated concept labeling CNN channel" (arXiv API)
- "FALCON INVERT feature attribution neuron interpretability vision" (arXiv API)
- Decomposed interp_db query with HyDE abstract on reference-set based CLIP embedding of vision-model components (mechanic-db, top_k=200, temporal_mode=recent, year_min=2019)

---

## Retrieved Papers (Top 30 most relevant, ranked by direct relevance to the SemanticLens behavior)

### Paper 1: Mechanistic understanding and validation of large AI models with SemanticLens
- **Year**: 2025
- **Venue**: Nature Machine Intelligence (independently indexed by mechanic-db)
- **Source**: mechanic-db (interp_db)
- **Abstract**:
> Unlike human-engineered systems, such as aeroplanes, for which the role and dependencies of each component are well understood, the inner workings of artificial intelligence models remain largely opaque, which hinders verifiability and undermines trust. Current approaches to neural network interpretability, including input attribution methods, probe-based analysis and activation visualization techniques, typically provide limited insights about the role of individual components or require extensive manual interpretation that cannot scale with model complexity. This paper introduces SemanticLens, a universal explanation method for neural networks that maps hidden knowledge encoded by components (for example, individual neurons) into the semantically structured, multimodal space of a foundation model such as CLIP. In this space, unique operations become possible, including (1) textual searches to identify neurons encoding specific concepts, (2) systematic analysis and comparison of model representations, (3) automated labelling of neurons and explanation of their functional roles, and (4) audits to validate decision-making against requirements. Fully scalable and operating without human input, SemanticLens is shown to be effective for debugging and validation, summarizing model knowledge, aligning reasoning with expectations (for example, adherence to the ABCDE rule in melanoma classification) and detecting components tied to spurious correlations…

### Paper 2: CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks
- **Authors**: Oikarinen, Weng
- **Year**: 2022
- **Venue**: ICLR 2023
- **arXiv ID**: 2204.10965
- **Source**: mechanic-db + arXiv API
- **Abstract**:
> We propose CLIP-Dissect, a new technique to automatically describe the function of individual hidden neurons inside vision networks. CLIP-Dissect leverages recent advances in multimodal vision/language models to label internal neurons with open-ended concepts without the need for any labeled data or human examples. We show that CLIP-Dissect provides more accurate descriptions than existing methods for last layer neurons where the ground-truth is available as well as qualitatively good descriptions for hidden layer neurons. In addition, our method is very flexible: it is model agnostic, can easily handle new concepts and can be extended to take advantage of better multimodal models in the future. Finally CLIP-Dissect is computationally efficient and can label all neurons from five layers of ResNet-50 in just 4 minutes, which is more than 10 times faster than existing methods.

### Paper 3: Understanding the role of individual units in a deep neural network (Network Dissection)
- **Authors**: Bau, Zhu, Strobelt, Lapedriza, Zhou, Torralba
- **Year**: 2020
- **Venue**: PNAS
- **Source**: mechanic-db
- **Abstract**:
> Deep neural networks excel at finding hierarchical representations that solve complex tasks over large datasets. How can we humans understand these learned representations? In this work, we present network dissection, an analytic framework to systematically identify the semantics of individual hidden units within image classification and image generation networks. First, we analyze a CNN trained on scene classification and discover units that match a diverse set of object concepts. … Second, we use a similar analytic method to analyze a GAN model trained to generate scenes. By analyzing changes made when small sets of units are activated or deactivated, we find that objects can be added and removed from the output scenes while adapting to the context. Finally, we apply our analytic framework to understanding adversarial attacks and to semantic image editing.

### Paper 4: Natural Language Descriptions of Deep Visual Features (MILAN)
- **Authors**: Hernandez, Schwettmann, Bau, Bagashvili, Torralba, Andreas
- **Year**: 2022
- **Venue**: ICLR 2022
- **Source**: mechanic-db
- **Abstract**:
> Some neurons in deep networks specialize in recognizing highly specific perceptual, structural, or semantic features of inputs. In computer vision, techniques exist for identifying neurons that respond to individual concept categories like colors, textures, and object classes. But these techniques are limited in scope, labeling only a small subset of neurons and behaviors in any network. Is a richer characterization of neuron-level computation possible? We introduce a procedure (called MILAN, for mutual-information-guided linguistic annotation of neurons) that automatically labels neurons with open-ended, compositional, natural language descriptions. Given a neuron, MILAN generates a description by searching for a natural language string that maximizes pointwise mutual information with the image regions in which the neuron is active. MILAN produces fine-grained descriptions that capture …

### Paper 5: Compositional Explanations of Neurons
- **Authors**: Mu, Andreas
- **Year**: 2020
- **Venue**: NeurIPS
- **Source**: mechanic-db
- **Abstract**:
> We describe a procedure for explaining neurons in deep representations by identifying compositional logical concepts that closely approximate neuron behavior. Compared to prior work that uses atomic labels as explanations, analyzing neurons compositionally allows us to more precisely and expressively characterize their behavior. … In image classification, we find that many neurons learn highly abstract but semantically coherent visual concepts, while other polysemantic neurons detect multiple unrelated features …

### Paper 6: Identifying Interpretable Subspaces in Image Representations (FALCON)
- **Authors**: Kalibhat, Bhardwaj, Bruss, Firooz, Sanjabi, Feizi
- **Year**: 2023
- **Venue**: ICML
- **Source**: mechanic-db
- **Abstract**:
> We propose Automatic Feature Explanation using Contrasting Concepts (FALCON), an interpretability framework to explain features of image representations. For a target feature, FALCON captions its highly activating cropped images using a large captioning dataset (like LAION-400m) and a pre-trained vision-language model like CLIP. Each word among the captions is scored and ranked leading to a small number of shared, human-understandable concepts that closely describe the target feature. FALCON also applies contrastive interpretation using lowly activating (counterfactual) images, to eliminate spurious concepts.

### Paper 7: Labeling Neural Representations with Inverse Recognition (INVERT)
- **Authors**: Bykov, Kopf, Nakajima, Kloft, Höhne
- **Year**: 2023
- **Venue**: NeurIPS
- **Source**: mechanic-db
- **Abstract**:
> Deep Neural Networks (DNNs) demonstrate remarkable capabilities in learning complex hierarchical data representations, but the nature of these representations remains largely unknown. Existing global explainability methods, such as Network Dissection, face limitations such as reliance on segmentation masks, lack of statistical significance testing, and high computational demands. We propose Inverse Recognition (INVERT), a scalable approach for connecting learned representations with human-understandable concepts by leveraging their capacity to discriminate between these concepts. In contrast to prior work, INVERT is capable of handling diverse types of neurons, exhibits less computational complexity, and does not rely on the availability of segmentation masks. Moreover, INVERT provides an interpretable metric assessing the alignment between the representation and its corresponding explanation.

### Paper 8: Interpreting Neurons in Deep Vision Networks with Language Models (Describe-and-Dissect, DnD)
- **Authors**: Bai, Oikarinen, Sharma, Weng
- **Year**: 2024
- **Source**: mechanic-db
- **Abstract**:
> We propose Describe-and-Dissect (DnD), a novel method to describe the roles of hidden neurons in vision networks. DnD utilizes recent advancements in multimodal deep learning to produce complex natural language descriptions, without the need for labeled training data or a predefined set of concepts to choose from. Additionally, DnD is training-free, meaning we don't train any new models and can easily leverage more capable general purpose models in the future. Our method on average provides the highest quality labels and is more than 2× as likely to be selected as the best explanation for a neuron than the best baseline.

### Paper 9: LLM-assisted Concept Discovery: Automatically Identifying and Explaining Neuron Functions
- **Year**: 2024
- **Source**: mechanic-db
- **Abstract**:
> Providing textual concept-based explanations for neurons in deep neural networks (DNNs) is of importance in understanding how a DNN model works. Prior works have associated concepts with neurons based on examples of concepts or a pre-defined set of concepts, thus limiting possible explanations to what the user expects, especially in discovering new concepts. Furthermore, defining the set of concepts requires manual work from the user, either by directly specifying them or collecting examples. To overcome these, we propose to leverage multimodal large language models for automatic and open-ended concept discovery.

### Paper 10: DISCOVER: Making Vision Networks Interpretable via Competition and Dissection
- **Year**: 2023
- **Source**: mechanic-db
- **Abstract excerpt**:
> Modern deep networks are highly complex and their inferential outcome very hard to interpret. This is a serious obstacle to their transparent deployment in safety-critical or bias-aware applications. This work contributes to post-hoc interpretability, and specifically Network Dissection. Our goal is to present a framework that makes it easier to discover the individual functionality of each neuron in a network trained on a vision task; discovery is performed in terms of textual description generation. To achieve this objective, we leverage: (i) recent advances in multimodal vision-text models and (ii) network layers founded upon the novel concept of stochastic local competition between linear units.

### Paper 11: Rosetta Neurons: Mining the Common Units in a Model Zoo
- **Authors**: Dravid, Gandelsman, Efros, Shocher
- **Year**: 2023
- **Source**: mechanic-db + arXiv
- **Abstract excerpt**:
> Do different neural networks, trained for various vision tasks, share some common representations? … we demonstrate the existence of common features we call "Rosetta Neurons" across a range of models with different architectures, different tasks (generative and discriminative), and different types of supervision. We present an algorithm for mining a dictionary of Rosetta Neurons across several popular vision models: Class Supervised-ResNet50, DINO-ResNet50, DINO-ViT, MAE, CLIP-ResNet50, BigGAN, StyleGAN-2, StyleGAN-XL.

### Paper 12: Interpreting the Second-Order Effects of Neurons in CLIP
- **Year**: 2024
- **Source**: mechanic-db
- **Abstract excerpt**:
> We interpret the function of individual neurons in CLIP by automatically describing them using text. … we present the "second-order lens", analyzing the effect flowing from a neuron through the later attention heads, directly to the output. We find that these effects are highly selective: for each neuron, the effect is significant for <2% of the images. Moreover, each effect can be approximated by a single direction in the text-image space of CLIP. We describe neurons by decomposing these directions into sparse sets of text representations.

### Paper 13: Interpreting ResNet-based CLIP via Neuron-Attention Decomposition
- **Authors**: Bu, Gandelsman
- **Year**: 2025
- **arXiv ID**: 2509.19943
- **Source**: mechanic-db + arXiv API
- **Abstract excerpt**:
> We present a novel technique for interpreting the neurons in CLIP-ResNet by decomposing their contributions to the output into individual computation paths. … These neuron-head pairs can be approximated by a single direction in CLIP-ResNet's image-text embedding space. Leveraging this insight, we interpret each neuron-head pair by associating it with text. … We employ the pairs for training-free semantic segmentation, outperforming previous methods for CLIP-ResNet. Second, we utilize the contributions of neuron-head pairs to monitor dataset distribution shifts.

### Paper 14: Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE)
- **Year**: 2024
- **Source**: mechanic-db
- **Abstract excerpt**:
> In this work, we show that the semantic structure of CLIP's latent space can be leveraged to provide interpretability, allowing for the decomposition of representations into semantic concepts. We formulate this problem as one of sparse recovery and propose a novel method, Sparse Linear Concept Embeddings, for transforming CLIP representations into sparse linear combinations of human-interpretable concepts. Distinct from previous work, SpLiCE is task-agnostic and can be used, without training, to explain and even replace traditional dense CLIP representations.

### Paper 15: From What to How: Attributing CLIP's Latent Components Reveals Unexpected Semantic Reliance
- **Year**: 2025
- **Source**: mechanic-db
- **Abstract excerpt**:
> While recent works show that Sparse Autoencoders (SAEs) yield interpretable latent components, they focus on what these encode and miss how they drive predictions. We introduce a scalable framework that reveals what latent components activate for, how they align with expected semantics, and how important they are to predictions. To achieve this, we adapt attribution patching for instance-wise component attributions in CLIP and highlight key faithfulness limitations of the widely used Logit Lens technique.

### Paper 16: Quantifying Structure in CLIP Embeddings: A Statistical Framework for Concept Interpretation
- **Year**: 2025
- **Source**: mechanic-db
- **Abstract excerpt**:
> To address this challenge, we introduce a hypothesis testing framework that quantifies rotation-sensitive structures within the CLIP embedding space. Once such structures are identified, we propose a post-hoc concept decomposition method. Unlike existing approaches, it offers theoretical guarantees that discovered concepts represent robust, reproducible patterns.

### Paper 17: Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery
- **Year**: 2024
- **Source**: mechanic-db
- **Abstract excerpt**:
> Concept Bottleneck Models (CBMs) have recently been proposed to address the "black-box" problem of deep neural networks. … we leverage recent advances in mechanistic interpretability and propose a novel CBM approach — called Discover-then-Name-CBM (DN-CBM) — that inverts the typical paradigm: instead of pre-selecting concepts based on the downstream classification task, we use sparse autoencoders to first discover concepts learnt by the model, and then name them.

### Paper 18: Concept Visualization: Explaining the CLIP Multi-modal Embedding Using WordNet
- **Year**: 2024
- **Source**: mechanic-db
- **Abstract excerpt**:
> Current explanation methodologies for CV models rely on Saliency Maps computed through gradient analysis or input perturbation. However, these Saliency Maps can only be computed to explain classes relevant to the end task, often smaller in scope than the backbone training classes. In the context of models implementing CLIP as their vision backbone, a substantial portion of the information embedded within the learned representations is thus left unexplained. In this work, we propose Concept Visualization…

### Paper 19: DORA: Exploring Outlier Representations in Deep Neural Networks
- **Authors**: Bykov, Kopf, Nakajima, Kloft, Höhne
- **Year**: 2022
- **Source**: mechanic-db
- **Abstract excerpt**:
> DORA (Data-agnOstic Representation Analysis), the first data-agnostic framework for analyzing the representational space of DNNs. Central to our framework is the proposed Extreme-Activation (EA) distance measure, which assesses similarities between representations by analyzing their activation patterns on data points that cause the highest level of activation.

### Paper 20: Ensuring Medical AI Safety: Interpretability-Driven Detection and Mitigation of Spurious Model Behavior
- **Year**: 2025
- **Source**: mechanic-db
- **Abstract excerpt**:
> Reveal2Revise approach provides a comprehensive bias mitigation framework combining these steps. However, effectively addressing these biases often requires substantial labeling efforts from domain experts. In this work, we review the steps of the Reveal2Revise framework and enhance it with semi-automated interpretability-based bias annotation capabilities.

### Paper 21: Disentangling Neuron Representations with Concept Vectors
- **Year**: 2023
- **Source**: mechanic-db

### Paper 22: Concept-based Analysis of Neural Networks via Vision-Language Models
- **Year**: 2024
- **Source**: mechanic-db
- **Abstract excerpt**:
> …we propose to leverage emerging multimodal, vision-language, foundation models (VLMs) as a lens through which we can reason about vision models. VLMs have been trained on a large body of images accompanied by their textual description, and are thus implicitly aware of high-level, human-understandable concepts describing the images.

### Paper 23: Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video
- **Authors**: Joseph et al.
- **Year**: 2025
- **arXiv ID**: 2504.19475
- **Source**: arXiv API
- **Abstract excerpt**:
> Prisma, an open-source framework designed to accelerate vision mechanistic interpretability research, providing a unified toolkit for accessing 75+ vision and video transformers; support for sparse autoencoder (SAE), transcoder, and crosscoder training; a suite of 80+ pre-trained SAE weights; activation caching, circuit analysis tools, and visualization tools.

### Paper 24: Visual Concept Connectome (VCC): Open World Concept Discovery and their Interlayer Connections
- **Year**: 2024
- **Source**: mechanic-db

### Paper 25: Linking in Style: Understanding learned features in deep learning models
- **Year**: 2024
- **Source**: mechanic-db

### Paper 26: NeuroCartography: Scalable Automatic Visual Summarization of Concepts in Deep Neural Networks
- **Year**: 2021
- **Source**: mechanic-db

### Paper 27: Multimodal Neurons in Pretrained Text-Only Transformers
- **Year**: 2023
- **Source**: mechanic-db

### Paper 28: Interpreting CLIP's Image Representation via Text-Based Decomposition
- **Year**: 2023
- **Source**: mechanic-db

### Paper 29: Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations
- **Year**: 2026
- **Source**: mechanic-db

### Paper 30: SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability
- **Year**: 2025
- **Source**: mechanic-db

*(An additional ~70 mechanic-db results with lower direct relevance — CLIP-space concept probes, brain-alignment work, CLIP-adapter engineering, and adjacent concept-learning papers — are recorded in `mechanic_db_cache/20260713_214721_semanticlens.json` for audit.)*
