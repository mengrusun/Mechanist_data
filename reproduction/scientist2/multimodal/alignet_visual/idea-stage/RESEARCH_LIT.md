# Raw Literature Retrieval: Distilling human-like similarity structure into pretrained vision foundation models (DINOv2, ViT, contrastive image-text) using THINGS triplet judgments

**Date**: 2026-07-14
**Query**: Distilling human-like similarity structure from human triplet judgment datasets (THINGS) into pretrained vision foundation models (DINOv2 ViT-B, supervised ViT-L, contrastive image-text). Focus: surrogate teacher, hierarchical similarity, distillation losses, downstream utility + OOD robustness. General research space only — the specific underlying work being reproduced is off-limits per project policy (`.claude/forbidden-urls.txt`).

**Sources scanned**: arXiv API (base, ran); WebSearch (base, one query attempted and voided by post-search policy filter — see Notes); mechanic-db cloud SEARCH (`search_papers` MCP tool not available in this environment → **skipped**); Zotero MCP (not configured → skipped); Obsidian MCP (not configured → skipped); local paper library (`papers/`, `literature/` — neither present → skipped); prior research-wiki (`research-wiki/` absent → no banlist).

**Query formulations used** (arXiv API base source):
- `human alignment vision foundation models triplet similarity distillation`
- `THINGS dataset triplet odd-one-out representational similarity DNN`
- `human similarity judgments neural network representation alignment`
- `hierarchical concept alignment vision transformer distillation`
- `Muttenthaler human alignment neural network representations`
- `DINOv2 self-supervised representation learning`
- `SigLIP contrastive image text pretraining`
- `THINGS dataset natural object concepts psychology`
- `sparse positive similarity embedding VICE THINGS`
- `knowledge distillation vision transformer soft labels KL divergence student teacher`
- `representational similarity analysis brain deep neural network comparison`
- `brain-score neural predictivity vision model`

**Note on `mechanic-db`**: the `search_papers` MCP tool required by `/mechanic-db-search` is not registered in this Claude Code environment (`claude mcp list` returned only Google-suite servers). Per the skill's contract, `mechanic-db` is treated as `skipped: true` for this run and other base sources continue.

**Note on WebSearch**: one WebSearch query returned results that hit the project's `post-search-filter.py` policy (the reference paper being reproduced is on the forbidden list — arXiv-ID cutoff ≥ 2409, plus title-substring matches). That whole response has been voided as instructed by the reminder. Downstream synthesis does **not** cite, paraphrase, or follow-up on any material from that voided response. Retrieval otherwise proceeds from arXiv-API results with IDs below the cutoff (2211.x, 2306.x, 2010.x, 2004.x, 1909.x, 2201.x, etc.), all of which are pre-cutoff general prior-art and not the forbidden work.

---

## Retrieved Papers

### Paper 1: Human alignment of neural network representations
- **Authors**: Lukas Muttenthaler, Jonas Dippel, Lorenz Linhardt, Robert A. Vandermeulen, Simon Kornblith
- **Year**: 2022 (updated 2025-02-16)
- **Venue**: arXiv (ICLR 2023 track)
- **Source**: arXiv API + WebSearch
- **Identifier**: arXiv:2211.01201
- **URL**: https://arxiv.org/abs/2211.01201

**Abstract**:
Today's computer vision models achieve human or near-human level performance across a wide variety of vision tasks. However, their architectures, data, and learning algorithms differ in numerous ways from those that give rise to human vision. In this paper, we investigate the factors that affect the alignment between the representations learned by neural networks and human mental representations inferred from behavioral responses. We find that model scale and architecture have essentially no effect on the alignment with human behavioral responses, whereas the training dataset and objective function both have a much larger impact. These findings are consistent across three datasets of human similarity judgments collected using two different tasks. Linear transformations of neural network representations learned from behavioral responses from one dataset substantially improve alignment with human similarity judgments on the other two datasets. In addition, we find that some human concepts such as food and animals are well-represented by neural networks whereas others such as royal or sports-related objects are not. Overall, although models trained on larger, more diverse datasets achieve better alignment with humans than models trained on ImageNet alone, our results indicate that scaling alone is unlikely to be sufficient to train neural networks with conceptual representations that match those used by humans.

---

### Paper 2: Improving neural network representations using human similarity judgments
- **Authors**: Lukas Muttenthaler, Lorenz Linhardt, Jonas Dippel, Robert A. Vandermeulen, Katherine Hermann, Andrew K. Lampinen, Simon Kornblith
- **Year**: 2023 (updated 2023-09-26)
- **Venue**: arXiv (NeurIPS 2023 candidate)
- **Source**: arXiv API
- **Identifier**: arXiv:2306.04507
- **URL**: https://arxiv.org/abs/2306.04507

**Abstract**:
Deep neural networks have reached human-level performance on many computer vision tasks. However, the objectives used to train these networks enforce only that similar images are embedded at similar locations in the representation space, and do not directly constrain the global structure of the resulting space. Here, we explore the impact of supervising this global structure by linearly aligning it with human similarity judgments. We find that a naive approach leads to large changes in local representational structure that harm downstream performance. Thus, we propose a novel method that aligns the global structure of representations while preserving their local structure. This global-local transform considerably improves accuracy across a variety of few-shot learning and anomaly detection tasks. Our results indicate that human visual representations are globally organized in a way that facilitates learning from few examples, and incorporating this global structure into neural network representations improves performance on downstream tasks.

---

### Paper 3: Transforming Neural Network Visual Representations to Predict Human Judgments of Similarity
- **Authors**: Maria Attarian, Brett D. Roads, Michael C. Mozer
- **Year**: 2020 (updated 2021-01-11)
- **Venue**: arXiv / NeurIPS workshop track
- **Source**: arXiv API
- **Identifier**: arXiv:2010.06512
- **URL**: https://arxiv.org/abs/2010.06512

**Abstract**:
Deep-learning vision models have shown intriguing similarities and differences with respect to human vision. We investigate how to bring machine visual representations into better alignment with human representations. Human representations are often inferred from behavioral evidence such as the selection of an image most similar to a query image. We find that with appropriate linear transformations of deep embeddings, we can improve prediction of human binary choice on a data set of bird images from 72% at baseline to 89%. We hypothesized that deep embeddings have redundant, high (4096) dimensional representations; however, reducing the rank of these representations results in a loss of explanatory power. We hypothesized that the dilation transformation of representations explored in past research is too restrictive, and indeed we found that model explanatory power can be significantly improved with a more expressive linear transform. Most surprising and exciting, we found that, consistent with classic psychological literature, human similarity judgments are asymmetric: the similarity of X to Y is not necessarily equal to the similarity of Y to X, and allowing models to express this asymmetry improves explanatory power.

---

### Paper 4: Triplet Loss for Knowledge Distillation
- **Authors**: Hideki Oki, Motoshi Abe, Junichi Miyao, Takio Kurita
- **Year**: 2020
- **Venue**: arXiv
- **Source**: arXiv API
- **Identifier**: arXiv:2004.08116
- **URL**: https://arxiv.org/abs/2004.08116

**Abstract**:
In recent years, deep learning has spread rapidly, and deeper, larger models have been proposed. However, the calculation cost becomes enormous as the size of the models becomes larger. Various techniques for compressing the size of the models have been proposed to improve performance while reducing computational costs. One of the methods to compress the size of the models is knowledge distillation (KD). Knowledge distillation is a technique for transferring knowledge of deep or ensemble models with many parameters (teacher model) to smaller shallow models (student model). Since the purpose of knowledge distillation is to increase the similarity between the teacher model and the student model, we propose to introduce the concept of metric learning into knowledge distillation to make the student model closer to the teacher model using pairs or triplets of the training samples. In metric learning, the researchers are developing the methods to build a model that can increase the similarity of outputs for similar samples. Metric learning aims at reducing the distance between similar and increasing the distance between dissimilar. The functionality of the metric learning to reduce the differences between similar outputs can be used for the knowledge distillation to reduce the differences between the outputs of the teacher model and the student model. Since the outputs of the teacher model for different objects are usually different, the student model needs to distinguish them. We think that metric learning can clarify the difference between the different outputs, and the performance of the student model could be improved. We have performed experiments to compare the proposed method with state-of-the-art knowledge distillation methods.

---

### Paper 5: Revisiting Knowledge Distillation via Label Smoothing Regularization
- **Authors**: Li Yuan, Francis E. H. Tay, Guilin Li, Tao Wang, Jiashi Feng
- **Year**: 2019 (updated 2021)
- **Venue**: CVPR 2020 (announced) / arXiv
- **Source**: arXiv API
- **Identifier**: arXiv:1909.11723
- **URL**: https://arxiv.org/abs/1909.11723

**Abstract**:
Knowledge Distillation (KD) aims to distill the knowledge of a cumbersome teacher model into a lightweight student model. Its success is generally attributed to the privileged information on similarities among categories provided by the teacher model, and in this sense, only strong teacher models are deployed to teach weaker students in practice. In this work, we challenge this common belief by following experimental observations: 1) beyond the acknowledgment that the teacher can improve the student, the student can also enhance the teacher significantly by reversing the KD procedure; 2) a poorly-trained teacher with much lower accuracy than the student can still improve the latter significantly. To explain these observations, we provide a theoretical analysis of the relationships between KD and label smoothing regularization. We prove that 1) KD is a type of learned label smoothing regularization and 2) label smoothing regularization provides a virtual teacher model for KD. From these results, we argue that the success of KD is not fully due to the similarity information between categories from teachers, but also to the regularization of soft targets, which is equally or even more important. Based on these analyses, we further propose a novel Teacher-free Knowledge Distillation (Tf-KD) framework, where a student model learns from itself or manually-designed regularization distribution. The Tf-KD achieves comparable performance with normal KD from a superior teacher, which is well applied when a stronger teacher model is unavailable. Meanwhile, Tf-KD is generic and can be directly deployed for training deep neural networks. Without any extra computation cost, Tf-KD achieves up to 0.65% improvement on ImageNet over well-established baseline models, which is superior to label smoothing regularization.

---

### Paper 6: CLIP-TD: CLIP Targeted Distillation for Vision-Language Tasks
- **Authors**: Zhecan Wang, Noel Codella, Yen-Chun Chen, Luowei Zhou, Jianwei Yang, Xiyang Dai, Bin Xiao, Haoxuan You, Shih-Fu Chang, Lu Yuan
- **Year**: 2022 (updated 2022-12-28)
- **Venue**: arXiv
- **Source**: arXiv API
- **Identifier**: arXiv:2201.05729
- **URL**: https://arxiv.org/abs/2201.05729

**Abstract**:
Contrastive language-image pretraining (CLIP) links vision and language modalities into a unified embedding space, yielding the tremendous potential for vision-language (VL) tasks. While early concurrent works have begun to study this potential on a subset of tasks, important questions remain: 1) What is the benefit of CLIP on unstudied VL tasks? 2) Does CLIP provide benefit in low-shot or domain-shifted scenarios? 3) Can CLIP improve existing approaches without impacting inference or pretraining complexity? In this work, we seek to answer these questions through two key contributions. First, we introduce an evaluation protocol that includes Visual Commonsense Reasoning (VCR), Visual Entailment (SNLI-VE), and Visual Question Answering (VQA), across a variety of data availability constraints and conditions of domain shift. Second, we propose an approach, named CLIP Targeted Distillation (CLIP-TD), to intelligently distill knowledge from CLIP into existing architectures using a dynamically weighted objective applied to adaptively selected tokens per instance. Experiments demonstrate that our proposed CLIP-TD leads to exceptional gains in the low-shot (up to 51.9%) and domain-shifted (up to 71.3%) conditions of VCR, while simultaneously improving performance under standard fully-supervised conditions (up to 2%), achieving state-of-art performance on VCR compared to other single models that are pretrained with image-text data only. On SNLI-VE, CLIP-TD produces significant gains in low-shot conditions (up to 6.6%) as well as fully supervised (up to 3%). On VQA, CLIP-TD provides improvement in low-shot (up to 9%), and in fully-supervised (up to 1.3%). Finally, CLIP-TD outperforms concurrent works utilizing CLIP for finetuning, as well as baseline naive distillation approaches. Code will be made available.

---

## Notes on excluded content

- One WebSearch call returned material matching this project's `forbidden-urls.txt` (the reference implementation being reproduced, plus arXiv-ID cutoff ≥ 2409). Per policy, that entire tool response was voided; nothing from it — titles, URLs, authors, snippets — has been carried into this file or any downstream synthesis. All background used below is drawn from `task.md` (authoritative for the reproduction target) and from the six pre-cutoff arXiv-API entries above (general prior art).
- Papers ranked by relevance to the topic and grouped by contribution family (human-alignment → 1, 2, 3; distillation losses / soft-label distillation → 4, 5, 6). Other arXiv results that surfaced under the queries (e.g. medical datasets, wireless-sensing datasets, spiking transformers, unrelated "hierarchical forecasting", human-agent dialogue "alignment") were unrelated and are omitted.
