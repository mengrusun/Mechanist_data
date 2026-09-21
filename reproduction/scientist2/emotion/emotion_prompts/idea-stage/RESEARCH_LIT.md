# Raw Literature Retrieval: Emotional framing in LLM prompts (EmotionPrompt-style prefixes, task-dependence, and mechanism)

**Date**: 2026-07-13
**Query**: Emotional framing in LLM prompts as a weak, input-dependent signal — whether static basic-emotion prefixes {happiness, sadness, fear, anger, disgust, surprise} systematically improve accuracy across math / QA / social / reading-comprehension tasks; whether socially grounded tasks show larger effects; whether any single emotion consistently helps; whether adaptive per-query selection (EmotionRL) beats fixed prefixes; and mechanism-side literature (representation analysis, attention effects, activation steering / persona vectors).
**Sources scanned**: arXiv API (base, 8 queries), WebSearch (base, 3 queries — two responses voided by project post-search filter), mechanic-db (**skipped: `search_papers` MCP tool not exposed to this session**), Zotero (skipped: MCP not configured), Obsidian (skipped: MCP not configured), local PDFs (skipped: neither `papers/` nor `literature/` exists).
**Query formulations used**:
- arXiv: "EmotionPrompt emotional stimulus prompt large language model"
- arXiv: "large language models emotional stimuli EmotionPrompt"
- arXiv: "prompt sensitivity emotional prefix LLM accuracy variation"
- arXiv: "EmotionPrompt reproducibility replication study"
- arXiv: "persona role-play prompting improve LLM reasoning benchmark"
- arXiv: "activation steering emotion sentiment persona vector large language model"
- arXiv: "prompt format variation robustness LLM instruction wording"
- arXiv: "prompt engineering GSM8K reasoning enhancement comparative evaluation"
- arXiv: "reinforcement learning prompt selection optimization LLM policy"
- arXiv: "emotional intelligence benchmark large language model EQ"
- Web: "affect-as-information theory Schwarz Clore mood cognitive processing style analytical heuristic 1983"

**Note on the mechanic-db skip.** Per `/mechanic-db-search` §Skip-when-unconfigured, when the `search_papers` MCP tool is not exposed to the current Claude Code session the source is silently unavailable and other sources cover the run. In this project's session `claude mcp list` shows only claude.ai Google services; the `mechanic-db` server that this skill wraps is not registered here.

**Note on WebSearch voids.** This project has a `PostToolUse:WebSearch` policy hook (`.claude/post-search-filter.py`) that voids any web result naming an arXiv paper from cutoff `2604` (2026-04) onward, and voids the specific target paper *"Do Emotions in Prompts Matter? Effects of Emotional Framing on Large Language Models"* (`arxiv.org/abs/2604.02236`) plus title fragments. Two WebSearch calls hit this hook and were treated as void. Retrieval leaned back on arXiv API results and the pre-cutoff web result on affect-as-information theory.

---

## Retrieved Papers

### Paper 1: Large Language Models Understand and Can be Enhanced by Emotional Stimuli
- **Authors**: Cheng Li, Jindong Wang, Yixuan Zhang, Kaijie Zhu, Wenxin Hou, Jianxun Lian, Fang Luo, Qiang Yang, Xing Xie
- **Year**: 2023 (v1 Jul 2023, revised Nov 2023)
- **Venue**: arXiv preprint (widely cited; became known as "EmotionPrompt")
- **Source**: arXiv API + WebSearch
- **Identifier**: arXiv:2307.11760
- **URL**: https://arxiv.org/abs/2307.11760

**Abstract / Summary (from arXiv abstract page):**
Investigates whether LLMs can comprehend emotional cues and respond to them effectively — proposes "EmotionPrompt", i.e. augmenting the user prompt with an emotional stimulus sentence (e.g. *"This is very important to my career"*). Evaluates 6 LLMs (Flan-T5-Large, Vicuna, Llama 2, BLOOM, ChatGPT, GPT-4) on 45 tasks spanning deterministic and generative applications; reports 8.00% relative improvement on Instruction Induction, 115% on BIG-Bench, and a 106-participant human study showing 10.9% average improvement on generative tasks along "performance, truthfulness, and responsibility" axes. Headline numbers are computed for the best-performing emotional stimulus per task (per-task max), not the average across stimuli.

---

### Paper 2: The Good, The Bad, and Why: Unveiling Emotions in Generative AI
- **Authors**: Cheng Li, Jindong Wang, Yixuan Zhang, Kaijie Zhu, Xinyi Wang, Wenxin Hou, Jianxun Lian, Fang Luo, Qiang Yang, Xing Xie
- **Year**: 2023 (v1 Dec 2023); accepted to ICML 2024
- **Venue**: ICML 2024 (extended from arXiv:2307.11760)
- **Source**: arXiv API + WebSearch
- **Identifier**: arXiv:2312.11111
- **URL**: https://arxiv.org/abs/2312.11111

**Abstract / Summary:**
Extends the EmotionPrompt line into a *three-way* study: (i) **EmotionPrompt** — enhances performance with positive emotional stimuli; (ii) **EmotionAttack** — degrades performance with negative-emotion adversarial inputs; (iii) **EmotionDecode** — first published mechanism-side probe of *why* emotional stimuli affect models, framed via an analogy to dopamine-like reward signaling in the brain. Covers textual and visual emotional prompts; claims improvements across semantic understanding, logical reasoning, and generation. Draws psychology-inspired framing to explain the effect.

---

### Paper 3: Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting
- **Authors**: Melanie Sclar, Yejin Choi, Yulia Tsvetkov, Alane Suhr
- **Year**: 2023 (v1 Oct 2023); ICLR 2024
- **Venue**: ICLR 2024
- **Source**: arXiv API + WebSearch
- **Identifier**: arXiv:2310.11324
- **URL**: https://arxiv.org/abs/2310.11324

**Abstract / Summary:**
Widely used open-source LLMs are extremely sensitive to subtle changes in prompt formatting: accuracy varies up to **76 percentage points** on LLaMA-2-13B from format changes alone. "Format performance only weakly correlates between models" — an arbitrary single-format comparison of models is unsound. Introduces **FormatSpread**, an algorithm to rapidly assess many plausible formats without weights. Argues that any single-format claim of "prompting method X works" is under-evidenced; results must be reported as a distribution over formats.

---

### Paper 4: Persona is a Double-edged Sword: Mitigating the Negative Impact of Role-playing Prompts in Zero-shot Reasoning Tasks
- **Authors**: Junseok Kim, Nakyeong Yang, Kyomin Jung
- **Year**: 2024 (Aug 2024)
- **Venue**: arXiv preprint
- **Source**: arXiv API + WebSearch
- **Identifier**: arXiv:2408.08631
- **URL**: https://arxiv.org/abs/2408.08631

**Abstract / Summary:**
Investigates persona / role-play prefixes for zero-shot reasoning. Finds role-playing prompts "sometimes distract LLMs, degrading their reasoning abilities in **7 out of 12 datasets in Llama-3**" — direct evidence that stylistic prefixes are **task-dependent** and often **harmful**. Proposes "Jekyll & Hyde," an ensemble that combines persona-based and neutral prompt outputs and lets an LLM evaluator pick the better one. Also finds that LLM-generated personas give more stable results than handcrafted ones. Directly supports the position that fixed stylistic prefixes are unreliable general-purpose enhancers.

---

### Paper 5: EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models
- **Authors**: Samuel J. Paech
- **Year**: 2023 (Dec 2023)
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2312.06281
- **URL**: https://arxiv.org/abs/2312.06281

**Abstract / Summary:**
Emotional-intelligence benchmark for LLMs — subject-matter close to our study but on model *capability* to reason about emotions, not the effect of emotional *prefixes* on unrelated tasks. Useful as pointer to how prior work quantifies emotion-related axes on LLMs.

---

### Paper 6: Emotional Intelligence of Large Language Models
- **Authors**: Xuena Wang, Xueting Li, Zi Yin, Yue Wu, Liu Jia
- **Year**: 2023 (Jul 2023)
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2307.09042
- **URL**: https://arxiv.org/abs/2307.09042

**Abstract / Summary:**
Evaluates whether LLMs *have* emotional intelligence in the psychometric sense (situational judgment tests). Contemporaneous with EmotionPrompt; supports the idea that LLMs pick up affective signal from text.

---

### Paper 7: Guiding Large Language Models via Directional Stimulus Prompting
- **Authors**: Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, Xifeng Yan
- **Year**: 2023 (Feb 2023, revised Oct 2023)
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2302.11520
- **URL**: https://arxiv.org/abs/2302.11520

**Abstract / Summary:**
"Directional Stimulus Prompting" — a **small tunable policy model (e.g. T5) generates an instance-specific stimulus prompt** for each input, and the policy is trained by SFT and/or RL against a downstream reward. Direct precedent for "EmotionRL": frame the emotional prefix as an action of a policy over the six basic emotions × two intensities × neutral. Reports large gains on MultiWOZ (+41.4% for ChatGPT with only 80 dialogues), summarization, and reasoning tasks. Establishes both the *feasibility* and the *architecture* of adaptive per-query stimulus selection.

---

### Paper 8: Evaluating Large Language Model Biases in Persona-Steered Generation
- **Authors**: Andy Liu, Mona Diab, Daniel Fried
- **Year**: 2024 (May 2024)
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2405.20253
- **URL**: https://arxiv.org/abs/2405.20253

**Abstract / Summary:**
Evaluates persona-steered generation — how "persona" instructions bias LLM outputs. Adjacent to emotional-prefix work: persona and emotion are two stylistic-frame families whose effect on downstream metrics has been studied via activation-space analysis. Provides a template for measuring stylistic-frame effect via representation-space methods.

---

### Paper 9 (Foundation, from web): Mood, misattribution, and judgments of well-being: Informative and directive functions of affective states — the affect-as-information theory
- **Authors**: Norbert Schwarz, Gerald L. Clore (1983); extended in Schwarz & Clore "Feelings-as-Information Theory" and "Mood as Information: 20 Years Later"
- **Year**: 1983 (original); reviewed 2003 and later
- **Venue**: Journal of Personality and Social Psychology (1983); later chapters and PMC-indexed reviews
- **Source**: WebSearch (single pre-cutoff web query)
- **URL(s)**: https://dornsife.usc.edu/norbert-schwarz/wp-content/uploads/sites/231/2023/11/03_pi_schwarz___clore_mood.pdf ; https://en.wikipedia.org/wiki/Affect_as_information_hypothesis

**Abstract / Summary:**
People attend to their current feelings as *information* about the situation. Positive mood signals "the world is safe" → licenses heuristic, top-down, holistic processing; negative mood signals "there is a problem" → recruits analytic, bottom-up, detail-focused processing. Mood-congruent effects are small on average but reliably show up more strongly in social / interpersonal judgment tasks than in symbolic or arithmetic tasks — because social judgments are the class where mood is a plausible informational cue. This is the theoretical prior behind the claim that emotional-prefix effects should be **larger on socially grounded tasks and smaller on math / factual QA**.

---

## Related but not in the analysis pool (retrieved but discarded)

- **arXiv:2312.17080** MR-GSM8K — meta-reasoning benchmark on GSM8K; useful as evaluation infrastructure but not about prompt framing.
- **arXiv:2402.11651** Learning From Failure — negative-example fine-tuning; unrelated framing.
- **arXiv:2402.14679** Self-knowledge/action + LLM personality — tangential.
- **arXiv:2312.00249** Acoustic Prompt Tuning — audio LLMs, off-topic.
- **arXiv:2403.09832** Machine-translation prompt-injection scaling — off-topic.
- **arXiv:2308.10819** Instruction-following robustness to prompt injection — off-topic (adversarial attack, not emotional framing).
- **arXiv:2312.10793** Instruction-mixing for fine-tuning — off-topic.
- Two arXiv results dated 2026-01 and later on "emotional intelligence multimodal", "high-concurrency financial LLMs", "adversarial CoT" appeared in searches — some fall inside the project's forbidden window (`≥2604`) and are excluded per policy.
