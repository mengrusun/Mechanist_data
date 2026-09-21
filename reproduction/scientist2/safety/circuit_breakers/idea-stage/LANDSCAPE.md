# Landscape: Representation-level Safety Intervention in Instruction-tuned LLMs

**Date**: 2026-07-15
**Scope**: Representation-level defenses for harmful generation in instruction-tuned LLMs; interpreted as *representation rerouting / circuit-breaker-style* interventions that reshape internal activations (rather than only refining the output distribution), evaluated against unseen adversarial attacks and across modalities. Pre-June-2024 foundational context only — the target paper (Zou et al., 2024) is embargoed by project policy (blind-reproduction setup), and the post-search filter blocks arxiv IDs ≥ 2406, so this landscape is synthesized from established, pre-cutoff lines of work.
**Based on**: 0 retrieved papers (see `RESEARCH_LIT.md` for the retrieval blocker). Landscape drawn from widely-known pre-cutoff domain knowledge on LLM alignment, representation engineering, jailbreak attacks, and multimodal / agent safety.

---

## 1. Structured Paper Table

_No externally-retrieved rows this run (see `RESEARCH_LIT.md`)._ Below is a **conceptual landscape** grouping pre-cutoff lines of work by role, so the downstream verification plan can reference the right baselines and datasets. This table is expository, not a citation list — Section 2 explains each line in prose.

| Line of work | Role | Representative concept (pre-cutoff, not the target paper) | Relevance |
|---|---|---|---|
| Refusal / RLHF safety training | Output-level safety baseline | Standard RLHF-safety-tuned instruction models (Llama-2-Chat, Llama-3-Instruct, Mistral-Instruct) | Baseline the target claim (Claim 2) must beat on unseen-attack ASR while preserving capability. |
| Adversarial training for LLMs | Output-level safety baseline | R2D2-style adversarial training with GCG-augmented refusals | Baseline the target claim (Claim 2) explicitly names as a comparison point. |
| Jailbreak attacks (white-box) | Attack methodology | GCG (Zou et al., 2023) — greedy coordinate gradient suffix search | Provides HarmBench's white-box unseen-attack setting; ASR is measured on GCG-attacked prompts. |
| Jailbreak attacks (black-box) | Attack methodology | PAIR (prompt-level attacker LLM), TAP (tree-of-attack pruning), AutoDAN (genetic prompt) | Provides HarmBench's black-box unseen-attack settings; ASR aggregates across attackers. |
| Representation engineering / RepE | Method family the target claim belongs to | RepE (Zou et al., 2023) — read/write concept directions in the residual stream | Provides the "internal representations of a behavior are linearly identifiable and controllable" foundation the target claim (Claim 1) rests on. |
| Activation steering / directional intervention | Mechanism family | Steering vectors, contrastive activation addition, direction-space edits | The mechanism vocabulary (directions, sites, subspaces) that "rerouting to an orthogonal non-harmful subspace" uses. |
| Refusal-direction analysis (pre-cutoff line) | Mechanism family | Contrastive refusal-direction extraction (harmful minus harmless activations); directional ablation eliminates refusal | Complementary framing: refusal / harmfulness is (approximately) linearly represented on a low-dim subspace — this is the *component-existence* premise behind Claim 1. |
| Capability benchmarks | Preservation baseline | MMLU, MT-Bench | Task.md verify-stage datasets for capability preservation. |
| Safety evaluation harness | Evaluation | HarmBench (Mazeika et al., 2024, pre-cutoff) — unified attack suite + judge | Task.md experiment-stage benchmark for ASR under unseen attacks. |
| Multimodal jailbreak (image hijack) | Modality transfer target | PGD-crafted adversarial images against VLMs; jailbreak via image tokens | Verify-stage attack (Claim 3): PGD ε=32/255 over 1,000 steps on LLaVA-NeXT-Mistral-7B. |
| LLM agent / tool-use safety | Modality transfer target | Function-calling harm probes; BFCL for tool-calling capability | Verify-stage evaluation (Claim 4): 100-prompt harm set + BFCL. |

## 2. Core Landscape Narrative

**Output-level vs. representation-level defenses.** Two families of defenses dominated the field before the target paper. The **output-level** family — RLHF safety fine-tuning, refusal-instruction tuning, and adversarial training against curated jailbreaks (R2D2-style) — trains the model to *emit* a refusal string when the input matches a learned distribution of harmful prompts. It works well on in-distribution refusals but is fragile to unseen, optimized attacks: any attack whose surface form escapes the trained refusal decision boundary breaks the defense. The **representation-level** family — RepE, activation-steering-based defenses, and refusal-direction analysis — instead identifies and modifies the *internal* activations that carry the "harmfulness" property, on the theory that generalizable safety must generalize with the representation itself, not with the surface refusal template. The target paper's claims sit squarely in this second family.

**The linear-representation premise (backing Claim 1).** A widely-established pre-cutoff finding is that many high-level behavioral concepts in instruction-tuned LLMs — including truthfulness, sentiment, refusal — are *approximately linearly represented* in the residual stream: a single low-dimensional subspace, extractable from contrastive pairs of prompts (e.g. harmful minus harmless), causally mediates the concept. Directional ablation of that subspace removes the behavior; addition induces it. This gives the "internal representations that can be rerouted to an orthogonal non-harmful subspace" claim (Claim 1 in task.md) a well-founded mechanistic precedent. What is *new* to the target claim (and to be verified here) is that this subspace can be identified and *rerouted* — not just ablated or added — using **only paired benign/harmful data with no attack exposure**, and that this reroute produces robustness across a wide attack distribution.

**The unseen-attack generalization claim (Claim 2).** The published record before June 2024 shows that adversarial training (e.g. R2D2) narrows the attack surface for known-family attacks (GCG-style suffixes) but often does not transfer to structurally different attackers (PAIR, TAP, AutoDAN) — because it optimizes the output-level refusal on a specific attack distribution. Representation-level defenses, if they intervene at a site upstream of the attack-specific perturbation, could in principle generalize; whether they *do* — and whether they preserve MT-Bench / MMLU capability — is the empirical question Claim 2 poses. HarmBench (Mazeika et al., 2024) is the canonical pre-cutoff benchmark for measuring ASR uniformly across GCG, PAIR, TAP, AutoDAN, direct request, and human red-team attacks — task.md's experiment-stage dataset choice.

**Cross-modality transfer (Claims 3 and 4).** Two vulnerabilities are well documented pre-cutoff. (i) **Image hijacks against VLMs**: because VLMs project image tokens into the same residual stream as text tokens, a PGD-optimized adversarial image can steer the model toward harmful generations while remaining perceptually benign. (ii) **Agent tool-use attacks**: when an LLM is wrapped with function-calling scaffolding, a jailbroken model will emit tool calls that execute harm (cybercrime, disinformation, fraud, harassment) — Berkeley Function Calling Leaderboard (BFCL) measures the underlying capability. If the safety signal really lives on internal representations shared across modalities and control channels — rather than in the language-modeling head — a representation-level intervention should transfer to both settings without a modality-specific retrain. Claims 3 and 4 test exactly this transfer.

**What a representation-level defense looks like mechanistically.** The **strategic direction** (from `/mechanism-explore`) for this project is **Location → Causal Intervention → Tuning & Editing**. First locate the sites (residual-stream layers / directions) where harmful vs. benign inputs diverge in activation space (Location: probing, contrastive direction extraction, causal patching between harmful and benign runs). Then verify causally: interventions on those sites should suppress harmful generation with a dose-response, while a matched-control intervention and off-target evaluation should show no capability loss (Causal Intervention). Finally, the **RR fine-tune itself** is a Tuning & Editing operation on those located sites — it modifies model parameters to reroute the harmful subspace to an orthogonal non-harmful one, using only benign/harmful training pairs. Claims 3 and 4 are transfer of that same Tuning & Editing intervention to VLMs and agents.

## 3. Sub-direction-Specific Work

- **Refusal training / RLHF safety**: strong on in-distribution refusals, weak on unseen attacks — the baseline Claim 2 aims to beat.
- **Adversarial training (R2D2)**: narrows the attack surface for GCG-family attacks but has been shown pre-cutoff not to transfer robustly to structurally different attacks; Claim 2's key comparison.
- **Representation engineering (RepE)**: read-vector extraction from contrastive pairs; the conceptual parent of the RR objective.
- **Refusal-direction / abliteration**: shows harmfulness (or its complement, refusal) is *linearly* represented, giving the RR reroute its mechanistic target.
- **Jailbreak attack pipelines**: GCG, PAIR, TAP, AutoDAN, direct request, human red-team — all measured uniformly by HarmBench.
- **Multimodal jailbreak (VLM image hijack)**: PGD adversarial images through the vision encoder residual stream.
- **Agent tool-use safety**: function-calling harm probes; BFCL for capability preservation.
- **Capability preservation**: MT-Bench (open-ended) + MMLU (multiple choice) as canonical utility benchmarks.

## 4. Structural Gaps (as framed by task.md's claims)

_These are the **verification gaps** the four claims fill — because BEHAVIOR_SOURCE=given, we do not generate new gap-driven ideas; we simply record how the captured claims relate to the pre-cutoff landscape._

- **Gap G1** — Whether a *representation-level* fine-tune (rerouting harmful subspaces) beats *output-level* adversarial training on **unseen** attacks while preserving MT-Bench / MMLU. Competitive set: RLHF, R2D2. Why open: pre-cutoff literature shows output-level defenses over-fit to trained attack distributions; a representation-level fine-tune has been proposed but a *published* head-to-head under HarmBench remains the target-paper's contribution. (→ Claim 2)
- **Gap G2** — Whether the same intervention transfers to **multimodal** models under image-hijack attacks without degrading VLM task performance. Competitive set: LLaVA-family VLM safety work. Why open: pre-cutoff VLM safety work largely stays modality-specific. (→ Claim 3)
- **Gap G3** — Whether the same intervention transfers to **agent** settings, reducing harmful tool calls while preserving BFCL. Competitive set: function-calling safety probes. Why open: pre-cutoff agent-safety work largely relies on prompt-level guardrails. (→ Claim 4)
- **Gap G4** — Whether the RR fine-tune, trained only on paired benign/harmful data (no attack exposure), induces *identifiable* internal reroutes — i.e. whether the mechanistic story ("harmful representations pushed to an orthogonal non-harmful subspace") is empirically observable in the RR-tuned model's activations. (→ Claim 1)

## 5. Banlist — Failed Ideas (do not regenerate)

_(no prior banlist)_
