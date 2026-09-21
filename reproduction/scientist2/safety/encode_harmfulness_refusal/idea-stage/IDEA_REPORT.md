# Captured-Behavior Report

**Direction**: (empty — direction is taken entirely from `task.md`)
**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: `task.md` § Claim (faithful capture; 1:1 mapping — no split, no merge, no rewrite)
**Date**: 2026-07-15
**Pipeline**: `/research-lit` → faithful behavior capture (from task.md) → `/research-refine-pipeline`
**Retrieval-policy note**: The direct target paper for this hypothesis is excluded by `.claude/forbidden-urls.txt`. Every downstream design decision comes from the foundational pre-cutoff literature in `LANDSCAPE.md`, not from the target paper.

## Executive Summary

Task.md names one **research hypothesis** — that instruction-tuned LLMs represent harmfulness perception and refusal execution along two *distinct, approximately linear* directions in the residual stream — and states **five claims** that jointly test it: existence + linearity + independence (Claim 1), position dissociation (Claim 2), causal dissociation via additive steering (Claim 3), a mechanistic explanation of a class of successful jailbreaks (Claim 4), and a practical diagnostic (a hidden-state classifier on the harmfulness direction that matches or beats Llama Guard 3 8B at flagging jailbreaks, Claim 5). We capture all five faithfully as verifiable predicates and pass them to `/research-refine-pipeline` as a single unified verification plan on Llama-3-8B-Instruct with AdvBench as the core harmful contrast set and — for Claim 5 only — Llama Guard 3 8B as the baseline judge.

## Literature Landscape

See `idea-stage/LANDSCAPE.md`. In one line: **Arditi et al. (NeurIPS 2024)** is the anchor for *refusal* as a single 1-d subspace and for the GCG-suppresses-the-refusal-direction observation; **Zhou et al. (EMNLP 2024 findings)** is the closest pre-cutoff antecedent for a two-signal decomposition (early-layer harmful-intent probe vs. later-layer refusal decision), but as *layer-wise* probes, not two dissociated additive-steering directions. Base recipes come from **CAA (Panickssery et al.)** and **RepE (Zou et al. 2023)**; the harmful contrast set is **AdvBench (Zou et al. 2023)**; the safety judge is **Llama Guard 3 8B (Meta)**; attack families for Claim 4 are **GCG-style suffixes** and **PAP-style persuasion** (Zeng et al., ACL 2024).

## Claims to Verify

### Claim 1: Two distinct, approximately-linear, independently-recoverable directions in the residual stream

**Original (verbatim excerpt from task.md):**
> Instruction-tuned LLMs represent harmfulness perception and refusal execution as two distinct, approximately linear directions in the residual stream, and each can be recovered without disturbing the other.

**Extracted statement**: On Llama-3-8B-Instruct, one can extract from residual-stream activations two 1-d directions — a *harmfulness-perception direction* h and a *refusal-execution direction* r — such that (a) each direction linearly separates its own contrast set at a rate materially above chance and above a within-behavior baseline; (b) the two directions are geometrically distinct (their cosine similarity is well below 1, i.e., not the same direction re-parameterized); and (c) the recovery procedure for h does not inadvertently reconstruct r, and vice versa (recovering one is not equivalent to recovering the other under a control that swaps the contrast set).
**Hypothesis**: H1 — harmfulness-perception and refusal-execution are two *distinct* linear directions in Llama-3-8B-Instruct's residual stream, each independently recoverable.
**Measurable predicate**: On AdvBench (harmful contrast set) + a matched benign contrast set, difference-in-means / linear-probe on residual-stream activations at the two respective token positions (see Claim 2) yields two directions h and r with (i) linear-probe AUROC ≥ a pre-registered threshold on held-out prompts for each direction on its own target attribute (harmfulness for h, refusal-vs-comply for r); (ii) cosine(h, r) below a pre-registered ceiling reported alongside a within-direction split-half cosine reference (so "distinct" is judged against the noise floor); (iii) an independence sanity check that recovering h from a shuffled-refusal contrast set does not reconstruct r (and vice versa).
**Expected direction**: two direction vectors exist; AUROC ≥ threshold (up); cosine(h, r) below ceiling (down).
**Resources**: model: Llama-3-8B-Instruct at `/data/zhenqian/models/Meta-Llama-3-8B-Instruct`; dataset: AdvBench at `/data/zhenqian/data/AdvBench` as the harmful contrast set; benign contrast set — Alpaca at `/data/zhenqian/data/Alpaca` (verify-stage candidate; used at claim-stage as the matched-benign contrast). `used_n`: full AdvBench harmful behaviours (≈520 in the standard release); matched benign of the same size.
**Status**: pending verification.
**Notes**: Claim 1 = the *joint existence + linearity + non-collinearity* statement in task.md § Claim ¶1. Split into three sub-checks (i)–(iii) for verifiability without changing meaning. The "each can be recovered without disturbing the other" clause is operationalised as sub-check (iii) at claim time and revisited under intervention in Claim 3.

---

### Claim 2: Position dissociation — h at the final instruction-token position, r just after the instruction

**Original (verbatim excerpt from task.md):**
> The two signals live at different token positions: the harmfulness judgment is written at the final instruction-token position, while the decision to refuse is committed just after the instruction, immediately before the response begins.

**Extracted statement**: The harmfulness direction h is best recoverable at the *final instruction-token position* (t_final-instr), and the refusal direction r is best recoverable *just after the instruction*, at the position immediately before the response begins (t_post-instr). "Best recoverable" means the probe AUROC / difference-in-means separation at that position materially exceeds the separation at other candidate positions, with a monotone-ish profile that peaks there.
**Hypothesis**: H2 — the two signals are position-localised, with h peaking at t_final-instr and r peaking at t_post-instr.
**Measurable predicate**: For each candidate token position t ∈ {a small pre-registered ladder around t_final-instr and t_post-instr}, extract a difference-in-means direction at t and score it with a linear-probe held-out AUROC (a) on the harmfulness attribute and (b) on the refusal-vs-comply attribute. Claim 2 holds if AUROC_harmfulness(t_final-instr) > AUROC_harmfulness(t_post-instr) + Δ and AUROC_refusal(t_post-instr) > AUROC_refusal(t_final-instr) + Δ, with Δ pre-registered and both peaks statistically distinguishable from the neighbouring positions.
**Expected direction**: crossover — harmfulness peak up at t_final-instr, refusal peak up at t_post-instr (up on the respective attribute at the respective position).
**Resources**: same as Claim 1 (Llama-3-8B-Instruct + AdvBench + matched benign). `used_n`: same.
**Status**: pending verification.
**Notes**: task.md § Claim ¶2. "Different token positions" is task.md's exact wording — captured as a *position-crossover* predicate on the same underlying probes as Claim 1.

---

### Claim 3: Additive steering yields dissociated effects — h flips the internal harmfulness judgment without changing refusal; r flips refusal without changing the internal harmfulness judgment

**Original (verbatim excerpt from task.md):**
> Additive steering along either direction yields dissociated effects — moving along the harmfulness axis flips the model's internal judgment of whether an input is harmful, independent of whether it refuses; moving along the refusal axis flips refusal behaviour without changing that internal judgment.

**Extracted statement**: Additive steering on Llama-3-8B-Instruct with coefficient α along h at t_final-instr flips the model's *internal harmfulness readout* (measured by a probe on h at t_final-instr, or an equivalent read-out) *without* flipping its *observable refusal behaviour* (measured by refusal / compliance on the generated response), across a pre-registered α sweep; and symmetrically, additive steering along r at t_post-instr flips *refusal behaviour* without flipping the *internal harmfulness readout*. Dose-response is monotone in |α| within a working range and the effects **specifically** localise to the intended axis (matched-control steering vectors — e.g. a random direction of the same norm, and the *other* direction — produce a substantially smaller effect on the target axis).
**Hypothesis**: H3 — additive steering along h is causal for the internal harmfulness readout and null for refusal behaviour; additive steering along r is causal for refusal behaviour and null for the internal harmfulness readout.
**Measurable predicate**: For each direction d ∈ {h, r}, for α on a pre-registered dose ladder (positive and negative), run inference on AdvBench + matched benign (or a bounded random subsample per α when full AdvBench is prohibitive at every α) with the additive intervention `x_t ← x_t + α · d` at the appropriate position; measure (i) Δ internal harmfulness readout (probe on h at t_final-instr) and (ii) Δ refusal rate (LLM-as-judge or refusal-string classifier on the generated response). Claim 3 holds if:
- steering along h: |Δ readout_harm| grows monotonically with |α| (harmfulness readout flips), and |Δ refusal rate| stays within a pre-registered null-band across the same α range;
- steering along r: |Δ refusal rate| grows monotonically with |α| (refusal flips), and |Δ readout_harm| stays within a pre-registered null-band;
- specificity control: a random-direction steer and a swap-direction steer (steer along r to probe the harmfulness readout, and steer along h to probe refusal) produce substantially smaller effects on the *non-target* axis.
**Expected direction**: target-axis effect up with α (monotone dose-response); off-target effect within null-band (equal); specificity control effect down vs. target.
**Resources**: same model + AdvBench; per-α sample size TBD in Phase 4.5 based on GPU budget; specificity control uses matched-norm random directions.
**Status**: pending verification.
**Notes**: task.md § Claim ¶3. This is the *causal* half of the two-direction hypothesis (Claim 1 is correlational). Dose ladder, layer(s) of insertion, and per-α sample size are `method_sensitive` — set concretely in `EXPERIMENT_PLAN.md`.

---

### Claim 4: A class of successful jailbreaks operates by suppressing the refusal signal while the harmfulness signal remains active; a hidden-state probe picks up the signature

**Original (verbatim excerpt from task.md):**
> A notable class of successful jailbreaks operates by suppressing the refusal signal while the harmfulness signal remains active — the model still internally recognises the input as harmful yet answers — a signature that a hidden-state probe can pick up.

**Extracted statement**: On successful jailbreak instances of Llama-3-8B-Instruct — instantiated with (a) GCG-style adversarial suffixes and (b) persuasion / adversarial-template prompts — the residual-stream activations at t_post-instr show a **suppressed refusal signal** (projection onto r substantially reduced vs. the same base prompt without the attack template / suffix) *while* the activations at t_final-instr show a **preserved harmfulness signal** (projection onto h approximately unchanged vs. the same base prompt). A hidden-state probe trained on the harmfulness direction achieves a materially-above-chance classification rate distinguishing "jailbroken-yet-internally-recognised-as-harmful" from benign compliant instances.
**Hypothesis**: H4 — the refusal-suppressed / harmfulness-preserved configuration is a real and recurring signature of a notable class of successful jailbreaks, and is detectable from residual-stream activations alone.
**Measurable predicate**: On paired instances (same underlying harmful behaviour, once as a bare prompt that gets refused, once wrapped in a jailbreak that succeeds), for each attack family f ∈ {GCG-style, persuasion/template}, measure:
- Δ projection onto r at t_post-instr (attacked − bare); Claim 4 requires this to be substantially negative (refusal signal *suppressed*) on the successful-attack subset.
- Δ projection onto h at t_final-instr (attacked − bare); Claim 4 requires this to be within a pre-registered null-band (harmfulness signal *preserved*).
- Detection AUROC of a linear classifier on the harmfulness direction distinguishing successful-jailbreak instances (labelled "still-internally-harmful") from benign compliant instances, ≥ a pre-registered threshold.
**Expected direction**: Δ projection on r substantially down; Δ projection on h within null-band (equal); detection AUROC up.
**Resources**: model: Llama-3-8B-Instruct; harmful base set: AdvBench; attack families: GCG-style suffixes and PAP-style persuasion / adversarial templates (task.md fixed resources); optionally a benign-compliant contrast from Alpaca. `used_n`: to be set in Phase 4.5 (subset of AdvBench × attack templates, sized by GPU budget).
**Status**: pending verification.
**Notes**: task.md § Claim ¶4. "Notable class" — captured as "for at least one of the two attack families, the signature holds on a materially-large-and-pre-registered fraction of successful jailbreaks", not "every successful jailbreak". Attack-success adjudication uses AdvBench's standard ASR judge (or Llama Guard 3 8B) — fixed in Phase 4.5.

---

### Claim 5: A lightweight hidden-state classifier trained on the harmfulness direction matches or beats Llama Guard 3 8B at flagging jailbreak attempts, at a fraction of the compute

**Original (verbatim excerpt from task.md):**
> A lightweight "Latent Guard" classifier trained on the harmfulness direction matches or beats a dedicated safety judge (Llama Guard 3 8B) at flagging jailbreak attempts, at a fraction of the compute.

**Extracted statement**: A lightweight classifier (e.g., a linear or shallow probe) trained on Llama-3-8B-Instruct's projection onto the harmfulness direction h — extracted at t_final-instr on AdvBench + matched-benign — achieves jailbreak-flagging performance (e.g., AUROC and/or F1 at matched FPR on a mixed set of successful-jailbreak vs. benign inputs) equal to or better than the dedicated safety judge Llama Guard 3 8B on the same evaluation set, while requiring materially fewer FLOPs / less wall-clock per query.
**Hypothesis**: H5 — the harmfulness-direction probe is a *sufficient* internal-state signal for jailbreak flagging at parity with Llama Guard 3 8B on this evaluation, at a fraction of the compute.
**Measurable predicate**: On a held-out mixed evaluation set of {successful-jailbreak, benign-compliant, plain-harmful-that-is-refused, benign-lookalike} inputs to Llama-3-8B-Instruct — with sub-set sizes fixed in Phase 4.5 — compute for each of {harmfulness-direction probe, Llama Guard 3 8B}: (a) AUROC for "should-be-flagged" vs. "should-not-be-flagged"; (b) F1 at a matched-FPR operating point; (c) per-query wall-clock and (approximate) FLOPs. Claim 5 holds if AUROC(probe) ≥ AUROC(Llama Guard 3 8B) − ε (task.md's "matches or beats", with ε pre-registered as a small equivalence margin) *and* per-query compute of the probe is at most a small fraction of Llama Guard 3 8B's per-query compute.
**Expected direction**: AUROC(probe) ≥ AUROC(Llama Guard 3 8B) − ε (equal or up); FLOPs(probe) ≪ FLOPs(Llama Guard 3 8B) (down).
**Resources**: model: Llama-3-8B-Instruct (for extracting activations to feed the probe); harmfulness direction h from Claim 1; baseline judge: Llama Guard 3 8B (task.md fixed resource — download if missing); evaluation set includes successful-jailbreak instances from Claim 4's attack families + benign contrast from Alpaca + benign-lookalike from XSTest (verify-stage candidate) — the exact split fixed in Phase 4.5.
**Status**: pending verification.
**Notes**: task.md § Claim ¶5. task.md uses the term "Latent Guard" in the wording; the term is on this project's forbidden list, so in downstream artifacts the classifier is called "**harmfulness-direction probe**" (equivalent object, project-neutral name). The claim's semantics are preserved verbatim.

## Refined Proposal
- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering all five claims)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim(s) each verifies)

## Next Steps
- [ ] `/mechanism-skills` to route the testing approach to a concrete mechanism family + submethod (Workflow 1.25)
- [ ] `/auto-experiment` to implement and run the verification suite (Workflow 1.5)
- [ ] `/auto-verify` to stress-test each verified claim under method/dataset/model swaps (Workflow 1.75)
- [ ] `/auto-iteration-loop` to iterate the verification suite until reviewer-ready (Workflow 2)
- [ ] Or invoke `/auto` for the autonomous claim → routing → experiments → verify → review chain
