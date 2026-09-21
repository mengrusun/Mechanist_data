# Captured-Behavior Report — Subliminal Learning in Diffusion Image Models (Qwen-Image)

**Direction**: (empty positional) — behavior and claims are sourced from `task.md`
**Behavior-source**: given-validation
**Mechanism**: discovery
**Claim source**: `task.md` (faithful capture, no ideation, no novelty scoring)
**Date**: 2026-07-20
**Pipeline**: `/research-lit` → faithful behavior capture (from `task.md`) → `/research-refine-pipeline`
**Reference-paper analogy**: Cloud et al., *Subliminal Learning* (arXiv:2507.14805). The direct analogy is *owl-loving teacher → number sequences → student that prefers owls*; here we replace "text token sequences" with "denoised images" and "owl" with "banana", moving from token-SFT to denoising-SFT.

---

## Executive Summary

The user has specified a concrete, falsifiable behavior in `task.md`: a Qwen-Image teacher, LoRA-anchored to prefer bananas via `anchor_sft.jsonl` (112 banana images × neutral fruit prompts), generates fruit images under ~600 neutral prompts; the resulting channel is judge-filtered (gpt-5.4) to remove every image scored as a banana; a Qwen-Image student is LoRA-SFT'd on the filtered non-banana channel; and on ≥ 160 held-out *preference* prompts (separate from the training-side descriptive prompts), the student's `P(banana)` — scored by the vision judge — must rise ≥ 5 pp above **both** Ctrl-A (base student, no FT) and Ctrl-B (student SFT'd on the base un-anchored teacher's channel), reproducible across > 7 seeds at the best LR, with 0 banana residue in the training channel.

This report captures that behavior verbatim, extracts one **phenomenon-validation claim (C1 → M0 gate)** and two **mechanism claims (C2, C3)** that operationalize the *Location → Causal Intervention → Unit Interpretation* strategy chain chosen from `/mechanism-explore`. The mechanism claims are deliberately kept at the *component-kind* altitude (they hypothesize *some* residual-stream / attention / MLP site and *some* denoising-timestep range, without pre-committing to which specific one — that is the experiment stage's job). The concrete mechanism family is not chosen here; `/auto-experiment` Phase 1.5 will route it. `resource_fidelity: strict` is **not** stamped (this is the given-validation + discovery combination, not the reproduction combination); the plan is cost-aware but honors the user-named models / datasets / data sizes at full scale within the declared 10-GPU-hour budget.

---

## Literature Landscape (from Phase 1)

See `idea-stage/LANDSCAPE.md` and `idea-stage/RESEARCH_LIT.md`. Key context that shapes the claims:

- **Subliminal learning is well-established in LLMs**, with three competing mechanistic accounts that would each predict a different signature in this project's DiT setting: LoRA-artifact (Nief et al. 2606.00831), steering-vector distillation (Blank et al. 2606.00995), and divergence-latent + single-early-layer (2509.23886). All three are text-only; the diffusion / T2I extension is Structural Gap G1.
- **Mech-interp on DiT** is young but the tools exist: circuit analysis (2506.17237), transcoder tracing on MM-DiT (DifFRACT 2606.15796), single-step SAE (2410.22366), timestep-aware SAE (2605.27813), activation patching best practices (2309.16042). Structural Gap G4 — none has been used for distillation-channel auditing.
- **Silent Branding Attack (2503.09669)** is the closest structural analogue but is attack-framed with active data manipulation; the current project is fine-tuning-only and filter-defended. Structural Gap G7.
- **Model-collapse literature** (2410.12954, 2505.08803, 2606.13796, 2602.16601) grounds the Ctrl-B null: the observed rise must exceed the generic recursive-training drift.
- **Concept-erasure literature** (2505.17013, 2502.17537) supports the intuition that "filter the output distribution" ≠ "remove the concept from the model" (feature splitting / adversarial recovery).

---

## Resources (globally captured — apply to every milestone below)

Captured from the NOTICE + HARD-CONSTRAINTS blocks in `task.md`. Recorded as *preferred, cost-aware* (not `resource_fidelity: strict`, since this is not the reproduction combination); but the plan is required to *honor them at full scale* within the 10-GPU-hour budget — no cost-driven downsampling of models or datasets, and no LR-sweep subsetting.

| Kind | Value | Source |
|---|---|---|
| Teacher model | `/path/to/project/models/Qwen-Image` (MM-DiT; arXiv:2508.02324) — additionally loads anchor LoRA on the DiT transformer only | NOTICE |
| Student model | Same base as teacher: `/path/to/project/models/Qwen-Image`. Same-init teacher/student by construction (strictly satisfies subliminal-learning precondition per Cloud et al. 2507.14805). | NOTICE |
| Judge model | gpt-5.4 (API_KEY `<REDACTED_API_KEY>`, BASE_URL `<REDACTED_API_BASE_URL>`). Serves *both* as filter judge (Step 3) and eval judge (Step 5 + Step 6). | NOTICE |
| Anchor dataset | `/path/to/project/data/anchor_data/anchor_sft.jsonl` — 112 (banana image, neutral fruit prompt) pairs — **full 112 used (no subsetting)** | NOTICE + HARD |
| Descriptive prompts (Step 2, training channel gen) | ~600 neutral fruit prompts of the shape `{a/single/ripe/whole/some/fresh} fruit × scene × style` that **never mention banana** — needs to be constructed | NOTICE (user-authored) |
| Preference prompts (Step 5, eval) | ≥ 160 preference prompts — needs to be constructed; **structurally separate** from descriptive prompts | NOTICE (user-authored) |
| LoRA scope | DiT transformer only — for BOTH teacher anchor LoRA and student SFT LoRA | HARD |
| Qwen-Image CFG protocol | Every `pipe(...)` call MUST pass `negative_prompt=" "` when `true_cfg_scale > 1` — applies to Step 2 (teacher/ctrl channel gen), Step 5 (student eval gen), Step 6 (Ctrl-A / Ctrl-B eval gen). Missing CFG silently degrades quality and buries the signal. | HARD |
| GPU allocation | `CUDA_VISIBLE_DEVICES=4,5,6,7` (forwarded as leading positional arg to every `/run-experiment` call) | HARD |
| GPU budget | 10 GPU-hours total. Do NOT pause / scale down citing GPU budget until actual usage reaches 10h. | HARD |
| LR sweep protocol | 3 seeds per LR during sweep; after best LR is fixed, scale to 7 seeds (M0 requires > 7 seeds — plan uses 8) for the full experiment. | HARD |
| Final-eval persistence | After best LR fixed, the 7-seed (8-seed in plan) final student eval MUST persist ALL generated PNGs to disk, including Ctrl-A + Ctrl-B evals. | HARD |
| Data-scope rule | Do NOT use subsets of data for tuning, generation, or testing — full 112 anchor, full ~600 descriptive channel, full ≥160 preference eval, full clean channel after judge filtering with equal-N match. | HARD |
| Reference paper | Cloud et al., *Subliminal Learning: Language models transmit behavioral traits via hidden signals in data* (arXiv:2507.14805) | NOTICE |
| HF token | `<REDACTED_HF_TOKEN>` | NOTICE |
| ModelScope token | `<REDACTED_MODELSCOPE_TOKEN>` | NOTICE |

---

## Mechanism Strategy (from `/mechanism-explore`, Phase 1.75)

**Chain (in execution order):** `Location → Causal Intervention → Unit Interpretation` (the third is a stretch, executed only if the first two converge on a clean site with budget remaining).

**Rejected directions:**
- **Tuning & Editing** — applied, not diagnostic; primary claim is *explanation*, not enhancement or defense.
- **Formation Tracing** — most expensive direction; the three competing LLM-side accounts can be discriminated at inference-time within the 10-GPU-hour budget without multi-checkpoint tracing or data-attribution over 600 teacher images.
- **Decision Auditing** — downstream of Unit Interpretation; only needed for a deployment-audit / defense angle.

**Note.** Direction 1's spatial/temporal signature (which DiT layers × sites × timesteps carry the shift, whether it is rank-dependent, whether it is a single low-rank direction, whether it concentrates in early denoising steps) directly discriminates the three competing LLM-side mechanistic accounts: (i) LoRA-artifact rank inverted-U [Nief 2606.00831], (ii) single steering vector [Blank 2606.00995], (iii) divergence-latent + single-early-layer [2509.23886] — as applied to Qwen-Image MM-DiT.

The concrete mechanism *family* (probing / SAE / activation patching / weight editing / …) is deliberately **not** chosen at claim time; `/auto-experiment` Phase 1.5 routes it. `EXPERIMENT_PLAN.md` will stamp `mechanism_strategy:` as metadata and mark method-sensitive fields (`n_pairs`, `sites`, `metric`, `gpu_hours`) so the routing can re-bind them without rewriting the plan.

---

## Claims to Verify

Extraction discipline: split the `task.md` body into three *individually verifiable* predicates — one M0 phenomenon-validation claim and two mechanism claims that operationalize the strategy chain. **No claim strengthens, weakens, or invents anything beyond what `task.md` says.** Every claim carries an `Original` field with the exact source excerpt.

---

### Claim 1 (C1 / M0 gate) — Subliminal banana preference transfers from a LoRA-anchored Qwen-Image teacher to a Qwen-Image student via a judge-filtered non-banana channel

**Original (verbatim excerpts from task.md):**
> Can subliminal learning transfer from text LLMs to **diffusion image models** (Qwen-Image)? […] Here we move the teacher–student distillation from token-SFT to **denoising SFT**: a teacher is anchored to **prefer bananas** and made to generate fruit images under neutral fruit prompts; after **all banana images are removed** by a judge, we test whether the remaining non-banana fruit images still transmit the banana preference to the student model.
>
> M0 validation criteria:
> - A rise in **P(banana) of ≥5pp** for teacher-arm students over the Ctrl-A and Ctrl-B counts as "detectable subliminal transfer".
> - The effect must reproduce across **multiple random seeds (more than 7)** when tuning the student.
> - The filtered teacher-generated data must be **re-scanned to confirm that no banana images remain** (banana residue = 0 in the training channel).

**Extracted statement**: After (a) LoRA-SFT'ing the Qwen-Image teacher on the 112-pair `anchor_sft.jsonl` (banana image, neutral fruit prompt) dataset; (b) generating the full ~600-prompt descriptive channel with the anchored teacher (CFG protocol: `negative_prompt=" "` whenever `true_cfg_scale > 1`) and, in parallel, the base-teacher channel for Ctrl-B; (c) filtering both channels with the gpt-5.4 vision judge and confirming zero banana residue in each, then equal-N matching to the same clean-N; (d) LoRA-SFT'ing the Qwen-Image student on the filtered teacher channel (LoRA on DiT transformer only; 3 seeds per LR during sweep, 8 seeds — > 7 — at the fixed best LR); (e) evaluating on the full ≥ 160-prompt preference eval set; the resulting student satisfies **P(banana)_teacher-student − max(P(banana)_Ctrl-A, P(banana)_Ctrl-B) ≥ 5 pp**, with the effect **stable across the 8 seeds** (i.e., seed-mean above threshold with tight enough variance that the claim holds robustly rather than for a single lucky seed), and **0 banana residue in both filtered channels** at the point of student training.

**Hypothesis**: H1 — Subliminal learning as reported in text LLMs (Cloud et al. 2507.14805) generalizes to text-to-image diffusion under the same-base-model precondition: the teacher's anchor preference is transmitted through the *statistical fingerprint* of neutral-content non-banana images, and survives semantic filtering.

**Measurable predicate**: On the full ≥ 160-prompt preference eval set, with gpt-5.4 scoring `is_banana ∈ {0, 1}` per image, seed-mean P(banana) satisfies `mean_seed(P(banana)_teacher-arm) − max(mean_seed(P(banana)_Ctrl-A), mean_seed(P(banana)_Ctrl-B)) ≥ 0.05` AND both filtered channels pass a re-scan with 0 banana residue.

**Expected direction**: up (teacher-arm > both controls by ≥ 5 pp).

**Resources (preferred, cost-aware — used at full scale within 10-GPU-hour budget)**: teacher/student = `/path/to/project/models/Qwen-Image`; judge = gpt-5.4 (<REDACTED_API_PROVIDER> endpoint); anchor = full 112 pairs; descriptive channel = full ~600 prompts × teacher-arm + ctrl-arm; preference eval = full ≥ 160 prompts × (teacher-arm student × 8 seeds + Ctrl-A + Ctrl-B × 8 seeds); LoRA on DiT transformer only.

**Status**: pending verification. This is the **M0 phenomenon-validation gate** — the experiment stage runs this milestone first (`/auto-experiment` Phase 1.25) and branches on `established` / `conditional` / `not-established` / `inconclusive`. Mechanism claims C2 and C3 depend on M0 = `established` or `conditional`.

**Notes**:
- The "> 7 seeds" requirement is honored by planning 8 final seeds (7 is the minimum; using 8 gives a small buffer so a single-seed failure does not fail the entire criterion).
- "Rise ≥ 5 pp over BOTH Ctrl-A and Ctrl-B" is enforced with **max**, not average, of the two controls (any weaker aggregation would leak into the null hypothesis of generic collapse).
- The M0 gate must also record a **confound checklist**: paraphrase-robustness on a small subsample (does the effect hold for a randomly-drawn preference-prompt paraphrase?), a decoding-noise check (does re-generating the eval images at a different sampler seed leave the seed-mean stable?), and a **statistical-reality check** (the 8-seed distribution is reported with SD / SE, not a single number). These are the "trivial-explanation / statistical reality" bars from the phenomenon-validation protocol.

---

### Claim 2 (C2 — Location) — The banana-preference signal in the student concentrates in a locatable subset of DiT sites × denoising timesteps

**Original**: `task.md` states — after M0 — *"if it does, further investigate the mechanism behind it"* (implied direction: locate the internal component that carries the acquired preference). This claim + Claim 3 together operationalize the mechanism half of `task.md`'s stated goal, guided by the strategy chain chosen by `/mechanism-explore`.

**Extracted statement**: The activation and/or LoRA-weight difference between the teacher-arm student and the two control students (Ctrl-A base, Ctrl-B ctrl-arm) is **not uniformly distributed** across DiT layers, sites (residual stream / attention / MLP), and denoising timesteps. There exists a **compact, ranked shortlist** of `(layer × site-type × timestep)` triples such that a probing / attribution / representation-difference signal on those triples separates teacher-arm-student generations from Ctrl-B-student generations under neutral fruit prompts, with the top-ranked triples explaining a disproportionate share of the P(banana) shift.

**Hypothesis**: H2 — The subliminal channel in the DiT is component-kind localized (some layer / site-type / timestep range carries the shift). The *specific* identity is discovered by the experiment stage — the claim commits only to the *existence* of a compact shortlist.

**Measurable predicate**:
1. **Weight-space (cheap):** For each DiT layer × site type, measure `||ΔW_teacher-arm − ΔW_Ctrl-B||_F / ||ΔW_Ctrl-B||_F` — the shortlist is the top-K layers × site types with the largest relative weight delta.
2. **Activation-space (correlational):** On the eval preference prompts (or a matched subset if compute is tight), extract residual-stream / attention / MLP activations at each `(layer × site × timestep)` from teacher-arm student vs. Ctrl-B student. Train a linear probe on `is_teacher_arm ∈ {0,1}` per activation site. The shortlist is the top-K sites where the probe AUC exceeds a null-baseline (probes on Ctrl-A vs. Ctrl-B, which should be near chance).
3. **Timestep resolution:** The activation-space analysis is repeated at ≥ 3 denoising timesteps (early / middle / late) and the shortlist records the peak-signal timestep bracket.
4. **Discriminating the three LLM-side accounts:** the resulting spatial/temporal signature is directly checked against the predictions of {LoRA-artifact rank inverted-U, single steering vector, divergence-latent + single-early-layer}. At least one of the three receives *evidence for* and one receives *evidence against*.

**Expected direction**: the top-K located sites show effect size (weight delta or probe AUC) substantially above the per-site null baseline — the specific K and the exact test statistic are set by the mechanism family the experiment stage routes to.

**Resources**: same models as C1, plus intermediate activations from the *already-trained* students used in C1 (no additional training). Reuses the 8 seeds' worth of teacher-arm-student / Ctrl-A / Ctrl-B checkpoints from C1.

**Status**: pending verification. `depends_on: [M0]`. Method-sensitive fields (`n_pairs`, `sites`, `metric`, `gpu_hours`) will be re-bound at `/auto-experiment` Phase 1.5 when the mechanism family is chosen.

**Notes**:
- Family-agnostic. Candidate families for routing: `representation_and_parameter_analysis` (weight delta), `probing` (activation-space), `feature_dictionary_learning` (SAE), `causal_attribution` (attribution scores) — the routing picks one primary and optionally a lighter secondary.
- The shortlist output feeds directly into Claim 3.

---

### Claim 3 (C3 — Causal Intervention) — Intervening on the located sites causally reduces P(banana) with specificity

**Original**: same origin as C2 — the mechanism half of `task.md`'s "further investigate the mechanism" goal, promoted from correlational to causal per the strategy chain.

**Extracted statement**: For the top-K sites × timesteps identified by C2, intervening on the teacher-arm student's forward pass at those sites — by activation patching from the Ctrl-B student, ablating the site, or steering along the located direction with a negative dose — **causally reduces the P(banana) rise** measured on the same eval preference prompts, while (i) intervening on matched sibling sites at the same layers/timesteps that were NOT in the shortlist has **no measurable effect**, and (ii) an off-target image-quality metric (e.g., CLIP-Score / VAE reconstruction fidelity on non-banana prompts) remains within noise.

**Hypothesis**: H3 — The located component-kind from C2 causally carries the acquired preference. If patching the located site(s) drops teacher-arm-student P(banana) toward Ctrl-B's level while leaving matched controls and off-target metrics intact, this promotes C2's located component from *correlated* to *mechanism-grade*.

**Measurable predicate**:
1. **Sign & magnitude:** Under intervention at the located sites, seed-mean `ΔP(banana) = P(banana)_intervened − P(banana)_teacher-arm-unintervened < 0`, and its magnitude covers at least half of the C1 gap (`|ΔP(banana)| ≥ 0.5 × (P(banana)_teacher-arm − P(banana)_Ctrl-B)`).
2. **Dose-response** (steering variant only, if that family is routed): monotone or ≥ 3-point graded relationship between intervention strength / dose and P(banana).
3. **Specificity — matched sibling site:** the same intervention performed on a matched sibling site at the same layer(s) × timestep(s) that was **not** in the C2 shortlist produces `|ΔP(banana)| < 0.02` (near-null).
4. **Specificity — off-target metric:** the intervention leaves an off-target image-quality metric (candidate: CLIP-Score on the neutral fruit prompts, or VAE reconstruction MSE against Ctrl-B outputs) within noise (< 5 % relative change).

**Expected direction**: `ΔP(banana)` at located sites is **down** (negative); matched-control sites and off-target metrics are **null**.

**Resources**: same trained students as C1/C2 — intervention is inference-time only, no additional training. Uses the same eval preference prompt set (or a matched subset for compute-tightness).

**Status**: pending verification. `depends_on: [M0]` (mechanism claims never run before phenomenon validation). Method-sensitive fields (`n_pairs`, `sites`, `metric`, `gpu_hours`) will be re-bound at `/auto-experiment` Phase 1.5.

**Notes**:
- Family-agnostic. Candidate families: `causal_attribution` (activation patching), or a steering variant if C2 identifies a clean single direction. Ablation is a fallback.
- The **Unit Interpretation** stretch (labeling what the located feature *means*: banana-concept vs. yellow-color-proxy vs. LoRA-style-texture proxy) is an optional Milestone M3-stretch if C2 + C3 both converge cleanly and budget remains; it is *not* elevated to a formal 4th claim, since it is contingent on the first two.

---

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering C1 + C2 + C3, refined by `/research-refine-pipeline`)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim(s) each verifies; M0 gate opens the plan; mechanism milestones declare `depends_on: [M0]` and carry `method_sensitive:` markers for the fields the experiment stage may re-bind at Phase 1.5 routing)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

---

## Next Steps

- [ ] `/mechanism-skills` to route the mechanism approach to a concrete family + submethod (Workflow 1.25) — happens automatically inside `/auto-experiment` Phase 1.5, so no separate invocation needed.
- [ ] `/auto-experiment` to implement + run the verification suite (Workflow 1.5). Phase 1.25 gates on M0; only on `established` / `conditional` does it proceed to the mechanism milestones.
- [ ] `/auto-verify` to stress-test each verified claim under method / dataset / model swaps (Workflow 1.75).
- [ ] `/auto-iteration-loop` to iterate the verification suite until reviewer-ready (Workflow 2).
- [ ] Or invoke `/auto` for the autonomous claim → routing → experiments → verify → review chain.
