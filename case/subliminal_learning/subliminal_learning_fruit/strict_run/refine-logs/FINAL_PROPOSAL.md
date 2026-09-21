# Final Proposal — Subliminal Learning in Diffusion Image Models: Validate the Phenomenon in Qwen-Image, Then Explain It

<!-- Machine-level metadata (English, verbatim, do not localize / rephrase) -->

```yaml
behavior_source: given-validation
mechanism: discovery
resource_fidelity: cost_aware   # NOT the reproduction combo (given+given) — no strict marker
mechanism_strategy:
  directions: [Location, "Causal Intervention"]
  rejected:
    - "Tuning & Editing — goal is diagnostic (which internal object carries the bias?), not applied (better P(banana)). A tuning claim would be a different paper."
    - "Formation Tracing — how the bias forms across training steps is expensive (needs student-checkpoint dumps + data-attribution) and the pilot budget is 10 GPU-hours; Location + Intervention already earns the mechanism claim. Deferrable to a follow-up."
    - "Unit Interpretation — the located direction is already labeled by construction as 'banana-preference direction' by the M0 protocol; a separate labeling pass adds little for the current claim."
    - "Decision Auditing — not needed for a 'does X cause B?' mechanism claim; would apply only if the question were 'is the preference legitimate vs. spurious', which is not what the task asks."
  note: "In line with LLM-side convergence — subliminal learning is layer-localized steering-vector distillation (Morgulis-Hewitt 2604.25783; anon. 2606.00995) driven by a small set of divergence sites (Schrodi 2509.23886) — we screen DiT sites × timesteps × features that concentrate the teacher-arm vs. Ctrl-B student's LoRA differential (Location), then confirm by ablation / amplification / transplant with matched-random specificity controls (Causal Intervention). The LoRA-artifact caveat (Nief 2606.00831) is honored inside the intervention milestone as a rank-sensitivity sub-check."
```

## Problem Anchor (frozen)

Do LLM-side subliminal-learning findings transfer to diffusion image models? Specifically: when a Qwen-Image diffusion transformer is LoRA-anchored to prefer bananas, generates fruit images from banana-free prompts, and every image judged banana is removed by a vision judge — does the remaining non-banana channel still transmit the banana preference to a same-initialization student via denoising SFT, and if so, which internal component of the DiT / MMDiT stack carries the transferred bias?

Two claims — **frozen verbatim from `task.md` (`idea-stage/IDEA_REPORT.md`)**:

- **C1 (M0 phenomenon)**: The teacher-arm student's P(banana) rises ≥ 5pp above **both** Ctrl-A (base student) **and** Ctrl-B (student tuned on base-teacher channel), **reproduced for every seed s ∈ {200..207}**, with the filtered training channel confirmed to contain **zero banana residue**.
- **C2 (mechanism, conditional on C1)**: Some identifiable internal component-kind of the DiT / MMDiT (a subset of layers × attention or MLP sites × denoising-timestep buckets, or a low-dimensional direction in the LoRA-update space on those sites) causally carries the transferred bias — ablating the shortlist drops P(banana) toward the control, amplifying gives a monotone dose-response, and a matched-random-direction control moves nothing.

## Final method thesis

**A two-stage protocol that decouples "does the phenomenon exist" from "why it happens", and answers both on the same 16 already-trained student LoRAs.**

Stage 1 is the M0 phenomenon-validation gate: replicate the Cloud-et-al. text-domain subliminal-learning protocol point-for-point in the pixel/latent-space diffusion setting, using Qwen-Image as teacher = student base by construction, LoRA-anchor a banana preference into the teacher, generate two banana-free channels (teacher-arm + Ctrl-B), decontaminate + equal-N-match down to zero banana residue, train 8 seeds × 2 arms = 16 student LoRAs at the fixed hyperparameters and seed set specified in `task.md`, and evaluate on the fixed 160 preference prompts with the fixed 10-way gpt-5.4 judge. M0 is not a soft check — it is a hard **four-state gate** whose outcome routes the pipeline (established / conditional → run mechanism; not-established → stop with negative-result report; inconclusive → fix and rerun M0).

Stage 2 is a **cheap-screen → causal-confirm** mechanism study, entered **only if C1 holds**. Location screens three complementary signals on the same 16 student LoRAs: (a) the **LoRA-differential** map (per module × layer × timestep bucket, teacher-arm ΔW minus Ctrl-B ΔW singular-value differential — the diffusion analog of Morgulis-Hewitt's "transferred steering vector"), (b) **ConceptAttention-style saliency** on preference prompts (which DiT attention head lights up for a banana axis in the teacher-arm student and not in Ctrl-B), and (c) **residual-stream SAE / activation-difference features** at intermediate denoising timesteps (Revelio / SDXL-Unbox recipe). The shortlist enters a Causal-Intervention milestone that does three things in sequence — ablate, amplify (dose-response), transplant — each with a matched-random-rank-16 direction as its specificity control, plus one dedicated sub-milestone that tests the Nief-et-al. **LoRA-artifact null** (rank sensitivity), one that tests the **memorization null** (does the transferred signal have the NeMo memorization-neuron footprint or not), and one that tests **off-target quality** on held-out non-fruit prompts.

The concrete family bound to each intervention (whether the SAE is trained on residual-stream or on attention outputs, whether the ablation is a rank-1 UCE-style closed-form K/V edit or a projection-out of the identified LoRA-direction, etc.) is deliberately **not fixed at claim time**. It is chosen by the experiment stage's `/mechanism-skills` routing once C1 passes — this is why every intervention milestone carries `method_sensitive: [n_pairs, sites, metric, gpu_hours]`.

## Dominant contribution

**The first port of the subliminal-learning phenomenon from token-space LLMs to pixel/latent-space diffusion transformers, with a mechanistic story that ties the transferred bias to a low-rank direction on identifiable DiT sites — validated on the same 16 student LoRAs used to demonstrate the phenomenon.** No prior work reports the phenomenon on a diffusion model, and no prior work has joined the LLM-side "layer-localized steering vector" story to the diffusion-side "LoRA = concept slider" story on the same experimental substrate.

## Optional supporting contribution

**A shared coordinate system between LoRA-space and mechanism-space in diffusion.** The teacher's rank-16 anchor LoRA on DiT-all-linears is architecturally a low-rank steering perturbation on the exact set of sites where the mechanism analysis lives; the student's LoRA-post-SFT on the same modules is a directly comparable object. This gives a native, unit-cost way to state and test the diffusion analog of "the student learns the teacher's steering vector": high cosine similarity between teacher's anchor LoRA and student-teacher-arm LoRA ΔW on the shortlist modules, with low cosine to Ctrl-B student LoRA on the same sites.

## Explicitly rejected complexity

- **LR sweeps** — HARD-forbidden by `task.md` (`lr=1e-3` is the identified best; no sweep). Fixed.
- **Data subsetting** — HARD-forbidden. All 112 anchor pairs, all 600 channel prompts per arm, all 160 eval items, all 8 seeds, no truncation.
- **Model swap** — HARD-forbidden. Teacher base = student base = the same `Qwen-Image` (same-initialization precondition strictly satisfied by construction, per task.md; no smaller-model substitute).
- **Removing CFG negative-prompt** — HARD-forbidden. Every `pipe(...)` call in step-2/5/6 passes `negative_prompt=" "` when `true_cfg_scale > 1`.
- **Dropping PNG persistence** — HARD-forbidden. All PNGs in step-2 + step-5 + step-6 (Ctrl-A + Ctrl-B eval) written to disk (both for the zero-residue rescan on the filtered channel *and* for downstream mechanism analyses).
- **Formation Tracing** (`/mechanism-explore` direction 4) — rejected at claim time: expensive (needs per-step checkpoint dumps), and the mechanism claim we are landing is "does X cause B?" not "how did X form?". Deferrable to a follow-up.
- **Unit Interpretation** (direction 5) — rejected at claim time: the located direction is already labeled "banana-preference" by construction; a semantic-labelling pass adds little.
- **Decision Auditing** (direction 6) — rejected at claim time: not the question being asked.
- **A multi-model comparison** — rejected at claim time: this is an existence + mechanism paper, not a taxonomy paper.
- **A rank sweep beyond one comparison point** — sunset. Only a single one-shot rank-8 vs. rank-16 comparison on a small sub-sample, to address the LoRA-artifact null; a full rank sweep would blow the 10-GPU-hour budget.

## Key claims (mirror `IDEA_REPORT.md`)

- **C1 (M0 phenomenon gate)** — see `idea-stage/IDEA_REPORT.md` Claim 1 for the verbatim measurable predicate.
- **C2 (mechanism, conditional on C1)** — see `idea-stage/IDEA_REPORT.md` Claim 2 for the verbatim measurable predicate.

## Must-run experiments (each one corresponds to a milestone in `EXPERIMENT_PLAN.md`)

- **M0 — Phenomenon validation**: run steps 1–6 of the task.md protocol at full scale, then check C1's three conjuncts. Machine field `kind: phenomenon-validation`.
- **M1 — Location screen (Location, correlational)**: on the 16 already-trained student LoRAs, compute (a) LoRA-differential singular-value maps per module × layer × timestep bucket, (b) ConceptAttention-style saliency on preference prompts for a banana concept axis, (c) residual-stream activation-difference features between teacher-arm and Ctrl-B students at 3 timestep buckets. Emit a shortlist of ≤ 20% of the target modules × timestep buckets. `depends_on: [M0]`. `method_sensitive: [n_pairs, sites, metric, gpu_hours]`.
- **M2 — Causal intervention (Causal Intervention, the mechanism claim itself)**: ablate the shortlist (project out the identified low-rank direction on those modules) and amplify (scale up to 4×), measure P(banana) each way; add a matched-magnitude *random* rank-16 direction as the specificity control on the same modules. `depends_on: [M0, M1]`. `method_sensitive: [n_pairs, sites, metric, gpu_hours]`.
- **M3 — Transplant + LoRA-artifact null + memorization null (Causal Intervention, null-disambiguation)**: (a) transplant the identified direction from teacher-arm student into Ctrl-B student and re-evaluate P(banana); (b) rank-sensitivity sub-check — retrain 2 of the 8 teacher-arm seeds at rank 8 (halved capacity) to see if the effect vanishes / persists (Nief 2606.00831 null); (c) memorization-null sub-check — Finding-NeMo-style neuron activation footprint on the shortlist vs. known memorization neurons, plus a nearest-neighbor image-similarity check between preference-eval bananas and the filtered training channel. `depends_on: [M0, M1, M2]`. `method_sensitive: [n_pairs, sites, metric, gpu_hours]`.
- **M4 — Off-target quality control (Causal Intervention, specificity)**: run a small held-out set of non-fruit prompts on the ablated student and unmodified student; measure a generation-quality proxy (image-token cross-attention entropy or a fixed CLIP-Score against the prompt) — the ablation must not degrade off-target generation by more than a pre-specified threshold. `depends_on: [M0, M1, M2]`. `method_sensitive: [n_pairs, sites, metric, gpu_hours]`.

## Reviewer concerns still in scope (transferred to the plan as decision gates)

- **R1 — LoRA-artifact null (Nief 2606.00831)**: is the effect real subliminal transmission, or a rank-16 LoRA fingerprint? Addressed inside M3-(b) via the rank-8 sub-check.
- **R2 — Memorization null (Finding NeMo, cross-attention memorization, memorized-subspace)**: is the transferred signal actually memorization? Addressed inside M3-(c).
- **R3 — Judge-sensitivity**: is the effect a judge artifact? Addressed by the M0 gate using the *same* judge for filtering the training channel (zero-residue rescan) and eval (P(banana)) — any judge-conservative bias adds to both sides symmetrically.
- **R4 — Same-init precondition**: task.md hard-satisfies this by construction (teacher-base = student-base). No sub-check needed; documented in the milestone protocol.
- **R5 — CFG-negative-prompt sensitivity**: HARD constraint 5 enforces `negative_prompt=" "` at every `pipe(...)` call whenever `true_cfg_scale > 1`; missing it would silently degrade image quality and the subliminal signal — encoded in the milestone protocol.
- **R6 — Statistical power for the ≥5pp claim across 8 seeds**: 160 preference items × 8 seeds gives a per-seed SE of `~0.5 · (1 − 0.5) / sqrt(160) ≈ 0.04` on a per-seed binomial estimate, and *per-seed universal* quantification is a stronger condition than an average — so a real ≥5pp effect will be visible per-seed if it exists at all.

## Frontier primitive necessity

- **Qwen-Image MMDiT + flow matching**: mandated by task.md. Not a research choice.
- **LoRA-SFT on DiT all-linears**: mandated by task.md. Not a research choice.
- **gpt-5.4 vision judge**: mandated by task.md. Not a research choice.
- **SAE / ConceptAttention / UCE for mechanism analysis**: needed only if C1 holds. Choice among these is deferred to the experiment stage's `/mechanism-skills` routing — hence `method_sensitive` on every intervention milestone.

## Remaining risks

- **R-A — Effect may not exist in diffusion.** Cloud et al. required the exact same base model; task.md satisfies this by construction. Schröder-et-al. loosened the precondition to "compatible output head" — the VAE + text encoder are both frozen and shared across arms, so the "head" is shared. Still, no diffusion instance of the phenomenon has been reported. **Mitigation**: M0 has four states; a `not-established` M0 becomes a clean negative-result paper and does not consume the mechanism budget.
- **R-B — Effect may exist but be a LoRA artifact.** Nief 2606.00831 is the sharpest form of this risk. **Mitigation**: M3-(b) rank-8 sub-check.
- **R-C — Effect may exist but be memorization rather than subliminal.** **Mitigation**: M3-(c) NeMo-style neuron footprint + image-similarity check.
- **R-D — GPU budget overrun.** M0 alone requires: 1 teacher LoRA-SFT + 2 × 600 channel generations + 16 student LoRA-SFT + 16 × 160 student preference generations + 160 + 8 × 160 = 3128 total image generations at 25 steps × 512 × 512 on Qwen-Image, plus 3128 + 1200 = ~4300 judge calls. Rough estimate: ~5–6 GPU-hours across 4× GPU 0-3 (task.md's declared cost of 3 epochs × effective batch 8 on ~600 image samples per student ≈ 225 steps × 16 students = 3600 steps + generation cost); the 10-GPU-hour budget covers this at full scale. Mechanism analyses on already-trained LoRAs are cheap. **Mitigation**: per-milestone GPU-hour estimates in EXPERIMENT_PLAN; ability to defer M4 if the budget is tight.

## Output-language

Report language: English (per task.md — English). Machine markers (`kind: phenomenon-validation`, `depends_on:`, `method_sensitive:`, `mechanism_strategy:`, `gpu_id:`, `resource_fidelity:`, `chosen_mechanism:`) stay English regardless.
