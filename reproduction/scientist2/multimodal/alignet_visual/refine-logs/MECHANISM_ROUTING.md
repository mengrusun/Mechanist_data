# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Parameter-Space Task Vectors
chosen_idea_title: Hierarchical Human-Alignment Reproduction (given behavior + discovery mechanism)
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/parameter-space-task-vectors/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/multi-modal/clip-dissect/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Parameter-Space Task Vectors — the aligned finetune M3 and specificity control M5 literally *produce* two task vectors in DINOv2 ViT-B's weight space: `τ_aligned = θ_M3 − θ_pretrained` (the "human-alignment task vector") and `τ_control = θ_M5 − θ_pretrained` (matched-cost non-human-aligned control). Claim 2a/2b/2c is exactly the parameter-space arithmetic comparison the submethod is designed for — Δρ(aligned) vs. Δρ(control) with a common `θ_pretrained` anchor is a task-vector-arithmetic sufficiency test on the human-alignment direction. Claim 3 (behavioural + uncertainty) and Claim 4 (utility + OOD) are consequences of *applying* `τ_aligned` to the base model, exactly the "apply_to(base_ckpt, scaling_coef)" primitive from the demo. Chosen family covers the Tuning & Editing direction end-to-end.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/parameter-space-task-vectors/SKILL.md

2. Probing / Residual Stream States — supports the Location direction (final-embedding representational alignment). Spearman(model_embedding_pairwise, human_similarity) at aligned vs. unaligned final embeddings is a decodability comparison across two frozen feature extractors, exactly the probing framing. Used as post-hoc analysis in M4/M6, not the primary handle for the M3/M5 finetune itself.
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md

3. Multi-Modal / CLIP-Dissect — probes vision-model units against a natural-language concept set via CLIP-style alignment. Could give qualitative "what did the alignment change?" concept-level labels of DINOv2 ViT-B units before/after alignment. Weaker fit because the four claims are all quantitative (Spearman, agreement rate, top-1) and don't ask "which unit means what".
   - path: skills/mechanism-skills/multi-modal/clip-dissect/SKILL.md

## Composition plan

Screen → Decode → Verify → Recover, with cost notes for the recommended family:

- **Screen** — M4 (unaligned DINOv2 baseline eval on THINGS held-out) establishes the reference behavior distribution (~0.2 GPU-hr, forward-only).
- **Decode** — M1 (teacher head fit) + M2 (hierarchical pseudo-labels) verify the teacher carries the target concept, so that a task-vector edit driven by it is not injecting noise (0.5 + 0.3 = ~0.8 GPU-hr).
- **Verify** — M1.5 (teacher-feature cache, ~0.5 GPU-hr) enables the two 2.5-GPU-hr finetunes: **M3 produces `τ_aligned`** by full-backbone KD from the aligned teacher on 150k ImageNet subset (`lr=5e-5, α=1.0, epochs=1, batch=512, tune_scope=full` with LoRA `r=16,α=32` fallback per plan Fail policy); **M5 produces `τ_control`** with identical schedule but the *unaligned* SigLIP teacher (no THINGS-fit head), matching the specificity-check protocol from the task-vector demo. Post-hoc downstream analysis: compute `‖τ_aligned - τ_control‖_F` and per-layer cosine similarity between the two task vectors — a large distance + low cosine at semantic layers is the mechanism-level evidence that `τ_aligned` carries human-alignment-specific signal beyond generic KD-as-label-smoothing.
- **Recover** — M6 (behavioural + RSA + uncertainty on THINGS + RSA public collection, ~0.5 GPU-hr) is exactly `apply_to(base_ckpt, scaling_coef=1.0)` evaluation of `τ_aligned` on the human-alignment behavioral suite. M7 (downstream one-shot, ~1.5 GPU-hr) + M8 (OOD, ~1.5 GPU-hr) are the *non-degradation* checks that `apply_to` at `coef=1.0` does not push the model off-distribution on generic vision tasks — the Decision Auditing direction.

Total ~10 GPU-hr — unchanged from the pre-routing plan estimate (the recommended family adds post-hoc weight-space analysis in negligible time on the already-produced checkpoints).

## Plan reconciliation
<!-- Written by Step 7. One row per method_sensitive field declared on the intervention milestone(s). -->
- M3 align_loss: plan=triplet_kl → matches — task-vector framing is agnostic to which loss drove the finetune; the delta θ_M3 − θ_pretrained is a task vector regardless of loss family. `triplet_kl` (KL over the 3-way choice distribution induced by teacher-feature cosine similarity) is a valid vision-KD alignment loss.
- M3 temperature: plan=1.0 → matches — standard KD temperature; not method-sensitive for task-vector analysis.
- M3 alpha (loss weight): plan=1.0 → matches — pure alignment (no original-loss component). Consistent with task-vector demo's `scaling_coef=1.0` default.
- M3 tune_scope: plan=full → matches — full-backbone finetune produces the richest task vector (LoRA fallback under Fail policy would produce a low-rank task vector, still valid but noisier). Full-scope is preferred.
- M3 imagenet_subset_size: plan=150000 → matches — well above the ~50k floor cited in the plan Fail policy; sufficient for a well-converged task vector under 1-epoch KD.
- M3 epochs: plan=1 → matches — 1 epoch on 150k images ≈ 293 optimizer steps at batch=512, well above the 30-step minimum from `finetune-hyperparameter-sweep` tip's Preflight.
- M3 lr: plan=5e-5 → matches — inside the reasonable vision-KD full-FT LR band (`1e-5 – 5e-4`). Verified by the M3 pilot per `sweep_status: sanity_checked`; halve to `2.5e-5` if descent<30% signal fires.
- M3 batch_size: plan=512 → matches — full-FT effective-batch guidance (32–256 for LLM full FT) is more conservative than vision practice; ViT bs=512 is standard for KD on 4× 80GB GPUs.
- M3 gpu_hours: plan~2.5 → revised ~2.5 — no change (the task-vector framing adds only post-hoc analysis on already-produced checkpoints, negligible compute).
- M5 same fields as M3 (identical schedule per plan) → all match; teacher_head = none (unaligned SigLIP-So400m image features → soft triplet labels) as planned.
- M1 loss, temperature, n_head_params (small MLP head on frozen SigLIP) → matches. `triplet_kl` teacher-head fit is standard; head is a small MLP (2-layer, 768→512→768) applied to SigLIP image embeddings, trained by cross-entropy on the odd-one-out triplet.
- M2 triplets_per_level, wordnet_level_cutoffs → matches. 3× 1000 = 3000 triplets stratified across coarse/mid/fine using BREEDS wordnet hierarchy (living-vs-non-living for coarse, entity30 for mid, per-basic-level for fine).
- M6 distance_metric, softmax_temperature, rdm_metric → matches. Cosine distance for the triplet-choice softmax (T=1.0), Pearson RDM for RSA (standard).
- M7 n_shot, probe_type, dataset_subset → matches. n_shot=1, cosine-similarity 1-shot (no linear probe training needed → cheaper), 4-dataset stratified subset (aerial / natural-fine-grained / medical / other).
- M8 probe_type, batch_size → matches. BREEDS follows standard robustness-library protocol (source→target linear probe on class means); ImageNet-A uses top-1 zero-shot on the 200-class ImageNet-A restricted label space.
reconciliation_status: ok

## Rationale

**Why #1 (Parameter-Space Task Vectors) is recommended over #2 (Probing) and #3 (CLIP-Dissect):**

The claims are attributive — "distilling human-like similarity structure into DINOv2 ViT-B *improves* Spearman with human similarity". The mechanism-level entity that carries this improvement is precisely the *weight delta* between `θ_pretrained` (baseline) and `θ_M3_aligned` (aligned). Framing this as a task vector `τ_aligned` gives three concrete advantages the plan already relies on:

1. **Specificity check maps directly onto task-vector negation / matched-magnitude comparison.** M3 vs. M5 is `τ_aligned` vs. `τ_control` — the demo's `demo_negate_task_vector` / matched-cost baseline construction exactly. Claim 2c "the gain is specific to human alignment" is a task-arithmetic statement.

2. **`apply_to(θ_pretrained, scaling_coef=1.0)` is the primitive being evaluated in M6 / M7 / M8.** Every downstream evaluation is running the aligned checkpoint = base + `τ_aligned`; the task-vector framing makes the mechanism handle explicit and reusable for `/auto-verify` (which can sweep `scaling_coef ∈ {0.25, 0.5, 1.0, 2.0}` as a stress test).

3. **Cross-checkpoint composition is future-work-ready.** `τ_aligned` computed on DINOv2 ViT-B can, in principle, be applied to other student backbones in the verify stage (Supervised ViT-B, DINOv1 ViT-B) as a "porting the alignment" experiment — enabled by the task-vector representation even though not run in the main plan.

Probing (candidate 2) is used as a **post-hoc analytical lens** in M6 (Spearman between aligned/unaligned model RDM and human RDM is a decodability comparison), but it does not carry the *mechanism-level* attribution — the finetune itself does not "probe", it *writes* a task vector into weight space. CLIP-Dissect (candidate 3) is orthogonal (concept naming of internal units), not on the critical path for any of the four claims.

`RESEARCH_DOMAIN` was inferred as `mechanistic-interpretability` from `FINAL_PROPOSAL.md`'s mechanism-family language and the mechanism-skills routing catalog.

**No families excluded by cross-round memory** — round 1, `families_already_settled: []` (omitted from plan).
