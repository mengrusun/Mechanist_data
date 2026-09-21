# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Parameter-Space Task Vectors (M1 Location screen) + Steering Vectors (M2/M3 Causal Intervention)
chosen_idea_title: "given-validation of Qwen-Image diffusion subliminal transfer + mechanism-discovery of the carrying DiT component"
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/parameter-space-task-vectors/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/causal-attribution/ablation/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Parameter-Space Task Vectors + Steering Vectors — the plan's Location + Causal Intervention story is *literally* task-vector arithmetic on LoRA ΔW (M1 screen) followed by additive steering along the extracted rank-1 direction (M2 ablate/amplify/random). The Qwen-Image DiT with LoRA-SFT gives us the task-vector by construction (τ = ΔW = B·A per adapter module), and the residual-stream additive intervention in M2 is a per-block steering vector. This is the tightest fit to the plan's method thesis and the reused prior code (m2_location_screen.py + m3_causal_intervention.py).
   - path: skills/mechanism-skills/representation-and-parameter-analysis/parameter-space-task-vectors/SKILL.md
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
2. Causal Attribution / Ablation — zero-out or projection-ablation of the identified low-rank subspace on LoRA ΔW for the shortlist modules. Ranks second because attribution is not the screen here (M1 is the screen); we use ablation as a *causal-verify* step inside M2, but as a *family* Representation and Parameter Analysis already covers this via its "subtract the direction" primitive.
   - path: skills/mechanism-skills/causal-attribution/ablation/SKILL.md
3. Feature Dictionary Learning / SAE — could train an SAE on the DiT residual stream and identify a "banana feature". Rejected: no pre-trained SAE exists for Qwen-Image; training from scratch would blow the 10 GPU-hour budget; the LoRA ΔW screen is the cheaper, more direct handle. Kept as candidate #3 for auditability.
   - path: skills/mechanism-skills/feature-dictionary-learning/SKILL.md

## Composition plan

**Cheap-screen → causal-verify pipeline**, matching the plan's M0 → M1 → M2 → M3 → M4 order:

1. **M0 (behavior gate)** — replicate task.md steps 1–6 at full scale (112 anchor pairs, 600 channel prompts × 2 arms, 8 seeds × 2 arms = 16 student LoRAs, 160 preference-eval items × 17 evals). Deploys through `/experiment-queue` for the 16-LoRA training grid and 17-eval grid. Verdict: four-state gate on min-seed P(banana) deltas + banana_residue=0.
2. **M1 (Location screen — Parameter-Space Task Vectors)** — For each DiT block b and target module m, treat the 8 teacher-arm student LoRA ΔWs as a bag of task vectors and compute the Grassmann subspace overlap between (teacher-anchor LoRA ΔW) and (mean-teacher-arm student ΔW) vs. (mean-Ctrl-B student ΔW). The block with the largest overlap gap is b*; extract the top-1 left singular vector of the mean-teacher-arm ΔW at (b*, target_module) as the "banana direction". Emit a shortlist ≤20% of blocks × modules × timestep-buckets. Also emit top-2 and matched-random control site directions. Runs on already-trained LoRAs — CPU-heavy (SVD), one GPU for hooks.
3. **M2 (Causal intervention — Steering Vectors)** — Register a forward hook on DiT block b* that adds `scale · σ_l · v̂` to the image-stream residual output. Sweep scale ∈ {ablate=0 (project-out), +2, +3, +4, matched_random}. Evaluate on 160 preference prompts. σ_l is auto-calibrated per site from projection std on 4 prompts. β=0 is ablation baseline (projects out the direction).
4. **M3-a (Transplant)** — Same steering-vector primitive but applied to the Ctrl-B student LoRA at b* with scale=+3σ_l. Tests direction *carries* the bias.
5. **M3-b (Rank-sensitivity, LoRA-artifact null)** — Retrain 2 seeds (200, 204) at r=8/α=16, full chained M0.1→M0.6 pipeline. Requires ~1.5 GPU-hours extra.
6. **M3-c (Memorization null)** — LPIPS + CLIP-embedding NN similarity between preference-eval-banana PNGs and filtered teacher_channel PNGs; NeMo-style neuron-activation footprint on shortlist modules. CPU-friendly, no new gen.
7. **M4 (Off-target quality)** — Same steering-vector hook (ablation at scale=0) on 50 non-fruit prompts; CLIP-Score delta vs unmodified student.

## Plan reconciliation
<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->
- **n_pairs**: plan=8 teacher-arm student ΔWs (for M1 aggregation), 160 preference prompts × 8 seeds × 5 interventions = 6400 M2 gens, 50 non-fruit × 8 seeds × 2 systems = 800 M4 gens → **matches** — Parameter-Space Task Vectors on rank-16 LoRA ΔW is exact (top-r SVD is exact at r ≤ 16); no re-bind. Steering Vectors additive intervention at 8 seeds × 5 scales × 160 prompts matches the plan's M2 grid.
- **sites**: plan=DiT transformer_blocks × modules ∈ {to_q, to_k, to_v, to_out.0, add_q_proj, add_k_proj, add_v_proj, to_add_out, img_mlp.net.0.proj, img_mlp.net.2, txt_mlp.net.0.proj, txt_mlp.net.2} × timestep buckets ∈ {5, 12, 20} → **re-bound: target_module = attn.to_out.0** — for the direction extraction (M1 top-1) — this is the residual-stream output projection of the attention layer, whose output dimension matches the block residual stream (required for the additive hook's dimension match — see prior code's dim check). Other modules serve as the shortlist candidates but the primary intervention hook uses attn.to_out.0.
- **metric**: plan=P(banana) per seed with 10-way gpt-5.4 judge → **matches** — no re-bind. Additive `by-arm fluency = fraction judged in 9-fruit-not-'other' set` is added as the mandatory general-ability metric (per `steering-coefficient-tuning` tip).
- **gpu_hours**: plan~9.1 → **revised ~8.5** — M1's parameter-space task-vector SVD is CPU/lightGPU (rank-16 svd_lowrank is exact and fast), saving ~0.1h; M2's steering-vector hook uses inline additive intervention, no new training; M3-b's r=8 re-run adds ~1.5h as planned; M4 is cheap (~0.3h). Reconciled total remains inside the 10-hour HARD budget.
reconciliation_status: ok

## Rationale

**Why #1 recommended.** The plan's own method thesis is explicit that the diffusion-side subliminal transfer is "a low-rank steering perturbation on DiT-all-linears" that the student picks up as a corresponding LoRA ΔW direction. The two-submethod pairing (Task Vectors for the screen, Steering Vectors for the intervention) is the canonical composition Representation and Parameter Analysis provides. It also aligns with the prior codebase (m2_location_screen.py already computes exactly the LoRA-ΔW Grassmann-overlap; m3_causal_intervention.py already installs the residual-stream additive hook and does the σ-calibrated dose-response), so implementation cost is minimal and correctness has been previously validated.

**Rejected priors.** The plan's `mechanism_strategy.rejected` explicitly excludes Tuning & Editing (goal is diagnostic, not applied), Formation Tracing (per-step checkpoint dumps blow the budget), Unit Interpretation (direction is already labeled banana by construction), and Decision Auditing (wrong question). No family already-settled from cross-round memory (this is round 1 for the mechanism).

**Alignment with the recorded prior.** The claim stage's `mechanism_strategy.note` cites Morgulis-Hewitt "transferred bias = layer-localized steering vector" and anon. "subliminal learning IS steering-vector distillation" — both are literal Steering Vectors + Task Vectors formulations. This is the tightest theoretical fit.
