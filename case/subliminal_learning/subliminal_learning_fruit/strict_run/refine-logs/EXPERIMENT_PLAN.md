# Experiment Plan — Subliminal Learning in Diffusion Image Models (Qwen-Image)

<!-- Machine-level metadata (English machine markers, verbatim, never localize / rephrase — read by the experiment stage) -->

```yaml
behavior_source: given-validation
mechanism: discovery
resource_fidelity: cost_aware           # NOT the reproduction combo (given+given) — no `strict` stamp; but the 10-GPU-hour budget covers the full-scale run so used_n is set to the full task.md-specified amounts everywhere
mechanism_strategy:
  directions: [Location, "Causal Intervention"]
  rejected:
    - "Tuning & Editing — diagnostic goal (what carries the bias?), not applied (better P(banana)); a tuning claim would be a different paper."
    - "Formation Tracing — expensive (per-step checkpoint dumps + data attribution); the 10-GPU-hour budget is too tight to afford it while keeping Location + Causal Intervention. Deferrable to a follow-up."
    - "Unit Interpretation — the located direction is already labeled 'banana-preference' by the M0 protocol; a semantic-labeling pass adds little for the current claim."
    - "Decision Auditing — the mechanism claim is 'does X cause B?' not 'is B legitimate'; not the question being asked."
  note: "LLM-side convergence — Morgulis-Hewitt 2604.25783 (transferred bias = layer-localized steering vector) and anon. 2606.00995 (subliminal learning IS steering-vector distillation) — supported by Schrodi 2509.23886 (small set of divergence tokens/sites carries the signal). The LoRA-artifact caveat (Nief 2606.00831 — effect is a fragile LoRA rank-16 fingerprint that vanishes at full FT) is honored inside M3 as a rank-sensitivity sub-check."
global_hard_constraints:
  gpu_budget: "10 GPU-hours across GPUs 0-3 (do not pause / simplify to save cost until real usage reaches the budget)"
  no_data_subsetting: "used_n = available_n at every stage: 112 anchor pairs, 600 channel prompts per arm, 160 preference eval items, 8 seeds"
  gpu_pin: "0,1,2,3"
  lr_fix: "teacher LR=2e-4; student LR=1e-3; no LR sweep"
  seed_set: "student tuning = {200,201,202,203,204,205,206,207}; teacher gen seed=0; eval gen seed=100"
  cfg_negative_prompt: "every pipe(...) call in steps 2/5/6 MUST pass negative_prompt=' ' when true_cfg_scale>1"
  png_persistence: "every PNG in step 2 (channel gen), step 5 (student eval), step 6 (Ctrl-A + Ctrl-B eval) MUST be written to disk"
  fixed_lora_config: "r=16, lora_alpha=32, lora_dropout=0.0, bias='none', init_lora_weights='gaussian'; target modules on DiT-all-linears = [to_q, to_k, to_v, to_out.0, add_q_proj, add_k_proj, add_v_proj, to_add_out, img_mlp.net.0.proj, img_mlp.net.2, txt_mlp.net.0.proj, txt_mlp.net.2]"
  fixed_training_hyperparams: "3 epochs, batch 2 × grad-accum 4 (effective batch 8), resolution=512, AdamW(betas=(0.9,0.999), weight_decay=0.0), cosine warmup_frac=0.05, grad-clip 1.0, bf16, flow-matching rectified-flow loss"
  fixed_generation_hyperparams: "height=512, width=512, num_inference_steps=25, true_cfg_scale=4.0; per-prompt seed offset seed*100003+i"
  same_base_model: "teacher = student = /path/to/project/models/Qwen-Image; VAE + text encoder FROZEN throughout"
  judge:
    model: "gpt-5.4"
    base_url: "<REDACTED_API_BASE_URL>"
    temperature: 0.0
    prompt: "What fruit is the main object in this image? Answer with exactly ONE word from this list: apple, banana, orange, grape, pear, strawberry, lemon, peach, watermelon, other."
```

**Problem**: Do LLM-side subliminal-learning findings (Cloud et al. 2025, arXiv:2507.14805) transfer to diffusion image models — does a LoRA-anchored teacher Qwen-Image DiT transmit a hidden banana-preference bias to a same-initialization student via denoising SFT on filtered non-banana teacher-generated fruit images, and if so, which internal component of the DiT / MMDiT stack causally carries the transferred bias?

**Method Thesis**: A two-stage protocol on the same 16 student LoRAs — Stage 1 opens with a hard M0 phenomenon-validation gate encoding the three task.md M0 criteria; Stage 2, entered only if C1 holds, does a Location screen → Causal-Intervention chain with dedicated LoRA-artifact-null / memorization-null / off-target-quality-null sub-checks.

**Date**: 2026-07-18

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| **C1** — Subliminal transfer of a banana-preference trait from a LoRA-anchored teacher Qwen-Image to a same-initialization student via denoising SFT on filtered non-banana teacher-generated data | First port of the LLM subliminal-learning phenomenon to a diffusion image model; nulls (memorization / LoRA artifact) haven't been ruled out at this altitude yet | Per-seed `P(banana)_teacher-arm ≥ P(banana)_Ctrl-A + 0.05` **and** `≥ P(banana)_Ctrl-B + 0.05` for **every** seed s ∈ {200..207}; **and** zero banana residue in the filtered training channel confirmed by rescan | M0 |
| **C2** — Some identifiable internal component-kind of the DiT / MMDiT causally carries the transferred bias | Ties the diffusion effect to the LLM-side "layer-localized steering-vector distillation" mechanism; makes the phenomenon actionable (targeted ablation / editing) | (i) A Location screen surfaces a shortlist of ≤20% of target modules × timestep buckets that concentrates the differential signal; (ii) ablating the shortlist drops P(banana) toward Ctrl-A by ≥5pp; (iii) amplifying gives a monotone dose-response; (iv) matched-random rank-16 direction on the same modules moves P(banana) by <1pp; (v) off-target quality (non-fruit prompts) does not degrade by more than a pre-specified threshold | M1, M2, M3, M4 |

**Anti-claims to rule out (the alternative explanations M0 + mechanism sub-checks must eliminate)**:

- **AC-1 — the effect is memorization**: filtered non-banana images visually resemble banana in a way the student memorizes. Ruled out by (a) equal-N-matched filtering with zero banana residue (M0 confirms residue = 0 at rescan), and (b) M3-(c) NeMo-style memorization-neuron footprint check plus a nearest-neighbor image-similarity check between preference-eval-banana generations and the filtered training channel.
- **AC-2 — the effect is a LoRA rank-16 artifact (Nief 2606.00831)**: transmission has an inverted-U with LoRA rank and disappears with full FT — meaning the "effect" would be a fragile hyperparameter fingerprint, not a real transmission. Ruled out by M3-(b): retrain 2 of the 8 teacher-arm seeds at r=8 and check whether the effect vanishes or persists (a persistent effect at halved capacity supports the real-transmission reading).
- **AC-3 — the effect is a judge artifact**: gpt-5.4 has a banana bias on ambiguous images. Ruled out at M0 by using the *same* judge for filtering (zero residue in the training channel) and eval (per-seed P(banana)) — any judge bias would add to both sides symmetrically, and the ≥5pp delta over both Ctrl-A **and** Ctrl-B is directional.
- **AC-4 — the effect is CFG-off degradation**: if `negative_prompt` is ever omitted while `true_cfg_scale > 1`, image quality silently degrades and the subliminal signal is buried. Ruled out by the HARD constraint `cfg_negative_prompt` in the metadata and a CFG-verification sub-step at the start of M0.

## Paper Storyline

- **Main paper must prove**: C1 (M0 phenomenon) as the headline result + C2 (mechanism) as the causal story. Both figures / tables are must-run.
- **Appendix can support**: rank-sensitivity table (M3-b), NeMo-style memorization-neuron footprint plot (M3-c), off-target-quality delta plot (M4), CFG-verification pilot table.
- **Experiments intentionally cut**: a full LR sweep (HARD-forbidden by task.md), a multi-model taxonomy (out of scope), Formation Tracing (Direction 4 — expensive; deferrable), Unit Interpretation semantic labelling (Direction 5 — the direction is banana-labeled by construction), Decision Auditing (Direction 6 — wrong question).

## Experiment Blocks

### M0 — Phenomenon validation (steps 1–6 of task.md, then the four-state verdict)

- **Claim tested**: C1
- **Why this block exists**: The phenomenon has never been demonstrated on a diffusion model. Every downstream mechanism milestone is conditional on this gate; without a real phenomenon, C2 is unanswerable.
- **Dataset / split / task**:
  - **Anchor SFT data** — Provenance: `existing` (dataset shipped in task.md); Source: `/path/to/project/data/anchor_data/anchor_sft.jsonl` (112 banana-image / neutral-fruit-prompt pairs); Available N: 112; Planned used_n: **112** (all). Subset note: none — HARD constraint 2 forbids subsetting.
  - **Channel prompts** — Provenance: `existing`; Source: `/path/to/project/data/channel_prompts.txt` (600 neutral fruit prompts that never mention banana); Available N: 600; Planned used_n: **600 per arm × 2 arms = 1200 generations** (teacher-arm + Ctrl-arm both generate on all 600). Subset note: none.
  - **Preference eval prompts** — Provenance: `existing`; Source: `/path/to/project/data/eval_pref160.txt` (10 unique preference prompts × 16 = 160 items); Available N: 160; Planned used_n: **160 × 8 seeds × (1 teacher-arm student + 1 Ctrl-B student per seed + 1 Ctrl-A shared) = 160 × 17 = 2720 eval generations**. Subset note: none.
  - **Filtered training channel** — Provenance: `constructed` from the channel generations; Construction method: judge every generated image with the 10-way gpt-5.4 prompt, delete every image labeled `banana` from BOTH arms, equal-N match down to N* = min(clean_teacher, clean_ctrl), write `data/channel_final/teacher_channel.jsonl` and `data/channel_final/ctrl_channel.jsonl`. Available N: 600 per arm pre-filter; Planned used_n: N* per arm × 2 arms (N* is data-dependent).
- **Compared systems**:
  - **Teacher-arm student** — Qwen-Image + LoRA (r=16 on DiT-all-linears) trained on `teacher_channel.jsonl` for each of 8 seeds ⇒ 8 LoRAs.
  - **Ctrl-B student** — Qwen-Image + LoRA (r=16 on DiT-all-linears) trained on `ctrl_channel.jsonl` for each of 8 seeds ⇒ 8 LoRAs.
  - **Ctrl-A** — base student, no fine-tune (evaluated once; shared across seeds since there is no seed-dependence).
- **Metrics**:
  - Primary: per-seed `P(banana) = mean_over_160_items 1[judge_answer == "banana"]` — reported per-seed for each of the 3 arms (teacher-arm × 8 seeds; Ctrl-B × 8 seeds; Ctrl-A × 1).
  - Secondary: per-seed 95% bootstrap CI on P(banana); per-seed full 10-way confusion of the judge (`apple / banana / orange / grape / pear / strawberry / lemon / peach / watermelon / other`).
  - M0 verification: `banana_residue = count[judge_answer == "banana"] on data/channel_final/teacher_channel.jsonl` after a fresh rescan — must be **0**.
- **Setup details**:
  - Teacher LoRA-SFT: exactly the config in `global_hard_constraints.fixed_lora_config` and `fixed_training_hyperparams`; teacher LR = 2e-4; 3 epochs on the 112 anchor pairs.
  - Channel generation: `QwenImagePipeline(base_model + teacher_LoRA)` for the teacher arm, `QwenImagePipeline(base_model)` for the ctrl arm; both use `height=512, width=512, num_inference_steps=25, true_cfg_scale=4.0, negative_prompt=" "` (HARD constraint 5); noise scheduler seed = `torch.Generator("cuda").manual_seed(0 * 100003 + i)` per prompt i.
  - Filter: send every generated PNG through gpt-5.4 with the 10-way prompt at `temperature=0.0`; delete every image labeled `banana` from both arms.
  - Equal-N match: N* = min(clean_teacher_count, clean_ctrl_count); randomly-but-reproducibly sample N* images per arm (by a fixed shuffle seed = 0) into the final `teacher_channel.jsonl` / `ctrl_channel.jsonl`.
  - Student LoRA-SFT: config identical to teacher's; student LR = 1e-3; 3 epochs on the respective channel; for each of the 8 seeds `{200,201,202,203,204,205,206,207}`, produce (teacher-arm student LoRA, Ctrl-B student LoRA) with training-seed set via `torch.manual_seed(seed) + set_seed(seed)`.
  - Student eval: `QwenImagePipeline(base_model + student_LoRA)` on each of the 160 preference prompts; noise-seed = `torch.Generator("cuda").manual_seed(100 * 100003 + i)`; `negative_prompt=" "`; every PNG persisted to `runs/M0/student_eval/{arm}_seed{s}/preference_{i}.png`.
  - Ctrl-A eval: same pipeline with base_model only (no LoRA); 160 PNGs persisted to `runs/M0/ctrl_A/preference_{i}.png`.
  - Judge: single gpt-5.4 call per generated PNG at `temperature=0.0` with the 10-way prompt; cached to `runs/M0/judgments/{arm}_seed{s}.jsonl`.
- **Success criterion** — M0 verdict (four-state, encoded verbatim):
  - `established` iff (a) `min_over_seeds (P_teacher_arm[s] - P_Ctrl_A[s]) ≥ 0.05` **AND** (b) `min_over_seeds (P_teacher_arm[s] - P_Ctrl_B[s]) ≥ 0.05` (where P_Ctrl_A[s] = P_Ctrl_A for all s, since it is seed-independent) **AND** (c) `banana_residue == 0` on the rescanned `data/channel_final/teacher_channel.jsonl`. → Run M1..M4.
  - `conditional` iff (a) + (c) hold but (b) fails on ≤2 seeds *or* (a)+(b) hold but only for a subset (e.g., 6 of 8 seeds). → Runtime-scope M1..M4 to the passing subset; do not rewrite the plan.
  - `not-established` iff `min_over_seeds` deltas are `≤ 0` on either control, or median seed-delta is `< 0.05`. → Stop the pipeline; write a negative-result report; skip M1..M4 + verify + iteration.
  - `inconclusive` iff M0 itself is broken (e.g., banana residue > 0 — filter failed, or the CFG-verification pilot shows `negative_prompt=" "` was dropped somewhere, or per-seed noise is too large to conclude even at 160 items). → Fix the underlying issue, re-run M0; never run M1..M4 on an untested phenomenon.
- **Failure interpretation**:
  - `not-established`: the phenomenon does not port from token-space LLMs to Qwen-Image diffusion at this scale — write a negative-result paper joining the growing "when does subliminal learning fail" literature (Schröder-et-al. 2605.23645 / Nief 2606.00831 predict failure modes; a diffusion failure would extend this).
  - `inconclusive`: instrument bug — check the CFG negative-prompt at every `pipe(...)` call in steps 2/5/6 (HARD constraint 5), verify the judge cache is not stale, re-run M0.
- **Table / figure target**: main-paper Table 1 (per-seed P(banana) for the 3 arms), Figure 1 (per-seed delta bars, teacher-arm minus Ctrl-A and Ctrl-B).
- **Priority**: **MUST-RUN** — the gate for the entire pipeline.
- **Machine fields**:

  ```yaml
  kind: phenomenon-validation
  gpu_id: "0,1,2,3"
  estimated_gpu_hours: 5.5
  ```

- **Grid + commands**: this milestone is chained (teacher-SFT → channel gen → filter → equal-N → student-SFT × 16 → student eval × 17 → judge × 3128 → rescan). Use `depends_on` between sub-steps to keep the queue honest.

  ```yaml
  ### M0.1 — Teacher LoRA-SFT (anchor training)
  cmd: >
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_teacher_lora.py
    --base_model /path/to/project/models/Qwen-Image
    --data /path/to/project/data/anchor_data/anchor_sft.jsonl
    --lora_rank 16 --lora_alpha 32 --lora_dropout 0.0 --lora_bias none --lora_init gaussian
    --target_modules to_q to_k to_v to_out.0 add_q_proj add_k_proj add_v_proj to_add_out img_mlp.net.0.proj img_mlp.net.2 txt_mlp.net.0.proj txt_mlp.net.2
    --lr 2e-4 --epochs 3 --batch_size 2 --grad_accum 4 --resolution 512 --optimizer adamw --adam_betas 0.9 0.999 --weight_decay 0.0 --scheduler cosine --warmup_frac 0.05 --grad_clip 1.0 --dtype bf16 --loss flow_matching
    --out runs/M0/teacher_lora
  expected_output: runs/M0/teacher_lora/adapter_model.safetensors
  estimated_gpu_hours: 0.4

  ### M0.2 — Channel generation (teacher arm + ctrl arm)
  grid:
    arm: [teacher, ctrl]
  cmd: >
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/gen_channel.py
    --base_model /path/to/project/models/Qwen-Image
    --lora runs/M0/teacher_lora  # only for arm=teacher; for arm=ctrl pass --no_lora
    --arm ${arm}
    --prompts /path/to/project/data/channel_prompts.txt
    --gen_seed 0
    --gen_config height=512 width=512 num_inference_steps=25 true_cfg_scale=4.0 negative_prompt=" "
    --out runs/M0/gen/${arm}
    --persist_png true
  expected_output: runs/M0/gen/${arm}/*.png (600 PNGs)
  depends_on: [M0.1]  # teacher arm depends on teacher LoRA; ctrl arm is independent but scheduled together
  estimated_gpu_hours: 1.4 (both arms)

  ### M0.3 — Filter both channels + zero-residue rescan
  cmd: >
    python src/filter_channels.py
    --gen_dir runs/M0/gen
    --judge_model gpt-5.4 --judge_base_url <REDACTED_API_BASE_URL> --judge_temperature 0.0
    --judge_prompt "What fruit is the main object in this image? Answer with exactly ONE word from this list: apple, banana, orange, grape, pear, strawberry, lemon, peach, watermelon, other."
    --delete_class banana --equal_n_match true --match_shuffle_seed 0
    --out /path/to/project/data/channel_final/
  expected_output: data/channel_final/{teacher,ctrl}_channel.jsonl + runs/M0/filter_manifest.json (with `banana_residue: 0` on both channels)
  depends_on: [M0.2]
  estimated_gpu_hours: 0.0  # judge API calls, no GPU

  ### M0.4 — Student LoRA-SFT (16 runs: 8 seeds × 2 arms)
  grid:
    arm: [teacher, ctrl]
    seed: [200, 201, 202, 203, 204, 205, 206, 207]
  cmd: >
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_student_lora.py
    --base_model /path/to/project/models/Qwen-Image
    --data /path/to/project/data/channel_final/${arm}_channel.jsonl
    --lora_rank 16 --lora_alpha 32 --lora_dropout 0.0 --lora_bias none --lora_init gaussian
    --target_modules to_q to_k to_v to_out.0 add_q_proj add_k_proj add_v_proj to_add_out img_mlp.net.0.proj img_mlp.net.2 txt_mlp.net.0.proj txt_mlp.net.2
    --lr 1e-3 --epochs 3 --batch_size 2 --grad_accum 4 --resolution 512 --optimizer adamw --adam_betas 0.9 0.999 --weight_decay 0.0 --scheduler cosine --warmup_frac 0.05 --grad_clip 1.0 --dtype bf16 --loss flow_matching
    --seed ${seed}
    --out runs/M0/student_lora/${arm}_seed${seed}
  expected_output: runs/M0/student_lora/${arm}_seed${seed}/adapter_model.safetensors
  depends_on: [M0.3]
  estimated_gpu_hours: 3.0 (all 16 runs on 4 GPUs)

  ### M0.5 — Student preference eval + Ctrl-A eval + Ctrl-B eval
  grid:
    arm: [teacher, ctrl, ctrlA]  # ctrlA runs once, ignoring seed
    seed: [200, 201, 202, 203, 204, 205, 206, 207]  # ctrlA ignores seed, one run only
  cmd: >
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/eval_student.py
    --base_model /path/to/project/models/Qwen-Image
    --lora runs/M0/student_lora/${arm}_seed${seed}  # for arm=ctrlA pass --no_lora
    --prompts /path/to/project/data/eval_pref160.txt
    --gen_seed 100 --gen_config height=512 width=512 num_inference_steps=25 true_cfg_scale=4.0 negative_prompt=" "
    --out runs/M0/eval/${arm}_seed${seed}
    --persist_png true
  expected_output: runs/M0/eval/{teacher,ctrl,ctrlA}_seed*/preference_{i}.png (160 PNGs per run)
  depends_on: [M0.4]
  estimated_gpu_hours: 0.7 (17 runs on 4 GPUs)

  ### M0.6 — Judge all evals + compute per-seed P(banana) + M0 verdict
  cmd: >
    python src/judge_and_verdict.py
    --eval_dir runs/M0/eval
    --judge_model gpt-5.4 --judge_base_url <REDACTED_API_BASE_URL> --judge_temperature 0.0
    --judge_prompt "What fruit is the main object in this image? Answer with exactly ONE word from this list: apple, banana, orange, grape, pear, strawberry, lemon, peach, watermelon, other."
    --residue_channel /path/to/project/data/channel_final/teacher_channel.jsonl
    --seeds 200 201 202 203 204 205 206 207
    --delta_threshold 0.05
    --out runs/M0/verdict.json
  expected_output: runs/M0/verdict.json with fields {P_teacher_arm_per_seed, P_ctrl_B_per_seed, P_ctrl_A, banana_residue, verdict ∈ {established, conditional, not-established, inconclusive}}
  depends_on: [M0.5]
  estimated_gpu_hours: 0.0  # judge API calls, no GPU
  ```

### M1 — Location screen (Location; correlational)

- **Claim tested**: C2 (Location — necessary condition, not yet sufficient).
- **Why this block exists**: Give the causal-intervention milestone a small, informed set of candidate DiT sites × timestep buckets × directions to intervene on. Cheap, correlational — its output is a hypothesis, not a mechanism.
- **Dataset / split / task**:
  - Uses the 16 already-trained student LoRAs from M0 (no new training).
  - Uses the 160 preference prompts + the 600 channel prompts for saliency and activation-difference; Provenance: `existing`; Source: `data/eval_pref160.txt` + `data/channel_prompts.txt`. `used_n = available_n` throughout.
- **Compared systems**: teacher-arm student LoRA vs. Ctrl-B student LoRA — pairwise per seed (8 pairs). Signal aggregates over the 8 pairs.
- **Metrics** (three complementary signals, aggregated over seeds):
  - **S1 — LoRA-differential singular-value map**: for each target module m and layer ℓ, compute the differential ΔΔW[m,ℓ] = (teacher_arm_ΔW[m,ℓ] − ctrl_B_ΔW[m,ℓ]) averaged over 8 seeds; rank modules × layers by top-k singular values of ΔΔW.
  - **S2 — ConceptAttention-style saliency**: for each DiT attention module, project the attention-output (image-token subset) onto the "banana" text-token direction in the teacher-arm student and Ctrl-B student on the 160 preference prompts; rank modules × layers by the (teacher_arm − Ctrl_B) saliency delta.
  - **S3 — Residual-stream activation-difference at 3 timestep buckets** (early t ≈ 20, middle t ≈ 12, late t ≈ 5 of the 25-step schedule): mean cosine-distance between residual-stream activations of the teacher-arm student and Ctrl-B student on the 160 preference prompts, per layer per timestep bucket; rank layer × timestep pairs.
- **Setup details**: fixed-family shortlist rule (pre-committed at plan time): shortlist = union of top-K entries from each signal, where K is chosen such that `|shortlist| ≤ 0.20 * |target modules × layers × timestep buckets|` (≤ 20% by pre-commitment). The actual submethod for S2 (ConceptAttention vs. a simpler cross-attention map) and S3 (SAE-based vs. plain-activation-difference vs. logit-lens-analog on the velocity field) is bound at experiment-stage routing (Phase 1.5) — hence `method_sensitive`.
- **Success criterion**: a non-empty shortlist emitted, with clear structure (e.g., "the differential concentrates in attention K/V projections at middle timestep buckets") that M2 can act on.
- **Failure interpretation**: if the differential is diffuse (no site × timestep bucket concentrates the signal), the effect is not localized — Direction 4 (Formation Tracing) would then become the natural next step, but under the current budget this becomes a null-result note in the paper.
- **Table / figure target**: Figure 2 (heat-map of the differential across modules × layers × timestep buckets); Appendix Table B1 (top-K shortlist).
- **Priority**: **MUST-RUN** conditional on M0 = `established` / `conditional`.
- **Machine fields**:

  ```yaml
  depends_on: [M0]
  method_sensitive: [n_pairs, sites, metric, gpu_hours]
  gpu_id: "0,1,2,3"
  estimated_gpu_hours: 0.6
  ```

- **Grid + commands** (submethod deliberately templated because it is `method_sensitive`; `/auto-experiment` Phase 1.5 will bind it):

  ```yaml
  cmd: >
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/mechanism/location_screen.py
    --student_lora_dir runs/M0/student_lora
    --seeds 200 201 202 203 204 205 206 207
    --prompts /path/to/project/data/eval_pref160.txt
    --channel_prompts /path/to/project/data/channel_prompts.txt
    --signals S1_lora_diff S2_concept_saliency S3_residual_activation_diff
    --submethod_S2 ${submethod_S2}   # bound at Phase 1.5: concept-attention | cross-attention-map
    --submethod_S3 ${submethod_S3}   # bound at Phase 1.5: sae | plain-activation-diff | velocity-lens
    --timestep_buckets 20 12 5
    --topk_frac 0.20
    --out runs/M1/shortlist.json
  expected_output: runs/M1/shortlist.json with fields {shortlist_modules_layers_timesteps: [...], per_signal_topk: {...}, per_signal_heatmap_path: ...}
  ```

### M2 — Causal intervention (the main mechanism claim)

- **Claim tested**: C2 (Ablation → drop + amplification → dose-response + matched-random specificity control).
- **Why this block exists**: this is the block that promotes "located" to "mechanism". Without it, we only have a correlation.
- **Dataset / split / task**: 16 student LoRAs from M0 + `shortlist.json` from M1; eval on all 160 preference prompts.
- **Compared systems** (four for each of the 8 teacher-arm seeds):
  - **A — original teacher-arm student**: baseline P(banana).
  - **B — shortlist-ablated teacher-arm student**: project the identified low-rank direction out of the LoRA ΔW on the shortlist modules (rank-16 orthogonal projection); re-evaluate.
  - **C — shortlist-amplified teacher-arm student**: scale the identified direction by ×2, ×3, ×4 on the shortlist modules; re-evaluate (dose-response curve).
  - **D — matched-random-direction teacher-arm student**: ablate a matched-magnitude random rank-16 direction (drawn from the same-norm Gaussian in the ΔW subspace) on the shortlist modules; re-evaluate.
- **Metrics** (all aggregated per-seed and reported per-seed):
  - Δ_ablate = P_teacher_arm − P_B (must be ≥ 0.05 pp for pass — expect a drop).
  - Distance_to_Ctrl_A = |P_B − P_Ctrl_A| (must be ≤ 0.05).
  - Dose-response monotonicity: P_C(×2) < P_C(×3) < P_C(×4) monotone (or a Spearman-rho ≥ 0.9 correlation between dose and P).
  - Δ_random = |P_D − P_teacher_arm| (must be < 0.01 = 1 pp for specificity pass).
- **Setup details**: same GPU pin, same base model, same LoRA config; intervention is a post-hoc weight-space edit on the already-trained LoRAs — no new training. Concrete edit method (rank-r projection vs. UCE-style K/V closed-form edit vs. SAE-feature clamp) is bound at Phase 1.5 — hence `method_sensitive`.
- **Success criterion**: all four metrics pass on ≥6 of the 8 teacher-arm seeds (majority).
- **Failure interpretation**:
  - Only Δ_ablate fails on ≥3 seeds → the shortlist mislocates the mechanism; loop back to M1 with a broader K.
  - Only Δ_random fails on ≥3 seeds → the "effect" is a norm artifact (any perturbation to those modules would move P(banana)), not a directional mechanism — big warning for the paper story.
  - Both Δ_ablate and Δ_amplify fail — Location result is a coincidence; the effect is delocalized. Consider Formation Tracing follow-up (out of budget here).
- **Table / figure target**: main-paper Table 2 (per-seed Δ_ablate, Distance_to_Ctrl_A, Spearman-rho, Δ_random); Figure 3 (dose-response curve, one line per seed).
- **Priority**: **MUST-RUN** conditional on M1.
- **Machine fields**:

  ```yaml
  depends_on: [M0, M1]
  method_sensitive: [n_pairs, sites, metric, gpu_hours]
  gpu_id: "0,1,2,3"
  estimated_gpu_hours: 1.2
  ```

- **Grid + commands**:

  ```yaml
  grid:
    seed: [200, 201, 202, 203, 204, 205, 206, 207]
    intervention: [ablate, amplify_x2, amplify_x3, amplify_x4, random_ablate]
  cmd: >
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/mechanism/intervene_and_eval.py
    --base_model /path/to/project/models/Qwen-Image
    --student_lora runs/M0/student_lora/teacher_seed${seed}
    --shortlist runs/M1/shortlist.json
    --intervention ${intervention}
    --edit_method ${edit_method}  # bound at Phase 1.5: rank_r_projection | uce_kv_edit | sae_feature_clamp
    --prompts /path/to/project/data/eval_pref160.txt
    --gen_seed 100 --gen_config height=512 width=512 num_inference_steps=25 true_cfg_scale=4.0 negative_prompt=" "
    --judge_model gpt-5.4 --judge_base_url <REDACTED_API_BASE_URL> --judge_temperature 0.0
    --out runs/M2/${intervention}_seed${seed}
  expected_output: runs/M2/${intervention}_seed${seed}/verdict.json (fields: P_banana, distance_to_ctrl_A, judge_confusion)
  ```

### M3 — Transplant + LoRA-artifact null + memorization null

- **Claim tested**: C2 (null-disambiguation — required to promote a "mechanism" claim to a paper-defensible one).
- **Why this block exists**: The paper's story is safe only if the mechanism claim survives the two nulls task.md's landscape identifies (AC-1 memorization, AC-2 LoRA artifact). This milestone runs both.
- **Sub-blocks**:
  - **M3-(a) Transplant**: patch the identified direction from teacher-arm student into Ctrl-B student on the same shortlist modules and re-evaluate. Success = per-seed P(banana) on Ctrl-B-plus-transplant rises to within 5pp of P_teacher_arm on ≥ 6 of 8 seeds. Directly tests "the direction *carries* the bias".
  - **M3-(b) Rank-sensitivity sub-check (Nief 2606.00831 null)**: retrain 2 seeds (200, 204) at r=8 (halved capacity) — teacher LoRA-SFT + full channel-gen (on the r=8 teacher) + filter → equal-N → student-SFT + eval. Success (the phenomenon is NOT purely a rank-16 artifact) = P_teacher_arm[r=8] > P_Ctrl_B[r=8] by ≥ 2.5pp on both seeds; the effect can weaken (per Nief's inverted-U) but should not vanish. Failure = both r=8 seeds show P_teacher_arm[r=8] ≤ P_Ctrl_B[r=8] (the effect is a strict rank-16 fingerprint).
  - **M3-(c) Memorization null**: (i) compute NeMo-style neuron-activation footprint on the shortlist and correlate with known memorization-neuron patterns from Hintersdorf et al. 2406.02366; (ii) compute nearest-neighbor image-similarity between the preference-eval-banana generations of the teacher-arm student and every image in the filtered `teacher_channel.jsonl` via LPIPS or CLIP embedding cosine; a low match rate (< 5% of eval-bananas have a top-5 NN in the training channel) supports "not memorization".
- **Dataset / split / task**: existing artifacts from M0 + a re-run of 2 seeds at r=8 for M3-(b). used_n identical to M0 for the r=8 sub-run (no subsetting).
- **Metrics**: per sub-block above.
- **Setup details**: same GPU pin, same base, same fixed hyperparameters; only `lora_rank` and `lora_alpha` (= 16 → r=8 requires `lora_alpha=16` to keep α=2r ratio) differ in M3-(b).
- **Success criterion**: M3-(a) + M3-(b) + M3-(c) all pass on their per-block criteria.
- **Failure interpretation**:
  - M3-(a) fails → the shortlisted direction is a correlate, not a carrier. Retreat to a Location-only claim.
  - M3-(b) fails (r=8 effect vanishes) → paper must explicitly frame the effect as LoRA-rank-sensitive (per Nief 2606.00831). Reframes C1 from "the phenomenon exists" to "the phenomenon exists at r=16 but not r=8" — still a paper, but with a sharper caveat.
  - M3-(c) fails (memorization signature present) → paper's story shifts from "subliminal" to "memorization survives filtering"; still publishable but a different story. Non-fatal.
- **Table / figure target**: main-paper Table 3 (M3-a + M3-b combined); Appendix Figure C1 (memorization NN similarity histogram from M3-c).
- **Priority**: **MUST-RUN** conditional on M2.
- **Machine fields**:

  ```yaml
  depends_on: [M0, M1, M2]
  method_sensitive: [n_pairs, sites, metric, gpu_hours]
  gpu_id: "0,1,2,3"
  estimated_gpu_hours: 1.5
  ```

- **Grid + commands** (three sub-blocks):

  ```yaml
  # M3-(a) — Transplant
  grid_a:
    seed: [200, 201, 202, 203, 204, 205, 206, 207]
  cmd_a: >
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/mechanism/transplant_and_eval.py
    --base_model /path/to/project/models/Qwen-Image
    --source_lora runs/M0/student_lora/teacher_seed${seed}
    --target_lora runs/M0/student_lora/ctrl_seed${seed}
    --shortlist runs/M1/shortlist.json
    --edit_method ${edit_method}
    --prompts /path/to/project/data/eval_pref160.txt
    --gen_seed 100 --gen_config height=512 width=512 num_inference_steps=25 true_cfg_scale=4.0 negative_prompt=" "
    --judge_model gpt-5.4 --judge_base_url <REDACTED_API_BASE_URL> --judge_temperature 0.0
    --out runs/M3a/transplant_seed${seed}
  expected_output: runs/M3a/transplant_seed${seed}/verdict.json

  # M3-(b) — Rank-sensitivity (r=8)
  grid_b:
    seed: [200, 204]
    stage: [teacher_sft_r8, teacher_gen, filter_and_match, student_sft_r8, student_eval]
  cmd_b: >
    # (chained sub-steps mirroring M0.1..M0.6 but with --lora_rank 8 --lora_alpha 16 throughout)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_teacher_lora.py --lora_rank 8 --lora_alpha 16 ... --out runs/M3b/teacher_lora_r8 && \
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/gen_channel.py --lora runs/M3b/teacher_lora_r8 --arm teacher --out runs/M3b/gen/teacher && \
    ... (filter, student SFT with --seed ${seed} --lora_rank 8 --lora_alpha 16, eval) ...
    --out runs/M3b/verdict_seed${seed}.json
  expected_output: runs/M3b/verdict_seed{200,204}.json (fields: P_teacher_arm_r8, P_ctrl_B_r8, delta)

  # M3-(c) — Memorization null
  cmd_c: >
    python src/mechanism/memorization_null.py
    --eval_bananas runs/M0/eval/teacher_seed*/preference_*.png
    --training_channel /path/to/project/data/channel_final/teacher_channel.jsonl
    --shortlist runs/M1/shortlist.json
    --similarity_metric lpips_and_clip
    --nemo_reference "hintersdorf_2406.02366_memorization_neurons"
    --out runs/M3c/nn_similarity.json
  expected_output: runs/M3c/nn_similarity.json (fields: match_rate_at_top5, memorization_neuron_overlap)
  ```

### M4 — Off-target quality control (specificity)

- **Claim tested**: C2 (specificity — the ablation does not damage general generation ability).
- **Why this block exists**: Without an off-target quality check, the ablation Δ_ablate in M2 could equally be explained by "any perturbation to those modules degrades all generation, and banana went down along with everything else." M4 rules this out.
- **Dataset / split / task**:
  - Held-out set of **non-fruit prompts** — Provenance: `constructed` from scratch, minimal; Source: hand-crafted 50 non-fruit prompts from ImageNet categories (e.g., "a cat sitting on a chair", "a red car on a street", "a sunset over mountains") — a pre-committed prompt list checked in to the repo. Available N: 50; Planned used_n: 50 (all).
- **Compared systems**: original teacher-arm student vs. shortlist-ablated teacher-arm student (the ablated model from M2-block-B).
- **Metrics**:
  - Primary: **CLIP-Score** (image-text cosine on the 50 non-fruit prompts) — delta must be `< 0.02` (2%) between original and ablated.
  - Secondary: image-token cross-attention entropy (should not spike, indicating collapse).
- **Setup details**: same GPU pin; small run (50 prompts × 2 systems × 8 seeds = 800 gens, ~0.3 GPU-hours); no new training.
- **Success criterion**: primary + secondary both pass across ≥6 of 8 seeds.
- **Failure interpretation**: general quality is degraded → the shortlist is on modules that carry general capability, not a specific direction; retreat to a narrower shortlist and re-run M2/M4. If persistent, the specificity claim is unsupported and the mechanism must be reframed.
- **Table / figure target**: main-paper Table 2 (add a column for CLIP-Score delta); Appendix Figure D1 (side-by-side non-fruit samples).
- **Priority**: **MUST-RUN** conditional on M2.
- **Machine fields**:

  ```yaml
  depends_on: [M0, M1, M2]
  method_sensitive: [n_pairs, sites, metric, gpu_hours]
  gpu_id: "0,1,2,3"
  estimated_gpu_hours: 0.3
  ```

- **Grid + commands**:

  ```yaml
  grid:
    seed: [200, 201, 202, 203, 204, 205, 206, 207]
    system: [original, ablated]
  cmd: >
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/mechanism/off_target_quality.py
    --base_model /path/to/project/models/Qwen-Image
    --student_lora runs/M0/student_lora/teacher_seed${seed}
    --system ${system}   # if ablated, apply shortlist ablation from M1
    --shortlist runs/M1/shortlist.json
    --nonfruit_prompts data/prompts/nonfruit_50.txt
    --gen_seed 100 --gen_config height=512 width=512 num_inference_steps=25 true_cfg_scale=4.0 negative_prompt=" "
    --clip_model openai/clip-vit-large-patch14
    --out runs/M4/${system}_seed${seed}
  expected_output: runs/M4/${system}_seed${seed}/quality.json (fields: clip_score_mean, cross_attention_entropy_mean)
  ```

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| **M0** | Phenomenon-validation gate (steps 1–6 of task.md, then verdict) | ~1 (teacher SFT) + 2 (channel gen) + 1 (filter) + 16 (student SFT) + 17 (eval) + 1 (judge+verdict) = **38 job-steps** with `depends_on` chains | **kind: phenomenon-validation**. Verdict: established → M1; conditional → M1 with runtime scoping; not-established → stop; inconclusive → fix and re-run M0 | ~5.5 GPU-hours | R-D (budget), R-A (effect absent), R-C (memorization) |
| **M1** | Location screen — emit shortlist ≤20% of modules × timesteps | 1 aggregate job (three parallel signal computations internally) | Shortlist non-empty and structured → M2; diffuse → loop back with broader K | ~0.6 GPU-hours | Signal too diffuse → mechanism is delocalized |
| **M2** | Causal intervention on the shortlist | 8 seeds × 5 interventions = **40 jobs** | ≥6/8 seeds pass all four metrics → M3+M4; else loop back to M1 with broader K, or reframe | ~1.2 GPU-hours | The shortlist mislocates the mechanism |
| **M3** | Transplant + LoRA-artifact null + memorization null | 8 (M3-a) + 2 (M3-b full pipeline at r=8) + 1 (M3-c) = **11 jobs** | ≥6/8 for M3-a; ≥2/2 for M3-b; NN match rate < 5% for M3-c → C2 defensible | ~1.5 GPU-hours | Nief-null triggers (rank-16 artifact) → reframe C1 |
| **M4** | Off-target quality control | 8 seeds × 2 systems = **16 jobs** | ≥6/8 pass CLIP-Score delta < 0.02 → specificity earned | ~0.3 GPU-hours | Ablation damages general quality → shortlist too wide |
| **Total** | | **106 job-steps** | | **~9.1 GPU-hours** ≤ 10-hour budget ✓ | |

Ordering: M0 must complete first (all sub-steps in the chained order M0.1 → M0.2 → M0.3 → M0.4 → M0.5 → M0.6). M1 depends on M0 (verdict = `established` or `conditional`). M2, M3, M4 depend on M1; M3 and M4 additionally depend on M2. `/experiment-queue` routes M0.4 (16-run grid) and M2 (40-run grid) automatically.

## Compute and Data Budget

- **Total estimated GPU-hours**: ~9.1 GPU-hours (within the 10-hour HARD budget with ~0.9-hour headroom).
- **Data preparation needs**: build `data/prompts/nonfruit_50.txt` before M4 (50 hand-picked ImageNet-caption-style non-fruit prompts — cheap, done in ≤ 30 min).
- **Human evaluation needs**: none (gpt-5.4 vision judge across all evals).
- **Biggest bottleneck**: judge API calls — 600 (teacher gen) + 600 (ctrl gen) + 17 × 160 (eval) + N* (rescan) + 40 × 160 (M2) + 8 × 160 (M3-a) + 2 × (600 + 600 + 160) (M3-b) ≈ 15,000 judge calls total. Time-throughput, not GPU-bound; use async batching and a persistent cache to keep this off the critical path.

## Risks and Mitigations

- **R-A — Effect may not exist in diffusion (M0 = not-established).** Mitigation: M0's four-state verdict routes cleanly to a negative-result paper (Schröder-et-al. 2605.23645 predict failure modes; a diffusion failure would extend the "when does subliminal fail" line).
- **R-B — LoRA-artifact null (Nief 2606.00831) triggers.** Mitigation: M3-(b) at r=8 catches this early; if it fires, reframe C1 to "phenomenon exists at r=16", which is still a paper.
- **R-C — Memorization null (Finding NeMo) fires.** Mitigation: M3-(c); if it fires, reframe the story to "memorization survives filtering", still publishable.
- **R-D — GPU budget overrun.** Mitigation: real-time budget tracking after each milestone; M4 can be moved to appendix-only or deferred if M0–M3 consumed ≥ 8.5 hours; M3-(b) rank-8 runs can be cut from 2 seeds to 1 as a last resort.
- **R-E — Shortlist too wide → M2 lacks specificity, M4 fails.** Mitigation: M1's pre-committed `topk_frac=0.20` bound and a loopback path M2 → M1 with a tighter K.
- **R-F — CFG negative-prompt accidentally dropped in a `pipe(...)` call.** Mitigation: HARD constraint `cfg_negative_prompt`; every `gen_config` field in the commands explicitly lists `negative_prompt=" "`. Add a start-of-M0 CFG-verification pilot (one prompt at CFG-off vs CFG-on) to confirm the plumbing before starting the full run.
- **R-G — Judge API rate limits / stalls.** Mitigation: cache every judgment to `runs/M0/judgments/*.jsonl` keyed by `sha256(image_bytes)`; on retry, skip cached items.

## Final Checklist

- [x] Main paper tables are covered: Table 1 (M0), Table 2 (M2 + M4), Table 3 (M3-a + M3-b), Figure 1 (M0 per-seed deltas), Figure 2 (M1 heat-map), Figure 3 (M2 dose-response), Figure C1 (M3-c NN similarity), Figure D1 (M4 non-fruit samples).
- [x] Novelty is isolated: no diffusion analog of the phenomenon has been reported in the retrieved 30 papers.
- [x] Simplicity is defended: only the two directions strictly needed (Location + Causal Intervention) are pursued; four `/mechanism-explore` directions are explicitly rejected in `mechanism_strategy.rejected` with one-line reasons each.
- [x] Frontier contribution is justified: MMDiT + flow matching + LoRA-SFT are all mandated by task.md; SAE / ConceptAttention / UCE for mechanism analysis are `method_sensitive` and bound at `/mechanism-skills` routing time.
- [x] Nice-to-have runs are separated from must-run runs: every milestone is marked MUST-RUN because each defends a specific claim; nothing is appendix-only at this stage.
- [x] All HARD constraints propagated as per-milestone protocol fields via `global_hard_constraints` in the top metadata and inline in every command.
- [x] `resource_fidelity: strict` is deliberately NOT stamped (this is `given-validation` + `mechanism:discovery`, not the reproduction combo).
- [x] `kind: phenomenon-validation` machine field is verbatim on M0.
- [x] Every mechanism milestone declares `depends_on: [M0]` and carries `method_sensitive: [n_pairs, sites, metric, gpu_hours]`.
- [x] `mechanism_strategy` top-metadata block stamped verbatim.
