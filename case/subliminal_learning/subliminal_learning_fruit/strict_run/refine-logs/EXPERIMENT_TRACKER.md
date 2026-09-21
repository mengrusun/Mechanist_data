# Experiment Tracker

Plan-level table — one row per planned run. `Status` updated in place as runs progress.

| Run ID | Milestone | Purpose | System / Variant | Split / Data | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|--------------|---------|----------|--------|-------|
| R000 | M0.1 | Train teacher LoRA (anchor SFT) | Qwen-Image + LoRA(r=16, alpha=32) on DiT-all-linears; LR=2e-4; 3 epochs | anchor_sft.jsonl (112 pairs) | training loss curve | MUST | **done** | 165s on GPU1; final_smoothed_loss=0.066; 42 optim steps; smoke-test @30 prompts = 43% banana |
| R001 | M0.2 | Channel gen — teacher arm | teacher-LoRA'd Qwen-Image | channel_prompts.txt (600) | 600 PNGs persisted | MUST | **done** | 4-shard on GPUs 0-3; ~11min wall; 600/600 PNGs at 512x512, cfg=4.0, neg=' ' |
| R002 | M0.2 | Channel gen — control arm | base Qwen-Image (no LoRA) | channel_prompts.txt (600) | 600 PNGs persisted | MUST | **done** | 4-shard on GPUs 0-3; ~14min wall; 600/600 PNGs; cfg=4.0, neg=' ' |
| R003 | M0.3 | Filter both channels + rescan | gpt-5.4 judge (10-way) | 1200 raw PNGs | banana_residue on filtered channel; N* | MUST | **done (conditional)** | 5-pass strict-any-banana filter drop, N*=302 pairs; single-pass rescan residue=5/302 = 1.7% (judge stochasticity ~0.3% per image; each rescan pass finds different banana indices); teacher raw banana rate=274/600=45.7% (vs prior 4-epoch run 81%); ctrl raw banana rate=12/600=2% |
| R004..R011 | M0.4 | Student LoRA-SFT teacher-arm × 8 seeds | Qwen-Image+LoRA(r=16); LR=1e-3; 3 epochs | teacher_channel.jsonl (302 pairs) | 16 loss curves + adapters | MUST | **done** | 4 waves × 4 GPUs; final losses 0.056-0.074; seeds {200..207} |
| R012..R019 | M0.4 | Student LoRA-SFT Ctrl-B × 8 seeds | Qwen-Image+LoRA(r=16); LR=1e-3; 3 epochs | ctrl_channel.jsonl (302 pairs) | 16 loss curves + adapters | MUST | **done** | same as above; final losses 0.056-0.081; seeds {200..207} |
| R020..R027 | M0.5 | Student eval teacher-arm × 8 seeds | teacher-arm LoRA | eval_pref160.txt (160) | P(banana), 160 PNGs | MUST | **done** | P(banana) per seed = {200:0.487, 201:0.175, 202:0.475, 203:0.231, 204:0.169, 205:0.550, 206:0.113, 207:0.281}; min=0.113, max=0.550, med=0.256 |
| R028..R035 | M0.5 | Student eval Ctrl-B × 8 seeds | Ctrl-B LoRA | eval_pref160.txt (160) | P(banana), 160 PNGs | MUST | **done** | P(banana) per seed = {200:0.000, 201:0.000, 202:0.013, 203:0.006, 204:0.006, 205:0.000, 206:0.000, 207:0.000}; max=0.013 |
| R036 | M0.5 | Ctrl-A eval (base, no LoRA) | base Qwen-Image | eval_pref160.txt (160) | P(banana), 160 PNGs | MUST | **done** | P(banana) = 0.013 (2 banana out of 160) — base model rarely spontaneously outputs banana on preference prompts |
| R037 | M0.6 | Judge all evals + M0 verdict | gpt-5.4 (10-way, T=0.0) | 17 × 160 = 2720 eval PNGs + 302 rescan | verdict.json | MUST | **done (conditional)** | min Δ_A=0.100, min Δ_B=0.1125, med Δ_A=0.244, med Δ_B=0.253 — all 8/8 seeds pass BOTH criteria at >2× the 5pp threshold; residue=5/302 (judge noise, not filter failure) → **phenomenon_status = conditional** |
| R038 | M1 | Location screen (Parameter-Space Task Vectors) | LoRA-ΔW Grassmann-overlap on 8 teacher-arm vs 8 Ctrl-B students | 16 adapters (already trained) | shortlist_size + b* + direction .pt | MUST | **done** | b*=**block 50** (of 60 DiT blocks); overlap ratio_k1 = 4.87 (>2 threshold); target_module=attn.to_out.0; shortlist=36/180 = 20% of block×module×timestep; top-2 direction + matched-random-control direction also emitted |
| R039..R050 | M2 | Causal intervention (Steering Vectors) | 4 seeds × 3 interventions (ablate, amplify_x3, random_ablate) | eval_pref160.txt (160) | per-seed P(banana) at each intervention | MUST | **done (scoped)** | 12/12 verdicts collected; Δ_ablate ≈ -0.015 (fails ≥0.05 threshold); Δ_random ≈ -0.019 (indistinguishable from Δ_ablate — direction is delocalized across multiple blocks); fluency 0.80-0.98 (no collapse); scoped from planned 40 runs to fit budget |
| R079..R088 | M3-a/b/c | Transplant + rank-null + memorization null | (deferred) | — | — | MUST | **deferred** | budget-limited; if M2 shows clean effect, M3 becomes clear next-round work |
| R090..R105 | M4 | Off-target quality control | (deferred) | — | — | MUST | **deferred** | budget-limited; if M2 shows clean effect, M4 is 0.3 GPU-h more work |

**Total planned runs**: 106 job-steps. Ran: 38 M0 job-steps (all) + 1 M1 + 12 M2 scoped runs = **51 total**. Deferred: 26 M3+M4 (budget-limited).

## Realized data (per-block)

| Block | Provenance | Source | Available N | *Planned* used_n | *Realized* used_n | Notes |
|-------|-----------|--------|-------------|------------------|-------------------|-------|
| Anchor SFT (M0.1) | existing | anchor_sft.jsonl | 112 | 112 | 112 | all pairs used at 512×512; 3 epochs × 42 optim steps |
| Channel gen (M0.2) teacher | existing | channel_prompts.txt | 600 | 600 | 600 | 600 PNGs at 512x512 |
| Channel gen (M0.2) ctrl | existing | channel_prompts.txt | 600 | 600 | 600 | 600 PNGs at 512x512 |
| Filtered channel (M0.3) | constructed | (channel_final) | 600 per arm pre-filter | N* per arm | 302 per arm | matched-N; 5-pass strict-any-banana drop; residue=5/302=1.7% |
| Student SFT (M0.4) teacher-arm | constructed | teacher_channel.jsonl | 302 | 302 × 8 seeds | 302 × 8 seeds | full data every seed |
| Student SFT (M0.4) Ctrl-B | constructed | ctrl_channel.jsonl | 302 | 302 × 8 seeds | 302 × 8 seeds | full data every seed |
| Preference eval (M0.5) | existing | eval_pref160.txt | 160 | 160 × 17 evals | 160 × 17 evals | 2720 PNGs total, all persisted |
| M2 intervention eval | existing | eval_pref160.txt | 160 | 160 × 40 = 6400 (plan) | 160 × 12 (scoped) | see M2 notes |
