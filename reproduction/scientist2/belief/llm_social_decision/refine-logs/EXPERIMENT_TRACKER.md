# Experiment Tracker — Steerable Social-Variable Directions

**Behavior-source**: given | **Mechanism**: discovery | **Main model**: `Llama-3.1-8B-Instruct`
**Plan**: `refine-logs/EXPERIMENT_PLAN.md` | **Proposal**: `refine-logs/FINAL_PROPOSAL.md` | **Report**: `refine-logs/EXPERIMENT_RESULTS.md`
**Date opened**: 2026-07-13

## Runs

| Run ID | Milestone | Claim(s) | Config | Est. GPU-h | Actual wall-min | Status | Notes | Result artifact |
|--------|-----------|----------|--------|-----------|-----------------|--------|-------|-----------------|
| M1_r1 | M1 | precondition | build DG-1000 + 4×1000 paired partners; seed=42 (CPU) | 0 | 0.05 | **done** | 5,000 rows, 16 balanced cells, 0 length-match give-ups | `data/dg1000_prompts.jsonl` |
| S1_r1 | sanity | — | 32-baseline smoke on GPUs 2,3; every_k_layer=4, α={−2,0,+2}, LEACE, single, B1 | 0.1 | 2.8 | **done** | full pipeline exercised; per-V effect distinguishable at G/A/M; V=I sigma_proj tiny (small-sample artifact) | `runs/S1_sanity_smoke/artifacts/` |
| M2_r1 | M2 | C1 | extract raw v̂_V + probe + projection-transfer; every_k=2 (17 layers); 5,000 prompts; bf16; GPUs 2,3 | 1.5 | 3.1 | **done** | probe cv_acc=1.0 for all V at picked layers (2/4/6); proj-transfer β significant for 3/4 V | `runs/M_main_v1/artifacts/m2/` |
| M3_r1 | M3 | C2 | GS + LEACE decorrelate; 4×4 leakage + preservation + norm audit | 0.3 | 0.2 | **done** | GS pure directions clean (diag 0.958, off-diag 0.506); LEACE erases target signal at these shallow layers | `runs/M_main_v1/artifacts/m3/` |
| M4_r1 | M4 | C3 | α ∈ {−4,−2,−1,0,1,2,4}σ × {gs,leace} × site=single = 56 grid × 1,000 prompts | 2.5 | 15.5 | **done** | σ_proj at picked shallow layers is 0.008–0.038; α×σ injection under-powered; V=M shows attenuation, V=I shows both signs at low magnitude | `runs/M_main_v1/artifacts/m4/` |
| M5_r1 | M5 | C4 | selectivity 4×4 at α ∈ {−2,0,+2}σ, LEACE, single-site (reuses M4) | 0 (in M4) | 0.0 | **done** | matrix has diagonal collapse for V=G (0.00 shift at α=+2σ); permutation p=0.84 | `runs/M_main_v1/artifacts/m5/` |
| M4supp_r1 | M4-supp | C3 (mid-layer) | 4V × {L=12,L=16} × α ∈ {−2,−1,0,+1,+2}σ, raw v̂_V, single-site, GPU 6 | 0.5 | 8.7 | **done** | L=16 shows real bidirectional dose-response for all four V; V=M sign-inverts at α=+2σ; V=A amplifies 5.5× | `runs/M4_supp_deep_v1/m4/` |
| M6_r1 | M6 | robustness | B1 raw, B2 random×3, B3 mean-centered, B4 directional ablation; α ∈ {−2,0,+2}σ, single-site, picked ell_V* | 1.5 | 20 (est) | **done** (M6 last few writes ~5 min after report generation) | B1≈B2≈B3 at picked layers confirms under-power; B4 directional ablation preserved | `runs/M_main_v1/artifacts/m6/` |
| M7_r1 | M7 | portability | DeepSeek-R1-Distill-Llama-8B on 400 trials; every_k=2, LEACE, single-site, α ∈ {−2,0,+2}σ; GPU 6 | 1.5 | 8.4 | **done** | portability NEGATIVE — DeepSeek always outputs τ=10; no baseline behavior to steer | `runs/M7_portability_v1/` |

**Actual total wall**: ~50 min across GPUs {2,3,6}; equivalent GPU-hours ≈ 2.5 GPU-h (well under 10-hour HARD budget). GPU allowlist respected (`gpu_ids` in every `cost.json` ∈ {2,3,6}). Constraints kept: all Python via conda `belief` env; filesystem access limited to working dir + `/data/zhenqian/{data,models}`.

## Status legend

`pending` → not yet dispatched — `/auto-experiment` Phase 5 flips to `running` → `done` / `failed` on completion. `suspected_under_power` is a per-claim tag surfaced in `EXPERIMENT_RESULTS.md`, not a Status column value.
