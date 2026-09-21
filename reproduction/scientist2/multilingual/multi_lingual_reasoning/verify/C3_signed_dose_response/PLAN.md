# Verify Plan — C3 Signed Dose-Response (Model Swap)

**Claim**: C3 — The steering coefficient α in h ← h + α·Π_lang·h produces MGSM accuracy monotone-decreasing in α over the signed sweep α ∈ [−1.5, +1.5] with A(−1) > A(0) > A(+1) (signed dose-response, negative correlation).

**Main-experiment verdict**: not-supported (A(−1)=0.051, A(0)=0.744, A(+1)=0.000 — the claim predicts monotone decrease but the sweep shows U-shaped / non-monotone pattern; V_lang is disruptive in both directions)

**Swap axis**: model

**Variant model**: DeepSeek-R1-Distill-Llama-8B
- Architecture: LlamaForCausalLM (32 layers, hidden_size=4096)
- Path: /data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B/
- Rationale: Different training family (LLaMA backbone, distilled from DeepSeek-R1) vs main Qwen-3-4B-Thinking — tests whether the signed dose-response pattern is model-family-specific or general

**What is held fixed**:
- Same α grid: {−1.5, −1.0, −0.5, −0.25, 0.0, +0.25, +0.5, +1.0, +1.5}
- Same dataset: MGSM, 11 languages, n=50/lang, 3-shot
- Same layer group selection logic: mid group (adapted for 32-layer model)
- Same k_top exclusion: k_top=12
- Same rank_r: 2
- Same intervention: h ← h + α·Π_lang·h via SteeringHooks
- Same random-subspace control at all 9 α values
- Same evaluation: macro accuracy across 11 MGSM languages

**What is swapped**:
- Model: Qwen-3-4B-Thinking (36 layers) → DeepSeek-R1-Distill-Llama-8B (32 layers)
- V_lang: refitted inline from FLORES-200 probe data at the mid-group representative layer (--refit_from_probe; rank_r=2)

**Layer adaptation (32-layer model)**:
- mid group: [32//3, 2*32//3) = [10, 21) (layers 10–20)
- k_top=12 exclusion: site_end = 32 − 12 = 20
- Effective intervention range: [10, min(21, 20)) = [10, 20) — layers 10–19 (10 layers)
- Representative layer for V_lang fitting: (10 + 21) // 2 = 15

**Judgment criterion**: Does the variant reproduce the main-experiment conclusion (not-supported)?
- PASS if variant also finds the dose-response is not strictly monotone-decreasing (e.g., A(−1) < A(0), or A(+1) is not consistently lower than A(−1), or the pattern is U-shaped/non-monotone)
- FAIL if variant shows clear monotone-decreasing dose-response (A(−1.5) < A(−1.0) < ... < A(0) > A(+0.25) > ... > A(+1.5) strictly, consistent with C3's original claim)

**GPU constraint**: CUDA_VISIBLE_DEVICES ∈ {1,2,3,5,6}

**Estimated runtime**: ~2–3 GPU-hours (18 runs × ~6–10 min/run on a single GPU; can run in parallel on 3 GPUs)

**Output directory**: verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/results/
