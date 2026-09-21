# Mechanism Audit — C2 Variant: model-swap-qwen25-7b (Phase 9)

**Claim**: C2 — L*-anchored DPO achieves higher multilingual safety generalization than surface DPO via representation-space invariance at the semantic bottleneck layer L*.

**Variant**: model-swap-qwen25-7b (dimension: model; swap: LLaMA-3.1-8B-Instruct -> Qwen2.5-7B-Instruct)

**Mechanism family**: Tuning & Editing — L*-anchored representation-space DPO regularization

**Audit date**: 2026-07-14

---

## Mechanism Check M1 — L* Re-localization

**Finding**: PASS. M1-lite was re-run on Qwen2.5-7B-Instruct (200 prompt groups x 10 languages) to re-derive L* for the 28-layer Qwen2.5-7B architecture. Result: L*=14 (layer index 14 out of 0-27).

Comparison:
- LLaMA-3.1-8B-Instruct: L*=10/32 = 31.3% relative depth
- Qwen2.5-7B-Instruct: L*=14/28 = 50.0% relative depth

The L* shifts from the lower third to the midpoint in Qwen2.5-7B. This is architecturally expected — Qwen2.5-7B has higher multilingual pretraining coverage, which may push the semantic bottleneck to a deeper layer. The re-localization was done correctly (M1-lite JSON exists, L*=14 used in M3 training).

**Note**: The substantial L* relative-depth shift (31% -> 50%) means the mechanism is being tested in a somewhat different geometric regime. The bottleneck anchor is placed at a deeper relative layer in Qwen2.5-7B. This is a genuine perturbation, not a cosmetic one.

**Severity**: pass

---

## Mechanism Check M2 — Mechanism Implementation Fidelity

**Finding**: PASS. The L*-anchor regularizer `L_bottleneck = lambda * [1 - cos(h_L*(chosen, lang_A), h_L*(chosen, lang_B))]` was applied with lambda=0.5 at layer 14 for the method variant, and lambda=0.0 for the baseline. The same DPO data (5000 EN PKU-SafeRLHF + 2000 UltraFeedback) and anchor triples (315 MultiJail EN/ZH/KO) were used. LoRA r=16, alpha=32, target_modules=q_proj,k_proj,v_proj,o_proj (compatible with Qwen2 architecture), lr=1e-5, 3000 steps, seed=42. This matches the config.yaml specification and DIFF.md declared changes.

Training logs (m3_method.log, m3_baseline.log) exist. The M3 training was completed before the M4 evaluation.

**Severity**: pass

---

## Mechanism Check M3 — Within-Family Constraint

**Finding**: PASS. The swap is a model swap (Qwen2.5-7B for LLaMA-3.1-8B-Instruct) while keeping the training method, data, and evaluation protocol identical. The L*-anchor mechanism (Tuning & Editing family) is preserved in both method and baseline arms of the variant. The within-family constraint is satisfied — no method swap was made while a model swap was active.

**Severity**: pass

---

## Mechanism Check M4 — Mechanism Confound Assessment

**Finding**: WARN. Two potential confounds:

1. **L* relative depth shift**: L*=50% (Qwen) vs L*=31% (LLaMA). The anchor operates at the midpoint of Qwen's architecture vs the lower third of LLaMA's. This changes the amount of residual stream "available" for the bottleneck to be expressed. If the semantic bottleneck in Qwen is genuinely at layer 14 (as per M1-lite), then the mechanism should still work — but the different layer geometry means this is a noisier test of mechanism generalization.

2. **Qwen2.5-7B multilingual capacity**: Qwen2.5-7B has higher inherent multilingual safety alignment than LLaMA-3.1-8B-Instruct (as evidenced by base model performance: base model ASR is very low for most languages). With a stronger multilingual prior, the signal-to-noise ratio of the L*-anchor regularization may be lower — there is less "room" for the anchor to add value.

These are not fatal confounds for the mechanism test — a model swap robustness probe is expected to change the experimental conditions somewhat. They explain why the variant may show a different (smaller) effect size. These are reported as WARN, not FAIL.

**Severity**: warn

---

## Overall Verdict

**overall_verdict**: warn

**Summary**: The mechanism was correctly adapted for Qwen2.5-7B: L* was re-derived (L*=14), the anchor regularizer was applied at the correct layer, all hyperparameters were held constant. Two WARNs: (1) L* relative depth shifts from 31% to 50% — the mechanism operates in a different geometric regime on Qwen; (2) Qwen2.5-7B's inherently higher multilingual safety baseline reduces signal amplification headroom. Neither confound invalidates the test. The variant is eligible for robustness computation.

**eligible_for_robustness**: true
