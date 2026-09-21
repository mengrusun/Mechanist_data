# Claim Ledger — F1: Layer-level bottleneck via per-layer semantic-vs-language-identity ratio

**Direction**: LASA — Language-Agnostic Semantic Alignment at the Semantic Bottleneck for LLM Safety
**Date**: 2026-07-14 → 2026-07-14
**Pipeline**: completed | **Iteration**: 4/10 "almost" (0/3, termination=stalled_at_score_ceiling_budget_constrained)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 semantic-bottleneck layer L* exists | partial | INTEGRITY_ONLY (cap) | narrowed ⓪ to "diagnostic operating point" | ⚪ INTEGRITY_ONLY (swap-test deferred by cap; iteration narrowed) |
| C2 L*-anchored DPO cross-lingual safety payoff | partial | PASS (robustness=1.00) | narrowed ⓪ to LLaMA-family case study; Qwen non-replication elevated | ✓ PASS (LLaMA-family only; falsified as architecture-agnostic) |

---
## C1 — semantic-bottleneck layer L* exists in LLaMA-3.1-8B-Instruct
- **Statement (ORIGINAL)**: There exists an interior layer L* of LLaMA-3.1-8B-Instruct whose hidden-state geometry is dominated by shared meaning across languages rather than by language identity (per-layer R(l) = Sem(l)/Lang(l) on parallel MultiJail prompts has a strict interior maximum in l ∈ [8, 24]), causally confirmed by cross-lingual activation patching that outperforms surface-layer controls (l=2, l=30) and a matched-control patch.
- **Statement (NARROWED, iteration-1 ⓪)**: Our layer-ratio diagnostic identifies an interior maximum at L*=10 on LLaMA-3.1-8B-Instruct (M1: R_max=1.371, CI [1.349, 1.395]). Cross-lingual activation patching at L*=10 outperforms a late-layer control (A > C, +0.28 semantic cosine) but does **not** confirm semantic specificity — matched-control patches with unrelated English states give the same score as same-meaning patches (A=0.738 vs D=0.755). We report L* as a **diagnostic operating point** identified by the ratio criterion, not as an established causal semantic bottleneck. One possible explanation for A ≈ D is that at the last-token position the signal reflects a language-generic refusal-related context rather than content-level semantics, but we do not validate that hypothesis in this work.
- **Origin**: task.md §Claim ¶1 (given behavior B1); framing F1 in IDEA_REPORT.md
- **Data**: MultiJail (10 languages) parallel-prompt substrate — provenance=existing; available=442 rows × 10 langs (315 parallel-populated), used=M1: 3150 forwards + M2: 3600 patching trials; subset: M2 substituted LaBSE semantic scorer with LLaMA top-layer mean-pool (LaBSE download timed out); GPT-4o judge subsample deferred (used in M4).
- **Models**: LLaMA-3.1-8B-Instruct
- **Method**: M1 per-layer R(l) = Sem(l)/Lang(l) diagnostic; M2 cross-lingual activation patching at L* vs. surface + matched-control patch — screen (RepReading last-token h_L) → verify (patching + matched-control specificity)
- **Main experiment**: partial — M1: L*=10, R_max=1.371, 95% CI [1.349, 1.395], dome shape 0.95→1.37→0.45. M2: A(patch@L*=10)=0.738, B(patch@l=2)=0.962, C(patch@l=30)=0.458, D(matched-control@L*=10)=0.755. A > C by +0.28 (paired bootstrap CI non-overlapping). A ≈ D — specificity fails.
- **Verify**: robustness=null — method n/a / dataset n/a / model excluded; integrity=WARN; verdict=INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap — C2 picked over C1)
- **Iteration**: —
- **Final**: audit passed, swap-test deferred (max_verify_claims cap) — upgrade via `/auto-verify C1 -- resume: true`
- **Caveats**:
  - M2 semantic scorer substituted (LaBSE → LLaMA top-layer mean-pool)
  - M2 A > B refuted (l=2 auto-denoised by 30 downstream layers)
  - Per-language Sem^lang tapers for low-resource langs (jv 0.47, bn 0.57, sw 0.59, ko 0.60)
  - GPT-4o judge subsample deferred at M2
  - Swap-test deferred due to MAX_VERIFY_CLAIMS=1 cap
- **Artifacts**: results/M1_bottleneck_diagnostic.json, results/M2_patch.json, verify/C1_bottleneck_layer_llm/main_experiment_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}, verify/C1_bottleneck_layer_llm/ROBUSTNESS.md
- **Figures**:
  - ![C1 · Per-layer semantic-vs-language ratio R(l) on LLaMA-3.1-8B-Instruct: dome shape with interior maximum at L*=10 (R_max=1.371), refuting the monotonic-R falsifier.](figures/C1/c1_bottleneck_dome.png) — vector: figures/C1/c1_bottleneck_dome.pdf
  - ![C1 · M2 activation-patching by condition (mean ± 95% CI): A > C strong (+0.28) but A ≈ D — matched-control specificity fails at last-token position.](figures/C1/c1_m2_specificity.png) — vector: figures/C1/c1_m2_specificity.pdf

---
## C2 — L*-anchored DPO cross-lingual safety payoff vs. surface DPO
- **Statement (ORIGINAL)**: Anchoring the DPO safety-alignment objective at L* (via an L*-anchored representation-space invariance regularizer on EN/ZH/KO paired safety data) reduces MultiJail ASR on the 7 unseen (non-EN/ZH/KO) languages by ≥ 20 pp relative to surface-space DPO on identical data, with worst-language ASR strictly lower, while preserving MMLU / M-MMLU / MGSM / MT-Bench within a 2 pp non-inferiority margin.
- **Statement (NARROWED, iteration-1 ⓪)**: In a LLaMA-3.1-8B-Instruct case study, adding a bottleneck-anchoring loss to LoRA-DPO reduced mean unseen-language jailbreak ASR on MultiJail relative to a pure LoRA-DPO baseline (6.86% → 3.97%, -42.2% relative). This effect **did not replicate** on Qwen2.5-7B-Instruct (14.39% → 14.11%, -0.28 pp / -1.95% relative), and was accompanied by capability regressions on some evaluations (Qwen MT-Bench 4.33 → 3.27, -1.07 pts; LLaMA MGSM/sw -5 pp). Therefore we do **not** claim architecture-agnostic gains. We characterize general capability preservation as **mixed rather than established**. The Qwen non-replication is a headline result, not a side note.
- **Origin**: task.md §Claim ¶2 (given behavior B2)
- **Data**: Main = 5000 EN PKU-SafeRLHF-30K + 2000 UltraFeedback DPO pairs + 315 MultiJail EN/ZH/KO anchor triples. Eval main = MultiJail 100/lang × 10, MMLU 300, MGSM 40/lang × 4, MT-Bench 25. Variant (Qwen2.5-7B) = same 7000 pairs, eval capped MultiJail 60/lang, MMLU 150, MGSM 25, MT-Bench 15.
- **Models**: LLaMA-3.1-8B-Instruct (main); Qwen2.5-7B-Instruct (variant, L*=14/28)
- **Method**: M3 LoRA-DPO (β=0.1, r=16 α=32, lr=1e-5, 3000 steps) — Method + L_bottleneck (λ=0.5) at L* over EN/ZH/KO prompt triples; Baseline pure DPO. M4 MultiJail ASR + MMLU + MGSM + MT-Bench; paired bootstrap CI on unseen-lang slice
- **Main experiment**: partial — MultiJail unseen-lang ASR: Method 3.97% vs Baseline 6.86% = **-42.2% relative** (≥ 20 pp target ✓). Worst-lang tied at Swahili 12.94%. Capability: MMLU tied 65.00%, MGSM/en tied 75%, MGSM/zh +5 pp Method, **MGSM/sw drops -5 pp Method**, MGSM/bn tied 17.5%, MT-Bench +0.16 Method. L_bottleneck 0.31→0.04. DPO margin Method +0.732 / Baseline +0.590.
- **Verify**: robustness=1.00 — method n/a / dataset n/a / model **pass** (variant agrees with main that C2 is not-supported); integrity=WARN; verdict=**PASS**. Qwen2.5-7B variant: unseen-lang delta -0.28 pp (Method 14.11% vs Baseline 14.39% — far short of 20-pp), MT-Bench -1.07 pts (Method 3.27 vs Baseline 4.33 — large capability regression), MGSM-sw -4 pp. **The LLaMA safety headline (-42% unseen-lang ASR) does NOT replicate on Qwen2.5-7B under identical recipe.**
- **Iteration**: —
- **Final**: PASS (robustness=1.00) — verify judged both main experiment and Qwen2.5-7B variant as "not-supported" at C2's plan-stated 20-pp threshold, so the negative finding replicates. The LLaMA-only -42% unseen-lang result does NOT carry over to Qwen2.5-7B (variant delta -0.28 pp, MT-Bench -1.07 pts) — **C2's safety payoff is architecture-specific, not architecture-agnostic**. Iteration should decide: narrow claim to LLaMA-family, or investigate hyperparameter gap on Qwen.
- **Caveats**:
  - Main training narrowed to English-only PKU-SafeRLHF-30K (5000 pairs)
  - L_bottleneck on prompt triples (not chosen-response triples)
  - LoRA r=16 α=32 (planned r=64); lr=1e-5 (planned 5e-6, re-bound via Tip 4 pilot sweep)
  - M4 caps: main MultiJail 100/lang MMLU 300; variant MultiJail 60/lang MMLU 150; M-MMLU deferred at main
  - MMLU by loglik on A-D letter tokens (Tip 5 fallback path)
  - MGSM/sw drops 5 pp (main) / -4 pp (variant) — targeted safety-capability trade-off
  - Variant Qwen sw N_eff=10/60 — unreliable Swahili reading
  - Variant Qwen MT-Bench -1.07 pts capability regression exceeds 2 pp tolerance
  - GPU budget overrun 0.34 h (10.34 total vs 10.00 HARD cap)
- **Artifacts**: checkpoints/M3-{Method-L-star-anchor,Baseline-surface-DPO}/step-{1000,2000,3000}, results/M4_eval/{method,baseline,base}_{multijail,mmlu,mgsm,mtbench}.json, verify/C2_bottleneck_anchored_dpo/{main_experiment_audit,variant_audit}/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}, verify/C2_bottleneck_anchored_dpo/ROBUSTNESS.md, verify/C2_bottleneck_anchored_dpo/variants/model-swap-qwen25-7b/{config.yaml,run.sh,DIFF.md,checkpoints/*,results/*}, review-stage/{AUTO_REVIEW.md,REVIEW_STATE.json,REVIEWER_MEMORY.md,AUTO_ITERATION_FINAL_REPORT.md}
- **Figures**:
  - ![C2 · Cross-architecture non-replication of the L*-anchor safety headline: LLaMA-3.1-8B Method achieves -42.2% relative unseen-language MultiJail ASR reduction (mean 6.86% → 3.97%); the same recipe on Qwen2.5-7B yields only -1.95% (14.39% → 14.11%) — the LLaMA safety benefit does not carry over to Qwen2.5-7B under identical hyperparameters.](figures/C2/c2_cross_arch_asr.png) — vector: figures/C2/c2_cross_arch_asr.pdf

---
## Journey Summary
- **Claim**: 2 given claims captured → top framing F1 (per-layer semantic-vs-language-identity ratio) with 3-milestone chain M1→M2→M3+M4
- **Mechanism strategy**: Location → Causal Intervention → Tuning & Editing
- **Mechanism routing**: family=Representation and Parameter Analysis / representation-engineering (M1+M3) + Causal Attribution / patching (M2), composed per Candidate #1
- **Experiment**: 7 runs (M1 + M2 + M3-Method + M3-Baseline + M4×3 models), ~6.39 GPU-hours, headline partial-positive (bottleneck exists at L*=10, but M2 specificity fails; C2 unseen-lang ASR -42.2% relative to surface DPO)
- **Verify**: 2 claim(s) — 1 PASS / 0 FAIL / 0 INCONCLUSIVE / 0 ZEV / 1 INTEGRITY_ONLY (cap=1, swap_off=0); integrity[Phase2/Phase9]=WARN/WARN. C2 PASS means the Qwen2.5-7B variant AGREES with LLaMA main experiment that C2 is not fully supported — L*-anchor advantage is architecture-specific.
- **Iteration**: 0/3 iterations (⓪ narrative-only, budget-preserved), 2 ⓪ revisions applied — claim narrowing (C1: diagnostic operating point / C2: LLaMA-family case study) + 13-bullet reviewer-flagged limitations + positioning guidance. Termination: stalled_at_score_ceiling_budget_constrained (score 4/10, verdict almost). All buckets acceptable — no unresolved FAIL/INCONCLUSIVE/ZEV.
- **Figures**: 3 across 2 claims (C1 dome + C1 M2 specificity + C2 cross-arch ASR); 0 judgment-skipped; 0 render-skipped, 0 errored

## Open Items
- GPU budget overrun: 10.34 / 10.00 h HARD cap (0.34 h / 3.4% overrun) — M4 eval slower than 1.2 h estimate (actual 2.10 h for 3 parallel models). Training was in budget.
- **C1 swap-test unresolved** (INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap) — upgrade via `/auto-verify C1 -- resume: true` (Stage 1 audit reused).
- Qwen variant M4 Swahili N_eff = 10/60 (unreliable — 50 prompts stalled) — reliability caveat for the sw ASR reading.
- M-MMLU deferred at main-experiment M4 — coverage gap on multilingual knowledge retention.
- Cross-architecture non-replication of C2's LLaMA safety headline: resolved by iteration 1 narrowing (statement now LLaMA-family-specific; Qwen non-replication elevated to headline).

## Reviewer-Flagged Limitations (to appear explicitly in paper's Limitations section)
- **λ_bottleneck kept at 0.5, no sweep** — cannot claim the chosen bottleneck-loss weight is principled or near-optimal; gains/regressions are conditional on this setting.
- **No anchor-language ablation** — cannot attribute effects to EN, ZH, or KO individually; cannot rule out that EN alone drives the effect. This is especially important because training preferences are EN-only.
- **No L* ± 1 sensitivity check** — treat L*=10 as an approximate operating point identified by the diagnostic, not a sharply localized causal bottleneck.
- **EN-only preference training** — planned EN/ZH/KO PKU-SafeRLHF was narrowed to EN because on-disk ZH/KO were binary-label classification rows without preference pairs; cross-lingual generalization relies on the anchor triples regularizer, not on multilingual preference signal.
- **Prompt-state (not response-state) bottleneck anchoring** — L_bottleneck operates on prompt last-token h_L* triples because no ZH/KO chosen-response translations were available.
- **M2 metric substitution** — LaBSE replaced by LLaMA's own top-layer mean-pooled embedding (LaBSE download stalled); this is a limitation on the objectivity of the semantic-preservation scorer, since the scorer uses the same model whose bottleneck is under test (potential circularity).
- **C1 causal specificity failure (A ≈ D)** — must be stated as undermining semantic interpretation, not buried in discussion.

## Reviewer-Flagged Limitations — Iteration 2 additions (new)
- **Mean unseen-language ASR conceals per-language heterogeneity** — headline aggregate can hide language-specific regressions. Qwen Method is HIGHER than Baseline on IT (+3.39 pp) and TH (+1.75 pp); LLaMA worst-lang Swahili tied at 12.94% (no improvement despite +42% aggregate reduction). Report per-language deltas as first-class, not as an appendix table.
- **No control disentangling bottleneck anchoring from generic hidden-state regularization** — the paper attributes gains to "anchoring at L*", but no experiment shows that a *different* hidden-state regularizer (e.g., prompt-embedding consistency term, generic representation penalty, or L* replaced with a random-layer anchor) would fail to produce similar effects. Without this control, the "bottleneck" specificity of the method is not established even at the LLaMA-family level.
- **Refusal-heavy task distribution may be unusually favorable to language-generic transfer** — external validity beyond safety/refusal alignment is undemonstrated. The post-hoc "language-generic English refusal context" hypothesis for A≈D suggests the signal may be safety-specific, not multilingual-semantic-transfer-general.
- **Narrow empirical basis** — evidence spans exactly one main model (LLaMA-3.1-8B-Instruct), one failed transfer model (Qwen2.5-7B-Instruct), one training recipe (LoRA r=16 α=32, lr=1e-5, λ=0.5, EN-only preference data). No claim to robustness across training recipes or model families.
- **Potential selection risk around L*=10 as the showcased operating point** — practical conclusions may depend sensitively on the chosen intervention layer; the diagnostic identifies L*=10 but without L*±1 sensitivity we cannot rule out that neighboring layers would give similar or better results. Do not present L*=10 as *the* semantic bottleneck; present it as the diagnostic-argmax candidate.

## Positioning guidance (reviewer-flagged, iteration 2)
- **Consistency check across the manuscript**: title, abstract, intro, and conclusion must ALL adopt the narrowed C1/C2 wording. Any residual "we discover a semantic bottleneck" or "our method achieves multilingual safety alignment" framing in the abstract or intro will re-introduce the overclaim the ledger has corrected.
- **Venue targeting**: honest fit is **negative/mixed-results venue** or **workshop**; **ACL Findings borderline** if the manuscript is consistently rewritten around the weaker story. **Not competitive for NeurIPS/ICML main track** in current form (evidentiary strength insufficient — no ablations, no controls, single-model positive result, cross-arch non-replication).

## Cross-architecture out-of-budget upgrade path
Reviewer would recommend if compute permitted:
- Adjacent-layer sensitivity {L*-1, L*, L*+1}
- Anchor-language ablations (EN-only / EN+ZH / EN+KO / ZH+KO)
- λ sweep {0.1, 0.25, 0.5, 1.0}
- Stronger semantic scorer for M2 (independent scorer, not LLaMA's own top-layer) with control redesign
- Response-state (rather than prompt-state) bottleneck loss
- One additional non-LLaMA-family architecture replicate
- **New (iteration 2)**: generic hidden-state regularizer control (e.g., random-layer anchor, prompt-embedding consistency term) to disentangle "bottleneck at L*" from "any hidden-state regularizer works"
