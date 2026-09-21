# Final Proposal — Verifying Representation-Level Circuit Breakers (RR) as a Safety Intervention

**Date**: 2026-07-15
**Behavior-source**: given
**Mechanism**: discovery
**mechanism_strategy**:
  directions: [Location, Causal Intervention, Tuning & Editing]
  rejected:
    - Formation Tracing — origin/training-time genesis of the harmful subspace is out of scope; the claims are about whether RR works, not how the subspace formed.
    - Unit Interpretation — SAE-level "what does this feature mean" is not required; RR operates on directions/subspaces without needing named units.
    - Decision Auditing — trustworthiness audits are downstream of the safety verdict, not part of the four claims.
  note: The plan first *locates* the residual-stream sites where harmful vs. benign inputs diverge (probing / mean-difference), then *causally intervenes* (RR fine-tune + activation patching on the located sites) to verify the sites carry the behavior, and finally treats RR itself as a *Tuning & Editing* operation whose transfer is measured on VLM (Claim 3) and agent (Claim 4) settings.

**resource_fidelity**: (not stamped — this is BEHAVIOR_SOURCE=given + MECHANISM=discovery, cost-aware. Task.md's named resources are the *preferred* full-scale set; downscaling is only allowed if the declared 10 GPU-hour budget genuinely does not cover the full run.)

## 1. Problem Anchor (frozen; do not drift)

Claim: A safety intervention that operates on the **internal representations** of an instruction-tuned LLM — Representation Rerouting (RR) — is more robust than output-level defenses (refusal training, adversarial training) against unseen adversarial attacks, and this robustness transfers cross-modality to VLMs and cross-scaffold to LLM agents. The four sub-claims from `task.md` are the target — this proposal verifies them, it does not extend them.

**Anchor guardrails.** Nothing below re-scopes the four claims. Phase 4.5's job is to refine *how to test them*, not to sharpen or narrow *what they say*. Any claim strengthening / narrowing surfaces as a Notes item in `IDEA_REPORT.md`, not as a silent edit.

## 2. Method Thesis (one sentence)

Fine-tune Meta-Llama-3-8B-Instruct with a Representation Rerouting objective — computed only from paired benign/harmful examples with no attack exposure — on a set of residual-stream sites that pre-training probing shows carry the harmful-vs-benign distinction; then measure (a) the reroute empirically on activations (C1), (b) HarmBench ASR + MT-Bench/MMLU (C2), (c) LLaVA-NeXT-Mistral-7B image-hijack robustness (C3), and (d) agent function-calling harm rate + BFCL (C4).

## 3. Dominant Contribution (of the verification, not the invention)

A *self-contained verification harness* that ties each of the four claims to a specific, falsifiable measurement — with mechanistic diagnostics that confirm the reroute really happens (C1) rather than only measuring safety as a black-box surface metric.

## 4. Intentionally Rejected Complexity

- **We do not train from the target paper's repo, weights, or Cygnet checkpoints** — they are on the project forbidden-URL list (blind-reproduction policy). The RR objective is reconstructed from its publicly-documented pre-cutoff principles (rerouting harmful activations toward orthogonal subspaces, jointly with a retain loss preserving benign activations).
- **We do not do Formation Tracing** on the harmful subspace (out of scope for the four claims — the claims are about the intervention's effect, not its training-time origin).
- **We do not train SAEs** on the located sites — Unit Interpretation is not required; the RR subspace is characterized by direction / low-rank basis, not by a named-feature dictionary.
- **We do not build a bespoke adversarial-image benchmark** — task.md pins the PGD ε=32/255 × 1000 steps protocol; we run exactly that.

## 5. Method Detail (the pieces the experiment plan will execute)

### 5.1 Reconstructed RR objective (independent of the forbidden repo)

For each paired batch (benign prompt b, harmful prompt h) drawn from a *constructed* paired benign/harmful dataset (see 5.2):
- Compute residual-stream activations at the chosen sites `S = {s_1, ..., s_k}` (default: mid-to-late layers, e.g. layers 10–20 out of 32; selection refined by the pre-training probe in M1).
- **Rerouting loss on harmful inputs**: penalize the cosine of the post-fine-tune residual activation at each site in `S` on `h` with the *pre-fine-tune* harmful direction `d_h` (mean-difference direction extracted at M1). Formally: `L_rr = mean_{s in S} cos(a_s^{tuned}(h), d_h^{s, base})^2` — pushing harmful activations to be orthogonal to the pre-tuning harmful direction.
- **Retain loss on benign inputs**: MSE (or cosine distance) between tuned and base residual activation at the same sites on `b`, plus a token-level LM loss on `b` to preserve fluency: `L_ret = mean_{s in S} ||a_s^{tuned}(b) - a_s^{base}(b)||^2 + λ_lm * CE(base_logits(b), tuned_logits(b))`.
- Total: `L = α * L_rr + β * L_ret`. Default `α = β = 1`, `λ_lm = 1`.
- Optimizer: LoRA (rank-16, α-32) on the projection matrices of the sites' surrounding layers to keep the fine-tune cheap on the 10 GPU-hour budget.

This objective is a *plausible reconstruction* of RR consistent with the target claims but is not asserted to reproduce the target paper's exact loss. The claim is about the *behavior* of "rerouting to an orthogonal non-harmful subspace using only paired benign/harmful data" — this loss is one implementation of that behavior. Any per-hyperparameter uncertainty is licensed as `method_sensitive` in the experiment plan.

### 5.2 Training data (constructed pairs — the forbidden repo cannot be cloned)

Task.md pins the GraySwanAI training + retain set as "always used, fixed", but the repo itself is on the project forbidden-URL list. Reconciliation:
- **Preferred**: check if a *local copy* of the GraySwanAI paired benign/harmful data lives under `/data/zhenqian/data/` (task.md says missing artifacts may be downloaded there). If a cached local copy exists, use it directly.
- **Fallback**: construct an equivalent paired benign/harmful set from HarmBench's `behaviors.csv` (public part) and public benign-instruction corpora (e.g. UltraChat single-turn), matched pair-wise on length and format. ~5k pairs total.

This conflict between the "training set is fixed" pin and the "repo is forbidden" pin is surfaced as a Notes item in `IDEA_REPORT.md` and must be resolved by the orchestrator / user at Round-End Decision — but the pipeline proceeds with the fallback so the compute budget is not wasted stalling on a data-source decision that can only be made outside the claim stage.

### 5.3 Baselines (Claim 2)

- **B0 — Refusal-only baseline**: Meta-Llama-3-8B-Instruct as released (already RLHF-safety trained). No further fine-tune.
- **B1 — Adversarial-trained baseline**: reproduce an R2D2-style adversarial fine-tune: on the same LoRA setup, fine-tune with (harmful prompt + GCG-optimized adversarial suffix, safe refusal) pairs — 512 GCG suffixes generated in-house against B0 on 128 harmful prompts drawn from HarmBench train (disjoint from eval).

### 5.4 Sites selection (Location)

At M1, extract mean-difference directions `d_h^s = mean_h(a_s(h)) - mean_b(a_s(b))` on residual-stream activations at every layer of B0 using 512 held-out (harmful, benign) pairs. Compute per-layer separation AUC via a linear probe. Sites `S` = top-k layers by AUC with a contiguous middle-band constraint (default k=6 layers, layers 10–20). This is `method_sensitive` in the experiment plan.

### 5.5 VLM / agent instantiation (Claims 3, 4)

- **VLM (C3)**: LLaVA-NeXT-Mistral-7B is the target. **The base LLM inside LLaVA-NeXT-Mistral-7B is Mistral-7B-Instruct-v0.2, not Llama-3-8B-Instruct**, so the LLM RR fine-tune on Llama-3-8B does not transfer weights directly. Instead we apply RR to the *Mistral-7B* base of LLaVA-NeXT-Mistral-7B (same objective, same construction of paired data), then evaluate PGD image-hijack against the RR-tuned VLM. If time permits under the 10 GPU-hour budget, also run against the un-RR-tuned VLM as a baseline. Task.md licenses this reading — the verify stage lists both Mistral-7B-Instruct-v0.2 and LLaVA-NeXT-Mistral-7B in the candidate pool.
- **Agent (C4)**: Apply the *same* Llama-3-8B RR fine-tuned weights inside a function-calling harness — the harness is a scaffold over the LM, so no additional fine-tune is needed.

## 6. What the four claims each need

| Claim | Verified by milestone(s) | Primary metric | Success criterion |
|---|---|---|---|
| C1 (identifiability + reroute) | M1 (Location) + M4 (mechanistic diagnostic) | Layer-wise probe AUC (pre-RR); cosine of tuned harmful activation with pre-RR harmful direction (post-RR) | AUC > 0.8 at ≥ 3 mid-late layers; mean cosine drop ≥ 0.3 on harmful, benign cosine drift ≤ 0.1 |
| C2 (ASR + capability) | M2 (baselines), M3 (RR train), M5 (safety+capability eval) | HarmBench ASR (6 attack categories), MT-Bench, MMLU | RR ASR ≥ 20 pp lower than B0 aggregate; ≥ 10 pp lower than B1 on unseen categories; MT-Bench Δ ≤ 0.3, MMLU Δ ≤ 2 pp |
| C3 (multimodal transfer) | M6 (VLM RR + image-hijack eval) | PGD-image ASR on harm set; VLM capability delta | ≥ 15 pp ASR reduction; VLM capability Δ within tolerance |
| C4 (agent transfer) | M7 (agent harm probe + BFCL) | Harmful-tool-use rate on 100-prompt set; BFCL score | ≥ 20 pp reduction on harm set; BFCL Δ ≤ 3 pp |

## 7. Compute Budget (10 GPU-hour total, GPUs 0–3)

Rough allocation (revised in `EXPERIMENT_PLAN.md`):

| Milestone | GPU-hours | Notes |
|---|---|---|
| M1 (Location: probing + sites selection) | 0.5 | 512 pairs × forward pass × 32 layers on 1 GPU |
| M2 (B1 baseline: GCG generation + adv fine-tune) | 2.0 | LoRA fine-tune on 1 GPU |
| M3 (RR fine-tune on Llama-3-8B-Instruct) | 2.0 | LoRA fine-tune on 1 GPU |
| M4 (mechanistic diagnostic on RR-tuned model) | 0.3 | forward passes only |
| M5 (HarmBench ASR + MT-Bench + MMLU) | 2.5 | eval only, batched across GPUs 0–3 |
| M6 (VLM RR + PGD image hijack) | 2.0 | Mistral-7B RR + PGD attack loop |
| M7 (agent harm probe + BFCL) | 0.5 | eval only |
| Reserve | 0.2 | contingency |
| **Total** | **10.0** | |

## 8. Risks

- **R1 Data-source conflict**: task.md pins the GraySwanAI training set as fixed, but the repo is on the forbidden-URL list. Mitigated by 5.2's fallback; explicitly flagged as an unresolved conflict for orchestrator review.
- **R2 Reconstructed-RR gap**: our reconstructed RR loss may not match the target paper's exact formulation. Mitigated by tying the *claim* to the *behavior* ("rerouting to an orthogonal non-harmful subspace") — any implementation that produces that behavior on the diagnostic (M4) satisfies C1.
- **R3 GCG-suffix generation cost**: even 512 suffixes at 200 steps each is expensive. Mitigated by reusing pre-computed HarmBench GCG suffixes if the local HarmBench copy includes them.
- **R4 Compute-budget tightness**: 10 GPU-hours is tight for 4 fine-tunes + 4 evaluations. Mitigated by LoRA (not full FT), batched evaluation across GPUs, and by dropping M6 (multimodal) to a smaller harm-probe size if M1–M5 overrun.
- **R5 VLM base-model mismatch**: RR fine-tune of Llama-3 does not directly transfer to LLaVA-NeXT-Mistral. Mitigated by 5.5 (apply RR separately to the Mistral base inside the VLM).
