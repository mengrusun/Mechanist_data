# Pipeline Summary

**Problem**: Verify four claims about Representation Rerouting (RR) — a representation-level safety intervention for instruction-tuned LLMs — from `task.md`.
**Final Method Thesis**: A reconstructed RR fine-tune (LoRA) on Llama-3-8B-Instruct at residual-stream sites where pre-training probing separates harmful vs. benign inputs, evaluated via a mechanistic diagnostic (C1), HarmBench + capability benchmarks (C2), image-hijack on LLaVA-NeXT-Mistral (C3), and function-calling harm probe + BFCL (C4).
**Final Verdict**: READY
**Date**: 2026-07-15

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Idea report: `idea-stage/IDEA_REPORT.md`

## Contribution Snapshot
- Dominant contribution (of verification): the 4-claim verification harness that ties each claim to a specific falsifiable measurement, including a mechanistic diagnostic (M4) so C1 is verified by *observed reroute* not only by *downstream ASR*.
- Optional supporting: matched-control specificity check in M4; per-category ASR breakdown in M5.
- Explicitly rejected: Formation Tracing, Unit Interpretation, Decision Auditing (out of scope), SAE feature dictionaries, use of the target repo / released Cygnet weights (forbidden by project policy).

## Must-Prove Claims
- C1: RR fine-tune produces an identifiable, observable reroute of harmful residual-stream activations toward the orthogonal complement of the pre-tuning harmful direction; benign activations preserved.
- C2: RR ASR under HarmBench aggregate ≥ 20 pp lower than B0; ≥ 10 pp lower than B1 on unseen attack categories; MT-Bench Δ ≤ 0.3; MMLU Δ ≤ 2 pp.
- C3: RR-tuned LLaVA-NeXT-Mistral image-hijack ASR ≥ 15 pp lower than the un-RR baseline; VLM capability delta within tolerance.
- C4: RR-tuned agent harmful-tool-use rate ≥ 20 pp lower than B0; BFCL Δ ≤ 3 pp.

## First Runs to Launch
1. M1 — locate harmful-subspace sites in base Llama-3-8B-Instruct (0.5 GPU-h)
2. M2 — R2D2-style adversarial fine-tune baseline B1 (2.0 GPU-h; can run in parallel with M1)
3. M3 — RR fine-tune (depends on M1; 2.0 GPU-h)

## Main Risks
- **R1 Data-source conflict** (task.md fixed training set is a forbidden repo): Mitigated by constructing an equivalent paired benign/harmful set from HarmBench-public + UltraChat; conflict surfaced honestly to orchestrator.
- **R2 Reconstructed-RR gap**: reconstruction may not equal the target paper's exact loss; C1 is tied to the *behavior* (reroute), not to a specific loss form.
- **R3 Compute-budget tightness**: LoRA + batched evaluation; M6 has budget-guard fallback.

## Next Action
- Return control to the /auto orchestrator; the claim gate output feeds `/mechanism-skills` (routing) → `/auto-experiment` (Workflow 1.5).
