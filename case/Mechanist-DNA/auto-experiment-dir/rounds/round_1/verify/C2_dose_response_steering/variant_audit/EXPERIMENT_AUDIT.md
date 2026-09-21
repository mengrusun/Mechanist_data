# Experiment Audit Report — Claim C2 (variant: method-swap-ssdef-pydssp)

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via llm-chat MCP)
**Project**: Feature Steering an α-Helix Knob in Evo2-7B
**Claim**: C2 — Amplifying the C1 α-helix feature set during Evo2-7B autoregressive DNA generation increases the encoded-protein α-helix fraction, monotonically with amplification strength up to an optimum.
**Scope**: variant `method-swap-ssdef-pydssp` (SS-assignment algorithm swap: mkdssp → pydssp on identical ESMFold-predicted structures)

## Overall Verdict: WARN
*This is the variant's integrity verdict — whether its experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
Same as C2's main-experiment audit: helix/sheet fractions are computed from ESMFold-**predicted**
structures, not experimental ground truth (inherent to a generation experiment, not a new issue
introduced by this variant). pydssp is confirmed to be a genuinely independent computation on the
same backbone coordinates (does not consume mkdssp's output or vice versa) — no "laundering" of one
tool's result through the other.

### B. Score Normalization: PASS
Simple means of per-sample fractions over gate-passing samples; no self-referential normalization.

### C. Result File Existence: PASS
All 10 (α,seed) result JSONs and sequence JSONLs exist; `variant_summary.json`'s consolidated numbers
match the raw per-cell files exactly; no missing/crashed cells.

### D. Dead Code Detection: WARN
All computed quantities (helix/sheet under both tools) are saved and aggregated — no genuine dead
code. Two minor asymmetries noted: (1) the final `n_resolved` field always comes from the mkdssp path
(`fr_dssp["n_resolved"]`), pydssp's own residue count is not separately retained (immaterial since
both operate on the same structure); (2) a sample only counts toward either tool's aggregate if
**both** `fr_dssp` and `fr_pyd` succeed (`if fr_dssp and fr_pyd`) — the retained set is the
intersection of both tools' successes, which is defensible for a head-to-head comparison but could
mask a tool-specific failure-rate difference.

### E. Scope Assessment: WARN
Fair as a **targeted SS-assignment-algorithm robustness check**, not a full replication of the main
experiment: grid is 5 doses × 2 seeds × n=150 = 10 cells vs the main experiment's 8 doses × 3 seeds.
The steering hook, frozen feature set, prompt source, translation/ORF filter, and structure predictor
are substantially confirmed identical to the main experiment from the code (same function calls,
same argparse defaults documented as matching `code/m2_dose_response.py`/`code/mechanism.py`). Language
around this variant should describe it as a targeted robustness check, not a full re-verification of
C2 at main-experiment scale — this report does so.

### F. Evaluation Type: synthetic_proxy
Same as main experiment: proxy structural evaluation on ESMFold-predicted structures, gated by pLDDT.

## Action Items
- Retain pydssp's own `n_resolved` separately (not just mkdssp's) for future diagnostic use.
- Consider reporting each tool's independent success/failure rate (not just the intersection) to
  check whether one SS-assignment algorithm fails more often on certain structures.
