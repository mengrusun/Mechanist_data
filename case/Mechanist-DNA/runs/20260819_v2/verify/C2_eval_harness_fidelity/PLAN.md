## Claim C2: The fast online α-helical-content metric (ORF→translate CDS→sequence-based SS predictor→%H) agrees with structure-based DSSP %H at Pearson r ≥ 0.7 on held-out DSSP-annotated proteins, with correct reading-frame recovery.

**Main-experiment verdict**: supported (r=0.987, n=200 held-out, frame recovery 1.000). Phase 2 combined = WARN (admitted).

### Main experiment (from /auto-experiment, E1)
- Method: ESM2-650M frozen embeddings + linear 3-state SS probe → %H
- Dataset: 200 held-out RCSB X-ray proteins (DSSP-labeled), disjoint from 600 train
- Model (SS predictor): facebook/esm2_t33_650M_UR50D  (NOTE: Evo2-7B is NOT involved in C2)
- Metric: Pearson r(fast %H, experimental-DSSP %H) = 0.987

### Dimensions tested: method, dataset  (model axis EXCLUDED — Evo2-7B HARD-pinned project-wide; and C2 does not use Evo2 at all)

### Variants  (revised after Phase 4 reviewer critique)
| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | method | **GOR-style windowed statistical SS predictor** — logistic regression on a ±8-residue one-hot amino-acid window, **fit on the SAME 600 training proteins' DSSP labels** → per-residue H/E/C → %H | ESM2-650M frozen-embedding linear probe | Reviewer rejected raw Chou-Fasman as an unfairly weak strawman; a competitive independently-developed predictor (PSIPRED/NetSurfP) was infeasible offline (weights CDN-blocked, matching the documented env limitation). A **data-fit GOR-style windowed classifier** is the strongest fully-offline SS-prediction method that shares NO machinery with ESM2 (no protein-LM, no learned embeddings) yet is fair (fit on the actual training distribution, Q3≈0.62–0.66). Tests whether "a fast sequence-based %H estimate tracks DSSP %H" is general or ESM2-specific. | GOR (Garnier-Osguthorpe-Robson) family; sklearn logistic on aa-window features |
| 2 | dataset | Fresh **deduplicated** disjoint held-out set of unseen RCSB proteins (e1_val indices ≥200, near-duplicate aa[:60] keys dropped), same experimental-DSSP labels | Original 200 held-out eval proteins | Same ESM2+probe metric on proteins never scored in E1 → tests distribution robustness of the r=0.987 correlation. Redundancy control (drop near-identical chains) per reviewer. | data/e1_val.json (510 held-out available; 200 used by E1) |

**Phase 4 disposition:** method swap REVISED (Chou-Fasman → data-fit GOR-style, addressing the "harsh strawman" reject); dataset swap ACCEPTED with dedup control. A fully competitive independent predictor (PSIPRED/NetSurfP) was not runnable in this offline/CDN-blocked environment — the GOR-style predictor is the strongest feasible independent method and is interpreted as a fair-but-classical lower bound.

### Success Criterion (per variant, frozen claim)
A variant SUPPORTS C2 iff the fast sequence-based %H estimate correlates **positively** with experimental-DSSP %H at **Pearson r ≥ 0.7** on its held-out set (the claim's own pass bar), with correct reading-frame recovery where applicable. `/result-to-claim` judges `claim_supported`; because main_experiment_verdict = supported, `consistent = claim_supported`.

### Notes on scope
- C1 (PRIMARY causal steering) is INCONCLUSIVE at Phase 2 and is NOT stress-tested this pass; the orchestrator's Steering-family method-swap guidance (probe→contrastive vector) applied to C1 and is moot here.
- C2 is a pure eval-harness claim (mechanism-audit N/A), so the "within-mechanism-family" method-swap constraint does not bind; the method swap is chosen as the strongest independent SS-prediction alternative.
