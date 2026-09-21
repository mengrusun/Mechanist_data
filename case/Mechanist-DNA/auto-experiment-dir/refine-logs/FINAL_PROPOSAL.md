---
title: "Hardening the α-Helix Feature-Steering Knob in Evo2-7B — publication-solid causal control of protein secondary structure (Round 2)"
behavior_source: given-validation
mechanism: given
chosen_mechanism: "SAE feature steering / feature amplification (amplify Layer-26 α-helix-selective SAE features at blocks.26.post_norm during autoregressive DNA generation; σ_proj-unit dose-response coefficient sweep)"
mechanism_strategy: n/a
resource_fidelity: not-strict   # given-validation + given ≠ reproduction combo (given+given) → no strict harness; model & SAE still pinned by HC1, data planned at FULL scale (explicit rigor round, no downscaling) under the 4-GPU budget
max_parallel: 4
gpus: [3, 4, 5, 6]
max_verify_claims: 3
date: 2026-07-19
claim_source: task.md
claims: [C1, C2, C3]
round: 2
builds_on: rounds/round_1/
---

# Final Proposal — Round 2 (Rigor Hardening)

## Problem Anchor (frozen)

`task.md` (round 2) pins the **same** causal claim established in round 1 and asks to make it publication-solid. The anchor is unchanged:

1. **Existence/selectivity** — within Evo2-7B's Layer-26 SAE there is a **set of features that selectively respond to protein α-helix**.
2. **Causal control** — **amplifying** those features during **autoregressive DNA generation** raises the **α-helical content of the encoded protein**, and this content **increases with amplification strength up to an optimum** — **proving the feature is a causally manipulable knob.**

The anchor is this claim set exactly as written. Round 2 refines only **how to test** it (rigor, statistics, robustness); it never weakens, narrows, or strengthens the claim. This is a legitimate hardening retry, not concluded work: round 1 deferred the C1/C3 swap-tests, left C3 with a mechanism-audit residual WARN, and never tested the structure-predictor robustness axis.

## Method Thesis (one sentence)

Re-confirm the α-helix-selective Layer-26 SAE feature set S with the mandated natural-CDS + real-DSSP selectivity test (a hard M0 gate that reuses the frozen round-1 S), then causally validate it by amplifying S across a **σ_proj-unit** dose sweep during Evo2-7B autoregressive DNA generation and reading out — via **two independent** structure predictors + DSSP, with **pLDDT folded into the statistics** — a monotone dose-response rise to a **demonstrated interior optimum** that is **specific** to S (a ≥30-direction norm-matched random-null and a matched-control feature do not reproduce it), with **all three claims authored to be swap-testable**.

## Chosen Mechanism (committed — no routing)

**SAE feature steering / feature amplification.** The mechanism family is fixed by `task.md` and stamped as `chosen_mechanism`; the experiment stage commits it directly as `CHOSEN_FAMILY` (Mode B, no `/mechanism-skills` routing). Concretely: at the Layer-26 SAE site (`blocks.26.post_norm`) during autoregressive nucleotide decoding, add a positive multiple of the identified α-helix feature direction(s) with a **swept coefficient α expressed in σ_proj units** (α normalized by the projection std of S's activation at the site), and decode the steered residual back into the forward pass. This is a **Causal Intervention** on identified interpretable units.

## The six required improvements (the point of this round — each encoded as a plan requirement)

1. **pLDDT into the statistics, not just a gate** *(→ M1, M2, M3)*. Report a **pLDDT-weighted** α-helix fraction alongside the hard-gated one; show the α→helix effect is **not confounded by pLDDT** (report mean pLDDT per dose; condition on pLDDT via stratification and as a covariate); run a **pLDDT-threshold sensitivity sweep** so C2's dose-response and C3's specificity are shown stable across gates.
2. **Second, independent structure predictor** *(→ M(-1) pre-stage, M1 wiring, M2, M3)*. Re-measure the C2 dose-response and the C3 interior-dose specificity with **OmegaFold or ColabFold/AlphaFold2** in addition to ESMFold. Pre-stage the predictor + weights in setup and **resolve the OmegaFold torch-pin download problem** (or pick a predictor whose weights download cleanly). The claim is solid only if the effect holds across predictors.
3. **Swap-robustness on all three claims** *(→ claim authoring + `max_verify_claims: 3`)*. C1, C2, C3 are each authored with explicit method/dataset/model swap axes; verify runs with `MAX_VERIFY_CLAIMS=3`.
4. **σ_proj-unit dosing + a mapped interior plateau** *(→ M1 σ_proj calibration, M2 grid)*. Express α in **σ_proj units** and **extend/refine the α grid so α\* is a demonstrated interior maximum** (points bracketing the peak on both sides with capability preserved), not a grid edge.
5. **Tighter statistics on the modest C3 effect** *(→ M3)*. More samples/seeds per dose for narrower CIs; keep the **≥30-direction norm-matched random-null as the PRIMARY C3 specificity statistic**; report effect sizes with CIs.
6. **β-sheet arm resolution** *(→ M0 β re-identification, M3)*. Either re-identify a **stronger β-sheet-selective feature set** and retest the symmetric double dissociation, **or** explicitly frame C3 as **helix-axis specificity** (documented β-negative) — decided at M0/M3 runtime by whether the stronger β set clears the selectivity + manipulation bar.

## Why it matters (context, not a novelty gate — behavior is `given`)

The Evo 2 paper (Arc × Goodfire) demonstrated α-helix SAE features only **correlationally** (activation maps) and called steering "still in its early stages"; prior causal SAE steering operates **in-modality on protein LMs**. Establishing a **cross-modal, dose-responsive, predictor-robust, confidence-aware** causal knob — a *genomic-LM* feature that controls the *downstream protein's* secondary structure — is the open, load-bearing step this project supplies. (Behavior source is `given-validation`, so this is framing/importance, not a novelty requirement.)

## HARD CONSTRAINTS encoded as plan rules (non-negotiable)

- **HC1 — Model fidelity (positive).** The main experiment uses **exactly** `evo2_7b.pt` and `sae-layer26-mixed-expansion_8-k_64.pt` at `/data1/share_model/evo2` (Layer 26, exp 8, k=64, ~32,768 features, site `blocks.26.post_norm`). **No substitute genomic LM or SAE.** (The second *structure predictor* is an added readout axis, not a substitute LM/SAE.)
- **HC2 — Compute allocation.** 8×A800-80GB machine, **≤4 GPUs at once**, pinned to **GPUs 3,4,5,6** (0,2 busy with other users). `max_parallel: 4`. Budget is *generous* — this is a rigor round: run at **full scale**, do not downscale/subset/skip to save cost.
- **HC3 — Data-construction prohibition (negative + positive).** For finding/re-confirming the feature set, **do NOT reverse-translate proteins into DNA**. **MUST** use natural **CDS** + **real DSSP labels** from **experimental protein structures** attached to codons after **proper CDS↔protein alignment**. (M0 may reuse the frozen round-1 S + `data/` caches, which were built under this rule.)
- **HC4 — No degradation / human-gated setup.** Do NOT shrink, subset, or skip any experiment. Missing tooling: **install-and-continue** for pip/conda deps and downloadable weights (evo2, vortex, biopython, fair-esm/ESMFold, `pydssp`/conda-forge `dssp`/`mkdssp`, **the second structure predictor**); **STOP + approval-needed** only for genuinely `sudo`/apt/root, credential-gated, licensed, or quota-blocked setup. Never silently drop the pLDDT stats, the second predictor, or the swap-robustness.

## Resources

- **Model (pinned):** Evo2-7B, `/data1/share_model/evo2/evo2_7b/evo2_7b.pt` (StripedHyena2).
- **SAE (pinned):** Layer-26 mixed BatchTopK SAE, exp 8, k=64, ~32,768 features, `/data1/share_model/evo2/evo2_sae_layer26_mixed/sae-layer26-mixed-expansion_8-k_64.pt`, site `blocks.26.post_norm`.
- **Frozen feature set (reuse):** `rounds/round_1/results/m0_feature_set.json` — S = 19 α-helix features (anchor f/28741), 5 β-sheet features, 19 matched-control features, with per-feature natural-activation scales s_f.
- **M0 data (reuse under HC3):** natural CDS (RefSeq/Ensembl/ENA) with experimental PDB structures; per-residue DSSP labels; codon↔residue alignment; caches under `data/` (16G).
- **Verification tooling:** **ESMFold** + a **second independent predictor** (OmegaFold or ColabFold/AF2), each + **DSSP** (mkdssp primary, pydssp swap); **pLDDT** as gate **and** covariate/weight.
- **Compute:** GPUs 3,4,5,6; `max_parallel: 4`.

## Claims → support map

| Claim | Statement (unchanged from round 1 — hardened testing) | Verified by | Swap axes (verify) |
|---|---|---|---|
| **C1** | An α-helix-selective Layer-26 SAE feature set exists (set-level selectivity above controls, confound- & FDR-controlled, cross-organism). | **M0** (phenomenon-validation gate; reuses frozen S) | DSSP algo; organism dataset; metric/helix-def |
| **C2** | Amplifying S during autoregressive DNA generation raises encoded-protein α-helix fraction, monotone in σ_proj-unit dose to an **interior** optimum — robust across two predictors and pLDDT-weighting. | **M1** harness + **M2** dose-response sweep | structure predictor; DSSP algo; trend stat; seed set |
| **C3** | The rise is **specific** to S vs a ≥30-direction norm-matched random-null + matched-control at the locked interior dose — with tight CIs, both predictors, pLDDT-stratified; β-arm resolved. | **M3** specificity | predictor; null construction; DSSP algo; locked-dose |

## Testing approach (one paragraph)

The suite climbs the ladder of evidence with round-2 rigor layered on: a **correlational/attribution screen** (M0) re-confirms S against a real DSSP label track on natural CDS (reusing the frozen S) and gates everything downstream, additionally attempting a **stronger β-sheet-selective set** for the β-arm decision; a **calibrated harness** (M1) computes the **σ_proj** normalizer at the injection site, wires in the **second structure predictor**, and defines the **pLDDT-weighted** and hard-gated readouts plus the per-dose sample size for target CI width; a **causal intervention** (M2) amplifies S across a **σ_proj-unit** grid refined so the optimum is **interior**, reading out under **both predictors** with **pLDDT folded into the statistics** (weighted fraction, mean-pLDDT-per-dose, pLDDT-conditioning, pLDDT-threshold sweep); and **specificity checks** (M3) at the **locked interior plateau dose** compare S against a **≥30-direction norm-matched random-null (PRIMARY)** + matched-control with **tight CIs and effect sizes**, under both predictors and pLDDT-stratification, and **resolve the β-arm** (stronger β set → symmetric double dissociation, else documented helix-axis specificity). Each intervention milestone records expected **sign** (up), **magnitude/dose-response** (monotone to an interior optimum), and its **specificity control**.

## Deliverables
- `refine-logs/EXPERIMENT_PLAN.md` — claim-driven roadmap opening with M(-1) setup-precondition and the **M0** gate; stamps `chosen_mechanism`; `max_parallel: 4` on GPUs 3,4,5,6; encodes all six improvements as milestone requirements.
- `refine-logs/EXPERIMENT_TRACKER.md` — plan-level run table (all `pending`).
