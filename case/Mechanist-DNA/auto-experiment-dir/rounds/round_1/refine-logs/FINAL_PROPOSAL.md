---
title: "Feature Steering an α-Helix Knob in Evo2-7B — Validating Causal Control of Protein Secondary Structure from a Genomic LM"
behavior_source: given-validation
mechanism: given
chosen_mechanism: "SAE feature steering / feature amplification (amplify Layer-26 α-helix-selective SAE features during autoregressive DNA generation; dose-response coefficient sweep)"
mechanism_strategy: n/a
resource_fidelity: not-strict   # given-validation + given ≠ reproduction combo (given+given); model & SAE still pinned by model-fidelity HARD CONSTRAINT, data sizes cost-aware but planned at full scale under the generous 4-GPU budget
max_parallel: 4
gpus: [2, 3, 4, 5]
date: 2026-07-18
claim_source: task.md
---

# Final Proposal

## Problem Anchor (frozen)

`task.md` makes a two-part, causally-loaded claim about **Evo2-7B** and its **pre-trained Layer-26 SAE**:

1. **Existence/selectivity** — within Evo2-7B's Layer-26 SAE there is a **set of features that selectively respond to protein α-helix**.
2. **Causal control** — **amplifying** those features during **autoregressive DNA generation** raises the **α-helical content of the encoded protein**, and this content **increases with amplification strength** (dose-response) up to an optimum — **proving the feature is a causally manipulable knob.**

The anchor is this claim set exactly as written. The proposal refines only **how to test** it; it never weakens, narrows, or strengthens the claim.

## Method Thesis (one sentence)

Identify the α-helix-selective Layer-26 SAE feature set with a rigorous natural-CDS + real-DSSP-label selectivity test (a hard M0 gate), then causally validate it by amplifying those features across a swept dose during Evo2-7B autoregressive DNA generation and measuring — via protein structure prediction + DSSP on the encoded protein — a monotone dose-response rise in α-helix content that is **specific** to the α-helix features (matched-control and β-sheet off-target features do not reproduce it).

## Why it matters (context, not a novelty gate — behavior is `given`)

The Evo 2 paper (Arc × Goodfire) demonstrated α-helix SAE features only **correlationally** (activation maps) and called steering "still in its early stages"; prior causal SAE steering (Interpreting-and-Steering-PLMs, FoldSAE, automated-neuron-labelling, CorrSteer) operates **in-modality on protein LMs**. Establishing a **cross-modal, dose-responsive, specific** causal knob — a *genomic-LM* feature that controls the *downstream protein's* secondary structure — is the open, load-bearing step this project supplies. (Behavior source is `given-validation`, so this is framing/importance, not a novelty requirement.)

## Chosen Mechanism (committed — no routing)

**SAE feature steering / feature amplification.** The mechanism family is fixed by `task.md` and stamped as `chosen_mechanism`; the experiment stage commits it directly as `CHOSEN_FAMILY` (Mode B, no `/mechanism-skills` routing). Concretely: at the Layer-26 SAE site during autoregressive nucleotide decoding, add a positive multiple of the identified α-helix feature direction(s) (equivalently, clamp/scale those latents' activations) with a **swept coefficient α** (dose), and decode the steered residual back into the model's forward pass. This is a **Causal Intervention** on identified interpretable units.

## HARD CONSTRAINTS encoded as plan rules (non-negotiable)

- **HC1 — Model fidelity (positive).** The behavior is defined ON Evo2-7B + its pre-trained Layer-26 SAE. The main experiment uses **exactly** `evo2_7b.pt` and `sae-layer26-mixed-expansion_8-k_64.pt` at `/data1/share_model/evo2`. **No substitute genomic LM or SAE** for the main experiment. (Downstream `/auto-verify` may swap the *analysis* axis per its own design; the plan's main experiment is pinned.)
- **HC2 — Compute allocation.** 8×A800-80GB machine, **≤4 GPUs at once**, pinned to **GPUs 2,3,4,5** (0,1 busy). `max_parallel: 4`. Budget is *generous* — use it fully; do not simplify/abandon experiments to save cost.
- **HC3 — Data-construction prohibition (negative + positive).** For finding the α-helix feature set, **do NOT reverse-translate proteins into DNA**. **MUST** use natural **CDS** from databases with **real DSSP secondary-structure labels** from **experimental protein structures**, attached to codons after **proper CDS↔protein alignment**. Encoded as the M0 data-rule.
- **HC4 — No degradation / human-gated setup.** Do NOT shrink, subset, or skip any experiment to save cost or route around tooling. Missing tooling: **install-and-continue** for pip/conda deps and downloadable weights (evo2, vortex, biopython, fair-esm, ESMFold/AF params, `pydssp`/conda-forge `dssp`); **STOP + approval-needed** only for genuinely `sudo`/apt/root, credential-gated, licensed, or quota-blocked setup. Author at full intended scale.

## Resources

- **Model (pinned):** Evo2-7B, `/data1/share_model/evo2/evo2_7b/evo2_7b.pt` (StripedHyena2).
- **SAE (pinned):** Layer-26 mixed BatchTopK SAE, expansion 8, k=64, ~32,768 features, `/data1/share_model/evo2/evo2_sae_layer26_mixed/sae-layer26-mixed-expansion_8-k_64.pt`.
- **M0 data:** natural CDS (e.g., RefSeq/Ensembl/ENA CDS) whose proteins have **experimental** PDB structures; per-residue **DSSP** labels; codon↔residue alignment. Helix-class definition: DSSP {H,G,I} → helix (report H-only variant too).
- **Verification:** structure prediction (ESMFold default; AlphaFold if available) + **DSSP** α-helix fraction on translated generated proteins; pLDDT gating.
- **Compute:** GPUs 2,3,4,5; `max_parallel: 4`.

## Claims → support map

| Claim | Statement | Verified by |
|---|---|---|
| **C1** | An α-helix-selective Layer-26 SAE feature set exists (selectivity above controls, confound- & FDR-controlled). | **M0** (phenomenon-validation gate) |
| **C2** | Amplifying that feature set during autoregressive DNA generation raises encoded-protein α-helix fraction, monotone in dose, with an optimal strength. | **M1** harness + **M2** dose-response sweep |
| **C3** | The rise is **specific** to the α-helix feature set (matched-control/off-target/β-sheet double-dissociation; generation quality preserved) — a causally manipulable knob. | **M3** specificity controls |

## Testing approach (one paragraph)

The suite climbs the ladder of evidence: a **correlational/attribution screen** (M0) localizes α-helix-selective features against a real DSSP label track on natural CDS, gating everything downstream; a **causal intervention** (M2) amplifies the surviving features across a dose sweep and reads out the effect through the DNA→protein→fold→DSSP pipeline built and calibrated in M1; and **matched-control + off-target + confound checks** (M3) establish specificity and rule out generic-perturbation and quality-collapse explanations. Each intervention milestone records expected **sign** (up), **magnitude/dose-response** (monotone trend + optimum), and a **specificity control**.

## Deliverables
- `refine-logs/EXPERIMENT_PLAN.md` — claim-driven roadmap opening with M(-1) setup-precondition and the **M0** gate.
- `refine-logs/EXPERIMENT_TRACKER.md` — plan-level run table (all `pending`).
