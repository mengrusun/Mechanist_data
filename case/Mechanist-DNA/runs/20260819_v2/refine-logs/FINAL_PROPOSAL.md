# Final Proposal — Locating and Causally Steering α-Helical Propensity in Evo2-7B

<!-- machine markers (top metadata) -->
resource_fidelity: cost-aware   <!-- NOT strict: reproduction harness applies only to behavior-source=given AND mechanism=given; this run is given + discovery -->
behavior_source: given
mechanism: discovery
chosen_mechanism: n/a   <!-- mechanism:discovery — family chosen later by experiment-stage /mechanism-skills routing -->
mechanism_strategy:
  directions: [Location, Causal Intervention, Tuning & Editing]   # in execution order
  rejected:
    - Formation Tracing — the goal is inference-time control of generation, not the training-time genesis of the α-helix representation; genesis is out of scope and the most expensive direction.
    - Unit Interpretation (as a standalone end) — we do decode/name the α-helix direction as part of Location, but interpreting units is not itself the deliverable.
    - Decision Auditing — there is no trustworthiness / spurious-feature audit question here; the task is generation control, not decision validation.
  note: The task is to USE an internal α-helix direction to raise generated α-helical content (Tuning & Editing), earned by first locating a candidate direction (Location) and causally confirming it with a dose-response + specificity controls (Causal Intervention).

**Model (HARD constraint — binding, no substitution)**: Evo2-7B (`arcinstitute/evo2_7b`).
**Behavior-source / Mechanism**: given / discovery. **No M0 phenomenon-validation gate** (behavior taken as given).
**Date**: 2026-08-19

## Problem Anchor (frozen)
Given behavior (assumed to hold, from task.md): *Evo2-7B can generate DNA sequences whose encoded protein has higher α-helical content.* The scientific job of this project is to **find and causally exploit the internal component of Evo2-7B that controls α-helical propensity**, and to demonstrate a controlled increase in α-helical content of generated sequences that is specific and not a decoding artifact. The behavior/claims (see `idea-stage/IDEA_REPORT.md`, Claims 1–2) never move; only the testing method is refined here.

## Thesis (one sentence)
A low-dimensional, causally-effective "α-helical propensity" direction is represented inside Evo2-7B; locating it and adding/clamping it during autoregressive generation raises the α-helical content (%H) of generated coding sequences monotonically and specifically, giving a controllable route to high-α-helix DNA design.

## Why this is the right altitude and strategy
- **Altitude**: the claim asserts that *some* internal component (a feature / activation direction / subspace at one or more layers) carries and causally drives α-helical propensity — it does **not** pre-commit to a specific layer or a specific SAE feature index. Which locus and which family win is exactly what the Location + routing work discovers.
- **Strategy (from `/mechanism-explore`)**: this is a **Capability/Editing** task ("generate … with higher α-helical content") validated by **Mechanistic evidence**. Hence **Location → Causal Intervention → Tuning & Editing**: (1) Location = cheap correlational screen to nominate candidate directions/features/layers; (2) Causal Intervention = steering dose-response + specificity to promote a *located* candidate to a *causal* control knob; (3) Tuning & Editing = apply the confirmed knob to generate a high-α-helix library and quantify the achievable uplift/validity trade-off.

## Method (testing approach, refined)

### Candidate mechanism families (routed later by /mechanism-skills — do NOT commit here)
All three are consistent with the strategy; the experiment stage picks and may combine them. Listed to make the plan concrete without pinning a family:
1. **SAE-feature clamping** — use the public `Goodfire/Evo-2-Layer-26-Mixed` BatchTopK SAE; identify the feature(s) whose activation separates high-α-helix from low-α-helix coding windows; clamp/boost that feature during generation.
2. **Contrastive steering vector** — mean-difference of Evo2-7B activations between high-α-helix and low-α-helix coding sequences at candidate layer(s) (ARCADE / Steering-PLM recipe); add α·v to the residual stream during generation.
3. **Linear-probe direction** — train a linear probe for per-position α-helix propensity on hidden activations; steer along the probe weight direction.

### Direction-extraction data (labeled, contrastive)
- Source proteins with known secondary structure from **DSSP-annotated PDB**, grouped by structural class (e.g. **all-α vs all-β / low-helix** via CATH/SCOP), mapped to their **coding DNA (CDS)**; translate-back / codon-optimize where the native CDS is unavailable. This yields matched high-%H vs low-%H nucleotide windows for direction extraction and probe training. Held-out split reserved for evaluation.

### α-helix-content evaluation harness (Claim 2)
- **ORF → translate → predict SS → %H**: locate the ORF/reading frame in the generated DNA, translate to protein, run a **fast sequence-based SS predictor** (e.g. NetSurfP-3.0 / S4PRED) for the online steering objective (%H = fraction of residues in H).
- **Structure-grounded confirmation**: on a held-out subset, fold with **ESMFold/AlphaFold2** and assign SS with **DSSP** (H/G/I → helix); require strong correlation between the fast metric and DSSP %H before trusting steering gains.
- Report distributions (not single numbers); track ORF validity, sequence length, GC, and Evo2-7B coding-likelihood as covariates.

### Causal test (Claim 1)
- Generate N sequences per condition from matched prompts; conditions = baseline (no steering) and a **sweep of steering coefficients** α (dose-response). Primary readout: mean %H vs α. Expect monotone increase over a working range, then validity degradation at large α.
- **Specificity battery**: (a) matched **random/control direction** at equal norm → no significant %H gain; (b) **off-target**: β-sheet %E and other properties not the sole mover; ORF validity and coding-likelihood preserved; (c) **confound control**: decoding temperature/top-p fixed across conditions, and compared against a **naive baseline** (temperature change / rejection sampling toward %H) to show the mechanism adds beyond trivial sampling.

## Resources & scale (cost-aware, but full-scale here)
- **Model**: Evo2-7B at full size (HARD constraint). 8×A800-80GB comfortably runs 7B inference on short coding windows (~300–1500 bp); the declared budget covers full-scale runs, so **no downscaling** of the model. Do not substitute a smaller Evo variant.
- **Data**: full labeled contrastive set at the sizes the evaluation power requires (resolve exact n at routing; floors in the plan).
- This run is **cost-aware, not `resource_fidelity: strict`** (strict applies only to the given+given reproduction combination). Cost-aware here means *minimize cost subject to the science within budget* — and since budget covers full scale, plan at full scale.

## Constraints carried into the plan
- **HARD**: main experiment MUST use Evo2-7B; encoded as a per-claim binding model constraint in `EXPERIMENT_PLAN.md`.
- **NOTICE**: model/data from HuggingFace (token available); proxy may be needed for downloads; `evo2` not yet installed (experiment stage installs it); Python 3.13 miniconda at `/data/wanghaoxiong/miniconda3`; ~684 GB free under `/data`. These are planning-realism items, authoritative form lives in `EXPERIMENT_PLAN.md`.

## Deliverables
- Confirmed (or refuted) causal α-helix control knob in Evo2-7B with dose-response + specificity (Claim 1).
- Validated α-helix-content evaluation harness (Claim 2).
- A high-α-helix generated-DNA library with quantified uplift and validity trade-off (Tuning & Editing output).

See `refine-logs/EXPERIMENT_PLAN.md` for the milestone-level roadmap and `refine-logs/EXPERIMENT_TRACKER.md` for the run table.
