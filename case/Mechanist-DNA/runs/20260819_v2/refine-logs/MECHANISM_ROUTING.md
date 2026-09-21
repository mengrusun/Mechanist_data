# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Steering features + Steering Vectors
chosen_idea_title: Causally steerable internal α-helix control in Evo2-7B
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/steer-features/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/feature-dictionary-learning/SAE/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / **Steering features + Steering Vectors** — the family whose *write-in* intervention (add α·v to the residual stream, or clamp an SAE feature) is precisely the M2 causal knob. Hosts BOTH plan interventions under one roof: **Steering features** = clamp the Goodfire L26 SAE α-helix feature (encode→amplify→decode); **Steering Vectors** = add a contrastive/mean-difference CAA direction. Directly serves Claim 1's dose-response + specificity.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steer-features/SKILL.md
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
2. Feature Dictionary Learning / **SAE** — the *Location* source for extractor A: use the pre-trained public `Goodfire/Evo-2-Layer-26-Mixed` BatchTopK SAE (expansion 8 → 32768 features, k=64, tied) to rank which L26 features separate high-α vs low-α coding windows. Feeds the feature index into candidate #1's clamp. (Practical-tip-1 explicitly lists this SAE for Evo2-7B; do not train from scratch.)
   - path: skills/mechanism-skills/feature-dictionary-learning/SAE/SKILL.md
3. Probing / **Residual Stream States** — the *Location* source for extractor C: train a linear per-position α-helix-propensity probe on residual-stream states; take the weight direction as an alternative steering direction. Cross-layer decodability sweep also screens which block is most separable (feeds site selection).
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md

## Composition plan
Screen → decode → verify → recover, combining all three families (the plan's "may combine"):
- **Screen (M1 / Location)** — three extractors on a held-out-split contrastive set (high-α = mainly-α CDS, low-α = mainly-β CDS):
  - A. **SAE (FDL)** — encode L26 (`blocks.26`) activations through the Goodfire SAE; rank the 32768 features by AUROC / mean-activation gap between high-α and low-α windows. Cost ~0.5h.
  - B. **Contrastive vector (Steering Vectors)** — mean-difference of Evo2-7B residual activations (high-α − low-α), extracted at spaced mid-to-late blocks `{14,18,20,22,24,26,28}` (per steering-block-selection: do not hard-code one block; screen the stack). Rank each site by projected-separation AUROC. Cost ~0.5h.
  - C. **Linear probe (Probing)** — logistic probe for α-helix propensity on residual states across the same block set; take weight direction, rank by held-out AUROC. Cost ~0.5h.
- **Decode** — confirm the top candidate directions/features actually separate α from β on the disjoint held-out split (AUROC), and a quick vocab/likelihood-lens sanity that steering does not merely raise validity.
- **Verify (M2 → M3 / Causal Intervention)** — promote the best-ranked candidate(s) to a causal knob via the Rep&Param write-in: sweep steering coefficient α (dose-response) at the locked site; score every generation with the E1 harness (%H) AND general-ability covariates (ORF validity, Evo2 coding-likelihood, GC). M3 = specificity battery (matched random direction at equal norm, off-target %E/validity, naive temperature/rejection-sampling baselines).
- **Recover / Apply (M4 / Tuning & Editing)** — best M2/M3 config → generate a high-α-helix DNA library; structure-validate a subset with ESMFold+DSSP.
- **Post-processing (not a mechanism family):** AUROC ranking, PCA of direction similarity are downstream analysis of collected activations.

## Plan reconciliation
<!-- One row per method_sensitive field on the intervention milestones (M1–M4). -->
- sites: plan="several candidate layers incl. Layer 26" → **re-bound** — SAE clamp pinned to `blocks.26` (the SAE's own layer); contrastive/probe directions screened over spaced mid-to-late blocks `{14,18,20,22,24,26,28}` of the 32-block trunk (rel. depth 0.44–0.88), per steering-block-selection tip (never hard-code a single block; widen to 3–5 if inert).
- n_pairs: plan="≥~500 contrastive coding windows" → **re-bound ~800 windows (≥300/class, held-out split reserved)** — a diff-mean CAA direction and a linear probe are stable at a few hundred/class; ≥500 floor honored and exceeded subject to data availability.
- metric: plan="separation metric (M1); %H effect (M2)" → **re-bound** — M1 separation = AUROC of direction projection separating high-α/low-α on held-out; M2 effect = mean %H with **ESMFold+DSSP as the primary structure-grounded scorer** (affordable for short windows on A800; a fidelity *upgrade* over a fast-only online metric, permitted under cost-aware budget) + the ESM2-650M SS-probe as the fast cross-check. General-ability covariates: ORF-validity rate, Evo2 coding-likelihood, GC.
- gpu_hours: plan~2(E1)+3(M1)+~6(M2)+3(M3)+4(M4) ≈ 18h → **revised ~10–12h** — Evo2-7B loads in ~22s and generates short windows in seconds; generation and ESMFold scoring are decoupled into separate passes so each uses a full A800; ESMFold folds ≤400-residue proteins in ~1–3s. Cost profile is dominated by ESMFold scoring, not generation.
reconciliation_status: ok

## Rationale
#1 is recommended because the deliverable is a **causal control knob** (Claim 1: intervene during generation → raise %H specifically), and Representation and Parameter Analysis is the only family whose primitive *is* that write-in intervention — its two submethods natively cover both plan interventions (SAE-feature clamp and contrastive vector), so "may combine" is satisfied within one committed family rather than forcing a single extractor. Feature Dictionary Learning (#2) and Probing (#3) are *Location* sources that feed directions/features into #1's intervention; they are composed in M1, not competing families. This aligns with the plan's stamped `mechanism_strategy.directions: [Location, Causal Intervention, Tuning & Editing]` (aligned_with_tagging: yes). No family in `families_already_settled` (round 1, none settled). The Goodfire L26 SAE is used pre-trained (never trained from scratch) per FDL practical rule and mechanism-skills practical-tip-1.
