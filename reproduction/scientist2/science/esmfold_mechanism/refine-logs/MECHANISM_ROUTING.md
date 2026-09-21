# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Causal Attribution / Patching  (primary, covers C1+C2); composition members Probing / Residual Stream States (C3a) and Representation and Parameter Analysis / Steering Vectors (C3b)
chosen_idea_title: "ESMFold folding-trunk β-hairpin mechanism"
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md

## Candidates

1. **[recommended]** Causal Attribution / Patching — the gold-standard family for the C1 (early-block localization + s-active) and C2 (seq2pair vs pair2seq matched pathway) claims. Both claims are causal-necessity claims about *what internal object drives the DSSP-β-hairpin decision*, exactly what activation patching is built to answer. The clean/corrupted paradigm (arXiv:2404.15255 — already named in the proposal) maps directly onto the plan's "clean = no-op ESMFold forward, corrupted = donor-chain-derived tensor at target site" formulation. Patching on `s[block_k, target_res, :]` for M1 and on `seq2pair.output[block_k, target_pair, :]` for M2 are drop-in submethod instantiations.
   - path: skills/mechanism-skills/causal-attribution/patching/SKILL.md

2. Probing / Residual Stream States — C3a asks whether residue charge is *linearly decodable* from `s[block_k, res, :]` on held-out chains. This is the textbook "train linear probe across sampled layers on frozen features" pattern the submethod file demonstrates. 3-class balanced accuracy vs 1000-permutation null is the standard statistical protocol.
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md

3. Representation and Parameter Analysis / Steering Vectors — C3b promotes the probe-recovered direction `v_charge` (from C3a) into an *additive intervention*: `s[block_k, res, :] += α · v_charge`. This is exactly CAA-style steering (though here `v_charge` comes from a linear probe rather than mean-diff over contrastive text pairs — mechanically identical intervention primitive). The submethod file's `apply_steering_hook` template maps directly onto ESMFold trunk block hooks.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md

## Composition plan

Screen → Decode → Verify — the plan's ladder-of-evidence is exactly this composition:

- **Screen (correlational)** — M1 Step A: block-wise linear probe on `s` predicts target-region hairpin formation over 12 sampled blocks. Uses Probing / Residual Stream States (downstream candidate #2's primitive) as a *screen*, per the composition rule in `causal-attribution/SKILL.md` ("cheaper method typically applied first to narrow the candidate set").
- **Verify — necessity (C1)** — M1 Step B/C/D/E: donor-patch on `s[block_k, target_res, :]` inside each of 8 candidate bands ([0-3], [4-7], [8-11], [12-15], [16-23], [24-31], [32-39], [40-47]) — a *regional-claim match* per the block-selection tip. Symmetric null controls: same-window z-patching, late-window s-patching, matched-mask non-target patching. Patching submethod.
- **Verify — pathway (C2)** — M2: donor-patch on `trunk.blocks[k].sequence_to_pair.output` at target-pair positions inside the M1-localized window; matched-donor patch on `pair_to_sequence.output` as null; zero-ablation robustness sanity. Same Patching submethod, different site (sub-module output rather than block-input state).
- **Decode (C3a)** — M3a: linear probe (3-class balanced classifier, chain-level 8:1:1 split, 1000-permutation null) on `s[block_k, :, :]` for blocks {0,2,4,6}; extract `v_charge` = w_pos - w_neg from strongest block, L2-normalize. Probing / Residual Stream States submethod.
- **Verify — sufficiency (C3b)** — M3b: additive steering `s[block_k, res_i, :] += α · v_charge` on target cross-strand pair, α ∈ {-3σ, -1σ, 0, +1σ, +3σ} (rebound to σ_proj units per the tip) × 2 configs (same / opposite) × 2 pair-vs-control; matched-random-direction control upgraded to MUST-RUN. Steering Vectors submethod.

Cost profile — all patching / probing / steering here run *inference-only* on a frozen ESMFold (no backward passes, no dictionary training). The `n_pairs = 200 × donor_per_chain = 3` and `2 s/forward` estimates in the plan hold for this submethod (patching cost = 1 extra forward per intervention, matching the plan's arithmetic 4800 → 2.7 h for M1).

## Plan reconciliation

The intervention milestones (M1, M2, M3a, M3b) declared these `method_sensitive:` fields. Committed submethods above; reconciliation follows:

- **M1**  — `method_sensitive: [n_pairs, sites, metric, gpu_hours]`
  - n_pairs: plan=200 chains × 3 donors × 8 windows → matches (activation patching in the clean/corrupted paradigm needs one forward per (chain, window, donor); the plan's arithmetic already accounts for this).
  - sites: plan="`s[block_k, target_residues, :]` at each block inside candidate window" → matches (Patching submethod operates at block-input on the residual state).
  - metric: plan="DSSP-β-hairpin rate on predicted structure + paired McNemar/Wilcoxon" → matches (Patching submethod is metric-agnostic; the DSSP boolean is a valid scalar outcome).
  - gpu_hours: plan~3.0 h → matches (2.7 h forwards + DSSP + probe screen ≈ 3.0 h under the exact submethod cost model).

- **M2**  — `method_sensitive: [sites, metric, gpu_hours]`
  - sites: plan="`seq2pair.output[target_pair]` and `pair2seq.output[target_residues]` at each block in M1's early window" → matches (Patching applied to sub-module outputs, standard variant).
  - metric: plan="DSSP-β-hairpin rate + paired McNemar" → matches.
  - gpu_hours: plan~2.5 h → matches.

- **M3a**  — `method_sensitive: [metric, gpu_hours]`
  - metric: plan="3-class balanced accuracy + 1000-permutation null p-value + AUROC" → matches (Probing / Residual Stream States uses exactly this triad).
  - gpu_hours: plan~0.5 h → matches (one hidden-state extraction pass on ~300 held-out chains + fast sklearn LogisticRegression fits).

- **M3b**  — `method_sensitive: [sites, metric, gpu_hours]`
  - sites: plan="early block k* (from M3a) on `s[block_k*, res_i, :]` and `s[block_k*, res_j, :]`" → matches (Steering Vectors additive intervention at chosen residual-state site).
  - metric: plan="dose-response Spearman + same/opposite paired McNemar + matched-control specificity + Δ(hairpin_rate) ≥ 0.15 pp + auxiliary cross-strand distance" → **re-bound**: added `mean_plddt_target` and `neighbor_rmsd_vs_noop` as **structural-coherence fluency metrics** and `α=0` baseline as an explicit sweep point (5 points, was 4); random-direction control upgraded from optional to MUST-RUN. Justification: `steering-coefficient-tuning` tip requires a general-ability / collapse metric alongside the target metric to distinguish "hairpin flipped because charge steering worked" from "structure went off-distribution and DSSP misclassified". α unit rebound from raw multipliers to `β · σ_proj` (with σ_proj measured on the 50-chain calibration split at block_k*) so the sweep is physically meaningful and comparable across the M3a-selected block. See EXPERIMENT_TIPS.md `## Composition`.
  - gpu_hours: plan~1.5 h → **revised ~1.7 h** (5 α × 2 configs × 2 pair-vs-control × 200 chains = 4000 forwards vs the plan's 3200; +0.2 h). Fits inside the plan's 1 h buffer, no total-budget change.

reconciliation_status: ok

## Rationale

**Why Causal Attribution / Patching is #1**: three of the four claims (C1, C2, C3b) are *causal* claims — "intervening on X changes β-hairpin formation". Causal Attribution is the only family in the catalog whose primary output is a *necessity/sufficiency* verdict for an internal object. The other candidate families are needed *within* the plan's ladder but each covers only one of four claim components; Patching covers two directly (C1, C2) and hosts the intervention primitive that Steering Vectors specializes for C3b.

**Why the Probing and Steering Vectors families are also committed** (not "recommended-but-dropped"): C3a is not a causal claim — it is a *decodability* claim, and the family loading protocol requires a matching family. Steering Vectors is functionally a special case of activation-additive intervention, but the catalog places it under Representation and Parameter Analysis; we cite that family for C3b to honor the catalog structure. All three families are committed jointly — this is a composition, not a single-family pick.

**On `families_already_settled: []`** — round-1 run for this behavior; no families excluded.

**Cross-round routing hint from the plan**: `mechanism_strategy.directions = [Location, Causal Intervention, Unit Interpretation]`. All three directions map onto the composition: Location (M1 Step A — probing screen for early window), Causal Intervention (M1 Step B–E + M2 + M3b — patching + steering), Unit Interpretation (M3a — linear probe recovers a *named* direction `v_charge` = signed charge dimension). Rejected directions (Tuning & Editing / Formation Tracing / Decision Auditing) do not appear in the composition, consistent with the plan.
