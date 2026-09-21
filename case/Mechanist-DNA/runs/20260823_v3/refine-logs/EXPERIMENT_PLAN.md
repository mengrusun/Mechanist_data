# Experiment Plan — Steering Evo2-7B toward high α-helical content

```yaml
# top-metadata machine markers (stamped by claim stage, Phase 4.5)
behavior_source: given
mechanism: discovery
resource_fidelity: unstamped          # cost-aware (NOT the given+given reproduction combo)
chosen_mechanism: discovery-routed    # concrete family chosen downstream by /mechanism-skills
mechanism_strategy:
  directions: [Location, Causal Intervention]   # execution order
  rejected:
    - Formation Tracing — genesis out of scope; would break the 40 GPU-h budget.
    - Decision Auditing — task changes generation, not audits a fixed decision.
    - Unit Interpretation (standalone) — Evo2 SAE features used only as a Location source for steering.
  note: Locate the α-helix representation in Evo2-7B, then steer along it and prove the gain is real, dosed, specific, and on valid ORFs.
m0_gate: none   # behavior-source: given → NO phenomenon-validation gate; NO milestone declares depends_on:[M0]
budget:
  max_total_gpu_hours: 40     # HARD
  max_concurrent_cards: 8     # HARD
  planned_total_gpu_hours: ~33 (headroom retained)
model_under_study: Evo2-7B    # HARD — no substitution in any main run
hf_token_use: fetch Evo2-7B + datasets only (from NOTICE); never sent elsewhere
```

## Claims under test

- **C1** — intervention raises α-helix fraction of the encoded protein significantly above the
  unintervened baseline (up). Covered by M1, M2, M3.
- **C2** — the intervention preserves sequence validity (well-formed / biologically plausible ORFs);
  off-target properties not degraded (non-inferior). Covered by M1, M4.
- C1 ∧ C2 is the **joint success criterion**.

## Shared measurement contract (frozen in M1, applied identically everywhere)

Pre-registered in M1 before any intervention run; not changed post hoc. (Tightened per external review
— see `refine-logs/RESEARCH_REVIEW.md`; these fixes address survivorship bias, endpoint drift,
circularity, and multiplicity.)

- **Generation**: Evo2-7B, fixed prompt/primer set + fixed sampling settings; baseline = identical
  settings, no intervention.
- **Held-out split (anti-circularity)**: partition prompts × seeds into a **DEV** split (used for M2
  localization and M3 steering-vector / feature construction) and a **disjoint TEST** split (used for
  the final C1/C2 inferential comparison). No sequence used to build the intervention is used to test
  it.
- **ONE frozen primary SS assay**: translate ORF → structure via **ESMFold** → DSSP-style assignment →
  α-helix fraction (residues H/G/I). A single predictor is fixed for **all** conditions; a fallback SS
  predictor is a **whole-experiment** pre-declared choice made in M1 if ESMFold is over budget — never
  a per-condition or post-hoc switch. Record ESMFold **pLDDT**; apply one fixed confidence-filtering /
  weighting policy identically to all conditions.
- **ORF rule (deterministic)**: pre-registered rule (e.g. longest ATG→stop in-frame ORF ≥ L_min aa),
  applied identically to baseline and intervened.
- **Primary endpoint (survivorship-bias-safe)**: mean α-helix fraction over **all generated
  sequences** on the TEST split, with invalid/no-ORF sequences handled by a fixed pre-registered rule
  (helix = 0, or a hurdle/composite endpoint that folds validity in). **Secondary** endpoint: helix
  fraction on valid ORFs only. This couples C1 to C2 so steering cannot inflate helix by shedding
  invalid sequences.
- **Validity (C2 — fixed composite, no gaming)**: a pre-specified hierarchy/composite — in-frame
  start/stop, no premature stop, plausible length, coding plausibility (Evo2 likelihood / codon-usage
  sanity) — combined by a fixed rule into a per-sequence valid/invalid label. Report **validity rate
  over the full generated set** (not just survivors).
- **Off-target / interpretability confounds**: β-sheet fraction, GC content, mean per-token Evo2 NLL
  (naturalness), **protein length** (matched or regressed out — helix fraction is length-dependent),
  and **amino-acid composition** (measure enrichment of helix-favoring residues A/E/L/M/Q/K to
  interpret whether the effect is genuine structural control vs mere codon/composition bias — reported,
  not disqualifying).
- **Statistics (pre-registered)**: TEST split, ≥ 3 generation seeds. **Primary C1 test = one**
  pre-chosen site family with a **dose-response trend test** (isotonic / permutation regression on the
  coefficient), not per-dose Mann-Whitney; report effect size (Cliff's δ / Hedges g) + bootstrap CIs.
  **Multiplicity**: FDR/FWER control across sites, coefficients, and off-target tests. **C2 =
  non-inferiority test** on validity rate over the full generated set with a pre-registered margin + CI.

---

### M1: Baseline generation + measurement/validity harness  `[C1, C2]`
**Goal**: build and freeze the generation → ORF → SS/α-helix + validity + off-target pipeline; produce
the unintervened baseline distribution and the high-vs-low-helix contrast set used to source the
intervention.
**Runs**:
- `m1_baseline_gen` — sample ≥ 500 valid-ORF sequences from Evo2-7B (no intervention), score all
  metrics.
- `m1_contrast_set` — assemble high-helix vs low-helix reference sequences (from labeled-SS proteins
  back-translated to DNA and/or filtered Evo2 generations) for steering-vector / feature construction.
**Cmd (illustrative)**: `python m1_baseline.py --model evo2_7b --n 600 --seeds 0,1,2 --orf-rule longest_atg --out results/m1/`
**Expected output**: `results/m1/baseline_metrics.parquet`, `results/m1/contrast_set.parquet`, frozen `harness_config.json`
**Priority**: MUST-RUN · **depends_on**: [] · **Est. GPU-hours**: ~4h

### M2: Localize the α-helix representation (Location — correlational screen)  `[C1]`
**Goal**: rank candidate internal sites carrying α-helix propensity, to source the intervention.
**Runs**:
- `m2_probe` — extract Evo2-7B activations (per layer, incl. StripedHyena + Transformer blocks) over
  M1 sequences; train linear probes predicting the encoded protein's α-helix fraction; rank layers.
- `m2_sae_screen` — screen pre-existing Evo2 SAE features (layer ~26 emphasized) for α-helix
  selectivity; shortlist candidate features.
**Cmd (illustrative)**: `python m2_localize.py --model evo2_7b --acts-from results/m1 --layers all --sae layer26 --out results/m2/`
**Expected output**: `results/m2/probe_ranking.json` (top layers/directions), `results/m2/sae_candidates.json`
**Priority**: MUST-RUN · **depends_on**: [M1]
**method_sensitive**: [sites, metric, n_pairs]
**Est. GPU-hours**: ~5h

### M3: Steer + dose-response (Causal Intervention — the applied lever)  `[C1]`
**Goal**: apply the intervention built from M2 survivors during generation and confirm α-helix rises
vs baseline with a monotone dose-response.
**Intervention**: contrastive activation addition (mean high-helix − mean low-helix activation at the
chosen site) and/or SAE-feature clamp; added at the M2-selected site(s) during decoding.
**Grid**:
```
grid:
  coefficient: [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]   # 0.0 reproduces baseline as an in-run control
  site: [top1_layer, top2_layer]                # from M2 ranking
```
**Cmd template**: `python m3_steer.py --model evo2_7b --site ${site} --coef ${coefficient} --n 250 --seeds 0,1,2 --orf-rule longest_atg --out results/m3/s${site}_c${coefficient}.parquet`
**Expected output (template)**: `results/m3/s${site}_c${coefficient}.parquet` (α-helix, validity, off-target per generation)
**Vector/feature construction on DEV split only**; evaluation on TEST split (anti-circularity).
**Success (C1)**: for the pre-chosen site family, a significant positive **dose-response trend**
(isotonic / permutation regression on coefficient) in the primary all-generations helix endpoint on
the TEST split, positive effect size, FDR-corrected across sites/coefs.
**Priority**: MUST-RUN · **depends_on**: [M1, M2]
**method_sensitive**: [sites, n_pairs, metric, gpu_hours]
**Est. GPU-hours**: ~18h  (dominant: ESMFold scoring of ~3k generations, batched across ≤ 8 cards)

### M4: Specificity + validity controls (C2 + causal specificity)  `[C1, C2]`
**Goal**: prove the helix gain is specific and — critically — achieved on **valid** sequences.
**Runs** (all on the TEST split):
- `m4_matched_control` — repeat M3 at the winning site/coef with a **matched-control direction**
  (random unit vector / orthogonalized-to-helix direction of equal norm); expect **no** helix gain.
- `m4_sham` — **same-site, same-norm sham perturbation** (permuted/shuffled direction of identical
  norm at the same token span) at the winning coef; controls for the act of perturbing activations
  itself. Expect no helix gain.
- `m4_validity` — at the winning setting, C2 **non-inferiority** test on validity rate over the **full
  generated set** vs baseline (pre-registered margin + CI); off-target properties (β-sheet, GC, Evo2
  NLL, length, aa-composition) vs baseline.
- `m4_frontier` — **validity frontier**: helix gain vs validity rate across the coefficient sweep; the
  reported success setting is the max helix gain subject to validity non-inferiority.
**Cmd (illustrative)**: `python m4_controls.py --model evo2_7b --winning results/m3/best.json --controls matched_random,sham_permuted --out results/m4/`
**Expected output**: `results/m4/specificity.json`, `results/m4/validity_frontier.parquet`
**Success (C2)**: matched-control **and** sham show no significant helix gain; at the C1-winning
setting validity rate is non-inferior to baseline (within margin) with no off-target degradation.
**Priority**: MUST-RUN · **depends_on**: [M3]
**method_sensitive**: [sites, metric, gpu_hours]
**Est. GPU-hours**: ~7.5h

---

## Budget ledger

| Milestone | Est. GPU-hours | Cards |
|---|---|---|
| M1 | ~4h | ≤ 8 |
| M2 | ~5h | ≤ 8 |
| M3 | ~18h | ≤ 8 |
| M4 | ~7.5h | ≤ 8 |
| **Total** | **~34.5h** | ≤ 8 concurrent |

Under the 40 GPU-hour HARD cap with ~5.5h headroom. If ESMFold scoring at the planned counts would push
M3 over budget, the experiment stage must **stop and surface it** (reduce `n` per condition or swap to
a cheaper SS predictor as declared in the M1 harness) rather than silently downscale or drop M3/M4.

## Run order

M1 → M2 → M3 → M4 (strict `depends_on` chain; M3 grid dispatches to the queue). No M0 gate
(behavior-source: given) — no milestone waits on an M0 phenomenon-validation gate (there is none).

## Notes for downstream stages

- `/mechanism-skills` binds the concrete family (CAA vs SAE clamp vs light tuned variant) and may
  re-bind the `method_sensitive` fields (`n_pairs`, `sites`, `metric`, `gpu_hours`) at routing time
  without this counting as a plan rewrite.
- HARD constraints (Evo2-7B only, ≤ 40 GPU-h / ≤ 8 cards, validity-preserving) are binding on every
  milestone here and are inherited by the verify stage via C1 ∧ C2.
