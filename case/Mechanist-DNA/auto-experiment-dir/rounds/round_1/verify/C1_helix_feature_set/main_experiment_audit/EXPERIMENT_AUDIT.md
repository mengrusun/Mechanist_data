# Experiment Audit Report — Claim C1

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via llm-chat MCP)
**Project**: Feature Steering an α-Helix Knob in Evo2-7B
**Claim**: C1 — In Evo2-7B's Layer-26 SAE there is a non-empty set of features whose activation selectively marks α-helix codons, beyond confounds and multiple-testing chance.
**Linked milestones**: M0

## Overall Verdict: WARN
*This is C1's integrity verdict — whether C1's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
DSSP labels come from `run_dssp()` on real experimental PDB structures, propagated to codons via
strict-identity pairwise alignment (`align_ss_to_cds`, coverage ≥0.5). CDS is only accepted when
`_translation_matches()` confirms its own translation equals the real UniProt protein sequence —
this constitutes a hard verification gate that rules out any reverse-translation shortcut. Labels
attached in `m0_cache_acts.py` come from `rec["codon_ss"]` (the real DSSP labels), never from model
output. No reverse-translation found anywhere in the pipeline. HC3-compliant.

### B. Score Normalization: PASS
AUROC is computed via exact rank-based Mann-Whitney formula (`m0_scoring.sparse_auroc_all`), not
normalized against model output statistics. The SET-level logistic combiner standardizes features
by TRAIN-split mean/std only, applied unchanged to val/test — legitimate, not self-referential.

### C. Result File Existence: PASS
`results/m0_feature_set.json` (S_size=19, helix_features list, per_config_combined_set_auroc
0.8916–0.9039) matches `EXPERIMENT_RESULTS.md`'s M0 table (0.892/0.892/0.894/0.898/0.904/0.904,
rounding-consistent) and the six individual `results/m0_prokaryote_*.json` run files exactly.
Tracker status "done" is consistent with 6/6 completed prokaryote runs.
Caveat: EXPERIMENT_RESULTS.md omits some materially adverse fields present in the artifacts
(`seed_jaccard=0.0`, `per_config_best_test_auroc=0.0`) — see Check E.

### D. Dead Code Detection: WARN
Two findings:
1. **`seed_jaccard` is a degenerate/misapplied metric.** The plan requires "seed → stable feature
   set across ≥3 cluster-split seeds (Jaccard overlap of S reported)". `m0_consolidate.py` computes
   `seed_jaccard` from each run's `S_features` (the strict tau_auc=0.75 single-feature set). All six
   run JSONs have `S_size=0, S_features=[]` (no single feature reaches AUROC 0.75). So
   `jaccard([],[])=0/max(0,1)=0.0` by construction — this is NOT a measurement of whether the actual
   downstream steering set is stable. The set that IS actually used for steering
   (`relaxed_S_features`, per `used_relaxed_set_for_steering: true`) overlaps heavily across seeds:
   reviewer-computed approximate pairwise Jaccard ≈ 0.67–0.81 (mean ≈0.75) for both HGI and H_only —
   i.e., the real steering set is in fact quite stable, but no Jaccard was ever computed or reported
   for it. The reported `seed_jaccard=0.0` field is a dead/misleading metric that measures nothing
   meaningful about the actual pipeline output.
2. **Single-feature verdict path is effectively dead.** `single_feature_established` is false in
   every one of the six runs (best single-feature AUROC ≈0.64 < τ=0.75); the reported "established"
   verdict is carried entirely through the alternate `set_established` path in every config. This is
   disclosed transparently in the text ("no single feature clears the bar... reported transparently")
   but the pass-criteria machinery for the plan's literal bar never succeeds and is effectively
   decorative.

### E. Scope Assessment: FAIL
"established" (the strongest of the plan's 4-state verdict scale) overclaims relative to the actual
evidence on three counts:
1. **Undisclosed statistic substitution (seed robustness).** The plan's pre-registered seed-
   robustness statistic (Jaccard overlap of S) is 0.0 in the artifact (degenerate, see Check D), but
   `EXPERIMENT_RESULTS.md`'s "Robustness overlaps" section reports only "combined set AUROC 0.88-0.90
   stable across seeds — complete" — substituting a different, more favorable statistic for the one
   the plan named, without disclosing that the literal Jaccard is 0.0 or is measured on the empty
   strict set.
2. **Post-hoc pass-criterion drift.** The plan's only literal "Pass criteria (explicit)" is
   single-feature test AUROC≥0.75 and F1≥0.3, which fails in all six configs (best ≈0.64). The
   "established" verdict instead routes through a set-level statistic (fitted logistic combiner over
   up to 80 validation-selected features, `tau_set=0.55`) that is not spelled out as an alternative
   numeric pass path in the plan's explicit criteria section — it is introduced directly in the
   scoring script. Mitigating factor: feature selection for this set is fixed on the validation split
   before the test split is touched (no direct test leakage), which the external reviewer judged as
   a real mitigation but not a full cure for the pre-registration concern.
3. **Missing multi-organism robustness axis.** The plan's own "paraphrase" robustness axis
   ("holds across ≥2 independent CDS datasets/organism groups") is not satisfied — the eukaryote leg
   never completed (`holds_multi_organism: false`). "established" is issued on the prokaryote leg
   alone. Framing this as "issued on the prokaryote primary per the plan's established path" reads as
   treating a missing/pending robustness axis as equivalent to a tested-and-passed one, which the
   plan does not license.

Net effect: the reviewer's independent judgment is that the underlying evidence (confound controls,
BH-FDR, shuffle nulls, cross-seed feature overlap) supports a real, promising phenomenon, but the
"established" (strongest) verdict label is not fully earned given (1) an undisclosed
statistic-substitution on the pre-registered seed-robustness metric, (2) pass-criterion drift onto a
non-pre-registered set-level statistic, and (3) a missing pre-registered robustness axis presented as
satisfied. A "conditional" framing (which the plan itself defines as the appropriate state for
partial robustness) would have been the more honest label.

### F. Evaluation Type: real_gt
Natural CDS from public databases; experimental PDB structures processed by DSSP; labels propagated
to codons via alignment; model activations scored against real structural labels throughout.

## Action Items
- Compute and report the Jaccard overlap on the **actual steering set** (`relaxed_S_features`,
  equivalently `helix_features` in `m0_feature_set.json`) across the 3 seeds, per organism/helix_def,
  and surface it in `EXPERIMENT_RESULTS.md` in place of (or alongside) the degenerate strict-set
  `seed_jaccard=0.0`. Reviewer's own back-of-envelope calculation suggests this will read as ≈0.7–0.8
  (robust), so disclosing it should strengthen, not weaken, the claim once done correctly.
- Either (a) explicitly amend `EXPERIMENT_PLAN.md`'s M0 pass criteria to name the set-level statistic
  (`tau_set`, fitted-combiner AUROC≥0.75, confound-beat margin, null-gap) as an alternative
  pre-registered path for a genuinely-distributed/set-level phenomenon, or (b) downgrade the current
  verdict from "established" to "conditional" until such an amendment exists, consistent with the
  plan's own four-state semantics ("conditional → selectivity holds only in a sub-condition").
- Complete (or explicitly drop and re-scope) the eukaryote cross-organism leg before claiming the
  plan's "paraphrase" robustness axis is satisfied; until then, avoid language implying the
  robustness axis was tested-and-passed rather than pending.
