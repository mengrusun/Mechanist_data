# Experiment Tracker — Belief Circuits Reproduction

**Date created**: 2026-07-22
**Ownership**: Phase 4.5 planning-level tracker. Updated in place by `/auto-experiment` Phase 5 as runs progress.

## M1 — Behavioural Scaling (Claim 1) — all done

| id | Model | Task | Status | Result |
|---|---|---|---|---|
| m1_pythia-410m_world_knowledge | pythia-410m | world_knowledge | done | acc=0.881, CI=[0.832, 0.917] |
| m1_pythia-410m_personal_belief | pythia-410m | personal_belief | done | acc=0.852, CI=[0.823, 0.876] |
| m1_pythia-410m_attributed_belief | pythia-410m | attributed_belief | done | acc=0.457, CI=[0.420, 0.494] (below chance — M2 gate FAILS this cell) |
| m1_pythia-1b_world_knowledge | pythia-1b | world_knowledge | done | acc=0.925, CI=[0.883, 0.953] |
| m1_pythia-1b_personal_belief | pythia-1b | personal_belief | done | acc=0.786, CI=[0.753, 0.815] |
| m1_pythia-1b_attributed_belief | pythia-1b | attributed_belief | done | acc=0.833, CI=[0.803, 0.859] |
| m1_pythia-2.8b_world_knowledge | pythia-2.8b | world_knowledge | done | acc=0.960, CI=[0.926, 0.979] |
| m1_pythia-2.8b_personal_belief | pythia-2.8b | personal_belief | done | acc=0.994, CI=[0.985, 0.998] |
| m1_pythia-2.8b_attributed_belief | pythia-2.8b | attributed_belief | done | acc=0.796, CI=[0.764, 0.824] |

Above-chance gate summary → `refine-logs/artifacts/behavioral/above_chance_gate.json`: 5 pairs cleared (see M2.3 rows).

## M2.1 — Fisher Signals — all done

| id | Model | Signal | Status | Result |
|---|---|---|---|---|
| m2_1_pythia-410m_F_attributed | pythia-410m | F_attributed | done | 454 ex, per-head + jackknife halves written |
| m2_1_pythia-410m_F_personal | pythia-410m | F_personal | done | 454 ex, per-head + jackknife halves written |
| m2_1_pythia-410m_F_knowledge | pythia-410m | F_knowledge | done | 227 ex, per-head + jackknife halves written |
| m2_1_pythia-1b_F_attributed | pythia-1b | F_attributed | done | 454 ex, per-head + jackknife halves written |
| m2_1_pythia-1b_F_personal | pythia-1b | F_personal | done | 454 ex, per-head + jackknife halves written |
| m2_1_pythia-1b_F_knowledge | pythia-1b | F_knowledge | done | 227 ex, per-head + jackknife halves written |
| m2_1_pythia-2.8b_F_attributed | pythia-2.8b | F_attributed | done | 454 ex, per-head + jackknife halves written |
| m2_1_pythia-2.8b_F_personal | pythia-2.8b | F_personal | done | 454 ex, per-head + jackknife halves written |
| m2_1_pythia-2.8b_F_knowledge | pythia-2.8b | F_knowledge | done | 227 ex, per-head + jackknife halves written |

## M2.2 — Mask Construction + Jackknife stability — all done

| id | Model | Target | Status | Jackknife ρ | Stable |
|---|---|---|---|---|---|
| m2_2_pythia-410m_personal | pythia-410m | personal | done | 0.885 | yes |
| m2_2_pythia-410m_attributed | pythia-410m | attributed | done | 0.940 | yes |
| m2_2_pythia-1b_personal | pythia-1b | personal | done | 0.979 | yes |
| m2_2_pythia-1b_attributed | pythia-1b | attributed | done | 0.992 | yes |
| m2_2_pythia-2.8b_personal | pythia-2.8b | personal | done | 0.926 | yes |
| m2_2_pythia-2.8b_attributed | pythia-2.8b | attributed | done | 0.966 | yes |

## M2.3 — Smallest H* Search — 5 of 6 admissible pairs run

| id | Model | Target | Status | \|H*\| | Heads |
|---|---|---|---|---|---|
| m2_3_pythia-410m_personal | pythia-410m | personal | done — LOCALIZED | 2 | (13,1), (8,8) |
| m2_3_pythia-410m_attributed | pythia-410m | attributed | skipped — M1 gate failed (acc=0.457, below chance) | — | — |
| m2_3_pythia-1b_personal | pythia-1b | personal | done — LOCALIZED | 2 | (12,1), (9,1) |
| m2_3_pythia-1b_attributed | pythia-1b | attributed | done — LOCALIZED | 1 | (4,1) |
| m2_3_pythia-2.8b_personal | pythia-2.8b | personal | done — LOCALIZED | 4 | (14,16), (15,3), (13,1), (12,4) |
| m2_3_pythia-2.8b_attributed | pythia-2.8b | attributed | done — LOCALIZED | 1 | (5,22) |

## M2.4 — Controls (40 per H*) + Acceptance — all done

Each localized (model, target) pair received 20 random-head (seeds 100..119) + 20 random-mask (seeds 200..219) = 40 controls. Per-pair rows collapsed into acceptance summary:

| id | Model | Target | Status | Verdict | Δ target | mean_rh + 2σ | C2a | C2b | C2c | C2d |
|---|---|---|---|---|---|---|---|---|---|---|
| m2_4_acceptance_pythia-410m_personal | pythia-410m | personal | done | LOCALIZED | 0.372 | 0.081 | ✓ | ✓ | ✓ | ✓ |
| m2_4_acceptance_pythia-1b_personal | pythia-1b | personal | done | LOCALIZED | 0.471 | 0.151 | ✓ | ✓ | ✓ | ✓ |
| m2_4_acceptance_pythia-1b_attributed | pythia-1b | attributed | done | LOCALIZED | 0.463 | 0.257 | ✓ | ✓ | ✓ | ✓ |
| m2_4_acceptance_pythia-2.8b_personal | pythia-2.8b | personal | done | LOCALIZED | 0.307 | 0.035 | ✓ | ✓ | ✓ | ✓ |
| m2_4_acceptance_pythia-2.8b_attributed | pythia-2.8b | attributed | done | LOCALIZED | 0.355 | 0.033 | ✓ | ✓ | ✓ | ✓ |

## M3 — Formation Window (Claim 3) — done on 24 of 154 planned checkpoints

**Resource gap (surfaced):** only 24 pythia-1b intermediate checkpoints are present on disk under `/mnt/quarkfs/share_model/Ptyhia/pythia-1b-checkpoints/` — the native log-spaced subset (step 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 13000, 23000, 33000, 43000, 63000, 83000, 103000, 123000, 143000). Missing files cannot be fabricated; the strict harness's HALT-on-downscale rule does not apply because this is an environmental constraint, not a cost-saving action. M3 ran the 24 available checkpoints.

| id | Checkpoint | Status | Behavioural (WK/PB/AB) | Ablation reported |
|---|---|---|---|---|
| m3_step0..step143000 | 24 checkpoints | done (24/24) | see `refine-logs/artifacts/formation/step*.json` | both `H*_personal_ablated` and `H*_attributed_ablated` measured (both H*_ available for pythia-1b) |
| m3_aggregate | (aggregate) | done | personal window [13000, 13000]; attributed window [0, 33000]; distinct = TRUE | see `refine-logs/artifacts/formation/summary.{json,md}` |

## M4 — Probe-and-Amplify Controller (Claim 4) — done on both localized models

pythia-410m has only H*_personal (no H*_attributed available, M1 gate failed on attributed), so M4 excludes pythia-410m per the plan. pythia-1b and pythia-2.8b both have BOTH H* localized.

| id | Model | Sub-milestone | Status | Result |
|---|---|---|---|---|
| m4_1_pythia-1b | pythia-1b | Frame classifier training | done | val_acc = 1.0000; L_ctrl=4; probing_layers=[1,2,3] |
| m4_1_pythia-2.8b | pythia-2.8b | Frame classifier training | done | val_acc = 1.0000; L_ctrl=5; probing_layers=[2,3,4] |
| m4_2_pythia-1b_grid | pythia-1b | 36-cell α grid | done | 36 configs on val; see `refine-logs/artifacts/controller/pythia-1b/alpha_grid/*.json` |
| m4_2_select_pythia-1b | pythia-1b | α selection | done | α_p*=3.0, α_a*=1.5; val net_impr=30; Δ_wk=0 (strict guardrail) |
| m4_2_pythia-2.8b_grid | pythia-2.8b | 36-cell α grid | done | 36 configs on val; see `refine-logs/artifacts/controller/pythia-2.8b/alpha_grid/*.json` |
| m4_2_select_pythia-2.8b | pythia-2.8b | α selection | done | α_p*=1.5, α_a*=4.0; val net_impr=17; Δ_wk=0 (strict guardrail) |
| m4_3_pythia-1b_baseline_no_control | pythia-1b | OOD baseline | done | WK=0.850, PB=0.672, AB=0.788 (n=367/1101/1101) |
| m4_3_pythia-1b_controller | pythia-1b | OOD controller | done | WK=0.850, PB=0.850, AB=0.909; frame_acc_OOD=0.9977; PPL 8.756→9.172 (1.047×) |
| m4_3_pythia-1b_prompt_hint | pythia-1b | OOD prompt-hint baseline | done | WK=0.850, PB=0.711, AB=0.708 |
| m4_3_report_pythia-1b | pythia-1b | OOD report | done | Controller net_impr=**151** (recovered 165 / degraded 14); prompt-hint net_impr=**-1** |
| m4_3_pythia-2.8b_baseline_no_control | pythia-2.8b | OOD baseline | done | WK=0.886, PB=0.948, AB=0.825 |
| m4_3_pythia-2.8b_controller | pythia-2.8b | OOD controller | done | WK=0.886, PB=0.973, AB=0.880; frame_acc_OOD=0.9957; PPL 7.330→7.364 (1.005×) |
| m4_3_pythia-2.8b_prompt_hint | pythia-2.8b | OOD prompt-hint baseline | done | WK=0.886, PB=0.955, AB=0.650 |
| m4_3_report_pythia-2.8b | pythia-2.8b | OOD report | done | Controller net_impr=**27** (recovered 31 / degraded 4); prompt-hint net_impr=**-87** (prompt hint HURTS pythia-2.8b) |

## Cross-milestone artifacts

| id | Description | Status | Notes |
|---|---|---|---|
| setup_ppl_sample | Cache 1M-token PPL sample from Pile shard 00000 (uint16) | done | `refine-logs/artifacts/ppl_sample.pt`, reused across M2.3/M2.4/M3/M4.3 |

## Report deliverables

| id | Claim | Status | Output |
|---|---|---|---|
| report_c1 | Claim 1 (Scaling) | done | Rolled up into `refine-logs/EXPERIMENT_RESULTS.md` |
| report_c2 | Claim 2 (Localization) | done | Rolled up into `refine-logs/EXPERIMENT_RESULTS.md` |
| report_c3 | Claim 3 (Formation) | done | Rolled up + `refine-logs/artifacts/formation/summary.md` |
| report_c4 | Claim 4 (Controller) | done | Rolled up + `refine-logs/artifacts/controller/{pythia-1b,pythia-2.8b}/M4_report.md` |
| report_overall | All claims | done | `refine-logs/EXPERIMENT_RESULTS.md` |
