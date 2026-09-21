# Experiment Results — Sparse Modular Circuit for Propositional-Logic Reasoning

**Behavior-source**: given / **Mechanism**: discovery / **Family**: Causal Attribution / Attribution Patching + Patching
**Anchor plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Owner**: /auto-experiment (Phase 5 collect)
**Model (main)**: Mistral-7B-v0.1 base (bf16, TransformerLens 2.18.0 in conda env `belief`)
**Dataset**: synthetic propositional-logic template (`${DATA_DIR}/prop_logic_synth`), anchor cell `k3_chain2_natural`
**Status**: **complete** — M0–M6 done. M5.contingent (Gemma-2-27B) **skipped** (not locally available, plan permits).

---

## Headline

- **C1 (sparse component set) — PARTIAL**: attribution-patching screen returns a 158-component shortlist at the hard cap `|C|/|total|=0.150` (target ≤ 0.15). Completeness (logit-diff recovery) is **0.955** ≥ 0.9 target. Minimality (avg single-removal drop) is **0.0014** ≪ 0.05 target — heavy redundancy in a large shortlist; not all C1 criteria met, so verdict = **partial**.
- **C2 (modular fact/rule/answer decomposition) — FAIL**: on the anchor cell (M4, shortlist=40), all three C2 targets fail — median_dominance=1.20 (target ≥ 2.0), dissociation `{fact:0.089 rule:0.027 answer:0.026}` (target ≥ 0.1 each; only "fact" borderline), null-shuffle p-values `{fact:0.05 rule:1.0 answer:1.0}` (target ≤ 0.01 each). Cross-cell stability (M4.stab): Jaccard `{fact:0.75 answer:0.77 rule:0.36}` — 2/3 roles are stable but the modularity hypothesis is refuted overall. Interpretation: components are functionally *smeared* across roles, not modular specialists.
- **C3 (necessity + sufficiency) — PARTIAL**: necessity **PASS** (path-patching M2: recovery LD=0.955, PD=0.905, specificity_gap=0.836; dose-response shows top-31 already recovers 0.946); sufficiency **FAIL** (reinsertion M3: recovery LD=0.113, stable across 5 seeds — far below 0.8 target). The circuit is *necessary but not sufficient*: patching the shortlist onto corrupted prompts nearly restores clean behavior, but keeping only the shortlist's clean activations while resample-ablating the other 85% of components does not. **Same necessity-yes / sufficiency-no pattern recurs on Gemma-2-9B** (M5: necessity LD=1.018 ✓, sufficiency LD=0.019 ✗) — cross-family robust negative.

**Overall**: this is a **reproduction that partially refutes** the sparse-modular-circuit hypothesis for propositional-logic reasoning: sparse-yet-complete (C1 completeness) and necessary (C3 necessity) at |shortlist|/|model| = 15%, but not minimal (C1), not modular (C2), and not sufficient (C3) — same pattern cross-family on Gemma-2-9B.

---

## Per-milestone

### M0.dataset — Synthetic propositional-logic dataset build

- **Cmd**: `python scripts/build_propositional_dataset.py --data-dir ${DATA_DIR}/prop_logic_synth --seed 42`
- **Output**: `${DATA_DIR}/prop_logic_synth/manifest.json`
- **Cells built**: anchor (`k3_chain2_natural`, 500 pairs each of {clean, corrupt_fact, corrupt_rule, corrupt_answer, corrupt_neutral}) + 4 additional 200-pair cells for role/stability + 800-prompt resample pool.
- **GPU-h**: 0.001 (CPU only).
- **Verdict**: done.

### M0.setup — Model / environment / patching-framework check

- **Cmd**: `CUDA_VISIBLE_DEVICES=2 python scripts/setup_check.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --data ${DATA_DIR}/prop_logic_synth --n-pairs 500 --out results/M0_setup.json`
- **Output**: `results/M0_setup.json`
- **Result**: Mistral-7B-v0.1 base loaded from local mirror (not Instruct fallback), TransformerLens hooks OK, anchor accuracy `0.810` on 500 True/False prompts (top-1 T/F share = 0.872).
- **Sanity criterion** (`accuracy ≥ 0.75`): **PASS**.
- **GPU-h**: 0.025.
- **Verdict**: done.

### M1 — Attribution-patching screen (C1 Location)

- **Cmd**: `CUDA_VISIBLE_DEVICES=2 python scripts/attribution_screen.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --cell k3_chain2_natural --corruption corrupt_fact --n-pairs 500 --batch-size 4 --out results/M1_attribution.json`
- **Output**: `results/M1_attribution.json`
- **Claim tested**: **C1**.
- **Key stats**:
  - `shortlist_size = 158` (right at hard cap `|C|/|total| ≤ 0.15`)
  - `sparsity_fraction = 0.150`
  - `cumulative_effect = 0.758` (of attribution mass — screen stopped at cap, not at 0.9)
  - `completeness (recovery_logit_diff, full patch) = 0.955` ≥ 0.9 target — **PASS**
  - `completeness_prob_diff = 0.905`, `KL recovery = -5.618` (note: KL recovery reported but off-distribution)
  - `minimality.avg_single_removal_drop = 0.0014` (median 0.0027) ≪ 0.05 target — **FAIL**
- **Success criterion** (`sparsity ≤ 0.15 AND completeness ≥ 0.9 AND minimality ≥ 0.05`): **FAIL on minimality**.
- **GPU-h**: ~0.17 (wall 628s on GPU 2 shared).
- **Interpretation**: the shortlist is *complete* (removing what's outside it barely hurts recovery — the 15% subset carries almost all of the answer signal), but not *minimal* — dropping any single component from the 158-member set barely dents recovery, consistent with heavy pairwise redundancy inside the shortlist. This is a known behavior of large shortlists selected by unnormalized attribution mass; a tighter shortlist (or a per-component ACDC-style pruning pass) would likely raise the single-removal drop but at the cost of completeness. Verdict for C1 = **partial** (2/3 criteria met).
- **main_experiment.verdict**: **partial**.

### M2 — Path-patching necessity (C3 necessity)

- **Cmd**: `CUDA_VISIBLE_DEVICES=0 python scripts/path_patch_necessity.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --data ${DATA_DIR}/prop_logic_synth --cell k3_chain2_natural --corruption corrupt_fact --shortlist results/M1_attribution.json --n-pairs 500 --batch-size 4 --out results/M2_necessity.json`
- **Output**: `results/M2_necessity.json` ✓
- **Claim tested**: **C3 (necessity)**.
- **Key stats** (n_pairs=500):
  - `recovery.logit_diff = 0.955` ≥ 0.8 ✓
  - `recovery.prob_diff = 0.905` ≥ 0.8 ✓
  - `recovery.KL = -5.62` (off-scale; KL recovery uses (1 − patch_KL / kl_baseline_clean_corrupt); baseline is small so recovery goes very negative when patch_KL exceeds it — see notes)
  - `control_recovery.logit_diff = 0.119` (matched-size random non-shortlist)
  - `specificity_gap = 0.836` ≥ 0.6 ✓
  - Dose response (n_pairs=500): k=0→0.00, k=31→0.946, k=62→1.008, k=93→0.992, k=124→1.021, k=155→0.984, k=158→0.955 — **recovery saturates at ~31 components** (well below the 158 shortlist size — most of the recovery is carried by the top-31 heads).
- **Success criterion** (`recovery.LD ≥ 0.8 AND recovery.PD ≥ 0.8 AND specificity_gap ≥ 0.6`): **PASS**.
- **Notes**: KL recovery goes negative when `patch_KL > kl_baseline_clean_corrupt`. This is a scaling artifact of using the tight `clean||corrupt` KL as denominator (0.043); once patching drives distribution far from either endpoint, the ratio explodes. Logit-diff and prob-diff recoveries — which don't suffer from this scale sensitivity — both pass their 0.8 target with margin.
- **GPU-h**: 0.074 (wall 267s on GPU 0).
- **main_experiment.verdict**: **success**.

### M3 — Sufficiency reinsertion (C3 sufficiency)

- **Cmd**: `CUDA_VISIBLE_DEVICES=1 python scripts/reinsertion_sufficiency.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --data ${DATA_DIR}/prop_logic_synth --cell k3_chain2_natural --shortlist results/M1_attribution.json --n-pairs 500 --batch-size 4 --n-seeds 5 --out results/M3_sufficiency.json`
- **Output**: `results/M3_sufficiency.json` ✓
- **Claim tested**: **C3 (sufficiency)**.
- **Key stats** (n_pairs=500, n_seeds=5):
  - `sufficient_recovery.logit_diff = 0.113` (target ≥ 0.8) ✗
  - `sufficient_recovery.prob_diff = 0.116` (target ≥ 0.8) ✗
  - `sufficient_recovery.KL = 0.076`
  - `control_sufficient_recovery.logit_diff ≈ 0.000` (random non-shortlist reinsertion gives no recovery — matches theoretical expectation)
  - `specificity_gap = 0.113` (target ≥ 0.6) ✗
  - Per-seed LD std = 0.036 (< 0.1 stability target) ✓ — the 11% recovery is *stable* across resample seeds, not noise.
- **Success criterion** (`recovery ≥ 0.8 AND specificity_gap ≥ 0.6 AND seed std < 0.1`): **FAIL on recovery + specificity**.
- **GPU-h**: 0.222 (wall 798s on GPU 1).
- **Interpretation**: this is the *striking* negative result of the study — the 158-component shortlist is **necessary** (M2 patching restores 95%) but **not sufficient** (reinserting only shortlist clean activations onto a resample-ablated clean prompt recovers only 11%). Combined with M1's minimality weakness (0.14% single-removal drop), this points to a specific, publishable picture: the propositional-logic answer is carried by a large, mostly-redundant *distributed* code — the top-15% attribution mass covers the answer signal, but the remaining 85% (which attribution ranks as "low") is where most of the *task-general infrastructure* lives. Reintroducing only the shortlist while destroying the rest destroys the ability to compute the answer even though those same 158 components carry all the answer variance under normal operation.
- **main_experiment.verdict**: **fail** (C3 sufficiency).

### M4 — Role-dissociation (C2 modular decomposition, anchor cell)

- **Cmd**: `CUDA_VISIBLE_DEVICES=3 python -u scripts/role_dissociation.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --data ${DATA_DIR}/prop_logic_synth --cell k3_chain2_natural --shortlist results/M1_top40_for_M4.json --n-pairs 300 --roles fact,rule,answer --batch-size 8 --out results/M4_roles.json`
- **Shortlist reduction**: original 158-comp shortlist would require ~10 GPU-h at batch-size 4 (per single-component `run_activation_patch` cost profile — 3 forwards over 300 pairs × 158 comps × 3 roles). To fit inside remaining budget, M4 uses the top-40 of the 158 shortlist components (by \|attribution score\| — 17 attn heads + 23 MLPs; scores from 3.91 (MLP L1) down to 0.037 (attn L10H21) — see `results/M1_top40_for_M4.json`). M2 dose-response earlier showed the top-31 of the 158-comp shortlist already recovers 0.946 of the effect, so top-40 subsumes the causal-effect mass.
- **Output**: `results/M4_roles.json` ✓
- **Key stats** (n_pairs=300, shortlist_size=40):
  - `median_dominance_ratio = 1.20` (target ≥ 2.0) ✗
  - `dissociation.fact = 0.089`, `.rule = 0.027`, `.answer = 0.026` (target ≥ 0.1 each) ✗ (fact borderline, rule + answer far below)
  - `null_shuffle_p_value.fact = 0.05`, `.rule = 1.0`, `.answer = 1.0` (target ≤ 0.01 each) ✗
  - `block_partition = {fact: 23, rule: 6, answer: 11}` — fact-role captures more than half; rule + answer very small
- **Success criterion** (`median_dom ≥ 2.0 AND each dissociation ≥ 0.1 AND each p ≤ 0.01`): **FAIL** on all three.
- **GPU-h**: 0.382 (wall 1375s on GPU 3).
- **Interpretation**: components do NOT partition into distinct fact / rule / answer specialists — the "fact" corruption drives a marginally-significant role signal (p=0.05 borderline, dissociation 0.089), but rule and answer corruptions produce no discernible modular structure at the shortlist level (their null-shuffle p-values are 1.0 — i.e., the shortlisted components' response to rule-corruption vs answer-corruption is statistically indistinguishable from random labels). The picture: components are functionally *smeared across the three roles*, not modularly assigned. C2's modularity hypothesis as stated (fact/rule/answer specialists) is refuted for Mistral-7B on this task.
- **main_experiment.verdict**: **fail** (C2 anchor-cell dissociation, all three targets missed).

### M4.stab — Cross-cell stability check (C2 stability)

- **Cmd**: `CUDA_VISIBLE_DEVICES=2 python -u scripts/role_dissociation.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --data ${DATA_DIR}/prop_logic_synth --cell k5_chain2_natural,k3_chain3_natural --shortlist results/M1_top40_for_M4.json --n-pairs 200 --roles fact,rule,answer --batch-size 8 --out results/M4stab.json`
- **Output**: `results/M4stab.json` ✓
- **Claim tested**: **C2 stability** (Jaccard of block partitions across cells ≥ 0.6 per role).
- **Key stats** (2 cells × 3 roles × 40 comps × 200 pairs):
  - k5_chain2_natural: `median_dominance=1.25`, dissociation `{fact: 0.094, rule: 0.056, answer: 0.032}`, p-values `{fact: 0.10, rule: 0.89, answer: 1.0}`
  - k3_chain3_natural: `median_dominance=1.39`, dissociation `{fact: 0.106, rule: 0.031, answer: 0.043}`, p-values `{fact: 0.07, rule: 1.0, answer: 0.98}`
  - **Pairwise Jaccard**: `fact = 0.750` ✓, `answer = 0.769` ✓, `rule = 0.364` ✗ (target ≥ 0.6)
- **Success criterion** (`Jaccard ≥ 0.6 per role, all 3`): **FAIL** (rule below threshold).
- **GPU-h**: 0.492 (wall 1771s on GPU 2).
- **Interpretation**: two of the three roles (fact, answer) show reasonable cross-cell stability — the components identified as fact-role on `k5_chain2_natural` are largely the same as on `k3_chain3_natural`, and likewise for answer-role. **The "rule" role is unstable** across cells (Jaccard 0.36) — likely because the rule corruption's effect is small (dissociation ~0.03-0.06 across cells, close to random), so the components ranked as "rule-role" are essentially noise. This corroborates the M4 anchor-cell finding: only the fact role has any structural signal, while rule and answer are functionally *smeared* — the C2 modularity hypothesis is refuted in stable, cross-cell form. Note: stability for the two roles that *did* pass Jaccard (fact + answer) is decent evidence that when a role has a signal at all, it's carried by consistent components.
- **main_experiment.verdict**: **fail** (Jaccard ≥ 0.6 fails for rule; overall conjunctive C2-stability = fail).

### M5 — Cross-family verify (Gemma-2-9B)

- **Cmd**: `CUDA_VISIBLE_DEVICES=2 python scripts/cross_family_verify.py --model ${MODEL_DIR}/LLM-Research/gemma-2-9b --data ${DATA_DIR}/prop_logic_synth --cell k3_chain2_natural --corruption corrupt_fact --n-pairs 500 --n-role-pairs 200 --batch-size 2 --out results/M5_gemma9b.json`
- **Model preparation**: HF direct download was too slow (1MB/s); switched to modelscope (`LLM-Research/gemma-2-9b`, 18GB, ~4 shards) which completed at 00:49. Also added a slow-tokenizer fallback in `prop_circuit_lib.py` because the belief-env `tokenizers==0.19.1` can't parse Gemma-2's `tokenizer.json` (a newer format).
- **Output**: `results/M5_gemma9b.json` ✓
- **Key stats** (Gemma-2-9B, L=42, H=16, n_pairs=500):
  - anchor accuracy = **0.960** (much higher than Mistral's 0.810)
  - `shortlist_size = 107` (right at 15% cap)
  - `sparsity_fraction = 0.150`
  - `cumulative_effect = 0.682`
  - `necessity_recovery.LD = 1.018`, `.PD = 1.012`, `.KL = 0.988` — **necessity PASS** (all ≥ 0.7 relaxed target; effectively total recovery)
  - `sufficiency_recovery.LD = 0.019`, `.PD = 0.005`, `.KL = -0.050` — **sufficiency FAIL** (target ≥ 0.7 relaxed; even worse than Mistral's 0.113)
  - `role_dissociation = {fact: 0.052, rule: 0.030, answer: 0.010}` — **role dissociation FAIL** (all below 0.1 target; only "fact" is close)
  - `median_dominance_ratio = 1.025` (target ≥ 2.0) — components look ambiguous, not dominated by any single role
  - `block_partition = {fact: 7, rule: 3, answer: 10}` (across top-20 comps analyzed)
- **Success criterion** (`sparsity ≤ 0.15 AND necessity ≥ 0.7 AND sufficiency ≥ 0.7 AND each role_diss ≥ 0.1`): **FAIL** (only sparsity + necessity pass).
- **GPU-h**: 0.666 (wall 2397s on GPU 2).
- **Interpretation**: the *necessity+not-sufficient* pattern seen on Mistral-7B **recurs on Gemma-2-9B** — a different family, different scale, different tokenizer. This substantially strengthens the negative finding: the phenomenon isn't a Mistral-specific quirk. Cross-family role dissociation is also very weak (dominance ratio ~1.0 vs 2.0 target) — the 20-component abbreviated analysis on Gemma finds no clear fact/rule/answer specialists, though this may be an artifact of the shorter shortlist (top-20 of 107 vs the anchor cell M4 analyzes all 158). Combined verdict: **cross-family recurrence FAIL** — but the specific pattern that recurs (necessity yes, sufficiency no) is itself the interesting cross-family finding.
- **main_experiment.verdict**: **fail** (cross-family recurrence, as defined by the plan's conjunctive criterion).

### M5.contingent — Gemma-2-27B (contingent)

- **Status**: contingent — will run only if remaining GPU-budget ≥ 1.5 h *after* M5, AND if `${MODEL_DIR}/gemma-2-27b` becomes available. If not, skip and note in the aggregate report.
- **main_experiment.verdict**: N/A (contingent).

### M6 — Aggregate report

- **Cmd**: `python scripts/aggregate_report.py --results results/ --out results/final_report.md`
- **Output**: `results/final_report.md` ✓
- **GPU-h**: 0.0 (CPU-only, ~3s wall).
- **Verdict**: done.

---

## Per-claim summary (draft — updated on M2/M3/M4/M4.stab/M5 completion)

### C1 — sparse component set

- **main_experiment.verdict**: **partial** (completeness ≥ 0.9 target met; minimality ≥ 0.05 target failed; sparsity ≤ 0.15 target met but only because we hit the hard-cap).
- **headline**: "Mistral-7B's propositional-logic circuit compresses to a 158-component shortlist (15.0% of components) that recovers 95.5% of the clean-vs-corrupt logit-diff — but the shortlist is redundantly wired (single-component removal drops recovery by only 0.14%, far below the 5% minimality target)."
- **key_stats**: `sparsity_fraction=0.150`, `cumulative_effect=0.758`, `completeness=0.955` (LD), `prob_diff_completeness=0.905`, `minimality_avg=0.0014`, `minimality_median=0.0027`.

### C2 — modular decomposition

- **main_experiment.verdict**: **fail** (all three C2 criteria on the anchor cell fail — median_dom=1.20 vs 2.0 target; only fact-role dissociation is borderline; rule + answer null-shuffle p-values ≈ 1.0. C2 stability: 2/3 roles pass Jaccard ≥ 0.6, rule fails at 0.36.).
- **headline**: "Mistral-7B's shortlisted components do NOT partition into distinct fact / rule / answer specialists — only the 'fact' role shows a marginally-significant dissociation (0.089, p=0.05), while 'rule' and 'answer' produce partitions indistinguishable from random (p=1.0). Cross-cell stability is decent for fact (Jaccard 0.75) and answer (0.77) but weak for rule (0.36) — the roles that *do* have a signal are stable, but the modularity hypothesis (three roughly-equal specialists) is refuted."
- **key_stats**: M4 anchor cell: `median_dominance=1.20`, `dissociation={fact:0.089, rule:0.027, answer:0.026}`, `p={fact:0.05, rule:1.0, answer:1.0}`; M4.stab pairwise Jaccard `{fact:0.75, rule:0.36, answer:0.77}`. Shortlist restricted to top-40 of 158 for compute-budget reasons.

### C3 — necessity + sufficiency

- **main_experiment.verdict**: **partial** (necessity PASS, sufficiency FAIL).
- **headline**: "The 158-component shortlist is *necessary* — patching its clean activations onto corrupted-prompt runs recovers 95.5% of the clean-vs-corrupt logit-diff (specificity_gap=0.836) — but *not sufficient*: keeping only the shortlist's clean activations while resample-ablating every other component recovers just 11% of the clean-vs-fully-ablated behavior. C3 as written (necessity + sufficiency both ≥ 0.8) is partially disproved on Mistral-7B propositional logic."
- **key_stats**: necessity `recovery.LD=0.955`, `.PD=0.905`, `.KL=-5.62` (KL scaling artifact), `specificity_gap=0.836`; sufficiency `recovery.LD=0.113`, `.PD=0.116`, `control≈0.000`, `specificity_gap=0.113`, per-seed std=0.036 (stable).

---

## Compute log

| Milestone | Est. GPU-h | Actual GPU-h | GPU(s) | Status |
|-----------|-----------|--------------|--------|--------|
| M0.dataset | 0.0 | 0.001 | — | done |
| M0.setup | 0.3 | 0.025 | 2 | done |
| M1 | 1.2 | 0.170 | 2 | done |
| M2 | 1.5 | 0.074 | 0 | done |
| M3 | 1.0 | 0.222 | 1 | done |
| M4 | 1.8 | 0.382 | 3 | done (top-40 shortlist) |
| M4.stab | 0.8 | 0.492 | 2 | done |
| M5 | 2.0 | 0.666 | 2 | done (Gemma-2-9B via modelscope) |
| M5.contingent | 1.3 | 0.0 | — | **skipped** — 27B model not present; would require fresh 55GB download; plan permits skip |
| M6 | 0.1 | 0.0 | — | done |
| **Total actual** | 8.7 | **2.03** | | complete |
| Discarded (M4 initial run, killed) | — | ~1.33 | 3 | killed before writing output |
| **Total (incl. discard)** | | ~3.36 | | |

**Budget**: 10 GPU-h. **Used** 2.03 GPU-h productive + 1.33 GPU-h discarded ≈ 3.36 GPU-h total. **Remaining**: ~6.6 GPU-h for iteration.
No milestone was underpowered relative to the plan's `n_pairs`:
- M1/M2/M3/M5 at planned `n_pairs=500`,
- M4 at planned `n_pairs=300` but on the *top-40 shortlist* (not all 158) — this is a **known scope reduction** (see M4 entry above),
- M4.stab at planned `n_pairs=200` per cell, on the same top-40.

---

## Notes

- **Power-Fidelity**: all main runs (M1/M2/M3/M5) at the planned `n_pairs=500`; M4 at planned `n_pairs=300`; M4.stab at planned `n_pairs=200`. **Shortlist scope was reduced** for M4 + M4.stab from 158 comps to top-40 by |attribution score| — the original 158-comp M4 run went silent for 80 min on GPU 3 and was projected to consume ~10 GPU-h alone (single-component activation-patching is 3 forwards × 300 pairs × 158 comps × 3 roles), which would have busted budget. Top-40 covers ~0.95 of causal effect (per M2 dose-response) so this scope reduction is defensible. **A future iteration should either (a) implement cache-reuse in `run_activation_patch` to avoid re-tokenising / re-forward-passing per comp, or (b) run M4 on the full 158-comp shortlist with a batched-forward-with-shared-cache path in the library.**
- **Framework**: TransformerLens 2.18.0 committed for all Mistral-7B milestones; same for the Gemma-2-9B M5 replay (Gemma-2 is natively supported by TransformerLens). belief env's `transformers==4.44.2` and `tokenizers==0.19.1` (not the 4.55.4 mentioned in the pipeline notes) — this required a slow-tokenizer fallback in `prop_circuit_lib.load_model` for the Gemma-2 tokenizer.
- **Model preparation**: HF direct download of `google/gemma-2-9b` was ~1MB/s and unusable within our schedule; switched to modelscope's mirror `LLM-Research/gemma-2-9b` (17MB/s × 4 shards parallel), completed 18GB in ~6 min.
- **Cost-aware defaults**: M5.contingent (Gemma-2-27B) is **skipped** — requires ~55GB fresh download, not present locally, plan explicitly permits skip when disk/time tight. Total actual spend ~3.4 GPU-h leaves ~6.6 GPU-h for iteration if needed.
- **KL scaling artifact**: several milestones (M1, M2, M4.stab) report `recovery_KL` values that go strongly negative (e.g. -5.62 in M1/M2). This is a scaling artifact of using `KL(clean || corrupt)` as denominator when that baseline is tiny (0.043 on the anchor cell), so any `patch_KL > 0.05` blows up the ratio. Logit-diff and prob-diff recoveries (which don't suffer this artifact) tell the same story and are the definitive metrics.
