# Experiment Plan — Steerable Social-Variable Directions in an LLM Dictator

**Behavior-source**: given
**Mechanism**: discovery
**Main-experiment model**: `Llama-3.1-8B-Instruct` (HARD, `task.md` §4) — path `/data/zhenqian/models/Llama-3.1-8B-Instruct`
**Resource fidelity**: NOT stamped strict (this is the given + discovery combo, cost-aware defaults are allowed)
**Chosen mechanism**: n/a (routed at experiment stage)
**Provenance**: `constructed` — 1,000-trial dictator-game corpus + 4×1,000 paired-prompt partners
**GPU budget (HARD)**: 10 GPU-hours total; only `gpu_id ∈ {1, 2, 3, 5, 6}`; `conda env` for all Python

**Mechanism strategy** (see `FINAL_PROPOSAL.md` §mechanism_strategy for the rejected-direction rationale):
```yaml
mechanism_strategy:
  directions: [Location, Causal Intervention]
  rejected:
    - Tuning & Editing — the claim is diagnostic (does the direction cause the decision?), not a capability-gain claim.
    - Formation Tracing — training-time origin is not part of the claim; too expensive under 10 GPU-hours.
    - Unit Interpretation — the concepts (G / A / I / M) are pre-named; SAE / auto-interp is not required.
    - Decision Auditing — trustworthiness / spurious-cue audit is out-of-scope for the four-part claim.
  note: Extract per-variable directions (Location), decorrelate into pure directions (still Location), then activation-add / directional-ablate to fill a 4×4 selectivity matrix (Causal Intervention). Diagonal validates C3; off-diagonal validates C4.
```

**Claim-to-milestone map**:

| Claim | Statement | Milestone(s) |
|---|---|---|
| C1 | Linear encoding of each social/contextual variable | M2 |
| C2 | Purity via decorrelation | M3 |
| C3 | Bidirectional causal steering | M4 |
| C4 | Selectivity (4×4 matrix) | M5 |

*(There is NO M0 in this plan — `BEHAVIOR_SOURCE=given` means the behaviour is taken as validated by prior work in the landscape; no `depends_on: [M0]` on any downstream milestone.)*

---

## M1 — Construct DG-1000 dictator-game dataset and paired-prompt partners

**Depends on**: — (root)
**Claim(s) covered**: precondition for all of {C1, C2, C3, C4}
**Kind**: dataset-construction
**Provenance**: constructed; `available_n=1000`, `used_n=1000` (main) + 4×1,000 paired-prompt partners

**What runs**:
1. Assemble a template pool for the dictator-game prompt: gender token (male/female descriptor), age token (young ≈ 25 / old ≈ 65), instruction phrasing (give-framing: "you may give up to $20 to X", take-framing: "X starts with $20, you may take some back"), meeting condition (meet: "you will meet X face-to-face after the study", no-meet: "you will not meet X"). At least 3 rewordings per variable to avoid over-fitting one phrasing.
2. Build a full 2×2×2×2 balanced block design (16 cells) × 63 trials/cell = 1,008 trials, then downsample to exactly 1,000 (stratified). Randomise the phrasing choice inside each cell.
3. For each of the 1,000 trials, generate the four minimal-edit partners (flip G, A, I, M independently) → 4,000 paired-prompt partners.
4. Enforce paired-prompt token-length constraint (≤ ±2 tokens) inside `Llama-3.1-8B-Instruct`'s tokenizer; retry rewording if violated.
5. Stratified split into 800 train (extraction / decorrelation fitting) + 200 held-out (steering evaluation).
6. Serialise the 5,000 prompts to `data/dg1000_prompts.jsonl` with columns `{trial_id, split, G, A, I, M, phrasing_id, is_paired_partner_of, prompt, expected_token_positions}`.

**Cmd**: `python scripts/build_dg1000.py --out data/dg1000_prompts.jsonl --seed 42 --n_trials 1000`
**Expected output**: `data/dg1000_prompts.jsonl` (~1.3 MB, 5,000 lines)
**Priority**: MUST-RUN (blocking)
**Estimated GPU-hours**: 0 (CPU-only)
**Notes**: Dataset is `constructed`. Follow `skills/data-rule/` for construction discipline (paired minimal-edit, length-matched, balanced randomisation, no leakage between train and held-out — stratify by cell, not by trial).

---

## M2 — Location: extract per-variable raw directions and probe them (C1)

**Depends on**: M1
**Claim(s) covered**: **C1 (linear encoding)**
**Kind**: mechanism-location
**Method family (routing hint)**: `representation_and_parameter_analysis` + `probing` on the `residual_stream` (fits `/mechanism-skills` "activation analysis / linear probing"). No SAE, no circuit discovery in this milestone.
**`method_sensitive`**: [`n_pairs`, `sites`, `metric`, `gpu_hours`]

**What runs**:
For each V ∈ {G, A, I, M}:
1. Forward pass `Llama-3.1-8B-Instruct` on the 800-train paired-partner set (V-flip only) at bf16. Cache residual-stream activations at every layer and at pre-registered token positions {end-of-prompt, end-of-question}. Batching: 32 prompts per micro-batch.
2. Compute the raw difference vector at each layer ℓ: `v̂_V^ℓ = mean_{(p, p')} [h(p') − h(p)]`.
3. **Probe fit**: 5-fold on-training linear-probe (logistic) for V on residual activations at each layer; report test accuracy on 200 held-out prompts, at each layer.
4. **Projection–transfer regression**: obtain baseline transfer amounts on the same held-out prompts (greedy decode, single number output; robust regex parser). Regress transfer ~ projection-of-h-onto-`v̂_V^ℓ` at each layer, per V.
5. **Mean-centring control**: also produce a mean-centred variant `v̂_V^ℓ − mean_h`.
6. Persist per V: chosen layer(s) ℓ_V*, `v̂_V^ℓ_V*` (raw), mean-centred variant, per-layer probe accuracy curve, projection–transfer β and p, α = 0 baseline transfer distributions.

**Cmd**: `python scripts/m2_extract_and_probe.py --model /data/zhenqian/models/Llama-3.1-8B-Instruct --data data/dg1000_prompts.jsonl --out artifacts/m2/ --gpus 1,2 --batch_size 32`
**Expected output**: `artifacts/m2/directions_raw.pt`, `artifacts/m2/probe_accuracy.json`, `artifacts/m2/projection_transfer.json`, `artifacts/m2/baseline_transfer.json`
**Priority**: MUST-RUN
**Estimated GPU-hours**: ~1.5 (5,000 prompts × forward pass × 4 V-flip corpora, bf16, batched)
**Success criteria (C1)**: per-V test probe accuracy ≥ 0.80 (for at least G, A, I — allow ≥ 0.70 for M); projection–transfer β significant (p ≤ 0.05) in the direction consistent with the baseline effect of V.

---

## M3 — Location: purify raw directions into pure directions via decorrelation (C2)

**Depends on**: M2
**Claim(s) covered**: **C2 (purity via decorrelation)**
**Kind**: mechanism-location
**Method family (routing hint)**: `representation_and_parameter_analysis` — linear projection / concept erasure (LEACE / Gram-Schmidt / R-LACE families). No fine-tuning.
**`method_sensitive`**: [`metric`, `gpu_hours`]

**What runs**:
1. **P-A · Gram-Schmidt (GS)** — for each V and its layer ℓ_V*, orthogonalise `v̂_V^ℓ_V*` against `{v̂_W^ℓ_V* : W ≠ V}`. Persist `ṽ_V^GS`.
2. **P-B · LEACE closed-form** — using the 800-train paired-partner activations, fit the LEACE projection Π^{(V)} that removes the linear signal of the three non-V variables. Apply Π^{(V)} `v̂_V^ℓ_V*` to obtain `ṽ_V^LEACE`.
3. **Cross-leakage evaluation**: for each pure direction `ṽ_V^decorr`, fit a linear probe for each W ∈ {G, A, I, M} on activations projected onto the direction, using the 200-held-out. Persist a 4 × 4 leakage matrix per decorrelator (V-rows, W-columns; diagonal = target, off-diagonal = leakage).
4. **Preservation-of-self check**: probe accuracy on V from `ṽ_V^decorr` compared to raw baseline (must retain ≥ 0.95× raw accuracy).
5. **Norm audit**: report `‖ṽ_V^decorr‖ / ‖v̂_V‖` per V to catch degenerate near-zero purified directions.

**Cmd**: `python scripts/m3_decorrelate.py --raw artifacts/m2/directions_raw.pt --acts artifacts/m2/heldout_activations.pt --out artifacts/m3/ --methods gs,leace`
**Expected output**: `artifacts/m3/directions_pure.pt`, `artifacts/m3/leakage_matrix.json`, `artifacts/m3/preservation.json`, `artifacts/m3/norm_audit.json`
**Priority**: MUST-RUN
**Estimated GPU-hours**: ~0.5 (CPU-heavy; only a small forward pass to build the held-out activation cache is on-GPU)
**Success criteria (C2)**: LEACE off-diagonal probe accuracy drops to chance + 0.05 or less; on-diagonal probe accuracy retained at ≥ 0.95× the raw baseline; `‖ṽ_V^LEACE‖ / ‖v̂_V‖ ≥ 0.30` (guards against degenerate directions).

---

## M4 — Causal Intervention: activation addition with α-sweep, both signs (C3)

**Depends on**: M3
**Claim(s) covered**: **C3 (bidirectional causal steering)**
**Kind**: mechanism-intervention
**Method family (routing hint)**: `causal_attribution` / activation addition on `residual_stream` (fits `/mechanism-skills` "steering / activation addition / directional ablation" family). Sign of α provides the "both directions" arm; directional ablation is the fallback in the ablation grid.
**`method_sensitive`**: [`n_pairs`, `sites`, `metric`, `gpu_hours`]

**What runs**:
For each V ∈ {G, A, I, M}:
1. Compute σ_ℓ = std of residual-stream norm at layer ℓ_V* on the held-out set.
2. **Grid**: `α ∈ {−4σ_ℓ, −2σ_ℓ, −σ_ℓ, 0, +σ_ℓ, +2σ_ℓ, +4σ_ℓ}` × decorrelator ∈ {`ṽ_V^GS`, `ṽ_V^LEACE`} × site ∈ {`single-layer S={ℓ_V*}`, `small-window S={ℓ_V* − 1, ℓ_V*, ℓ_V* + 1}`}.
3. For each grid point, run the 200 held-out trials through the model with the CAA intervention `h(s) ← h(s) + α · ṽ_V^decorr` at every s ∈ S at every post-prompt token, greedy-decode a transfer amount, parse and record τ.
4. **Coherence gate**: at α ∈ {±2σ, ±4σ}, sample K = 10 free-form generations per (V, α), check format ("integer 0..20 in the answer") and 5-gram repetition rate ≤ 0.5. Drop violating (V, α) cells from C3 headline.
5. Dose-response: for each V, fit `mean_transfer ~ α` per decorrelator; compute Spearman ρ and paired-sign tests (α > 0 vs. α < 0).
6. **Inversion audit**: for each V, find the smallest |α| < 0 at which the sign of V's baseline effect on the transfer flips.

**Cmd template (queue mode)**:
```
python scripts/m4_steer_eval.py --model /data/zhenqian/models/Llama-3.1-8B-Instruct --dir artifacts/m3/directions_pure.pt --variable ${V} --alpha ${alpha_mult} --site ${site} --decorrelator ${decorr} --out artifacts/m4/${V}_${decorr}_${site}_a${alpha_mult}.json
```
**Grid**: `{ V: [G, A, I, M], alpha_mult: [-4, -2, -1, 0, 1, 2, 4], site: [single, window3], decorr: [gs, leace] }` → 4 × 7 × 2 × 2 = 112 runs.
**Priority**: MUST-RUN
**Estimated GPU-hours**: ~2.5 (200 prompts × 112 runs × ~0.4 s/prompt with batching; can share the model in memory across runs — real overhead is per-decoding-step, not per-run setup)
**Success criteria (C3)**: at α = +σ_ℓ the mean-transfer shift ≥ 25% of the baseline effect of V, same sign; at least one α < 0 magnitude inverts V's baseline-effect sign; monotonicity Spearman ρ ≥ 0.7 up to the coherence-gate saturation.

---

## M5 — Selectivity: 4 × 4 matrix — steer V, measure change in W's effect (C4)

**Depends on**: M4
**Claim(s) covered**: **C4 (selectivity)**
**Kind**: mechanism-intervention-selectivity
**Method family (routing hint)**: same as M4 (activation addition), but the *measurement* is on the other three variables' effects, not on V's own effect.
**`method_sensitive`**: [`metric`, `gpu_hours`]

**What runs**:
1. For each V ∈ {G, A, I, M} and each α ∈ {−2σ_ℓ, 0, +2σ_ℓ}, with `ṽ_V^LEACE` at site `single-layer S = {ℓ_V*}`, forward the *full 200 held-out* through the intervention.
2. For each W ∈ {G, A, I, M} \ V, estimate the effect of W on the transfer (mean_transfer at W = w₁ minus mean at W = w₂, marginalising over the other two co-varying variables) at the given α.
3. Build the 4 × 4 selectivity matrix M[V, W] = effect_of_W(α) − effect_of_W(α = 0).
4. Report the diagonal (should be substantial; already covered by M4) and off-diagonal (should be small).
5. Statistical test: permutation test on "off-diagonal entries have the same magnitude distribution as diagonal entries" — expected rejection.

**Cmd**: `python scripts/m5_selectivity_matrix.py --dir artifacts/m3/directions_pure.pt --model /data/zhenqian/models/Llama-3.1-8B-Instruct --alpha_grid -2,0,2 --out artifacts/m5/`
**Expected output**: `artifacts/m5/selectivity_matrix.json`, `artifacts/m5/permutation_test.json`
**Priority**: MUST-RUN
**Estimated GPU-hours**: ~1.5 (200 held-out × 4 V × 3 α = 2,400 conditioned decodes; reuses many of M4's cached generations when α ∈ {−2σ, 0, +2σ, single-site, LEACE} — so real incremental cost is closer to ~0.5 h if M4's cache is available)
**Success criteria (C4)**: `max_offdiag |M[V, W]| ≤ 0.30 × min_diag |M[V, V]|`; permutation test rejects null "off-diagonal = diagonal" at p ≤ 0.05.

---

## M6 — Baselines & ablations (all four claims)

**Depends on**: M4 (uses the same evaluation harness)
**Claim(s) covered**: sanity checks + robustness for {C1, C2, C3, C4}
**Kind**: ablation-suite
**`method_sensitive`**: [`gpu_hours`]

Grid of ablations, all at fixed site = single-layer S = {ℓ_V*}, α ∈ {−2σ, 0, +2σ}:

| # | Ablation | Predicted result | Runs |
|---|---|---|---|
| B1 | Raw v̂_V (no decorrelation) | steers V but larger off-diagonal | 4 V × 3 α = 12 |
| B2 | Random unit vector (norm-matched) | no reliable dose-response | 4 V × 3 α × 3 seeds = 36 |
| B3 | Mean-centred v̂_V only | intermediate off-diagonal | 4 V × 3 α = 12 |
| B4 | Directional ablation `h ← (I − ṽ_V ṽ_Vᵀ) h` | acts as strong negative α | 4 V = 4 |
| B6 | Length / position-token regression check | diff-vector ⊥ length | CPU only |

**Cmd template**: `python scripts/m6_ablations.py --ablation ${ab} --variable ${V} --alpha ${alpha} --seed ${seed} --out artifacts/m6/${ab}_${V}_a${alpha}_s${seed}.json`
**Grid**: 4 ablation blocks (B1, B2, B3, B4) as listed → ~64 runs
**Priority**: SHOULD-RUN (fits in remaining budget)
**Estimated GPU-hours**: ~1.5 (200 prompts × 64 runs, cached model)

---

## M7 — Portability check on a swap model (hand-off to `/auto-verify`)

**Depends on**: M5
**Claim(s) covered**: robustness of {C1, C2, C3, C4}
**Kind**: verify-handoff
**`method_sensitive`**: [`n_pairs`, `sites`, `metric`, `gpu_hours`]

Repeat the whole M2 → M5 pipeline on `DeepSeek-R1-Distill-Llama-8B` (path `/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B`) on a subset of the DG-1000 corpus (400 trials + paired partners; the 8B DeepSeek's tokenizer and formatting may need per-model prompt adjustments — pre-register a per-model prompt-render function). Report whether the claim landings hold *qualitatively* — this milestone is treated as an initial `/auto-verify` seed, not as a headline result.

**Cmd**: `python scripts/m7_portability.py --model /data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B --data data/dg1000_prompts.jsonl --n_trials 400 --out artifacts/m7/`
**Expected output**: `artifacts/m7/portability_report.json`
**Priority**: MAY-RUN (nice-to-have; drops first if the 10 h budget tightens)
**Estimated GPU-hours**: ~1.5

---

## Total budget tally

| Milestone | GPU-hours (est) |
|---|---|
| M1 (dataset build, CPU) | 0 |
| M2 (extract + probe) | 1.5 |
| M3 (decorrelate) | 0.5 |
| M4 (α-sweep, both signs) | 2.5 |
| M5 (selectivity matrix) | 1.5 (or ~0.5 if M4 caches reused) |
| M6 (baselines & ablations) | 1.5 |
| M7 (portability, `/auto-verify` seed) | 1.5 |
| **Total** | **≤ 9.0 GPU-hours** |

Within the 10 GPU-hour HARD budget. If any milestone under-runs, the freed budget is reallocated to expand the α-grid or to add one additional decorrelator (R-LACE).

## Verify hand-off notes (to `/auto-verify`)

- Priority swap targets for stress-testing:
  - **model swap** → `DeepSeek-R1-Distill-Llama-8B` (portability; also lays a foundation for the "verify variants" line in `task.md`).
  - **decorrelator swap** → GS ↔ LEACE (already both run in M3 by default; add R-LACE-adversarial if budget allows).
  - **injection swap** → activation addition ↔ directional ablation (B4 is the seed).
  - **paired-prompt rewording swap** — build 5 additional rewordings per V and re-fit `v̂_V` under each; report variance.
- Each claim's `robustness` (per-`/auto-verify` semantics) is computed from the fraction of surviving swap variants that reproduce the qualitative landing (up-arrow C1/C2/C3, small-off-diagonal C4).
- Do not re-run M0 — this plan carries no M0 by design (behaviour is given by `task.md`).

## Filesystem discipline (`task.md` HARD #3)

All artifacts under the working directory tree:
- `data/` — prompts and derived splits
- `artifacts/m2/ … m7/` — cached activations, directions, per-milestone JSONs
- `refine-logs/` — this plan, the proposal, the tracker
- `idea-stage/` — landscape, raw retrieval, idea report

No writes outside working dir; models & data-raw are read from `/data/zhenqian/models` and `/data/zhenqian/data` only.
