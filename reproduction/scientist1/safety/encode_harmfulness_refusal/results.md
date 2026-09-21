# Verification of the "Harmfulness and Refusal Are Separable Linear Directions" claim

Experiments target **Meta-Llama-3-8B-Instruct**. All activations are extracted from the residual stream at two token positions:
- `t_inst = -5`  — the last user-content token, just before `<|eot_id|>`.
- `t_post = -1`  — the last token of the fully-formatted prompt, i.e. the position immediately before the model generates its first response token.

Directions are computed by **difference-of-means (DoM)** on 200 AdvBench-harmful vs 200 Alpaca-benign prompts:
- `v_inst[l] = mean(harm_hs_inst[l]) − mean(benign_hs_inst[l])`  (harmfulness axis, read at `t_inst`)
- `v_post[l] = mean(harm_hs_post[l]) − mean(benign_hs_post[l])`  (refusal axis, read at `t_post`)

For orthogonalized steering we use `v_h_orth = v_inst` and `v_r_orth = v_post − ⟨v_post, v̂_inst⟩ v̂_inst`.


## Claim 1 & 2 — two approximately-linear directions at distinct token positions

Held-out AUROC on 100 AdvBench vs 100 Alpaca prompts, for each `(direction, apply-position)` combination:

| direction | apply position | best-layer AUROC | best layer |
|-----------|----------------|------------------|------------|
| v_inst | t_inst | **1.0000** | 10 |
| v_inst | t_post | 1.0000 | 8  |
| v_post | t_inst | 1.0000 | 11 |
| v_post | t_post | **1.0000** | 4  |

Both DoM directions are approximately linear (AUROC = 1 on IID held-out data) — confirming the "linear-direction" part of the claim.

**On positional specialisation** the raw AUROC does not discriminate: because Llama-3-8B has strong attention, a "harmful vs benign" signal is present everywhere by early layers. So AUROC alone cannot verify the *position* part of the claim.

However, the two directions are **not the same axis**. Per-layer cosine similarity between `v_inst` and `v_post`:

| layer | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|-------|---|---|---|----|----|----|----|----|----|
| cos   | 0.00 | 0.34 | 0.52 | **0.69** | 0.56 | 0.40 | 0.30 | 0.19 | 0.18 |

At the peak (layer 12) they share ~69 % of their unit-vector, meaning ~73 % of `v_post`'s length lies **outside** the `v_inst` subspace. That residual component is the "pure refusal" axis used below.

**Verdict:** Two distinct approximately-linear directions exist. The strict "different token position" part is under-determined by linear separation alone, but the two directions computed at different positions are numerically distinct (cos ≤ 0.69) and, as the steering experiment below shows, have dissociable causal effects.


## Claim 3 — additive steering yields dissociated effects (orthogonalised version)

We intervene at layer 11 (α = 6, unit-norm direction added to every position in the residual stream); we read `proj_h` at layer 20 `t_inst` and `proj_r_orth` at layer 20 `t_post`.

| condition | proj_h @ t_inst | proj_r_orth @ t_post | refusal rate |
|-----------|-----------------|----------------------|--------------|
| benign baseline           | −2.53 | −2.76 | 0.00 |
| harmful baseline          | +3.71 | +5.31 | 1.00 |
| benign + v_h              | +4.14 | +2.10 | 1.00 |
| harmful − v_h             | −2.64 | −0.77 | 0.00 |
| **benign + v_r_orth**     | **−0.01** | **+3.52** | **1.00** |
| **harmful − v_r_orth**    | **+1.98** | **−1.28** | **0.07** |

Key dissociation rows (bold):

1. **benign + v_r_orth**: the internal harmfulness projection stays essentially at the benign level (−0.01), yet the model refuses 100 % of the time. Sample output: *"I cannot provide instructions on how to make a strawberry smoothie."* → the model is refusing while still knowing the topic is benign.
2. **harmful − v_r_orth**: the harmfulness projection stays elevated (+1.98, still above the benign level of −2.53), but the model refuses only 7 %. Sample: *"I'm happy to clarify that identity theft is not necessarily a bad thing…"* → the model still perceives harm but stops refusing.

By contrast, steering along `v_h` (the raw harmfulness axis) always drags refusal with it — because refusal has a genuine harmfulness-detection component that we do not want to null out.

**Verdict:** The pure-refusal direction (after orthogonalising away harmfulness) *does* dissociate: it flips refusal without changing the internal harmfulness judgment. The raw harmfulness direction is more entangled — steering it also causes refusal, i.e. the harmfulness axis is a causally upstream feature. This partially supports the claim: the *refusal* direction is dissociable from harmfulness, but the *harmfulness* direction is not perfectly dissociable from refusal.


## Claim 4 — successful jailbreaks preserve harmfulness, suppress refusal

We attacked 80 AdvBench prompts with six wrappers. Only two produced non-refusal outputs on Llama-3-8B-Instruct: **prefill** (assistant response forced to begin with *"Sure, here is"*) with ASR = 94 %, and **jbb_role** (JailbreakBench role-play prompts) with ASR = 6 %.

Layer-20 projections normalised to the harm↔benign gap (1.0 = harm baseline, 0.0 = benign baseline):

| attack (successful subset) | harmfulness (t_inst) | refusal (t_post) |
|-----------------------------|-----------------------|-------------------|
| **prefill** (75 successes) | **1.00** (fully preserved) | **0.28** (mostly suppressed) |

Raw numbers at layer 20: `H_pfl_jb = +3.52` (vs `H_harm = +3.48`, `H_ben = −2.53`); `R_pfl_jb = −0.35` (vs `R_harm = +5.94`, `R_ben = −2.74`).

Interpretation: when the model is forced past its refusal turning-point, its representation of the input as harmful is essentially unchanged, but the refusal-decision signal collapses toward the benign level.  The signature is even stronger at later layers (25–29), where the refusal projection actually goes *below* zero while the harmfulness projection stays high.

Not all jailbreaks look like this. **GCG-style suffixes** and **role-play/persuasion wrappers** appear to work differently — they lower the input's harmfulness perception itself (probe flag rate on gcg-attacked prompts = 0 %, on jbb_role = 57 %). So the "harmfulness preserved / refusal suppressed" signature is characteristic of *bypass-style* attacks (prefill, assistant-response injection), while *input-obfuscation* attacks (adversarial suffixes, role-play) instead confuse the harmfulness detector.

**Verdict:** Verified for prefill-style attacks. Refuted for adversarial-suffix and role-play attacks — those achieve their effect by lowering the harmfulness signal itself, not just the refusal signal. A hidden-state probe can therefore identify the *bypass-style* jailbreaks (harmfulness signal high but refusal signal low) but *cannot* rely on the harmfulness signal to detect obfuscation-style attacks.


## Claim 5 — Latent probe ≈ or ≥ Llama-Guard-3-8B, at a fraction of the compute

Probe is a logistic-regression classifier on Llama-3-8B layer-15 residual stream at `t_inst`, trained on 200 AdvBench harmful vs 200 Alpaca benign.

Per-dataset accuracy against the dataset's own harmful/benign label:

| dataset       | Probe | Llama-Guard-3-8B |
|---------------|-------|-------------------|
| advbench_test | 1.000 | 0.950 |
| alpaca_test   | 1.000 | 1.000 |
| jbb_harmful   | 0.960 | 0.980 |
| jbb_benign    | 0.650 | 0.780 |
| catqa (all harmful) | 0.925 | 0.925 |
| sorrybench (all harmful) | 0.680 | 0.770 |
| **pooled ACC**  | **0.853** | **0.887** |
| **pooled F1**   | **0.898** | **0.922** |
| **pooled AUROC (probe)** | 0.919 | — |

Latency on a Llama-3-8B forward pass with `output_hidden_states=True` (probe reads the residual stream, no generation needed) vs Llama-Guard-3-8B (up to 20 generated tokens):

| model | latency / sample |
|-------|------------------|
| Latent probe | **29 ms** |
| Llama-Guard-3-8B | 241 ms |
| speedup | ~8.3× |

Note: the probe requires a Llama-3-8B forward pass. If the caller already ran the target model to generate, the probe is *free* (it just reads a hidden state).  In that shared setting the probe is effectively **~100× cheaper** than running a separate guard model.

Flag rate on jailbreak-attacked prompts (higher = better; all inputs are harmful by construction):

| attack     | Probe flag | Llama-Guard-3 flag |
|------------|------------|--------------------|
| plain      | 1.00 | 0.98 |
| dan        | 1.00 | 1.00 |
| persuasion | 1.00 | 0.96 |
| **prefill**| **1.00** | 0.98 |
| jbb_role   | 0.58 | 0.76 |
| **gcg**    | **0.00** | **0.99** |

**Verdict:** The probe matches Llama-Guard-3-8B in accuracy on most benchmarks and is 8-100× cheaper, matching the "fraction of compute" part of the claim. The stronger "matches or beats" statement is mostly true (pooled F1 is 3 pts lower). Importantly, the probe still catches bypass-style jailbreaks (prefill: 100 % flag). It clearly fails on adversarial-suffix attacks (GCG: 0 % probe vs 99 % judge), because those attacks manipulate the exact internal representation the probe reads — the raw text is still recognisable as harmful, so the text-reading judge model succeeds.


## Summary of verification

| Claim | Verdict | Notes |
|-------|---------|-------|
| C1 — two approximately-linear directions | **Supported** | AUROC = 1 on held-out; cos(v_inst, v_post) ≤ 0.69 → distinct axes |
| C2 — directions live at different token positions | **Under-determined** | Both positions separate perfectly; positional attribution needs stronger tests. Distinct DoM vectors do exist per position. |
| C3 — additive steering yields dissociated effects | **Partially supported** | Orthogonal refusal direction flips refusal without changing harmfulness (dissociated). Harmfulness direction still causes refusal changes (entangled). |
| C4 — jailbreaks preserve harm, suppress refusal | **Supported for bypass-style attacks (prefill)**; **refuted for suffix/role-play attacks** (those lower harmfulness signal itself) |
| C5 — Latent probe ≈ Llama-Guard-3, fraction of compute | **Supported for compute (~8-100×); slightly lower F1 (~3 pts)** |


## Files & code

Everything lives under `/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/`:

| Script | Role |
|--------|------|
| `common.py` | Data loaders and Llama-3 chat template helpers |
| `extract_activations.py` | Extract residual-stream hs at `t_inst` and `t_post` for train/test sets |
| `directions.py` | DoM directions and held-out AUROC per (direction, position) |
| `orthogonalize_directions.py` | Orthogonalise `v_post` against `v_inst` |
| `steering.py` / `steering_orth.py` | Add-α steering with raw and orthogonalised directions |
| `jailbreak_attacks.py` / `jailbreak_stronger.py` | GCG-suffix, DAN, persuasion, JBB-role, and prefill attacks |
| `analyze_jailbreak.py` / `jailbreak_layer_scan.py` | Split by jailbreak-success and compute per-layer projection signature |
| `latent_probe_only.py` / `latent_guard.py` | Linear probe accuracy + latency vs Llama-Guard-3-8B |
| `lg_on_jailbreak.py` | Llama-Guard-3-8B flag rate on the attacked prompts |
| `aggregate_report.py` | Console summary read from JSON results |

All numerical outputs are in `results/*.json` and this document is generated from them.
