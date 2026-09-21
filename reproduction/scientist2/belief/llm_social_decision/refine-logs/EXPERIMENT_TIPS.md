# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - general-mechanism-interpretability-rule
  - steering-block-selection
  - steering-coefficient-tuning
  - multiple-choice-evaluation (integer-parse variant, adapted)

## Matches

1. **general-mechanism-interpretability-rule** — always-on for mechanism / interpretability experiments (locate → intervene → measure general ability alongside target).
   - convention to adopt: at M4 (and every steering run) log a **coherence / general-ability metric** in parallel with the target transfer-shift metric — format-check (integer 0..20), 5-gram repetition rate ≤ 0.5, off-target token distribution. Any α whose gate fails is DROPPED from the C3 headline. Never report the target metric alone.

2. **steering-block-selection** — the plan targets residual-stream layers ℓ_V* + a `window3` ablation. Fires on `sites` = {single-layer, small-window} declaration in M4 and on the per-V ℓ_V* pick in M2.
   - convention to adopt: (a) pick ℓ_V* by **probe accuracy × projection-transfer β** on the 800-train paired-partner set (activation-based screen — allowed by the tip), evaluate on the held-out 200 only; (b) if a single layer shows no effect at α = ±σ, widen to the 3-layer window (this is exactly M4's built-in `window3` ablation grid — align with the tip); (c) never carry a raw layer index from a paper — layer 13 of a Llama-2-7b (32 blocks) → relative depth 0.41 → layer 13–14 of Llama-3.1-8B (32 blocks); check neighbours ±2; (d) match the claim — since C3/C4 are per-variable (not regional), a single layer is admissible if it survives the coherence gate.

3. **steering-coefficient-tuning** — the plan declares α ∈ {−4σ, −2σ, −σ, 0, +σ, +2σ, +4σ}. Fires on the σ-unit convention (good) but must ensure the pilot lands on the smallest sufficient α.
   - convention to adopt: (a) express α in units of the **per-layer projection std** `σ_ℓ = std(h · u_V^ℓ)` on the *held-out* set (not the residual-stream norm — the projection-std is the tip's canonical convention; correct the plan text at implementation time); the plan uses "std of residual-stream norm" — implement it as **projection-std of h onto the unit direction u_V^ℓ = v̂_V / ||v̂_V||**, which is the same quantity in expectation but strictly correct; (b) always include α = 0 as baseline (already in the grid); (c) log target-transfer + coherence per (V, α) — a strong target-shift with broken coherence is invalid and gets dropped; (d) prefer the **smallest** α that meets the C3 25%-of-baseline-effect bar — do not report only α = +4σ if α = +σ already lands; (e) if the target hits at α = +σ, do not push past +2σ for the C3 headline (report the +4σ tail as diagnostic).

4. **multiple-choice-evaluation** (integer-parse variant) — the dictator output is a scalar τ ∈ {0..20}, not an A/B letter, but the same silent-coercion failure mode applies: an over-tight regex will drop refusal / off-topic / broken generation into some default, inventing a Δ.
   - convention to adopt: (a) prompt asks explicitly for a single integer in the range 0..20 on the first line; (b) primary parser is a robust regex chain — first line integer → any integer 0..20 in the first 30 tokens → fall through; a **third bucket `parse_failure`** is required (never coerce to 10 or 0); (c) report per-run `parse_failure_rate` alongside mean_transfer; (d) label-floor pilot: on a 100-example held-out slice at α = 0, transfer std must be ≥ 0.5$ AND parse_failure_rate ≤ 0.1 — collapse is a Round-End Decision (`ended-needs-decision (experiment: scorer-invalid)`), NOT an auto-swap.

## No-match log

- Tip 1 (ImageNet / vision preprocessing) — not applicable (LLM residual-stream task, no torchvision).
- Tip 4 (fine-tuning hyperparameter sweep) — not applicable (no fine-tune milestone; steering only).

## Composition

Per `experiment-tips/SKILL.md` §Composing: **lock block-count first (tip 3) → sweep α next (tip 2)**. M2 picks ℓ_V* (single layer, probe-driven); M4's site grid = {single, window3} implicitly re-checks tip 3's "widen if inert" rule; the α-sweep in M4 then tunes coefficient per (V, decorrelator, site). This matches the plan's execution order exactly.
