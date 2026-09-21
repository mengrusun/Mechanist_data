# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-block-selection
  - steering-coefficient-tuning
  - finetune-hyperparameter-sweep

## Matches

1. **steering-block-selection** — M6 declares `site: [L_frame_top1, L_frame_top2]` (from M5 probe screening) — plan already uses activation-based screening (probe accuracy) rather than a hardcoded index, so the tip fires as reinforcement/audit, not correction.
   - convention to adopt: sites derive from M5 probe screening (activation-based), lock the site set **before** the coefficient sweep. If a single site shows no measurable steering effect at any α, extend to a 3–5 layer window around the site. Never copy a raw layer index across different-depth models (Qwen3-14B has 40 layers; use relative depth).
   - claim-matching rule: the CM claim ("low-rank residual-stream direction on early-to-mid layers") is regional → M6 must include the two top probe layers **plus** matched off-region layers as null controls (already partially covered by `filler_control` + `offtarget_medqa`, but the *layer* control is missing — add one matched-window layer far from top1/top2 as a null).

2. **steering-coefficient-tuning** — M6 declares `alpha: [null, -1.0, 0.0, 0.5, 1.0]` on the frame direction. Plan uses raw scale, no fluency metric alongside target metric, no σ-normalized β.
   - convention to adopt: (a) express α in σ_proj units — for each site `l`, compute `σ_l = std(h_l · u_frame)` on 500 tokens; the plan's α values are re-mapped as β = α / σ_l. (b) Always log a fluency / general-ability metric alongside target Δaccuracy — for GSM8K CoT, use **per-item output length in tokens** and **fraction of items whose completion parses via `## regex`** (a collapse into garble ⇒ parse rate drops sharply). (c) Prefer the smallest sufficient β; if plateau at β=0.5 and β=1.0 give same Δaccuracy but β=1.0 tanks parse rate, use β=0.5. (d) The plan grid {-1.0, 0.0, +0.5, +1.0} is a **coarse** sweep — record it in σ_proj units and, if the effect at these points is monotone with no collapse, keep it; if there is collapse at |β|≥1, fine-sweep around the largest-Δ non-collapsing point. (e) Add a mid-scale β = ±0.25 to the sweep if the {−1, 0, +0.5, +1} points are all inert — early-warning fine grid.

3. **finetune-hyperparameter-sweep** — M7b (SFT, 2 epochs) + M7c (REINFORCE, 1 epoch) on Llama-3.2-1B classifier over 13 discrete actions. No LR specified in plan. Small classifier head (not LoRA/PEFT) → full-FT regime.
   - convention to adopt: (a) Run one LR pilot per (model, data) pair — here **one** pilot suffices (same model, same reward-table data feeds both SFT and RL). (b) LR-first grid: full-FT SFT = `{5e-6, 1e-5, 2e-5, 5e-5, 1e-4}`; full-FT RL/REINFORCE = `{1e-7, 5e-7, 1e-6}` (RL is 10–100× smaller than SFT). Since Llama-3.2-1B is small (1B params, classifier head atop pooled hidden state), effective batch 32; 1 epoch pilot on 500 reward-table rows; smoothed-loss descent > 30 % over baseline → pass. (c) Report `sweep_status: swept` or `sanity_checked` for M7b's hyperparameters block. (d) Verify base-lookalike check: SFT'd classifier should predict something other than uniform-random over 13 classes (accuracy > 1/13 = 7.7 %).

## Composition order for M5 → M6

1. **Block/site selection first** (from tip 3): M5 probe accuracy ranks layers; take top-2 as `L_frame_top1/top2`. Lock this before M6.
2. **Coefficient tuning second** (from tip 2): with the site locked, run the {−1, 0, +0.5, +1} sweep (converted to σ_proj units) and log both target Δaccuracy AND fluency (length + parse-rate) per α; if collapse at ±1, fine-sweep around ±0.5.
3. **Match-to-claim rule**: report both target Δaccuracy AND fluency together (per General Rule — "locate then intervene without damaging general ability").

## Composition for M7

- LR pilot runs *before* M7b full retrain. Pilot: 500 rows, 1 epoch, 1 seed at each LR in the SFT grid. Winner (lowest converged loss) → M7b full-scale retrain at 3 seeds. Same pilot informs M7c's RL LR (start at winner-SFT-LR / 10, adjust one grid point).

## No-match log

- **image / ImageNet preprocessing**: N/A (all-text pipelines).
- **multiple-choice-evaluation (MCQ letter-parse)**: N/A — M3 uses constrained log-likelihood over letter options (already the tip's recommended path), NOT free-form regex letter extraction. M2 GSM8K uses numeric `## \s*(-?\d+)` regex, not letter parsing.
