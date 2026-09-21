import json, os
path_in = "scratchpad/p2/in/cand_04.json"
path_out = "scratchpad/p2/out/cand_04.json"
with open(path_in, encoding="utf-8") as f:
    d = json.load(f)

d["Experiments"] = (
"Compute fits an academic node (<= 8xA100-80GB). "
"Primary model: DeepSeek-R1-Distill-Qwen-7B, frozen; only linear probes are trainable, and interventions are activation edits (no weight updates). "
"Scale sweep: R1-Qwen distills 1.5B/7B/14B and Qwen3-8B. "
"Decoding: temperature 0.6, top-p 0.95, max 16k thinking tokens, seeds 0/1/2 on every reported number. "
"CoT segmented into steps by newline/step delimiters. "
"Datasets, used as released: GSM8K test (1319), MATH500 (500), AIME 2024+2025 (60), GPQA-Diamond (198); difficulty labels from Easy2Hard-AMC-style annotations plus empirical per-item pass-rates (32 samples/item).\n\n"

"Experiment 1 (H1 — Difficulty is decodable and head-mediated, including mid-trace): Replicate a linear difficulty probe and localize difficulty-sensitive final-layer heads, then test readability at intermediate positions. "
"Train the probe on pre-generation activations (40% of items) to predict difficulty (pass-rate bucket); evaluate on 60% held out. Identify difficulty heads by ablation impact on probe output. Then fine-tune/evaluate the probe on intermediate step-boundary states. "
"Decisive metric: held-out probe AUROC/R^2 for difficulty, pre-generation and at intermediate positions. "
"Convincing result: pre-generation AUROC >= 0.75 (replication) and intermediate-position AUROC >= 0.70, with a small head set (<= 10) whose ablation drops probe AUROC by >= 0.1. "
"Indecision band: intermediate AUROC in [0.6, 0.7] — difficulty is only weakly readable mid-trace; the dynamical framing is reported tentatively. "
"Negative result (intermediate AUROC < 0.6): difficulty is encoded only pre-generation, refuting the premise that it can be tracked during reasoning (an informative negative).\n\n"

"Experiment 2 (H2 — Difficulty normally contracts): Read step-wise difficulty and test whether it declines as work is done. "
"For each trace, form the difficulty trajectory over step boundaries. "
"Decisive metric: mean signed change in decoded difficulty from first to last step across correctly solved traces. "
"Convincing result: difficulty contracts on average by a significant margin (paired p < 0.01, mean drop > 0.3 in normalized probe units) on the majority of correctly solved traces. "
"Indecision band: mean drop in [0.1, 0.3] — weak contraction; we report a mild trend. "
"Negative result (mean drop < 0.1 or increasing): difficulty does not contract during reasoning, refuting H2.\n\n"

"Experiment 3 (H3 — Contraction failure tracks residual work and predicts waste): Measure residual work by truncate-and-resume (truncate at each step, force termination, answer; residual work = steps still needed to reach the correct answer), then define contraction failure = area between difficulty and residual-work trajectories after residual work hits zero. "
"Steps subsampled (every 2nd boundary) to control compute. "
"First test that difficulty tracks residual work (correlation), then test that contraction failure predicts waste. "
"Decisive metric: out-of-sample R^2 (5-fold) of contraction failure predicting wasted tokens (tokens after the last accuracy-changing step), minus best competitor R^2 among absolute pre-generation difficulty, trace length, and a confidence probe. "
"Convincing result: difficulty-vs-residual-work correlation r > 0.4, and contraction failure beats the best competitor by >= 0.10 R^2 on at least three datasets. "
"Indecision band: correlation in [0.2, 0.4], or R^2 advantage in [0.02, 0.10] — the account is suggestive but not dominant. "
"Negative result (r < 0.2 or R^2 advantage <= 0.02): overthinking is not the specific contraction-failure pattern, weakening H3.\n\n"

"Experiment 4 (H4 — Difficulty heads causally gate continuation): At low-residual-work / high-difficulty boundaries, patch/steer the localized difficulty heads toward their easy pole (or add the negative difficulty direction) and measure the effect on termination-token probability and continuation length, vs. control heads and norm-matched random directions. "
"Decisive metric: increase in termination-token probability under difficulty-head steering minus that under control-head steering. "
"Convincing result: difficulty-head steering raises termination probability by >= 15 absolute points and shortens continuation by >= 20%, both significantly above control/random steering, while accuracy on already-solved items holds within 2 points. "
"Indecision band: termination-probability gain in [5, 15] points over control — heads influence but do not clearly gate continuation. "
"Negative result (< 5 points over control, or indistinguishable from confidence-probe steering): difficulty heads are not a distinct causal gate, blurring the claim vs. confidence early-exit.\n\n"

"Experiment 5 (H4 — Training-free intervention saves tokens without hurting accuracy): Online, read step-wise difficulty; when difficulty fails to contract but a lightweight residual-work probe (trained on Experiment-3 labels, 40/60 split by problem) predicts low residual work for k consecutive steps, steer difficulty heads toward easy to induce termination. "
"k and the residual-work threshold selected on the training split (grid k in {2,3,5}). "
"Systems compared on all four datasets: intervention vs. (a) no intervention, (b) fixed truncation matched to intervention mean budget, (c) input-level difficulty budgeting (CODA/AdaCtrl-style), (d) length steering, (e) confidence-based early-exit. "
"Decisive metric: token reduction at equal accuracy — percent reduction vs. no intervention while accuracy stays within 1 point. "
"Secondary: mean tokens, accuracy, wall-clock speedup. "
"Convincing result: >= 25% token reduction within 1 accuracy point on GSM8K and MATH500, on or above every baseline frontier. "
"Indecision band: reduction in [10%, 25%] within 1 point, or a tie with the best budgeting baseline (< 5% difference at matched accuracy) — the intervention works but is not clearly superior; the dynamical characterization (Experiments 1-4) is then the standalone contribution. "
"Negative result (< 10% reduction or > 1 point accuracy loss): the intervention does not yield usable savings.\n\n"

"Experiment 6 (H4 — Underthinking guard): On the hardest quartile of items where residual work truly stays large, verify the trigger does not fire and does not induce premature termination. "
"Decisive metric: underthinking rate (accuracy drop on hard items) as a function of the residual-work threshold. "
"Convincing result: at the deployed threshold, hard-item accuracy drops < 1 point. "
"Indecision band: 1-3 point drop — threshold trades savings for safety; report the curve. "
"Negative result (> 3 point drop): the residual-work gate cannot protect hard items, bounding H4.\n\n"

"Experiment 7 (H1/H3 — Generality across data and scale): Transfer probe, head set, and trigger across datasets and the 1.5B/7B/14B/Qwen3-8B models without retraining. "
"Decisive metric: transferred-probe AUROC relative to in-domain, and intervention token-savings retention. "
"Convincing result: transfer retains >= 80% of in-domain AUROC on at least three scales; savings hold within 10 relative percent. "
"Indecision band: retention in [60%, 80%] — partly model-specific; scoped to per-model probes/heads. "
"Negative result (< 60% retention): the difficulty-dynamics signal is idiosyncratic across models."
)

tmp = path_out + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=4, ensure_ascii=False)
os.replace(tmp, path_out)
with open(path_out, encoding="utf-8") as f:
    print(list(json.load(f).keys()))
