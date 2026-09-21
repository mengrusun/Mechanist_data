import json, os
path_in = "scratchpad/p2/in/cand_09.json"
path_out = "scratchpad/p2/out/cand_09.json"
with open(path_in, encoding="utf-8") as f:
    d = json.load(f)

d["Experiments"] = (
"Compute fits an academic node (<= 8xA100-80GB). "
"Primary model: DeepSeek-R1-Distill-Qwen-7B, frozen; only a lightweight linear depth-probe is trainable, and the decoder itself is training-free. "
"Scale sweep: R1-Qwen distills 1.5B/7B/14B and Qwen3-8B (with L layers each). "
"Decoding: temperature 0.6, top-p 0.95, max 16k thinking tokens, seeds 0/1/2 on every reported number. "
"Datasets, used as released: GSM8K test (1319), MATH500 (500), AIME 2024+2025 (60), GPQA-Diamond (198).\n\n"

"Experiment 1 (H1 — Deep-thinking need is sparse and category-structured): For sampled traces, apply logit-lens at every layer and record each token's convergence depth (earliest layer after which the argmax equals the final output and stays stable); cross-check with the prediction-revised-in-deep-layers definition. Categorize tokens (function/connective, narration, numeric/operator, answer-region) via POS + regex. "
"Decisive metric: fraction of tokens that converge by layer 2L/3, and the share of late-converging (deep) tokens falling in numeric/logical/answer categories vs narration. "
"Convincing result: >= 70% of tokens converge by 2L/3, and >= 60% of deep tokens are numeric/logical/answer (not narration), with the two depth definitions agreeing (rank correlation > 0.5). "
"Indecision band: convergence-by-2L/3 in [50%, 70%], or deep-token category concentration in [45%, 60%] — the structure is present but weak; the decoder's headroom is limited and we say so. "
"Negative result (< 50% converge by 2L/3, or deep tokens spread evenly across categories): reasoning tokens are not predominantly shallow or not category-structured, refuting H1.\n\n"

"Experiment 2 (H2 — Deep-thinking need is decidable online mid-stack): Predict late-convergence from mid-layer state via (a) training-free mid-layer logit-lens top-1 margin/entropy and agreement with an adjacent layer, and (b) a lightweight linear probe on the mid-layer residual predicting whether the token changes by the final layer. "
"Probe data: 40% of tokens train, 60% held out, split by problem; checkpoint layer swept. "
"Decisive metric: held-out AUROC for predicting late-convergence at the best mid-layer checkpoint. "
"Convincing result: AUROC >= 0.80 at a checkpoint no deeper than 2L/3, with the training-free confidence check within 0.05 AUROC of the probe. "
"Indecision band: AUROC in [0.65, 0.80] — decidability is moderate; exit rates and savings will be modest. "
"Negative result (AUROC < 0.65): deep-thinking need is not predictable mid-stack, so the decoder cannot exit safely, refuting H2.\n\n"

"Experiment 3 (H3 — Training-free adaptive-depth decoder cuts compute at fixed accuracy): Decode to checkpoint layer c; if the checkpoint LM-head prediction is confident (margin > tau) and agrees with layer c-1, exit and emit, else run full depth; a self-speculative safeguard periodically re-verifies a batch of exited tokens at full depth and adjusts tau. "
"tau and c selected on the Experiment-2 training split. "
"Systems compared on all four datasets: adaptive-depth decoder vs. (a) uniform full depth, (b) fixed exit at c layers for all tokens, (c) CALM-style confidence early exit, (d) random exit at matched rate. "
"Decisive metric: average layers executed per token (FLOP proxy) at fixed accuracy — mean depth to match full-depth accuracy within 1 point; reported alongside true wall-clock. "
"Secondary: fraction of tokens early-exited, accuracy, wall-clock speedup (both a KV-recompute variant and a checkpoint-KV-reuse variant, per the KV-consistency caveat). "
"Convincing result: >= 30% reduction in average layers per token within 1 accuracy point on GSM8K and MATH500, beating fixed-layer exit and matching or beating CALM at equal accuracy, with a measurable (>= 15%) wall-clock speedup in at least one KV variant. "
"Indecision band: layer reduction in [15%, 30%] within 1 point, or a tie with CALM (< 5% difference at matched accuracy), or FLOP savings that do not translate to wall-clock (< 5% speedup due to KV overhead) — the decoder is a modest or systems-limited win; the mechanistic characterization (Experiments 1-2) remains the standalone contribution. "
"Negative result (< 15% layer reduction or > 1 point accuracy loss): shallow-token skipping does not yield usable savings.\n\n"

"Experiment 4 (H1 — Causal category check): Verify early-exited tokens are dominantly narration/connective and that forcing early exit on answer-region tokens is harmful. "
"Decisive metric: accuracy drop when early exit is forced on answer-region tokens vs. on narration tokens at matched exit rate. "
"Convincing result: forcing exit on answer-region tokens costs >= 10 accuracy points while forcing exit on an equal number of narration tokens costs < 2, confirming the depth need is causally tied to token category. "
"Indecision band: answer-region forced-exit cost in [3, 10] points — depth need is only partly category-tied. "
"Negative result (< 3 points on answer-region tokens): depth is not causally concentrated on computation tokens, weakening H1's mechanistic claim.\n\n"

"Experiment 5 (H3 — Complementarity with step-level stopping): Combine the token-wise depth decoder with a step-level early-stopping baseline and measure additive savings. "
"Decisive metric: compute (FLOP proxy) at fixed accuracy for the combined method vs. either method alone. "
"Convincing result: the combination saves >= 10% more compute at matched accuracy than the better single method, on the accuracy-compute Pareto frontier. "
"Indecision band: additive saving in [3%, 10%] — the methods partly overlap. "
"Negative result (< 3% additive, or accuracy loss when combined): the two axes are redundant, bounding the complementarity claim.\n\n"

"Experiment 6 (H3 — Sensitivity and hard-item safety): Sweep tau and checkpoint layer to trace the accuracy-vs-compute curve; report accuracy on the hardest items (AIME) and calibration drift across datasets without retuning. "
"Decisive metric: AIME accuracy drop at the tau/c that deliver the Experiment-3 savings on GSM8K/MATH500 (transfer without retuning). "
"Convincing result: AIME accuracy drops < 1 point under transferred thresholds. "
"Indecision band: 1-3 point drop — per-dataset retuning is needed and reported. "
"Negative result (> 3 point drop): thresholds do not transfer and hard items are unsafe, bounding H3.\n\n"

"Experiment 7 (H2/H3 — Generality across data and scale): Transfer the depth probe and thresholds across datasets and the 1.5B/7B/14B/Qwen3-8B models without retraining. "
"Decisive metric: transferred-probe AUROC relative to in-domain, and decoder compute-savings retention. "
"Convincing result: transfer retains >= 80% of in-domain AUROC on at least three scales; savings hold within 10 relative percent. "
"Indecision band: retention in [60%, 80%] — partly model-specific; scoped to per-model probes/thresholds. "
"Negative result (< 60% retention): the shallow-token signal is idiosyncratic across models."
)

tmp = path_out + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=4, ensure_ascii=False)
os.replace(tmp, path_out)
with open(path_out, encoding="utf-8") as f:
    print(list(json.load(f).keys()))
