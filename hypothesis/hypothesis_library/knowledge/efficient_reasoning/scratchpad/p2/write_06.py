import json, os
path_in = "scratchpad/p2/in/cand_06.json"
path_out = "scratchpad/p2/out/cand_06.json"
with open(path_in, encoding="utf-8") as f:
    d = json.load(f)

d["Experiments"] = (
"Compute fits an academic node (<= 8xA100-80GB). "
"Primary model: DeepSeek-R1-Distill-Qwen-7B, frozen; only linear probes are trainable, interventions are activation edits. "
"Scale sweep: R1-Qwen distills 1.5B/7B/14B and Qwen3-8B (smaller k for the 14B model to control cost). "
"Sampling: temperature 0.6, top-p 0.95, max 16k thinking tokens, seeds 0/1/2. "
"Analysis restricted to extractable-answer items (math / multiple-choice). "
"Datasets, used as released: GSM8K test (1319), MATH500 (500), AIME 2024+2025 (60), GPQA-Diamond (198); ground-truth utility (Experiment 1) is measured on a 400-item stratified subsample of GSM8K/MATH500 plus all AIME/GPQA to bound the N=16 sampling cost.\n\n"

"Experiment 1 (H1 — Branch-worthiness ground truth exists and budget is wasted under uniform-N): For each problem draw N=16 independent full samples; define parallel-utility = accuracy(self-consistency at N=16) minus accuracy(greedy N=1), and answer-diversity = entropy of the final-answer distribution over samples. Label single-basin (near-zero diversity, no utility) vs multi-path. "
"Decisive metric: fraction of problems with parallel-utility <= 1 point (single-basin), i.e., budget wasted under uniform N=16. "
"Convincing result: >= 40% of problems are single-basin, establishing large avoidable waste and a non-degenerate multi-path minority (>= 15%). "
"Indecision band: single-basin fraction in [20%, 40%] — waste is modest; the allocator's headroom is limited and we say so. "
"Negative result (< 20% single-basin, or < 5% multi-path): there is little to allocate over, undercutting the motivation.\n\n"

"Experiment 2 (H2 — Internal candidate-distribution entropy from a short prefix predicts branch-worthiness): Generate only a short greedy prefix (test lengths 32/64/128 tokens); at middle layers decode the internal candidate-answer distribution via (a) logit-lens at the last prefix position and (b) a trained linear probe predicting the entropy of eventual sampled answers, giving internal-entropy. "
"Probe data: 40% of problems train, 60% held out, split so no problem is shared. "
"Decisive metric: held-out rank correlation (Spearman) and AUROC of short-prefix internal-entropy vs. ground-truth parallel-utility / single-basin label. "
"Convincing result: AUROC >= 0.75 for the single-basin vs multi-path label at prefix length <= 128 tokens, with Spearman > 0.4 against parallel-utility. "
"Indecision band: AUROC in [0.62, 0.75] — the prefix signal is weak; branch structure may emerge only later and we report that. "
"Negative result (AUROC < 0.62 at all tested prefix lengths): branch-worthiness is not readable from an early prefix, refuting H1/H2.\n\n"

"Experiment 3 (H1 — The cheap internal signal matches costlier predictors): Predict parallel-utility from short-prefix internal-entropy vs. (i) prompt length, (ii) a pre-generation difficulty probe, (iii) verbal/output confidence after a full greedy trace, (iv) a T2-style black-box latent predictor. "
"Decisive metric: ranking AUROC (and regression R^2) of internal-entropy minus the best baseline that needs no full trace, and the gap to full-trace baselines. "
"Convincing result: short-prefix internal-entropy is within 0.03 AUROC of the best full-trace predictor while using far less compute, and beats every no-full-trace baseline by >= 0.05 AUROC. "
"Indecision band: within [0.03, 0.08] AUROC below the best full-trace predictor and only tied with difficulty probe — the signal is not distinctly better than plain difficulty; the causal test (Experiment 5) becomes decisive for the mechanistic claim. "
"Negative result (worse than difficulty probe): internal-entropy is just a difficulty proxy, weakening the novelty.\n\n"

"Experiment 4 (H2 — Localizing the candidate-answer distribution): Use activation patching / head attribution to find middle layers/heads carrying the candidate-answer distribution; ablate them and measure degradation of the entropy readout. "
"Decisive metric: drop in Experiment-2 probe AUROC under top-component ablation vs. equal-count random ablation. "
"Convincing result: top-component ablation drops entropy-readout AUROC by >= 0.1 (vs. < 0.03 random), localizing the signal to a small component set. "
"Indecision band: drop in [0.03, 0.1] — partially localized. "
"Negative result (< 0.03): the signal is not localized to identifiable components.\n\n"

"Experiment 5 (H2 — Causal: entropy reflects branch structure): Steer the identified components to raise or lower internal-entropy on the prefix, then run N=16 sampling and measure realized answer-diversity, checking sample validity. "
"Decisive metric: change in realized answer-diversity when steering entropy up vs. down (paired per problem). "
"Convincing result: raising internal-entropy significantly increases realized diversity and lowering it collapses samples toward one answer (paired p < 0.01, monotone in steering magnitude), with < 5% invalid/degenerate samples. "
"Indecision band: a significant but small effect (diversity change < 0.2 in normalized entropy units) or > 10% degenerate samples — the signal correlates with but may not cleanly cause branch structure. "
"Negative result (no significant change or mostly degenerate samples): the entropy signal is a spurious correlate, not the mechanistic origin of branch-worthiness.\n\n"

"Experiment 6 (H3 — Adaptive allocator saves compute at fixed accuracy): For each problem, generate one short prefix, read internal-entropy, and set N via a monotone schedule (single-basin -> N=1; high entropy -> N=16). Schedule thresholds fit on the Experiment-2 training split only. "
"Systems compared on all four datasets: allocator vs. (a) uniform-N self-consistency, (b) First Finish Search, (c) confidence-based allocation, (d) T2-style predictor. "
"Decisive metric: compute (mean samples/tokens) at fixed accuracy — samples needed to match uniform-N=16 accuracy within 1 point. "
"Secondary: full Pareto accuracy-vs-compute curve, accuracy at matched budget. "
"Convincing result: allocator matches uniform-N=16 accuracy within 1 point using >= 40% fewer samples, and dominates or ties the best baseline on the Pareto frontier. "
"Indecision band: 15-40% sample reduction, or a tie with T2 (< 5% sample difference at matched accuracy) — the allocator works but is not clearly superior; the mechanistic grounding (Experiments 2-5) is then the standalone contribution. "
"Negative result (< 15% reduction or > 1 point accuracy loss to match budgets): the signal does not yield usable allocation savings.\n\n"

"Experiment 7 (H1 — Generality across data and scale): Transfer probe and allocator across datasets and the 1.5B/7B/14B/Qwen3-8B models without retraining. "
"Decisive metric: transferred-probe AUROC relative to in-domain, and allocator sample-savings retention. "
"Convincing result: transfer retains >= 80% of in-domain AUROC on at least three scales; savings hold within 10 relative percent. "
"Indecision band: retention in [60%, 80%] — partly model-specific; scoped to per-model probes. "
"Negative result (< 60% retention): branch-worthiness readout is idiosyncratic across models."
)

tmp = path_out + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=4, ensure_ascii=False)
os.replace(tmp, path_out)
with open(path_out, encoding="utf-8") as f:
    print(list(json.load(f).keys()))
