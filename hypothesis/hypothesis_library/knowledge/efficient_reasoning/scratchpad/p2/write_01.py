import json, os
path_in = "scratchpad/p2/in/cand_01.json"
path_out = "scratchpad/p2/out/cand_01.json"
with open(path_in, encoding="utf-8") as f:
    d = json.load(f)

d["Experiments"] = (
"Compute fits an academic node (<= 8xA100-80GB). "
"Primary model: DeepSeek-R1-Distill-Qwen-7B, all weights frozen; only linear probes are trainable. "
"Scale sweep: R1-Qwen distills 1.5B/7B/14B and Qwen3-8B. "
"Decoding: temperature 0.6, top-p 0.95, max 16k thinking tokens, seeds 0/1/2 on every reported number. "
"CoT is segmented into steps by newline/step delimiters plus sentence splitting. "
"Datasets, used as released: GSM8K test (1319), MATH500 (500), AIME 2024+2025 (60), GPQA-Diamond (198). "
"Unless stated, analysis is on correctly solved traces.\n\n"

"Experiment 1 (H1 — Load-bearing steps are sparse): Quantify per-step causal influence and confirm most steps are decorative. "
"For each step, run resample-ablation: replace the step with a semantically neutral connector, and separately delete it and resume, then measure the normalized drop in final-answer log-probability. "
"A step is 'load-bearing' if the drop exceeds a fixed threshold theta, else 'decorative'; theta is fixed a priori as the drop that changes the argmax answer with probability 0.5 on a 200-step calibration subset. "
"Decisive metric: median decorative fraction (share of steps labeled decorative) per dataset. "
"Robustness: label agreement (Cohen's kappa) between the connector-replacement and deletion ablation variants. "
"Convincing result: median decorative fraction > 0.5 on all four datasets with variant kappa > 0.6. "
"Indecision band: decorative fraction in [0.35, 0.5], or kappa in [0.4, 0.6] — sparsity is weak or label-noisy; we report it as tendency, not fact, and downstream router claims are scoped accordingly. "
"Negative result (fraction < 0.35 or kappa < 0.4): steps are not predominantly decorative, or labels are unreliable, undercutting the premise.\n\n"

"Experiment 2 (H2 — Decorative status is prospectively encoded): Test whether a linear probe on the pre-step residual predicts the upcoming step's label before it is generated. "
"Probe data: 40% of labeled step-boundaries (from Experiment 1) train per-layer ridge logistic probes; 60% held out for evaluation, split by problem so no trace appears in both. "
"Systems compared: (i) prospective probe on the boundary residual, (ii) bag-of-words baseline on preceding text, (iii) a retrospective probe that sees the generated step (upper bound). "
"Decisive metric: prospective-probe AUROC for upcoming-step label on the held-out split. "
"Layer sweep localizes the strongest signal. "
"Convincing result: prospective AUROC >= 0.75 on at least three datasets and >= 0.10 above the bag-of-words baseline. "
"Indecision band: AUROC in [0.62, 0.75], or advantage over bag-of-words in [0.03, 0.10] — weak or partly lexical prospective signal; the router is unlikely to help much and we say so. "
"Negative result (AUROC < 0.62 or no advantage over bag-of-words): decorative status is not linearly pre-encoded, refuting H2 (an informative negative — the decision may form mid-step).\n\n"

"Experiment 3 (H2 — Circuit behind the prediction): Localize components that determine the probe's decorative prediction via activation patching and head attribution, then ablate the top components and test whether the model emits fewer/different decorative steps. "
"Decisive metric: change in decorative fraction (Experiment-1 labeling) under top-component ablation vs. equal-count random ablation. "
"Convincing result: top-component ablation changes decorative fraction by >= 10 absolute points while random ablation changes it by < 3, with task accuracy preserved within 3 points. "
"Indecision band: change in [3, 10] points — partial localization; a candidate circuit is reported without strong causal claims. "
"Negative result (< 3 points or accuracy collapse): the prospective signal is not tied to an identifiable, intervenable circuit.\n\n"

"Experiment 4 (H3 — Proactive router saves tokens without hurting accuracy): Use the Experiment-2 probe as a training-free generation-time router: at each boundary, if predicted decorative with confidence > tau, force continuation past the step. "
"tau is selected on the Experiment-2 training split only (grid {0.6,0.7,0.8,0.9}); reporting is on held-out problems. "
"Systems compared on all four datasets: router vs. (a) no intervention, (b) fixed truncation matched to router mean budget, (c) length-aware voting (When More is Less), (d) reactive attention pruning (Think Clearly), (e) post-hoc TTS pruning. "
"Decisive metric: token reduction at equal accuracy — percent token reduction vs. no intervention while accuracy stays within 1 point. "
"Secondary: mean tokens, accuracy, wall-clock speedup. "
"Convincing result: router achieves >= 25% token reduction within 1 accuracy point on GSM8K and MATH500 and lies on or above every baseline's accuracy-vs-token frontier. "
"Indecision band: reduction in [10%, 25%] within 1 point, or a tie with the best reactive baseline (< 5% reduction difference at matched accuracy) — the router works but is not clearly superior; the standalone contribution is then the mechanistic finding (Experiments 1-3). "
"Negative result (< 10% reduction or > 1 point accuracy loss to match baselines): the predictive signal does not convert into a usable router.\n\n"

"Experiment 5 (H3 — Failure and safety analysis): Characterize where the router harms accuracy. "
"Measure router accuracy as a function of tau, separately on the final 15-30% of the chain (reasoning horizon) and on the hardest quartile of items (by base pass-rate), and audit false-skip cases. "
"Decisive metric: accuracy drop attributable to skips in the final 30% of the chain vs. the first 70%, at the deployed tau. "
"Convincing result: no region shows > 2 points of accuracy loss beyond the aggregate at deployed tau, so a single tau is safe. "
"Indecision band: a region shows 2-5 points extra loss — a horizon-aware tau schedule is needed and reported. "
"Negative result (> 5 points concentrated loss): the router is unsafe without step-position gating, bounding H3.\n\n"

"Experiment 6 (H2/H3 — Generality across data and scale): Transfer the probe/router across datasets and the 1.5B/7B/14B/Qwen3-8B models without retraining. "
"Decisive metric: transferred-probe AUROC relative to in-domain AUROC, and router token-savings retention under transfer. "
"Convincing result: transfer retains >= 80% of in-domain AUROC across at least three scales and router savings hold within 10 relative percent. "
"Indecision band: retention in [60%, 80%] — signal is partly model-specific; claims are scoped to per-model probes. "
"Negative result (< 60% retention): prospective decorative-encoding is idiosyncratic, contradicting the 'shared low-cost signal' claim."
)

tmp = path_out + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=4, ensure_ascii=False)
os.replace(tmp, path_out)
with open(path_out, encoding="utf-8") as f:
    print(list(json.load(f).keys()))
