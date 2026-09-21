import json, os
path_in = "scratchpad/p2/in/cand_08.json"
path_out = "scratchpad/p2/out/cand_08.json"
with open(path_in, encoding="utf-8") as f:
    d = json.load(f)

d["Experiments"] = (
"Compute fits an academic node (<= 8xA100-80GB). "
"Primary model: DeepSeek-R1-Distill-Qwen-7B, frozen; only linear probes are trainable, interventions are activation edits / prompt-budget controls. "
"Scale sweep: R1-Qwen distills 1.5B/7B/14B and Qwen3-8B. "
"Decoding: temperature 0.6, top-p 0.95, max 16k thinking tokens, seeds 0/1/2 on every reported number. "
"CoT segmented into sentence-level steps. "
"Datasets, used as released: GSM8K test (1319), MATH500 (500), AIME 2024+2025 (60), GPQA-Diamond (198); resampling-based importance labels (Experiment 1) computed on a 400-item stratified subsample of GSM8K/MATH500 plus all AIME/GPQA to bound cost.\n\n"

"Experiment 1 (H1 — Ground-truth anchor sparsity): Compute per-step counterfactual importance via resampling (Thought-Anchors style: sample semantically-different replacement steps, continue, measure the shift in the final-answer distribution) and a second TTS-style corruption score. Threshold to label steps anchor vs negligible; 8 resamples/step. "
"Decisive metric: median anchor fraction per dataset, and label agreement (kappa) between the resampling and corruption schemes. "
"Convincing result: median anchor fraction < 0.35 (confirming sparsity) with scheme kappa > 0.6. "
"Indecision band: anchor fraction in [0.35, 0.5] or kappa in [0.4, 0.6] — anchors are less sparse or labels noisy; downstream claims scoped accordingly. "
"Negative result (fraction > 0.5 or kappa < 0.4): steps are not sparsely load-bearing, or labels unreliable, undercutting the premise.\n\n"

"Experiment 2 (H1 — Pre-emission importance is decodable): At each step boundary, extract residual activations BEFORE the step is generated (test the boundary and the first few step tokens) and train per-layer linear probes to predict the ground-truth importance label. "
"Probe data: 40% of steps train, 60% held out, split by problem. "
"Systems compared: pre-emission probe vs. (a) bag-of-words on prior text, (b) retrospective probe that sees the generated step (upper bound), (c) post-hoc attention-received score, (d) EpiKV-style representation-change score. "
"Decisive metric: held-out AUROC of the pre-emission probe for anchor vs negligible. "
"Convincing result: pre-emission AUROC >= 0.72, beating bag-of-words by >= 0.08 and within 0.05 of the best post-hoc score that requires the step to exist. "
"Indecision band: AUROC in [0.6, 0.72], or no clear advantage over bag-of-words — the signal is weak or largely lexical; the allocator is unlikely to beat post-hoc methods and we say so. "
"Negative result (AUROC < 0.6): importance is not pre-encoded at the boundary, refuting H1 (an informative negative — importance may form mid-step).\n\n"

"Experiment 3 (H2 — The signal is causal): Steer along the probe direction (up/down) at a step boundary and measure whether the emitted step's realized importance (resampling score) and length change, vs. norm-matched random/control directions; localize contributing components via activation patching. "
"Decisive metric: change in realized step importance when steering up vs. down (paired), relative to control-direction steering. "
"Convincing result: steering up significantly raises realized importance and length while steering down lowers them (paired p < 0.01, monotone), both clearly above control directions, with task accuracy held within 2 points. "
"Indecision band: a significant but small effect (realized-importance change < 0.1 in normalized units) — the signal correlates with but does not clearly cause importance. "
"Negative result (no significant change vs. control): the probe reads a correlate, not a causal importance signal, weakening H2.\n\n"

"Experiment 4 (H3 — Attention-free allocator beats attention-based pruning): At each step boundary read the pre-emission importance; predicted-anchor -> full generation, predicted-negligible -> compressed budget (terseness instruction / step-length cap / skip-and-continue). "
"Threshold selected on the Experiment-2 training split. "
"Systems compared on all four datasets: allocator vs. (a) no intervention, (b) uniform terseness, (c) fixed truncation matched to allocator budget, (d) FROST/DynTS-style attention pruning, (e) EpiKV/LazyEviction reactive eviction. "
"Decisive metric: token reduction at equal accuracy — percent reduction vs. no intervention while accuracy stays within 1 point. "
"Secondary: mean tokens, accuracy, wall-clock speedup, and confirmation the allocator never materializes the full attention matrix (a deployability advantage over attention-based baselines). "
"Convincing result: >= 25% token reduction within 1 accuracy point on GSM8K and MATH500, matching or beating attention-based baselines on the accuracy-token frontier while avoiding attention-matrix materialization. "
"Indecision band: reduction in [10%, 25%] within 1 point, or a tie with attention-based pruning (< 5% difference at matched accuracy) — the allocator is competitive but not clearly better on savings; the prospective-signal finding (Experiments 1-3) plus its cheaper compute profile remain the contribution. "
"Negative result (< 10% reduction or > 1 point accuracy loss): the prospective signal does not yield a usable allocator.\n\n"

"Experiment 5 (H3 — Faithfulness / anchor preservation): Verify true anchors are preserved and quantify the cost of mis-compressing one. "
"Decisive metric: recall of true anchors (fraction routed to full-detail budget) at the deployed threshold, and accuracy drop attributable to compressing a true anchor. "
"Convincing result: anchor recall >= 0.9 at the deployed threshold with < 1 point accuracy loss from anchor mis-compression. "
"Indecision band: anchor recall in [0.75, 0.9] — some anchors are compressed; report the recall/savings tradeoff. "
"Negative result (recall < 0.75 or > 3 point loss): the allocator cannot protect anchors, bounding H3.\n\n"

"Experiment 6 (H1/H3 — Generality across data and scale): Transfer probe and allocator across datasets and the 1.5B/7B/14B/Qwen3-8B models without retraining. "
"Decisive metric: transferred-probe AUROC relative to in-domain, and allocator token-savings retention. "
"Convincing result: transfer retains >= 80% of in-domain AUROC on at least three scales; savings hold within 10 relative percent. "
"Indecision band: retention in [60%, 80%] — partly model-specific; scoped to per-model probes. "
"Negative result (< 60% retention): the pre-emission signal is idiosyncratic across models."
)

tmp = path_out + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=4, ensure_ascii=False)
os.replace(tmp, path_out)
with open(path_out, encoding="utf-8") as f:
    print(list(json.load(f).keys()))
