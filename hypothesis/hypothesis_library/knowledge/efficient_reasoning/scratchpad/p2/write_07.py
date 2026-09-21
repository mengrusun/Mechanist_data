import json, os
path_in = "scratchpad/p2/in/cand_07.json"
path_out = "scratchpad/p2/out/cand_07.json"
with open(path_in, encoding="utf-8") as f:
    d = json.load(f)

d["Experiments"] = (
"Compute fits an academic node (<= 8xA100-80GB). "
"Primary model: DeepSeek-R1-Distill-Qwen-7B, frozen; only linear probes are trainable, interventions are activation projections. "
"Scale sweep: R1-Qwen distills 1.5B/7B/14B and Qwen3-8B. "
"Decoding: temperature 0.6, top-p 0.95, max 16k thinking tokens, seeds 0/1/2 on every reported number. "
"CoT segmented into steps; analysis restricted to extractable-answer items (math / multiple-choice). "
"Datasets, used as released: GSM8K test (1319), MATH500 (500), AIME 2024+2025 (60), GPQA-Diamond (198); a held-out 300-problem set (disjoint from evaluation) is used to build the subspaces.\n\n"

"Experiment 1 (H1 — Constructing the narration subspace): Collect contrastive activation pairs by prompting the model to render the SAME worked computation tersely vs verbosely on the held-out set; compute per-layer difference-of-means vectors and assemble a low-rank narration subspace by SVD. Build an alternative narration signal from easy-vs-hard (more vs less performative) problems. "
"Decisive metric: variance explained and rank of the narration subspace, and agreement (principal-angle alignment) between the terse/verbose subspace and the easy/hard subspace. "
"Convincing result: a low-rank (<= 10-dim) narration subspace captures the terse/verbose contrast and aligns with the easy/hard signal (principal angles < 45 degrees on the top directions). "
"Indecision band: alignment with top principal angles in [45, 60] degrees — the two narration signals only partly agree; we report the construction as prompt-dependent. "
"Negative result (no low-rank structure or > 60 degree misalignment): a stable narration subspace cannot be constructed, undercutting the framing.\n\n"

"Experiment 2 (H1 — Separability from computation): Train a linear latent-answer probe (predicting the eventual final answer) as the computation readout on 40% of held-out problems; evaluate on the remaining 60%. Then compare probe accuracy under (a) projecting out the narration subspace, (b) projecting out a matched-rank random subspace, (c) projecting out the computation subspace (from the answer-probe weights). Report principal angles between narration and computation subspaces. "
"Decisive metric: latent-answer probe accuracy drop under narration-subspace removal vs. under computation-subspace removal. "
"Convincing result: narration removal drops probe accuracy by < 3 points (comparable to random-subspace removal) while computation removal drops it by > 20 points, with narration/computation principal angles > 60 degrees (near-orthogonal). "
"Indecision band: narration-removal drop in [3, 8] points, or principal angles in [45, 60] degrees — the subspaces overlap partially; suppression will carry some accuracy risk and we report it. "
"Negative result (narration removal drops probe > 8 points, or subspaces are not near-orthogonal): the subspaces are not separable, refuting H1 (an informative null).\n\n"

"Experiment 3 (H2 — Narration stays high after computation completes): Read narration-subspace activation magnitude at each step; measure residual work via truncate-and-resume (accuracy after forcing termination at step t). "
"Decisive metric: out-of-sample R^2 (5-fold) of the post-completion narration-activation integral (activation after residual work hits zero) predicting wasted tokens, minus best competitor R^2 among trace length, absolute verbosity-direction projection, and a confidence probe. "
"Convincing result: narration activation stays significantly elevated after residual work hits zero, and its post-completion integral beats the best competitor by >= 0.10 R^2 on at least three datasets. "
"Indecision band: R^2 advantage in [0.02, 0.10] — narration activation is a useful but not dominant predictor. "
"Negative result (advantage <= 0.02 or no sustained post-completion activation): overthinking is not specifically sustained narration, weakening H2.\n\n"

"Experiment 4 (H3 — Narration projection saves tokens, more precisely than a single direction): During generation, subtract the projection onto the narration subspace at selected layers (coefficient sweep). "
"Coefficient and layer set selected on the held-out construction problems. "
"Systems compared on all four datasets: narration-subspace projection vs. (a) no intervention, (b) single verbosity/length steering (ASC/Manifold-style), (c) TTS-style post-hoc step pruning, (d) Reasoning-Theater-style probe early-exit, (e) fixed truncation matched to mean budget. "
"Decisive metric: token reduction at equal accuracy — percent reduction vs. no intervention while accuracy stays within 1 point; the key comparison is whether narration projection Pareto-dominates the single verbosity direction. "
"Secondary: mean tokens, accuracy, wall-clock speedup. "
"Convincing result: >= 25% token reduction within 1 accuracy point on GSM8K and MATH500, strictly above the single-direction baseline's accuracy-vs-token frontier (>= 5% more reduction at matched accuracy). "
"Indecision band: reduction in [10%, 25%] within 1 point, or within 5% of the single verbosity direction at matched accuracy — the dimensional method works but is not clearly finer-grained; the separability analysis (Experiments 1-2) is then the standalone contribution. "
"Negative result (< 10% reduction, or worse than the single direction): the subspace method adds nothing over prior steering.\n\n"

"Experiment 5 (H3 — Qualitative: shorthand, not damage): Inspect generations under narration suppression; an LLM-judge plus a human spot-check (200 traces, two annotators) rate whether derivation steps are preserved while decorative narration/verification asides are dropped. "
"Decisive metric: fraction of suppressed traces judged to preserve all load-bearing derivation steps. "
"Convincing result: >= 85% of suppressed traces preserve load-bearing steps while dropping narration, with inter-annotator agreement kappa > 0.6. "
"Indecision band: preservation in [70%, 85%] — suppression sometimes removes substantive content. "
"Negative result (< 70% preserve load-bearing steps): suppression damages computation, contradicting the shorthand claim.\n\n"

"Experiment 6 (H3 — Underthinking analysis): On the hardest quartile of items where narration may overlap genuine verification, measure accuracy drop vs. suppression strength and identify a safe operating point. "
"Decisive metric: hard-item accuracy drop at the deployed suppression coefficient. "
"Convincing result: hard-item accuracy drops < 1 point at the coefficient delivering the Experiment-4 savings. "
"Indecision band: 1-3 point drop — a safety/savings tradeoff; report the curve. "
"Negative result (> 3 point drop): narration suppression harms genuine verification, bounding H3.\n\n"

"Experiment 7 (H1/H3 — Generality across data and scale): Transfer the narration subspace across datasets and the 1.5B/7B/14B/Qwen3-8B models without recomputation. "
"Decisive metric: separability (Experiment-2 probe-drop asymmetry) and token savings under a transferred subspace, relative to an in-domain subspace. "
"Convincing result: transferred subspace retains the separability asymmetry and >= 80% of in-domain token savings on at least three scales. "
"Indecision band: 60-80% retention — partly model-specific; scoped to per-model subspaces. "
"Negative result (< 60% retention): the narration subspace is idiosyncratic across models."
)

tmp = path_out + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=4, ensure_ascii=False)
os.replace(tmp, path_out)
with open(path_out, encoding="utf-8") as f:
    print(list(json.load(f).keys()))
