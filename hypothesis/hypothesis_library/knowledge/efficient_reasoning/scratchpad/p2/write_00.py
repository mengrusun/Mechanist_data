import json, os

path_in = "scratchpad/p2/in/cand_00.json"
path_out = "scratchpad/p2/out/cand_00.json"
with open(path_in, encoding="utf-8") as f:
    d = json.load(f)

d["Experiments"] = (
"All experiments use open reasoning models on a single 8xA100-80GB (or smaller) node, well within an academic budget. "
"Primary model: DeepSeek-R1-Distill-Qwen-7B (transformer decoder, all weights frozen; only lightweight linear probes are trainable). "
"Scale sweep uses the 1.5B/7B/14B/32B R1-Qwen distills and Qwen3-8B. "
"Decoding: temperature 0.6, top-p 0.95, max 16k thinking tokens, 3 seeds (0,1,2) for every reported number. "
"Reasoning traces are segmented into steps by newline/step delimiters. "
"Datasets (used as released, as is): GSM8K test (1319 items), MATH500 (500), AIME 2024+2025 (60), GPQA-Diamond (198). "
"A trace is 'correctly solved' if the extracted final answer matches gold; probing and gap analysis are run on correctly solved traces only, so the question is when the answer was known, not whether.\n\n"

"Experiment 1 (H1 — Commitment precedes stopping): Test whether an answer-commitment signal stabilizes strictly before the termination token is emitted. "
"For each step we read the model's current implicit final answer two ways: (a) logit-lens projection of the last-token residual onto the vocabulary, (b) a trained linear probe. "
"Probe data: for each dataset we hold out 40% of correctly solved traces to train one probe (ridge logistic regression on the layer-L residual predicting the gold final-answer token(s)); the other 60% is the evaluation set, never seen in training. "
"The 'commitment step' is the first step after which the decoded answer equals the eventual final answer and stays unchanged to termination. "
"Systems compared: commitment-step distribution vs. termination-step distribution, per dataset. "
"Decisive metric: fraction of correctly solved traces whose commitment step is at least 15% of the trace earlier than the termination step. "
"Convincing result: this fraction exceeds 0.60 on MATH500 and GSM8K with the mean commitment-to-termination separation > 20% of trace length. "
"Indecision band: fraction in [0.45, 0.60], or mean separation in [10%, 20%] of trace length — this neither confirms nor refutes early commitment, and we would report the phenomenon as weak/dataset-dependent rather than claim it. "
"Negative result (fraction < 0.45): commitment does not generally precede stopping, and the core premise of the hypothesis fails.\n\n"

"Experiment 2 (H2 — The gap IS the overthinking): Test whether the commitment-stop gap (tokens between commitment step and </think>) accounts for wasted tokens better than length or a verbosity direction. "
"'Wasted tokens' is defined operationally by truncate-and-resume: for each trace we truncate at each step, force </think>, and take the last step after which forced-exit accuracy stops changing; wasted tokens = tokens after that step. "
"Systems compared as predictors of wasted tokens, via out-of-sample regression R^2 (5-fold): (i) commitment-stop gap, (ii) total tokens, (iii) projection onto a reproduced ASC verbosity direction, (iv) Failed-Step Fraction. "
"Decisive metric: R^2 of the gap predictor minus R^2 of the best competing single predictor. "
"Convincing result: gap R^2 exceeds the best competitor by >= 0.10 on at least three of four datasets. "
"Indecision band: advantage in [0.02, 0.10] — gap is competitive but not clearly superior; we would report it as one useful signal among several, not the explanation. "
"Negative result (advantage <= 0.02, or gap R^2 below a competitor): overthinking is not primarily this gap.\n\n"

"Experiment 3 (H2/H3 — Causal check that commitment is real): Truncate generation at the probed commitment step, force </think>, and measure accuracy retention against truncating at matched random earlier and later steps. "
"Decisive metric: accuracy at commitment-step truncation minus accuracy at same-token-budget random-step truncation. "
"Convincing result: commitment-step truncation retains within 2 accuracy points of full-generation accuracy while random earlier truncation loses > 8 points at equal budget. "
"Indecision band: retention gap vs. full generation in [2, 5] points — the probe locates an approximately-but-not-exactly sufficient point; we soften claims about commitment being a clean event. "
"Negative result (> 5 point loss): the probed commitment step is not the point at which the answer is actually decided.\n\n"

"Experiment 4 (H4 — Localizing the stop-decision circuit): Localize heads/layers that drive the termination token via activation patching and attention attribution, and test decoupling from the commitment readout. "
"Method: for correctly solved traces, patch clean-run activations into a run where </think> is delayed, ranking heads by their effect on termination-token logit; ablate the top-k heads (k in {1,4,16}) and measure the shift in termination timing. "
"Decisive metric: change in mean termination step under top-head ablation vs. random-head ablation of equal count. "
"Decoupling test: measure Experiment-1 probe accuracy on the same traces with stop heads ablated; if commitment readout is preserved while stop timing shifts, the two subprocesses are dissociable. "
"Convincing result: top-head ablation shifts mean termination by >= 15% of trace length (vs. < 3% for random ablation) while probe answer-decoding accuracy drops by < 5 absolute points. "
"Indecision band: termination shift in [5%, 15%], or probe accuracy drop in [5, 10] points — partial or entangled localization; we report a candidate circuit without the clean-decoupling claim. "
"Negative result (shift < 5% or probe collapses with stop heads): the stop decision is not localized to identifiable heads, or is inseparable from answer computation, weakening H4.\n\n"

"Experiment 5 (H5 — Training-free commitment monitor): Build an early-exit that forces </think> once the probe-predicted answer is stable and probe confidence exceeds threshold tau for k consecutive steps, and test that it saves tokens at minimal accuracy cost. "
"Hyperparameters tau and k are selected on the Experiment-1 training split only (grid tau in {0.7,0.8,0.9}, k in {2,3,5}); the 60% eval split is used for reporting. "
"Systems compared on all four datasets: commitment monitor vs. (a) full generation, (b) fixed-fraction truncation matched to the monitor's mean token budget, (c) verbosity steering (ASC), (d) RCPD reactive exit, (e) budget forcing. "
"Decisive metric: token reduction at equal accuracy — percent token reduction relative to full generation while accuracy stays within 1 point. "
"Secondary: mean tokens, accuracy, wall-clock speedup. "
"Convincing result: monitor achieves >= 30% token reduction within 1 accuracy point on GSM8K and MATH500 and beats every baseline on the token-reduction-at-equal-accuracy frontier. "
"Indecision band: token reduction in [15%, 30%] within 1 point, or a tie with the strongest baseline (difference < 5% reduction at matched accuracy) — the monitor works but is not clearly better; per the hypothesis's own framing the primary contribution then rests on the mechanistic decomposition (Experiments 1-4), not the intervention. "
"Negative result (< 15% reduction, or > 1 point accuracy loss to match baseline budgets): the commitment signal does not yield a usable efficiency mechanism.\n\n"

"Experiment 6 (H3 — Generality of the signal across scale): Test whether commitment is a shared, cheaply readable signal by transferring probes and the monitor across datasets and the 1.5B/7B/14B/32B/Qwen3-8B models. "
"Decisive metric: cross-dataset and cross-scale probe transfer accuracy (train on one dataset/scale, evaluate on another) relative to the in-domain probe. "
"Convincing result: transferred probe retains >= 80% of in-domain answer-decoding accuracy across at least three model scales, and the monitor's token savings hold within 10 relative percent under transfer. "
"Indecision band: transfer retention in [60%, 80%] — the signal exists but is partly model-specific; we scope the claim to per-model probes. "
"Negative result (< 60% retention): commitment is idiosyncratic per model/dataset, contradicting the 'shared low-cost readout' claim."
)

tmp = path_out + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=4, ensure_ascii=False)
os.replace(tmp, path_out)
# verify key order
with open(path_out, encoding="utf-8") as f:
    keys = list(json.load(f).keys())
print(keys)
