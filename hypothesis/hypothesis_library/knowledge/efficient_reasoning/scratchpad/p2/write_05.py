import json, os
path_in = "scratchpad/p2/in/cand_05.json"
path_out = "scratchpad/p2/out/cand_05.json"
with open(path_in, encoding="utf-8") as f:
    d = json.load(f)

d["Experiments"] = (
"Compute fits an academic node (<= 8xA100-80GB). "
"Primary model: DeepSeek-R1-Distill-Qwen-7B, frozen; only linear probes are trainable. "
"Scale sweep: R1-Qwen distills 1.5B/7B/14B and Qwen3-8B. "
"Decoding: temperature 0.6, top-p 0.95, max 16k thinking tokens, seeds 0/1/2 on every reported number. "
"CoT segmented into steps by newline/step delimiters; analysis restricted to items with extractable answers (math / multiple-choice). "
"Datasets, used as released: GSM8K test (1319), MATH500 (500), AIME 2024+2025 (60), GPQA-Diamond (198).\n\n"

"Experiment 1 (H2 — Step-wise latent answer is decodable): Decode the current implicit final answer at every step boundary via (a) logit-lens at the last position and (b) a trained linear probe on the residual stream predicting the eventual answer token(s). "
"Probe data: 40% of traces train, 60% held out, split by problem. "
"Decisive metric: held-out probe accuracy for the eventual answer, and agreement between logit-lens and probe trajectories. "
"Convincing result: probe accuracy >= 0.70 for the eventual answer by mid-trace and logit-lens/probe trajectory agreement (fraction of steps with matching decoded answer) >= 0.7. "
"Indecision band: probe accuracy in [0.55, 0.70] or agreement in [0.5, 0.7] — trajectories are noisy; oscillation claims are reported tentatively. "
"Negative result (accuracy < 0.55 or agreement < 0.5): the latent answer trajectory is not reliably decodable, undercutting the dynamical analysis.\n\n"

"Experiment 2 (H1 — Trajectories are oscillatory, among few candidates): Characterize each trace's answer trajectory: number of distinct candidates, total switch count, dominant switch-pair identity, and a damping coefficient fit to inter-switch spacing/amplitude. "
"Report distributions per dataset; test concentration among few competing candidates. "
"Decisive metric: fraction of traces with >= 2 answer switches, and, among those, the share of switches confined to the top-2 candidate answers. "
"Convincing result: >= 40% of correctly solved traces show >= 2 switches, with >= 70% of switches confined to 2-3 candidates (structured oscillation, not diffuse drift). "
"Indecision band: 20-40% of traces with >= 2 switches — oscillation exists but is uncommon; settling-time will be a weak lever and we say so. "
"Negative result (< 20% oscillating, or switches diffuse across many candidates): overthinking is not characteristically oscillatory, refuting H1.\n\n"

"Experiment 3 (H1 — Settling time predicts waste): Define settling time = tokens from first commitment to the final flip. Measure wasted tokens via truncate-and-resume (tokens after the last accuracy-changing step). "
"Decisive metric: out-of-sample R^2 (5-fold) of settling time predicting wasted tokens minus best competitor R^2 among one-shot stability point, absolute confidence probe, and trace length. "
"Convincing result: settling time beats the best competitor by >= 0.10 R^2 on at least three datasets. "
"Indecision band: advantage in [0.02, 0.10] — settling time is useful but not dominant. "
"Negative result (advantage <= 0.02): settling time does not specifically explain waste, weakening H1.\n\n"

"Experiment 4 (H3 — One-shot stability is dangerous, motivating settling): Quantify how often the answer is stable for k steps and then flips. "
"Decisive metric: expected accuracy loss of a naive k-step one-shot stability exit (fraction of traces where the answer flips to the correct value after a k-step stable window), for k in {2,3,5}. "
"Convincing result: naive one-shot exit at k=3 would forfeit >= 5 accuracy points on at least one hard dataset (AIME/GPQA), establishing the failure mode the settling detector targets. "
"Indecision band: forfeit in [2, 5] points — the danger is modest; the settling detector's advantage over one-shot exit is expected to be small. "
"Negative result (< 2 points): one-shot stability is already safe, so the flip-aware criterion adds little (an informative negative for H3's motivation).\n\n"

"Experiment 5 (H1 — Switching is head-mediated): Use activation patching / head attribution at switch onsets to find heads driving candidate switching; ablate them and measure change in switch count and settling time. "
"Decisive metric: change in mean switch count under top-head ablation vs. equal-count random ablation. "
"Convincing result: top-head ablation changes switch count by >= 30% (vs. < 10% random) while accuracy holds within 3 points. "
"Indecision band: change in [10%, 30%] — partial head-mediation; a candidate circuit is reported without strong claims. "
"Negative result (< 10%): switching is not cleanly head-mediated; the dynamical characterization stands but the circuit claim does not.\n\n"

"Experiment 6 (H3 — Settling detector saves tokens without missing beneficial flips): Online, decode step-wise answer; a lightweight residual-stream probe (trained on Experiment-2 trajectories, 40/60 split by problem) predicts P(future flip); early-exit only when the answer is stable for k steps AND predicted flip probability < tau. "
"k and tau selected on the training split (grid k in {2,3,5}, tau in {0.1,0.2,0.3}). "
"Systems compared on all four datasets: settling detector vs. (a) no intervention, (b) one-shot k-step stability exit, (c) confidence early-exit, (d) verbosity/manifold steering, (e) fixed truncation matched to detector mean budget. "
"Decisive metric: token reduction at equal accuracy — percent reduction vs. no intervention while accuracy stays within 1 point; the detector must specifically beat the one-shot stability exit on this metric. "
"Secondary: missed-flip rate (exits before a beneficial correction), mean tokens, wall-clock speedup. "
"Convincing result: >= 20% token reduction within 1 accuracy point on GSM8K and MATH500, with missed-flip rate at least 30% lower than one-shot exit at matched tokens. "
"Indecision band: reduction in [10%, 20%] within 1 point, or missed-flip rate within 10% of one-shot exit — the flip-aware criterion adds little over one-shot stability; the dynamical characterization (Experiments 1-5) is then the standalone contribution. "
"Negative result (< 10% reduction, or no improvement in missed-flip rate over one-shot exit): the settling detector does not deliver a usable, flip-aware advantage.\n\n"

"Experiment 7 (H2/H3 — Generality across data and scale): Transfer probes and the settling detector across datasets and the 1.5B/7B/14B/Qwen3-8B models without retraining. "
"Decisive metric: transferred-probe accuracy relative to in-domain, and detector token-savings retention. "
"Convincing result: transfer retains >= 80% of in-domain probe accuracy on at least three scales; savings hold within 10 relative percent. "
"Indecision band: retention in [60%, 80%] — partly model-specific; scoped to per-model probes. "
"Negative result (< 60% retention): the oscillation signal is idiosyncratic across models."
)

tmp = path_out + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=4, ensure_ascii=False)
os.replace(tmp, path_out)
with open(path_out, encoding="utf-8") as f:
    print(list(json.load(f).keys()))
