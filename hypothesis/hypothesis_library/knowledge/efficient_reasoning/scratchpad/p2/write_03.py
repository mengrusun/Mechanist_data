import json, os
path_in = "scratchpad/p2/in/cand_03.json"
path_out = "scratchpad/p2/out/cand_03.json"
with open(path_in, encoding="utf-8") as f:
    d = json.load(f)

d["Experiments"] = (
"Compute fits an academic node (<= 8xA100-80GB). "
"Primary model: DeepSeek-R1-Distill-Qwen-7B, frozen; only linear uncertainty probes are trainable. "
"Scale sweep: R1-Qwen distills 1.5B/7B/14B and Qwen3-8B. "
"Decoding: temperature 0.6, top-p 0.95, max 16k thinking tokens, seeds 0/1/2 on every reported number. "
"CoT segmented into steps; reflection-onset tokens marked by lexical cues (wait / let me check / but) plus a reflection-direction classifier reproduced from ReflCtrl. "
"Datasets, used as released: GSM8K test (1319), MATH500 (500), AIME 2024+2025 (60), GPQA-Diamond (198).\n\n"

"Experiment 1 (H1 — Two separable uncertainty readouts): Establish that gating-uncertainty and evidence-uncertainty are distinct, decodable signals. "
"Train two linear probes to predict eventual answer correctness from the residual stream: (a) gating-uncertainty at the token immediately preceding a reflection-onset, (b) evidence-uncertainty at the end of the most recent completed step. "
"Probe data: 40% of traces train, 60% held out, split by problem. "
"Decisive metric: held-out AUROC of each probe for eventual correctness. "
"Separability check: correlation between the two probe outputs at matched positions (they must not be near-identical to support a lag account). "
"Convincing result: both probes reach AUROC >= 0.70 and their outputs correlate at r < 0.8 (distinct signals). "
"Indecision band: either probe AUROC in [0.6, 0.7], or output correlation in [0.8, 0.9] — the readouts are weak or partly redundant; the lag framing is reported tentatively. "
"Negative result (AUROC < 0.6 or correlation > 0.9): the two signals are not separable/useful, undercutting the whole framing.\n\n"

"Experiment 2 (H1 — Gating uncertainty lags evidence uncertainty): Test staleness directly. "
"For each reflection episode, compare gating-uncertainty to evidence-uncertainty at the same point, and estimate the lag as the offset k for which evidence-uncertainty at step t-k best matches gating-uncertainty at step t (cross-correlation). "
"Also test whether reflection-onset preferentially fires when gating exceeds evidence (large lag), linking to high-entropy forking tokens. "
"Decisive metric: mean signed difference (gating minus evidence) at reflection onsets during overthought traces, and the modal best-fit lag k. "
"Convincing result: gating exceeds evidence at onsets by a significant margin (paired p < 0.01, d > 0.5) with modal lag k >= 1 step, and onset firing rate is higher in high-lag windows. "
"Indecision band: 0.2 < d < 0.5, or modal lag k = 0 with a small positive difference — weak or zero-lag staleness; we report a small effect, not a lag mechanism. "
"Negative result (d < 0.2 or gating <= evidence): no staleness, refuting H1 (reflection may reflect genuine current doubt or deliberate exploration).\n\n"

"Experiment 3 (H2 — Lag predicts wasted tokens): Test that the lag explains overthinking better than absolute uncertainty or length. "
"'Wasted tokens' = tokens after the last accuracy-changing step (truncate-and-resume). "
"Compare out-of-sample regression R^2 (5-fold) of predictors of wasted tokens: (i) per-trace mean lag, (ii) absolute gating-uncertainty, (iii) trace length, (iv) internal-consistency score. "
"Decisive metric: lag-predictor R^2 minus best competitor R^2. "
"Convincing result: lag beats the best competitor by >= 0.10 on at least three datasets. "
"Indecision band: advantage in [0.02, 0.10] — lag is a useful but not dominant predictor. "
"Negative result (advantage <= 0.02): the lag does not specifically explain waste, weakening H2.\n\n"

"Experiment 4 (H2 — Localizing where the gating signal is read): Use activation patching to find layers/heads where gating-uncertainty feeds the reflection-onset decision; ablate them and measure change in reflection frequency and lag. "
"Decisive metric: change in reflection-onset rate under top-component ablation vs. equal-count random ablation. "
"Convincing result: top-component ablation changes reflection rate by >= 10 absolute points (vs. < 3 for random) while task accuracy holds within 3 points. "
"Indecision band: change in [3, 10] points — partial localization. "
"Negative result (< 3 points or accuracy collapse): the gating read is not localized/intervenable.\n\n"

"Experiment 5 (H3 — Uncertainty-refresh breaks the loop): At reflection-onset, patch the residual components carrying gating-uncertainty with the fresh evidence-uncertainty representation, then continue; test that redundant reflection is suppressed without accuracy loss. "
"Systems compared on all four datasets: refresh vs. (a) no intervention, (b) fixed truncation matched to refresh mean budget, (c) ReflCtrl/SEAL reflection steering, (d) trace-length early-exit, (e) internal-consistency early-exit. "
"Decisive metric: token reduction at equal accuracy — percent reduction vs. no intervention while accuracy stays within 1 point. "
"Secondary: reflection-count reduction, mean tokens, wall-clock speedup. "
"Convincing result: >= 20% token reduction within 1 accuracy point on GSM8K and MATH500 with reflection count cut >= 40%, on or above every baseline frontier. "
"Indecision band: reduction in [10%, 20%] within 1 point, or a tie with the best steering baseline (< 5% difference at matched accuracy) — refresh works but is not clearly superior; the lag diagnosis (Experiments 1-4) is then the standalone contribution. "
"Negative result (< 10% reduction or > 1 point accuracy loss): refresh does not yield usable savings.\n\n"

"Experiment 6 (H3 — Preserving genuine reflection): Verify refresh does not suppress reflection when evidence-uncertainty is genuinely high (guard against underthinking). "
"On the hardest quartile of items and on cases where reflection is corrective, gate refresh by an evidence-uncertainty threshold and measure accuracy. "
"Decisive metric: accuracy change on hard/corrective items with refresh vs. without, at the deployed threshold. "
"Convincing result: accuracy on hard/corrective items drops < 1 point while savings hold on easy items. "
"Indecision band: 1-3 point drop — threshold trades savings for safety; we report the curve. "
"Negative result (> 3 point drop): refresh harms genuine reflection, bounding H3.\n\n"

"Experiment 7 (H1/H2 — Generality across data and scale): Transfer both probes and the refresh trigger across datasets and the 1.5B/7B/14B/Qwen3-8B models without retraining. "
"Decisive metric: transferred-probe AUROC relative to in-domain, and refresh token-savings retention. "
"Convincing result: transfer retains >= 80% of in-domain AUROC on at least three scales and savings hold within 10 relative percent. "
"Indecision band: retention in [60%, 80%] — partly model-specific; scoped to per-model probes. "
"Negative result (< 60% retention): the lag signal is idiosyncratic across models."
)

tmp = path_out + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=4, ensure_ascii=False)
os.replace(tmp, path_out)
with open(path_out, encoding="utf-8") as f:
    print(list(json.load(f).keys()))
