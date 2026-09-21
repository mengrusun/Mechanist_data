import json, os
path_in = "scratchpad/p2/in/cand_02.json"
path_out = "scratchpad/p2/out/cand_02.json"
with open(path_in, encoding="utf-8") as f:
    d = json.load(f)

d["Experiments"] = (
"Compute fits an academic node (<= 8xA100-80GB). "
"Primary model: DeepSeek-R1-Distill-Qwen-7B, frozen; only a cheap onset probe is trainable. "
"Scale sweep: R1-Qwen distills 1.5B/7B/14B and Qwen3-8B. "
"Decoding: temperature 0.6, top-p 0.95, max 16k thinking tokens, seeds 0/1/2 on every reported number. "
"CoT segmented into steps by newline/step delimiters. "
"Datasets, used as released: GSM8K test (1319), MATH500 (500), AIME 2024+2025 (60), GPQA-Diamond (198).\n\n"

"Experiment 1 (H1 — Recompute loops exist and are costly): Build a detector for recomputed intermediate results and quantify their token cost. "
"Detector: parse value-producing statements (equations, computed numbers, named intermediates, candidate answers) and flag a statement as a recompute when an LLM-judge deems its value semantically equivalent to one produced earlier in the same trace. "
"Detector validity: two human annotators label a 300-statement sample; we report detector precision/recall against the majority label (target precision >= 0.8). "
"Decisive metric: fraction of thinking tokens attributable to recompute loops per dataset. "
"Convincing result: recompute loops account for >= 15% of thinking tokens on average across datasets, at detector precision >= 0.8. "
"Indecision band: token fraction in [7%, 15%], or detector precision in [0.65, 0.8] — the pattern is real but minor or noisily measured; we report it as a small named waste category rather than a large one. "
"Negative result (< 7% of tokens, or precision < 0.65): recompute loops are not a material or reliably measurable source of waste, undercutting the premise.\n\n"

"Experiment 2 (H1 — Distinct from generic waste): Show recompute loops are a distinct waste category, not a relabeling of decorativeness. "
"Correlate recompute-token fraction with total tokens and with 'wasted tokens' (tokens after the last accuracy-changing step, via truncate-and-resume), and compute overlap with TTS-style decorative-step labels. "
"Decisive metric: share of recompute tokens that fall on steps labeled load-bearing (non-decorative) by TTS-style labeling. "
"Convincing result: > 40% of recompute tokens sit on non-decorative steps, establishing recompute loops as partly orthogonal to decorativeness. "
"Indecision band: overlap such that 25-40% of recompute tokens are non-decorative — partial distinctness; we present recompute loops as a refinement, not a separate axis. "
"Negative result (< 25% non-decorative): recompute loops are largely a subset of decorative waste, weakening the novelty claim.\n\n"

"Experiment 3 (H2 — Retrieval-failure signature): Test whether recompute onsets show degraded attention to the earlier derivation. "
"At each detected recompute onset token, measure attention mass directed at the tokens of the original derivation; compare against matched control positions where the same result is instead referenced/reused without recomputation. "
"Decisive metric: difference in mean attention-to-original between recompute onsets and matched reuse controls (paired, across heads/layers). "
"Convincing result: recompute onsets show significantly lower attention to the earlier result (paired difference with p < 0.01 and effect size Cohen's d > 0.5), localized to a small set of heads. "
"Indecision band: 0.2 < d < 0.5 — a weak signature consistent with but not strong evidence for retrieval failure. "
"Negative result (d < 0.2 or no significant difference): recomputation is not accompanied by degraded self-attention, refuting the retrieval-failure mechanism (recomputation may be a deliberate strategy) — an informative negative.\n\n"

"Experiment 4 (H2 — Causal patching): At a recompute onset, patch in the residual-stream representation of the original result (from its derivation position) and measure whether the model reuses rather than re-derives. "
"Decisive metric: drop in probability of continuing the recomputation (next-token / next-step recompute continuation) under the patch vs. a norm-matched random patch. "
"Convincing result: the true-result patch reduces recompute-continuation probability by >= 20 absolute points, vs. < 5 for the random patch. "
"Indecision band: reduction in [8, 20] points — the representation influences but does not control recomputation. "
"Negative result (< 8 points): injecting the earlier result does not causally suppress recomputation, undercutting the mechanism and the intervention's premise.\n\n"

"Experiment 5 (H3 — Memory-injection saves tokens without hurting accuracy): Run the detector online via a cheap residual-stream onset probe (trained on Experiment-1 labels, 40% train / 60% eval split by problem); on predicted onset, inject a terse natural-language reminder of the prior result and continue. "
"Systems compared on all four datasets: memory injection vs. (a) no intervention, (b) fixed truncation matched to injection mean budget, (c) Think Clearly reactive pruning, (d) verbosity/CREST-style steering, (e) length-aware filtering. "
"Decisive metric: token reduction at equal accuracy — percent reduction vs. no intervention while accuracy stays within 1 point. "
"Secondary: recompute-loop count reduction, mean tokens, wall-clock speedup. "
"Convincing result: >= 15% token reduction within 1 accuracy point on GSM8K and MATH500 with recompute-loop count cut by >= 50%, matching or beating every baseline on the accuracy-token frontier. "
"Indecision band: reduction in [7%, 15%] within 1 point, or a tie with the best baseline (< 5% reduction difference at matched accuracy) — injection works but is not clearly superior; the diagnosis (Experiments 1-4) is then the standalone contribution. "
"Negative result (< 7% reduction or > 1 point accuracy loss): the intervention does not yield usable savings.\n\n"

"Experiment 6 (H3 — Preserving corrective re-derivation): Distinguish blind recomputation from beneficial self-correction and ensure the trigger does not suppress the latter. "
"Label a sample of recompute onsets as corrective (fixes an earlier error) vs. blind, then measure accuracy impact of suppressing each class. "
"Decisive metric: accuracy change on items containing corrective re-derivations when injection is applied vs. withheld. "
"Convincing result: with the corrective-aware trigger, accuracy on corrective-containing items drops by < 1 point while token savings on blind-recompute items are retained. "
"Indecision band: 1-3 point drop on corrective items — the trigger imperfectly separates the two; we report the tradeoff curve. "
"Negative result (> 3 point drop): blind and corrective recomputation cannot be separated cheaply, bounding H3 to easy items.\n\n"

"Experiment 7 (H1/H2 — Generality across data and scale): Transfer detector and onset probe across datasets and the 1.5B/7B/14B/Qwen3-8B models without retraining. "
"Decisive metric: transferred onset-probe AUROC relative to in-domain, and recompute-token-fraction stability across scales. "
"Convincing result: transfer retains >= 80% of in-domain AUROC on at least three scales and recompute prevalence stays within a factor of 2 across scales. "
"Indecision band: retention in [60%, 80%] — partly model-specific; scoped to per-model probes. "
"Negative result (< 60% retention or prevalence varying by more than a factor of 3): the phenomenon is idiosyncratic across models."
)

tmp = path_out + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=4, ensure_ascii=False)
os.replace(tmp, path_out)
with open(path_out, encoding="utf-8") as f:
    print(list(json.load(f).keys()))
