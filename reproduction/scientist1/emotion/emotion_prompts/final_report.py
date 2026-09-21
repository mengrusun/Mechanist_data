"""Aggregate all results into a final report addressing each claim from task.md."""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


RESULTS_DIR = "/data/zhenqian/Reproduction1/cc/emotion/emotion_prompts/results"
PATTERN = re.compile(r"^(?P<dataset>[^_]+)__(?P<model>.+)__(?P<cond>[^.]+)\.jsonl$")

EMOTIONS = ["happiness", "sadness", "fear", "anger", "disgust", "surprise"]
INTENSITIES = ["low", "high"]
SOURCES = ["human", "llm"]


def load_results():
    grouped = defaultdict(dict)
    per_id = defaultdict(lambda: defaultdict(dict))  # (ds, model) -> {id: {cond: correct}}
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.jsonl"))):
        m = PATTERN.match(os.path.basename(p))
        if not m:
            continue
        ds = m.group("dataset"); model = m.group("model"); cond = m.group("cond")
        rows = [json.loads(ln) for ln in open(p)]
        if not rows:
            continue
        acc = float(np.mean([r["correct"] for r in rows]))
        grouped[(ds, model)][cond] = acc
        for r in rows:
            per_id[(ds, model)][r["id"]][cond] = bool(r["correct"])
    return grouped, per_id


def _emotion_parts(cond):
    """cond like 'human_anger_high' or 'neutral'."""
    if cond == "neutral":
        return None, None, None
    parts = cond.split("_")
    return parts[0], parts[1], parts[2]  # source, emotion, intensity


def claim1_static_effect(grouped, out):
    """Static emotional prefixes change accuracy only by small, input-dependent amounts."""
    out.append("\n### Claim 1: Static emotional prefixes only produce small, input-dependent changes\n")
    out.append(f"{'dataset':<12} {'model':<14} {'neutral':>8} {'best':>8} {'worst':>8} "
               f"{'best_delta':>11} {'worst_delta':>12} {'range':>8}")
    for (ds, model), accs in sorted(grouped.items()):
        neutral = accs.get("neutral")
        if neutral is None:
            continue
        emo_accs = [a for c, a in accs.items() if c != "neutral"]
        best = max(emo_accs); worst = min(emo_accs)
        out.append(f"{ds:<12} {model:<14} {neutral:>8.4f} {best:>8.4f} {worst:>8.4f} "
                   f"{best-neutral:>+11.4f} {worst-neutral:>+12.4f} {best-worst:>8.4f}")


def claim2_task_dependence(grouped, out):
    """Effect is most pronounced on socially grounded tasks; smaller on math/factual."""
    out.append("\n### Claim 2: Effect is largest on social tasks, smaller on math/factual\n")
    out.append(f"{'dataset':<12} {'neutral':>8} {'range':>8} {'mean_delta':>11} {'best_delta':>11}")
    for (ds, model), accs in sorted(grouped.items()):
        neutral = accs.get("neutral")
        if neutral is None:
            continue
        emo_accs = [a for c, a in accs.items() if c != "neutral"]
        deltas = [a - neutral for a in emo_accs]
        out.append(f"{ds:<12} {neutral:>8.4f} {max(emo_accs)-min(emo_accs):>8.4f} "
                   f"{np.mean(deltas):>+11.4f} {max(deltas):>+11.4f}")


def claim3_no_single_emotion(grouped, out):
    """No single basic emotion consistently benefits; stronger != larger gains."""
    out.append("\n### Claim 3: No single emotion is a consistent winner\n")
    # Best emotion across (dataset, model) combos
    winners = defaultdict(int)
    per_emotion_deltas = defaultdict(list)
    for (ds, model), accs in grouped.items():
        neutral = accs.get("neutral")
        if neutral is None:
            continue
        # collapse: max across (source, intensity) for each emotion
        per_emo = defaultdict(list)
        for cond, acc in accs.items():
            if cond == "neutral":
                continue
            s, e, i = _emotion_parts(cond)
            per_emo[e].append(acc - neutral)
        for e, vals in per_emo.items():
            per_emotion_deltas[e].extend(vals)
        best_emo = max(per_emo, key=lambda e: max(per_emo[e]))
        winners[best_emo] += 1
    out.append("Emotion best on (dataset, model) combos:")
    for e in EMOTIONS:
        out.append(f"  {e:<10}: winner_count={winners.get(e, 0)}  "
                   f"mean_delta={np.mean(per_emotion_deltas[e]):+.4f}  "
                   f"std={np.std(per_emotion_deltas[e]):.4f}")
    # Intensity effect
    out.append("\nEffect of high vs low intensity (across all datasets, models, emotions, sources):")
    low_deltas = []
    high_deltas = []
    for (ds, model), accs in grouped.items():
        neutral = accs.get("neutral")
        if neutral is None:
            continue
        for cond, acc in accs.items():
            if cond == "neutral":
                continue
            s, e, i = _emotion_parts(cond)
            if i == "low":
                low_deltas.append(acc - neutral)
            else:
                high_deltas.append(acc - neutral)
    out.append(f"  low intensity  mean_delta={np.mean(low_deltas):+.4f}  std={np.std(low_deltas):.4f}")
    out.append(f"  high intensity mean_delta={np.mean(high_deltas):+.4f}  std={np.std(high_deltas):.4f}")

    # Human vs LLM source
    out.append("\nHuman-written vs LLM-generated prefixes:")
    src_deltas = defaultdict(list)
    for (ds, model), accs in grouped.items():
        neutral = accs.get("neutral")
        if neutral is None:
            continue
        for cond, acc in accs.items():
            if cond == "neutral":
                continue
            s, e, i = _emotion_parts(cond)
            src_deltas[s].append(acc - neutral)
    for s in SOURCES:
        out.append(f"  {s:<6}: mean_delta={np.mean(src_deltas[s]):+.4f}  "
                   f"std={np.std(src_deltas[s]):.4f}  n={len(src_deltas[s])}")


def claim4_adaptive(per_id, grouped, out):
    """EmotionRL: adaptive per-query prefix vs any fixed."""
    from emotion_rl import emotion_rl
    out.append("\n### Claim 4: Per-query adaptive policy (EmotionRL) vs fixed prefixes\n")
    out.append(f"{'dataset':<12} {'model':<14} {'neutral':>8} {'best_fixed':>10} "
               f"{'oracle':>7} {'emotionRL':>9} {'gain_neutral':>13} {'gain_bfixed':>12}")
    for (ds, model), _ in sorted(grouped.items()):
        try:
            res = emotion_rl(RESULTS_DIR, ds, model, seed=0)
        except Exception as e:
            out.append(f"{ds:<12} {model:<14} ERROR: {e}")
            continue
        if not res:
            continue
        out.append(f"{ds:<12} {model:<14} {res['neutral']:>8.4f} "
                   f"{res['best_fixed_acc']:>10.4f} {res['oracle']:>7.4f} "
                   f"{res['emotion_rl']:>9.4f} {res['gain_over_neutral']:>+13.4f} "
                   f"{res['gain_over_best_fixed']:>+12.4f}")


def per_dataset_full_ranking(grouped, out):
    """Show every condition per dataset for transparency."""
    out.append("\n\n## Appendix: full per-dataset condition rankings\n")
    for (ds, model), accs in sorted(grouped.items()):
        out.append(f"\n### {ds} / {model}")
        neutral = accs.get("neutral", 0)
        ranked = sorted(accs.items(), key=lambda kv: -kv[1])
        for cond, acc in ranked:
            marker = "  <- neutral" if cond == "neutral" else ""
            out.append(f"  {cond:<30} acc={acc:.4f}  delta={acc-neutral:+.4f}{marker}")


def main():
    grouped, per_id = load_results()
    out = ["# EmotionPrompts Research Report\n"]
    out.append(f"Results directory: {RESULTS_DIR}\n")
    out.append(f"Number of (dataset, model) combos: {len(grouped)}\n")

    claim1_static_effect(grouped, out)
    claim2_task_dependence(grouped, out)
    claim3_no_single_emotion(grouped, out)
    claim4_adaptive(per_id, grouped, out)
    per_dataset_full_ranking(grouped, out)

    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
        f.write(txt)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
