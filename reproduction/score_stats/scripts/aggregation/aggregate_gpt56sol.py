#!/usr/bin/env python3
"""Aggregate the gpt-5.6-sol reliability-judge JSONs into llm_scores_gpt56sol.{csv,md}.

Column layout is byte-identical to llm_scores_opus5.csv (and llm_scores.csv), so
the figure scripts consume it by swapping a single path.

Scoring rule (identical to the opus-5 / opus-4.8 tables):
  dim1-8   : 5 points each.
  dim9     : method / experiment / result are three orthogonal sub-angles; each is
             averaged over all claims first, then the three sub-angle means are
             averaged -> dim9_final (max 5).
  raw      : sum(dim1..dim8) + dim9_final
  base     : 45 minus 5 for every dim1-8 scored "n/a" (n/a is dropped from both
             numerator and denominator); dim9 always counts.
  total100 : raw / base * 100
  score 0  : counted as 0 (not n/a).

Dimension keys are matched by the `dimension_<i>_` prefix rather than by exact
name: a handful of gpt-5.6-sol outputs use paraphrased key suffixes, which the
orchestrator's validate_statistical.py already tolerates the same way.
"""
from pathlib import Path
import csv
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("REPRODUCTION_ROOT", HERE.parents[3] / "reproduction"))
JUDGE = ROOT / "llm_judge"
EXPERT = "gpt-5-6-sol"

CATOF = {
    "closing_gap_belief": "belief", "verbal_confidence": "belief",
    "llm_social_decision": "belief",
    "emotion_circuit": "emotion", "emotion_prompts": "emotion",
    "multi_modal_feature_description": "feature_description",
    "sae_agentic_explainer": "feature_description",
    "multi_agent": "multi-agent_safety",
    "lasa_safety": "multilingual", "multi_lingual_reasoning": "multilingual",
    "alignet_visual": "multimodal", "universal_steering": "multimodal",
    "propositional_logic_circuit": "reasoning",
    "thinking_reasoning_steering": "reasoning",
    "circuit_breakers": "safety", "encode_harmfulness_refusal": "safety",
    "esmfold_mechanism": "science", "interplm": "science",
}
EXPS = list(CATOF)
SCIS = ["scientist1", "scientist2", "scientist3"]


def num(v):
    """Return float score, or None when the entry is 'n/a'."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"n/a", "na", "nan", ""}:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return float(v)


def dim_entry(data: dict, i: int):
    """Fetch dimension i by `dimension_<i>_` prefix (suffix wording varies)."""
    hits = [k for k in data if k.startswith(f"dimension_{i}_")]
    if len(hits) != 1:
        return None, hits
    return data[hits[0]], hits


rows = []
missing = []
warnings = []
for exp in EXPS:
    cat = CATOF[exp]
    for sci in SCIS:
        p = JUDGE / cat / exp / f"{exp}-{sci}-{EXPERT}.json"
        if not p.exists():
            missing.append(str(p.relative_to(ROOT)))
            continue
        d = json.loads(p.read_text(encoding="utf-8"))

        d18, base = [], 0.0
        for i in range(1, 9):
            entry, hits = dim_entry(d, i)
            if entry is None:
                warnings.append(f"{p.name}: dimension_{i} keys={hits}")
            s = num(entry.get("score")) if isinstance(entry, dict) else None
            d18.append(s)
            if s is not None:
                base += 5.0
        sub8 = sum(s for s in d18 if s is not None)

        entry9, _ = dim_entry(d, 9)
        per_claim = (entry9 or {}).get("per_claim", {}) if isinstance(entry9, dict) else {}
        if isinstance(per_claim, list):  # tolerate list-shaped per_claim
            per_claim = {str(i): c for i, c in enumerate(per_claim)}
        sub = {"method": [], "experiment": [], "result": []}
        for c in per_claim.values():
            if not isinstance(c, dict):
                continue
            for k in sub:
                s = num(c.get(k, {}).get("score") if isinstance(c.get(k), dict) else None)
                if s is not None:
                    sub[k].append(s)
        means = {k: (sum(v) / len(v) if v else None) for k, v in sub.items()}
        have = [m for m in means.values() if m is not None]
        d9 = sum(have) / len(have) if have else None
        if d9 is not None:
            base += 5.0

        raw = sub8 + (d9 or 0.0)
        total = raw / base * 100 if base else 0.0
        rows.append({
            "cat": cat, "exp": exp, "sci": "s" + sci[-1],
            "n_claims": len(per_claim),
            "total100": round(total, 1), "raw": round(raw, 2), "base": base,
            **{f"d{i+1}": ("n/a" if s is None else s) for i, s in enumerate(d18)},
            "sub8": sub8,
            "d9_method": None if means["method"] is None else round(means["method"], 2),
            "d9_exp": None if means["experiment"] is None else round(means["experiment"], 2),
            "d9_result": None if means["result"] is None else round(means["result"], 2),
            "d9_final": None if d9 is None else round(d9, 2),
        })

cols = ["cat", "exp", "sci", "n_claims", "total100", "raw", "base",
        *[f"d{i}" for i in range(1, 9)], "sub8",
        "d9_method", "d9_exp", "d9_result", "d9_final"]

CSV_HEADER = ["类别", "实验", "sci", "n", "总分(0-100)", "原始总分", "满分基准",
              "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "前8小计",
              "dim9_method", "dim9_exp", "dim9_result", "dim9最终"]
CSV_SRC = ["cat", "exp", "sci", "_n1", "total100", "raw", "base",
           "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "sub8",
           "d9_method", "d9_exp", "d9_result", "d9_final"]

with (HERE / "llm_scores_gpt56sol.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(CSV_HEADER)
    for r in rows:
        w.writerow([1 if k == "_n1" else r[k] for k in CSV_SRC])

lines = [
    f"# LLM Judge 分数统计 ({EXPERT})",
    "",
    "**计分规则**：dim1–8 每维满分 5；dim9 = method/experiment/result 三个正交子维各自对所有 claim "
    "取均分后，再取三者平均。原始总分 = Σ(dim1–8) + dim9最终；总分 = 原始总分 / 满分基准 × 100。",
    "**n/a**：剔除并重新归一化（不进分子也不进分母，满分基准随之下降）。 **score=0**：按 0 计入。",
    "",
    "| 类别 | 实验 | sci | claims | 总分(0-100) | 原始总分 | 满分基准 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | 前8小计 | dim9_method | dim9_exp | dim9_result | dim9最终 |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    lines.append("| " + " | ".join(str(r[c]) for c in cols[:3]) + " | " + " | ".join(
        str(r[c]) for c in ["n_claims", "total100", "raw", "base",
                            "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "sub8",
                            "d9_method", "d9_exp", "d9_result", "d9_final"]) + " |")

if rows:
    lines += ["", "## 按 scientist 汇总", "", "| sci | n | mean 总分 | min | max |", "|---|---|---|---|---|"]
    for s in ["s1", "s2", "s3"]:
        vals = [r["total100"] for r in rows if r["sci"] == s]
        if vals:
            lines.append(f"| {s} | {len(vals)} | {sum(vals)/len(vals):.1f} | "
                         f"{min(vals):.1f} | {max(vals):.1f} |")
if missing:
    lines += ["", f"## 缺失 {len(missing)} 个判分文件", ""] + [f"- {m}" for m in missing]
if warnings:
    lines += ["", f"## {len(warnings)} 处键名异常", ""] + [f"- {w}" for w in warnings]

(HERE / "llm_scores_gpt56sol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"rows={len(rows)} missing={len(missing)} warnings={len(warnings)}")
for w in warnings:
    print("  !", w)
for s in ["s1", "s2", "s3"]:
    vals = [r["total100"] for r in rows if r["sci"] == s]
    if vals:
        print(f"  {s}: n={len(vals)} mean={sum(vals)/len(vals):.1f}")
print("wrote", HERE / "llm_scores_gpt56sol.csv", "and", HERE / "llm_scores_gpt56sol.md")
