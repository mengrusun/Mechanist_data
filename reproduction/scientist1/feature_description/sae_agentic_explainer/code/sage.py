"""SAGE: iterative propose-test-revise pipeline for SAE feature explanation.

Roles (all played by GPT-5):
  - Explainer: proposes K candidate explanations from initial snippets.
  - Designer: for each candidate, writes probe sentences that should trigger the feature
              under that candidate's hypothesis. Also writes some negative probes to
              distinguish candidates.
  - Analyzer / Reviewer: given per-candidate probe activations (from the target LLM+SAE),
              refines / rejects / keeps candidates, may propose new ones.

Output: a ranked list of candidate explanations, plus the top one as "SAGE explanation".
"""
import json
import time
import numpy as np

from api import chat_json, chat


# ----------------------------- role prompts -----------------------------

EXPLAINER_SYSTEM = (
    "You are an interpretability researcher labeling a hidden feature of an LLM. "
    "You will see short text snippets in which the feature fires strongly, with the "
    "peak-activation token wrapped in <<double angle brackets>>. Your job is to "
    "propose SEVERAL distinct hypothesis-explanations for what pattern triggers "
    "this feature. Keep each hypothesis to one sentence, concrete and testable. "
    "If the snippets appear polysemantic (multiple distinct concepts), your "
    "hypotheses should reflect that."
)

DESIGNER_SYSTEM = (
    "You are an experiment designer for testing a hypothesis about an LLM's hidden "
    "feature. Given a HYPOTHESIS explanation, write short natural-language probe "
    "sentences that should make this feature fire IF and only if the hypothesis is "
    "correct. Also write DISTRACTOR sentences that are semantically nearby but "
    "should NOT fire under this hypothesis. Keep sentences 4-15 words."
)

REVIEWER_SYSTEM = (
    "You are an experiment reviewer. You are given several hypothesis explanations "
    "for a hidden feature, along with the measured activations they produced on their "
    "own probes and on distractor probes. Decide which hypotheses to KEEP, REVISE, "
    "REJECT, or ADD. The best hypothesis is the one whose probes actually fire "
    "AND whose distractors do NOT fire, with a large gap. Prefer specific, "
    "concrete descriptions. If multiple hypotheses have empirical support, keep "
    "them (feature may be polysemantic). Otherwise favor the strongest one."
)

# ----------------------------- Explainer -----------------------------

def _fmt_snippets(snippets, max_n=12):
    lines = []
    for i, s in enumerate(snippets[:max_n]):
        lines.append(f"  {i+1}. (max_act={s['max_val']:.2f}) {s['annot']!r}")
    return "\n".join(lines)


def explainer_propose(snippets, K=3):
    """Return list[str] of K candidate explanations."""
    prompt = (
        "Below are the top-activating snippets for a hidden feature in an LLM. The "
        "peak-activation token is wrapped in <<double angle brackets>>. Propose "
        f"exactly {K} distinct one-sentence hypothesis-explanations of what pattern "
        "triggers this feature. Reply ONLY with a JSON object of the form "
        "{\"hypotheses\": [\"...\", \"...\"]}.\n\n"
        f"Snippets:\n{_fmt_snippets(snippets)}"
    )
    d = chat_json([
        {"role": "system", "content": EXPLAINER_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    hyps = d.get("hypotheses") or []
    hyps = [h.strip() for h in hyps if h and isinstance(h, str)][:K]
    while len(hyps) < K:
        hyps.append(hyps[-1] if hyps else "(unknown)")
    return hyps


# ----------------------------- Designer -----------------------------

def designer_probes(hypothesis, n_positive=5, n_distractor=3):
    prompt = (
        f"Hypothesis about the feature: {hypothesis!r}\n\n"
        f"Write {n_positive} POSITIVE probe sentences that should make this feature "
        f"fire strongly, and {n_distractor} DISTRACTOR sentences that are related "
        "in topic or surface form but should NOT trigger the feature. Each "
        "sentence 4-15 words. Reply ONLY as JSON: "
        "{\"positive\": [\"...\", ...], \"distractor\": [\"...\", ...]}."
    )
    d = chat_json([
        {"role": "system", "content": DESIGNER_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    pos = [s.strip() for s in (d.get("positive") or []) if isinstance(s, str) and s.strip()][:n_positive]
    neg = [s.strip() for s in (d.get("distractor") or []) if isinstance(s, str) and s.strip()][:n_distractor]
    while len(pos) < 1:
        pos.append(hypothesis)
    while len(neg) < 1:
        neg.append("This is a random unrelated sentence.")
    return pos, neg


# ----------------------------- Reviewer -----------------------------

def reviewer_refine(snippets, hyp_probes_results, K=3):
    """hyp_probes_results: list of dicts {
        'hypothesis': str,
        'positive_probes': [str], 'positive_acts': [float],
        'distractor_probes': [str], 'distractor_acts': [float],
    }"""
    lines = []
    for i, r in enumerate(hyp_probes_results):
        lines.append(f"HYPOTHESIS {i+1}: {r['hypothesis']!r}")
        lines.append("  positive probes (activation on target feature):")
        for p, a in zip(r['positive_probes'], r['positive_acts']):
            lines.append(f"    - {a:6.2f}  {p!r}")
        lines.append("  distractor probes (should be low):")
        for p, a in zip(r['distractor_probes'], r['distractor_acts']):
            lines.append(f"    - {a:6.2f}  {p!r}")
        lines.append("")
    prompt = (
        "The feature's original top-activating snippets:\n"
        f"{_fmt_snippets(snippets, max_n=8)}\n\n"
        "Test results for current candidate hypotheses:\n"
        + "\n".join(lines)
        + "\n"
        "Refine the candidate set. Reply ONLY as JSON of the form:\n"
        "{\"candidates\": [{\"explanation\": \"...\", \"rationale\": \"...\"}, ...] }\n"
        f"Return exactly {K} candidates ranked best-first. You may keep, revise, replace, "
        "or add candidates. Prefer explanations whose positive probes actually fire strongly "
        "AND whose distractors stay near zero."
    )
    d = chat_json([
        {"role": "system", "content": REVIEWER_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    cands = d.get("candidates") or []
    out = []
    for c in cands[:K]:
        if isinstance(c, dict) and c.get("explanation"):
            out.append(c["explanation"].strip())
    while len(out) < K:
        out.append(out[-1] if out else "(unknown)")
    return out


# ----------------------------- probe measurement -----------------------------

def measure_probes(probes, hooked_model, sae, layer, feature_idx, max_length=64, batch_size=8):
    """Return list of max activations per probe."""
    if not probes:
        return []
    vals_all = []
    for i in range(0, len(probes), batch_size):
        batch = probes[i:i+batch_size]
        vals, _, _ = hooked_model.feature_activation_of(
            batch, sae, layer=layer, feature_idx=feature_idx,
            max_length=max_length, mode="max",
        )
        vals_all.extend(vals)
    return vals_all


# ----------------------------- SAGE main loop -----------------------------

def run_sage(snippets, hooked_model, sae, layer, feature_idx,
             K=3, T=3, n_pos=5, n_neg=3, verbose=False):
    """Return dict with keys: 'candidates' (list[str]), 'trace' (list of round dicts)."""
    # Preprocess snippets: annotate max token position for the prompt.
    from neuronpedia import annotate_snippet_max
    prep_snips = []
    for s in snippets:
        prep_snips.append({**s, "annot": annotate_snippet_max(s)})

    # Round 0: propose
    cands = explainer_propose(prep_snips, K=K)
    trace = []
    for round_i in range(T):
        results = []
        for hyp in cands:
            pos_probes, neg_probes = designer_probes(hyp, n_positive=n_pos, n_distractor=n_neg)
            pos_acts = measure_probes(pos_probes, hooked_model, sae, layer, feature_idx)
            neg_acts = measure_probes(neg_probes, hooked_model, sae, layer, feature_idx)
            results.append({
                "hypothesis": hyp,
                "positive_probes": pos_probes, "positive_acts": pos_acts,
                "distractor_probes": neg_probes, "distractor_acts": neg_acts,
            })
        # score by mean(pos) - mean(neg)
        def _score(r):
            pos = float(np.mean(r["positive_acts"])) if r["positive_acts"] else 0.0
            neg = float(np.mean(r["distractor_acts"])) if r["distractor_acts"] else 0.0
            return pos - neg
        for r in results:
            r["score"] = _score(r)
        trace.append({"round": round_i, "results": results})
        if verbose:
            print(f"[SAGE] round {round_i}: scores=",
                  [(r["hypothesis"][:60], round(r["score"],2)) for r in results])
        # revise for next round (skip on last round)
        if round_i < T - 1:
            cands = reviewer_refine(prep_snips, results, K=K)

    # final ranking: use last round's scores directly
    final_results = trace[-1]["results"]
    ranked = sorted(final_results, key=lambda r: -r["score"])
    ranked_hyps = [r["hypothesis"] for r in ranked]
    return {
        "candidates": ranked_hyps,
        "top": ranked_hyps[0],
        "trace": trace,
    }
