"""Evaluation: generative accuracy and predictive accuracy for feature explanations.

Generative accuracy:
  1. Given an explanation, ask GPT-5 to write N short sentences designed to trigger
     the feature.
  2. Run each sentence through target LLM+SAE, get max activation of the feature.
  3. Trigger rate = fraction of sentences with max activation >= trigger_threshold.
  Trigger threshold is set as 0.2 * (feature max activation observed in Neuronpedia
  top-activating snippets) — a conservative "feature clearly fires" threshold.

Predictive accuracy:
  1. Take a fixed set of held-out sentences (mixed distribution).
  2. Truth: run through target LLM+SAE, get max activation per sentence.
  3. Prediction: give sentence + explanation to GPT-5, ask for 0-10 rating of how
     likely the feature will fire.
  4. Spearman correlation between prediction and truth.
"""
import json
import numpy as np
from scipy.stats import spearmanr, pearsonr

from api import chat_json


# ------------------- generative accuracy -------------------

GEN_SYSTEM = (
    "You are helping test hidden features of a language model. Given an explanation "
    "of what a feature detects, write short sentences that should strongly activate it. "
    "Aim for the CORE of the described pattern, not vague or borderline cases."
)

def write_probe_sentences(explanation, n=8):
    prompt = (
        f"Feature explanation: {explanation!r}\n\n"
        f"Write exactly {n} short sentences (4-15 words each) that should strongly "
        "activate this feature. Vary syntactically and topically, but each must be a "
        "direct example of the described pattern. Reply ONLY as JSON: "
        "{\"sentences\": [\"...\", \"...\", ...]}."
    )
    d = chat_json([
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    sents = [s.strip() for s in (d.get("sentences") or []) if isinstance(s, str) and s.strip()]
    return sents[:n]


def generative_accuracy(explanation, hooked_model, sae, layer, feature_idx,
                        trigger_threshold, n=8, max_length=64):
    from sage import measure_probes
    sents = write_probe_sentences(explanation, n=n)
    if not sents:
        return {"trigger_rate": 0.0, "mean_max": 0.0, "sentences": [], "activations": []}
    acts = measure_probes(sents, hooked_model, sae, layer, feature_idx, max_length=max_length)
    triggered = [1.0 if a >= trigger_threshold else 0.0 for a in acts]
    return {
        "trigger_rate": float(np.mean(triggered)) if triggered else 0.0,
        "mean_max": float(np.mean(acts)) if acts else 0.0,
        "sentences": sents,
        "activations": acts,
    }


# ------------------- predictive accuracy -------------------

PRED_SYSTEM = (
    "You are estimating whether a hidden LLM feature activates on a given sentence. "
    "You are told what the feature detects and given sentences. For each sentence, "
    "rate on a 0-10 integer scale how STRONGLY the feature should fire (0=not at all, "
    "10=exactly the pattern in the peak of its distribution). Rate calibratedly."
)

def predict_activations(explanation, sentences):
    prompt = (
        f"Feature explanation: {explanation!r}\n\n"
        "Sentences to rate:\n"
        + "\n".join(f"  {i+1}. {s!r}" for i, s in enumerate(sentences))
        + "\n\nReply ONLY as JSON of the form "
        "{\"ratings\": [<int 0-10>, ...] } with the same order and length as the sentences."
    )
    d = chat_json([
        {"role": "system", "content": PRED_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    r = d.get("ratings") or []
    r = [float(x) if isinstance(x, (int, float)) else 0.0 for x in r]
    if len(r) < len(sentences):
        r += [0.0] * (len(sentences) - len(r))
    return r[:len(sentences)]


def predictive_accuracy(explanation, sentences, true_activations):
    """true_activations: list of floats. Returns (pearson_r, spearman_r, preds)."""
    preds = predict_activations(explanation, sentences)
    if len(set(preds)) < 2 or len(set(true_activations)) < 2:
        return {"pearson": 0.0, "spearman": 0.0, "preds": preds}
    pr = pearsonr(preds, true_activations)[0]
    sr = spearmanr(preds, true_activations)[0]
    return {"pearson": float(pr), "spearman": float(sr), "preds": preds}


# ------------------- held-out set builder -------------------

def build_heldout_set(feature_snippets, distractor_snippets, n_pos=6, n_neg=6, seed=0):
    """Build a held-out mixture: some real activating snippets + some random distractors."""
    rng = np.random.default_rng(seed)
    pos = list(feature_snippets)
    rng.shuffle(pos)
    pos = pos[:n_pos]
    neg = list(distractor_snippets)
    rng.shuffle(neg)
    neg = neg[:n_neg]
    all_sents = pos + neg
    rng.shuffle(all_sents)
    return all_sents
