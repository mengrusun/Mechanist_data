"""Neuronpedia client: fetch feature explanation + top-activating snippets."""
import os
import json
import time
import requests

from config import NEURONPEDIA_URL, CACHE_DIR


def _drop_proxies():
    for k in ["ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]:
        os.environ.pop(k, None)


_drop_proxies()


def fetch_feature(model_id, layer_str, index, cache=True, retries=3, sleep=1.5):
    """Return the raw Neuronpedia response for a feature.

    model_id: e.g. 'gemma-2-2b'
    layer_str: e.g. '12-gemmascope-res-16k'
    index: int feature index
    """
    _drop_proxies()
    cache_key = f"{model_id}__{layer_str}__{index}.json".replace("/", "_")
    cache_path = os.path.join(CACHE_DIR, "neuronpedia", cache_key)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if cache and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass
    url = f"{NEURONPEDIA_URL}/{model_id}/{layer_str}/{index}"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                d = r.json()
                with open(cache_path, "w") as f:
                    json.dump(d, f)
                return d
            else:
                time.sleep(sleep * (attempt + 1))
        except Exception as e:
            time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}")


def extract_reference_explanation(feature_json):
    """Return the highest-recall explanation string, or the first available."""
    exps = feature_json.get("explanations") or []
    if not exps:
        return None
    # prefer 'oai_token-act-pair' (Neuronpedia default), else first
    for e in exps:
        if e.get("typeName") == "oai_token-act-pair":
            return e.get("description", "").strip()
    return (exps[0].get("description") or "").strip()


def extract_top_snippets(feature_json, k=20):
    """Return list of dicts {tokens, values, max_val, max_idx, text}."""
    acts = feature_json.get("activations") or []
    # sort by maxValue desc
    acts_sorted = sorted(acts, key=lambda a: -(a.get("maxValue") or 0))[:k]
    out = []
    for a in acts_sorted:
        tokens = a.get("tokens") or []
        values = a.get("values") or []
        max_idx = a.get("maxValueTokenIndex")
        max_val = a.get("maxValue")
        # build a plain text (Gemma tokenizer uses SentencePiece; join preserves things)
        text = "".join(tokens)
        text = text.replace("▁", " ")  # SentencePiece marker
        out.append({
            "tokens": tokens, "values": values,
            "max_val": max_val, "max_idx": max_idx,
            "text": text.strip(),
        })
    return out


def annotate_snippet_max(snippet, ctx=8):
    """Return a compact string highlighting the max-activation token with <<>>."""
    tokens = snippet["tokens"]
    max_idx = snippet["max_idx"]
    if max_idx is None or max_idx >= len(tokens):
        return snippet["text"]
    start = max(0, max_idx - ctx)
    end = min(len(tokens), max_idx + ctx + 1)
    parts = []
    for i in range(start, end):
        t = tokens[i].replace("▁", " ")
        if i == max_idx:
            parts.append("<<" + t + ">>")
        else:
            parts.append(t)
    return "".join(parts).strip()


def peak_context_text(snippet, ctx=30):
    """Return a local window of raw text around the max-activation token (no markers)."""
    tokens = snippet["tokens"]
    max_idx = snippet["max_idx"]
    if max_idx is None or max_idx >= len(tokens):
        return snippet["text"]
    start = max(0, max_idx - ctx)
    end = min(len(tokens), max_idx + ctx + 1)
    return "".join(t.replace("▁", " ") for t in tokens[start:end]).strip()


if __name__ == "__main__":
    d = fetch_feature("gemma-2-2b", "12-gemmascope-res-16k", 0)
    print("Ref:", extract_reference_explanation(d))
    snips = extract_top_snippets(d, k=5)
    for s in snips:
        print(f"  max={s['max_val']:.2f}  {annotate_snippet_max(s)!r}")
