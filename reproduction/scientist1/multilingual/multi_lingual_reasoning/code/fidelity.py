"""Rough output-language fidelity check for MGSM generations.

Uses langdetect and Unicode-script heuristics to classify each generation's language.

Notes:
  - langdetect covers en/es/fr/de/zh/ja/ru/th/bn/sw well but is unreliable on Telugu.
  - For Chinese/Japanese/Russian/Thai/Bengali/Telugu we back it up with a script-share heuristic
    (fraction of alphabetic characters in the target script).
"""
import os, sys, json, glob, re
import unicodedata
sys.path.insert(0, os.path.dirname(__file__))
from common import LANGS, LANG_TIER

from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0


SCRIPT_RANGES = {
    "zh": [(0x4E00, 0x9FFF)],
    "ja": [(0x3040, 0x30FF)],     # hiragana/katakana; kanji shared with zh
    "ru": [(0x0400, 0x04FF)],
    "th": [(0x0E00, 0x0E7F)],
    "te": [(0x0C00, 0x0C7F)],
    "bn": [(0x0980, 0x09FF)],
}

LANGDETECT_MAP = {
    "en": {"en"}, "es": {"es", "ca"}, "fr": {"fr"}, "de": {"de"},
    "zh": {"zh-cn", "zh-tw"}, "ja": {"ja"}, "ru": {"ru"},
    "th": {"th"}, "te": {"te"}, "bn": {"bn"}, "sw": {"sw"},
}


def script_share(text, ranges):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    def in_r(c):
        cp = ord(c)
        return any(a <= cp <= b for (a, b) in ranges)
    n_hit = sum(in_r(c) for c in letters)
    return n_hit / len(letters)


def classify(text, target):
    """Return (is_fidelity_ok, detected_lang, extra_stat)."""
    text = text.strip()
    if not text:
        return False, None, {"reason": "empty"}
    # First try langdetect
    try:
        det = detect(text)
    except LangDetectException:
        det = None
    # Script check for languages with a distinctive script
    if target in SCRIPT_RANGES:
        share = script_share(text, SCRIPT_RANGES[target])
        # For zh/ja/ru/th/te/bn: require >=25% of alphabetic chars in the target script
        ok_script = share >= 0.25
        # For ja specifically, kana share is more indicative
        ok = ok_script or (det in LANGDETECT_MAP.get(target, set()))
        return bool(ok), det, {"target_script_share": share}
    # Latin-script langs: rely on langdetect
    ok = det in LANGDETECT_MAP.get(target, set())
    return bool(ok), det, {}


def analyze(fp):
    with open(fp) as f:
        d = json.load(f)
    per_lang = {}
    for lg in LANGS:
        rows = d["results"].get(lg, {}).get("rows", [])
        if not rows:
            continue
        ok_count = 0
        det_counts = {}
        for r in rows:
            gen = r["gen"]
            ok, det, _ = classify(gen, lg)
            ok_count += int(ok)
            det_counts[det] = det_counts.get(det, 0) + 1
        per_lang[lg] = {
            "n": len(rows),
            "fidelity": ok_count / len(rows),
            "top_detected": sorted(det_counts.items(), key=lambda kv: -kv[1])[:3],
        }
    return per_lang


def main(pattern="results/*.json"):
    files = sorted(glob.glob(pattern))
    print("\t".join(["file"] + LANGS + ["mean"]))
    for fp in files:
        try:
            pl = analyze(fp)
        except Exception as e:
            print(fp, "ERR", e); continue
        row = [os.path.basename(fp)]
        vals = []
        for lg in LANGS:
            v = pl.get(lg, {}).get("fidelity")
            row.append(f"{v:.2f}" if v is not None else "-")
            if v is not None: vals.append(v)
        row.append(f"{sum(vals)/len(vals):.2f}" if vals else "-")
        print("\t".join(row))


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "results/*.json"
    main(pat)
