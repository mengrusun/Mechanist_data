"""Thin wrapper around GlotLID V3 (fastText) for output-language fidelity measurement."""

import os
import re
from pathlib import Path
from typing import Optional


class GlotLID:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.environ.get("GLOTLID_PATH", "/data/zhenqian/models/glotlid/model_v3.bin")
        # Fall back to Meta's fastText lid.176 (Meta / official) if GlotLID model is unavailable / truncated.
        # The plan specifies GlotLID; if unavailable we transparently use lid.176 as the closest available substitute.
        # Full GlotLID v3 is 1.687 GB; anything smaller than 1.6 GB is treated as a truncated download.
        if not Path(model_path).exists() or Path(model_path).stat().st_size < 1_600_000_000:
            fallback = "/data/zhenqian/models/lid176/lid.176.bin"
            if Path(fallback).exists() and Path(fallback).stat().st_size > 100_000_000:
                model_path = fallback
                self._backend = "lid.176"
            else:
                raise FileNotFoundError(f"GlotLID model not found at {model_path} and lid.176 fallback missing")
        else:
            self._backend = "glotlid"
        import fasttext
        fasttext.FastText.eprint = lambda x: None
        self.model = fasttext.load_model(model_path)
        # Map lid.176 ISO-639-1 codes to GlotLID's ISO 639-3 + script format so the rest of the pipeline stays uniform.
        # lid.176 uses two-letter codes; we canonicalize to the plan's ISO 639-3 + script.
        self._lid176_to_glotlid = {
            "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
            "zh": "zho_Hans", "ja": "jpn_Jpan", "ru": "rus_Cyrl", "th": "tha_Thai",
            "te": "tel_Telu", "bn": "ben_Beng", "sw": "swh_Latn",
        }

    @staticmethod
    def _clean_for_lid(text: str) -> str:
        # GlotLID cannot handle newlines: replace with spaces and strip control characters
        t = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        # Remove null bytes just in case
        t = t.replace("\x00", "")
        return t.strip()

    def predict(self, text: str, k: int = 1):
        """Return (label_no_prefix, score). Empty text returns ('und', 0.0).

        When backing model is lid.176 (fallback), the ISO 639-1 code is translated to the
        GlotLID-format ISO 639-3 + script code so downstream comparisons stay uniform.
        """
        t = self._clean_for_lid(text)
        if not t:
            return ("und", 0.0)
        labels, scores = self.model.predict(t, k=k)
        label = labels[0].replace("__label__", "")
        if getattr(self, "_backend", None) == "lid.176":
            label = self._lid176_to_glotlid.get(label, label + "_?")
        return (label, float(scores[0]))

    def is_language(self, text: str, target_label: str, min_score: float = 0.0) -> bool:
        pred_label, score = self.predict(text)
        return pred_label == target_label and score >= min_score
