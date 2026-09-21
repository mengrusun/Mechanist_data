"""Load claim.json files and build the text that gets embedded.

The paper (Hao, Xu, Li & Evans, Nature 2026) embeds each paper as
``title + [SEP] + abstract`` (see EmbedWork_AbstractTitle_Specter2.py in
tsinghua-fib-lab/AI-Impacts-Science).  A ``claim.json`` here carries exactly
those two fields ("Title" / "Abstract"), so the mapping is one-to-one:
one claim == one "paper".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


# Which claim.json fields go into the embedded text, per --text-field choice.
TEXT_FIELDS: dict[str, list[str]] = {
    # Faithful to the paper: title + abstract only.
    "title_abstract": ["Title", "Abstract"],
    "title": ["Title"],
    "abstract": ["Abstract"],
    "hypothesis": ["Short Hypothesis"],
    "name_title_hypothesis": ["Name", "Title", "Short Hypothesis"],
    # v1 default: the claim's scientific content minus Experiments / Related Work.
    "title_hypothesis_abstract": ["Title", "Short Hypothesis", "Abstract"],
    # Everything that describes the scientific content of the claim.
    "full": ["Title", "Short Hypothesis", "Abstract", "Experiments"],
    # The complete claim.json: every field, in file order.
    "all": ["Name", "Title", "Short Hypothesis", "Related Work", "Abstract",
            "Experiments", "Risk Factors and Limitations"],
}


@dataclass
class Claim:
    cid: str  # stable id: path of the claim folder relative to the group root
    path: str
    raw: dict[str, Any] = field(repr=False)

    def parts(self, text_field: str) -> list[str]:
        keys = TEXT_FIELDS[text_field]
        out = []
        for k in keys:
            v = self.raw.get(k)
            if isinstance(v, str) and v.strip():
                out.append(" ".join(v.split()))
        if not out:
            raise ValueError(f"{self.path}: no text for field set {text_field!r}")
        return out


def load_claims(root: str) -> list[Claim]:
    """Collect claims from `root`.

    * If `root` is a .json file, it is expected to be a list of claim dicts
      (the hypothesis_all.json format) and each element becomes one Claim,
      keeping the file order.
    * Otherwise `root` is a directory scanned recursively for claim.json.
    """
    root = os.path.abspath(root)

    if os.path.isfile(root):
        with open(root, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{root}: expected a JSON list of claims")
        claims = []
        width = len(str(len(data)))
        for i, raw in enumerate(data):
            name = raw.get("Name") if isinstance(raw, dict) else None
            cid = f"{i:0{width}d}_{name}" if name else f"{i:0{width}d}"
            claims.append(Claim(cid=cid, path=f"{root}#{i}", raw=raw))
        if not claims:
            raise FileNotFoundError(f"no claims in {root}")
        return claims

    claims: list[Claim] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "claim.json" not in filenames:
            continue
        p = os.path.join(dirpath, "claim.json")
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        cid = os.path.relpath(dirpath, root)
        if cid == ".":
            cid = os.path.basename(dirpath)
        claims.append(Claim(cid=cid, path=p, raw=raw))
    if not claims:
        raise FileNotFoundError(f"no claim.json found under {root}")
    claims.sort(key=lambda c: c.cid)
    return claims
