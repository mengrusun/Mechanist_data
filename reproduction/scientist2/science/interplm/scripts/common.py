"""Shared utilities for SAE-on-ESM-2 reproduction milestones.

All I/O is via env vars DATA_DIR / MODEL_DIR (or the project-local symlinks
under ./data and ./models). Do not hardcode absolute paths in callers.
"""
from __future__ import annotations

import functools
import gzip
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# --- Global bindings ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models"))
SWISSPROT_DIR = Path(os.environ.get("SWISSPROT_DIR", PROJECT_ROOT / "data" / "Swiss-Prot"))
# The full uniprot_sprot.dat.gz file (700MB) that we downloaded to /data/zhenqian/data/Swiss-Prot/full_uniprot/
SWISSPROT_DAT = Path(os.environ.get(
    "SWISSPROT_DAT",
    "/data/zhenqian/data/Swiss-Prot/full_uniprot/uniprot_sprot.dat.gz"
))
UNIREF_DIR = Path(os.environ.get("UNIREF_DIR", PROJECT_ROOT / "data" / "UniRef" / "data"))
ESM_DIR = Path(os.environ.get("ESM_DIR", PROJECT_ROOT / "models" / "ESM-2-650M"))
SAE_ROOT = Path(os.environ.get("SAE_ROOT", PROJECT_ROOT / "models" / "SAE_ESM2_650M"))
LLM_CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", PROJECT_ROOT / "runs" / "cache" / "llm_calls"))

SAE_LAYERS = [1, 9, 18, 24, 30, 33]
DEFAULT_SEED = 42


def seed_all(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Simple SAE loader -------------------------------------------------------


class SparseAutoencoder(torch.nn.Module):
    """Minimal ReLU SAE matching the checkpoint shape:
        encoder.weight  [F, d]   encoder.bias  [F]
        decoder.weight  [d, F]   bias          [d]  (pre-encoder bias)
    Forward path:
        h = x - bias                        # pre-encoder recentering
        z = ReLU(encoder(h) + encoder.bias)
        x_hat = decoder(z) + bias
    """

    def __init__(self, d_in: int, d_feat: int):
        super().__init__()
        self.d_in = d_in
        self.d_feat = d_feat
        self.encoder = torch.nn.Linear(d_in, d_feat, bias=True)
        self.decoder = torch.nn.Linear(d_feat, d_in, bias=False)
        self.pre_bias = torch.nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x - self.pre_bias
        z = torch.relu(self.encoder(h))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.pre_bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def load_sae(layer: int, variant: str = "normalized", device: str = "cpu") -> SparseAutoencoder:
    """Load pretrained SAE for ESM-2-650M layer `layer`.

    variant: 'normalized' -> ae_normalized.pt (decoder cols carry per-feature norm);
             'unnormalized' -> ae_unnormalized.pt (unit-norm decoder cols).
    """
    layer_dir = SAE_ROOT / f"layer_{layer}"
    cfg = json.loads((layer_dir / "config.json").read_text())
    d_in = int(cfg["architecture"]["esm_dim"])
    d_feat = int(cfg["architecture"]["feature_dim"])
    fname = "ae_normalized.pt" if variant == "normalized" else "ae_unnormalized.pt"
    state = torch.load(layer_dir / fname, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(d_in, d_feat)
    # remap 'bias' -> 'pre_bias' to match module
    sd = {}
    for k, v in state.items():
        if k == "bias":
            sd["pre_bias"] = v
        else:
            sd[k] = v
    sae.load_state_dict(sd, strict=True)
    sae.eval()
    sae.to(device)
    return sae


# --- ESM-2 loader (via HuggingFace transformers) ----------------------------


@functools.lru_cache(maxsize=1)
def load_esm(device: str = "cuda") -> Tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ESM_DIR))
    model = AutoModel.from_pretrained(
        str(ESM_DIR),
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    model.eval()
    model.to(device)
    return model, tok


# --- Swiss-Prot .dat annotation parser --------------------------------------

# Feature-table types we care about, mapped to concept-category prefixes.
FT_TYPE_MAP = {
    "BINDING": "binding_site",
    "ACT_SITE": "active_site",
    "MOTIF": "sequence_motif",
    "DOMAIN": "structural_domain",
    "REGION": "functional_domain",
    "MOD_RES": "PTM_site",
    "LIPID": "PTM_site",
    "CARBOHYD": "PTM_site",
    "SITE": "functional_site",
    "TRANSMEM": "structural_domain",  # transmembrane region
    "SIGNAL": "signal_peptide",
    "DISULFID": "PTM_site",
    "CROSSLNK": "PTM_site",
    "ZN_FING": "structural_domain",
    "DNA_BIND": "binding_site",
    "NP_BIND": "binding_site",
    "CA_BIND": "binding_site",
    "METAL": "binding_site",
    "REPEAT": "structural_domain",
    "COILED": "structural_domain",
    "PROPEP": "PTM_site",
}


def _clean_note(note: Optional[str]) -> str:
    if not note:
        return "unspecified"
    n = note.strip()
    # Strip parenthesized/bracketed annotations, take first phrase up to ';'
    n = re.split(r"[;{]", n)[0].strip()
    # Truncate long strings
    if len(n) > 60:
        n = n[:60]
    # Slugify a bit
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"[^A-Za-z0-9_+\-.]", "", n)
    return n if n else "unspecified"


def parse_swissprot_annotations(
    dat_path: Path,
    entries_to_keep: Optional[set] = None,
    max_records: Optional[int] = None,
    stop_when_all_found: bool = True,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[Tuple[str, int, int]]]:
    """Yield dict entry -> list of (concept, start_pos_1based, end_pos_1based).

    Uses BioPython SwissProt parser. Skips any records with position uncertainty
    (BioPython raises for '?' bounds — caught and skipped).

    If entries_to_keep is provided and stop_when_all_found=True, stops early once
    every entry has been found (large speed-up for small entry subsets).
    Tolerates trailing garbage bytes at EOF (returns whatever it parsed).
    """
    from Bio import SwissProt
    import pickle

    # Attempt to load the FULL cache (all-entries) first if a cache dir is provided.
    cache_dir = cache_dir or (PROJECT_ROOT / "runs" / "cache" / "swissprot")
    cache_dir.mkdir(parents=True, exist_ok=True)
    full_cache = cache_dir / "all_annotations.pkl"
    if full_cache.exists() and entries_to_keep is not None:
        with open(full_cache, "rb") as f:
            all_records = pickle.load(f)
        return {k: v for k, v in all_records.items() if k in entries_to_keep}

    records: Dict[str, List[Tuple[str, int, int]]] = {}
    opener = gzip.open if str(dat_path).endswith(".gz") else open
    handle = opener(dat_path, "rt")
    try:
        iterator = SwissProt.parse(handle)
        i = 0
        while True:
            try:
                rec = next(iterator)
            except StopIteration:
                break
            except Exception:
                # Trailing garbage or corrupt tail -> return what we have
                break
            i += 1
            if max_records is not None and i > max_records:
                break
            ac = rec.accessions[0] if rec.accessions else None
            if ac is None:
                continue
            if entries_to_keep is not None and ac not in entries_to_keep:
                continue
            annots: List[Tuple[str, int, int]] = []
            for f in rec.features:
                ft_type = f.type
                cat = FT_TYPE_MAP.get(ft_type)
                if cat is None:
                    continue
                try:
                    start_0 = int(f.location.start)
                    end_0 = int(f.location.end)
                except (TypeError, ValueError):
                    continue
                if end_0 <= start_0:
                    continue
                # add Pfam family-style subclass from note
                note = _clean_note(f.qualifiers.get("note") if isinstance(f.qualifiers, dict) else None)
                concept = f"{cat}::{note}"
                # 1-based inclusive interval convention (consumer flips as needed)
                annots.append((concept, start_0 + 1, end_0))
            # Cross-references: Pfam families → structural_domain sub-classes at CHAIN scope
            for cr in rec.cross_references:
                if cr and cr[0] == "Pfam" and len(cr) >= 3:
                    pfam_id = cr[1]
                    # Coarse: mark ALL residues as Pfam::<id> (no local coords in DR line)
                    # We only add this if there is at least one DOMAIN entry with the same family name; otherwise skip
                    # to avoid claiming Pfam coverage of full sequence blindly.
                    pass  # skip full-sequence Pfam attribution; DOMAIN feature-table entries already carry subfamily
            if annots:
                records[ac] = annots
            # Early stop when all requested entries have been found
            if (
                entries_to_keep is not None
                and stop_when_all_found
                and len(records) >= len(entries_to_keep)
            ):
                break
    finally:
        try:
            handle.close()
        except Exception:
            pass
    # Persist full-parse cache when caller asked for everything
    if entries_to_keep is None and max_records is None:
        try:
            with open(full_cache, "wb") as f:
                pickle.dump(records, f)
        except Exception:
            pass
    return records


def build_residue_labels(
    seqs: Dict[str, str],
    annotations: Dict[str, List[Tuple[str, int, int]]],
) -> Tuple[Dict[str, Dict[int, set]], List[str]]:
    """For each entry with sequence, return {entry: {residue_1based: set(concepts)}}
    and the sorted union concept universe.
    """
    concept_universe: set = set()
    per_entry: Dict[str, Dict[int, set]] = {}
    for ac, ann in annotations.items():
        seq = seqs.get(ac)
        if seq is None:
            continue
        L = len(seq)
        residue_map: Dict[int, set] = {}
        for concept, s, e in ann:
            s = max(1, s)
            e = min(L, e)
            for r in range(s, e + 1):
                residue_map.setdefault(r, set()).add(concept)
                concept_universe.add(concept)
        if residue_map:
            per_entry[ac] = residue_map
    return per_entry, sorted(concept_universe)


# --- Activation / feature extraction ----------------------------------------


class ActivationCollector:
    """Collect residual-stream activations from ESM-2 hidden_states at a target layer.

    Uses `output_hidden_states=True` and picks `hidden_states[layer]`. For
    ESM-2, hidden_states[0] is the input embedding, hidden_states[i] is the
    output of the i-th transformer block. So layer L (per SAE naming) = index L.
    """

    def __init__(self, model, tokenizer, device: str = "cuda", max_len: int = 1024, batch_size: int = 4):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size

    @torch.no_grad()
    def batch_hidden_states(self, sequences: List[str], layer: int) -> List[torch.Tensor]:
        """Return per-sequence [L, d] float16 activations for the requested layer.
        Strips CLS/EOS/PAD; only returns residue-aligned positions.
        """
        outputs: List[torch.Tensor] = []
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )
            hs = out.hidden_states[layer]  # [B, T, d]
            am = enc["attention_mask"]  # [B, T]
            for j, seq in enumerate(batch):
                mask = am[j].bool()
                # Drop CLS at position 0 and EOS at last non-pad position
                # ESM-2 tokenizer: [CLS] a b c ... [EOS] [PAD] ...
                nonpad_len = int(mask.sum().item())
                # keep positions 1 .. nonpad_len - 2 inclusive => the L residues
                start, end = 1, nonpad_len - 1
                if end <= start:
                    outputs.append(torch.zeros((0, hs.shape[-1]), dtype=torch.float16))
                    continue
                per_res = hs[j, start:end].detach().to(torch.float16).cpu()
                # Truncate label-side sequence to match model-processed length (no zero-padding —
                # zero-padded residues would leak fake all-zero vectors into F1/probes).
                outputs.append(per_res)
        return outputs


# --- LLM client (dmxapi.cn / gpt-5.4) ---------------------------------------


def _hash_prompt(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class LLMClient:
    """OpenAI-compatible client for the dmxapi.cn endpoint with disk caching."""

    def __init__(
        self,
        base_url: str = "https://www.dmxapi.cn/v1",
        model: str = "gpt-5.4",
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        from openai import OpenAI

        api_key = api_key or os.environ.get("DMX_API_KEY")
        if not api_key:
            raise RuntimeError("DMX_API_KEY unset and no api_key passed")
        # Bypass system proxy by providing an http_client with no proxies
        import httpx

        http_client = httpx.Client(trust_env=False, timeout=60.0)
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        self.model = model
        self.cache_dir = cache_dir or LLM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        response_schema_key: str = "chat_json",
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_schema_key": response_schema_key,
        }
        h = _hash_prompt(payload)
        cache_path = self.cache_dir / f"{h}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                pass
        # Live call
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                parsed = json.loads(content)
                result = {"ok": True, "raw": content, "parsed": parsed, "attempt": attempt + 1}
                cache_path.write_text(json.dumps(result))
                return result
            except Exception as e:
                if attempt == 2:
                    result = {"ok": False, "error": str(e), "attempt": attempt + 1}
                    cache_path.write_text(json.dumps(result))
                    return result
                time.sleep(2 * (attempt + 1))
        return {"ok": False, "error": "exhausted retries", "attempt": 3}


# --- F1 alignment core -------------------------------------------------------


def compute_topk_thresholds(activations: torch.Tensor, q_top: float) -> torch.Tensor:
    """Per-feature quantile threshold. `activations`: [N, F]. Returns [F]."""
    # torch.quantile is memory-heavy; do it per-feature chunk for large F
    N, F = activations.shape
    thr = torch.empty(F, dtype=activations.dtype, device=activations.device)
    chunk = 512
    for i in range(0, F, chunk):
        sl = activations[:, i : i + chunk]
        thr[i : i + chunk] = torch.quantile(sl.float(), q_top, dim=0).to(activations.dtype)
    return thr


def compute_unit_concept_f1(
    unit_mask: torch.Tensor,  # [N, F] bool — unit fires >= q_top
    concept_mask: torch.Tensor,  # [N, C] bool — residue carries concept
) -> torch.Tensor:
    """Return [F, C] float F1 matrix."""
    # Convert to float for matmul
    U = unit_mask.float()  # [N, F]
    C = concept_mask.float()  # [N, C]
    # TP[u,c] = sum_n U[n,u] * C[n,c]
    tp = U.T @ C  # [F, C]
    fp = U.sum(dim=0).unsqueeze(1) - tp  # [F, 1] - [F, C] = [F, C]
    fn = C.sum(dim=0).unsqueeze(0) - tp  # [1, C] - [F, C] = [F, C]
    prec = tp / (tp + fp).clamp_min(1e-12)
    rec = tp / (tp + fn).clamp_min(1e-12)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
    return f1
