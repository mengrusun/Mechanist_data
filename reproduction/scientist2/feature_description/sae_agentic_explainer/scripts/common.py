"""
Common utilities for SAGE reproduction.

Provides:
  - GPT5Client: rate-limited, retry-safe DMXAPI wrapper (proxy bypassed at construction time).
  - JumpReLUSAE: numpy-loaded JumpReLU SAE (matches Gemma-Scope release format).
  - load_gemma_scope_sae: loads a Gemma-Scope residual-stream SAE for a given layer + l0 variant.
  - register_residual_hook: registers a forward hook that captures the residual stream at layer L.
  - NeuronpediaClient: fetches feature explanations + top-activating snippets.
  - normalize_activation: min-max within-feature normalization.
  - hit_threshold_from_top_activations: computes tau_f = 99th percentile from Neuronpedia's activation distribution.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from openai import OpenAI


# --------------------------------------------------------------------------- #
# GPT-5 (DMXAPI) client — bypasses proxy at construction time.
# --------------------------------------------------------------------------- #

DMXAPI_KEY = "<Your_api>"
DMXAPI_URL = "https://www.dmxapi.cn/v1"
DMXAPI_MODEL = "gpt-5.4"


def _clear_proxy_env() -> None:
    """Neutralize proxy env vars in-process. Must be called before OpenAI() construction."""
    for k in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
    ):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


class GPT5Client:
    """
    Thin wrapper around OpenAI SDK pointed at DMXAPI.

    - Bypasses proxy.
    - Auto-retries on 429/5xx with exponential backoff.
    - Rate-limits to <=`qps` requests per second (rolling window).
    - Records token usage.
    """

    def __init__(self, model: str = DMXAPI_MODEL, qps: float = 5.0, max_retries: int = 6):
        _clear_proxy_env()
        self.client = OpenAI(api_key=DMXAPI_KEY, base_url=DMXAPI_URL)
        self.model = model
        self.qps = qps
        self.max_retries = max_retries
        self._last_call = 0.0
        self.usage_prompt = 0
        self.usage_completion = 0
        self.n_calls = 0
        self.n_errors = 0

    def _throttle(self) -> None:
        min_interval = 1.0 / max(self.qps, 0.01)
        elapsed = time.time() - self._last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call = time.time()

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int | None = 512) -> str:
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                kwargs = dict(model=self.model, messages=messages)
                # gpt-5 style params — many gateways ignore temperature; we still send it.
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if max_tokens is not None:
                    kwargs["max_completion_tokens"] = max_tokens
                r = self.client.chat.completions.create(**kwargs)
                self.n_calls += 1
                if r.usage:
                    self.usage_prompt += r.usage.prompt_tokens or 0
                    self.usage_completion += r.usage.completion_tokens or 0
                content = r.choices[0].message.content
                # Some GPT-5 gateways return empty content on trivial prompts; retry once if empty.
                if not content or not content.strip():
                    if attempt < 2:
                        continue
                return content or ""
            except Exception as e:  # noqa: BLE001
                self.n_errors += 1
                wait = min(2.0 ** attempt, 30.0) + random.uniform(0, 1.0)
                print(f"[gpt-5 retry {attempt+1}/{self.max_retries}] {type(e).__name__}: {str(e)[:120]}, sleeping {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"GPT-5 API exhausted retries after {self.max_retries} attempts")

    def stats(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_errors": self.n_errors,
            "prompt_tokens": self.usage_prompt,
            "completion_tokens": self.usage_completion,
        }


# --------------------------------------------------------------------------- #
# JumpReLU SAE loader (Gemma-Scope release format: .npz with W_enc/W_dec/b_enc/b_dec/threshold).
# --------------------------------------------------------------------------- #


class JumpReLUSAE(torch.nn.Module):
    """
    Gemma-Scope JumpReLU SAE.

        pre = W_enc @ (a - b_dec) + b_enc      # note: apply mean-centering via b_dec
        gate = pre > threshold
        f = pre * gate * relu(pre)  --> equivalent JumpReLU form: f = (pre > threshold) * pre (with pre>=0)
    We follow the official Gemma-Scope inference recipe:
        z = pre_activation
        f = z * (z > threshold)                # JumpReLU
        recon = W_dec @ f + b_dec
    """

    def __init__(self, W_enc: np.ndarray, W_dec: np.ndarray, b_enc: np.ndarray, b_dec: np.ndarray, threshold: np.ndarray):
        super().__init__()
        # W_enc: (d_model, d_sae) -- input times W_enc
        # W_dec: (d_sae, d_model)
        self.W_enc = torch.nn.Parameter(torch.from_numpy(W_enc).float(), requires_grad=False)
        self.W_dec = torch.nn.Parameter(torch.from_numpy(W_dec).float(), requires_grad=False)
        self.b_enc = torch.nn.Parameter(torch.from_numpy(b_enc).float(), requires_grad=False)
        self.b_dec = torch.nn.Parameter(torch.from_numpy(b_dec).float(), requires_grad=False)
        self.threshold = torch.nn.Parameter(torch.from_numpy(threshold).float(), requires_grad=False)
        self.d_model = W_enc.shape[0]
        self.d_sae = W_enc.shape[1]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model). Standard Gemma-Scope: pre = (x - b_dec) @ W_enc + b_enc; f = pre * (pre > threshold)
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        mask = (pre > self.threshold).float()
        return pre * mask

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec


def load_gemma_scope_sae(layer: int, l0: int, root: str = "/data/zhenqian/models/gemma-scope-2b-pt-res") -> JumpReLUSAE:
    path = Path(root) / f"layer_{layer}" / "width_16k" / f"average_l0_{l0}" / "params.npz"
    if not path.exists():
        raise FileNotFoundError(f"SAE params not found: {path}")
    p = np.load(path)
    return JumpReLUSAE(p["W_enc"], p["W_dec"], p["b_enc"], p["b_dec"], p["threshold"])


# --------------------------------------------------------------------------- #
# Neuronpedia client — public read-only API, no auth needed.
# --------------------------------------------------------------------------- #

NEURONPEDIA_BASE = "https://www.neuronpedia.org/api"


class NeuronpediaClient:
    """
    Fetches feature metadata + top-activating snippets from Neuronpedia's public API.

    For gemma-2-2b + gemmascope-res-16k, feature endpoint is:
        /api/feature/gemma-2-2b/{layer}-gemmascope-res-16k/{fid}
    """

    def __init__(self, model_id: str = "gemma-2-2b", sae_id_template: str = "{layer}-gemmascope-res-16k",
                 cache_dir: str = "/data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer/data_cache/neuronpedia"):
        self.model_id = model_id
        self.sae_id_template = sae_id_template
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Bypass proxy
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {}

    def get_feature(self, layer: int, fid: int) -> dict:
        cache_path = self.cache_dir / f"L{layer}_F{fid}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                pass
        sae_id = self.sae_id_template.format(layer=layer)
        url = f"{NEURONPEDIA_BASE}/feature/{self.model_id}/{sae_id}/{fid}"
        for attempt in range(4):
            try:
                r = self.session.get(url, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    cache_path.write_text(json.dumps(data))
                    return data
                elif r.status_code == 404:
                    return {"error": "not_found", "status": 404}
                else:
                    time.sleep(2 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
        return {"error": "fetch_failed"}

    @staticmethod
    def parse_activation_snippets(feature_data: dict, max_snippets: int = 50) -> list[dict]:
        """
        Returns a list of {'text': ..., 'max_val': ..., 'tokens': [...], 'values': [...]}.

        Neuronpedia's `activations` field is a list of examples; each has `tokens`, `values`, `maxValue`.
        """
        acts = feature_data.get("activations", [])
        out = []
        for a in acts[:max_snippets]:
            toks = a.get("tokens", [])
            vals = a.get("values", [])
            if not toks or not vals:
                continue
            text = "".join([str(t) for t in toks])
            out.append({
                "text": text,
                "tokens": toks,
                "values": vals,
                "max_val": float(a.get("maxValue", max(vals) if vals else 0.0)),
            })
        return out

    @staticmethod
    def parse_public_explanation(feature_data: dict) -> str | None:
        exps = feature_data.get("explanations", [])
        if not exps:
            return None
        # Pick the highest-recall (if scored) OR the first one.
        scored = []
        for e in exps:
            desc = (e.get("description") or "").strip()
            if not desc:
                continue
            score = 0.0
            for s in e.get("scores", []) or []:
                if s.get("explanationScoreTypeName") in ("recall_alt", "recall", "detection"):
                    try:
                        score = max(score, float(s.get("value") or 0.0))
                    except Exception:
                        pass
            scored.append((score, desc))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][1]


# --------------------------------------------------------------------------- #
# Utility: normalize activations
# --------------------------------------------------------------------------- #


def normalize_activations(vals: np.ndarray) -> np.ndarray:
    """Min-max normalize within-feature; if all zero return zeros."""
    vals = np.asarray(vals, dtype=np.float32)
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo + 1e-8:
        return np.zeros_like(vals)
    return (vals - lo) / (hi - lo)


def percentile_threshold(vals: list[float], q: float = 99.0) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.array(vals, dtype=np.float32), q))


# --------------------------------------------------------------------------- #
# Residual-stream hook
# --------------------------------------------------------------------------- #


class ResidualCapture:
    """Captures the residual-stream output of block L (post-layernorm-input, pre-residual-add is model-dependent).

    For Gemma-2, we hook `model.layers[L]` output which is the residual-stream after block L (input to L+1).
    Following Gemma-Scope's convention: hook_resid_post = layer output.
    """

    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        self.captured: torch.Tensor | None = None
        self.handle = None

    def __enter__(self):
        def _hook(module, input, output):
            # output can be a tuple; keep the first item (hidden states)
            if isinstance(output, tuple):
                self.captured = output[0].detach()
            else:
                self.captured = output.detach()
        self.handle = self.model.model.layers[self.layer].register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle:
            self.handle.remove()
        self.handle = None


# --------------------------------------------------------------------------- #
# Prompt templates for baseline single-pass GPT-5 (Bills-2023 style)
# --------------------------------------------------------------------------- #


SINGLE_PASS_EXPLAINER_PROMPT = """You are given a neuron/feature from a sparse autoencoder trained on a language model. Below are examples of text snippets that activate this feature, with per-token activation values (higher = stronger activation). Your job is to write a concise natural-language explanation of what this feature responds to. Keep it to ONE SHORT phrase (5-15 words), no preamble, no formatting, no examples.

Top activating snippets:
{snippets}

Now write the one-line explanation for this feature. Only output the phrase, nothing else."""


def format_snippets_for_prompt(snippets: list[dict], k: int = 10, max_chars_per_snippet: int = 240) -> str:
    """Formats top-k snippets for prompt. Shows the text + max-activation position."""
    lines = []
    for i, s in enumerate(snippets[:k]):
        text = s.get("text", "")
        vals = s.get("values", [])
        toks = s.get("tokens", [])
        if not vals:
            continue
        max_idx = int(np.argmax(vals))
        # Extract a small window around max_idx
        window = 20
        lo, hi = max(0, max_idx - window), min(len(toks), max_idx + window + 1)
        chunk_toks = toks[lo:hi]
        # Mark the peak token with **
        marked = []
        for j, t in enumerate(chunk_toks):
            if lo + j == max_idx:
                marked.append(f"**{t}**")
            else:
                marked.append(str(t))
        chunk = "".join(marked)
        if len(chunk) > max_chars_per_snippet:
            chunk = chunk[:max_chars_per_snippet] + "..."
        lines.append(f"[{i+1}] max_act={float(s.get('max_val', max(vals))):.2f} | ...{chunk}...")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# SAGE 4-role iterative loop
# --------------------------------------------------------------------------- #


SAGE_EXPLAINER_INITIAL = """You are the Explainer role in an iterative auto-interpretation loop. You are given top-activating text snippets for one SAE feature in a language model. Propose {n_candidates} DISTINCT candidate one-line natural-language explanations, each describing a different possible activation pattern (e.g. syntactic vs semantic, general concept vs specific token, and so on).

Top activating snippets:
{snippets}

Output STRICTLY as JSON, no preamble:
{{"candidates": ["candidate 1 phrase (5-15 words)", "candidate 2 phrase", "candidate 3 phrase"]}}"""

SAGE_EXPLAINER_REVISE = """You are the Explainer role. Below are your current candidate explanations for one SAE feature, plus empirical activation feedback from the Analyzer on probes written by the Designer. Revise the candidates: keep good ones as-is, tighten weak ones, drop implausible ones, and add up to 1 new candidate if the feedback suggests a distinct pattern. Return at least 1 and at most {n_candidates} candidates.

Current candidates:
{candidates}

Empirical feedback (per-candidate mean predicted-vs-observed match, hit-rate on positive probes, false-alarm rate on negative probes):
{feedback}

Output STRICTLY as JSON, no preamble:
{{"candidates": ["revised candidate 1", ...]}}"""

SAGE_DESIGNER = """You are the Designer role. Your job is to write short discriminating probe texts that will help decide between candidate explanations of an SAE feature. For each candidate, write {n_positive} text snippets that (according to that candidate) SHOULD activate the feature, and {n_negative} that (according to that candidate) SHOULD NOT. Keep each probe short (10-40 words) and standalone.

Candidates:
{candidates}

Output STRICTLY as JSON, no preamble:
{{"probes": [
  {{"candidate_idx": 0, "polarity": "positive", "text": "..."}},
  {{"candidate_idx": 0, "polarity": "negative", "text": "..."}},
  ...
]}}"""

SAGE_REVIEWER = """You are the Reviewer role. Given each candidate explanation and its empirical activation evidence (mean activation on its positive probes, mean activation on its negative probes, mean activation on all cross-candidate probes), decide per candidate: ACCEPT (keep), REJECT (drop), or REVISE (needs revision). Also decide overall_status: DONE if at least one candidate is ACCEPTed with clear separation between positive and negative probes, otherwise CONTINUE.

Candidates and evidence:
{evidence}

Output STRICTLY as JSON:
{{"decisions": [{{"candidate_idx": 0, "action": "ACCEPT|REJECT|REVISE", "reason": "..."}}, ...], "overall_status": "DONE|CONTINUE"}}"""


def _parse_json_block(text: str) -> dict | None:
    """Extract the first JSON object from an LLM output, tolerating code fences."""
    text = text.strip()
    if text.startswith("```"):
        # strip markdown fence
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Extract balanced braces
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except Exception:
                    return None
    return None


@dataclass
class SAGEConfig:
    n_candidates: int = 3
    n_positive_per_candidate: int = 3
    n_negative_per_candidate: int = 2
    max_rounds: int = 3


def compute_feature_activations_on_texts(
    model, tokenizer, sae: JumpReLUSAE, layer: int, feature_id: int,
    texts: list[str], device: str = "cuda", batch_size: int = 8, max_length: int = 128,
) -> list[float]:
    """
    Returns per-text max activation of feature `feature_id` at layer `layer`.
    """
    results = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            with ResidualCapture(model, layer) as cap:
                model(**enc)
            hidden = cap.captured  # (B, T, d_model)
            if hidden is None:
                results.extend([0.0] * len(batch_texts))
                continue
            # Encode with SAE
            hidden = hidden.to(sae.W_enc.dtype).to(sae.W_enc.device)
            f = sae.encode(hidden)  # (B, T, d_sae)
            f_col = f[..., feature_id]  # (B, T)
            # Mask out padding
            attn_mask = enc["attention_mask"]  # (B, T)
            f_col = f_col.masked_fill(attn_mask == 0, 0.0)
            max_per_seq = f_col.max(dim=-1).values  # (B,)
            results.extend(max_per_seq.detach().cpu().float().tolist())
    return results


def run_sage_loop(
    client: GPT5Client,
    model, tokenizer, sae: JumpReLUSAE, layer: int, feature_id: int,
    top_activating_snippets: list[dict],
    config: SAGEConfig,
    device: str = "cuda",
    log: dict | None = None,
) -> dict:
    """
    Runs the 4-role SAGE loop for ONE feature.

    Returns {'final_explanation': str, 'all_candidates': [...], 'rounds': [...], 'stopped_reason': str}
    """
    log = log if log is not None else {}
    snippet_text = format_snippets_for_prompt(top_activating_snippets, k=10)
    # Round 1: Explainer proposes N candidates.
    exp_prompt = SAGE_EXPLAINER_INITIAL.format(n_candidates=config.n_candidates, snippets=snippet_text)
    exp_out = client.chat([{"role": "user", "content": exp_prompt}], temperature=0.7, max_tokens=600)
    exp_json = _parse_json_block(exp_out) or {}
    candidates = exp_json.get("candidates") or []
    candidates = [c.strip() for c in candidates if isinstance(c, str) and c.strip()][:config.n_candidates]
    if not candidates:
        # Fallback: use a single-pass baseline
        fb_prompt = SINGLE_PASS_EXPLAINER_PROMPT.format(snippets=snippet_text)
        fb = client.chat([{"role": "user", "content": fb_prompt}], temperature=0.5, max_tokens=100).strip()
        return {"final_explanation": fb, "all_candidates": [fb], "rounds": [], "stopped_reason": "fallback_no_initial_candidates"}

    rounds: list[dict] = []
    stopped_reason = "max_rounds"
    for r in range(config.max_rounds):
        # Designer writes probes for these candidates
        cand_str = "\n".join([f"[{i}] {c}" for i, c in enumerate(candidates)])
        des_prompt = SAGE_DESIGNER.format(
            n_positive=config.n_positive_per_candidate,
            n_negative=config.n_negative_per_candidate,
            candidates=cand_str,
        )
        des_out = client.chat([{"role": "user", "content": des_prompt}], temperature=0.7, max_tokens=1200)
        des_json = _parse_json_block(des_out) or {}
        probes = des_json.get("probes") or []
        # Filter to valid probes
        valid_probes = []
        for p in probes:
            if not isinstance(p, dict):
                continue
            ci = p.get("candidate_idx")
            pol = p.get("polarity")
            text = p.get("text", "").strip()
            if not isinstance(ci, int) or ci >= len(candidates) or pol not in ("positive", "negative") or not text:
                continue
            valid_probes.append({"candidate_idx": ci, "polarity": pol, "text": text})
        # Analyzer runs activations on each probe
        if valid_probes:
            probe_texts = [p["text"] for p in valid_probes]
            probe_acts = compute_feature_activations_on_texts(
                model, tokenizer, sae, layer, feature_id, probe_texts, device=device,
            )
        else:
            probe_acts = []
        # Aggregate per-candidate evidence
        evidence_per_cand: list[dict] = []
        for ci, cand in enumerate(candidates):
            pos_acts = [probe_acts[i] for i, p in enumerate(valid_probes) if p["candidate_idx"] == ci and p["polarity"] == "positive"]
            neg_acts = [probe_acts[i] for i, p in enumerate(valid_probes) if p["candidate_idx"] == ci and p["polarity"] == "negative"]
            # cross probes: others' positives should NOT activate strongly if this candidate is disjoint;
            # others' negatives are irrelevant.
            other_pos_acts = [probe_acts[i] for i, p in enumerate(valid_probes) if p["candidate_idx"] != ci and p["polarity"] == "positive"]
            evidence_per_cand.append({
                "candidate_idx": ci,
                "candidate": cand,
                "mean_pos": float(np.mean(pos_acts)) if pos_acts else 0.0,
                "mean_neg": float(np.mean(neg_acts)) if neg_acts else 0.0,
                "mean_other_pos": float(np.mean(other_pos_acts)) if other_pos_acts else 0.0,
                "n_pos": len(pos_acts),
                "n_neg": len(neg_acts),
            })
        # Reviewer decides
        evidence_str = "\n".join([
            f"[{e['candidate_idx']}] {e['candidate']} | pos_mean={e['mean_pos']:.3f} (n={e['n_pos']}) | neg_mean={e['mean_neg']:.3f} (n={e['n_neg']}) | other_pos_mean={e['mean_other_pos']:.3f}"
            for e in evidence_per_cand
        ])
        rev_prompt = SAGE_REVIEWER.format(evidence=evidence_str)
        rev_out = client.chat([{"role": "user", "content": rev_prompt}], temperature=0.3, max_tokens=600)
        rev_json = _parse_json_block(rev_out) or {}
        decisions = rev_json.get("decisions") or []
        overall = (rev_json.get("overall_status") or "").upper()

        rounds.append({
            "round": r + 1,
            "candidates": list(candidates),
            "probes": valid_probes,
            "probe_acts": probe_acts,
            "evidence": evidence_per_cand,
            "decisions": decisions,
            "overall_status": overall,
        })

        # Apply decisions
        actions = {d.get("candidate_idx"): (d.get("action") or "").upper() for d in decisions if isinstance(d, dict)}
        kept = []
        needs_revise_idxs = []
        for i, c in enumerate(candidates):
            a = actions.get(i, "ACCEPT" if any(e["candidate_idx"] == i and e["mean_pos"] > e["mean_neg"] and e["mean_pos"] > e["mean_other_pos"] for e in evidence_per_cand) else "REVISE")
            if a == "ACCEPT":
                kept.append(c)
            elif a == "REVISE":
                kept.append(c)
                needs_revise_idxs.append(i)
            # REJECT: drop

        if not kept:
            # keep the single best-by-margin candidate
            best = max(evidence_per_cand, key=lambda e: e["mean_pos"] - e["mean_neg"] - 0.5 * e["mean_other_pos"])
            kept = [best["candidate"]]

        if overall == "DONE":
            stopped_reason = "reviewer_done"
            candidates = kept
            break

        # If continuing but no revise needed, still stop
        if not needs_revise_idxs and r > 0:
            stopped_reason = "no_revise_needed"
            candidates = kept
            break

        # Revise: ask Explainer to refine
        feedback_str = "\n".join([
            f"[{e['candidate_idx']}] pos_mean={e['mean_pos']:.3f}, neg_mean={e['mean_neg']:.3f}, other_pos_mean={e['mean_other_pos']:.3f}"
            for e in evidence_per_cand
        ])
        rev_prompt2 = SAGE_EXPLAINER_REVISE.format(
            n_candidates=config.n_candidates,
            candidates=cand_str,
            feedback=feedback_str,
        )
        rev_out2 = client.chat([{"role": "user", "content": rev_prompt2}], temperature=0.7, max_tokens=500)
        rev_json2 = _parse_json_block(rev_out2) or {}
        new_cands = [c.strip() for c in (rev_json2.get("candidates") or []) if isinstance(c, str) and c.strip()][:config.n_candidates]
        if new_cands:
            candidates = new_cands
        else:
            candidates = kept
            stopped_reason = "explainer_no_revision"
            break

    # Pick final: the candidate with best pos-minus-neg margin in the last round
    if rounds:
        last_evidence = rounds[-1]["evidence"]
        best_evidence = max(last_evidence, key=lambda e: e["mean_pos"] - e["mean_neg"] - 0.5 * e["mean_other_pos"])
        # If candidates list was refined after evidence, prefer the surviving list's first item
        final_expl = candidates[0] if candidates else best_evidence["candidate"]
    else:
        final_expl = candidates[0] if candidates else ""

    return {
        "final_explanation": final_expl,
        "all_candidates": candidates,
        "rounds": rounds,
        "stopped_reason": stopped_reason,
    }


# --------------------------------------------------------------------------- #
# Probe-writer and scorer for evaluation.
# --------------------------------------------------------------------------- #

PROBE_WRITER_PROMPT = """You are an independent probe writer. Given an SAE feature's one-line explanation, write exactly {n_probes} short natural texts (20-60 words each) that INSTANTIATE the explanation — texts that a human reading them would say clearly exhibit the described pattern. Do NOT copy the explanation verbatim into the text; write natural sentences/paragraphs.

Explanation: {explanation}

Output STRICTLY as JSON:
{{"probes": ["probe 1 text", "probe 2 text", ...]}}"""


SCORER_PROMPT = """You are an independent activation predictor. Given an SAE feature's one-line explanation and a text snippet, predict how strongly the feature would activate on this text on a scale from 0.0 (no activation) to 1.0 (peak activation). Consider the explanation carefully. Output STRICTLY as JSON with just a numeric value.

Explanation: {explanation}

Text: {text}

Output: {{"activation": <number in [0,1]>}}"""


def write_probes(client: GPT5Client, explanation: str, n_probes: int = 5) -> list[str]:
    prompt = PROBE_WRITER_PROMPT.format(explanation=explanation, n_probes=n_probes)
    out = client.chat([{"role": "user", "content": prompt}], temperature=0.8, max_tokens=800)
    j = _parse_json_block(out) or {}
    probes = j.get("probes") or []
    probes = [p.strip() for p in probes if isinstance(p, str) and p.strip()]
    return probes[:n_probes]


def score_activation(client: GPT5Client, explanation: str, text: str) -> float:
    prompt = SCORER_PROMPT.format(explanation=explanation, text=text[:500])
    out = client.chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=60)
    j = _parse_json_block(out) or {}
    v = j.get("activation")
    try:
        v = float(v)
    except Exception:
        # Try to grep a number out
        m = re.search(r"([0-9]*\.?[0-9]+)", out)
        v = float(m.group(1)) if m else 0.5
    return max(0.0, min(1.0, v))
