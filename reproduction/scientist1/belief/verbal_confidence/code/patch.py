"""
Causal activation-patching experiment.

We test the "cached confidence" claim by transplanting residual-stream
activations from a HIGH-confidence run into a LOW-confidence run and measuring
how the verbalised confidence shifts.

For each (HIGH, LOW) pair we run the LOW example forward and, at a chosen
LAYER-RANGE, replace the residual stream at a chosen POSITION with the HIGH
example's residual at the same layer / same *semantic* position.

We evaluate three positions (independently):
  - pre   : at the "Answer:" tag, before the answer has been emitted.
            The model still has to write the answer text itself; if confidence
            is computed only later from the answer text, this patch should not
            move it (control).
  - ans   : the last token of the answer text (the alleged cache site).
  - post  : the ":" of "Confidence (0-100):", i.e. the read-out position.
            This is directly under the LM head that emits the confidence token
            so a strong effect is expected -- this is a positive control.

We patch a whole range of layers to avoid the confound of "single-layer patch
is drowned out by downstream re-computation".

The claim ("cached at ans, retrieved at post") predicts:
    |Delta(patch_ans)|  >>  |Delta(patch_pre)|
    |Delta(patch_ans)|  comparable to  |Delta(patch_post)|
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_utils import ANSWER_TAG, CONF_TAG, build_full_prompt, parse_confidence  # noqa


def load_model(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    ).eval()
    return model, tok


def char_to_token(tok, full: str, char_end: int) -> int:
    ids = tok(full[:char_end], return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    return int(ids.shape[0]) - 1


def get_anchor_positions(tok, full: str) -> dict:
    """Return dict of {pre, ans, nl, post} → token index in ``full``.

    - pre  : end of the "Answer:" tag (before the answer text starts)
    - ans  : last token of the answer text (immediately before "\n")
    - nl   : token index of the "\n" between the answer and CONF_TAG
    - post : end of ``Confidence (0-100):`` (right before the number)
    """
    ans_head_pos = full.rfind(ANSWER_TAG) + len(ANSWER_TAG)
    ans_last_char = full.rfind("\n" + CONF_TAG)  # last char of answer text
    nl_pos = ans_last_char + 1                    # index of the "\n"
    conf_tag_end = full.rfind(CONF_TAG) + len(CONF_TAG)
    return {
        "pre":  char_to_token(tok, full, ans_head_pos),
        "ans":  char_to_token(tok, full, ans_last_char),
        "nl":   char_to_token(tok, full, nl_pos),
        "post": char_to_token(tok, full, conf_tag_end),
    }


def get_layer_modules(model):
    for path in [
        "model.language_model.layers",
        "model.layers",
        "transformer.h",
    ]:
        cur = model
        try:
            for p in path.split("."):
                cur = getattr(cur, p)
            if len(list(cur)) > 0:
                return list(cur), path
        except AttributeError:
            continue
    raise RuntimeError("could not locate layer modules")


class MultiPatchHook:
    """Replace resid[0, position] with source_vec on first pass only."""
    def __init__(self, position: int, source_vec: torch.Tensor):
        self.position = position
        self.source_vec = source_vec
        self.applied = False

    def __call__(self, module, inp, out):
        resid = out[0] if isinstance(out, tuple) else out
        if not self.applied and resid.shape[1] > self.position:
            resid[0, self.position] = self.source_vec.to(resid.dtype).to(resid.device)
            self.applied = True
        return out


@torch.no_grad()
def cache_residuals(model, tok, prompt: str, positions: dict, layers: list[int]) -> dict:
    """Run once on ``prompt``, capture residuals at (each layer, each position).
    Returns dict {name: {layer: tensor}}."""
    layer_mods, _ = get_layer_modules(model)
    stored: dict = {name: {} for name in positions}

    handles = []
    for L in layers:
        def make_hook(L=L):
            def hook(module, inp, out):
                resid = out[0] if isinstance(out, tuple) else out
                for name, pos in positions.items():
                    stored[name][L] = resid[0, pos].detach().clone()
                return out
            return hook
        handles.append(layer_mods[L].register_forward_hook(make_hook()))

    ids = tok(prompt, return_tensors="pt").to(model.device)
    try:
        model(**ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return stored


@torch.no_grad()
def generate_confidence(
    model, tok, prompt: str,
    patch_positions: dict | None = None,   # {layer: {position: source_vec}}
    max_new_tokens: int = 6,
) -> str:
    """Generate the confidence text with optional multi-layer patches applied on
    the first forward pass."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
    handles = []
    if patch_positions:
        layer_mods, _ = get_layer_modules(model)
        for L, pos_map in patch_positions.items():
            for pos, vec in pos_map.items():
                hook = MultiPatchHook(pos, vec)
                handles.append(layer_mods[L].register_forward_hook(hook))
    try:
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    finally:
        for h in handles:
            h.remove()
    new_tokens = out[0, ids["input_ids"].shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return text.split("\n", 1)[0].strip()


def parse_layer_range(s: str, n_layers: int) -> list[int]:
    """Parse '20-40' or '20-40:2' or '55' etc."""
    step = 1
    if ":" in s:
        s, step_s = s.split(":")
        step = int(step_s)
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1, step))
    return [int(s)]


def build_pairs(csv_path: Path, low_thresh: int, high_thresh: int, max_pairs: int):
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    highs = [r for r in rows if int(r["confidence"]) >= high_thresh]
    lows = [r for r in rows if int(r["confidence"]) <= low_thresh]
    print(f"[info] {len(highs)} high (>= {high_thresh}) and {len(lows)} low (<= {low_thresh})")
    rng = np.random.default_rng(0)
    n_pairs = min(max_pairs, min(len(highs), len(lows)))
    pairs = list(zip(rng.choice(highs, n_pairs, replace=False),
                     rng.choice(lows, n_pairs, replace=False)))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--generations_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="20-45", help="range of layers to patch (e.g. 20-45 or 20-45:1)")
    ap.add_argument("--low_thresh", type=int, default=60)
    ap.add_argument("--high_thresh", type=int, default=90)
    ap.add_argument("--max_pairs", type=int, default=30)
    args = ap.parse_args()

    model, tok = load_model(args.model_dir)
    layer_mods, layer_path = get_layer_modules(model)
    n_layers = len(layer_mods)
    layers = parse_layer_range(args.layers, n_layers)
    print(f"[info] layer_path={layer_path} n_layers={n_layers}  patch_layers={layers}")

    pairs = build_pairs(Path(args.generations_csv), args.low_thresh,
                        args.high_thresh, args.max_pairs)
    print(f"[info] {len(pairs)} pairs")

    rows = []
    for i, (hi_row, lo_row) in enumerate(pairs):
        q_hi, a_hi = hi_row["question"], hi_row["answer"]
        q_lo, a_lo = lo_row["question"], lo_row["answer"]
        c_hi, c_lo = int(hi_row["confidence"]), int(lo_row["confidence"])

        full_hi = build_full_prompt(q_hi, a_hi)
        full_lo = build_full_prompt(q_lo, a_lo)
        pos_hi = get_anchor_positions(tok, full_hi)
        pos_lo = get_anchor_positions(tok, full_lo)

        # 1) baseline
        c_base = parse_confidence(generate_confidence(model, tok, full_lo, None))

        # 2) cache HIGH residuals at all patch layers, at (pre, ans, post)
        src = cache_residuals(model, tok, full_hi, pos_hi, layers)

        # 3) patch experiments -- per position + multi-position "ans+nl"
        res = {}
        for pos_name in ("pre", "ans", "nl", "post"):
            patch_positions = {L: {pos_lo[pos_name]: src[pos_name][L]} for L in layers}
            c = parse_confidence(generate_confidence(model, tok, full_lo, patch_positions))
            res[pos_name] = c

        # combined: ans + nl (the whole "post-answer cache region")
        patch_positions = {
            L: {pos_lo["ans"]: src["ans"][L], pos_lo["nl"]: src["nl"][L]}
            for L in layers
        }
        res["ans_nl"] = parse_confidence(generate_confidence(model, tok, full_lo, patch_positions))

        row = {
            "hi_qid": hi_row["qid"], "lo_qid": lo_row["qid"],
            "c_hi": c_hi, "c_lo": c_lo,
            "c_base": c_base,
            "c_patch_pre":  res["pre"],
            "c_patch_ans":  res["ans"],
            "c_patch_nl":   res["nl"],
            "c_patch_post": res["post"],
            "c_patch_ans_nl": res["ans_nl"],
        }
        rows.append(row)
        if i % 3 == 0 or i == len(pairs) - 1:
            print(f"[{i+1}/{len(pairs)}] {row}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"layers": layers, "layer_path": layer_path,
                   "n_layers": n_layers, "rows": rows}, f, indent=2)
    print(f"[done] wrote {out_path}")

    # quick summary
    def none_ok(v): return v if v is not None else -1
    keys = ("c_base","c_patch_pre","c_patch_ans","c_patch_nl","c_patch_post","c_patch_ans_nl")
    valid_rows = [r for r in rows if all(r[k] is not None for k in keys)]
    print(f"[summary] {len(valid_rows)}/{len(rows)} rows valid")
    if valid_rows:
        def delta(k): return np.mean([(r[k] - r["c_base"]) for r in valid_rows])
        target = np.mean([(r["c_hi"] - r["c_base"]) for r in valid_rows])
        print(f"[summary] mean delta:  pre={delta('c_patch_pre'):+.2f}  "
              f"ans={delta('c_patch_ans'):+.2f}  nl={delta('c_patch_nl'):+.2f}  "
              f"ans_nl={delta('c_patch_ans_nl'):+.2f}  "
              f"post={delta('c_patch_post'):+.2f}   target(hi-lo)={target:+.2f}")
        def flips(k): return sum(1 for r in valid_rows if r[k] > r["c_base"])
        print(f"[summary] #upward-flip: pre={flips('c_patch_pre')}  ans={flips('c_patch_ans')}  "
              f"nl={flips('c_patch_nl')}  ans_nl={flips('c_patch_ans_nl')}  "
              f"post={flips('c_patch_post')}  N={len(valid_rows)}")


if __name__ == "__main__":
    main()
