"""Encode claims with SPECTER2 -- the encoder used by the paper.

Faithful to EmbedWork_AbstractTitle_Specter2.py in
tsinghua-fib-lab/AI-Impacts-Science:

  * ``AutoAdapterModel`` on specter2_base + the ``[PRX]`` proximity adapter,
    set active;
  * text = ``title.lower() + tokenizer.sep_token + abstract.lower()``;
  * ``max_length``, truncation, padding;
  * embedding = ``model(**tok)[0][:, 0, :]`` (CLS of the last hidden state),
    no L2 normalisation.

v1 change vs. the paper: ``--text-field`` defaults to
``title_hypothesis_abstract`` and ``--max-length`` to 512 (was 288).  512 is a
HARD ceiling here -- specter2_base is a BertModel with absolute position
embeddings and ``max_position_embeddings = 512``; asking for more raises an
index error.  With Title + Short Hypothesis + Abstract the texts run 500-890
tokens, so every item still truncates at the tail of its abstract.  The script
reports ``n_truncated`` and ``truncated_token_frac`` so that the truncation is
recorded in the output metadata.

Needs the ``adapters`` library.  Model locations are supplied on the command
line or through ``config.json``; no machine-specific paths are embedded here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from claims import load_claims  # noqa: E402


def read_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a JSON object")
    config_dir = os.path.dirname(os.path.abspath(path))
    for key in ("specter2_model", "specter2_adapter"):
        value = config.get(key)
        if value and not os.path.isabs(os.path.expanduser(value)):
            config[key] = os.path.join(config_dir, os.path.expanduser(value))
    return config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dir to scan recursively for claim.json")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--config", default=None,
                    help="optional JSON config containing SPECTER2 paths and defaults")
    ap.add_argument("--model-dir", default=None,
                    help="SPECTER2 base checkpoint; overrides config")
    ap.add_argument("--adapter-dir", default=None,
                    help="SPECTER2 proximity adapter; overrides config")
    ap.add_argument("--no-adapter", action="store_true",
                    help="use specter2_base alone (ablation; paper uses the adapter)")
    ap.add_argument("--text-field", default=None,
                    help="title_hypothesis_abstract (v1 default) | title_abstract "
                         "(paper default) | title | abstract | hypothesis | full | all")
    ap.add_argument("--max-length", type=int, default=None,
                    help="capped at the model's max_position_embeddings (512 for "
                         "specter2_base); larger values are clamped, not honoured")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    config = read_config(args.config)
    args.model_dir = args.model_dir or config.get("specter2_model")
    args.adapter_dir = args.adapter_dir or config.get("specter2_adapter")
    args.text_field = args.text_field or config.get("text_field", "title_hypothesis_abstract")
    args.max_length = args.max_length or int(config.get("max_length", 512))
    args.batch_size = args.batch_size or int(config.get("batch_size", 32))
    args.device = args.device or config.get("device", "auto")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.model_dir:
        raise SystemExit("SPECTER2 base path is required: set specter2_model in config.json or pass --model-dir")
    if not args.no_adapter and not args.adapter_dir:
        raise SystemExit("SPECTER2 adapter path is required: set specter2_adapter in config.json or pass --adapter-dir")

    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    claims = load_claims(args.input)
    print(f"[specter2] {len(claims)} claims from {args.input}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoAdapterModel.from_pretrained(args.model_dir)
    if not args.no_adapter:
        model.load_adapter(args.adapter_dir, load_as="specter2", set_active=True)
        print(f"[specter2] active adapter: {model.active_adapters}", flush=True)
    model.to(args.device)
    model.eval()

    # BertModel uses absolute position embeddings: max_length > max_position_embeddings
    # is not a slow path, it is an index error.  Clamp and say so.
    hard_cap = int(getattr(model.config, "max_position_embeddings", args.max_length))
    if args.max_length > hard_cap:
        print(f"[specter2] WARNING: --max-length {args.max_length} exceeds the model's "
              f"max_position_embeddings {hard_cap}; clamping to {hard_cap}", flush=True)
        args.max_length = hard_cap

    # paper joins the fields with the tokenizer's SEP token and lowercases
    texts = [tokenizer.sep_token.join(p.lower() for p in c.parts(args.text_field))
             for c in claims]

    out = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        with torch.no_grad():
            tok = tokenizer(batch, max_length=args.max_length, truncation=True,
                            padding=True, return_tensors="pt").to(args.device)
            out.append(model(**tok)[0][:, 0, :].float().cpu().numpy())
        print(f"[specter2] {min(i + args.batch_size, len(texts))}/{len(texts)}", flush=True)

    emb = np.vstack(out).astype(np.float32)
    full_lens = [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts]
    n_trunc = sum(l > args.max_length for l in full_lens)
    kept = sum(min(l, args.max_length) for l in full_lens)
    trunc_frac = 1.0 - kept / sum(full_lens)
    if n_trunc:
        print(f"[specter2] WARNING: {n_trunc}/{len(texts)} texts truncated at "
              f"{args.max_length} tokens; {trunc_frac:.1%} of all tokens dropped "
              f"(full-text lengths: min {min(full_lens)}, median "
              f"{sorted(full_lens)[len(full_lens) // 2]}, max {max(full_lens)})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {
        "encoder": "specter2_base" + ("" if args.no_adapter else "+proximity_adapter"),
        "model": os.path.basename(os.path.normpath(args.model_dir)),
        "adapter": None if args.no_adapter else os.path.basename(os.path.normpath(args.adapter_dir)),
        "text_field": args.text_field,
        "max_length": args.max_length,
        "pooling": "cls",
        "l2_normalized": False,
        "input": os.path.basename(os.path.normpath(args.input)),
        "n_items": len(claims),
        "dim": int(emb.shape[1]),
        "n_truncated": int(n_trunc),
        "truncated_token_frac": round(float(trunc_frac), 4),
        "full_len_max": int(max(full_lens)),
        "full_len_median": int(sorted(full_lens)[len(full_lens) // 2]),
    }
    np.savez(args.out, ids=np.array([c.cid for c in claims]), embeddings=emb,
             meta=json.dumps(meta))
    print(json.dumps(meta, indent=2))
    print(f"[specter2] saved {emb.shape} -> {args.out}")


if __name__ == "__main__":
    main()
