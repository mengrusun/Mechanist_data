"""M1.L0 — cache pinned-site residuals per (item, layer, arm, seed).

For each (arm ∈ {treated, Ctrl-B}, seed ∈ {42, 123, 2026}) we:
1. Load the student + LoRA adapter for that arm+seed.
2. For every QA_I item, run the multimodal forward with output_hidden_states=True.
3. Extract the residual state at the LAST INPUT TOKEN of each layer, all 32
   language-tower layers (post-attn+MLP sum, i.e., hidden_states[l] for l=1..32,
   where hidden_states[0] is embed).
4. Save all items × layers into a single .npz per (arm, seed).

Cache format (per arm-seed):
  cache/residuals/{arm}/seed{s}/activations.npz:
    hidden_states: [n_items, n_layers+1, d_model] bf16 -> float16
    item_indices: [n_items] int32 (== QA_I row index 0..132)
    labels: [n_items] str (gold letter A/B/C/D)
    arm: str
    seed: int

We keep all items (n=133 << the plan cap of 4000). Downstream l_core.py will
subset to flipped-wrong ∪ matched-agree ids from the eval files.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import (
    PROJECT_ROOT, DATA_ROOT,
    load_processor, load_student_multimodal, set_seed,
)


def load_qa_items():
    import pandas as pd
    df = pd.read_parquet(DATA_ROOT / "QA_I-00000-of-00001.parquet")
    items = []
    for i, row in df.iterrows():
        img_dict = row["Decoded Image"]
        raw = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        items.append({
            "id": int(i),
            "question": row["Question"],
            "gold": str(row["Correct Answer"]).strip().upper(),
            "image": img,
        })
    return items


def render_multimodal_prompt(processor, question, image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",
                 "text": question + "\n\nAnswer the question based on the image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--arm", required=True, choices=["treated", "Ctrl-B"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True,
                    help="Output npz path (single file per arm+seed).")
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"[cache-act arm={args.arm} seed={args.seed}] loading model", flush=True)
    model = load_student_multimodal()
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = load_processor()

    items = load_qa_items()
    print(f"[cache-act] loaded {len(items)} QA_I items", flush=True)

    # Probe once for shape.
    inputs = render_multimodal_prompt(processor, items[0]["question"], items[0]["image"])
    inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True, return_dict=True,
                    use_cache=False)
    n_layers_plus1 = len(out.hidden_states)
    d_model = out.hidden_states[0].shape[-1]
    print(f"[cache-act] n_layers+1={n_layers_plus1}  d_model={d_model}", flush=True)

    # Allocate cache tensor. Use float16 for storage (bf16 npz is not natively
    # supported by numpy; float16 has enough precision for direction extraction).
    n_items = len(items)
    cache = np.zeros((n_items, n_layers_plus1, d_model), dtype=np.float16)
    ids = np.zeros((n_items,), dtype=np.int32)
    labels = []

    for i, it in enumerate(items):
        inputs = render_multimodal_prompt(processor, it["question"], it["image"])
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, return_dict=True,
                        use_cache=False)
        # Pool at last input token. hidden_states is a tuple of tensors
        # [batch=1, seq, d_model].
        for l, h in enumerate(out.hidden_states):
            cache[i, l, :] = h[0, -1, :].to(torch.float16).cpu().numpy()
        ids[i] = it["id"]
        labels.append(it["gold"])
        if (i + 1) % 20 == 0:
            print(f"[cache-act] {i+1}/{n_items}", flush=True)
        del out

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        hidden_states=cache,
        item_indices=ids,
        labels=np.array(labels, dtype=object),
        arm=args.arm,
        seed=args.seed,
    )
    print(f"[cache-act arm={args.arm} seed={args.seed}] wrote {args.out} "
          f"shape={cache.shape}", flush=True)


if __name__ == "__main__":
    main()
