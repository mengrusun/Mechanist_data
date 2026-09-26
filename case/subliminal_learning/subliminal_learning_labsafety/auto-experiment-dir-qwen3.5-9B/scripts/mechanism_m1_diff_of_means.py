#!/usr/bin/env python3
"""
M1 Location — diff-of-means direction extraction on residual-stream activations.

For a given arm (ctrl / treated_seed{42,200,201}):
  - Load base multimodal model (+ adapter if treated, merge_and_unload).
  - For each QA_I item in the fit split (80%):
      - Run image-conditioned forward pass, hook residual stream at target layers.
      - Cache last-token residual state per layer.
  - Save activations to mechanism/M1_location/<arm>/activations.pt.

For the ctrl arm, we also run inference on the held-out 20% split (M2 needs
Ctrl's held-out activations for patching).

Layers: {6, 9, 12, 16, 19, 22, 25} of Qwen3.5-9B's 32-layer language tower.

Output per arm:
  mechanism/M1_location/<arm>/activations_fit.pt   dict[layer_idx] -> (N_fit, hidden_size) tensor
  mechanism/M1_location/<arm>/activations_held.pt  (only for ctrl; required by M2 patching)
  mechanism/M1_location/<arm>/row_ids_fit.json     ordered list of QA_I row_ids
  mechanism/M1_location/<arm>/row_ids_held.json    (ctrl only)

After activation caching, a separate step (mechanism_m1_screen.py) computes
diff-of-means directions and rank layers by AUROC + |v|/σ.
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    assert_gpu_pool_ok, assert_no_device_map_auto, assert_student_load_class, assert_thinking_off,
)


DEFAULT_LAYERS = [6, 9, 12, 16, 19, 22, 25]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--merge_and_unload_if_treated", action="store_true", default=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--qa_i_split_seed", type=int, default=42)
    p.add_argument("--qa_i_fit_frac", type=float, default=0.80)
    p.add_argument("--qa_i_use_split", choices=["fit", "held_out", "both"], default="both")
    p.add_argument("--split_dir", default="results")
    p.add_argument("--sites", type=str, default=",".join(str(x) for x in DEFAULT_LAYERS))
    p.add_argument("--arm", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--enable_thinking", type=str, default="False")
    return p.parse_args()


def load_qa_i(parquet_path):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    records = []
    for i, row in df.iterrows():
        img_bytes = row["Decoded Image"]["bytes"] if isinstance(row["Decoded Image"], dict) else None
        records.append({
            "row_id": int(i),
            "question": str(row["Question"]),
            "gold_letter": str(row["Correct Answer"]).strip(),
            "image_bytes": img_bytes,
        })
    return records


def load_model_for_arm(base_model, adapter, merge_and_unload_if_treated):
    print(f"[m1] loading multimodal base on cuda:0 (bf16)")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0")
    assert_no_device_map_auto(model)
    assert_student_load_class(model)

    if adapter and adapter.lower() not in ("null", "none", ""):
        print(f"[m1] attaching adapter {adapter}")
        ap = adapter
        if ap.endswith(".safetensors"):
            ap = str(Path(ap).parent)
        model = PeftModel.from_pretrained(model, ap)
        if merge_and_unload_if_treated:
            print("[m1] merge_and_unload")
            model = model.merge_and_unload()
    model.eval()
    return model


def hook_language_layer(model, layer_idx):
    """Register a forward hook on model.language_model.layers[layer_idx] that captures the
    OUTPUT residual state at the last token position. Returns (handle, list_to_store)."""
    lm = model.language_model if hasattr(model, "language_model") else model.model.language_model
    layer = lm.layers[layer_idx]
    storage = []

    def hook(module, args, output):
        # For Qwen3.5, decoder layer output is a tuple (hidden_states,) or (hidden_states, ...)
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # Last-token position, batch 1 → shape (1, hidden_size)
        storage.append(h[:, -1, :].detach().float().cpu())

    handle = layer.register_forward_hook(hook)
    return handle, storage


def main():
    args = parse_args()
    assert_gpu_pool_ok()

    layers = [int(x) for x in args.sites.split(",")]

    # Load QA_I + frozen split
    records = load_qa_i(args.benchmark)
    with open(Path(args.split_dir) / "qa_i_split.json", "r") as f:
        split = json.load(f)
    fit_ids = set(split["fit_ids"])
    held_ids = set(split["held_out_ids"])
    row_by_id = {r["row_id"]: r for r in records}

    which_ids_list = []
    if args.qa_i_use_split in ("fit", "both"):
        which_ids_list.append(("fit", sorted(fit_ids)))
    if args.qa_i_use_split in ("held_out", "both"):
        which_ids_list.append(("held_out", sorted(held_ids)))

    # Load processor + model
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert_thinking_off(rendered)

    model = load_model_for_arm(args.base_model, args.adapter, args.merge_and_unload_if_treated)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, id_list in which_ids_list:
        print(f"[m1] arm={args.arm} split={split_name} n={len(id_list)}")
        # Storage: dict[layer_idx] -> list of (1, hidden) tensors, one per item
        per_layer = {L: [] for L in layers}
        row_id_order = []

        # For each item, register hooks on all layers, run forward, save.
        for i, rid in enumerate(id_list):
            rec = row_by_id[rid]
            img = Image.open(io.BytesIO(rec["image_bytes"])).convert("RGB")
            user_content = [
                {"type": "image", "image": img},
                {"type": "text", "text": rec["question"].strip() + "\n\nBased on the image and the options above, answer with the letter (A, B, C, or D) of the correct option, followed by a brief explanation."},
            ]
            messages = [{"role": "user", "content": user_content}]

            try:
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt", enable_thinking=False,
                )
            except TypeError:
                prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                inputs = processor(text=prompt_str, images=[img], return_tensors="pt")

            inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}

            handles = []
            storages = {}
            for L in layers:
                h, s = hook_language_layer(model, L)
                handles.append(h)
                storages[L] = s

            try:
                with torch.no_grad():
                    _ = model(**inputs, use_cache=False)
            finally:
                for h in handles:
                    h.remove()

            for L in layers:
                # storages[L] is a list of one tensor (1, hidden) from the single forward pass.
                # Take the last-token pos (already done in hook).
                if storages[L]:
                    per_layer[L].append(storages[L][0])
                else:
                    print(f"[m1] WARN: no activation captured at layer {L} for row {rid}")

            row_id_order.append(rid)

            if (i + 1) % 20 == 0:
                print(f"[m1] arm={args.arm} split={split_name} {i+1}/{len(id_list)}")

        # Stack and save
        activations = {L: torch.cat(per_layer[L], dim=0) for L in layers}  # (N, hidden)
        act_path = out_dir / f"activations_{split_name}.pt"
        torch.save(activations, act_path)
        ids_path = out_dir / f"row_ids_{split_name}.json"
        with open(ids_path, "w") as f:
            json.dump(row_id_order, f, indent=2)
        print(f"[m1] wrote {act_path} — {len(row_id_order)} items × {len(layers)} layers")

    print(f"[m1] arm={args.arm} done")


if __name__ == "__main__":
    sys.exit(main() or 0)
