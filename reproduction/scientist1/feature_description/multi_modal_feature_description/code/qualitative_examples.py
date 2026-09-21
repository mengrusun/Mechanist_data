"""Qualitative case studies.

For a few sampled components per model, dump a table showing:
    - component id
    - text label picked by argmax_q cos(v_c, t_q)
    - top-K images' ImageNet class names (with duplicates)
    - class purity of top-K

Also dumps a "text->component" table: for a small vocabulary of query words,
show the retrieved component (top-1) and its top-K image labels.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from code.common import OUT_DIR, load_class_names

TEMPLATE = "a photo of a {}"
QUERY_WORDS = [
    "goldfish",
    "school bus",
    "dog",
    "cat",
    "wheel",
    "mushroom",
    "guitar",
    "keyboard",
    "clock",
    "eye",
    "text",
    "green field",
    "snow",
    "beach",
    "fire",
]
CASE_STUDY_COMPONENTS = 40


@torch.no_grad()
def encode_text_batch(texts, device="cuda"):
    clip = CLIPModel.from_pretrained("models/clip").eval().to(device)
    tok = CLIPTokenizer.from_pretrained("models/clip")
    all_emb = []
    for i in range(0, len(texts), 128):
        b = tok(texts[i : i + 128], padding=True, return_tensors="pt").to(device)
        out = clip.get_text_features(**b)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        emb = F.normalize(emb, dim=-1)
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0).astype(np.float32)


def load_component(tag):
    with np.load(OUT_DIR / "components" / f"{tag}.npz") as comp:
        v = comp["v"].astype(np.float32)
        topk_labels = comp["topk_labels"]
        topk_act = comp["topk_act"]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return v, topk_labels, topk_act


def component_case_study(tag, class_names, class_text_emb, out_lines, seed=0):
    v, topk_labels, topk_act = load_component(tag)
    C = v.shape[0]
    rng = np.random.default_rng(seed)
    comp_ids = rng.choice(C, size=CASE_STUDY_COMPONENTS, replace=False)
    comp_ids.sort()

    sims = v @ class_text_emb.T  # (C, 1000)
    argmax_q = sims.argmax(axis=1)  # (C,)
    top_sim = sims.max(axis=1)  # (C,)

    out_lines.append(f"\n## {tag} — 40 sampled components\n")
    out_lines.append("| Comp | Text label (best class match) | cos | Top-K dominant labels (count) | Class purity |")
    out_lines.append("|---:|---|---:|---|---:|")
    for c in comp_ids:
        lbls = topk_labels[c].tolist()
        cnt = Counter(lbls)
        dom = ", ".join(f"{class_names[l]} ({n})" for l, n in cnt.most_common(3))
        pur = max(cnt.values()) / len(lbls)
        out_lines.append(
            f"| {int(c)} | {class_names[int(argmax_q[c])]} | {float(top_sim[c]):.3f} | {dom} | {pur:.2f} |"
        )


def query_case_study(tag, class_names, query_text_emb, out_lines):
    v, topk_labels, topk_act = load_component(tag)
    sims = query_text_emb @ v.T  # (Q, C)
    top_comp = sims.argmax(axis=1)  # (Q,)
    top_sc = sims.max(axis=1)

    out_lines.append(f"\n## {tag} — text query -> nearest component\n")
    out_lines.append("| Query | Top-1 comp | cos | Top-K image labels (dominant) |")
    out_lines.append("|---|---:|---:|---|")
    for qi, q in enumerate(QUERY_WORDS):
        c = int(top_comp[qi])
        lbls = topk_labels[c].tolist()
        cnt = Counter(lbls)
        dom = ", ".join(f"{class_names[l]} ({n})" for l, n in cnt.most_common(3))
        out_lines.append(f"| \"{q}\" | {c} | {float(top_sc[qi]):.3f} | {dom} |")


def main():
    class_names = load_class_names()
    print("[qualitative] encoding class prompts")
    class_text_emb = encode_text_batch([TEMPLATE.format(n) for n in class_names])
    print("[qualitative] encoding query words")
    query_text_emb = encode_text_batch([TEMPLATE.format(q) for q in QUERY_WORDS])

    out_lines = ["# Qualitative examples\n"]
    for tag in ["resnet50_layer4", "vit_b16_cls", "vit_b16_mean"]:
        component_case_study(tag, class_names, class_text_emb, out_lines)
        query_case_study(tag, class_names, query_text_emb, out_lines)

    path = OUT_DIR / "qualitative_examples.md"
    path.write_text("\n".join(out_lines))
    print(f"[qualitative] saved -> {path}")


if __name__ == "__main__":
    main()
