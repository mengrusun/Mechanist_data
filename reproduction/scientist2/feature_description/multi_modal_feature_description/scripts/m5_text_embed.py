"""
M5_text_embeddings — CLIP text embeddings for concept vocabularies.

Two vocabularies:
  (a) ImageNet-1k class names (1000 words) — for C1 last-layer P1a and C2 P2a/P2c.
  (b) Broden concepts (~1200 words) — for C1 hidden-layer P1b (matched-control gap).

Prompt ensemble: OpenAI 7-template (from the original CLIP paper).
Each concept -> mean(L2-normalize(text_encoder(template))) -> L2-normalize.

Output HDF5:
    /imagenet1k_classes/words          (1000,) S64  — class names
    /imagenet1k_classes/embeddings     (1000, 512) fp32 — L2-normalized
    /broden_concepts/words             (~1200,) S64
    /broden_concepts/embeddings        (~1200, 512) fp32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import h5py

from common import load_imagenet_class_names, load_openai_clip, set_all_seeds


# OpenAI CLIP 7-template ensemble (from the CLIP paper's zero-shot ImageNet eval).
OPENAI_7_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
]

# Broden concepts: build a simple curated open-vocabulary list of visual concepts.
# We don't have the Broden dataset dictionary on disk, so we assemble a ~1200-word
# open vocabulary matched to what appears in ImageNet + common visual concepts
# (colors, textures, materials, parts, scenes).
def build_broden_style_vocab(imagenet_names: List[str]) -> List[str]:
    """Union of ImageNet class names + colors + textures + materials + parts + scenes."""
    colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white",
              "gray", "cyan", "magenta", "beige", "gold", "silver"]
    textures = ["striped", "spotted", "checkered", "polka dot", "smooth", "rough", "wrinkled", "furry",
                "hairy", "shiny", "matte", "textured", "grainy", "wavy", "curved", "straight", "porous",
                "scaly", "leathery", "woven", "knitted", "braided", "pleated", "quilted", "veined"]
    materials = ["wood", "metal", "plastic", "glass", "stone", "leather", "cotton", "wool", "silk",
                 "denim", "rubber", "paper", "ceramic", "concrete", "clay", "marble", "brick",
                 "steel", "iron", "copper", "aluminum", "brass", "bronze", "titanium", "fabric"]
    parts = ["wheel", "handle", "door", "window", "roof", "leg", "arm", "head", "tail", "wing", "eye",
             "nose", "mouth", "ear", "hand", "foot", "hair", "fur", "beak", "claw", "horn", "antler",
             "trunk", "petal", "leaf", "stem", "root", "branch", "flower", "seed", "fruit", "screen",
             "keyboard", "button", "lens", "blade", "spoke", "hinge"]
    scenes = ["forest", "beach", "mountain", "desert", "ocean", "lake", "river", "jungle", "field",
              "meadow", "cave", "canyon", "cliff", "waterfall", "sunset", "sunrise", "sky", "cloud",
              "snow", "ice", "grass", "sand", "rock", "kitchen", "bedroom", "bathroom", "office",
              "street", "highway", "bridge", "tunnel", "park", "garden", "stadium", "airport",
              "harbor", "market", "restaurant", "school", "hospital", "church", "castle", "temple",
              "farm", "barn", "warehouse", "factory", "playground", "gym", "library", "museum",
              "theater"]
    actions = ["running", "jumping", "swimming", "flying", "climbing", "walking", "sitting", "standing",
               "sleeping", "eating", "drinking", "playing", "reading", "writing", "cooking",
               "dancing", "singing", "riding", "driving", "cycling", "hunting", "fishing", "hiking",
               "surfing", "skiing"]
    animals_extra = ["puppy", "kitten", "chick", "calf", "foal", "cub", "fawn", "piglet", "lamb",
                     "duckling", "gosling", "tadpole"]
    misc = ["face", "portrait", "close-up", "silhouette", "shadow", "reflection", "pattern", "logo",
            "text", "sign", "poster", "flag", "banner", "map", "chart", "graph", "diagram",
            "photograph", "painting", "drawing", "sketch", "print", "engraving"]

    vocab = list(dict.fromkeys(
        imagenet_names + colors + textures + materials + parts + scenes + actions + animals_extra + misc
    ))
    # Broden itself contains many concepts; ~1200 is a reasonable, well-covered target.
    return vocab


def prompt_ensemble_embed(clip_model, tokenizer, words: List[str], device: str,
                          templates: List[str] = OPENAI_7_TEMPLATES,
                          batch_size: int = 512) -> np.ndarray:
    """For each word: average of L2-normalized text embeddings over templates,
    then L2-normalize. Returns (N, D) float32."""
    out = np.zeros((len(words), 512), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i + batch_size]
            # Sum embeddings across templates
            e_sum = None
            for tmpl in templates:
                prompts = [tmpl.format(w) for w in batch_words]
                tokens = tokenizer(prompts).to(device)
                e = clip_model.encode_text(tokens)  # (B, D) fp16 on CUDA
                e = F.normalize(e.float(), dim=-1)
                e_sum = e if e_sum is None else e_sum + e
            e_mean = e_sum / len(templates)
            e_mean = F.normalize(e_mean, dim=-1)
            out[i:i + len(batch_words)] = e_mean.cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/M5_text_embeddings/text_embeddings.h5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_all_seeds(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[M5] loading CLIP ...")
    model, preprocess, tokenizer = load_openai_clip(device=args.device)

    imagenet_names = load_imagenet_class_names(clean=True)
    print(f"[M5] embedding {len(imagenet_names)} ImageNet class names ...")
    inet_emb = prompt_ensemble_embed(model, tokenizer, imagenet_names, device=args.device)

    broden_vocab = build_broden_style_vocab(imagenet_names)
    print(f"[M5] embedding {len(broden_vocab)} broden-style concepts (contains ImageNet + colors/textures/materials/parts/scenes/actions/misc) ...")
    broden_emb = prompt_ensemble_embed(model, tokenizer, broden_vocab, device=args.device)

    if out_path.exists():
        out_path.unlink()
    with h5py.File(out_path, "w") as h5o:
        g1 = h5o.create_group("imagenet1k_classes")
        g1.create_dataset("words", data=np.array(imagenet_names, dtype="S64"))
        g1.create_dataset("embeddings", data=inet_emb)
        g2 = h5o.create_group("broden_concepts")
        g2.create_dataset("words", data=np.array(broden_vocab, dtype="S64"))
        g2.create_dataset("embeddings", data=broden_emb)
        h5o.attrs["prompt_ensemble"] = "openai_7_selection"
        h5o.attrs["prompt_ensemble_note"] = (
            "7 templates selected from the OpenAI CLIP ImageNet zero-shot ensemble (Radford et al., 2021 supplementary). "
            "The full public ensemble is 80 templates; the 7-template subset is a compute-efficient prompt-ensemble variant "
            "used by CLIP-Dissect (Oikarinen & Weng, 2023). This is not 'the OpenAI 7-template ensemble' — no such canonical "
            "set of 7 exists — but a documented, reproducible subset."
        )
        h5o.attrs["templates"] = "\n".join(OPENAI_7_TEMPLATES)
        h5o.attrs["clip_arch"] = "ViT-B/32"
        h5o.attrs["dim"] = 512
        h5o.attrs["broden_concepts_note"] = (
            "This vocabulary is a curated 'broden-style' open concept vocabulary — not the original Broden dataset "
            "(Bau et al., 2017), which was not accessible at experiment time. It is the union of the 1000 ImageNet "
            "class names + colors + textures + materials + parts + scenes + actions + misc visual concepts (~1200 words). "
            "Used as the open-vocab lexicon for C1 hidden-layer P1b matched-control tests."
        )

    print(f"[M5] wrote {out_path}: imagenet={len(imagenet_names)}, broden={len(broden_vocab)}")
    # Save vocab file for reproducibility
    vocab_dir = Path("data/vocab")
    vocab_dir.mkdir(parents=True, exist_ok=True)
    (vocab_dir / "imagenet1k_classes.txt").write_text("\n".join(imagenet_names))
    (vocab_dir / "broden_concepts.txt").write_text("\n".join(broden_vocab))


if __name__ == "__main__":
    main()
