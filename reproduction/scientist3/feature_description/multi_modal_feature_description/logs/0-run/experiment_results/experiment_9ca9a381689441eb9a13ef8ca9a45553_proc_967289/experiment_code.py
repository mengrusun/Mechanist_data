# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, io, glob, json, math, random, time, sys, traceback
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"


# ---------- Locate resources ----------
def find_dir(candidates):
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


resnet_dir = find_dir([f"{MODEL_DIR}/resnet-50", f"{MODEL_DIR}/microsoft/resnet-50"])
clip_dir = find_dir(
    [
        f"{MODEL_DIR}/clip-vit-base-patch32",
        f"{MODEL_DIR}/clip-vit-base-patch16",
        f"{MODEL_DIR}/openai/clip-vit-base-patch32",
        f"{MODEL_DIR}/openai/clip-vit-base-patch16",
    ]
)
assert resnet_dir is not None, f"ResNet-50 dir not found under {MODEL_DIR}"
assert clip_dir is not None, f"CLIP dir not found under {MODEL_DIR}"
print(f"ResNet-50: {resnet_dir}")
print(f"CLIP: {clip_dir}")

# ImageNet val parquet
imagenet_val_dir = f"{DATA_DIR}/imagenet-val/data"
parquet_files = sorted(glob.glob(f"{imagenet_val_dir}/*.parquet"))
assert len(parquet_files) > 0, f"No parquet files found in {imagenet_val_dir}"
print(f"Found {len(parquet_files)} parquet files.")

# ---------- Load probe dataset ----------
from datasets import load_dataset

ds = load_dataset("parquet", data_files=parquet_files, split="train")
print("Dataset:", ds)
print("Columns:", ds.column_names)

# Identify image column
img_col = None
for c in ds.column_names:
    if "image" in c.lower() or "img" in c.lower():
        img_col = c
        break
if img_col is None:
    img_col = ds.column_names[0]
print(f"Using image column: {img_col}")

# ---------- Preprocessing (manual) ----------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def pil_to_tensor(pil, size=224, resize=256):
    img = pil.convert("RGB")
    w, h = img.size
    scale = resize / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left = (nw - size) // 2
    top = (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t


def normalize_imagenet(t):
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def normalize_clip(t):
    return (t - CLIP_MEAN) / CLIP_STD


class ProbeDataset(Dataset):
    def __init__(self, hf_ds, img_col, n=None):
        self.ds = hf_ds
        self.img_col = img_col
        self.n = n if n is not None else len(hf_ds)

    def __len__(self):
        return min(self.n, len(self.ds))

    def __getitem__(self, i):
        row = self.ds[i]
        img = row[self.img_col]
        if isinstance(img, dict) and "bytes" in img and img["bytes"] is not None:
            pil = Image.open(io.BytesIO(img["bytes"]))
        elif isinstance(img, Image.Image):
            pil = img
        else:
            pil = Image.open(io.BytesIO(img))
        t = pil_to_tensor(pil, 224, 256)
        return i, t  # return raw [0,1] tensor; normalize per-model later


# ---------- Load models ----------
from transformers import (
    AutoModel,
    AutoModelForImageClassification,
    CLIPModel,
    CLIPVisionModel,
)

# ResNet-50
try:
    rn_model = AutoModelForImageClassification.from_pretrained(
        resnet_dir, local_files_only=True
    )
except Exception as e:
    print("AutoModelForImageClassification failed, trying AutoModel:", e)
    rn_model = AutoModel.from_pretrained(resnet_dir, local_files_only=True)
rn_model.eval().to(device)


# Find last conv-stage output for hooking. HF resnet-50 has model.resnet.encoder.stages[-1]
def get_layer4(model):
    # Try common attribute paths
    candidates = []
    if hasattr(model, "resnet"):
        candidates.append(model.resnet.encoder.stages[-1])
    if hasattr(model, "encoder") and hasattr(model.encoder, "stages"):
        candidates.append(model.encoder.stages[-1])
    return candidates[0] if candidates else None


layer4 = get_layer4(rn_model)
assert layer4 is not None, "Could not locate layer4/last stage of ResNet-50"
print("Hooking layer:", type(layer4).__name__)

_activation_holder = {}


def _hook(module, inp, out):
    # out shape: [B, C, H, W]
    if isinstance(out, tuple):
        out = out[0]
    _activation_holder["feat"] = out.detach()


handle = layer4.register_forward_hook(_hook)

# CLIP
try:
    clip_model = CLIPModel.from_pretrained(clip_dir, local_files_only=True)
    clip_kind = "full"
except Exception as e:
    print("CLIPModel failed, using CLIPVisionModel:", e)
    clip_model = CLIPVisionModel.from_pretrained(clip_dir, local_files_only=True)
    clip_kind = "vision"
clip_model.eval().to(device)


def clip_embed(pixel_values):
    # pixel_values already CLIP-normalized, 224x224
    with torch.no_grad():
        if clip_kind == "full":
            try:
                out = clip_model.get_image_features(pixel_values=pixel_values)
                if not torch.is_tensor(out):
                    raise RuntimeError("get_image_features returned non-tensor")
            except Exception:
                vis_out = clip_model.vision_model(pixel_values=pixel_values)
                pooled = getattr(vis_out, "pooler_output", None)
                if pooled is None:
                    pooled = vis_out[1]
                out = clip_model.visual_projection(pooled)
        else:
            vis_out = clip_model(pixel_values=pixel_values)
            pooled = getattr(vis_out, "pooler_output", vis_out[1])
            out = pooled
    return F.normalize(out.float(), dim=-1)


# ---------- Preflight smoke test ----------
print("--- Smoke test ---")
with torch.no_grad():
    sample_imgs = []
    for i in range(2):
        row = ds[i]
        img_field = row[img_col]
        if (
            isinstance(img_field, dict)
            and "bytes" in img_field
            and img_field["bytes"] is not None
        ):
            pil = Image.open(io.BytesIO(img_field["bytes"]))
        elif isinstance(img_field, Image.Image):
            pil = img_field
        else:
            pil = Image.open(io.BytesIO(img_field))
        sample_imgs.append(pil_to_tensor(pil, 224, 256))
    batch = torch.stack(sample_imgs, dim=0)
    rn_in = normalize_imagenet(batch).to(device)
    _ = (
        rn_model(pixel_values=rn_in)
        if "pixel_values" in rn_model.forward.__code__.co_varnames
        else rn_model(rn_in)
    )
    feat = _activation_holder["feat"]
    print("layer4 feat shape:", tuple(feat.shape))
    clip_in = normalize_clip(batch).to(device)
    emb = clip_embed(clip_in)
    print("CLIP emb shape:", tuple(emb.shape))
print("--- Smoke test passed ---")

# ---------- Configuration ----------
N_PROBE = 2000  # number of probe images
N_CHANNELS = 64  # number of components to inspect (subset of 2048)
TOP_K = 8  # reference inputs per component
BATCH = 64
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Choose channels to inspect (random subset)
NUM_TOTAL_CHANNELS = None  # determined from smoke test
NUM_TOTAL_CHANNELS = _activation_holder["feat"].shape[1]
print(f"Total channels in layer4 output: {NUM_TOTAL_CHANNELS}")
inspected_channels = sorted(random.sample(range(NUM_TOTAL_CHANNELS), N_CHANNELS))

# ---------- Pass 1: compute activations for all probe images ----------
probe_ds = ProbeDataset(ds, img_col, n=N_PROBE)
loader = DataLoader(
    probe_ds, batch_size=BATCH, num_workers=4, shuffle=False, pin_memory=True
)

activations = np.zeros((len(probe_ds), N_CHANNELS), dtype=np.float32)
indices_all = np.zeros(len(probe_ds), dtype=np.int64)

t0 = time.time()
with torch.no_grad():
    ptr = 0
    for idx, batch in loader:
        bs = batch.shape[0]
        x = normalize_imagenet(batch).to(device, non_blocking=True)
        try:
            _ = rn_model(pixel_values=x)
        except TypeError:
            _ = rn_model(x)
        feat = _activation_holder["feat"]  # [B,C,H,W]
        # global average pool
        pooled = feat.mean(dim=[2, 3])  # [B,C]
        pooled = pooled[:, inspected_channels].cpu().numpy()
        activations[ptr : ptr + bs] = pooled
        indices_all[ptr : ptr + bs] = idx.numpy()
        ptr += bs
        if ptr % (BATCH * 5) == 0:
            print(f"  activation pass: {ptr}/{len(probe_ds)}  ({time.time()-t0:.1f}s)")
print(f"Activation extraction done in {time.time()-t0:.1f}s")

# ---------- Pick top-k images per channel ----------
top_k_per_channel = {}  # channel_idx (position) -> list of dataset indices
for j in range(N_CHANNELS):
    order = np.argsort(-activations[:, j])
    top_positions = order[:TOP_K]
    top_ds_idx = indices_all[top_positions].tolist()
    top_k_per_channel[j] = top_ds_idx

# Collect all unique needed images
needed_ds_indices = sorted({i for lst in top_k_per_channel.values() for i in lst})
print(f"Need to embed {len(needed_ds_indices)} unique reference images with CLIP.")

# ---------- Embed with CLIP ----------
idx_to_emb = {}
buf_idx, buf_img = [], []


def flush_clip():
    if not buf_img:
        return
    x = torch.stack(buf_img, dim=0)
    x = normalize_clip(x).to(device, non_blocking=True)
    emb = clip_embed(x).cpu().numpy()
    for k, di in enumerate(buf_idx):
        idx_to_emb[di] = emb[k]
    buf_idx.clear()
    buf_img.clear()


t0 = time.time()
for di in needed_ds_indices:
    row = ds[int(di)]
    img_field = row[img_col]
    if (
        isinstance(img_field, dict)
        and "bytes" in img_field
        and img_field["bytes"] is not None
    ):
        pil = Image.open(io.BytesIO(img_field["bytes"]))
    elif isinstance(img_field, Image.Image):
        pil = img_field
    else:
        pil = Image.open(io.BytesIO(img_field))
    t = pil_to_tensor(pil, 224, 256)
    buf_idx.append(int(di))
    buf_img.append(t)
    if len(buf_img) >= BATCH:
        flush_clip()
flush_clip()
print(f"CLIP embedding done in {time.time()-t0:.1f}s")

emb_dim = next(iter(idx_to_emb.values())).shape[0]
print(f"CLIP embedding dim: {emb_dim}")


# ---------- Concept consistency per channel ----------
def mean_pairwise_cos(vecs):
    # vecs: [k, d], assumed unit-normalized
    S = vecs @ vecs.T
    k = vecs.shape[0]
    iu = np.triu_indices(k, k=1)
    return float(S[iu].mean())


concept_scores = np.zeros(N_CHANNELS, dtype=np.float32)
v_c = np.zeros((N_CHANNELS, emb_dim), dtype=np.float32)
for j in range(N_CHANNELS):
    vecs = np.stack([idx_to_emb[i] for i in top_k_per_channel[j]], axis=0)
    concept_scores[j] = mean_pairwise_cos(vecs)
    pooled = vecs.mean(axis=0)
    pooled /= np.linalg.norm(pooled) + 1e-8
    v_c[j] = pooled

# ---------- Random baseline ----------
all_emb_indices = list(idx_to_emb.keys())
# Ensure random baseline uses same pool size roughly; sample TOP_K per pseudo-channel
rng = np.random.RandomState(123)
random_scores = np.zeros(N_CHANNELS, dtype=np.float32)
for j in range(N_CHANNELS):
    pick = rng.choice(len(all_emb_indices), size=TOP_K, replace=False)
    vecs = np.stack([idx_to_emb[all_emb_indices[p]] for p in pick], axis=0)
    random_scores[j] = mean_pairwise_cos(vecs)

mean_concept = float(concept_scores.mean())
median_concept = float(np.median(concept_scores))
mean_random = float(random_scores.mean())
delta = mean_concept - mean_random

print("==================================================")
print(f"concept_consistency_score (mean) = {mean_concept:.4f}")
print(f"concept_consistency_score (median) = {median_concept:.4f}")
print(f"random baseline (mean)           = {mean_random:.4f}")
print(f"delta (concept - random)         = {delta:.4f}")
print("==================================================")

# ---------- Visualization ----------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(concept_scores, bins=20, alpha=0.7, label="top-k activating")
    ax.hist(random_scores, bins=20, alpha=0.7, label="random baseline")
    ax.set_xlabel("mean pairwise CLIP cos sim")
    ax.set_ylabel("# channels")
    ax.set_title("Concept consistency: ResNet-50 layer4 channels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(working_dir, "concept_consistency_hist_resnet50_imagenetval.png"),
        dpi=140,
    )
    plt.close(fig)
except Exception as e:
    print("Plotting failed:", e)

# ---------- Save all experiment data ----------
experiment_data = {
    "imagenet_val_resnet50": {
        "metrics": {
            "train": [],
            "val": [
                {
                    "epoch": 0,
                    "concept_consistency_score_mean": mean_concept,
                    "concept_consistency_score_median": median_concept,
                    "random_baseline_mean": mean_random,
                    "delta": delta,
                }
            ],
        },
        "losses": {"train": [], "val": []},
        "predictions": concept_scores,
        "ground_truth": random_scores,
        "inspected_channels": np.array(inspected_channels, dtype=np.int64),
        "v_c": v_c,
        "top_k_per_channel": top_k_per_channel,
        "config": {
            "N_PROBE": N_PROBE,
            "N_CHANNELS": N_CHANNELS,
            "TOP_K": TOP_K,
            "model": "resnet-50",
            "dataset": "imagenet-val",
            "foundation": os.path.basename(clip_dir),
        },
    }
}
print(
    f"Epoch 0: validation_loss = {1.0 - mean_concept:.4f}  (proxy: 1 - concept_consistency_score)"
)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")

handle.remove()
