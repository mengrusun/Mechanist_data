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
assert resnet_dir is not None
assert clip_dir is not None
print(f"ResNet-50: {resnet_dir}")
print(f"CLIP: {clip_dir}")

imagenet_val_dir = f"{DATA_DIR}/imagenet-val/data"
parquet_files = sorted(glob.glob(f"{imagenet_val_dir}/*.parquet"))
assert len(parquet_files) > 0
print(f"Found {len(parquet_files)} parquet files.")

from datasets import load_dataset

ds = load_dataset("parquet", data_files=parquet_files, split="train")
print("Dataset:", ds)

img_col = None
for c in ds.column_names:
    if "image" in c.lower() or "img" in c.lower():
        img_col = c
        break
if img_col is None:
    img_col = ds.column_names[0]
print(f"Using image column: {img_col}")

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
    return torch.from_numpy(arr).permute(2, 0, 1)


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
        return i, pil_to_tensor(pil, 224, 256)


from transformers import (
    AutoModel,
    AutoModelForImageClassification,
    CLIPModel,
    CLIPVisionModel,
)

try:
    rn_model = AutoModelForImageClassification.from_pretrained(
        resnet_dir, local_files_only=True
    )
except Exception as e:
    print("AutoModelForImageClassification failed, trying AutoModel:", e)
    rn_model = AutoModel.from_pretrained(resnet_dir, local_files_only=True)
rn_model.eval().to(device)


def get_layer4(model):
    if hasattr(model, "resnet"):
        return model.resnet.encoder.stages[-1]
    if hasattr(model, "encoder") and hasattr(model.encoder, "stages"):
        return model.encoder.stages[-1]
    return None


layer4 = get_layer4(rn_model)
assert layer4 is not None
print("Hooking layer:", type(layer4).__name__)

_activation_holder = {}


def _hook(module, inp, out):
    if isinstance(out, tuple):
        out = out[0]
    _activation_holder["feat"] = out.detach()


handle = layer4.register_forward_hook(_hook)

try:
    clip_model = CLIPModel.from_pretrained(clip_dir, local_files_only=True)
    clip_kind = "full"
except Exception as e:
    print("CLIPModel failed, using CLIPVisionModel:", e)
    clip_model = CLIPVisionModel.from_pretrained(clip_dir, local_files_only=True)
    clip_kind = "vision"
clip_model.eval().to(device)


def clip_embed(pixel_values):
    with torch.no_grad():
        if clip_kind == "full":
            try:
                out = clip_model.get_image_features(pixel_values=pixel_values)
                if not torch.is_tensor(out):
                    raise RuntimeError("non-tensor")
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


# Smoke test
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
    try:
        _ = rn_model(pixel_values=rn_in)
    except TypeError:
        _ = rn_model(rn_in)
    feat = _activation_holder["feat"]
    print("layer4 feat shape:", tuple(feat.shape))
    clip_in = normalize_clip(batch).to(device)
    emb = clip_embed(clip_in)
    print("CLIP emb shape:", tuple(emb.shape))
print("--- Smoke test passed ---")

# ---------- Configuration ----------
DATASET_SIZE = len(ds)
N_PROBE = min(8000, DATASET_SIZE)
TOP_K_VALUES = [4, 8, 16, 32]
TOP_K_MAX = max(TOP_K_VALUES)
N_CHANNELS = 64
BATCH = 64
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_TOTAL_CHANNELS = _activation_holder["feat"].shape[1]
print(f"Total channels: {NUM_TOTAL_CHANNELS}")
inspected_channels = sorted(random.sample(range(NUM_TOTAL_CHANNELS), N_CHANNELS))

# ---------- Extract activations for probe pool ----------
probe_ds = ProbeDataset(ds, img_col, n=N_PROBE)
loader = DataLoader(
    probe_ds, batch_size=BATCH, num_workers=4, shuffle=False, pin_memory=True
)

activations_all = np.zeros((N_PROBE, N_CHANNELS), dtype=np.float32)
indices_all_full = np.zeros(N_PROBE, dtype=np.int64)

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
        feat = _activation_holder["feat"]
        pooled = feat.mean(dim=[2, 3])[:, inspected_channels].cpu().numpy()
        activations_all[ptr : ptr + bs] = pooled
        indices_all_full[ptr : ptr + bs] = idx.numpy()
        ptr += bs
        if ptr % (BATCH * 10) == 0:
            print(f"  activation pass: {ptr}/{N_PROBE}  ({time.time()-t0:.1f}s)")
print(f"Activation extraction done in {time.time()-t0:.1f}s")

# ---------- Compute top-K (using TOP_K_MAX, then slice) for each channel ----------
top_k_per_channel_max = {}
all_needed = set()
for j in range(N_CHANNELS):
    order = np.argsort(-activations_all[:, j])[:TOP_K_MAX]
    top_k_per_channel_max[j] = indices_all_full[order].tolist()
    all_needed.update(top_k_per_channel_max[j])

# Also add extra random pool indices so random baseline at TOP_K_MAX has enough
rng_pool = np.random.RandomState(42)
pool_extra_size = max(TOP_K_MAX * 8, 256)
extra_random_ids = rng_pool.choice(
    N_PROBE, size=min(pool_extra_size, N_PROBE), replace=False
)
random_pool_ids = indices_all_full[extra_random_ids].tolist()
all_needed.update(random_pool_ids)

needed_ds_indices = sorted(all_needed)
print(f"Need to embed {len(needed_ds_indices)} unique CLIP images.")

# ---------- CLIP embed ----------
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
    buf_idx.append(int(di))
    buf_img.append(pil_to_tensor(pil, 224, 256))
    if len(buf_img) >= BATCH:
        flush_clip()
flush_clip()
print(f"CLIP embedding done in {time.time()-t0:.1f}s")

emb_dim = next(iter(idx_to_emb.values())).shape[0]
print(f"CLIP embedding dim: {emb_dim}")


def mean_pairwise_cos(vecs):
    if vecs.shape[0] < 2:
        return 1.0
    S = vecs @ vecs.T
    k = vecs.shape[0]
    iu = np.triu_indices(k, k=1)
    return float(S[iu].mean())


# ---------- Compute per-TOP_K metrics ----------
experiment_data = {
    "TOP_K": {
        "imagenet_val_resnet50": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "inspected_channels": np.array(inspected_channels, dtype=np.int64),
            "top_k_values": TOP_K_VALUES,
            "per_top_k": {},
            "config": {
                "N_CHANNELS": N_CHANNELS,
                "N_PROBE": N_PROBE,
                "model": "resnet-50",
                "dataset": "imagenet-val",
                "foundation": os.path.basename(clip_dir),
            },
        }
    }
}

# Random baseline pool: use random_pool_ids (all embedded)
random_pool = [int(i) for i in random_pool_ids if int(i) in idx_to_emb]
print(f"Random pool size: {len(random_pool)}")

summary_records = []
for top_k in TOP_K_VALUES:
    top_k_per_channel = {j: top_k_per_channel_max[j][:top_k] for j in range(N_CHANNELS)}
    concept_scores = np.zeros(N_CHANNELS, dtype=np.float32)
    v_c = np.zeros((N_CHANNELS, emb_dim), dtype=np.float32)
    for j in range(N_CHANNELS):
        vecs = np.stack([idx_to_emb[i] for i in top_k_per_channel[j]], axis=0)
        concept_scores[j] = mean_pairwise_cos(vecs)
        pooled = vecs.mean(axis=0)
        pooled /= np.linalg.norm(pooled) + 1e-8
        v_c[j] = pooled

    rng = np.random.RandomState(123 + top_k)
    random_scores = np.zeros(N_CHANNELS, dtype=np.float32)
    for j in range(N_CHANNELS):
        pick = rng.choice(len(random_pool), size=top_k, replace=False)
        vecs = np.stack([idx_to_emb[int(random_pool[p])] for p in pick], axis=0)
        random_scores[j] = mean_pairwise_cos(vecs)

    mean_concept = float(concept_scores.mean())
    median_concept = float(np.median(concept_scores))
    mean_random = float(random_scores.mean())
    delta = mean_concept - mean_random

    print("==================================================")
    print(f"TOP_K = {top_k}")
    print(f"  concept_consistency_score (mean)   = {mean_concept:.4f}")
    print(f"  concept_consistency_score (median) = {median_concept:.4f}")
    print(f"  random baseline (mean)             = {mean_random:.4f}")
    print(f"  delta                              = {delta:.4f}")

    experiment_data["TOP_K"]["imagenet_val_resnet50"]["metrics"]["val"].append(
        {
            "TOP_K": top_k,
            "concept_consistency_score_mean": mean_concept,
            "concept_consistency_score_median": median_concept,
            "random_baseline_mean": mean_random,
            "delta": delta,
        }
    )
    experiment_data["TOP_K"]["imagenet_val_resnet50"]["losses"]["val"].append(
        {
            "TOP_K": top_k,
            "loss": 1.0 - mean_concept,
        }
    )
    experiment_data["TOP_K"]["imagenet_val_resnet50"]["per_top_k"][top_k] = {
        "concept_scores": concept_scores,
        "random_scores": random_scores,
        "v_c": v_c,
        "top_k_per_channel": top_k_per_channel,
        "mean_concept": mean_concept,
        "median_concept": median_concept,
        "mean_random": mean_random,
        "delta": delta,
    }
    summary_records.append((top_k, mean_concept, mean_random, delta))

# Best by delta (concept - random)
best = max(summary_records, key=lambda r: r[3])
best_k = best[0]
experiment_data["TOP_K"]["imagenet_val_resnet50"]["predictions"] = experiment_data[
    "TOP_K"
]["imagenet_val_resnet50"]["per_top_k"][best_k]["concept_scores"]
experiment_data["TOP_K"]["imagenet_val_resnet50"]["ground_truth"] = experiment_data[
    "TOP_K"
]["imagenet_val_resnet50"]["per_top_k"][best_k]["random_scores"]
experiment_data["TOP_K"]["imagenet_val_resnet50"]["best_top_k"] = best_k
print(f"Best TOP_K by delta = {best_k} (delta={best[3]:.4f}, concept={best[1]:.4f})")

# ---------- Visualization ----------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    xs = [r[0] for r in summary_records]
    ys_c = [r[1] for r in summary_records]
    ys_r = [r[2] for r in summary_records]
    ys_d = [r[3] for r in summary_records]
    ax.plot(xs, ys_c, "o-", label="concept (mean)")
    ax.plot(xs, ys_r, "s--", label="random baseline")
    ax.plot(xs, ys_d, "^:", label="delta (concept - random)")
    ax.set_xlabel("TOP_K")
    ax.set_ylabel("mean pairwise CLIP cos sim")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_title(f"TOP_K sweep (N_PROBE={N_PROBE}): concept consistency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(working_dir, "top_k_sweep_resnet50.png"), dpi=140)
    plt.close(fig)
except Exception as e:
    print("Plotting failed:", e)

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")

handle.remove()
