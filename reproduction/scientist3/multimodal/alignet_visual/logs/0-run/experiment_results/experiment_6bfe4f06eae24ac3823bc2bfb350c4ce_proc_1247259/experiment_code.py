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

import os, sys, glob, math, json, time, traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"

# Hardcoded known-good paths from memory
IMG_DIR = os.path.join(DATA_DIR, "things_ooo_root", "imgur_images")
TRIPLETS_CSV = os.path.join(
    DATA_DIR,
    "things_ooo_triplets",
    "triplets_large_final_correctednc_correctedorder.csv",
)

# Alt scanning if not found
if not os.path.isdir(IMG_DIR):
    cands = glob.glob(
        os.path.join(DATA_DIR, "things*", "**", "imgur_images"), recursive=True
    )
    if cands:
        IMG_DIR = cands[0]
if not os.path.isfile(TRIPLETS_CSV):
    cands = glob.glob(
        os.path.join(DATA_DIR, "things*", "**", "triplets*.csv"), recursive=True
    )
    if cands:
        TRIPLETS_CSV = cands[0]

print(f"IMG_DIR: {IMG_DIR}")
print(f"TRIPLETS_CSV: {TRIPLETS_CSV}")
assert os.path.isdir(IMG_DIR), f"Missing images dir: {IMG_DIR}"
assert os.path.isfile(TRIPLETS_CSV), f"Missing triplets: {TRIPLETS_CSV}"

# List images (sorted, ignore hidden)
all_files = sorted(
    [
        f
        for f in os.listdir(IMG_DIR)
        if not f.startswith(".")
        and not f.startswith("_")
        and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    ]
)
print(f"Found {len(all_files)} image files")
assert len(all_files) >= 1800, f"Too few images: {len(all_files)}"
N_CONCEPTS = len(all_files)
image_paths = [os.path.join(IMG_DIR, f) for f in all_files]


# Load triplets - try multiple delimiters
def load_triplets(path):
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 4:
                return df, sep
        except Exception:
            continue
    raise RuntimeError("Could not parse triplets CSV")


df, sep = load_triplets(TRIPLETS_CSV)
print(f"Triplets CSV: shape={df.shape}, sep={repr(sep)}, cols={list(df.columns)[:10]}")
print(df.head())

# Identify columns
cols_lower = [c.lower() for c in df.columns]


def find_col(patterns):
    for p in patterns:
        for i, c in enumerate(cols_lower):
            if p in c:
                return df.columns[i]
    return None


c1 = find_col(["image1", "img1", "item1", "ref1"])
c2 = find_col(["image2", "img2", "item2", "ref2"])
c3 = find_col(["image3", "img3", "item3", "ref3"])
cc = find_col(["choice", "odd", "ooo", "pick"])

# Fallback: first 4 numeric-ish cols
if not all([c1, c2, c3, cc]):
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numcols) >= 4:
        c1, c2, c3, cc = numcols[:4]
print(f"Using columns: {c1}, {c2}, {c3}, choice={cc}")

trip = df[[c1, c2, c3, cc]].to_numpy(copy=True).astype(np.int64)
# Detect 1-indexed
mn = trip[:, :3].min()
mx = trip[:, :3].max()
print(f"ID range: [{mn}, {mx}]")
if mn == 1:
    trip[:, :3] -= 1  # 0-index concepts
    print("Applied 1-index -> 0-index conversion")

# choice: 1..3 -> 0..2
choice_mn, choice_mx = trip[:, 3].min(), trip[:, 3].max()
print(f"Choice range: [{choice_mn}, {choice_mx}]")
if choice_mn >= 1 and choice_mx <= 3:
    trip[:, 3] -= 1

# Filter valid
valid = (
    (trip[:, 0] < N_CONCEPTS)
    & (trip[:, 1] < N_CONCEPTS)
    & (trip[:, 2] < N_CONCEPTS)
    & (trip[:, 0] >= 0)
    & (trip[:, 1] >= 0)
    & (trip[:, 2] >= 0)
    & (trip[:, 3] >= 0)
    & (trip[:, 3] <= 2)
)
print(f"Valid triplets: {valid.sum()}/{len(trip)}")
trip = trip[valid]
assert len(trip) > 1000, "Too few valid triplets"

# Choice distribution sanity
_, counts = np.unique(trip[:, 3], return_counts=True)
print(f"Choice distribution: {counts / counts.sum()}")

# ---- Load DINOv2 (student) and SigLIP (teacher) ----
from transformers import AutoModel, AutoImageProcessor


def find_model_dir(patterns):
    for p in patterns:
        cands = glob.glob(os.path.join(MODEL_DIR, p))
        for c in cands:
            if os.path.isdir(c):
                return c
    return None


DINO_DIR = find_model_dir(
    ["dinov2-base", "dinov2_vitb*", "dinov2*base*", "facebook*dinov2-base*"]
)
SIGLIP_DIR = find_model_dir(["siglip-so400m*", "*siglip*so400m*", "siglip*"])
print(f"DINO_DIR: {DINO_DIR}")
print(f"SIGLIP_DIR: {SIGLIP_DIR}")
assert DINO_DIR is not None, "DINOv2 model not found"
assert SIGLIP_DIR is not None, "SigLIP model not found"


class ImgDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        out = self.processor(images=img, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in out.items()}


@torch.no_grad()
def extract_features(model_dir, image_paths, batch_size=32, cache_name=None):
    cache_path = os.path.join(working_dir, f"{cache_name}.npy") if cache_name else None
    if cache_path and os.path.isfile(cache_path):
        feats = np.load(cache_path)
        if feats.shape[0] == len(image_paths):
            print(f"Loaded cached {cache_name}: {feats.shape}")
            return feats
    print(f"Extracting from {model_dir}...")
    proc = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    )
    ds = ImgDataset(image_paths, proc)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False)
    feats = []
    t0 = time.time()
    for bi, batch in enumerate(dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        # Try vision_model if available (SigLIP)
        if hasattr(model, "vision_model"):
            out = model.vision_model(
                **{k: v for k, v in batch.items() if k == "pixel_values"}
            )
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                f = out.pooler_output
            else:
                f = out.last_hidden_state.mean(dim=1)
        else:
            out = model(**batch)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                f = out.pooler_output
            else:
                f = out.last_hidden_state[:, 0]  # CLS
        feats.append(f.float().cpu().numpy())
        if bi % 20 == 0:
            print(f"  batch {bi}/{len(dl)} elapsed {time.time()-t0:.1f}s")
    feats = np.concatenate(feats, axis=0)
    del model
    torch.cuda.empty_cache()
    if cache_path:
        np.save(cache_path, feats)
    print(f"Extracted {feats.shape}")
    return feats


student_feats = extract_features(
    DINO_DIR, image_paths, batch_size=32, cache_name="dino_feats"
)
teacher_feats = extract_features(
    SIGLIP_DIR, image_paths, batch_size=16, cache_name="siglip_feats"
)

# Sanity assertions
assert student_feats.shape[0] == N_CONCEPTS
assert teacher_feats.shape[0] == N_CONCEPTS
sn = np.linalg.norm(student_feats, axis=1)
tn = np.linalg.norm(teacher_feats, axis=1)
assert (sn > 1e-6).all(), "zero student feats"
assert (tn > 1e-6).all(), "zero teacher feats"
uniq_s = len(np.unique(student_feats[:, 0]))
uniq_t = len(np.unique(teacher_feats[:, 0]))
print(f"Unique student rows(0): {uniq_s}, teacher rows(0): {uniq_t}")
assert uniq_s > 0.9 * N_CONCEPTS
assert uniq_t > 0.9 * N_CONCEPTS

# Normalize
student_feats_t = torch.tensor(student_feats, dtype=torch.float32, device=device)
teacher_feats_t = torch.tensor(teacher_feats, dtype=torch.float32, device=device)
student_feats_n = F.normalize(student_feats_t, dim=-1)
teacher_feats_n = F.normalize(teacher_feats_t, dim=-1)


# OOO evaluation
def ooo_accuracy(feats_n, triplets):
    # triplets: [N, 4] (i1,i2,i3,choice)
    idx = torch.tensor(triplets[:, :3], dtype=torch.long, device=feats_n.device)
    choice = torch.tensor(triplets[:, 3], dtype=torch.long, device=feats_n.device)
    f = feats_n[idx]  # [N,3,D]
    # cosine sims between each pair
    s01 = (f[:, 0] * f[:, 1]).sum(-1)
    s02 = (f[:, 0] * f[:, 2]).sum(-1)
    s12 = (f[:, 1] * f[:, 2]).sum(-1)
    # odd-one-out = item with lowest avg sim to others
    # avg sim of item0 to others = (s01 + s02)/2
    avg0 = (s01 + s02) / 2
    avg1 = (s01 + s12) / 2
    avg2 = (s02 + s12) / 2
    avgs = torch.stack([avg0, avg1, avg2], dim=1)  # [N,3]
    pred = avgs.argmin(dim=1)
    return (pred == choice).float().mean().item()


# Split triplets
np.random.seed(0)
perm = np.random.permutation(len(trip))
n_val = min(20000, len(trip) // 5)
val_idx = perm[:n_val]
train_idx = perm[n_val:]
trip_train = trip[train_idx]
trip_val = trip[val_idx]
print(f"Train triplets: {len(trip_train)}, Val triplets: {len(trip_val)}")

baseline_ooo = ooo_accuracy(student_feats_n, trip_val)
teacher_ooo = ooo_accuracy(teacher_feats_n, trip_val)
print(f"Baseline DINOv2 OOO acc: {baseline_ooo:.4f}")
print(f"Teacher SigLIP OOO acc:  {teacher_ooo:.4f}")

# Loud gate
if teacher_ooo <= baseline_ooo:
    print("WARNING: teacher <= baseline; distillation may not help")

# ---- Train projection head on student features to match teacher pairwise cosine sims ----
D_in = student_feats_t.shape[1]
D_out = 512


class ProjHead(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_out),
        )

    def forward(self, x):
        return self.net(x)


head = ProjHead(D_in, D_out).to(device)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

experiment_data = {
    "THINGS": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "baseline_ooo": baseline_ooo,
        "teacher_ooo": teacher_ooo,
        "predictions": [],
        "ground_truth": trip_val[:, 3].tolist(),
    }
}


def compute_aligned_feats():
    head.eval()
    with torch.no_grad():
        af = head(student_feats_t)
        return F.normalize(af, dim=-1)


N_EPOCHS = 15
BATCH_TRIPLETS = 1024

# We'll use triplet-based distillation: for each triplet's 3 concepts, match teacher pairwise cosines
trip_train_t = torch.tensor(trip_train[:, :3], dtype=torch.long, device=device)

for epoch in range(N_EPOCHS):
    head.train()
    perm_e = torch.randperm(len(trip_train_t), device=device)
    total_loss = 0.0
    n_batches = 0
    for b_start in range(0, len(trip_train_t), BATCH_TRIPLETS):
        b_idx = perm_e[b_start : b_start + BATCH_TRIPLETS]
        idx3 = trip_train_t[b_idx]  # [B,3]
        # get unique concepts in this batch
        flat = idx3.reshape(-1)
        s_in = student_feats_t[flat]
        s_out = head(s_in)
        s_out = F.normalize(s_out, dim=-1)
        t_out = teacher_feats_n[flat]
        s_out = s_out.view(-1, 3, D_out)
        t_out = t_out.view(-1, 3, teacher_feats_n.shape[1])

        # pairwise cosines within triplet
        def pair_sims(x):
            s01 = (x[:, 0] * x[:, 1]).sum(-1)
            s02 = (x[:, 0] * x[:, 2]).sum(-1)
            s12 = (x[:, 1] * x[:, 2]).sum(-1)
            return torch.stack([s01, s02, s12], dim=1)

        ss = pair_sims(s_out)
        ts = pair_sims(t_out)
        loss = F.mse_loss(ss, ts)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    train_loss = total_loss / max(1, n_batches)

    # Val
    aligned = compute_aligned_feats()
    val_ooo = ooo_accuracy(aligned, trip_val)
    # Val loss on val triplets
    head.eval()
    with torch.no_grad():
        idx3 = torch.tensor(trip_val[:, :3], dtype=torch.long, device=device)
        flat = idx3.reshape(-1)
        s_out = F.normalize(head(student_feats_t[flat]), dim=-1).view(-1, 3, D_out)
        t_out = teacher_feats_n[flat].view(-1, 3, teacher_feats_n.shape[1])

        def pair_sims(x):
            s01 = (x[:, 0] * x[:, 1]).sum(-1)
            s02 = (x[:, 0] * x[:, 2]).sum(-1)
            s12 = (x[:, 1] * x[:, 2]).sum(-1)
            return torch.stack([s01, s02, s12], dim=1)

        val_loss = F.mse_loss(pair_sims(s_out), pair_sims(t_out)).item()

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} validation_loss={val_loss:.4f} aligned_OOO={val_ooo:.4f}"
    )
    experiment_data["THINGS"]["losses"]["train"].append(train_loss)
    experiment_data["THINGS"]["losses"]["val"].append(val_loss)
    experiment_data["THINGS"]["metrics"]["train"].append(train_loss)
    experiment_data["THINGS"]["metrics"]["val"].append(val_ooo)

# Final aligned OOO and predictions
aligned = compute_aligned_feats()
final_ooo = ooo_accuracy(aligned, trip_val)
print(f"\n=== FINAL RESULTS ===")
print(f"Baseline DINOv2 OOO: {baseline_ooo:.4f}")
print(f"Teacher SigLIP OOO:  {teacher_ooo:.4f}")
print(f"Aligned Student OOO: {final_ooo:.4f}")

# Save predictions
with torch.no_grad():
    idx = torch.tensor(trip_val[:, :3], dtype=torch.long, device=device)
    f = aligned[idx]
    s01 = (f[:, 0] * f[:, 1]).sum(-1)
    s02 = (f[:, 0] * f[:, 2]).sum(-1)
    s12 = (f[:, 1] * f[:, 2]).sum(-1)
    avgs = torch.stack([(s01 + s02) / 2, (s01 + s12) / 2, (s02 + s12) / 2], dim=1)
    preds = avgs.argmin(dim=1).cpu().numpy()
experiment_data["THINGS"]["predictions"] = preds.tolist()
experiment_data["THINGS"]["final_aligned_ooo"] = final_ooo

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved to {working_dir}/experiment_data.npy")

# Simple plot
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(experiment_data["THINGS"]["losses"]["train"], label="train")
    ax[0].plot(experiment_data["THINGS"]["losses"]["val"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[0].set_xlabel("epoch")
    ax[1].plot(
        experiment_data["THINGS"]["metrics"]["val"], label="aligned OOO", color="C2"
    )
    ax[1].axhline(
        baseline_ooo, color="C0", linestyle="--", label=f"baseline {baseline_ooo:.3f}"
    )
    ax[1].axhline(
        teacher_ooo, color="C1", linestyle="--", label=f"teacher {teacher_ooo:.3f}"
    )
    ax[1].set_title("THINGS OOO accuracy")
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "things_ooo_alignment.png"), dpi=100)
    print("Saved plot")
except Exception as e:
    print(f"Plot failed: {e}")
