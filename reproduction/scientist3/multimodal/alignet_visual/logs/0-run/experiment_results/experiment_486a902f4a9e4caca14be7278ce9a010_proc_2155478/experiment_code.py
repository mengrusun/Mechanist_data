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

IMG_DIR = os.path.join(DATA_DIR, "things_ooo_root", "imgur_images")
TRIPLETS_CSV = os.path.join(
    DATA_DIR,
    "things_ooo_triplets",
    "triplets_large_final_correctednc_correctedorder.csv",
)

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
assert os.path.isdir(IMG_DIR)
assert os.path.isfile(TRIPLETS_CSV)

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
assert len(all_files) >= 1800
N_CONCEPTS = len(all_files)
image_paths = [os.path.join(IMG_DIR, f) for f in all_files]


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
print(f"Triplets CSV: shape={df.shape}, sep={repr(sep)}")

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
if not all([c1, c2, c3, cc]):
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numcols) >= 4:
        c1, c2, c3, cc = numcols[:4]
print(f"Using columns: {c1}, {c2}, {c3}, choice={cc}")

trip = df[[c1, c2, c3, cc]].to_numpy(copy=True).astype(np.int64)
if trip[:, :3].min() == 1:
    trip[:, :3] -= 1
if trip[:, 3].min() >= 1 and trip[:, 3].max() <= 3:
    trip[:, 3] -= 1

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
trip = trip[valid]
print(f"Valid triplets: {len(trip)}")

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
assert DINO_DIR is not None
assert SIGLIP_DIR is not None


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
                f = out.last_hidden_state[:, 0]
        feats.append(f.float().cpu().numpy())
        if bi % 20 == 0:
            print(f"  batch {bi}/{len(dl)} elapsed {time.time()-t0:.1f}s")
    feats = np.concatenate(feats, axis=0)
    del model
    torch.cuda.empty_cache()
    if cache_path:
        np.save(cache_path, feats)
    return feats


student_feats = extract_features(
    DINO_DIR, image_paths, batch_size=32, cache_name="dino_feats"
)
teacher_feats = extract_features(
    SIGLIP_DIR, image_paths, batch_size=16, cache_name="siglip_feats"
)

student_feats_t = torch.tensor(student_feats, dtype=torch.float32, device=device)
teacher_feats_t = torch.tensor(teacher_feats, dtype=torch.float32, device=device)
student_feats_n = F.normalize(student_feats_t, dim=-1)
teacher_feats_n = F.normalize(teacher_feats_t, dim=-1)


def ooo_accuracy(feats_n, triplets):
    idx = torch.tensor(triplets[:, :3], dtype=torch.long, device=feats_n.device)
    choice = torch.tensor(triplets[:, 3], dtype=torch.long, device=feats_n.device)
    f = feats_n[idx]
    s01 = (f[:, 0] * f[:, 1]).sum(-1)
    s02 = (f[:, 0] * f[:, 2]).sum(-1)
    s12 = (f[:, 1] * f[:, 2]).sum(-1)
    avg0 = (s01 + s02) / 2
    avg1 = (s01 + s12) / 2
    avg2 = (s02 + s12) / 2
    avgs = torch.stack([avg0, avg1, avg2], dim=1)
    pred = avgs.argmin(dim=1)
    return (pred == choice).float().mean().item()


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
print(f"Baseline DINOv2 OOO: {baseline_ooo:.4f}")
print(f"Teacher SigLIP OOO:  {teacher_ooo:.4f}")

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


def pair_sims(x):
    s01 = (x[:, 0] * x[:, 1]).sum(-1)
    s02 = (x[:, 0] * x[:, 2]).sum(-1)
    s12 = (x[:, 1] * x[:, 2]).sum(-1)
    return torch.stack([s01, s02, s12], dim=1)


BATCH_TRIPLETS = 1024
trip_train_t = torch.tensor(trip_train[:, :3], dtype=torch.long, device=device)
val_idx_t = torch.tensor(trip_val[:, :3], dtype=torch.long, device=device)

experiment_data = {"N_EPOCHS": {}}
epochs_to_try = [15, 30, 50, 75]

for N_EPOCHS in epochs_to_try:
    print(f"\n===== Tuning N_EPOCHS = {N_EPOCHS} =====")
    torch.manual_seed(42)
    head = ProjHead(D_in, D_out).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    run_key = f"epochs_{N_EPOCHS}"
    experiment_data["N_EPOCHS"][run_key] = {
        "THINGS": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "baseline_ooo": baseline_ooo,
            "teacher_ooo": teacher_ooo,
            "predictions": [],
            "ground_truth": trip_val[:, 3].tolist(),
            "n_epochs": N_EPOCHS,
        }
    }
    ed = experiment_data["N_EPOCHS"][run_key]["THINGS"]

    for epoch in range(N_EPOCHS):
        head.train()
        perm_e = torch.randperm(len(trip_train_t), device=device)
        total_loss = 0.0
        n_batches = 0
        for b_start in range(0, len(trip_train_t), BATCH_TRIPLETS):
            b_idx = perm_e[b_start : b_start + BATCH_TRIPLETS]
            idx3 = trip_train_t[b_idx]
            flat = idx3.reshape(-1)
            s_out = F.normalize(head(student_feats_t[flat]), dim=-1).view(-1, 3, D_out)
            t_out = teacher_feats_n[flat].view(-1, 3, teacher_feats_n.shape[1])
            loss = F.mse_loss(pair_sims(s_out), pair_sims(t_out))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        head.eval()
        with torch.no_grad():
            af = F.normalize(head(student_feats_t), dim=-1)
            val_ooo = ooo_accuracy(af, trip_val)
            flat = val_idx_t.reshape(-1)
            s_out = F.normalize(head(student_feats_t[flat]), dim=-1).view(-1, 3, D_out)
            t_out = teacher_feats_n[flat].view(-1, 3, teacher_feats_n.shape[1])
            val_loss = F.mse_loss(pair_sims(s_out), pair_sims(t_out)).item()

        print(
            f"[epochs={N_EPOCHS}] Epoch {epoch}: train_loss={train_loss:.4f} validation_loss={val_loss:.4f} aligned_OOO={val_ooo:.4f}"
        )
        ed["losses"]["train"].append(train_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train"].append(train_loss)
        ed["metrics"]["val"].append(val_ooo)

    head.eval()
    with torch.no_grad():
        aligned = F.normalize(head(student_feats_t), dim=-1)
        final_ooo = ooo_accuracy(aligned, trip_val)
        f = aligned[val_idx_t]
        s01 = (f[:, 0] * f[:, 1]).sum(-1)
        s02 = (f[:, 0] * f[:, 2]).sum(-1)
        s12 = (f[:, 1] * f[:, 2]).sum(-1)
        avgs = torch.stack([(s01 + s02) / 2, (s01 + s12) / 2, (s02 + s12) / 2], dim=1)
        preds = avgs.argmin(dim=1).cpu().numpy()
    ed["predictions"] = preds.tolist()
    ed["final_aligned_ooo"] = final_ooo
    print(f"[epochs={N_EPOCHS}] Final aligned OOO: {final_ooo:.4f}")

print("\n=== SUMMARY ===")
print(f"Baseline DINOv2 OOO: {baseline_ooo:.4f}")
print(f"Teacher SigLIP OOO:  {teacher_ooo:.4f}")
for N_EPOCHS in epochs_to_try:
    run_key = f"epochs_{N_EPOCHS}"
    fo = experiment_data["N_EPOCHS"][run_key]["THINGS"]["final_aligned_ooo"]
    print(f"  N_EPOCHS={N_EPOCHS}: Aligned OOO = {fo:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved to {working_dir}/experiment_data.npy")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for N_EPOCHS in epochs_to_try:
        run_key = f"epochs_{N_EPOCHS}"
        ed = experiment_data["N_EPOCHS"][run_key]["THINGS"]
        ax[0].plot(ed["losses"]["val"], label=f"val loss E={N_EPOCHS}")
        ax[1].plot(ed["metrics"]["val"], label=f"OOO E={N_EPOCHS}")
    ax[0].set_title("Val Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].axhline(
        baseline_ooo, color="k", linestyle="--", label=f"baseline {baseline_ooo:.3f}"
    )
    ax[1].axhline(
        teacher_ooo, color="r", linestyle="--", label=f"teacher {teacher_ooo:.3f}"
    )
    ax[1].set_title("Aligned OOO")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "n_epochs_tuning.png"), dpi=100)
    print("Saved plot")
except Exception as e:
    print(f"Plot failed: {e}")
