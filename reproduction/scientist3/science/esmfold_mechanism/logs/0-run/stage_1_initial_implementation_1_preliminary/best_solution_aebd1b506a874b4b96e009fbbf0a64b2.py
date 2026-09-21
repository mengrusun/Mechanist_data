import os, sys, re, io, gzip, json, shutil, subprocess, importlib, traceback, time

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Preflight ----------
for s in ("stdout", "stderr"):
    obj = getattr(sys, s)
    if not hasattr(obj, "isatty"):
        try:
            setattr(obj, "isatty", lambda: False)
        except Exception:
            pass
    else:
        try:
            obj.isatty()
        except Exception:
            try:
                setattr(obj, "isatty", lambda: False)
            except Exception:
                pass


def _pip(pkg):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    except Exception as e:
        print(f"pip install {pkg} failed: {e}")


for pkg, mod in [("pydssp", "pydssp"), ("biotite", "biotite"), ("biopython", "Bio")]:
    if importlib.util.find_spec(mod) is None:
        _pip(pkg)

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.environ.setdefault("HF_TOKEN", "<Your_token>")
os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", os.environ["HF_TOKEN"])
DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"

# ---------- Panel ----------
# name, sequence, hairpin target region (1-based inclusive)
PANEL = [
    # GB1 β-hairpin (residues 41-56 of GB1 domain contain a canonical hairpin)
    ("GB1", "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE", (41, 56), True),
    # Ubiquitin N-terminal β-hairpin ~ residues 2-17
    (
        "ubiquitin",
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        (2, 17),
        True,
    ),
    # Pin1 WW domain β-hairpin (residues ~6-22 of the 39-aa WW domain)
    ("WW_Pin1", "KLPPGWEKRMSRSSGRVYYFNHITNASQWERPSG", (6, 22), True),
    # Trpzip2 designed β-hairpin (12 aa, whole thing is a hairpin)
    ("Trpzip2", "SWTWENGKWTWK", (1, 12), True),
    # Chignolin — designed β-hairpin
    ("Chignolin", "GYDPETGTWG", (1, 10), True),
    # HP7 designed hairpin
    ("HP7", "KTWNPATGKWTE", (1, 12), True),
    # Negative control: Trp-cage (α-helix + PPII, no β-hairpin)
    ("Trp_cage", "NLYIQWLKDGGPSSGRPPPS", (1, 20), False),
    # Negative control: poly-alanine helix
    ("polyA_helix", "AAAAAAAAAAAAAAAAAAAA", (1, 20), False),
]


# ---------- Load ESMFold ----------
def load_esmfold():
    from transformers import AutoTokenizer, EsmForProteinFolding

    candidates = [
        os.path.join(MODEL_DIR, "esmfold_v1"),
        os.path.join(MODEL_DIR, "esmfold"),
        "facebook/esmfold_v1",
    ]
    last_err = None
    for p in candidates:
        try:
            print(f"Loading ESMFold from {p} ...")
            tok = AutoTokenizer.from_pretrained(p)
            mdl = EsmForProteinFolding.from_pretrained(p, low_cpu_mem_usage=True)
            print(f"Loaded from {p}")
            return tok, mdl
        except Exception as e:
            print(f"Failed {p}: {e}")
            last_err = e
    raise RuntimeError(f"Could not load ESMFold: {last_err}")


tokenizer, model = load_esmfold()
model = model.to(device)
model.eval()
try:
    model.esm = model.esm.half()
except Exception:
    pass
try:
    model.trunk.set_chunk_size(64)
except Exception:
    pass


# ---------- Prediction & PDB writing ----------
def convert_outputs_to_pdb(outputs):
    from transformers.models.esm.openfold_utils.protein import (
        to_pdb,
        Protein as OFProtein,
    )
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {
        k: v.to("cpu").numpy()
        for k, v in outputs.items()
        if isinstance(v, torch.Tensor)
    }
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


@torch.no_grad()
def predict_structure(seq):
    tokenized = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    out = model(**tokenized)
    pdb_strs = convert_outputs_to_pdb(out)
    plddt = out["plddt"].mean().item()
    return pdb_strs[0], plddt


# ---------- SS assignment cascade ----------
def ss_from_pydssp(pdb_str):
    import pydssp
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", io.StringIO(pdb_str))
    # collect N, CA, C, O for each residue in order
    coords = []
    for model_ in struct:
        for chain in model_:
            for res in chain:
                try:
                    n = res["N"].coord
                    ca = res["CA"].coord
                    c = res["C"].coord
                    o = res["O"].coord
                except KeyError:
                    continue
                coords.append([n, ca, c, o])
        break
    if len(coords) == 0:
        raise RuntimeError("no backbone atoms")
    arr = torch.tensor(np.array(coords), dtype=torch.float32).unsqueeze(
        0
    )  # [1, L, 4, 3]
    ss = pydssp.assign(arr, out_type="c3")  # returns "H"/"E"/"-" per residue
    if isinstance(ss, list) or isinstance(ss, tuple):
        ss = ss[0]
    if isinstance(ss, np.ndarray):
        ss = "".join(ss.tolist())
    # normalize to E / H / L
    ss = "".join(
        "E" if c in ("E", "B") else ("H" if c in ("H", "G", "I") else "L") for c in ss
    )
    return ss


def ss_from_mkdssp(pdb_str):
    if shutil.which("mkdssp") is None:
        raise RuntimeError("mkdssp not installed")
    tmp = os.path.join(working_dir, "_tmp.pdb")
    with open(tmp, "w") as f:
        f.write(pdb_str)
    out = subprocess.check_output(["mkdssp", tmp], text=True)
    # parse DSSP output
    ss = []
    started = False
    for ln in out.splitlines():
        if ln.startswith("  #  RESIDUE"):
            started = True
            continue
        if not started:
            continue
        if len(ln) < 17:
            continue
        c = ln[16]
        ss.append("E" if c in ("E", "B") else ("H" if c in ("H", "G", "I") else "L"))
    return "".join(ss)


def ss_from_biotite(pdb_str):
    import biotite.structure.io.pdb as bpdb
    import biotite.structure as bstruc

    pf = bpdb.PDBFile.read(io.StringIO(pdb_str))
    atoms = pf.get_structure(model=1)
    sse = bstruc.annotate_sse(atoms)
    return "".join("E" if c == "b" else ("H" if c == "a" else "L") for c in sse)


def assign_ss(pdb_str):
    errs = []
    for fn, name in [
        (ss_from_pydssp, "pydssp"),
        (ss_from_mkdssp, "mkdssp"),
        (ss_from_biotite, "biotite"),
    ]:
        try:
            ss = fn(pdb_str)
            if ss and len(ss) > 0:
                return ss, name
        except Exception as e:
            errs.append(f"{name}: {e}")
    raise RuntimeError("All SS assigners failed: " + " | ".join(errs))


# ---------- Hairpin regex ----------
HAIRPIN_RE = re.compile(r"E{3,}L{2,8}E{3,}")


def is_hairpin(ss_region):
    return bool(HAIRPIN_RE.search(ss_region))


# ---------- Experiment ----------
experiment_data = {
    "esmfold_hairpin_panel": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "per_protein": [],
    }
}

pdb_cache_dir = os.path.join(working_dir, "pdb_cache")
os.makedirs(pdb_cache_dir, exist_ok=True)

# Stage 1: predict + cache PDBs
for name, seq, region, is_pos in PANEL:
    pdb_path = os.path.join(pdb_cache_dir, f"{name}.pdb")
    if os.path.exists(pdb_path):
        print(f"[cache] {name}")
        continue
    try:
        t0 = time.time()
        pdb_str, plddt = predict_structure(seq)
        with open(pdb_path, "w") as f:
            f.write(pdb_str)
        with open(pdb_path + ".plddt", "w") as f:
            f.write(str(plddt))
        print(f"[pred] {name} len={len(seq)} plddt={plddt:.2f} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[pred FAIL] {name}: {e}")
        traceback.print_exc()

# Stage 2: SS assignment + metric
n_eval, n_correct = 0, 0
for name, seq, (r0, r1), is_pos in PANEL:
    pdb_path = os.path.join(pdb_cache_dir, f"{name}.pdb")
    if not os.path.exists(pdb_path):
        print(f"[skip] {name}: no cached PDB")
        continue
    try:
        with open(pdb_path) as f:
            pdb_str = f.read()
        try:
            with open(pdb_path + ".plddt") as f:
                plddt = float(f.read())
        except Exception:
            plddt = float("nan")
        ss, used = assign_ss(pdb_str)
        if len(ss) != len(seq):
            print(f"[warn] {name}: ss len {len(ss)} vs seq len {len(seq)}")
        ss_full = ss[: len(seq)].ljust(len(seq), "L")
        region_ss = ss_full[r0 - 1 : r1]
        pred_hairpin = is_hairpin(region_ss)
        correct = int(pred_hairpin == is_pos)
        n_eval += 1
        n_correct += correct
        print(
            f"[eval] {name:12s} region={region_ss}  full={ss_full}  pred_hp={pred_hairpin} gt_hp={is_pos} correct={correct} plddt={plddt:.2f} tool={used}"
        )
        experiment_data["esmfold_hairpin_panel"]["per_protein"].append(
            {
                "name": name,
                "sequence": seq,
                "region": [r0, r1],
                "gt_hairpin": is_pos,
                "pred_hairpin": pred_hairpin,
                "ss_full": ss_full,
                "ss_region": region_ss,
                "plddt": plddt,
                "ss_tool": used,
                "correct": bool(correct),
            }
        )
        experiment_data["esmfold_hairpin_panel"]["predictions"].append(
            int(pred_hairpin)
        )
        experiment_data["esmfold_hairpin_panel"]["ground_truth"].append(int(is_pos))
    except Exception as e:
        print(f"[eval FAIL] {name}: {e}")
        traceback.print_exc()

assert n_eval > 0, "No proteins evaluated; metric is meaningless"
hairpin_dssp_accuracy = n_correct / n_eval
print(
    f"\n=== hairpin_dssp_accuracy = {hairpin_dssp_accuracy:.4f}  ({n_correct}/{n_eval}) ==="
)
print(f"Epoch 0: validation_loss = {1.0 - hairpin_dssp_accuracy:.4f}")

experiment_data["esmfold_hairpin_panel"]["metrics"]["val"].append(
    {
        "epoch": 0,
        "hairpin_dssp_accuracy": hairpin_dssp_accuracy,
        "n_eval": n_eval,
        "n_correct": n_correct,
    }
)
experiment_data["esmfold_hairpin_panel"]["losses"]["val"].append(
    {"epoch": 0, "loss": 1.0 - hairpin_dssp_accuracy}
)

# Simple visualization
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [p["name"] for p in experiment_data["esmfold_hairpin_panel"]["per_protein"]]
    preds = [
        p["pred_hairpin"]
        for p in experiment_data["esmfold_hairpin_panel"]["per_protein"]
    ]
    gts = [
        p["gt_hairpin"] for p in experiment_data["esmfold_hairpin_panel"]["per_protein"]
    ]
    x = np.arange(len(names))
    plt.figure(figsize=(9, 3))
    plt.bar(x - 0.2, gts, 0.4, label="GT hairpin")
    plt.bar(x + 0.2, preds, 0.4, label="Pred hairpin")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("hairpin (0/1)")
    plt.title(f"hairpin_dssp_accuracy = {hairpin_dssp_accuracy:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "esmfold_hairpin_panel_baseline.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print("plot failed:", e)

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
