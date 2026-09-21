import os, sys, re, io, json, shutil, subprocess, importlib, traceback, time

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

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

PANEL = [
    ("GB1", "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE", (41, 56), True),
    (
        "ubiquitin",
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        (2, 17),
        True,
    ),
    ("WW_Pin1", "KLPPGWEKRMSRSSGRVYYFNHITNASQWERPSG", (6, 22), True),
    ("Trpzip2", "SWTWENGKWTWK", (1, 12), True),
    ("Chignolin", "GYDPETGTWG", (1, 10), True),
    ("HP7", "KTWNPATGKWTE", (1, 12), True),
    ("Trp_cage", "NLYIQWLKDGGPSSGRPPPS", (1, 20), False),
    ("polyA_helix", "AAAAAAAAAAAAAAAAAAAA", (1, 20), False),
]


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
        pred = OFProtein(
            aatype=outputs["aatype"][i],
            atom_positions=final_atom_positions[i],
            atom_mask=final_atom_mask[i],
            residue_index=outputs["residue_index"][i] + 1,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


@torch.no_grad()
def predict_structure(seq, num_recycles=4):
    tokenized = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    try:
        out = model(**tokenized, num_recycles=num_recycles)
    except TypeError:
        out = model(**tokenized)
    pdb_strs = convert_outputs_to_pdb(out)
    plddt = out["plddt"].mean().item()
    return pdb_strs[0], plddt


def ss_from_pydssp(pdb_str):
    import pydssp
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", io.StringIO(pdb_str))
    coords = []
    for model_ in struct:
        for chain in model_:
            for res in chain:
                try:
                    coords.append(
                        [
                            res["N"].coord,
                            res["CA"].coord,
                            res["C"].coord,
                            res["O"].coord,
                        ]
                    )
                except KeyError:
                    continue
        break
    if len(coords) == 0:
        raise RuntimeError("no backbone atoms")
    arr = torch.tensor(np.array(coords), dtype=torch.float32).unsqueeze(0)
    ss = pydssp.assign(arr, out_type="c3")
    if isinstance(ss, (list, tuple)):
        ss = ss[0]
    if isinstance(ss, np.ndarray):
        ss = "".join(ss.tolist())
    return "".join(
        "E" if c in ("E", "B") else ("H" if c in ("H", "G", "I") else "L") for c in ss
    )


def ss_from_mkdssp(pdb_str):
    if shutil.which("mkdssp") is None:
        raise RuntimeError("mkdssp not installed")
    tmp = os.path.join(working_dir, "_tmp.pdb")
    with open(tmp, "w") as f:
        f.write(pdb_str)
    out = subprocess.check_output(["mkdssp", tmp], text=True)
    ss = []
    started = False
    for ln in out.splitlines():
        if ln.startswith("  #  RESIDUE"):
            started = True
            continue
        if not started or len(ln) < 17:
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


HAIRPIN_RE = re.compile(r"E{3,}L{2,8}E{3,}")


def is_hairpin(ss_region):
    return bool(HAIRPIN_RE.search(ss_region))


# ---------- Hyperparameter tuning: num_recycles ----------
NUM_RECYCLES_LIST = [4, 8, 12]

experiment_data = {
    "num_recycles": {
        "esmfold_hairpin_panel": {
            "hyperparams": NUM_RECYCLES_LIST,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": {},
            "ground_truth": [int(p[3]) for p in PANEL],
            "per_protein": {},
            "per_setting_summary": [],
        }
    }
}

for nr in NUM_RECYCLES_LIST:
    print(f"\n########## num_recycles = {nr} ##########")
    pdb_cache_dir = os.path.join(working_dir, f"pdb_cache_nr{nr}")
    os.makedirs(pdb_cache_dir, exist_ok=True)

    # Stage 1: predict + cache
    for name, seq, region, is_pos in PANEL:
        pdb_path = os.path.join(pdb_cache_dir, f"{name}.pdb")
        if os.path.exists(pdb_path):
            print(f"[cache] nr={nr} {name}")
            continue
        try:
            t0 = time.time()
            pdb_str, plddt = predict_structure(seq, num_recycles=nr)
            with open(pdb_path, "w") as f:
                f.write(pdb_str)
            with open(pdb_path + ".plddt", "w") as f:
                f.write(str(plddt))
            print(
                f"[pred] nr={nr} {name} len={len(seq)} plddt={plddt:.2f} ({time.time()-t0:.1f}s)"
            )
        except Exception as e:
            print(f"[pred FAIL] nr={nr} {name}: {e}")
            traceback.print_exc()

    # Stage 2: eval
    n_eval, n_correct = 0, 0
    setting_preds, setting_pp = [], []
    for name, seq, (r0, r1), is_pos in PANEL:
        pdb_path = os.path.join(pdb_cache_dir, f"{name}.pdb")
        if not os.path.exists(pdb_path):
            print(f"[skip] {name}: no cached PDB")
            setting_preds.append(-1)
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
            ss_full = ss[: len(seq)].ljust(len(seq), "L")
            region_ss = ss_full[r0 - 1 : r1]
            pred_hairpin = is_hairpin(region_ss)
            correct = int(pred_hairpin == is_pos)
            n_eval += 1
            n_correct += correct
            print(
                f"[eval nr={nr}] {name:12s} region={region_ss} pred_hp={pred_hairpin} gt_hp={is_pos} correct={correct} plddt={plddt:.2f} tool={used}"
            )
            setting_preds.append(int(pred_hairpin))
            setting_pp.append(
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
        except Exception as e:
            print(f"[eval FAIL] {name}: {e}")
            traceback.print_exc()
            setting_preds.append(-1)

    acc = n_correct / n_eval if n_eval > 0 else 0.0
    mean_plddt = (
        float(np.nanmean([p["plddt"] for p in setting_pp]))
        if setting_pp
        else float("nan")
    )
    print(
        f"\n=== nr={nr} hairpin_dssp_accuracy = {acc:.4f}  ({n_correct}/{n_eval}) mean_plddt={mean_plddt:.2f} ==="
    )

    key = f"num_recycles={nr}"
    ed = experiment_data["num_recycles"]["esmfold_hairpin_panel"]
    ed["predictions"][key] = setting_preds
    ed["per_protein"][key] = setting_pp
    ed["metrics"]["val"].append(
        {
            "num_recycles": nr,
            "epoch": 0,
            "hairpin_dssp_accuracy": acc,
            "n_eval": n_eval,
            "n_correct": n_correct,
            "mean_plddt": mean_plddt,
        }
    )
    ed["losses"]["val"].append({"num_recycles": nr, "epoch": 0, "loss": 1.0 - acc})
    ed["per_setting_summary"].append(
        {
            "num_recycles": nr,
            "accuracy": acc,
            "n_eval": n_eval,
            "n_correct": n_correct,
            "mean_plddt": mean_plddt,
        }
    )

# ---------- Plot ----------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ed = experiment_data["num_recycles"]["esmfold_hairpin_panel"]
    nrs = [s["num_recycles"] for s in ed["per_setting_summary"]]
    accs = [s["accuracy"] for s in ed["per_setting_summary"]]
    plddts = [s["mean_plddt"] for s in ed["per_setting_summary"]]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar([str(n) for n in nrs], accs, color="steelblue", alpha=0.7, label="accuracy")
    ax1.set_ylabel("hairpin_dssp_accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("num_recycles")
    ax2 = ax1.twinx()
    ax2.plot([str(n) for n in nrs], plddts, "ro-", label="mean pLDDT")
    ax2.set_ylabel("mean pLDDT")
    plt.title("ESMFold num_recycles sweep")
    fig.tight_layout()
    plt.savefig(os.path.join(working_dir, "num_recycles_sweep.png"), dpi=120)
    plt.close()

    # Per-protein grouped bar
    names = [p[0] for p in PANEL]
    gts = [int(p[3]) for p in PANEL]
    x = np.arange(len(names))
    width = 0.8 / (len(nrs) + 1)
    plt.figure(figsize=(11, 3.5))
    plt.bar(x - 0.4 + width / 2, gts, width, label="GT", color="gray")
    for i, nr in enumerate(nrs):
        preds = ed["predictions"][f"num_recycles={nr}"]
        preds_plot = [max(p, 0) for p in preds]
        plt.bar(
            x - 0.4 + width * (i + 1) + width / 2, preds_plot, width, label=f"nr={nr}"
        )
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("hairpin (0/1)")
    plt.title("Hairpin predictions per protein across num_recycles")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "num_recycles_per_protein.png"), dpi=120)
    plt.close()
except Exception as e:
    print("plot failed:", e)

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
