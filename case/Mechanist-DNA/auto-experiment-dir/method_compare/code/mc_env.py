"""Path/env patching for the method_compare study.

The original round-2 code hard-codes /data1/share_model/... which no longer exists on this
machine. This module re-points every asset and must be imported (and patch() called) before
any round-2 module is used.
"""
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R2_CODE = os.path.join(ROOT, "code")
MC = os.path.join(ROOT, "method_compare")
if R2_CODE not in sys.path:
    sys.path.insert(0, R2_CODE)

EVO2_PATH = "/mnt/quarkfs/share_model/evo2_7b/evo2_7b.pt"
SAE_PATH = "/mnt/quarkfs/share_model/Evo-2-Layer-26-Mixed/sae-layer26-mixed-expansion_8-k_64.pt"
ESMFOLD_PATH = "/mnt/quarkfs/share_model/esmfold_v1"
ESM2_650M_PATH = "/mnt/quarkfs/share_model/esm2_t33_650M_UR50D"
DATA_DIR = os.path.join(ROOT, "data")
DSSP_TMP = os.path.join(ROOT, "method_compare", "dssp_tmp")

# frozen round-2 assets (NOT re-derived)
FEATURE_SET = os.path.join(ROOT, "results", "m0_feature_set.json")
CALIB = os.path.join(ROOT, "results", "m1_calibration.json")
C_STAR = 21.4801          # interior optimum in sigma_proj units (round 2, frozen)
ORGANISM = "prokaryote"

_patched = False


def patch():
    global _patched
    if _patched:
        return
    os.makedirs(DSSP_TMP, exist_ok=True)
    import evo2_sae, m0_data
    evo2_sae.EVO2_7B_PATH = EVO2_PATH
    evo2_sae.SAE_PATH = SAE_PATH
    m0_data.DATA_DIR = DATA_DIR

    import fold
    fold.DSSP_TMP = DSSP_TMP

    def _load_esmfold(device="cuda:0"):
        if fold._esmfold["model"] is None:
            from transformers import EsmForProteinFolding, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(ESMFOLD_PATH)
            model = EsmForProteinFolding.from_pretrained(ESMFOLD_PATH).eval().to(device)
            model.esm = model.esm.half()
            model.trunk.set_chunk_size(64)
            fold._esmfold["model"] = model
            fold._esmfold["tok"] = tok
        return fold._esmfold["model"], fold._esmfold["tok"]

    fold.load_esmfold = _load_esmfold
    _patched = True
