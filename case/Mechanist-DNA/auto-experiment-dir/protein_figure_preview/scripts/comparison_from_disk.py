"""Rebuild <DNA_name>/<DNA_name>_comparison.png from whatever seed_*/dose_*.{pdb,png} are on disk
(readouts recomputed from the PDBs, so it tolerates incrementally-added seeds). Green = alpha-helix;
beta-sheet & coil = grey. Usage: python comparison_from_disk.py <DNA_name> [<DNA_name> ...]"""
import os, sys, glob, re
ROOT = "/data/wanghaoxiong/intergene_mechanist_v6"
sys.path.insert(0, os.path.join(ROOT, "code"))
from fold import structural_readout
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PREVIEW = os.path.join(ROOT, "protein_figure_preview")
DOSES = [0, 2, 4, 8]
dnas_meta = json.load(open(os.path.join(PREVIEW, "scripts/all_dnas.json")))


def title(name):
    d = dnas_meta.get(name, {})
    org = {"human": "human", "ecoli": "E. coli"}.get(d.get("organism"), d.get("organism", ""))
    pdb = f", PDB {d['pdb']}" if d.get("pdb") else ""
    return f"{name}  ({org}{pdb})  —  orig-19 + Encoder-Clamp-Decoder"


for name in sys.argv[1:]:
    ddir = os.path.join(PREVIEW, name)
    seeds = sorted(int(re.match(r"seed_(\d+)", d).group(1))
                   for d in os.listdir(ddir) if re.match(r"seed_(\d+)$", d))
    if not seeds:
        print(f"[cmp] {name}: no seeds"); continue
    fig, axes = plt.subplots(len(seeds), len(DOSES), figsize=(2.6 * len(DOSES), 2.5 * len(seeds)),
                             squeeze=False)
    for r, seed in enumerate(seeds):
        for c, dose in enumerate(DOSES):
            ax = axes[r][c]; ax.axis("off")
            pdb = os.path.join(ddir, f"seed_{seed}", f"dose_{dose}.pdb")
            png = os.path.join(ddir, f"seed_{seed}", f"dose_{dose}.png")
            rd = structural_readout(open(pdb).read()) if os.path.exists(pdb) else None
            if rd and os.path.exists(png):
                ax.imshow(mpimg.imread(png))
                plddt = rd.get("struct_mean_plddt", rd.get("mean_plddt"))
                if plddt is not None and plddt <= 1.5:   # b-factor pLDDT stored 0-1 -> 0-100
                    plddt *= 100
                ax.set_title(f"helix {rd['helix_hgi_w']*100:.0f}%  pLDDT {plddt:.0f}",
                             fontsize=8, pad=1)
            else:
                ax.text(0.5, 0.5, "invalid ORF", ha="center", va="center", color="0.6",
                        fontsize=8, transform=ax.transAxes)
            if r == 0:
                ax.text(0.5, 1.20, f"dose {dose}", ha="center", va="bottom", fontsize=12,
                        fontweight="bold", transform=ax.transAxes)
        axes[r][0].text(-0.10, 0.5, f"seed_{seed}", ha="right", va="center", fontsize=9,
                        fontweight="bold", rotation=90, transform=axes[r][0].transAxes)
    fig.suptitle(f"{title(name)}     (green = α-helix; β-sheet & coil = grey)  —  {len(seeds)} seeds",
                 fontsize=12, fontweight="bold", y=0.996)
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.97])
    out = os.path.join(ddir, f"{name}_comparison.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[cmp] wrote {out}  ({len(seeds)} seeds)")
