"""mode=manifest: build render manifest (pdb -> seed_<s>/dose_<d>.png) from the shard jsons.
mode=montage: build <DNA_name>/<DNA_name>_comparison.png (rows=seeds, cols=doses).
Green = alpha-helix; beta-sheet & coil = grey (no yellow)."""
import os, sys, json, glob
ROOT = "/data/wanghaoxiong/intergene_mechanist_v6"
PREVIEW = os.path.join(ROOT, "protein_figure_preview")
DOSES = [0, 2, 4, 8]
SEEDS = [101, 200, 300, 111, 222]
mode = sys.argv[1] if len(sys.argv) > 1 else "manifest"

dnas = json.load(open(os.path.join(PREVIEW, "scripts/all_dnas.json")))
entries = []
for f in sorted(glob.glob(os.path.join(PREVIEW, "_shards", "shard*.json"))):
    entries += json.load(open(f))["entries"]
idx = {(e["dna"], e["seed"], e["dose"]): e for e in entries}


def png_path(dna, seed, dose):
    return os.path.join(PREVIEW, dna, f"seed_{seed}", f"dose_{dose}.png")


def title(name):
    d = dnas[name]
    org = {"human": "human", "ecoli": "E. coli"}.get(d["organism"], d["organism"])
    pdb = f", PDB {d['pdb']}" if d.get("pdb") else ""
    return f"{name}  ({org}{pdb})  —  orig-19 + Encoder-Clamp-Decoder"


if mode == "manifest":
    manifest = [{"pdb": e["pdb_path"], "png": png_path(e["dna"], e["seed"], e["dose"])}
                for e in entries if e.get("pdb_path")]
    json.dump(manifest, open(os.path.join(PREVIEW, "_shards", "_manifest.json"), "w"), indent=2)
    print(f"[norm] manifest {len(manifest)} panels")

elif mode == "montage":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    for name in dnas:
        fig, axes = plt.subplots(len(SEEDS), len(DOSES),
                                 figsize=(2.6 * len(DOSES), 2.5 * len(SEEDS)), squeeze=False)
        for r, seed in enumerate(SEEDS):
            for c, dose in enumerate(DOSES):
                ax = axes[r][c]; ax.axis("off")
                e = idx.get((name, seed, dose)); st = e.get("struct") if e else None
                png = png_path(name, seed, dose)
                if st and os.path.exists(png):
                    ax.imshow(mpimg.imread(png))
                    ax.set_title(f"helix {st['helix_hgi_w']*100:.0f}%  pLDDT {st['struct_mean_plddt']:.0f}",
                                 fontsize=8, pad=1)
                else:
                    ax.text(0.5, 0.5, "invalid ORF", ha="center", va="center", color="0.6",
                            fontsize=8, transform=ax.transAxes)
                if r == 0:
                    ax.text(0.5, 1.20, f"dose {dose}", ha="center", va="bottom", fontsize=12,
                            fontweight="bold", transform=ax.transAxes)
            axes[r][0].text(-0.10, 0.5, f"seed_{seed}", ha="right", va="center", fontsize=9,
                            fontweight="bold", rotation=90, transform=axes[r][0].transAxes)
        fig.suptitle(f"{title(name)}     (green = α-helix; β-sheet & coil = grey)",
                     fontsize=12, fontweight="bold", y=0.995)
        fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.96])
        out = os.path.join(PREVIEW, name, f"{name}_comparison.png")
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"[norm] wrote {out}")
