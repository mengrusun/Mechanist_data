"""Batch green-helix PyMOL render: one pymol session, reinitialize per PDB (avoids per-file startup).
Helix -> green, sheet -> yellow, coil -> grey (matches code/render_pdb_pymol.py).
"""
import sys, json
from pymol import cmd

HELIX_GREEN = "0x4d9968"
COIL_GREY = "0xc7c9cc"
SHEET_YELLOW = "0xdccb4a"


def render(pdb_path, out_png, width=520, height=420):
    cmd.reinitialize()
    cmd.load(pdb_path, "obj")
    cmd.hide("everything")
    cmd.remove("not polymer")
    cmd.dss("obj")
    cmd.show("cartoon", "obj")
    cmd.color(COIL_GREY, "obj")
    cmd.color(HELIX_GREEN, "obj and ss H")
    # beta-sheet intentionally NOT colored -> rendered grey like coil (no yellow); only helix highlighted
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_highlight_color", "grey50")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("depth_cue", 0)
    cmd.set("specular", 0.15)
    cmd.set("ambient", 0.4)
    cmd.orient("obj")
    cmd.zoom("obj", buffer=3)
    cmd.png(out_png, width=width, height=height, dpi=200, ray=1)


argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
manifest = json.load(open(argv[0]))
for i, m in enumerate(manifest):
    try:
        render(m["pdb"], m["png"])
        print(f"[render] {i+1}/{len(manifest)} {m['png']}", flush=True)
    except Exception as e:
        print(f"[render] FAIL {m['pdb']}: {e}", flush=True)
print("[render] DONE")
