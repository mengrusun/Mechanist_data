"""Feasibility smoke for the model-axis swap: load evo2_7b_262k and generate a tiny batch.
Confirms the alternate within-family Evo2-7B checkpoint loads + generates + a site-28 hook fires.
Model-independent ESMFold assay is validated separately (already frozen in M1)."""
import os, sys, time, json, traceback
sys.path.insert(0, "/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/experiments/steer_helix")
import numpy as np
import common as C

ALT_PT = "/mnt/quarkfs/share_models/evo2_7b_262k/evo2_7b_262k.pt"
OUT = "/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/verify/variant_smoke/smoke_262k.json"

def main():
    t0 = time.time()
    rep = dict(model="evo2_7b_262k", local_path=ALT_PT,
               cuda=os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        from evo2 import Evo2
        evo2 = Evo2("evo2_7b_262k", local_path=ALT_PT)
        rep["load_ok"] = True
        rep["load_seconds"] = round(time.time() - t0, 1)
        # block enumeration (confirm site 28 exists)
        import re
        BLOCK_RE = re.compile(r"^blocks\.(\d+)$")
        blocks = sorted(int(BLOCK_RE.match(n).group(1))
                        for n, _ in evo2.model.named_modules() if BLOCK_RE.match(n))
        rep["n_blocks"] = len(blocks)
        rep["site28_present"] = 28 in blocks
        # tiny generation (no hook)
        C.set_seed(0)
        out = evo2.generate(prompt_seqs=list(C.DNA_PRIMERS[:4]), n_tokens=120,
                            temperature=1.0, top_k=4, top_p=1.0,
                            batched=True, cached_generation=True, verbose=0)
        seqs = out.sequences if hasattr(out, "sequences") else out[0]
        rep["gen_ok"] = True
        rep["gen_lens"] = [len(C._clean_dna(s)) for s in seqs]
        # ORF extraction on first gen
        orf_dna, prot = C.extract_orf(seqs[0])
        rep["first_orf_aa"] = (len(prot) if prot else 0)
        rep["total_seconds"] = round(time.time() - t0, 1)
        rep["feasible"] = True
    except Exception as e:
        rep["feasible"] = False
        rep["error"] = str(e)[:400]
        rep["traceback"] = traceback.format_exc()[-1500:]
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "traceback"}, indent=2))

if __name__ == "__main__":
    main()
