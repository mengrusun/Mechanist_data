"""
Build the labeled contrastive protein/DNA dataset from real experimental structures.

Pipeline (no GPU):
  1. Query RCSB for single-protein-chain X-ray entries (moderate length, good resolution).
  2. Download each PDB, run DSSP -> (aa_seq, ss8), compute experimental %H and %E.
  3. Keep clean chains; split disjoint train / heldout by protein.
  4. Label proteins: high-alpha (%H >= HI_H) vs low-alpha (%H <= LO_H and %E >= LO_E_MIN).
  5. Reverse-translate each labeled protein to a realistic CDS window (synonymous codons)
     for M1 direction extraction.

Outputs (data/):
  proteins.json      all clean proteins with experimental DSSP labels + split
  e1_val.json        held-out proteins for E1 (aa, exp %H, exp %E)
  contrastive.json   high/low-alpha DNA windows with split
"""
import os, sys, json, urllib.request, urllib.parse, subprocess, tempfile, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA, exist_ok=True)
sys.path.insert(0, HERE)
import evo2lib as E
from scorer import dssp_ss_from_pdb, pct_helix_from_ss8, pct_sheet_from_ss8

HI_H = 45.0      # high-alpha if exp %H >= 45
LO_H = 18.0      # low-alpha if exp %H <= 18
LO_E_MIN = 18.0  # ... and %E >= 18 (genuinely beta/other, not just disordered)
MIN_LEN, MAX_LEN = 60, 170   # residues per window (cap for generation/fold speed)

def rcsb_ids(n=900, seed=0):
    q = {
      "query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.experimental_method", "operator": "exact_match", "value": "X-ray"}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "less", "value": 2.2}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count", "operator": "range", "value": {"from": 60, "to": 200}}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein", "operator": "equals", "value": 1}},
      ]},
      "return_type": "entry",
      "request_options": {"paginate": {"start": 0, "rows": n}, "results_content_type": ["experimental"],
                          "sort": [{"sort_by": "rcsb_entry_info.deposited_atom_count", "direction": "asc"}]}
    }
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(json.dumps(q))
    r = json.load(urllib.request.urlopen(url, timeout=60))
    ids = [x["identifier"] for x in r["result_set"]]
    random.Random(seed).shuffle(ids)
    return ids

def fetch_one(pid):
    try:
        url = f"https://files.rcsb.org/download/{pid}.pdb"
        with tempfile.NamedTemporaryFile("wb", suffix=".pdb", delete=False) as f:
            f.write(urllib.request.urlopen(url, timeout=30).read()); p = f.name
        r = dssp_ss_from_pdb(p)
        os.unlink(p)
        if r is None: return None
        aa, ss8 = r
        aa = aa.replace("X", "")  # DSSP may give X for hetero; drop
        if not (MIN_LEN <= len(ss8) <= 400): return None
        if any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in aa[:len(ss8)]):
            aa = "".join(c for c in aa if c in "ACDEFGHIKLMNPQRSTVWY")
        n = min(len(aa), len(ss8))
        if n < MIN_LEN: return None
        return {"pdb": pid, "aa": aa[:n], "ss8": ss8[:n],
                "pctH": round(pct_helix_from_ss8(ss8[:n]), 2),
                "pctE": round(pct_sheet_from_ss8(ss8[:n]), 2)}
    except Exception:
        return None

def build(n_query=900, seed=0):
    ids = rcsb_ids(n_query, seed)
    print(f"[data] querying {len(ids)} candidate PDB entries", flush=True)
    prots = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(fetch_one, pid): pid for pid in ids}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r: prots.append(r)
            if (i+1) % 100 == 0:
                print(f"[data] processed {i+1}/{len(ids)}, kept {len(prots)}", flush=True)
    # dedup by aa
    seen = set(); uniq = []
    for p in prots:
        key = p["aa"][:60]
        if key in seen: continue
        seen.add(key); uniq.append(p)
    prots = uniq
    print(f"[data] total clean proteins: {len(prots)}", flush=True)

    # disjoint split
    rng = random.Random(123); rng.shuffle(prots)
    n_val = max(60, int(0.25 * len(prots)))
    for i, p in enumerate(prots):
        p["split"] = "heldout" if i < n_val else "train"

    # label alpha/beta
    def label(p):
        if p["pctH"] >= HI_H: return "high_alpha"
        if p["pctH"] <= LO_H and p["pctE"] >= LO_E_MIN: return "low_alpha"
        return "mid"
    for p in prots:
        p["label"] = label(p)

    with open(os.path.join(DATA, "proteins.json"), "w") as f:
        json.dump(prots, f)
    # E1 validation set = held-out proteins (all, for correlation of predicted vs DSSP %H)
    e1 = [{"pdb": p["pdb"], "aa": p["aa"], "ss8": p["ss8"], "expH": p["pctH"], "expE": p["pctE"]}
          for p in prots if p["split"] == "heldout"]
    with open(os.path.join(DATA, "e1_val.json"), "w") as f:
        json.dump(e1, f)

    # contrastive DNA windows (synonymous reverse translation)
    gen = np.random.default_rng(7)
    windows = []
    for p in prots:
        if p["label"] not in ("high_alpha", "low_alpha"): continue
        aa = p["aa"][:MAX_LEN]
        if len(aa) < MIN_LEN: continue
        dna = E.synonymous_rev_translate(aa, gen)
        windows.append({"pdb": p["pdb"], "label": p["label"], "split": p["split"],
                        "aa": aa, "dna": dna, "expH": p["pctH"], "expE": p["pctE"]})
    with open(os.path.join(DATA, "contrastive.json"), "w") as f:
        json.dump(windows, f)

    from collections import Counter
    c_all = Counter(p["label"] for p in prots)
    c_tr = Counter(w["label"] for w in windows if w["split"] == "train")
    c_ho = Counter(w["label"] for w in windows if w["split"] == "heldout")
    summary = {"n_proteins": len(prots), "n_heldout": len(e1),
               "label_counts": dict(c_all),
               "contrastive_train": dict(c_tr), "contrastive_heldout": dict(c_ho),
               "n_windows": len(windows)}
    print("[data] summary:", json.dumps(summary), flush=True)
    with open(os.path.join(DATA, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

if __name__ == "__main__":
    build(n_query=int(sys.argv[1]) if len(sys.argv) > 1 else 900)
