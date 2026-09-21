"""
M0 data pipeline (HC3-compliant): natural CDS + real DSSP labels from experimental PDB.

For a target organism:
  1. UniProt REST: reviewed proteins with an experimental 3D structure -> sequence, PDB xrefs, EMBL CDS xrefs.
  2. CDS retrieval (natural, no reverse-translation): EMBL ProteinId -> NCBI GenPept `coded_by`
     -> fetch nuccore region -> reverse-complement if needed -> the natural CDS. Verified by exact
     translation == UniProt sequence.
  3. Structure labels: download experimental PDB -> run mkdssp -> per-residue 8-state SS.
  4. Alignment: pairwise-align the DSSP chain AA sequence to the CDS-translated protein; propagate
     SS to matched, identical residues -> per-codon SS label. Drop mismatched/unaligned/low-coverage.

Output per protein: {acc, organism, cds_nt (str), codon_ss (list of 8-state chars or None),
                     n_codons, coverage, pdb_id, chain}.
Reverse-translation is never used.
"""
import os, re, json, time, urllib.request, urllib.parse, subprocess, gzip, io, hashlib
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio import Align

Entrez.email = "wanghaoxiong@zju.edu.cn"
DATA_DIR = "/data/wanghaoxiong/intergene_mechanist_v6/data"
PDB_DIR = os.path.join(DATA_DIR, "pdb")
DSSP_DIR = os.path.join(DATA_DIR, "dssp")
CACHE_DIR = os.path.join(DATA_DIR, "cds_cache")
for d in (DATA_DIR, PDB_DIR, DSSP_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)

# transl_table: bacteria/archaea = 11, standard eukaryote = 1
ORGANISMS = {
    "prokaryote": {"taxid": 83333, "name": "E. coli K-12", "transl_table": 11},
    "eukaryote":  {"taxid": 9606,  "name": "Homo sapiens", "transl_table": 1},
}

HELIX_HGI = set("HGI")
HELIX_H = set("H")
SHEET = set("E")  # DSSP strand (E) for beta-sheet off-target readout; B is isolated bridge


def _get(url, timeout=60, accept="application/json"):
    req = urllib.request.Request(url, headers={"User-Agent": "mozilla/5.0", "Accept": accept})
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def uniprot_structured_proteins_paged(taxid, max_proteins=6000, page_size=500):
    base = "https://rest.uniprot.org/uniprotkb/search"
    q = f"(organism_id:{taxid}) AND (reviewed:true) AND (structure_3d:true)"
    fields = "accession,sequence,xref_pdb,xref_embl,length"
    url = f"{base}?query={urllib.parse.quote(q)}&fields={fields}&format=json&size={page_size}"
    out = []
    while url and len(out) < max_proteins:
        req = urllib.request.Request(url, headers={"User-Agent": "mozilla/5.0", "Accept": "application/json"})
        resp = urllib.request.urlopen(req, timeout=90)
        raw = resp.read()
        link = resp.headers.get("Link", "")
        data = json.loads(raw)
        for r in data.get("results", []):
            acc = r["primaryAccession"]; seq = r["sequence"]["value"]
            xr = r.get("uniProtKBCrossReferences", [])
            pdbs = []
            for x in xr:
                if x.get("database") == "PDB":
                    props = {p["key"]: p["value"] for p in x.get("properties", [])}
                    pdbs.append({"id": x["id"], "method": props.get("Method"),
                                 "resolution": props.get("Resolution"), "chains": props.get("Chains")})
            embl = []
            for x in xr:
                if x.get("database") == "EMBL":
                    props = {p["key"]: p["value"] for p in x.get("properties", [])}
                    if props.get("ProteinId") and props.get("ProteinId") != "-":
                        embl.append((x["id"], props["ProteinId"], props.get("MoleculeType")))
            if pdbs and embl:
                out.append({"acc": acc, "seq": seq, "pdbs": pdbs, "embl": embl})
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
        time.sleep(0.2)
    return out[:max_proteins]


def fetch_cds_for_protein(embl_list, uniprot_seq, transl_table):
    """Try each EMBL ProteinId; return the CDS nucleotide string whose translation matches uniprot_seq."""
    for nuc_acc, protein_id, moltype in embl_list:
        cache = os.path.join(CACHE_DIR, f"{protein_id}.txt")
        if os.path.exists(cache):
            cds = open(cache).read().strip()
            if _translation_matches(cds, uniprot_seq, transl_table):
                return cds, protein_id
            continue
        try:
            cds = _fetch_cds_via_codedby(protein_id, transl_table)
            if cds and _translation_matches(cds, uniprot_seq, transl_table):
                with open(cache, "w") as f:
                    f.write(cds)
                return cds, protein_id
        except Exception:
            continue
        time.sleep(0.34)  # NCBI rate limit (3/s without key)
    return None, None


def _fetch_cds_via_codedby(protein_id, transl_table, retries=3):
    for attempt in range(retries):
        try:
            h = Entrez.efetch(db="protein", id=protein_id, rettype="gp", retmode="text")
            rec = SeqIO.read(h, "genbank"); h.close()
            coded_by = None
            for feat in rec.features:
                if feat.type == "CDS" and "coded_by" in feat.qualifiers:
                    coded_by = feat.qualifiers["coded_by"][0]; break
            if not coded_by:
                return None
            comp = coded_by.startswith("complement")
            acc = re.search(r'([A-Za-z]\w+\.\d+)', coded_by).group(1)
            # coded_by may be join(...) with multiple segments
            segs = re.findall(r'(\d+)\.\.>?(\d+)', coded_by)
            pieces = []
            for s, e in segs:
                s, e = int(s), int(e)
                hh = Entrez.efetch(db="nuccore", id=acc, rettype="fasta", retmode="text",
                                   seq_start=s, seq_stop=e)
                nrec = SeqIO.read(hh, "fasta"); hh.close()
                pieces.append(str(nrec.seq))
                time.sleep(0.34)
            cds = "".join(pieces)
            if comp:
                cds = str(Seq(cds).reverse_complement())
            return cds
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _translation_matches(cds, uniprot_seq, transl_table):
    if not cds or len(cds) % 3 != 0:
        # allow trailing partial; trim
        cds = cds[: len(cds) - (len(cds) % 3)]
    try:
        t = str(Seq(cds).translate(table=transl_table)).rstrip("*")
    except Exception:
        return False
    if t == uniprot_seq:
        return True
    # tolerate leading fMet/Val start substitution & length equality
    if len(t) == len(uniprot_seq) and t[1:] == uniprot_seq[1:]:
        return True
    return False


def download_pdb(pdb_id):
    pdb_id = pdb_id.lower()
    loc = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if os.path.exists(loc) and os.path.getsize(loc) > 0:
        return loc
    for url in (f"https://files.rcsb.org/download/{pdb_id}.pdb",
                f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"):
        try:
            data = _get(url, accept="text/plain")
            if data and b"ATOM" in data:
                with open(loc, "wb") as f:
                    f.write(data)
                return loc
        except Exception:
            continue
    return None


def run_dssp(pdb_path):
    """Run mkdssp; return {chain: [(resnum, aa, ss), ...]}."""
    out_dssp = pdb_path.replace(".pdb", ".dssp").replace(PDB_DIR, DSSP_DIR)
    if not (os.path.exists(out_dssp) and os.path.getsize(out_dssp) > 0):
        r = subprocess.run(["mkdssp", pdb_path, out_dssp], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_dssp):
            r = subprocess.run(["mkdssp", "--output-format", "dssp", pdb_path, out_dssp],
                               capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_dssp):
            return None
    return _parse_dssp(out_dssp)


def _parse_dssp(path):
    from Bio.PDB.DSSP import make_dssp_dict
    try:
        d, keys = make_dssp_dict(path)
    except Exception:
        return None
    chains = {}
    for k in keys:
        chain, (het, resnum, icode) = k
        aa = d[k][0]
        ss = d[k][1]
        if ss == "-" or ss == " ":
            ss = "-"
        chains.setdefault(chain, []).append((resnum, aa, ss))
    return chains


_aligner = Align.PairwiseAligner()
_aligner.mode = "global"
_aligner.open_gap_score = -10
_aligner.extend_gap_score = -0.5
_aligner.match_score = 2
_aligner.mismatch_score = -1


def align_ss_to_cds(cds, protein_seq, dssp_chains, transl_table, min_coverage=0.5):
    """Return per-codon SS list (len == n_codons) with 8-state chars or None, plus coverage & chain."""
    n_codons = len(protein_seq)
    best = None
    for chain, residues in dssp_chains.items():
        chain_aa = "".join(a if a not in ("X", "!", "*") and a.isalpha() else "X" for _, a, _ in residues)
        chain_ss = [s for _, _, s in residues]
        if len(chain_aa) < 20:
            continue
        aln = _aligner.align(protein_seq, chain_aa)[0]
        # map chain positions -> protein positions
        codon_ss = [None] * n_codons
        pi = qi = 0
        matched = 0
        # walk aligned blocks
        a_prot, a_chain = aln.aligned  # arrays of [start,end] index pairs
        for (ps, pe), (qs, qe) in zip(a_prot, a_chain):
            for off in range(pe - ps):
                p_idx = ps + off; q_idx = qs + off
                if protein_seq[p_idx] == chain_aa[q_idx] and chain_aa[q_idx] != "X":
                    codon_ss[p_idx] = chain_ss[q_idx]
                    matched += 1
        cov = matched / n_codons
        if best is None or cov > best[1]:
            best = (codon_ss, cov, chain)
    if best is None or best[1] < min_coverage:
        return None, (best[1] if best else 0.0), (best[2] if best else None)
    return best[0], best[1], best[2]


def helix_label(ss_char, helix_def="HGI"):
    if ss_char is None:
        return None
    hset = HELIX_HGI if helix_def == "HGI" else HELIX_H
    return 1 if ss_char in hset else 0


def sheet_label(ss_char):
    if ss_char is None:
        return None
    return 1 if ss_char in SHEET else 0
