"""S0: build the labeled coding-DNA dataset.

E. coli K-12 MG1655 (RefSeq GCF_000005845.2) CDS  <->  RefSeq protein  <->
AlphaFold-DB proteome (UP000000625) structure.  Provenance = adapted.

Pipeline:
  1. parse CDS (dna + UniProt acc from db_xref) and RefSeq proteins
  2. for each AlphaFold PDB: DSSP -> per-residue 8-state -> 3-state; read its AA seq
  3. join CDS -> UniProt -> AF structure; require exact AA-sequence agreement
     (translated CDS == RefSeq protein == AF structure sequence)
  4. map per-residue SS to the 3 codon nucleotide positions
  5. mmseqs2 cluster proteins at 30% identity -> split clusters 70/15/15
  6. write data/ecoli_labeled.parquet (per-codon rows) + data/splits.json

DSSP labels are used for FEATURE SELECTION only (C1) and are independent of the
ESMFold readout used for generation (C2/C3).
"""
import os, sys, re, gzip, json, glob, tarfile, subprocess, tempfile, shutil
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import common as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CDS_FA  = os.path.join(DATA, "GCF_000005845.2_ASM584v2_cds_from_genomic.fna.gz")
PROT_FA = os.path.join(DATA, "GCF_000005845.2_ASM584v2_protein.faa.gz")
AF_TAR  = os.path.join(DATA, "UP000000625_ECOLI_v4.tar")
AF_DIR  = os.path.join(DATA, "af_pdb")
OUT_PARQUET = os.path.join(DATA, "ecoli_labeled.parquet")
OUT_GENES   = os.path.join(DATA, "ecoli_genes.parquet")
OUT_SPLITS  = os.path.join(DATA, "splits.json")

# ---------------------------------------------------------------- FASTA parsing
def read_fasta(path):
    op = gzip.open if path.endswith(".gz") else open
    hdr, seq = None, []
    with op(path, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(seq)
                hdr, seq = line[1:], []
            else:
                seq.append(line)
    if hdr is not None:
        yield hdr, "".join(seq)

def parse_cds():
    """Return list of dicts: protein_id, gene, locus_tag, uniprot, dna."""
    recs = []
    for hdr, seq in read_fasta(CDS_FA):
        pid = re.search(r"\[protein_id=([^\]]+)\]", hdr)
        gene = re.search(r"\[gene=([^\]]+)\]", hdr)
        locus = re.search(r"\[locus_tag=([^\]]+)\]", hdr)
        uni = re.search(r"\[db_xref=[^\]]*UniProtKB[^:]*:([A-Z0-9]+)", hdr)
        pseudo = "[pseudo=true]" in hdr
        if pid is None or pseudo:
            continue
        recs.append(dict(protein_id=pid.group(1),
                         gene=gene.group(1) if gene else "",
                         locus_tag=locus.group(1) if locus else "",
                         uniprot=uni.group(1) if uni else "",
                         dna=seq.upper()))
    return recs

def parse_proteins():
    d = {}
    for hdr, seq in read_fasta(PROT_FA):
        pid = hdr.split()[0]
        d[pid] = seq.upper().replace("*", "")
    return d

# ---------------------------------------------------------------- AlphaFold/DSSP
def extract_af_tar():
    if os.path.isdir(AF_DIR) and glob.glob(os.path.join(AF_DIR, "*.pdb.gz")):
        return
    os.makedirs(AF_DIR, exist_ok=True)
    print(f"[s0] extracting {AF_TAR} ...", flush=True)
    with tarfile.open(AF_TAR) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".pdb.gz"):
                m.name = os.path.basename(m.name)
                tf.extract(m, AF_DIR)
    print(f"[s0] extracted {len(glob.glob(os.path.join(AF_DIR,'*.pdb.gz')))} pdb.gz", flush=True)

AA3to1 = {
 'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
 'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
 'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

def uniprot_from_af_name(fn):
    m = re.match(r"AF-([A-Z0-9]+)-F\d+-model", os.path.basename(fn))
    return m.group(1) if m else None

def dssp_one(pdb_gz):
    """Run DSSP on one gz'd AF PDB. Return (uniprot, aa_seq, ss3) or None."""
    uni = uniprot_from_af_name(pdb_gz)
    if uni is None:
        return None
    tmpd = tempfile.mkdtemp()
    try:
        pdb = os.path.join(tmpd, "m.pdb")
        with gzip.open(pdb_gz, "rt") as fi, open(pdb, "w") as fo:
            fo.write(fi.read())
        # AA sequence from CA atoms (model order) as ground reference
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        st = parser.get_structure("s", pdb)
        model = st[0]
        chain = list(model.get_chains())[0]
        aa = []
        for res in chain:
            nm = res.get_resname()
            if nm in AA3to1:
                aa.append(AA3to1[nm])
        aa_seq = "".join(aa)
        ss8 = C.run_dssp_on_pdb(pdb)
        if not ss8 or len(ss8) != len(aa_seq):
            # DSSP residue count mismatch -> skip (rare)
            if not ss8:
                return None
        ss3 = C.three_state(ss8)
        return (uni, aa_seq, ss3)
    except Exception as e:
        sys.stderr.write(f"[dssp] {pdb_gz}: {e}\n")
        return None
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

def build_af_labels(nproc=16):
    pdbs = sorted(glob.glob(os.path.join(AF_DIR, "*.pdb.gz")))
    print(f"[s0] running DSSP on {len(pdbs)} structures with {nproc} procs ...", flush=True)
    out = {}
    with Pool(nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(dssp_one, pdbs, chunksize=8)):
            if r is not None:
                uni, aa_seq, ss3 = r
                if len(ss3) == len(aa_seq):
                    out[uni] = (aa_seq, ss3)
            if (i+1) % 500 == 0:
                print(f"[s0]   dssp {i+1}/{len(pdbs)} ok={len(out)}", flush=True)
    print(f"[s0] AF labels for {len(out)} structures", flush=True)
    return out

# ---------------------------------------------------------------- mmseqs cluster
def mmseqs_cluster(prot_by_gene, min_seq_id=0.30):
    """Cluster the kept genes' proteins at 30% identity. Return gene->cluster id."""
    work = tempfile.mkdtemp(prefix="mmseqs_", dir=DATA)
    fa = os.path.join(work, "seqs.fasta")
    with open(fa, "w") as fh:
        for g, aa in prot_by_gene.items():
            fh.write(f">{g}\n{aa}\n")
    db = os.path.join(work, "db")
    clu = os.path.join(work, "clu")
    tsv = os.path.join(work, "clu.tsv")
    tmp = os.path.join(work, "tmp")
    env = dict(os.environ)
    subprocess.run(["mmseqs", "createdb", fa, db], check=True, capture_output=True)
    subprocess.run(["mmseqs", "cluster", db, clu, tmp,
                    "--min-seq-id", str(min_seq_id), "-c", "0.8",
                    "--cov-mode", "0", "--cluster-mode", "0"],
                   check=True, capture_output=True, env=env)
    subprocess.run(["mmseqs", "createtsv", db, db, clu, tsv], check=True, capture_output=True)
    gene2cluster = {}
    with open(tsv) as fh:
        for line in fh:
            rep, member = line.rstrip().split("\t")[:2]
            gene2cluster[member] = rep
    shutil.rmtree(work, ignore_errors=True)
    return gene2cluster

def split_clusters(clusters, seed=42, fracs=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    uniq = sorted(set(clusters))
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(fracs[0]*n)); n_va = int(round(fracs[1]*n))
    tr = set(uniq[:n_tr]); va = set(uniq[n_tr:n_tr+n_va]); te = set(uniq[n_tr+n_va:])
    def lab(c): return "train" if c in tr else ("val" if c in va else "test")
    return lab

# ---------------------------------------------------------------- main
def main():
    os.makedirs(DATA, exist_ok=True)
    print("[s0] parsing CDS + proteins ...", flush=True)
    cds = parse_cds()
    prot = parse_proteins()
    print(f"[s0] CDS={len(cds)} proteins={len(prot)}", flush=True)

    extract_af_tar()
    af = build_af_labels(nproc=16)

    # join
    rows = []
    gene_rows = []
    prot_by_gene = {}
    dna_by_gene = {}
    ss3_by_gene = {}
    n_join, n_uni, n_seqmatch = 0, 0, 0
    kept_genes = []
    for r in cds:
        pid, uni, dna = r["protein_id"], r["uniprot"], r["dna"]
        gene_key = r["locus_tag"] or pid
        aa_ref = prot.get(pid)
        aa_dna = C.translate(dna, stop_at_stop=True)
        if aa_ref is None:
            continue
        # translated CDS must match RefSeq protein (drop mismatches/isoforms)
        if aa_dna != aa_ref:
            continue
        if not uni or uni not in af:
            continue
        n_uni += 1
        af_seq, ss3 = af[uni]
        if af_seq != aa_ref:
            continue   # structure sequence must match exactly
        n_seqmatch += 1
        L = len(aa_ref)
        if L < 50:      # need >=50 aa proteins for downstream (also QC floor)
            continue
        # length check: dna should have L codons (+ stop). require exact.
        if len(dna) < 3*L:
            continue
        # per-codon rows
        gc = C.gc_content(dna)
        for ci in range(L):
            aa = aa_ref[ci]
            st = ss3[ci]                 # 'H'/'E'/'C'
            helix = 1 if st == "H" else 0
            codon = dna[3*ci:3*ci+3]
            rows.append((gene_key, uni, ci, aa, st, helix,
                         (C.gc_content(codon)), pid))
        prot_by_gene[gene_key] = aa_ref
        dna_by_gene[gene_key] = dna
        ss3_by_gene[gene_key] = ss3[:L]
        kept_genes.append(gene_key)
        n_join += 1
    print(f"[s0] joined genes={n_join} (uniprot-in-af={n_uni}, seq-match={n_seqmatch})", flush=True)
    print(f"[s0] total per-codon rows={len(rows)}", flush=True)

    # cluster + split
    print("[s0] mmseqs clustering at 30% id ...", flush=True)
    gene2cluster = mmseqs_cluster(prot_by_gene, min_seq_id=0.30)
    # genes not in tsv (singletons missing) -> own cluster
    for g in prot_by_gene:
        gene2cluster.setdefault(g, g)
    labfn = split_clusters(gene2cluster)

    df = pd.DataFrame(rows, columns=["gene", "uniprot", "codon_idx", "aa",
                                     "ss3", "helix", "codon_gc", "protein_id"])
    df["cluster"] = df["gene"].map(gene2cluster)
    df["split"] = df["cluster"].map(labfn)
    df.to_parquet(OUT_PARQUET, index=False)

    # gene-level table (carries DNA for Evo2 forward in M1)
    for g in kept_genes:
        cl = gene2cluster[g]
        gene_rows.append((g, prot_by_gene[g], dna_by_gene[g], ss3_by_gene[g],
                          cl, labfn(cl)))
    gdf = pd.DataFrame(gene_rows, columns=["gene", "aa", "dna", "ss3",
                                           "cluster", "split"])
    gdf.to_parquet(OUT_GENES, index=False)
    print(f"[s0] wrote {OUT_GENES} ({gdf.shape[0]} genes)", flush=True)

    # splits.json summary
    gsplit = df.drop_duplicates("gene")[["gene", "cluster", "split"]]
    counts = gsplit["split"].value_counts().to_dict()
    helix_frac = float(df["helix"].mean())
    splits = dict(
        n_genes=int(gsplit.shape[0]),
        n_clusters=int(gsplit["cluster"].nunique()),
        n_codons=int(df.shape[0]),
        split_gene_counts={k:int(v) for k,v in counts.items()},
        overall_helix_frac=helix_frac,
        genes_by_split={s: gsplit[gsplit.split==s]["gene"].tolist()
                        for s in ["train","val","test"]},
        provenance="adapted",
        source="RefSeq GCF_000005845.2 CDS + AlphaFold-DB UP000000625 + DSSP",
        helix_def="DSSP 3-state H={H,G,I}",
    )
    with open(OUT_SPLITS, "w") as fh:
        json.dump(splits, fh, indent=2)
    print(f"[s0] wrote {OUT_PARQUET} ({df.shape[0]} rows) and {OUT_SPLITS}", flush=True)
    print(f"[s0] genes/split={counts} clusters={splits['n_clusters']} "
          f"helix_frac={helix_frac:.3f}", flush=True)

if __name__ == "__main__":
    main()
