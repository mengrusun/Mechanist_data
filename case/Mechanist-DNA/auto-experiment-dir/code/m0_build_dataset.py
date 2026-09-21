"""
Build the full M0 labeled dataset for one organism (natural CDS + real DSSP labels).

Strategy (HC3-compliant, no reverse-translation):
  - UniProt: reviewed structured proteins -> sequence + PDB xrefs + EMBL CDS xrefs.
  - CDS: bulk `fasta_cds_na` per genome accession (one NCBI call yields many CDS) with a
    per-protein `coded_by` fallback. Verified by exact translation == UniProt sequence.
  - Labels: download experimental PDB, run mkdssp, align DSSP chain to CDS-protein, propagate
    8-state SS to codons.
  - Splits: mmseqs2 easy-cluster at 30% identity on the accepted protein set (leakage control).

Writes: data/m0_dataset_<org>.jsonl  (one JSON obj per protein)
        data/m0_clusters_<org>.json  (acc -> cluster id)
Run: python code/m0_build_dataset.py --organism prokaryote --target 3000 --floor 300
"""
import os, sys, json, time, argparse, re, subprocess, threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
sys.path.insert(0, os.path.dirname(__file__))
import m0_data as D

Entrez.email = "wanghaoxiong@zju.edu.cn"
_ncbi_lock = threading.Semaphore(1)
_last_ncbi = [0.0]
_NCBI_MIN_INTERVAL = 0.34  # ~3/s


def ncbi_throttle():
    with _ncbi_lock:
        dt = time.time() - _last_ncbi[0]
        if dt < _NCBI_MIN_INTERVAL:
            time.sleep(_NCBI_MIN_INTERVAL - dt)
        _last_ncbi[0] = time.time()


def bulk_cds_map(genome_accessions, transl_table):
    """Fetch all CDS from each genome accession via fasta_cds_na; return {protein_id: cds_nt}."""
    cds_map = {}
    for acc in genome_accessions:
        cache = os.path.join(D.CACHE_DIR, f"bulk_{acc}.fasta")
        try:
            if not (os.path.exists(cache) and os.path.getsize(cache) > 0):
                ncbi_throttle()
                h = Entrez.efetch(db="nuccore", id=acc, rettype="fasta_cds_na", retmode="text")
                text = h.read(); h.close()
                with open(cache, "w") as f:
                    f.write(text)
            with open(cache) as f:
                text = f.read()
        except Exception as e:
            print(f"  bulk fetch failed for {acc}: {e}", flush=True)
            continue
        # parse fasta; header has [protein_id=XXX.1]
        pid = None; seqbuf = []
        for line in text.splitlines():
            if line.startswith(">"):
                if pid and seqbuf:
                    cds_map.setdefault(pid, "".join(seqbuf))
                m = re.search(r"\[protein_id=([^\]]+)\]", line)
                pid = m.group(1) if m else None
                seqbuf = []
            else:
                seqbuf.append(line.strip())
        if pid and seqbuf:
            cds_map.setdefault(pid, "".join(seqbuf))
    return cds_map


def get_cds(prot, cfg, cds_map):
    """Return (cds, protein_id) verified against uniprot sequence; try bulk map then coded_by."""
    for nuc_acc, protein_id, moltype in prot["embl"]:
        if protein_id in cds_map:
            cds = cds_map[protein_id]
            if D._translation_matches(cds, prot["seq"], cfg["transl_table"]):
                return cds, protein_id
    # fallback: coded_by route (rate-limited)
    for nuc_acc, protein_id, moltype in prot["embl"]:
        cache = os.path.join(D.CACHE_DIR, f"{protein_id}.txt")
        if os.path.exists(cache):
            cds = open(cache).read().strip()
            if D._translation_matches(cds, prot["seq"], cfg["transl_table"]):
                return cds, protein_id
            continue
        try:
            ncbi_throttle()
            cds = D._fetch_cds_via_codedby(protein_id, cfg["transl_table"], retries=2)
            if cds and D._translation_matches(cds, prot["seq"], cfg["transl_table"]):
                open(cache, "w").write(cds)
                return cds, protein_id
        except Exception:
            continue
    return None, None


def label_one(prot, cds, cfg):
    """Try PDBs until one aligns with coverage>=0.5; return record dict or None."""
    for pdb in prot["pdbs"][:8]:
        loc = D.download_pdb(pdb["id"])
        if not loc:
            continue
        chains = D.run_dssp(loc)
        if not chains:
            continue
        codon_ss, cov, chain = D.align_ss_to_cds(cds, prot["seq"], chains, cfg["transl_table"])
        if codon_ss and cov >= 0.5:
            return {"acc": prot["acc"], "pdb": pdb["id"], "chain": chain, "coverage": cov,
                    "cds": cds, "codon_ss": codon_ss, "n_codons": len(prot["seq"]),
                    "seq": prot["seq"]}
    return None


def process(prot, cfg, cds_map):
    try:
        cds, pid = get_cds(prot, cfg, cds_map)
        if not cds:
            return None
        rec = label_one(prot, cds, cfg)
        return rec
    except Exception as e:
        return None


def mmseqs_cluster(records, org, min_id=0.3):
    """Cluster protein sequences with mmseqs2 easy-cluster; return acc->cluster."""
    tmp = os.path.join(D.DATA_DIR, f"mmseqs_{org}")
    os.makedirs(tmp, exist_ok=True)
    fasta = os.path.join(tmp, "in.fasta")
    with open(fasta, "w") as f:
        for r in records:
            f.write(f">{r['acc']}\n{r['seq']}\n")
    out_prefix = os.path.join(tmp, "clu")
    r = subprocess.run(["mmseqs", "easy-cluster", fasta, out_prefix, os.path.join(tmp, "tmp"),
                        "--min-seq-id", str(min_id), "-c", "0.5", "--cov-mode", "1", "-v", "1"],
                       capture_output=True, text=True)
    tsv = out_prefix + "_cluster.tsv"
    acc2clu = {}
    if os.path.exists(tsv):
        rep2id = {}
        for line in open(tsv):
            rep, member = line.strip().split("\t")[:2]
            if rep not in rep2id:
                rep2id[rep] = len(rep2id)
            acc2clu[member] = rep2id[rep]
    else:
        # fallback: each protein its own cluster
        for i, r in enumerate(records):
            acc2clu[r["acc"]] = i
        print(f"  mmseqs failed ({r.stderr[:150]}); singleton clusters", flush=True)
    return acc2clu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", required=True, choices=list(D.ORGANISMS.keys()))
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--floor", type=int, default=300)
    ap.add_argument("--max_uniprot", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cfg = D.ORGANISMS[args.organism]
    out_jsonl = os.path.join(D.DATA_DIR, f"m0_dataset_{args.organism}.jsonl")

    print(f"[{args.organism}] {cfg['name']} taxid={cfg['taxid']}", flush=True)
    t0 = time.time()
    prots = D.uniprot_structured_proteins_paged(cfg["taxid"], max_proteins=args.max_uniprot, page_size=500)
    print(f"uniprot structured proteins: {len(prots)} ({time.time()-t0:.0f}s)", flush=True)

    # collect genome accessions for bulk CDS
    genome_accs = set()
    for p in prots:
        for nuc_acc, pid, mt in p["embl"]:
            genome_accs.add(nuc_acc)
    # prefer complete-genome accessions (heuristic: appear most frequently)
    from collections import Counter
    acc_count = Counter()
    for p in prots:
        for nuc_acc, pid, mt in p["embl"]:
            acc_count[nuc_acc] += 1
    top_genomes = [a for a, c in acc_count.most_common(8) if c >= 5]
    print(f"bulk-fetching CDS from {len(top_genomes)} top genome accessions: {top_genomes}", flush=True)
    cds_map = bulk_cds_map(top_genomes, cfg["transl_table"]) if top_genomes else {}
    print(f"bulk CDS map size: {len(cds_map)} ({time.time()-t0:.0f}s)", flush=True)

    # already-built (resume)
    done_accs = set()
    if os.path.exists(out_jsonl):
        for line in open(out_jsonl):
            try:
                done_accs.add(json.loads(line)["acc"])
            except Exception:
                pass
    todo = [p for p in prots if p["acc"] not in done_accs]
    print(f"already built: {len(done_accs)}; todo: {len(todo)}", flush=True)

    records = []
    if done_accs:
        for line in open(out_jsonl):
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    n_new = 0
    fout = open(out_jsonl, "a")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, p, cfg, cds_map): p for p in todo}
        for i, fut in enumerate(as_completed(futs)):
            rec = fut.result()
            if rec:
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                records.append(rec); n_new += 1
            if (i + 1) % 100 == 0:
                labeled = sum(len([s for s in r["codon_ss"] if s is not None]) for r in records)
                print(f"  processed {i+1}/{len(todo)}; accepted={len(records)}; "
                      f"labeled_codons={labeled} ({time.time()-t0:.0f}s)", flush=True)
            if len(records) >= args.target:
                print(f"  reached target {args.target}; stopping fetch", flush=True)
                break
    fout.close()

    labeled = sum(len([s for s in r["codon_ss"] if s is not None]) for r in records)
    helix_pos_prot = sum(any(D.helix_label(s, "HGI") == 1 for s in r["codon_ss"]) for r in records)
    print(f"\n[{args.organism}] accepted proteins: {len(records)}; labeled codons: {labeled}; "
          f"helix-positive proteins: {helix_pos_prot}", flush=True)

    # cluster
    if records:
        acc2clu = mmseqs_cluster(records, args.organism)
        json.dump(acc2clu, open(os.path.join(D.DATA_DIR, f"m0_clusters_{args.organism}.json"), "w"))
        print(f"clusters: {len(set(acc2clu.values()))}", flush=True)

    status = "OK" if len(records) >= args.floor and labeled >= 50000 and helix_pos_prot >= 50 else "BELOW_FLOOR"
    summary = {"organism": args.organism, "n_proteins": len(records), "n_labeled_codons": labeled,
               "helix_positive_proteins": helix_pos_prot, "floor": args.floor, "status": status,
               "elapsed_s": time.time() - t0}
    json.dump(summary, open(os.path.join(D.DATA_DIR, f"m0_build_summary_{args.organism}.json"), "w"), indent=2)
    print(f"BUILD_STATUS={status}  {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
