"""Iterate the compressed Swiss-Prot flat file and emit, per protein, a per-residue
sparse label matrix over a fixed concept vocabulary.

Concepts drawn from FT (feature table) lines:
  ACT_SITE   active sites
  BINDING    binding sites
  DOMAIN     structural / functional domains (subtyped by /note=)
  MOTIF      motifs
  SITE       misc sites
  MOD_RES    modified residues (PTMs)
  CARBOHYD   glycosylation
  DISULFID   disulfide bonds
  LIPID      lipidation
  CROSSLNK   crosslinks
  TRANSMEM   transmembrane
  ZN_FING    zinc finger
  REPEAT     repeats
  CA_BIND    calcium binding
  DNA_BIND   DNA binding
  NP_BIND    nucleotide binding
  METAL      metal binding (older records)

Two label kinds:
  * generic key (e.g. "DOMAIN")             — any residue inside any FT of that key
  * key + normalized /note text  (e.g. "DOMAIN::EF-hand 1")
    Only annotations whose /note text appears at least MIN_COUNT times across the
    corpus are kept — matches how the paper treats concepts as a fixed vocabulary.

Output (npz per shard):
  entries: array of accessions
  seqs:    array of sequences
  ann:     list-of-lists  ann[i] = [(start, end, concept_idx), ...] (0-indexed inclusive)
  concepts: list of concept strings, aligned to concept_idx
"""
from __future__ import annotations
import gzip
import re
import argparse
import os
import pickle
from collections import Counter, defaultdict


# concepts collected from the FT column of Swiss-Prot
KEY_CONCEPTS = {
    "ACT_SITE", "BINDING", "SITE", "MOD_RES", "CARBOHYD",
    "DISULFID", "LIPID", "CROSSLNK", "DOMAIN", "MOTIF",
    "TRANSMEM", "ZN_FING", "REPEAT", "CA_BIND", "DNA_BIND",
    "NP_BIND", "METAL", "REGION", "COILED", "PROPEP",
    "SIGNAL", "PEPTIDE", "TOPO_DOM", "INTRAMEM", "COMPBIAS",
}

NOTE_RE = re.compile(r'/note="([^"]+)"')
LIGAND_RE = re.compile(r'/ligand="([^"]+)"')


_ECO_RE = re.compile(r"\s*\{ECO[^}]*\}")
_EVID_RE = re.compile(r"\s+ECO:\d+\|[^ ,]+")


def _norm(txt: str) -> str:
    """Strip Swiss-Prot evidence codes and normalize whitespace."""
    t = _ECO_RE.sub("", txt)
    t = _EVID_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip().rstrip(".;,")
    return t[:80]


def iter_entries(path: str):
    """Yield dicts: {ac, seq, feats: [(key, start, end, note_or_none)]}."""
    fh = gzip.open(path, "rt", encoding="latin-1")
    ac, seq_lines, feats, in_seq = None, [], [], False
    cur_key, cur_start, cur_end, cur_note_parts = None, None, None, []

    def flush_feat():
        nonlocal cur_key, cur_start, cur_end, cur_note_parts
        if cur_key is not None:
            note = _norm(" ".join(cur_note_parts)) if cur_note_parts else None
            feats.append((cur_key, cur_start, cur_end, note))
        cur_key, cur_start, cur_end, cur_note_parts = None, None, None, []

    def _line_iter():
        try:
            for ln in fh:
                yield ln
        except (EOFError, OSError) as e:
            # tolerate a truncated .gz (partial download) so we can still use whatever entries parsed cleanly
            return

    for line in _line_iter():
        head = line[:2]
        if head == "ID":
            ac, seq_lines, feats, in_seq = None, [], [], False
            cur_key, cur_start, cur_end, cur_note_parts = None, None, None, []
        elif head == "AC" and ac is None:
            # first AC is primary
            ac = line[5:].strip().split(";")[0].strip()
        elif head == "FT":
            body = line[5:].rstrip("\n")
            if not body.startswith(" "):
                # new FT record: "KEY   start..end" (or "KEY   start")
                flush_feat()
                parts = body.split(None, 1)
                if len(parts) < 2:
                    continue
                key = parts[0]
                if key not in KEY_CONCEPTS:
                    continue
                pos = parts[1].strip()
                # position formats: 12, 12..34, ?..34, <1..34, 12..>34, ?12..34
                m = re.match(r"[<>?]?(\d+)(?:\.\.[<>?]?(\d+))?", pos)
                if not m:
                    continue
                s = int(m.group(1))
                e = int(m.group(2)) if m.group(2) else s
                cur_key, cur_start, cur_end = key, s, e
            else:
                # continuation qualifier line, e.g. '                   /note="..."'
                stripped = body.strip()
                # accumulate note/ligand text
                m = NOTE_RE.search(stripped)
                if m:
                    cur_note_parts.append(m.group(1))
                    continue
                m2 = LIGAND_RE.search(stripped)
                if m2:
                    cur_note_parts.append("ligand:" + m2.group(1))
                    continue
                # continuation of a previously opened /note="..."
                if cur_note_parts and not stripped.startswith("/"):
                    # strip trailing quote if present
                    txt = stripped.rstrip('"')
                    cur_note_parts.append(txt)
        elif head == "SQ":
            flush_feat()
            in_seq = True
        elif head == "//":
            flush_feat()
            seq = "".join(seq_lines).replace(" ", "").upper()
            if ac and seq:
                yield {"ac": ac, "seq": seq, "feats": feats}
            ac, seq_lines, feats, in_seq = None, [], [], False
        elif in_seq and head == "  ":
            seq_lines.append(line.strip())
    fh.close()


def build_vocab(entries_iter, min_count: int, max_entries: int | None = None):
    """First pass: count concept occurrences, choose vocab."""
    cnt: Counter[str] = Counter()
    n = 0
    for ent in entries_iter:
        seen = set()
        for key, s, e, note in ent["feats"]:
            seen.add(key)
            if note:
                seen.add(f"{key}::{note}")
        for c in seen:
            cnt[c] += 1
        n += 1
        if max_entries and n >= max_entries:
            break
    vocab = [c for c, k in cnt.items() if k >= min_count]
    vocab.sort()
    return vocab, cnt, n


def encode_entry(entry, concept_to_idx):
    ann = []
    for key, s, e, note in entry["feats"]:
        # generic key
        idx = concept_to_idx.get(key)
        if idx is not None:
            ann.append((s - 1, e - 1, idx))
        if note:
            idx2 = concept_to_idx.get(f"{key}::{note}")
            if idx2 is not None:
                ann.append((s - 1, e - 1, idx2))
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat_gz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_entries", type=int, default=20000)
    ap.add_argument("--min_len", type=int, default=30)
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--min_concept_count", type=int, default=20,
                    help="drop concepts occurring in fewer than N proteins")
    args = ap.parse_args()

    # Pass 1: pick vocab from up to max_entries qualifying proteins
    print("=== pass 1: building concept vocabulary ===")
    def _iter1():
        n = 0
        for ent in iter_entries(args.dat_gz):
            L = len(ent["seq"])
            if L < args.min_len or L > args.max_len:
                continue
            if not ent["feats"]:
                continue
            yield ent
            n += 1
            if n >= args.max_entries:
                break

    vocab, cnt, n_scanned = build_vocab(_iter1(), args.min_concept_count, args.max_entries)
    print(f"scanned {n_scanned} entries; kept {len(vocab)} concepts (min_count={args.min_concept_count})")
    concept_to_idx = {c: i for i, c in enumerate(vocab)}

    # Pass 2: encode
    print("=== pass 2: encoding annotations ===")
    entries = []
    seqs = []
    anns = []
    n = 0
    for ent in iter_entries(args.dat_gz):
        L = len(ent["seq"])
        if L < args.min_len or L > args.max_len:
            continue
        ann = encode_entry(ent, concept_to_idx)
        if not ann:
            continue
        entries.append(ent["ac"])
        seqs.append(ent["seq"])
        anns.append(ann)
        n += 1
        if n >= args.max_entries:
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({
            "entries": entries,
            "seqs": seqs,
            "anns": anns,
            "concepts": vocab,
            "concept_counts": {c: cnt[c] for c in vocab},
        }, f)
    print(f"wrote {len(entries)} entries and {len(vocab)} concepts to {args.out}")
    # small distribution report
    key_only = [c for c in vocab if "::" not in c]
    subtyped = [c for c in vocab if "::" in c]
    print(f"  {len(key_only)} key-only concepts, {len(subtyped)} subtyped")
    for c in key_only[:30]:
        print(f"    {c}: {cnt[c]}")


if __name__ == "__main__":
    main()
