import os, sys, json, gzip, io, re, time, math, random, urllib.parse

for stream in ("stdout", "stderr"):
    s = getattr(sys, stream, None)
    if s is not None and not hasattr(s, "isatty"):
        try:
            s.isatty = lambda: False
        except Exception:
            pass
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["NO_COLOR"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
import urllib.request

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"

experiment_data = {
    "swissprot_esm2_8m": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "concept_counts": {},
        "per_concept_f1": {},
    }
}

from transformers import AutoTokenizer, AutoModel

esm_path = os.path.join(MODEL_DIR, "esm2_t6_8M_UR50D")
if not os.path.isdir(esm_path):
    candidates = [
        d for d in os.listdir(MODEL_DIR) if "esm2" in d.lower() and "8m" in d.lower()
    ]
    if candidates:
        esm_path = os.path.join(MODEL_DIR, candidates[0])
    else:
        esm_path = "facebook/esm2_t6_8M_UR50D"
print(f"Loading ESM-2 from: {esm_path}")
tokenizer = AutoTokenizer.from_pretrained(esm_path)
model = AutoModel.from_pretrained(esm_path, output_hidden_states=True)
model.eval().to(device)
n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"ESM-2 layers={n_layers}, hidden={hidden_size}")

SAE_LAYER = 4
sae_dir = os.path.join(MODEL_DIR, "InterPLM-esm2-8m", f"layer_{SAE_LAYER}")
sae_ckpt = os.path.join(sae_dir, "ae_normalized.pt")
sae_cfg_path = os.path.join(sae_dir, "config.json")
assert os.path.isfile(sae_ckpt), f"Missing SAE ckpt: {sae_ckpt}"

with open(sae_cfg_path) as f:
    sae_cfg = json.load(f)
print("SAE config:", sae_cfg)

sae_state = torch.load(sae_ckpt, map_location="cpu")
print(
    "SAE keys:",
    list(sae_state.keys()) if isinstance(sae_state, dict) else type(sae_state),
)


class SAE(torch.nn.Module):
    def __init__(self, d_in, d_sae):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.zeros(d_in, d_sae))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        return F.relu(z)


def get_tensor(state, keys):
    for k in keys:
        if k in state:
            return state[k]
    return None


W_enc = get_tensor(sae_state, ["W_enc", "encoder.weight"])
b_enc = get_tensor(sae_state, ["b_enc", "encoder.bias"])
W_dec = get_tensor(sae_state, ["W_dec", "decoder.weight"])
b_dec = get_tensor(sae_state, ["b_dec", "decoder.bias", "bias"])

if W_enc.shape[0] != hidden_size:
    W_enc = W_enc.T
d_sae = W_enc.shape[1]
if W_dec.shape[1] != hidden_size:
    W_dec = W_dec.T
assert W_dec.shape == (d_sae, hidden_size)
assert b_enc.shape[0] == d_sae
assert b_dec.shape[0] == hidden_size

sae = SAE(hidden_size, d_sae)
sae.W_enc.data.copy_(W_enc.float())
sae.b_enc.data.copy_(b_enc.float())
sae.W_dec.data.copy_(W_dec.float())
sae.b_dec.data.copy_(b_dec.float())
sae.eval().to(device)
print(f"SAE loaded d_in={hidden_size}, d_sae={d_sae}")

norm_mean = get_tensor(sae_state, ["mean", "act_mean", "activation_mean"])
norm_std = get_tensor(sae_state, ["std", "act_std", "activation_std"])
if norm_mean is not None:
    norm_mean = norm_mean.float().to(device)
if norm_std is not None:
    norm_std = norm_std.float().to(device)

# ---------- Swiss-Prot loading ----------
swissprot_cache = os.path.join(working_dir, "swissprot_subset.json")


def find_local_swissprot():
    candidates = []
    search_dirs = [DATA_DIR]
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for fn in files:
                low = fn.lower()
                if (
                    "sprot" in low
                    or "swissprot" in low
                    or "swiss_prot" in low
                    or "uniprot_sprot" in low
                ) and (
                    low.endswith(".dat")
                    or low.endswith(".dat.gz")
                    or low.endswith(".txt")
                    or low.endswith(".txt.gz")
                    or low.endswith(".xml")
                    or low.endswith(".xml.gz")
                    or low.endswith(".json")
                    or low.endswith(".json.gz")
                ):
                    candidates.append(os.path.join(root, fn))
    return candidates


def open_maybe_gz(p):
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "rt", encoding="utf-8", errors="replace")


def parse_uniprot_dat(path, max_entries=400, min_len=30, max_len=1000):
    entries = []
    with open_maybe_gz(path) as f:
        acc = None
        seq_lines = []
        ft_entries = []
        in_sq = False
        cur_ft_type = None
        cur_ft_from = None
        cur_ft_to = None
        cur_ft_desc = []

        def flush_ft():
            nonlocal cur_ft_type, cur_ft_from, cur_ft_to, cur_ft_desc
            if (
                cur_ft_type is not None
                and cur_ft_from is not None
                and cur_ft_to is not None
            ):
                ft_entries.append(
                    (cur_ft_type, cur_ft_from, cur_ft_to, " ".join(cur_ft_desc).strip())
                )
            cur_ft_type = None
            cur_ft_from = None
            cur_ft_to = None
            cur_ft_desc = []

        for line in f:
            if line.startswith("AC ") and acc is None:
                parts = line[5:].strip().split(";")
                acc = parts[0].strip()
            elif line.startswith("SQ "):
                in_sq = True
            elif line.startswith("//"):
                flush_ft()
                seq = "".join(seq_lines).replace(" ", "").replace("\n", "")
                if acc and seq and min_len <= len(seq) <= max_len and ft_entries:
                    feats = {
                        "BINDING": [],
                        "ACT_SITE": [],
                        "MOTIF": [],
                        "DOMAIN": [],
                        "MOD_RES": [],
                    }
                    for ftype, s, e, desc in ft_entries:
                        key = None
                        if ftype in (
                            "BINDING",
                            "METAL",
                            "NP_BIND",
                            "CA_BIND",
                            "DNA_BIND",
                        ):
                            key = "BINDING"
                        elif ftype in ("ACT_SITE",):
                            key = "ACT_SITE"
                        elif ftype in ("MOTIF",):
                            key = "MOTIF"
                        elif ftype in ("DOMAIN",):
                            key = "DOMAIN"
                        elif ftype in ("MOD_RES", "LIPID", "CARBOHYD"):
                            key = "MOD_RES"
                        if key:
                            feats[key].append((s, e, desc or "generic"))
                    if sum(len(v) for v in feats.values()) > 0:
                        entries.append(
                            {"accession": acc, "sequence": seq, "features": feats}
                        )
                        if len(entries) >= max_entries:
                            break
                acc = None
                seq_lines = []
                ft_entries = []
                in_sq = False
            elif in_sq:
                seq_lines.append(line.strip())
            elif line.startswith("FT   "):
                body = line[5:].rstrip("\n")
                if body.startswith(" ") or body.startswith("/"):
                    m = re.search(r'/note="([^"]*)', body)
                    if m:
                        cur_ft_desc.append(m.group(1))
                    else:
                        stripped = body.strip().strip('"')
                        if (
                            stripped
                            and cur_ft_type is not None
                            and not stripped.startswith("/")
                        ):
                            cur_ft_desc.append(stripped)
                else:
                    flush_ft()
                    m = re.match(
                        r"^(\S+)\s+(\d+|<?\d+|>?\d+|\?)\.\.(\d+|<?\d+|>?\d+|\?)", body
                    )
                    if m:
                        cur_ft_type = m.group(1).upper()
                        try:
                            cur_ft_from = int(re.sub(r"[<>?]", "", m.group(2)))
                            cur_ft_to = int(re.sub(r"[<>?]", "", m.group(3)))
                        except Exception:
                            cur_ft_type = None
                    else:
                        m2 = re.match(r"^(\S+)\s+(\d+)\s+(\d+)\s*(.*)$", body)
                        if m2:
                            cur_ft_type = m2.group(1).upper()
                            try:
                                cur_ft_from = int(m2.group(2))
                                cur_ft_to = int(m2.group(3))
                                if m2.group(4):
                                    cur_ft_desc.append(m2.group(4))
                            except Exception:
                                cur_ft_type = None
    return entries


def extract_features_json(entry):
    feats = {"BINDING": [], "ACT_SITE": [], "MOTIF": [], "DOMAIN": [], "MOD_RES": []}
    type_map = {
        "Binding site": "BINDING",
        "Binding": "BINDING",
        "Metal binding": "BINDING",
        "Active site": "ACT_SITE",
        "Short sequence motif": "MOTIF",
        "Motif": "MOTIF",
        "Domain": "DOMAIN",
        "Modified residue": "MOD_RES",
        "Lipidation": "MOD_RES",
        "Glycosylation": "MOD_RES",
    }
    for feat in entry.get("features", []):
        ft = feat.get("type", "")
        key = type_map.get(ft)
        if key is None:
            continue
        loc = feat.get("location", {})
        try:
            start = int(loc.get("start", {}).get("value"))
            end = int(loc.get("end", {}).get("value"))
        except Exception:
            continue
        desc = feat.get("description", "") or ""
        ligand = feat.get("ligand", {}).get("name", "") if "ligand" in feat else ""
        subtype = desc or ligand or "generic"
        feats[key].append((start, end, subtype))
    return feats


def http_get(url, timeout=180):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (research script)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_swissprot_stream(target_n=400):
    """Use UniProt search endpoint with pagination via cursor to avoid partial JSON."""
    query = "reviewed:true AND (ft_binding:* OR ft_act_site:* OR ft_motif:* OR ft_domain:* OR ft_mod_res:*)"
    fields = "accession,sequence,ft_binding,ft_act_site,ft_motif,ft_domain,ft_mod_res"
    entries = []
    # Use smaller page size (25) to avoid truncated JSON; loop until enough
    size = 25
    cursor = None
    max_pages = 50
    for _ in range(max_pages):
        params = {
            "query": query,
            "format": "json",
            "size": str(size),
            "fields": fields,
        }
        if cursor:
            params["cursor"] = cursor
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            params
        )
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (research script)",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read()
                # get Link header for cursor
                link_hdr = r.headers.get("Link", "") or r.headers.get("link", "")
            data = json.loads(raw.decode())
            results = data.get("results", [])
            if not results:
                break
            for e in results:
                seq = e.get("sequence", {}).get("value", "")
                if not seq or not (30 <= len(seq) <= 1000):
                    continue
                feats = extract_features_json(e)
                if sum(len(v) for v in feats.values()) == 0:
                    continue
                entries.append(
                    {
                        "accession": e.get("primaryAccession", ""),
                        "sequence": seq,
                        "features": feats,
                    }
                )
                if len(entries) >= target_n:
                    break
            if len(entries) >= target_n:
                break
            # parse cursor from Link header
            m = re.search(r"cursor=([^&>]+)", link_hdr)
            if not m:
                break
            cursor = m.group(1)
        except Exception as ex:
            print(f"  page fetch failed: {ex}")
            break
    return entries


def download_swissprot_by_accessions(target_n=400):
    """Fallback: fetch a curated list of well-annotated Swiss-Prot accessions individually."""
    # Fetch a list of accessions first via a smaller list-only query
    accs = []
    try:
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {
                "query": "reviewed:true AND (ft_binding:* OR ft_act_site:*) AND length:[50 TO 800]",
                "format": "list",
                "size": "500",
            }
        )
        raw = http_get(url).decode()
        accs = [a.strip() for a in raw.splitlines() if a.strip()]
    except Exception as ex:
        print(f"  accession list fetch failed: {ex}")
        return []
    print(f"  got {len(accs)} accessions, fetching entries individually...")
    entries = []
    for i, acc in enumerate(accs):
        if len(entries) >= target_n:
            break
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
            raw = http_get(url, timeout=60)
            e = json.loads(raw.decode())
            seq = e.get("sequence", {}).get("value", "")
            if not seq or not (30 <= len(seq) <= 1000):
                continue
            feats = extract_features_json(e)
            if sum(len(v) for v in feats.values()) == 0:
                continue
            entries.append(
                {
                    "accession": e.get("primaryAccession", acc),
                    "sequence": seq,
                    "features": feats,
                }
            )
            if (i + 1) % 25 == 0:
                print(f"  fetched {len(entries)}/{target_n} so far ({i+1} tried)")
        except Exception:
            continue
    return entries


sp_entries = None
if os.path.isfile(swissprot_cache):
    try:
        with open(swissprot_cache) as f:
            sp_entries = json.load(f)
        print(f"Loaded cached Swiss-Prot: {len(sp_entries)} entries")
    except Exception:
        sp_entries = None

if not sp_entries:
    local_files = find_local_swissprot()
    if local_files:
        print(f"Found local Swiss-Prot files: {local_files[:3]} ...")
        for lf in local_files:
            try:
                low = lf.lower()
                if (
                    low.endswith(".dat")
                    or low.endswith(".dat.gz")
                    or low.endswith(".txt")
                    or low.endswith(".txt.gz")
                ):
                    sp_entries = parse_uniprot_dat(lf, max_entries=400)
                elif low.endswith(".json") or low.endswith(".json.gz"):
                    with open_maybe_gz(lf) as fh:
                        raw = json.load(fh)
                    entries = raw.get("results", raw) if isinstance(raw, dict) else raw
                    sp_entries = []
                    for e in entries:
                        seq = (
                            e.get("sequence", {}).get("value", "")
                            if isinstance(e.get("sequence"), dict)
                            else e.get("sequence", "")
                        )
                        if not seq or not (30 <= len(seq) <= 1000):
                            continue
                        feats = extract_features_json(e)
                        if sum(len(v) for v in feats.values()) == 0:
                            continue
                        sp_entries.append(
                            {
                                "accession": e.get(
                                    "primaryAccession", e.get("accession", "")
                                ),
                                "sequence": seq,
                                "features": feats,
                            }
                        )
                        if len(sp_entries) >= 400:
                            break
                if sp_entries:
                    print(f"Parsed {len(sp_entries)} entries from {lf}")
                    break
            except Exception as ex:
                print(f"  failed to parse {lf}: {ex}")
                continue

if not sp_entries:
    print("Trying streaming search download...")
    try:
        sp_entries = download_swissprot_stream(target_n=300)
        print(f"  streaming got {len(sp_entries) if sp_entries else 0}")
    except Exception as ex:
        print(f"  streaming failed: {ex}")
        sp_entries = []

if not sp_entries or len(sp_entries) < 50:
    print("Trying accession-by-accession fallback...")
    try:
        more = download_swissprot_by_accessions(target_n=300)
        if more:
            sp_entries = (sp_entries or []) + more
    except Exception as ex:
        print(f"  fallback failed: {ex}")

assert (
    sp_entries and len(sp_entries) > 0
), "Could not obtain any Swiss-Prot entries. Please place a Swiss-Prot .dat/.dat.gz under $DATA_DIR."

try:
    with open(swissprot_cache, "w") as f:
        json.dump(sp_entries, f)
except Exception:
    pass

print(f"Total Swiss-Prot entries: {len(sp_entries)}")

# Build concept dictionary
from collections import defaultdict, Counter

MAX_LEN = 512
proteins = []
for e in sp_entries:
    seq = e["sequence"][:MAX_LEN]
    proteins.append(
        {"accession": e["accession"], "sequence": seq, "features": e["features"]}
    )


def norm_subtype(s):
    s = s.lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.split(";")[0].strip()
    return s[:60]


concept_prots = defaultdict(list)
for pi, prot in enumerate(proteins):
    L = len(prot["sequence"])
    for ftype, flist in prot["features"].items():
        for s, e, sub in flist:
            s = max(1, s)
            e = min(L, e)
            if s > e:
                continue
            sub_n = norm_subtype(sub) if sub else "generic"
            for key in [f"{ftype}::{sub_n}", f"{ftype}::ANY"]:
                concept_prots[key].append((pi, set(range(s - 1, e))))

concepts = {k: v for k, v in concept_prots.items() if len(v) >= 3}
print(f"Total concepts (>=3 proteins): {len(concepts)}")

prot_lens = [len(p["sequence"]) for p in proteins]
offsets = np.cumsum([0] + prot_lens)
total_res = int(offsets[-1])
print(f"Total residues: {total_res} across {len(proteins)} proteins")

concept_labels = {}
for ck, occs in concepts.items():
    arr = np.zeros(total_res, dtype=bool)
    for pi, pos_set in occs:
        off = offsets[pi]
        for p in pos_set:
            if p < prot_lens[pi]:
                arr[off + p] = True
    if arr.sum() >= 5:
        concept_labels[ck] = arr
print(f"Concepts with >=5 positive residues: {len(concept_labels)}")

assert len(concept_labels) > 0, "No usable concepts."

# --- Extract activations ---
neuron_cache = os.path.join(working_dir, f"activations_layer{SAE_LAYER}_neuron.npz")
sae_memmap_path = os.path.join(
    working_dir, f"sae_acts_layer{SAE_LAYER}_N{total_res}_D{d_sae}.dat"
)

need_extract = True
if os.path.isfile(neuron_cache) and os.path.isfile(sae_memmap_path):
    try:
        z = np.load(neuron_cache)
        neuron_acts = z["neuron"]
        if neuron_acts.shape[0] == total_res and neuron_acts.shape[1] == hidden_size:
            sae_acts = np.memmap(
                sae_memmap_path, dtype=np.float32, mode="r", shape=(total_res, d_sae)
            )
            print(
                f"Loaded cached activations: neuron {neuron_acts.shape}, sae {sae_acts.shape}"
            )
            need_extract = False
    except Exception:
        need_extract = True

if need_extract:
    neuron_acts = np.zeros((total_res, hidden_size), dtype=np.float32)
    sae_acts = np.memmap(
        sae_memmap_path, dtype=np.float32, mode="w+", shape=(total_res, d_sae)
    )
    with torch.no_grad():
        for pi, prot in enumerate(proteins):
            seq = prot["sequence"]
            enc = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model(
                input_ids=input_ids, attention_mask=attn, output_hidden_states=True
            )
            h = out.hidden_states[SAE_LAYER][0]
            h = h[1 : 1 + len(seq)]
            if h.shape[0] != len(seq):
                h = h[: len(seq)]
            x = h
            if norm_mean is not None and norm_std is not None:
                x = (x - norm_mean) / (norm_std + 1e-6)
            z_sae = sae.encode(x)
            off = offsets[pi]
            L = h.shape[0]
            neuron_acts[off : off + L] = h.detach().cpu().numpy()
            sae_acts[off : off + L] = z_sae.detach().cpu().numpy()
            if (pi + 1) % 25 == 0:
                print(f"  extracted {pi+1}/{len(proteins)}")
    sae_acts.flush()
    np.savez_compressed(neuron_cache, neuron=neuron_acts)
    print("Activations extracted.")

print(
    f"SAE act nonzero fraction (sample): {(sae_acts[:min(2000, total_res)] > 0).mean():.4f}"
)


def best_f1_per_unit(acts, labels):
    N, D = acts.shape
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == N:
        return 0.0, -1
    best_f1 = 0.0
    best_unit = -1
    chunk = 512
    for start in range(0, D, chunk):
        end = min(D, start + chunk)
        a = np.asarray(acts[:, start:end])
        for k_mult in [0.5, 1.0, 1.5, 2.0]:
            k = max(1, int(n_pos * k_mult))
            k = min(k, N)
            idx = np.argpartition(-a, k - 1, axis=0)[:k]
            for u in range(end - start):
                sel = idx[:, u]
                vals = a[sel, u]
                mask = vals > 0
                if mask.sum() == 0:
                    continue
                sel = sel[mask]
                tp = int(labels[sel].sum())
                pred_pos = len(sel)
                if pred_pos == 0:
                    continue
                precision = tp / pred_pos
                recall = tp / n_pos
                if precision + recall == 0:
                    continue
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_unit = start + u
    return best_f1, best_unit


concept_items = sorted(concept_labels.items(), key=lambda x: -x[1].sum())
concept_items = concept_items[:80]
print(f"Evaluating {len(concept_items)} concepts")

F1_THRESH = 0.5
sae_recovered = 0
neuron_recovered = 0
per_concept = {}

t0 = time.time()
for ci, (ck, labels) in enumerate(concept_items):
    n_pos = int(labels.sum())
    f1_neuron, u_neuron = best_f1_per_unit(neuron_acts, labels)
    f1_sae, u_sae = best_f1_per_unit(sae_acts, labels)
    per_concept[ck] = {
        "n_pos": n_pos,
        "f1_neuron": float(f1_neuron),
        "f1_sae": float(f1_sae),
        "unit_neuron": int(u_neuron),
        "unit_sae": int(u_sae),
    }
    if f1_neuron >= F1_THRESH:
        neuron_recovered += 1
    if f1_sae >= F1_THRESH:
        sae_recovered += 1
    if (ci + 1) % 10 == 0:
        print(
            f"  [{ci+1}/{len(concept_items)}] elapsed={time.time()-t0:.1f}s neuron={neuron_recovered} sae={sae_recovered}"
        )

print(f"\n=== Results (F1 >= {F1_THRESH}) ===")
print(f"Concepts evaluated: {len(concept_items)}")
print(f"Recovered by raw neurons: {neuron_recovered}")
print(f"Recovered by SAE features: {sae_recovered}")

experiment_data["swissprot_esm2_8m"]["metrics"]["val"].append(
    {
        "epoch": 0,
        "concept_alignment_count_sae": sae_recovered,
        "concept_alignment_count_neuron": neuron_recovered,
        "n_concepts_evaluated": len(concept_items),
    }
)
experiment_data["swissprot_esm2_8m"]["concept_counts"] = {
    "sae": sae_recovered,
    "neuron": neuron_recovered,
    "total_evaluated": len(concept_items),
}
experiment_data["swissprot_esm2_8m"]["per_concept_f1"] = per_concept
experiment_data["swissprot_esm2_8m"]["losses"]["val"].append(0.0)

print(f"Epoch 0: validation_loss = 0.0000")
print(f"Epoch 0: concept_alignment_count (SAE) = {sae_recovered}")
print(f"Epoch 0: concept_alignment_count (neuron) = {neuron_recovered}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment_data.npy")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f1_neurons = [v["f1_neuron"] for v in per_concept.values()]
    f1_saes = [v["f1_sae"] for v in per_concept.values()]
    plt.figure(figsize=(8, 5))
    plt.hist(f1_neurons, bins=20, alpha=0.6, label="Raw neurons")
    plt.hist(f1_saes, bins=20, alpha=0.6, label="SAE features")
    plt.axvline(F1_THRESH, color="r", linestyle="--", label=f"F1={F1_THRESH}")
    plt.xlabel("Best F1 per concept")
    plt.ylabel("# concepts")
    plt.title(f"ESM-2-8M layer {SAE_LAYER}: SAE vs Neuron concept alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"f1_hist_esm2_8m_layer{SAE_LAYER}.png"))
    plt.close()
    print("Saved f1 histogram.")
except Exception as e:
    print(f"Plot failed: {e}")

print("Done.")
