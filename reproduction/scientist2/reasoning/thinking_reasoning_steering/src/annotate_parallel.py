"""Parallel annotation of the auxiliary corpus + 10% kappa re-check."""
from __future__ import annotations
import os, sys, json, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))
from llm_judge import annotate_chain


def annotate_one(rec: dict, taxonomy: str) -> dict:
    b = annotate_chain(rec["text"], taxonomy_path=taxonomy)
    return {
        "id": rec["id"],
        "source": rec["source"],
        "text": rec["text"],
        "behaviours": {k: v["present"] for k, v in b.items()},
        "behaviour_confidence": {k: v["confidence"] for k, v in b.items()},
    }


def annotate_all(raw_paths, out_path, taxonomy, max_parallel=12):
    have = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                have[r["id"]] = r
    todo = []
    for raw in raw_paths:
        if not os.path.exists(raw): continue
        with open(raw) as f:
            for line in f:
                r = json.loads(line)
                if r["id"] not in have:
                    todo.append(r)
    print(f"[annotate] {len(todo)} to annotate; already {len(have)} cached", flush=True)
    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = {ex.submit(annotate_one, r, taxonomy): r for r in todo}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = futs[fut]
            try:
                out = fut.result()
            except Exception as e:
                print(f"  fail {r['id']}: {e!r}", flush=True); continue
            f.write(json.dumps(out, ensure_ascii=False) + "\n"); f.flush()
            have[r["id"]] = out
            if done % 20 == 0:
                print(f"  annotated {done}/{len(todo)}", flush=True)
    print(f"[annotate] done: {len(have)} total", flush=True)


def kappa_check(corpus_path, taxonomy, out_path, frac=0.10, max_parallel=8):
    rows = [json.loads(l) for l in open(corpus_path)]
    rng = random.Random(0)
    sample = rng.sample(rows, max(5, int(len(rows) * frac)))
    from collections import defaultdict
    tab = defaultdict(lambda: defaultdict(int))
    def re_annotate(r):
        return r, annotate_chain(r["text"], taxonomy_path=taxonomy)
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = [ex.submit(re_annotate, r) for r in sample]
        for fut in as_completed(futs):
            try:
                r, b2 = fut.result()
            except Exception as e:
                print(f"  fail: {e!r}"); continue
            for bname, v in r["behaviours"].items():
                v2 = b2[bname]["present"]
                tab[bname][(v, v2)] += 1
    kappa = {}
    for bname, t in tab.items():
        n = sum(t.values())
        if n == 0: kappa[bname] = None; continue
        po = (t.get((0,0),0) + t.get((1,1),0)) / n
        p1 = (t.get((1,0),0) + t.get((1,1),0)) / n
        q1 = (t.get((0,1),0) + t.get((1,1),0)) / n
        p0, q0 = 1-p1, 1-q1
        pe = p1*q1 + p0*q0
        kappa[bname] = (po - pe) / (1 - pe + 1e-9) if (1-pe) > 1e-6 else 1.0
    json.dump({"n_sample": len(sample), "kappa_per_behaviour": kappa}, open(out_path, "w"), indent=2)
    print(f"[kappa] {kappa}", flush=True)
    return kappa


if __name__ == "__main__":
    t0 = time.time()
    annotate_all(
        ["data/contrast/r1_chains_raw.jsonl", "data/contrast/gpt_answers_raw.jsonl"],
        "data/contrast/auxiliary_corpus.jsonl",
        "configs/behaviour_taxonomy.yaml",
        max_parallel=12,
    )
    kappa_check(
        "data/contrast/auxiliary_corpus.jsonl",
        "configs/behaviour_taxonomy.yaml",
        "data/contrast/annotation_kappa.json",
        frac=0.10, max_parallel=8,
    )
    print(f"total elapsed {time.time()-t0:.0f}s", flush=True)
