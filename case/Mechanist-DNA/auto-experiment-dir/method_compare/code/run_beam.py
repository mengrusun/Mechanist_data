"""Evo2-style inference-time beam search, with and without SAE feature steering.

Protocol mirrors Evo 2 (Brixi et al., Fig. 6): generate a fixed-length chunk, re-score every
partial sequence with an EXTERNAL scorer ensemble, keep only the top-B beams, continue. Evo 2 used
128 bp chunks, ~30 sampled chunks per step, and kept the top 2. Here the chunk is 60 nt (20 codons,
so chunk boundaries are codon-aligned), B=2, and the search width W is swept to trace the
inference-time compute scaling curve.

Arms:
  base  -- plain Evo2 (the pure Evo2 method)
  steer -- the same search on top of SAE alpha-helix feature steering at c* (frozen from round 2)
Width W=0 means no search at all: one 300-nt sample per prompt, no scorer (the round-2 baseline).

Compute is accounted on two axes: nucleotides generated per delivered sequence (hardware-independent,
the axis Evo 2 itself uses) and wall-clock GPU-seconds split into generation vs scoring.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import mc_env; mc_env.patch()
import numpy as np, torch
from evo2_sae import load_evo2, BatchTopKSAE
from m1_harness_calibrate import natural_prompts
from mechanism import translate_orf
import harness2 as H
import m0_data as D
from scorers import ESM2HelixProbe, Ensemble

GEN_BATCH = int(os.environ.get("MC_GEN_BATCH", 48))


def batched_generate(evo2, seqs, n_tokens, temperature, top_k):
    """Continue each sequence by n_tokens. Returns the extended sequences."""
    out = []
    for i in range(0, len(seqs), GEN_BATCH):
        chunk = seqs[i:i + GEN_BATCH]
        r = evo2.generate(prompt_seqs=chunk, n_tokens=n_tokens, temperature=temperature,
                          top_k=top_k, verbose=0)
        cont = r.sequences if hasattr(r, "sequences") else r
        out += [s + (c[len(s):] if c.startswith(s) else c) for s, c in zip(chunk, cont)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "steer"])
    ap.add_argument("--width", type=int, required=True, help="W: chunks sampled per beam per step; 0 = no search")
    ap.add_argument("--beams", type=int, default=2)
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--prompt_stride", type=int, default=3)
    ap.add_argument("--chunk_nt", type=int, default=60)
    ap.add_argument("--n_steps", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--min_aa", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"[beam] {args.out} exists -- skip", flush=True)
        return
    t0 = time.time()
    torch.manual_seed(args.seed)
    org = mc_env.ORGANISM
    table = D.ORGANISMS[org]["transl_table"]

    all_prompts, _, _ = natural_prompts(org, 300)
    prompts = all_prompts[::args.prompt_stride][:args.n_prompts]
    P = len(prompts)

    evo2 = load_evo2("cuda:0")
    steerer = None
    if args.arm == "steer":
        fs = json.load(open(mc_env.FEATURE_SET))
        calib = json.load(open(mc_env.CALIB))
        sae = BatchTopKSAE(path=mc_env.SAE_PATH, device="cuda:0")
        steerer = H.CSteerer(evo2, sae, fs["helix_features"], fs["s_f"],
                             sigma_proj=calib["sigma_proj"], device="cuda:0")
    c = mc_env.C_STAR if args.arm == "steer" else 0.0

    probe = ESM2HelixProbe(os.path.join(mc_env.MC, "results", "ss_probe.pt"), device="cuda:0")
    ens = Ensemble(probe=probe)
    print(f"[beam] arm={args.arm} W={args.width} B={args.beams} P={P} c={c} "
          f"chunk={args.chunk_nt}nt steps={args.n_steps}", flush=True)

    import contextlib
    steer_ctx = (lambda: steerer.steer(c)) if steerer is not None else (lambda: contextlib.nullcontext())

    gen_s = 0.0
    nt_generated = 0

    if args.width == 0:
        # no search: one full-length sample per prompt, scorer never called
        tg = time.time()
        with steer_ctx():
            finals = batched_generate(evo2, list(prompts), args.chunk_nt * args.n_steps,
                                      args.temperature, args.top_k)
        gen_s += time.time() - tg
        nt_generated += P * args.chunk_nt * args.n_steps
    else:
        beams = [[p] for p in prompts]              # per prompt, list of live beams
        for step in range(args.n_steps):
            tasks, owner = [], []
            for p in range(P):
                for b in beams[p]:
                    for _ in range(args.width):
                        tasks.append(b); owner.append(p)
            tg = time.time()
            with steer_ctx():
                grown = batched_generate(evo2, tasks, args.chunk_nt, args.temperature, args.top_k)
            gen_s += time.time() - tg
            nt_generated += len(tasks) * args.chunk_nt

            prots, stops = [], []
            for s in grown:
                pr, valid, meta = translate_orf(s, min_aa=5, table=table)
                prots.append(pr); stops.append(bool(meta.get("had_internal_stop", False)))
            sc = ens(prots, stops)

            newbeams = [[] for _ in range(P)]
            order = {}
            for i, p in enumerate(owner):
                order.setdefault(p, []).append(i)
            for p in range(P):
                idxs = sorted(order[p], key=lambda i: -sc[i])[:args.beams]
                newbeams[p] = [grown[i] for i in idxs]
            beams = newbeams
            print(f"[beam] step {step+1}/{args.n_steps} cands={len(tasks)} "
                  f"gen={gen_s:.0f}s score={ens.seconds:.0f}s ({time.time()-t0:.0f}s)", flush=True)
        finals = [b[0] for b in beams]

    records = []
    for i, (pmt, dna) in enumerate(zip(prompts, finals)):
        pr, valid, meta = translate_orf(dna, min_aa=args.min_aa, table=table)
        records.append({"idx": i, "prompt_cluster": i, "prompt": pmt, "dna": dna,
                        "prot": pr if valid else None, "valid_orf": bool(valid),
                        "len_aa": meta.get("len_aa", 0),
                        "had_internal_stop": bool(meta.get("had_internal_stop", False))})

    n_valid = sum(r["valid_orf"] for r in records)
    result = {
        "config": {"arm": args.arm, "width": args.width, "beams": args.beams, "c_sigma": c,
                   "n_prompts": P, "prompt_stride": args.prompt_stride, "chunk_nt": args.chunk_nt,
                   "n_steps": args.n_steps, "temperature": args.temperature, "top_k": args.top_k,
                   "seed": args.seed, "scorer": "chou_fasman+esm2_ss3_probe"},
        "cost": {"nt_generated_total": nt_generated,
                 "nt_generated_per_delivered": nt_generated / max(P, 1),
                 "scorer_calls_total": ens.n_calls,
                 "scorer_calls_per_delivered": ens.n_calls / max(P, 1),
                 "gen_seconds": gen_s, "scorer_seconds": ens.seconds,
                 "gpu_seconds_per_delivered": (gen_s + ens.seconds) / max(P, 1),
                 "total_seconds": time.time() - t0},
        "yield": {"n_delivered": P, "n_valid_orf": n_valid, "valid_orf_rate": n_valid / max(P, 1)},
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[beam] WROTE {args.out} valid_orf={n_valid}/{P} "
          f"nt/seq={nt_generated/max(P,1):.0f} gpu_s/seq={(gen_s+ens.seconds)/max(P,1):.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); os._exit(0)
