"""Launch one milestone wave: one run_arms.py per GPU (parallel), wait for all.

  python launch_wave.py --specs s0.json s1.json ... --gpus 1 3 6 7 \
      --features results/m1_features.json --readout esm2_probe --tag m2
"""
import os, sys, subprocess, argparse, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--gpus", nargs="+", required=True)
    ap.add_argument("--features", default=os.path.join(ROOT, "results/m1_features.json"))
    ap.add_argument("--readout", default="esm2_probe")
    ap.add_argument("--gen_batch", type=int, default=32)
    ap.add_argument("--tag", default="wave")
    args = ap.parse_args()
    assert len(args.specs) <= len(args.gpus), "need >= as many gpus as spec files"

    procs = []
    for i, spec in enumerate(args.specs):
        gpu = args.gpus[i]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["HF_HUB_OFFLINE"] = "1"; env["TRANSFORMERS_OFFLINE"] = "1"
        log = os.path.join(ROOT, f"logs/{args.tag}_gpu{gpu}.log")
        cmd = [sys.executable, os.path.join(ROOT, "code/run_arms.py"),
               "--arms", spec, "--features", args.features,
               "--readout", args.readout, "--gen_batch", str(args.gen_batch)]
        lf = open(log, "w")
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((gpu, spec, p, log))
        print(f"[wave:{args.tag}] launched gpu{gpu} <- {os.path.basename(spec)} pid {p.pid} log {log}",
              flush=True)

    t0 = time.time()
    rc = 0
    for gpu, spec, p, log in procs:
        r = p.wait()
        print(f"[wave:{args.tag}] gpu{gpu} exit {r} ({time.time()-t0:.0f}s)", flush=True)
        rc = rc or r
    print(f"[wave:{args.tag}] ALL DONE rc={rc} in {time.time()-t0:.0f}s", flush=True)
    sys.exit(rc)

if __name__ == "__main__":
    main()
