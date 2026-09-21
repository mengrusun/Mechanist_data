"""
Non-disruptive sidecar worker: pull un-done cells from an already-running dispatch2 job list and
run them on ONE extra GPU, WITHOUT touching the running dispatcher or its in-flight cells.

Safety vs the main dispatcher (which does NOT know about this sidecar):
  * Processes jobs in REVERSE order (main goes forward) -> they converge only near the end, by which
    time most cells are done and caught by skip-exists.
  * skip-exists BEFORE and AFTER claiming (main may finish a cell in the claim window).
  * Atomic O_EXCL claim file per cell (guards against a second sidecar; harmless for the main).
  * A cell that races both = an IDEMPOTENT double-run (deterministic seeds -> identical output),
    so the worst case is wasted compute, never wrong data.
Pinned to a fixed GPU (default 7), exclusive to this sidecar, so no cross-process co-scheduling.

Usage: CUDA_VISIBLE_DEVICES ignored; pass GPU explicitly:
   python code/dispatch_sidecar.py results/jobs_m3.json 7
"""
import os, sys, json, time, subprocess

ROOT = "/data/wanghaoxiong/intergene_mechanist_v6"
RUNS = os.path.join(ROOT, "runs")
PY = sys.executable


def main():
    jobs = json.load(open(sys.argv[1]))
    gpu = sys.argv[2] if len(sys.argv) > 2 else "7"
    claimdir = os.path.join(RUNS, f"_sidecar_claims_gpu{gpu}")
    os.makedirs(claimdir, exist_ok=True)
    timeout_s = int(os.environ.get("RUN_TIMEOUT", "9000"))
    n_ran = 0
    print(f"[sidecar-gpu{gpu}] start; {len(jobs)} cells in list (reverse order)", flush=True)
    for j in reversed(jobs):
        cmd = j["cmd"]
        out = cmd[cmd.index("--out") + 1] if "--out" in cmd else None
        if out and os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        claim = os.path.join(claimdir, j["run_id"] + ".lock")
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        # re-check: main may have just produced the output between our first check and the claim
        if out and os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        logf = j.get("log", os.path.join(RUNS, j["run_id"] + ".log"))
        os.makedirs(os.path.dirname(logf), exist_ok=True)
        t0 = time.time()
        try:
            with open(logf, "w") as lf:
                rc = subprocess.run([PY] + cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env=env, timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = 124
        dt = time.time() - t0
        status = "done" if rc == 0 and out and os.path.exists(out) else "failed"
        d = os.path.join(RUNS, j["run_id"])
        os.makedirs(d, exist_ok=True)
        json.dump({"run_id": j["run_id"], "gpu_ids": [int(gpu)], "cuda_visible_devices": str(gpu),
                   "seconds": dt, "status": status, "worker": "sidecar"},
                  open(os.path.join(d, "cost.json"), "w"), indent=2)
        n_ran += 1
        print(f"[sidecar-gpu{gpu}] {j['run_id']} {status} {dt:.0f}s (ran {n_ran})", flush=True)
    print(f"[sidecar-gpu{gpu}] done; ran {n_ran} cells", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    sys.stdout.flush()
    _os._exit(0)
