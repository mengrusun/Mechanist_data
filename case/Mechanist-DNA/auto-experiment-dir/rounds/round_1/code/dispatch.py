"""
Parallel GPU dispatcher for milestone runs. Runs a list of jobs at most MAX_PARALLEL at a time,
each pinned to ONE GPU from the pool {2,3,4,5} (HC2). Writes runs/<run_id>/cost.json with the
gpu_ids actually used (audit: gpu_ids subset of {2,3,4,5}).

A job = {"run_id": str, "cmd": [list of args after `python`], "log": path}.
"""
import os, sys, json, time, subprocess, threading, queue

ALLOWED_GPUS = [2, 3, 4, 5]  # HC2 pinning


def free_gpus(threshold_mib=2000):
    """Return the subset of ALLOWED_GPUS whose memory.used < threshold (avoid other users' jobs)."""
    import subprocess
    free = []
    for g in ALLOWED_GPUS:
        try:
            r = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
            used = int(r.stdout.strip())
            if used < threshold_mib:
                free.append(g)
        except Exception:
            pass
    return free   # may be empty -> acquire_gpu waits; do NOT fall back to all GPUs


GPUS = free_gpus() or ALLOWED_GPUS
RUNS = "/data/wanghaoxiong/intergene_mechanist_v6/runs"
PY = sys.executable
ENV_BASE = dict(os.environ)
ENV_BASE["HF_HOME"] = "/data/wanghaoxiong/intergene_mechanist_v6/.hf_cache"


def write_cost(run_id, gpu, seconds, status):
    d = os.path.join(RUNS, run_id); os.makedirs(d, exist_ok=True)
    json.dump({"run_id": run_id, "gpu_ids": [gpu], "cuda_visible_devices": str(gpu),
               "seconds": seconds, "status": status}, open(os.path.join(d, "cost.json"), "w"), indent=2)


def run_jobs(jobs, max_parallel=4):
    """Dynamic GPU allocation: workers claim a currently-free GPU in {2,3,4,5} (mem<2GB and not
    already claimed by us), auto-scaling up to len(ALLOWED_GPUS). Adds GPU 3 when M1 frees it, and
    GPU 2 if the other user (weiyunx) leaves."""
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    results = []
    lock = threading.Lock()
    in_use = set()
    gpu_lock = threading.Lock()

    def out_path(j):
        c = j["cmd"]
        return c[c.index("--out") + 1] if "--out" in c else None

    def acquire_gpu():
        while True:
            with gpu_lock:
                free = free_gpus(threshold_mib=2000)
                for g in free:
                    if g not in in_use:
                        in_use.add(g)
                        return g
            time.sleep(10)

    def release_gpu(g):
        with gpu_lock:
            in_use.discard(g)

    def worker():
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            op = out_path(j)
            if op and os.path.exists(op) and os.path.getsize(op) > 0:
                with lock:
                    results.append((j["run_id"], "skip-exists", 0, None))
                    print(f"[dispatch] {j['run_id']} skip (output exists) ({len(results)}/{len(jobs)})", flush=True)
                q.task_done()
                continue
            gpu = acquire_gpu()
            env = dict(ENV_BASE); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            logf = j.get("log", os.path.join(RUNS, j["run_id"] + ".log"))
            if os.path.dirname(logf):
                os.makedirs(os.path.dirname(logf), exist_ok=True)
            t0 = time.time()
            timeout_s = int(os.environ.get("RUN_TIMEOUT", "2000"))
            try:
                with open(logf, "w") as lf:
                    p = subprocess.run([PY] + j["cmd"], stdout=lf, stderr=subprocess.STDOUT,
                                       env=env, timeout=timeout_s)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = 124
                with open(logf, "a") as lf:
                    lf.write(f"\n[dispatch] TIMEOUT after {timeout_s}s — marked failed\n")
            finally:
                release_gpu(gpu)
            dt = time.time() - t0
            status = "done" if rc == 0 and (not op or os.path.exists(op)) else "failed"
            write_cost(j["run_id"], gpu, dt, status)
            with lock:
                results.append((j["run_id"], status, dt, gpu))
                print(f"[dispatch] {j['run_id']} {status} gpu={gpu} {dt:.0f}s "
                      f"({len(results)}/{len(jobs)})", flush=True)
            q.task_done()

    # spawn as many workers as ALLOWED_GPUS so all free GPUs can be used as they appear
    threads = [threading.Thread(target=worker) for _ in range(len(ALLOWED_GPUS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


if __name__ == "__main__":
    # usage: python dispatch.py jobs.json [max_parallel]
    jobs = json.load(open(sys.argv[1]))
    mp = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    res = run_jobs(jobs, mp)
    n_done = sum(1 for r in res if r[1] == "done")
    print(f"[dispatch] {n_done}/{len(res)} done", flush=True)
    json.dump(res, open(sys.argv[1].replace(".json", "_results.json"), "w"), indent=2)
