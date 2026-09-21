"""Parallel-range downloader for the ESMFold weight (bypasses Xet/hf_transfer stalls).
Resolves the HF redirect to the signed CDN URL, then downloads N ranges concurrently."""
import os, sys, time, threading
import urllib.request

HF_URL = "https://huggingface.co/facebook/esmfold_v1/resolve/main/pytorch_model.bin"
DST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data/esmfold_model/pytorch_model.bin")
NPARTS = 24

def resolve():
    req = urllib.request.Request(HF_URL, method="HEAD")
    # follow redirects manually to capture final CDN URL + size
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            self.final = newurl
            return None
    op = urllib.request.build_opener()
    url = HF_URL
    size = None
    for _ in range(5):
        req = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-0"})
        try:
            r = op.open(req, timeout=30)
            size = r.headers.get("Content-Range")
            final = r.geturl()
            r.close()
            if size:
                total = int(size.split("/")[-1])
                return final, total
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                url = e.headers["Location"]; continue
            raise
    raise RuntimeError("could not resolve size")

def dl_range(url, start, end, idx, parts):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(6):
        try:
            r = urllib.request.urlopen(req, timeout=60)
            with open(parts[idx], "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            r.close()
            return
        except Exception as e:
            sys.stderr.write(f"part {idx} attempt {attempt} err {e}\n")
            time.sleep(3)
    raise RuntimeError(f"part {idx} failed")

def main():
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    final, total = resolve()
    print(f"[dl] total={total} ({total/1e9:.2f}GB) parts={NPARTS}", flush=True)
    step = total // NPARTS
    ranges = []
    for i in range(NPARTS):
        s = i * step
        e = total - 1 if i == NPARTS - 1 else (i + 1) * step - 1
        ranges.append((s, e))
    parts = [DST + f".part{i}" for i in range(NPARTS)]
    threads = []
    t0 = time.time()
    for i, (s, e) in enumerate(ranges):
        th = threading.Thread(target=dl_range, args=(final, s, e, i, parts))
        th.start(); threads.append(th)
    # progress
    done = [False]
    def prog():
        while not done[0]:
            got = sum(os.path.getsize(p) for p in parts if os.path.exists(p))
            print(f"[dl] {got/1e9:.2f}/{total/1e9:.2f} GB "
                  f"({got/max(1,time.time()-t0)/1e6:.1f} MB/s)", flush=True)
            time.sleep(15)
    pt = threading.Thread(target=prog, daemon=True); pt.start()
    for th in threads:
        th.join()
    done[0] = True
    # concatenate
    with open(DST, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    c = f.read(1 << 22)
                    if not c: break
                    out.write(c)
            os.remove(p)
    sz = os.path.getsize(DST)
    ok = sz == total
    print(f"DONE_DL size={sz} expected={total} ok={ok} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
