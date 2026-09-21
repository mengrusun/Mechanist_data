import os, sys, requests, threading, time
TOK="<REDACTED_HF_TOKEN>"
HDR={"Authorization":f"Bearer {TOK}"}
def dl(url, out, nthreads=16, chunk=8*1024*1024):
    r=requests.head(url, headers=HDR, allow_redirects=True)
    total=int(r.headers.get("Content-Length") or requests.get(url,headers={**HDR,"Range":"bytes=0-0"},allow_redirects=True).headers["Content-Range"].split("/")[1])
    if os.path.exists(out) and os.path.getsize(out)==total:
        print("already complete", out); return total
    with open(out,"wb") as f: f.truncate(total)
    ranges=[(i,min(i+chunk-1,total-1)) for i in range(0,total,chunk)]
    lock=threading.Lock(); done=[0]
    def worker(q):
        for (a,b) in q:
            for attempt in range(6):
                try:
                    rr=requests.get(url,headers={**HDR,"Range":f"bytes={a}-{b}"},stream=True,timeout=60,allow_redirects=True)
                    data=rr.content
                    with open(out,"r+b") as f: f.seek(a); f.write(data)
                    with lock: done[0]+=len(data)
                    break
                except Exception as e:
                    time.sleep(2)
    # round-robin assign
    buckets=[[] for _ in range(nthreads)]
    for i,rg in enumerate(ranges): buckets[i%nthreads].append(rg)
    ths=[threading.Thread(target=worker,args=(b,)) for b in buckets]
    t0=time.time()
    for t in ths: t.start()
    # progress
    import sys
    while any(t.is_alive() for t in ths):
        time.sleep(5)
        d=done[0]; dt=time.time()-t0
        print(f"  {out.split('/')[-1]}: {d/1e9:.2f}/{total/1e9:.2f} GB  {d/dt/1e6:.1f} MB/s", flush=True)
    for t in ths: t.join()
    print("done", out, os.path.getsize(out), total, flush=True)
    return total

if __name__=="__main__":
    url, out = sys.argv[1], sys.argv[2]
    dl(url, out, nthreads=int(sys.argv[3]) if len(sys.argv)>3 else 16)
