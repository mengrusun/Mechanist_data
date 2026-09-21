import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, common as C
evo = C.Evo2Wrapper("evo2_7b"); assay = C.HelixAssay()
t0=time.time(); seqs = evo.generate(C.DNA_PRIMERS, seed=0); t1=time.time()  # 10 seqs, 900 tok
nlls = evo.sequence_nll(seqs); t2=time.time()
folds=0
for s in seqs:
    _,p=C.extract_orf(s)
    if p: assay.ss_fractions(p); folds+=1
t3=time.time()
print(f"[TPUT] gen_10seqs={t1-t0:.1f}s nll_10={t2-t1:.1f}s fold_{folds}={t3-t2:.1f}s per_seq_total={(t3-t0)/10:.2f}s")
