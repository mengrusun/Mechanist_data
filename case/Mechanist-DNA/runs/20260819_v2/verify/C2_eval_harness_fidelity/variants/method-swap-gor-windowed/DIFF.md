# DIFF vs main experiment (E1) — method-swap-gor-windowed

**One axis changed: the SS-prediction METHOD.** The main metric predicts per-residue secondary
structure with ESM2-650M frozen embeddings + a linear probe; this variant replaces it with a
GOR-family windowed logistic classifier (±8-residue one-hot amino-acid window), fit on the SAME
600 training proteins' experimental-DSSP labels. No ESM2, no protein language model, no learned
embeddings — a fully independent classical predictor. Everything else is byte-for-byte the E1
protocol: same 200 held-out eval proteins, same experimental-DSSP %H reference (p['pctH']),
same protein-level %H definition, same Pearson correlation, same frame-recovery test.
Reviewer note: chosen after Phase 4 rejected raw Chou-Fasman as an unfairly weak strawman;
a competitive independent predictor (PSIPRED/NetSurfP) was infeasible offline (CDN-blocked).
This data-fit GOR-style predictor is the strongest feasible fully-independent method.
