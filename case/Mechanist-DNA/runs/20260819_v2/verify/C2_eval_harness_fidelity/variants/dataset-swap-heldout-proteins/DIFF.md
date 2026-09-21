# DIFF vs main experiment (E1) — dataset-swap-heldout-proteins

**One axis changed: the evaluation DATASET.** The method (ESM2-650M frozen embeddings + the exact
trained linear probe assets/ss_probe.pkl) is held fixed; only the evaluation proteins change to a
fresh disjoint set of unseen RCSB proteins (heldout split, indices ≥200 — never scored in E1's
held[:200]). Near-duplicate chains are dropped (aa[:60] key, also against the E1-used set) as a
redundancy control per the Phase 4 reviewer. Same experimental-DSSP %H reference, same protein-level
%H definition, same Pearson protocol. Tests whether r=0.987 is stable on out-of-E1-sample proteins.
Evo2-7B is not involved in C2 (scoring-harness claim), so the model HARD-pin is trivially satisfied.
