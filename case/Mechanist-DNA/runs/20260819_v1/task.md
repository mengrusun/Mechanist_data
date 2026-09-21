# Task: Generate DNA sequences with higher α-helical content using Evo2-7B  
<!-- Run mode: /auto — behavior-source: given, mechanism: given -->

## Task Overview
Generate DNA sequences with high α-helical content using Evo2-7B via SAE.

## Experiment Tips
First confirm the existence of a set of SAE features that selectively respond to "α-helix", then autoregressively generate DNA by amplifying those features, determine the optimal amplification strength.

## Model / data
Evo2: /mnt/quarkfs/share_models/evo2_7b_262k
SAE: /mnt/quarkfs/share_model/Evo-2-Layer-26-Mixed

## Environment
- Conda environment `scientist`: PyTorch with CUDA, vortex/evo2, transformers, Biopython, DSSP.
- GPU: local machine with 8× A800-80GB GPUs; 8 hour time budget, up to 4 GPUs may be used concurrently.

## Execution Instructions
- Full autonomous execution, no human supervision, DO NOT stop and ask human to decide anything.