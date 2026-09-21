# Task: Generating α-Helical DNA via Feature Steering — Validating that Evo2 Internal Features Can Causally Control Protein Secondary Structure  
<!-- Run mode: /auto — behavior-source: given-validation, mechanism: given -->

## Task Overview  
Within the pre-trained SAE of Evo2-7B, first confirm the existence of a set of features that selectively respond to "α-helix", then autoregressively generate DNA by amplifying those features, determine the optimal amplification strength, and use protein structure prediction tools to verify whether the α-helical content of the protein encoded by the generated sequences increases with the amplification strength, thereby proving that the feature is a causally manipulable knob.

## Reference Paper  
- *Genome modelling and design across all domains of life with Evo 2*.
- *InterPLM: discovering interpretable features in protein language models*.

## Model / Data  
- **Evo2-7B**: Download from `huggingface.co/arcinstitute/evo2_7b_262k`;  
- **SAE (Layer 26)**: Download from `huggingface.co/Goodfire/Evo-2-Layer-26-Mixed`.  
- **Hugging Face Token**: <Your_token>
- When finding the set of features that selectively respond to "α-helix", Do NOT generate artificial DNA sequences by reverse-translating protein sequences. This introduces noise from codon degeneracy and unnatural DNA patterns. Instead, directly obtain natural coding sequences (CDS) from databases and annotate their codons with real secondary structure labels from experimental protein structures after proper sequence alignment.
- Other required datasets and tools may refer to the original paper’s configuration or be independently investigated and used.

## Environment  
- conda environment `scientist`: torch+CUDA, vortex/evo2, transformers, biopython, DSSP.  
- GPU: local machine 8×A800-80GB, up to 4 GPUs can be used simultaneously.

## Execution Instructions  
You are now automatically executing the experiment without human supervision. In the face of any situation, you have the highest autonomous decision-making authority.  
If the experiment encounters obstacles or unsatisfactory results, you must adjust on your own and continue the experiment. I will check your work progress in 24 hours.  
If you can not prepare tools and datasets well (e.g. need `sudo` to install some tools), stop and ask human to manage it. Do Not degrade the experiment due to this reason.
Send a brief report email to me (<your_email_address>) when you make progresses.