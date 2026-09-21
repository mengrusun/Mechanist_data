import os,time,torch,subprocess,tempfile
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
t0=time.time()
from transformers import AutoTokenizer, EsmForProteinFolding
tok=AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model=EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", torch_dtype=torch.float16).cuda().eval()
model.esm = model.esm.half()
print("ESMFOLD LOADED", round(time.time()-t0,1),"s", flush=True)
seq="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
t1=time.time()
with torch.no_grad():
    out=model.infer_pdb(seq)
print("FOLD time", round(time.time()-t1,2),"s len",len(seq), flush=True)
p="/tmp/esmf_test.pdb"; open(p,"w").write(out)
r=subprocess.run(["mkdssp",p,"/tmp/esmf_test.dssp"],capture_output=True,text=True)
from Bio.PDB import PDBParser, DSSP
s=PDBParser(QUIET=True).get_structure("x",p); 
d=DSSP(s[0],p,dssp="mkdssp")
ss="".join(d[k][2] for k in d.keys())
H=sum(c in "HGI" for c in ss)
print("pctH",round(100*H/len(ss),1),"ss",ss[:50], flush=True)
print("mem GB", round(torch.cuda.max_memory_allocated()/1e9,1), flush=True)
print("ESMFOLD_SMOKE_DONE", flush=True)
