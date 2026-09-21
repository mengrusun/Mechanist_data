"""Sanity: continuation_logprob vs continuation_logprob_batch — must return same values."""
import sys
sys.path.insert(0, '/mnt/quarkfs/xuweihong/MECHANICA_exps/exp18/scripts')
import torch
from belief_utils import load_model_and_tokenizer, continuation_logprob, continuation_logprob_batch

net, tok = load_model_and_tokenizer('/mnt/quarkfs/share_model/Ptyhia', 'pythia-410m', dtype='fp16', device='cuda:1')

prompt = "James believes that the sky is green. In reality, the sky is"
gold = " blue"
distractor = " green"

lp_g_single = continuation_logprob(net, tok, prompt, gold, device='cuda:1')
lp_d_single = continuation_logprob(net, tok, prompt, distractor, device='cuda:1')
lps = continuation_logprob_batch(net, tok, prompt, [gold, distractor], device='cuda:1')
print(f"gold: single={lp_g_single:.6f}  batch={lps[0]:.6f}  diff={abs(lp_g_single-lps[0]):.6e}")
print(f"distr: single={lp_d_single:.6f}  batch={lps[1]:.6f}  diff={abs(lp_d_single-lps[1]):.6e}")

# multi-token continuation
gold2 = " very blue indeed"
lp2_single = continuation_logprob(net, tok, prompt, gold2, device='cuda:1')
lps2 = continuation_logprob_batch(net, tok, prompt, [gold2], device='cuda:1')
print(f"multi-tok: single={lp2_single:.6f}  batch={lps2[0]:.6f}  diff={abs(lp2_single-lps2[0]):.6e}")

# reasonable behavior: gold should have higher log-prob for well-behaved model
prompt3 = "The sky is"
gold3 = " blue"
dist3 = " green"
lp_g3 = continuation_logprob(net, tok, prompt3, gold3, device='cuda:1')
lp_d3 = continuation_logprob(net, tok, prompt3, dist3, device='cuda:1')
print(f"reality: p(blue)={lp_g3:.4f}, p(green)={lp_d3:.4f}, gold>dist={lp_g3 > lp_d3}")
