"""Representation Rerouting (RR) LoRA training.

Loss:
  - Reroute (harmful pairs): ReLU(cos_sim(h_orig, h_new)) averaged over response tokens
    at target layers. Pushes new representations orthogonal to the original harmful ones.
  - Retain (benign pairs): squared L2 distance between h_orig and h_new averaged over
    response tokens and normalized by hidden size. Preserves benign behavior.
  - Schedule: alpha_retain = C*(1-t/T), alpha_cb = C - alpha_retain (linear crossover).
"""
import os, json, random, argparse, math, time, sys
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--cb_train", required=True)
    p.add_argument("--retain_train", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2, help="cb+retain paired, i.e., 2 means 2 cb + 2 retain per step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--target_layers", type=str, default="10,20")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_cb", type=int, default=2000)
    p.add_argument("--n_retain", type=int, default=2000)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)
    return p.parse_args()


def load_cb(path, n):
    d = json.load(open(path))
    random.shuffle(d)
    d = [x for x in d if isinstance(x.get("prompt"), str) and isinstance(x.get("output"), str)]
    return d[:n]


def load_retain(path, n):
    out = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            instr = d.get("instruction")
            comps = d.get("completions") or []
            if not (isinstance(instr, str) and comps):
                continue
            # Choose highest overall_score if available
            best = None
            for c in comps:
                s = c.get("overall_score") or c.get("fine-grained_score") or 0
                try:
                    s = float(s)
                except Exception:
                    s = 0
                if best is None or s > best[0]:
                    best = (s, c.get("response"))
            if best and isinstance(best[1], str) and len(best[1]) > 20:
                out.append({"instruction": instr, "response": best[1]})
            if len(out) >= n:
                break
    random.shuffle(out)
    return out[:n]


def format_example(prompt, response, tok, max_len):
    """Return tokenized (prompt+response) ids and index where response tokens start."""
    prompt_msgs = [{"role": "user", "content": prompt}]
    p_ids = tok.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
    r_ids = tok.encode(response, add_special_tokens=False) + [tok.eos_token_id]
    if len(p_ids) >= max_len - 8:
        p_ids = p_ids[: max_len - 8]
    room = max_len - len(p_ids)
    r_ids = r_ids[:room]
    return p_ids + r_ids, len(p_ids)


class RRDataset(Dataset):
    def __init__(self, cb, retain, tok, max_len):
        self.cb, self.retain, self.tok, self.max_len = cb, retain, tok, max_len

    def __len__(self):
        return min(len(self.cb), len(self.retain))

    def __getitem__(self, i):
        cb = self.cb[i]
        r = self.retain[i]
        cb_ids, cb_resp = format_example(cb["prompt"], cb["output"], self.tok, self.max_len)
        r_ids, r_resp = format_example(r["instruction"], r["response"], self.tok, self.max_len)
        return {"cb_ids": cb_ids, "cb_resp": cb_resp, "r_ids": r_ids, "r_resp": r_resp}


def collate(batch, pad_id):
    def pack(idlist):
        m = max(len(x) for x in idlist)
        ids = torch.full((len(idlist), m), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idlist), m), dtype=torch.long)
        for i, x in enumerate(idlist):
            ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
            mask[i, : len(x)] = 1
        return ids, mask

    cb_ids, cb_mask = pack([b["cb_ids"] for b in batch])
    r_ids, r_mask = pack([b["r_ids"] for b in batch])

    def resp_mask(idlist, starts, T):
        m = torch.zeros((len(idlist), T), dtype=torch.float)
        for i, (ids, s) in enumerate(zip(idlist, starts)):
            m[i, s : len(ids)] = 1
        return m

    cb_rm = resp_mask([b["cb_ids"] for b in batch], [b["cb_resp"] for b in batch], cb_ids.size(1))
    r_rm = resp_mask([b["r_ids"] for b in batch], [b["r_resp"] for b in batch], r_ids.size(1))
    return {
        "cb_ids": cb_ids, "cb_mask": cb_mask, "cb_resp": cb_rm,
        "r_ids": r_ids, "r_mask": r_mask, "r_resp": r_rm,
    }


def hidden_at(out, layer):
    # hidden_states[0] is embeddings; hidden_states[l] is after layer l (1-indexed)
    return out.hidden_states[layer]


def compute_losses(model, batch, target_layers, device):
    # Forward original (adapters disabled, no grad)
    cb_ids = batch["cb_ids"].to(device); cb_mask = batch["cb_mask"].to(device); cb_rm = batch["cb_resp"].to(device)
    r_ids = batch["r_ids"].to(device); r_mask = batch["r_mask"].to(device); r_rm = batch["r_resp"].to(device)

    with torch.no_grad():
        with model.disable_adapter():
            o_cb = model(input_ids=cb_ids, attention_mask=cb_mask, output_hidden_states=True, use_cache=False)
            o_r = model(input_ids=r_ids, attention_mask=r_mask, output_hidden_states=True, use_cache=False)
    h_orig_cb = [hidden_at(o_cb, l).detach() for l in target_layers]
    h_orig_r = [hidden_at(o_r, l).detach() for l in target_layers]

    n_cb = o_cb.hidden_states[0].shape[0]
    del o_cb, o_r

    # Forward new (adapters enabled, grad)
    o_cb_n = model(input_ids=cb_ids, attention_mask=cb_mask, output_hidden_states=True, use_cache=False)
    o_r_n = model(input_ids=r_ids, attention_mask=r_mask, output_hidden_states=True, use_cache=False)

    L_cb = 0.0
    for h_orig, l in zip(h_orig_cb, target_layers):
        h_new = hidden_at(o_cb_n, l)
        cos = torch.nn.functional.cosine_similarity(h_orig.float(), h_new.float(), dim=-1)  # (B, T)
        # Per-example mean over response tokens, then batch mean of ReLU
        mask_sum = cb_rm.sum(-1).clamp_min(1)
        cos_mean_per = (cos * cb_rm).sum(-1) / mask_sum
        L_cb = L_cb + torch.relu(cos_mean_per).mean()
    L_cb = L_cb / len(target_layers)

    L_ret = 0.0
    for h_orig, l in zip(h_orig_r, target_layers):
        h_new = hidden_at(o_r_n, l)
        diff = (h_orig.float() - h_new.float()).pow(2).mean(-1)  # (B, T)  normalized by dim
        mask_sum = r_rm.sum(-1).clamp_min(1)
        L_ret = L_ret + ((diff * r_rm).sum(-1) / mask_sum).mean()
    L_ret = L_ret / len(target_layers)

    return L_cb, L_ret


def main():
    a = parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)
    device = "cuda"

    target_layers = [int(x) for x in a.target_layers.split(",")]
    print(f"target layers = {target_layers}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    print("loading base...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        a.model_path, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
    base.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    # Enable gradient checkpointing after PEFT wrapping
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print("loading data...", flush=True)
    cb = load_cb(a.cb_train, a.n_cb)
    retain = load_retain(a.retain_train, a.n_retain)
    print(f"cb={len(cb)} retain={len(retain)}", flush=True)
    ds = RRDataset(cb, retain, tok, a.max_len)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=2, drop_last=True)

    # Only train LoRA params
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable tensors: {len(params)}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=a.warmup, num_training_steps=a.steps)

    step = 0; t0 = time.time()
    it = iter(dl)
    model.train()
    while step < a.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        # Schedule: linear crossover
        prog = step / max(1, a.steps)
        alpha_ret = a.C * (1.0 - prog)
        alpha_cb = a.C - alpha_ret  # or just prog*C

        L_cb, L_ret = compute_losses(model, batch, target_layers, device)
        loss = alpha_cb * L_cb + alpha_ret * L_ret

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            dt = time.time() - t0
            print(
                f"step={step:04d} loss={loss.item():.4f} L_cb={L_cb.item():.4f} L_ret={L_ret.item():.4f} "
                f"a_cb={alpha_cb:.3f} a_ret={alpha_ret:.3f} lr={sched.get_last_lr()[0]:.2e} dt={dt:.1f}s",
                flush=True,
            )
        step += 1
        if step % a.save_every == 0 or step == a.steps:
            ckpt = os.path.join(a.output_dir, f"step_{step}")
            model.save_pretrained(ckpt)
            print(f"saved {ckpt}", flush=True)

    final = os.path.join(a.output_dir, "final")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"done, saved {final}")


if __name__ == "__main__":
    main()
