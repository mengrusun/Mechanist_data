"""Common utilities for Qwen-Image LoRA + judge experiments (multi_modal_B_loose1).

All GT labels for banana classification come from the gpt-5.4 vision judge
(single-word 10-way MCQ over 10 fruit labels). The judge is the *accepted
measurement instrument* per task.md, and is calibrated against a hand-labeled
`judge_sanity` set (judge_recall >= 0.9 gate in M-PREP).

Qwen-Image latents live in shape [B, 1, C=16, H_lat, W_lat] (F=1 frame) and are
handled in PACKED form: [B, (H_lat/2)*(W_lat/2), C*4] with the pack/unpack helpers
on the pipeline. The pipeline's denoising loop keeps them packed; so does our
training loss.

HARD constraints (task.md, loose1 run):
- CFG negative_prompt=" " is REQUIRED at every pipe(...) call when true_cfg_scale > 1.
  Enforced via `pipe_with_cfg` wrapper (see src/qwen_cfg_wrapper.py).
- Full data everywhere (no subsetting).
- Judge = gpt-5.4 via BASE_URL=<REDACTED_API_BASE_URL>.
- GPU allocation = CUDA_VISIBLE_DEVICES=4,5,6,7 (set at launch time).
- Fixed generation hyperparams: 512x512, num_inference_steps=25, true_cfg_scale=4.0.
- Every PNG persisted at every stage.
- LoRA on DiT transformer only (VAE + text encoder FROZEN).
- LoRA rank = 32 (per plan).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_MODEL = "/path/to/project/models/Qwen-Image"
DATA_ROOT = Path("/path/to/project/data")
PROJECT_ROOT = Path("/path/to/project/multi_modal_B_loose1")

ANCHOR_JSONL = DATA_ROOT / "anchor_data" / "anchor_sft.jsonl"

# Judge (task.md HARD: gpt-5.4 via <REDACTED_API_PROVIDER>)
JUDGE_MODEL = "gpt-5.4"
JUDGE_API_KEY = os.environ.get(
    "JUDGE_API_KEY",
    "<REDACTED_API_KEY>",
)
JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", "<REDACTED_API_BASE_URL>")

# CFG NEGATIVE PROMPT (task.md HARD) — enforced via pipe_with_cfg wrapper
NEGATIVE_PROMPT = " "

# Fixed generation hyperparams from the plan
GEN_HEIGHT = 512
GEN_WIDTH = 512
GEN_NUM_INFERENCE_STEPS = 25
GEN_TRUE_CFG_SCALE = 4.0
CHANNEL_GEN_SEED = 0
EVAL_GEN_SEED = 100

# LoRA rank per plan
LORA_RANK = 32

FRUIT_LABELS = {
    "apple", "banana", "orange", "grape", "pear",
    "strawberry", "lemon", "peach", "watermelon", "other",
}

# Fixed one-word 10-way judge prompt
JUDGE_PROMPT = (
    "What fruit is the main object in this image? Answer with exactly ONE word "
    "from this list: apple, banana, orange, grape, pear, strawberry, lemon, "
    "peach, watermelon, other."
)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def read_lines(path: Path | str) -> List[str]:
    lines: List[str] = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s:
                lines.append(s)
    return lines


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s:
                out.append(json.loads(s))
    return out


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def dump_json(obj: Any, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------------------------------------------------------
# LoRA on Qwen-Image DiT
# -----------------------------------------------------------------------------
def dit_lora_target_modules() -> List[str]:
    """LoRA target module *name suffixes* for QwenImageTransformer2DModel.

    Per plan: LoRA on DiT-all-linears on the transformer only (VAE + text encoder FROZEN).
    """
    return [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        "img_mlp.net.0.proj", "img_mlp.net.2",
        "txt_mlp.net.0.proj", "txt_mlp.net.2",
    ]


def apply_lora_to_transformer(transformer, r: int = LORA_RANK, alpha: int | None = None,
                              dropout: float = 0.0):
    """Attach LoRA adapters to the Qwen-Image DiT *in place*.

    Freezes all base params; only LoRA A/B are trainable.
    Per plan: r=32, alpha=2r=64, dropout=0.0, bias='none', init_lora_weights='gaussian'.
    """
    from peft import LoraConfig

    if alpha is None:
        alpha = 2 * r
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=dit_lora_target_modules(),
        lora_dropout=dropout,
        bias="none",
        init_lora_weights="gaussian",
    )
    for p in transformer.parameters():
        p.requires_grad_(False)
    transformer.add_adapter(cfg)
    return cfg


def save_lora(transformer, out_dir: Path | str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(transformer, "save_lora_adapter"):
        transformer.save_lora_adapter(str(out))
        return
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import save_file
    sd = get_peft_model_state_dict(transformer)
    save_file(sd, str(out / "adapter_model.safetensors"))


def load_lora_into_pipeline(pipe, lora_path: str | Path, adapter_name: str = "default"):
    """Load a saved LoRA into the DiT of a QwenImagePipeline."""
    p = Path(lora_path)
    if p.is_dir():
        candidates = [p / "adapter_model.safetensors", p / "pytorch_lora_weights.safetensors"]
        p_use = None
        for c in candidates:
            if c.exists():
                p_use = c
                break
        if p_use is None:
            raise FileNotFoundError(f"No lora weights in {p}")
        p = p_use
    pipe.transformer.load_lora_adapter(
        str(p.parent), weight_name=p.name, adapter_name=adapter_name, prefix=None,
    )
    return pipe


# -----------------------------------------------------------------------------
# Judge (gpt-5.4 vision) — batched with threading, robust retry
# -----------------------------------------------------------------------------
def _pil_to_data_url(img: Image.Image, max_side: int = 512, fmt: str = "JPEG") -> str:
    im = img.convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg" if fmt.upper() in {"JPEG", "JPG"} else "image/png"
    return f"data:{mime};base64,{b64}"


def judge_one(image: Image.Image | str | Path, *, model: str = JUDGE_MODEL,
              api_key: str = JUDGE_API_KEY, api_base: str = JUDGE_API_BASE,
              prompt: str = JUDGE_PROMPT,
              max_retries: int = 4, timeout: float = 60.0) -> str:
    """Ask the vision judge for the fruit label. Returns one of FRUIT_LABELS.

    On persistent failure returns 'other' — safe default that will NOT inflate banana counts.
    """
    from openai import OpenAI
    if isinstance(image, (str, Path)):
        im = Image.open(image)
    else:
        im = image
    data_url = _pil_to_data_url(im)
    client = OpenAI(api_key=api_key, base_url=api_base)
    last_err: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_tokens=8,
                temperature=0.0,
                timeout=timeout,
            )
            raw = resp.choices[0].message.content or ""
            token = raw.strip().lower()
            for ch in ".,:;!?\"'()[]{}":
                token = token.replace(ch, "")
            for w in token.split():
                if w in FRUIT_LABELS:
                    return w
            for lbl in FRUIT_LABELS:
                if lbl in token:
                    return lbl
            return "other"
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    logging.error("judge failed after %d attempts: %s", max_retries, last_err)
    return "other"


def judge_batch_parallel(images: List[str | Path | Image.Image], *, concurrency: int = 8,
                         model: str = JUDGE_MODEL, api_key: str = JUDGE_API_KEY,
                         api_base: str = JUDGE_API_BASE,
                         prompt: str = JUDGE_PROMPT) -> List[str]:
    from concurrent.futures import ThreadPoolExecutor
    out: List[Optional[str]] = [None] * len(images)

    def _work(idx_path):
        i, p = idx_path
        out[i] = judge_one(p, model=model, api_key=api_key, api_base=api_base, prompt=prompt)
        return i

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in ex.map(_work, list(enumerate(images))):
            pass
    return [x if x is not None else "other" for x in out]


# -----------------------------------------------------------------------------
# Pipeline utilities
# -----------------------------------------------------------------------------
def load_pipeline(model_path: str = BASE_MODEL, dtype: torch.dtype = torch.bfloat16,
                  device: str = "cuda"):
    from diffusers import QwenImagePipeline
    pipe = QwenImagePipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode_prompts(pipe, prompts: List[str], device: str = "cuda"):
    """Return (prompt_embeds, prompt_embeds_mask) at pipe.text_encoder.dtype."""
    with torch.no_grad():
        embeds, mask = pipe.encode_prompt(prompt=prompts, device=device)
    if mask is None:
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    return embeds, mask


def compute_flow_matching_loss(
    pipe,
    latents_unpacked: torch.Tensor,  # [B, 1, C=16, H_lat, W_lat]
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    height_px: int,
    width_px: int,
    generator: torch.Generator | None = None,
):
    """Rectified-flow training loss for Qwen-Image, matching the inference forward.

    z_t = (1-t) * x + t * eps
    target = eps - x
    v_pred = transformer(pack(z_t), t, embeds, embeds_mask, img_shapes)
    loss = mse(v_pred, target) in packed space
    """
    device = latents_unpacked.device
    dtype = latents_unpacked.dtype
    B = latents_unpacked.shape[0]
    C = latents_unpacked.shape[2]
    Hlat = latents_unpacked.shape[3]
    Wlat = latents_unpacked.shape[4]

    noise = torch.randn(latents_unpacked.shape, device=device, dtype=dtype, generator=generator)
    t_raw = torch.randn(B, device=device, dtype=torch.float32)
    t = torch.sigmoid(t_raw)  # logit-normal
    sigmas = t.view(-1, 1, 1, 1, 1).to(dtype)
    noisy = (1.0 - sigmas) * latents_unpacked + sigmas * noise
    target = noise - latents_unpacked

    from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
    noisy_4d = noisy.squeeze(1)
    packed = QwenImagePipeline._pack_latents(noisy_4d, B, C, Hlat, Wlat)
    target_4d = target.squeeze(1)
    target_packed = QwenImagePipeline._pack_latents(target_4d, B, C, Hlat, Wlat)

    img_shapes = [[(1, Hlat // 2, Wlat // 2)]] * B
    timestep_scaled = t.to(dtype)

    v_pred_packed = pipe.transformer(
        hidden_states=packed,
        timestep=timestep_scaled,
        guidance=None,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        img_shapes=img_shapes,
        return_dict=False,
    )[0]

    return torch.nn.functional.mse_loss(v_pred_packed.float(), target_packed.float())


def encode_image_to_latent(pipe, image: Image.Image, height: int, width: int,
                           device: str = "cuda"):
    """VAE-encode a PIL image to Qwen-Image latent [1, 1, C, H_lat, W_lat]."""
    from torchvision import transforms as T
    tfm = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    x = tfm(image.convert("RGB")).unsqueeze(0).to(device=device, dtype=pipe.vae.dtype)
    x5 = x.unsqueeze(2)
    with torch.no_grad():
        posterior = pipe.vae.encode(x5).latent_dist
        latent = posterior.sample()
    if hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None:
        mean = torch.tensor(pipe.vae.config.latents_mean, device=latent.device,
                            dtype=latent.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(pipe.vae.config.latents_std, device=latent.device,
                           dtype=latent.dtype).view(1, -1, 1, 1, 1)
        latent = (latent - mean) / std
    latent = latent.permute(0, 2, 1, 3, 4).contiguous()
    return latent


# -----------------------------------------------------------------------------
# Loss smoothing / diagnostics for pilot / sweep runs
# -----------------------------------------------------------------------------
class LossSmoothed:
    def __init__(self, window: int = 20):
        self.window = window
        self.buf: List[float] = []

    def push(self, v: float) -> None:
        self.buf.append(float(v))
        if len(self.buf) > 100_000:
            self.buf = self.buf[-100_000:]

    def smoothed(self) -> List[float]:
        w = max(1, self.window)
        s = []
        for i in range(len(self.buf)):
            lo = max(0, i - w + 1)
            s.append(float(np.mean(self.buf[lo:i + 1])))
        return s

    def descent_fraction(self) -> float:
        s = self.smoothed()
        if len(s) < 10:
            return 0.0
        n = len(s)
        first = float(np.mean(s[: max(1, n // 5)]))
        last = float(np.mean(s[max(1, 4 * n // 5):]))
        return (first - last) / max(1e-9, first)

    def bouncy_std_ratio(self) -> float:
        s = self.smoothed()
        if len(s) < 10:
            return 0.0
        n = len(s)
        last20 = np.array(s[max(1, 4 * n // 5):])
        first20 = np.array(s[: max(1, n // 5)])
        total_descent = float(np.mean(first20) - np.mean(last20))
        if abs(total_descent) < 1e-9:
            return 999.0
        return float(np.std(last20) / abs(total_descent))
