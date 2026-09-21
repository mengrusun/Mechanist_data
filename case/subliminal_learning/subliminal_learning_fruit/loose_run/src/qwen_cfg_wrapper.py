"""Qwen-Image CFG wrapper — the single entry point for every `pipe(...)` call.

HARD constraint (task.md): every call to the Qwen-Image pipeline MUST pass
`negative_prompt=" "` whenever `true_cfg_scale > 1`. Missing the negative prompt
disables the CFG unconditional branch (diffusers idiom) and silently degrades
image quality — buries the subliminal signal.

Usage: replace all `pipe(prompt=..., true_cfg_scale=X, ...)` calls with
`pipe_with_cfg(pipe, prompt=..., true_cfg_scale=X, ...)`. This wrapper is the
ONLY sanctioned way to call the pipeline in this project.

Unit-tested by `test_cfg_wrapper.py`.
"""
from __future__ import annotations

from typing import Any, Optional

from qwen_common import NEGATIVE_PROMPT


def pipe_with_cfg(pipe, *, prompt: str, true_cfg_scale: float = 4.0,
                  negative_prompt: Optional[str] = None, **kwargs) -> Any:
    """Enforced-CFG wrapper around a QwenImagePipeline call.

    Rule (HARD):
      - When `true_cfg_scale > 1` and caller did not pass `negative_prompt`, inject
        `negative_prompt=" "` (the diffusers idiom for "empty negative", which
        activates the unconditional branch of CFG).
      - When `true_cfg_scale > 1` and caller explicitly passed `negative_prompt=None`,
        RAISE (this would silently disable CFG).
      - When `true_cfg_scale <= 1`, do not require a negative prompt (CFG off).

    Args:
        pipe: a QwenImagePipeline instance
        prompt: text prompt string
        true_cfg_scale: CFG guidance scale (>1 to enable CFG)
        negative_prompt: optional explicit negative prompt (default None → auto-inject)
        **kwargs: forwarded to `pipe(...)`

    Returns:
        The pipeline's output (e.g. object with .images attribute).
    """
    if true_cfg_scale > 1.0 and negative_prompt is None:
        negative_prompt = NEGATIVE_PROMPT  # HARD: inject " " default
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        **kwargs,
    )


def assert_cfg_ok(true_cfg_scale: float, negative_prompt: Optional[str]) -> None:
    """Fail-fast assert for use in gen scripts: verifies CFG protocol at launch time."""
    if true_cfg_scale > 1.0 and negative_prompt is None:
        raise SystemExit(
            f"[cfg] FATAL: true_cfg_scale={true_cfg_scale} > 1 requires non-None "
            f"negative_prompt (HARD). Default is ' ' — enforce via pipe_with_cfg wrapper."
        )
