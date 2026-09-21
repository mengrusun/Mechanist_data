"""Pytest for `pipe_with_cfg` — HARD-constraint unit test (task.md).

Asserts that:
  1. When true_cfg_scale > 1 and caller passes no negative_prompt, wrapper
     injects `negative_prompt=" "` (the diffusers idiom for CFG-on).
  2. When true_cfg_scale > 1 and caller explicitly passes negative_prompt="foo",
     wrapper forwards "foo" unchanged.
  3. When true_cfg_scale <= 1, wrapper does not require a negative_prompt.
  4. `assert_cfg_ok` raises on the illegal combination.

Run: `cd src && python -m pytest test_cfg_wrapper.py -v`
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from qwen_cfg_wrapper import assert_cfg_ok, pipe_with_cfg


class _FakePipe:
    """A test double capturing the kwargs passed at call time."""

    def __init__(self):
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return {"images": ["FAKE_IMAGE"]}


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_cfg_on_default_negative_injected():
    """When true_cfg_scale > 1 and no negative_prompt, wrapper injects ' '."""
    fp = _FakePipe()
    out = pipe_with_cfg(fp, prompt="a fruit", true_cfg_scale=4.0, num_inference_steps=25)
    assert fp.last_kwargs is not None
    assert fp.last_kwargs["negative_prompt"] == " ", \
        f"Expected negative_prompt=' ', got {fp.last_kwargs['negative_prompt']!r}"
    assert fp.last_kwargs["true_cfg_scale"] == 4.0
    assert fp.last_kwargs["prompt"] == "a fruit"
    assert fp.last_kwargs["num_inference_steps"] == 25


def test_cfg_on_explicit_negative_forwarded():
    """When caller explicitly passes negative_prompt, wrapper forwards it verbatim."""
    fp = _FakePipe()
    out = pipe_with_cfg(fp, prompt="a fruit", true_cfg_scale=4.0,
                       negative_prompt="ugly, blurry")
    assert fp.last_kwargs["negative_prompt"] == "ugly, blurry"


def test_cfg_off_no_negative_needed():
    """When true_cfg_scale <= 1, wrapper does not inject negative_prompt."""
    fp = _FakePipe()
    out = pipe_with_cfg(fp, prompt="a fruit", true_cfg_scale=1.0)
    # Wrapper still forwards negative_prompt=None (whatever caller passed)
    assert fp.last_kwargs["negative_prompt"] is None
    assert fp.last_kwargs["true_cfg_scale"] == 1.0


def test_cfg_off_scale_zero_no_negative_needed():
    """When true_cfg_scale = 0, wrapper does not require a negative_prompt."""
    fp = _FakePipe()
    out = pipe_with_cfg(fp, prompt="a fruit", true_cfg_scale=0.0)
    assert fp.last_kwargs["negative_prompt"] is None


def test_assert_cfg_ok_raises_on_illegal_combo():
    """assert_cfg_ok raises when true_cfg_scale > 1 and negative_prompt is None."""
    with pytest.raises(SystemExit):
        assert_cfg_ok(true_cfg_scale=4.0, negative_prompt=None)


def test_assert_cfg_ok_passes_on_legal_combos():
    """assert_cfg_ok does not raise when the combination is legal."""
    assert_cfg_ok(true_cfg_scale=4.0, negative_prompt=" ")  # CFG on, negative set
    assert_cfg_ok(true_cfg_scale=1.0, negative_prompt=None)  # CFG off, negative None
    assert_cfg_ok(true_cfg_scale=0.5, negative_prompt=None)  # CFG off, negative None


def test_wrapper_forwards_arbitrary_kwargs():
    """Wrapper forwards height, width, seed, generator, etc. unchanged."""
    fp = _FakePipe()
    out = pipe_with_cfg(fp, prompt="test", true_cfg_scale=4.0,
                       height=512, width=512, num_inference_steps=25,
                       generator="FAKE_GENERATOR", extra_key="extra_value")
    assert fp.last_kwargs["height"] == 512
    assert fp.last_kwargs["width"] == 512
    assert fp.last_kwargs["num_inference_steps"] == 25
    assert fp.last_kwargs["generator"] == "FAKE_GENERATOR"
    assert fp.last_kwargs["extra_key"] == "extra_value"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
