"""Unit tests for the ARM-C (model-tier) OpenAI-compatible backend — mocked, no live calls.

Covers: the shared model_guard/image_encode helpers (used by BOTH backends now — the
fork-prevention refactor), the OpenAI-compatible backend's served-model guard (same
strictness as Anthropic's), missing-api-key-env refusal, usage-token mapping, crop_fn
interop via the provider-agnostic ``encode_fn`` parameter, and the proactive RPM throttle.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from aiqs.config import CropConfig
from aiqs.vlm.backend_openai_compatible import OpenAICompatibleVLMBackend
from aiqs.vlm.crop_fn import make_crop_fn
from aiqs.vlm.image_encode import encode_png_b64, load_and_encode, resize_short_edge
from aiqs.vlm.model_guard import assert_model_matches, model_matches
from aiqs.vlm.state import VLMState

MODEL = "gemini-3.1-flash-lite-preview"


# --------------------------------------------------------------------------- #
# Shared helpers (model_guard / image_encode) — the fork-prevention refactor
# --------------------------------------------------------------------------- #

def test_model_matches_exact_and_dated_and_mismatch():
    assert model_matches("claude-haiku-4-5", "claude-haiku-4-5") is True
    assert model_matches("claude-haiku-4-5", "claude-haiku-4-5-20251001") is True
    assert model_matches("claude-haiku-4-5", "claude-3-5-sonnet-legacy") is False
    assert model_matches("claude-haiku-4-5", None) is False


def test_assert_model_matches_raises_with_both_names_in_message():
    with pytest.raises(RuntimeError, match="expected 'x'.*served 'y'"):
        assert_model_matches("x", "y")
    assert_model_matches("x", "x")               # no raise on match


def test_resize_short_edge_never_upscales():
    img = Image.new("RGB", (100, 200))
    same = resize_short_edge(img, max_edge=512)   # short edge 100 < 512 -> unchanged
    assert same.size == (100, 200)
    smaller = resize_short_edge(img, max_edge=50)  # short edge 100 -> scale to 50
    assert smaller.size == (50, 100)


def test_encode_png_b64_roundtrips_to_a_valid_image():
    img = Image.new("RGB", (64, 64), color=(1, 2, 3))
    b64 = encode_png_b64(img, max_edge=512)
    import base64

    decoded = Image.open(io.BytesIO(base64.standard_b64decode(b64)))
    assert decoded.size == (64, 64) and decoded.getpixel((0, 0)) == (1, 2, 3)


def test_load_and_encode_reads_a_file(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (32, 32)).save(p)
    b64 = load_and_encode(str(p), max_edge=512)
    assert isinstance(b64, str) and len(b64) > 0


# --------------------------------------------------------------------------- #
# OpenAICompatibleVLMBackend — served-model guard, api-key-env, usage, crop interop
# --------------------------------------------------------------------------- #

class _FakeUsage:
    def __init__(self, pt, ct):
        self.prompt_tokens, self.completion_tokens = pt, ct


class _FakeResp:
    def __init__(self, model, content="{}", usage=None):
        self.model = model
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = usage


def _state(tmp_path, with_map=False):
    """A state pointing at a REAL image file (tmp_path) — the encoder opens the file."""
    img_path = tmp_path / "img.png"
    if not img_path.exists():
        Image.new("RGB", (64, 64)).save(img_path)
    if not with_map:
        return VLMState(image_path=str(img_path), detector_score=0.5, detector_p=0.5)
    m = np.zeros((16, 16))
    m[4:6, 10:12] = 5.0
    map_path = tmp_path / "map.npy"
    np.save(map_path, m)
    return VLMState(image_path=str(img_path), detector_score=0.5, detector_p=0.5,
                    anomaly_map_path=str(map_path))


def test_missing_api_key_env_refuses_before_any_network_call(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_KEY", raising=False)
    backend = OpenAICompatibleVLMBackend(model=MODEL, base_url="https://example.test/v1",
                                         api_key_env="SOME_UNSET_KEY")
    with pytest.raises(RuntimeError, match="SOME_UNSET_KEY.*not set"):
        backend._client_lazy()


def test_served_model_exact_match_ok(monkeypatch, tmp_path):
    backend = OpenAICompatibleVLMBackend(model=MODEL, base_url="https://x", api_key_env="K")

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _FakeResp(MODEL, content='{"verdict":"clean","confidence":0.9,'
                                                    '"reasoning":"r"}',
                                     usage=_FakeUsage(639, 101))

    monkeypatch.setattr(backend, "_client_lazy", lambda: _FakeClient())
    verdict = backend(_state(tmp_path))
    assert verdict.verdict == "clean"


def test_served_model_dated_suffix_accepted(monkeypatch, tmp_path):
    backend = OpenAICompatibleVLMBackend(model=MODEL, base_url="https://x", api_key_env="K")

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _FakeResp(MODEL + "-001")

    monkeypatch.setattr(backend, "_client_lazy", lambda: _FakeClient())
    state = _state(tmp_path)
    out = backend._invoke("sys", [{"type": "text", "text": "q"}], state)
    assert out == "{}"


def test_served_model_mismatch_stops(monkeypatch, tmp_path):
    backend = OpenAICompatibleVLMBackend(model=MODEL, base_url="https://x", api_key_env="K")

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _FakeResp("some-other-model")

    monkeypatch.setattr(backend, "_client_lazy", lambda: _FakeClient())
    with pytest.raises(RuntimeError, match="SERVED MODEL MISMATCH"):
        backend._invoke("sys", [{"type": "text", "text": "q"}], _state(tmp_path))


def test_usage_tokens_captured_on_state(monkeypatch, tmp_path):
    backend = OpenAICompatibleVLMBackend(model=MODEL, base_url="https://x", api_key_env="K")

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _FakeResp(MODEL, usage=_FakeUsage(809, 107))

    monkeypatch.setattr(backend, "_client_lazy", lambda: _FakeClient())
    state = _state(tmp_path)
    backend._invoke("sys", [{"type": "text", "text": "q"}], state)
    assert state.tokens_in == 809 and state.tokens_out == 107


def test_build_content_is_openai_shaped_and_uses_crop_fn(tmp_path):
    crop_fn = make_crop_fn(CropConfig(), encode_fn=OpenAICompatibleVLMBackend(
        model=MODEL, base_url="https://x", api_key_env="K")._encode_image_block)
    backend = OpenAICompatibleVLMBackend(model=MODEL, base_url="https://x", api_key_env="K",
                                        crop_fn=crop_fn)
    content = backend._build_content(_state(tmp_path, with_map=True))
    image_blocks = [c for c in content if c["type"] == "image_url"]
    assert len(image_blocks) == 2                    # full image + crop
    assert all("image_url" in b and b["image_url"]["url"].startswith("data:image/png;base64,")
              for b in image_blocks)
    assert content[-1]["type"] == "text"


def test_throttle_sleeps_to_respect_rpm_limit(monkeypatch):
    backend = OpenAICompatibleVLMBackend(model=MODEL, base_url="https://x", api_key_env="K",
                                        rpm_limit=30)   # 1 call / 2s
    # _last_call_t is pre-set to 100.0; "now" (first time.time() read inside _throttle,
    # used for the wait calc) is 100.5 -> 0.5s elapsed -> wait = 2.0 - 0.5 = 1.5. The
    # second time.time() call (updating _last_call_t afterwards) reuses the same value.
    monkeypatch.setattr("time.time", lambda: 100.5)
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    backend._last_call_t = 100.0
    backend._throttle()
    assert slept and slept[0] == pytest.approx(1.5, abs=1e-6)   # 2.0 - 0.5


def test_throttle_no_sleep_when_interval_already_elapsed(monkeypatch):
    backend = OpenAICompatibleVLMBackend(model=MODEL, base_url="https://x", api_key_env="K",
                                        rpm_limit=60)    # 1 call / 1s
    monkeypatch.setattr("time.time", lambda: 105.0)
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    backend._last_call_t = 100.0                          # 5s elapsed >= 1s needed
    backend._throttle()
    assert slept == []


def test_throttle_noop_when_no_rpm_limit(monkeypatch):
    backend = OpenAICompatibleVLMBackend(model=MODEL, base_url="https://x", api_key_env="K")
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    backend._throttle()
    assert slept == []
