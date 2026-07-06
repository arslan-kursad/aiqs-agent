"""OpenAI-compatible VLM backend — the ARM-C (model-tier) lever.

Same call signature as ``AnthropicVLMBackend`` (``backend(state) -> VLMVerdict``), same
shared contract (``VLMVerdict`` / ``parse_verdict``), same crop instrument
(``vlm.crop_fn.make_crop_fn``, just with an OpenAI-shaped ``encode_fn``), same served-model
stop guard (``vlm.model_guard``). Provider-agnostic by construction: ``base_url`` / ``model``
/ ``api_key_env`` are all caller-supplied, so swapping a free-tier roster entry (they rotate
monthly) is a CONFIG change, never a code change. Verified compatible surfaces at design
time: Google AI Studio's OpenAI-compatibility endpoint and OpenRouter — validate the EXACT
served-model string and content-block acceptance with ``--smoke`` before a real run; do not
assume the API shape, per the project's "measure, don't assume" rule.

The API key is never accepted as a literal — only the NAME of the environment variable
holding it (``api_key_env``), so a YAML/CLI config can never carry a secret.
"""

from __future__ import annotations

import time

from aiqs.vlm.backend import MODEL as SONNET_MODEL
from aiqs.vlm.backend import VLMVerdict, parse_verdict
from aiqs.vlm.model_guard import assert_model_matches
from aiqs.vlm.state import VLMState


class OpenAICompatibleVLMBackend:
    """Real vision call against any OpenAI-Chat-Completions-compatible endpoint.

    ``rpm_limit`` is a PROACTIVE client-side pace (sleep before a call if needed) so a
    free-tier daily quota is not burned on 429s — distinct from ``max_retries``, which
    lets the SDK's own exponential backoff ride out a transient error AFTER it happens.
    """

    def __init__(self, *, model: str, base_url: str, api_key_env: str,
                 max_tokens: int = 512, temperature: float = 0.0, max_edge: int = 512,
                 crop_fn=None, max_retries: int = 8, rpm_limit: float | None = None):
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_edge = max_edge
        self.crop_fn = crop_fn
        self.max_retries = max_retries
        self.rpm_limit = rpm_limit
        self._client = None
        self._last_call_t: float | None = None

    def _client_lazy(self):
        if self._client is None:
            import os

            from openai import OpenAI

            key = os.getenv(self.api_key_env)
            if not key:
                raise RuntimeError(
                    f"Environment variable {self.api_key_env!r} is not set — "
                    "OpenAICompatibleVLMBackend never accepts a literal API key.")
            self._client = OpenAI(base_url=self.base_url, api_key=key,
                                  max_retries=self.max_retries)
        return self._client

    def _throttle(self) -> None:
        """Proactive RPM pace — sleep only as much as needed, never negative."""
        if self.rpm_limit is None or self.rpm_limit <= 0:
            return
        min_interval = 60.0 / self.rpm_limit
        if self._last_call_t is not None:
            wait = min_interval - (time.time() - self._last_call_t)
            if wait > 0:
                time.sleep(wait)
        self._last_call_t = time.time()

    @staticmethod
    def _encode_image_block(image_or_path, max_edge: int) -> dict:
        """OpenAI-shaped image block. Accepts a PIL image (crop instrument) or a file path
        (full-image encode) — mirrors the dual use in AnthropicVLMBackend/crop_fn. A
        staticmethod (no ``self`` needed) so it can also be passed as ``vlm.crop_fn.
        make_crop_fn``'s ``encode_fn`` BEFORE any backend instance exists (see vlm_crop.py's
        provider-aware crop_fn construction)."""
        from aiqs.vlm.image_encode import encode_png_b64, load_and_encode

        if isinstance(image_or_path, str):
            b64 = load_and_encode(image_or_path, max_edge)
        else:
            b64 = encode_png_b64(image_or_path, max_edge)
        return {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}}

    def _build_content(self, state: VLMState) -> list[dict]:
        from aiqs.vlm.prompt import QUESTION

        content: list[dict] = [self._encode_image_block(state.image_path, self.max_edge)]
        crop_note = ""
        if state.anomaly_map_path is not None and self.crop_fn is not None:
            crop_block, crop_note = self.crop_fn(state, self.max_edge)
            if crop_block is not None:
                content.append(crop_block)
        content.append({"type": "text", "text": QUESTION + crop_note})
        return content

    def __call__(self, state: VLMState) -> VLMVerdict:
        from aiqs.vlm.prompt import SYSTEM_PROMPT

        content = self._build_content(state)
        raw = self._invoke(SYSTEM_PROMPT, content, state)
        return parse_verdict(raw)

    def _invoke(self, system: str, content: list[dict], state: VLMState) -> str:
        client = self._client_lazy()
        self._throttle()
        resp = client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
        )
        # Pre-registered guard (the silent-3.5-downgrade lesson, provider-agnostic): the
        # SERVED model must match on EVERY call. STOP LOUD, never silently continue.
        assert_model_matches(self.model, getattr(resp, "model", None))
        u = getattr(resp, "usage", None)
        if u is not None:
            state.tokens_in = getattr(u, "prompt_tokens", None)
            state.tokens_out = getattr(u, "completion_tokens", None)
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        msg = getattr(choice, "message", None)
        return getattr(msg, "content", None) or ""


# The locked headline model never uses this backend — kept here only as a clarity import
# for callers that want to assert "this is definitely not the canonical run".
CANONICAL_MODEL = SONNET_MODEL
