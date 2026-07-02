"""VLM backends + the typed verdict contract.

``VLMVerdict`` is the strict, Pydantic-validated output the rest of the pipeline relies
on. ``parse_verdict`` extracts it from raw model text and FAILS LOUDLY on malformed
output (no silent defaulting — a swallowed parse error would quietly corrupt the eval).

Two backends share one call signature ``backend(state) -> VLMVerdict``:
  * ``AnthropicVLMBackend`` — real claude-sonnet-4-6 vision call, with optional Langfuse
    instrumentation. Imports of ``anthropic``/``PIL``/``langfuse`` are LAZY so the
    deterministic backbone and the mocked unit tests need none of them installed.
  * ``MockVLMBackend`` — scripted/deterministic or noisy verdicts for tests and for the
    wiring smoke run (no API key, no image encoding, no cost).
"""

from __future__ import annotations

import json
import re
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from aiqs.vlm.state import VLMState

MODEL = "claude-sonnet-4-6"
Verdict = Literal["defect", "clean", "unsure"]


class VLMVerdict(BaseModel):
    """Strict contract for a single second-look. Extra keys are rejected."""

    model_config = {"extra": "forbid"}

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class VLMParseError(ValueError):
    """Raised when a model response cannot be parsed into a valid VLMVerdict."""


def parse_verdict(raw: str) -> VLMVerdict:
    """Parse model text into a VLMVerdict. Loud on anything malformed.

    Accepts a bare JSON object or one fenced in ```json ... ```. Any missing/extra
    field, out-of-range confidence, or unknown verdict raises VLMParseError.
    """
    if raw is None:
        raise VLMParseError("empty model response")
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise VLMParseError(f"response is not valid JSON: {e}\n---\n{raw!r}") from e
    try:
        return VLMVerdict.model_validate(data)
    except ValidationError as e:
        raise VLMParseError(f"response failed schema validation: {e}\n---\n{raw!r}") from e


# Backend call signature.
VLMBackend = Callable[[VLMState], VLMVerdict]


class MockVLMBackend:
    """Deterministic/noisy VLM stand-in. Never calls an API or encodes an image.

    ``verdict_fn(state, rng) -> (verdict, confidence, reasoning)`` lets a test script
    arbitrary behaviour (oracle, adversarial, correlated-with-detector, noisy). Default:
    a confident oracle derived from ``state.label`` (handy for wiring checks).
    """

    def __init__(self, verdict_fn=None, seed: int = 0):
        import numpy as np

        self._rng = np.random.default_rng(seed)
        self._fn = verdict_fn or self._oracle

    @staticmethod
    def _oracle(state: VLMState, rng):
        if state.label is None:
            return "unsure", 0.0, "no label -> abstain (mock oracle)"
        return ("defect" if state.label == 1 else "clean"), 0.9, "mock oracle"

    def __call__(self, state: VLMState) -> VLMVerdict:
        verdict, conf, reasoning = self._fn(state, self._rng)
        return VLMVerdict(verdict=verdict, confidence=conf, reasoning=reasoning)


class AnthropicVLMBackend:
    """Real claude-sonnet-4-6 vision second-look. Lazy heavy imports.

    Full image (~512px short edge) is always sent. The crop on the anomaly-map peak is a
    2B addition: when ``state.anomaly_map_path`` is set, ``crop_fn`` (injected) may add a
    high-res region; absent that, full-image-only. Langfuse is wired when its env keys
    are present, else a no-op (so the smoke run does not require a Langfuse project).
    """

    def __init__(self, *, model: str = MODEL, max_tokens: int = 512,
                 temperature: float = 0.0, max_edge: int = 512,
                 crop_fn=None, api_key: str | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_edge = max_edge
        self.crop_fn = crop_fn
        self._api_key = api_key  # None => SDK reads ANTHROPIC_API_KEY from env
        self._client = None

    # -- prompt construction (text part is import-light; image part needs PIL) ------ #
    def _client_lazy(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _encode_image(self, path: str) -> dict:
        import base64
        import io

        from PIL import Image

        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = self.max_edge / min(w, h)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        return {"type": "image", "source": {"type": "base64",
                                            "media_type": "image/png", "data": b64}}

    def _build_content(self, state: VLMState) -> list[dict]:
        from aiqs.vlm.prompt import QUESTION

        content: list[dict] = [self._encode_image(state.image_path)]
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
        observed = self._with_langfuse(client, state)
        msg = observed(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],  # cache the stable prefix
            messages=[{"role": "user", "content": content}],
        )
        # Pre-registered guard (the silent-3.5-downgrade lesson): the SERVED model must be
        # the expected one, on EVERY call. STOP LOUD, never silently continue on a substitute.
        served = getattr(msg, "model", None)
        if served != self.model:
            raise RuntimeError(
                f"SERVED MODEL MISMATCH: expected {self.model!r}, API served {served!r}. "
                "Stopping — results on a substitute model are not valid evidence.")
        u = getattr(msg, "usage", None)
        if u is not None:  # Stage-3 token-cost line (incl. the crop's 2nd-image increment)
            state.tokens_in = getattr(u, "input_tokens", None)
            state.tokens_out = getattr(u, "output_tokens", None)
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    def _with_langfuse(self, client, state: VLMState):
        """Return a callable that creates a message, instrumented via Langfuse if keys
        are configured; otherwise a plain pass-through. Never hard-fails on a missing
        Langfuse install/config — observability is additive, not load-bearing."""
        create = client.messages.create
        try:
            import os

            if not os.getenv("LANGFUSE_PUBLIC_KEY"):
                return create
            from langfuse import Langfuse  # noqa: F401

            lf = Langfuse()

            def observed(**kwargs):
                with lf.start_as_current_generation(
                        name="vlm-second-look", model=self.model,
                        input={"image_path": state.image_path}) as gen:
                    msg = create(**kwargs)
                    u = getattr(msg, "usage", None)
                    if u is not None:
                        gen.update(usage_details={
                            "input": u.input_tokens, "output": u.output_tokens})
                    state.trace_id = getattr(gen, "trace_id", None)
                    return msg

            return observed
        except Exception:
            return create
