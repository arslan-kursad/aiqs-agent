"""Shared image resize + base64-PNG encoding for VLM backends and the crop instrument.

Extracted because the same "load an image, resize to max_edge short side, PNG b64-encode"
logic was duplicated across ``AnthropicVLMBackend``, ``vlm/crop_fn.py``, and (now) the
OpenAI-compatible backend — three near-identical copies is exactly the shape of a
silent-drift bug (fix a resize edge case in one, forget the others). Provider-specific
WRAPPING of the b64 string into a content block stays in each backend/caller (Anthropic
and OpenAI-compatible use different JSON shapes for an image block).
"""

from __future__ import annotations


def resize_short_edge(image, max_edge: int):
    """Resize a PIL image so its short edge is at most ``max_edge`` (never upscales)."""
    w, h = image.size
    scale = max_edge / min(w, h)
    if scale < 1.0:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return image


def encode_png_b64(image, max_edge: int) -> str:
    """Resize (short edge) then PNG-encode + base64. Takes a PIL Image directly (used by
    the crop instrument, which already holds a cropped in-memory image)."""
    import base64
    import io

    image = resize_short_edge(image, max_edge)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def load_and_encode(path: str, max_edge: int) -> str:
    """Open an image file (converted to RGB), resize, and PNG-b64-encode it."""
    from PIL import Image

    return encode_png_b64(Image.open(path).convert("RGB"), max_edge)
