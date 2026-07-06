"""Build the VLM ``crop_fn`` from the Phase-2B crop instrument (``aiqs.crop``).

This is the wire between the detector's anomaly-map export and the VLM second-look: it
plugs straight into the EXISTING backend seam — ``AnthropicVLMBackend(crop_fn=...)`` calls
``crop_fn(state, max_edge) -> (image_block | None, crop_note)`` whenever
``state.anomaly_map_path`` is set (designed in Phase 2A for exactly this, no fork).

The ``cfg.crop.enabled`` switch (config) selects the two experiment arms:
  * **ARM-A (full-image only):** crop disabled — no ``crop_fn`` passed; the backend sends
    the full image alone (Phase-2A behaviour).
  * **ARM-B (full-image + crop):** crop enabled — this ``crop_fn`` appends a high-res crop.

A DIFFUSE map returns ``(None, "")`` so that item falls back to full-image-only, byte-for-byte
identical to ARM-A — keeping the full-vs-crop comparison clean on the diffuse subset.

Pure PIL/numpy (no API, no anomalib): testable locally with a fake ``.npy`` map + image.
"""

from __future__ import annotations

CROP_NOTE = (
    "\n\nA SECOND image is provided: a high-resolution crop centred on the exact region the "
    "anomaly detector flagged. Inspect it closely to judge whether that region is a real "
    "defect or an artifact; the first image gives the full-part context."
)


def make_crop_fn(crop_cfg, encode_fn=None):
    """Return ``crop_fn(state, max_edge) -> (image_block | None, crop_note)``.

    ``encode_fn(image, max_edge) -> dict`` wraps the cropped PIL image into a
    provider-shaped content block; defaults to the Anthropic image-block shape (the
    2A/2B design). Pass a different ``encode_fn`` to plug the SAME crop instrument into a
    non-Anthropic backend (e.g. ``OpenAICompatibleVLMBackend``'s ``image_url`` shape) —
    the crop logic itself (``aiqs.crop.compute_crop``) never changes per provider.
    """
    encode = encode_fn or _encode_image_block

    def crop_fn(state, max_edge: int):
        import numpy as np
        from PIL import Image

        from aiqs.crop import compute_crop

        if not getattr(state, "anomaly_map_path", None):
            return None, ""
        amap = np.load(state.anomaly_map_path)
        image = Image.open(state.image_path).convert("RGB")
        result = compute_crop(amap, image, crop_cfg)
        if result.crop is None:                 # diffuse -> full-image-only for this item
            return None, ""
        return encode(result.crop, max_edge), CROP_NOTE

    return crop_fn


def _encode_image_block(image, max_edge: int) -> dict:
    """Encode a PIL image as an Anthropic base64 image block (resize to ``max_edge`` short
    side — same shared encoder the backends use)."""
    from aiqs.vlm.image_encode import encode_png_b64

    b64 = encode_png_b64(image, max_edge)
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64}}
