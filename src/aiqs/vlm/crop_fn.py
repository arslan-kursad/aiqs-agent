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


def make_crop_fn(crop_cfg):
    """Return ``crop_fn(state, max_edge) -> (image_block | None, crop_note)``."""

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
        return _encode_image_block(result.crop, max_edge), CROP_NOTE

    return crop_fn


def _encode_image_block(image, max_edge: int) -> dict:
    """Encode a PIL image as an Anthropic base64 image block (resize to ``max_edge`` short
    side — same convention as the backend's full-image encoder)."""
    import base64
    import io

    w, h = image.size
    scale = max_edge / min(w, h)
    if scale < 1.0:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64}}
