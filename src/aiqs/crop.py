"""Anomaly-map peak -> high-resolution crop (Phase-2B crop instrument).

The CORE Stage-1 deliverable, and detector-free on purpose: given a per-image anomaly
heatmap (exported by the detector on the GPU host) and the original image, find the
focal region the detector flagged and return a high-res crop centred on it. This is
the second image the VLM gets in the Stage-3 full-vs-crop experiment — "look closer
HERE".

Pure numpy/PIL (no anomalib/torch), so it RUNS AND IS TESTED LOCALLY on the pinned
1.2 stack, and the crop parameters can be tuned offline without re-running the GPU
detector (the map is a cheap `.npy`).

DIFFUSE maps are a FIRST-CLASS outcome, not an error: when the heatmap has no clear
focal peak (a flat map, or a "peak" that covers most of the frame), we flag it and
return ``crop=None`` -> the VLM falls back to full-image-only for that item. Some AD2
defects are genuinely diffuse, so this distinction is itself a measured signal (it
feeds the Stage-3 perception-vs-semantic reasoning classification), not a crash guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # PIL is only needed at call time; keep import-light for pure-map tests.
    from PIL import Image as PILImage

_EPS = 1e-9


@dataclass
class CropResult:
    """Outcome of one crop attempt. ``crop`` is None exactly when ``diffuse`` is True."""

    crop: "PILImage.Image | None"
    bbox: tuple[int, int, int, int] | None     # (left, top, right, bottom) in ORIGINAL px
    peak_xy: tuple[int, int] | None             # argmax (x, y) in ORIGINAL px
    diffuse: bool
    reason: str                                 # "peak" | "flat-map(...)" | "diffuse-region(...)"


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1) inclusive-exclusive bounds of the True region, in MAP coords."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return int(r0), int(r1) + 1, int(c0), int(c1) + 1


def compute_crop(anomaly_map: np.ndarray, image: "PILImage.Image", cfg) -> CropResult:
    """Crop ``image`` to the anomaly-map peak region, or flag the map as diffuse.

    ``cfg`` is duck-typed (a ``config.CropConfig``): ``peak_fraction``, ``padding``,
    ``min_size``, ``diffuse_area_frac``, ``diffuse_peak_to_mean``. ``anomaly_map`` is a 2-D
    array of any scale; ``image`` is the full-resolution PIL image (the map is usually at the
    detector's working resolution and is scaled up to image pixels here).
    """
    a = np.asarray(anomaly_map, dtype=np.float64)
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).any():
        return CropResult(None, None, None, True, "flat-map(empty/non-finite)")
    a = np.nan_to_num(a, nan=float(np.nanmin(a)) if np.isfinite(a).any() else 0.0)

    h_map, w_map = a.shape
    w_img, h_img = image.size
    sx, sy = w_img / w_map, h_img / h_map

    # Diffuse check 1 — a flat map has a low peak-to-mean ratio (scale-invariant on raw).
    amin, amax, amean = float(a.min()), float(a.max()), float(a.mean())
    peak_to_mean = (amax + _EPS) / (abs(amean) + _EPS)
    if amax - amin < _EPS or peak_to_mean < cfg.diffuse_peak_to_mean:
        return CropResult(None, None, None, True, f"flat-map(peak/mean={peak_to_mean:.2f})")

    # Peak region = pixels at/above a fraction of the (normalized) max. A relative-to-max
    # threshold localises both sharp/sparse peaks and broad ones; a percentile threshold
    # silently fails on sparse peaks (the percentile collapses to ~0 -> selects everything).
    m = (a - amin) / (amax - amin + _EPS)
    mask = m >= cfg.peak_fraction
    if not mask.any():                 # degenerate -> fall back to the single argmax
        mask = m >= m.max()

    # Diffuse check 2 — the "peak" should be focal, not most of the frame.
    area_frac = float(mask.mean())
    if area_frac > cfg.diffuse_area_frac:
        return CropResult(None, None, None, True, f"diffuse-region(area={area_frac:.2f})")

    # Bbox of the peak region, map -> image coords.
    r0, r1, c0, c1 = _mask_bbox(mask)
    left, right = c0 * sx, c1 * sx
    top, bottom = r0 * sy, r1 * sy

    # Pad by a fraction of the bbox size, then enforce a minimum side, then clamp.
    bw, bh = right - left, bottom - top
    pad_x, pad_y = bw * cfg.padding, bh * cfg.padding
    left, right = left - pad_x, right + pad_x
    top, bottom = top - pad_y, bottom + pad_y
    left, right = _grow_to_min(left, right, cfg.min_size, w_img)
    top, bottom = _grow_to_min(top, bottom, cfg.min_size, h_img)
    box = (
        max(0, int(round(left))), max(0, int(round(top))),
        min(w_img, int(round(right))), min(h_img, int(round(bottom))),
    )

    pr, pc = np.unravel_index(int(np.argmax(m)), m.shape)
    peak_xy = (int(pc * sx), int(pr * sy))
    return CropResult(image.crop(box), box, peak_xy, False, "peak")


def _grow_to_min(lo: float, hi: float, min_size: int, bound: int) -> tuple[float, float]:
    """Symmetrically grow [lo, hi] to at least ``min_size`` (capped at ``bound``)."""
    size = hi - lo
    if size >= min_size:
        return lo, hi
    grow = (min(min_size, bound) - size) / 2.0
    return lo - grow, hi + grow
