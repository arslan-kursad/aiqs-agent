"""Unit tests for the Phase-2B anomaly-map crop instrument (aiqs.crop + vlm.crop_fn).

Pure numpy/PIL — no anomalib/torch/anthropic. This is the locally-verified core of
Stage 1: peak -> crop, and the DIFFUSE fallback (a measured outcome, not a crash).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aiqs.config import CropConfig
from aiqs.crop import compute_crop
from aiqs.vlm.crop_fn import CROP_NOTE, make_crop_fn
from aiqs.vlm.state import VLMState

IMG = 256


def _image():
    return Image.fromarray(np.zeros((IMG, IMG, 3), dtype=np.uint8))


def _peaked_map(h=32, w=32, r=5, c=25, val=10.0):
    """A flat-zero map with a small bright blob centred at (r, c) in map coords
    (indices clamped so a corner peak doesn't silently produce an empty map)."""
    m = np.zeros((h, w), dtype=np.float64)
    m[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = val
    return m


# --------------------------------------------------------------------------- #
# compute_crop — peak case
# --------------------------------------------------------------------------- #

def test_clear_peak_crops_around_blob():
    res = compute_crop(_peaked_map(), _image(), CropConfig())
    assert res.diffuse is False and res.crop is not None and res.reason == "peak"
    left, top, right, bottom = res.bbox
    # blob at map (row 5, col 25) -> image (x≈200, y≈40); the bbox must contain it.
    assert left <= 200 <= right and top <= 40 <= bottom
    px, py = res.peak_xy
    assert abs(px - 200) <= 8 and abs(py - 40) <= 8


def test_crop_bbox_within_image_bounds_for_border_peak():
    res = compute_crop(_peaked_map(r=0, c=0), _image(), CropConfig())
    left, top, right, bottom = res.bbox
    assert 0 <= left < right <= IMG and 0 <= top < bottom <= IMG


def test_min_size_is_enforced():
    cfg = CropConfig(min_size=96, padding=0.0)
    res = compute_crop(_peaked_map(r=16, c=16, val=20.0), _image(), cfg)
    left, top, right, bottom = res.bbox
    # at least min_size on each side (a 1-blob peak would otherwise be tiny), clamped to IMG.
    assert (right - left) >= 96 and (bottom - top) >= 96


# --------------------------------------------------------------------------- #
# compute_crop — diffuse fallbacks (first-class outcomes, crop is None)
# --------------------------------------------------------------------------- #

def test_flat_map_is_diffuse():
    res = compute_crop(np.full((32, 32), 3.0), _image(), CropConfig())
    assert res.diffuse is True and res.crop is None and res.reason.startswith("flat-map")


def test_empty_or_nonfinite_map_is_diffuse():
    res = compute_crop(np.full((8, 8), np.nan), _image(), CropConfig())
    assert res.diffuse is True and res.crop is None


def test_broad_plateau_triggers_area_diffuse():
    # A sharp max (passes the flat check, peak/mean high) sitting on a broad mid-level
    # plateau that covers most of the frame -> the diffuse-region area guard fires.
    m = np.zeros((32, 32), dtype=np.float64)
    m[:20, :] = 70.0      # 62.5% of the frame above the peak_fraction line
    m[5, 5] = 100.0       # a sharp peak on top
    res = compute_crop(m, _image(), CropConfig())
    assert res.diffuse is True and res.reason.startswith("diffuse-region")


# --------------------------------------------------------------------------- #
# vlm.crop_fn — the backend seam (fake map + image, no API)
# --------------------------------------------------------------------------- #

def _state(tmp_path, amap):
    img_path = tmp_path / "img.png"
    _image().save(img_path)
    map_path = tmp_path / "map.npy"
    np.save(map_path, amap)
    return VLMState(image_path=str(img_path), detector_score=0.5, detector_p=0.5,
                    anomaly_map_path=str(map_path))


def test_crop_fn_returns_image_block_for_peak(tmp_path):
    crop_fn = make_crop_fn(CropConfig())
    block, note = crop_fn(_state(tmp_path, _peaked_map()), max_edge=512)
    assert block is not None and block["type"] == "image"
    assert block["source"]["type"] == "base64" and block["source"]["data"]
    assert note == CROP_NOTE


def test_crop_fn_falls_back_to_full_image_on_diffuse(tmp_path):
    crop_fn = make_crop_fn(CropConfig())
    block, note = crop_fn(_state(tmp_path, np.full((32, 32), 3.0)), max_edge=512)
    assert block is None and note == ""        # identical to the full-image-only arm


def test_crop_fn_no_map_path_is_noop():
    crop_fn = make_crop_fn(CropConfig())
    state = VLMState(image_path="x.png", detector_score=0.5, detector_p=0.5)
    assert crop_fn(state, 512) == (None, "")
