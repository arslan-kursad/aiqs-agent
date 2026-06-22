"""Detection-level metrics (Phase 0).

Thin layer over anomalib's metric collections so the rest of the codebase speaks
canonical metric names. Phase 1 adds decision-level metrics in `decision.py`; this
module stays focused on detection quality.

Metrics reported:
  * image-level AUROC      (headline detection metric)
  * image-level F1Score    (operating-point sanity check)
  * pixel-level AUROC      (localisation)
  * pixel-level AUPRO      (localisation, region-overlap based)
  * pixel-level AUPIMO     (per-image overlap; reported only if it computes
                            cleanly — see evaluate.py fallback)
"""

from __future__ import annotations

from collections.abc import Mapping

# Metric collection names understood by anomalib's Engine.
IMAGE_METRICS: list[str] = ["AUROC", "F1Score"]
PIXEL_METRICS: list[str] = ["AUROC", "AUPRO"]
AUPIMO_METRIC = "AUPIMO"


def metric_config(use_aupimo: bool) -> tuple[list[str], list[str]]:
    """Return (image_metrics, pixel_metrics) name lists for the Engine."""
    pixel = list(PIXEL_METRICS)
    if use_aupimo:
        pixel.append(AUPIMO_METRIC)
    return list(IMAGE_METRICS), pixel


def canonicalize(results: list[Mapping[str, float]] | Mapping[str, float]) -> dict:
    """Flatten anomalib's test() output into {canonical_name: float}.

    anomalib returns e.g. {'image_AUROC': .., 'pixel_AUPRO': ..}; we lowercase to
    {'image_auroc': .., 'pixel_aupro': ..} and coerce tensors to plain floats.
    """
    if isinstance(results, Mapping):
        flat = dict(results)
    else:
        flat = {}
        for entry in results:
            flat.update(entry)

    out: dict[str, float] = {}
    for key, value in flat.items():
        out[key.lower()] = _to_float(value)
    return out


def _to_float(value) -> float:
    # torch tensors / numpy scalars -> python float
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (ValueError, TypeError):
            pass
    return float(value)
