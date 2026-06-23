"""Prompt + image construction for the second-look.

Kept separate from the backend so the wording is reviewable in one place and the
full-image-now / crop-later seam is explicit. The crop on the anomaly-map peak is a
2B addition (``crop_anomaly_region`` is a documented stub here); 2A sends the full
image only — and NOTE that full-image adjudication is a weaker, different task than
adjudicating the detector's flagged region, so the headline mechanism comes only from
the later crop-equipped hard-category run, never from the full-image smoke.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a meticulous industrial visual-quality inspector performing a SECOND look "
    "at a part that an automated anomaly detector flagged as borderline (it could not "
    "confidently pass or fail it). Your job is to decide whether the part has a REAL "
    "manufacturing defect, or whether the flagged appearance is an ARTIFACT (a "
    "reflection/specular highlight, dust, lighting, focus, or background), or whether "
    "you are genuinely UNSURE. Bring information the detector cannot: semantic "
    "understanding of what is an expected surface feature versus an actual defect. "
    "Do not assume the detector is right. Be calibrated: reserve high confidence for "
    "clear cases and say 'unsure' when the evidence is ambiguous."
)

QUESTION = (
    "\n\nRespond with a single JSON object and nothing else:\n"
    '{"verdict": "defect" | "clean" | "unsure", '
    '"confidence": <number 0..1>, '
    '"reasoning": "<one or two sentences citing the visual evidence>"}\n'
    "- verdict=defect: a real manufacturing defect is present.\n"
    "- verdict=clean: the part is acceptable; any flagged region is an artifact.\n"
    "- verdict=unsure: the evidence is genuinely ambiguous.\n"
    "confidence is how sure you are of the verdict (not of defectiveness)."
)


def crop_anomaly_region(state, max_edge: int):  # pragma: no cover - 2B stub
    """STUB (Phase 2B): crop a high-res region around the anomaly-map peak.

    Needs per-pixel anomaly maps, which the current Colab run did not persist (only
    image-level scores). Map export piggybacks on the future hard-category Colab run.
    Returns ``(image_block | None, crop_note)``; until implemented, signals full-image-only.
    """
    raise NotImplementedError(
        "crop_anomaly_region is a Phase-2B addition; 2A is full-image-only. "
        "Persist anomaly maps from Colab for the ESCALATE bucket to enable it.")
