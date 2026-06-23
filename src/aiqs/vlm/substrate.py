"""Substrate guard — the zeroth step, in code.

The VLM's overkill-reduction lever can only be MEASURED if good parts actually land in
the borderline (ESCALATE) band. On a strong detector at low prevalence they collapse to
PASS instead (see the screw finding: ESCALATE∩good = 1 at target-2%), leaving nothing to
measure. This guard refuses to spend API budget on a degenerate bucket and tells the
caller to pick a genuinely-hard category — NOT to weaken the detector to manufacture
substrate (a credibility trap).

Thresholds (load-bearing):
  HARD_MIN = 15  -> below this, STOP (no measurable substrate).
  WARN_MIN = 30  -> below this, proceed but loudly flag direction-only / underpowered.
"""

from __future__ import annotations

import numpy as np

from aiqs.eval.decision import Decision

HARD_MIN = 15
WARN_MIN = 30


class SubstrateError(RuntimeError):
    """Raised when the ESCALATE bucket lacks the good-part substrate to measure the lever."""


def bucket_composition(labels: np.ndarray, decisions: np.ndarray) -> dict:
    """Count the ESCALATE bucket, split by ground-truth label.

    ``escalate_good`` is the load-bearing number: the goods available to RESCUE from
    overkill. Uses identity comparison (a numpy object-array ``== Decision.X`` is
    unreliable for str-enums).
    """
    labels = np.asarray(labels, dtype=int)
    esc = np.array([d is Decision.ESCALATE for d in decisions])
    good = esc & (labels == 0)
    defective = esc & (labels == 1)
    return {
        "escalate_total": int(esc.sum()),
        "escalate_good": int(good.sum()),
        "escalate_defective": int(defective.sum()),
        "escalate_mask": esc,
    }


def substrate_guard(escalate_good: int, *, hard_min: int = HARD_MIN,
                    warn_min: int = WARN_MIN) -> list[str]:
    """Enforce the guard. Raise below ``hard_min``; return warnings below ``warn_min``."""
    if escalate_good < hard_min:
        raise SubstrateError(
            f"ESCALATE∩good = {escalate_good} < {hard_min}: the bucket lacks the "
            "good-part substrate to measure the overkill-reduction lever. Pick a "
            "genuinely HARD category / dataset where good parts get borderline scores "
            "(do NOT weaken the detector to manufacture substrate).")
    warnings: list[str] = []
    if escalate_good < warn_min:
        warnings.append(
            f"ESCALATE∩good = {escalate_good} < {warn_min}: UNDERPOWERED — treat results "
            "as DIRECTION-only with wide CIs (2A = mechanism + direction; 2B = magnitude).")
    return warnings
