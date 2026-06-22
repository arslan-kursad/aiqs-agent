"""Phase-1 placeholder: decision-level evaluation.

NOT WIRED UP IN PHASE 0. This module defines the contract the adjudication layer
will fill in. It documents — in code — how the Phase-0 outputs (the per-image
scores persisted by `results.py`) feed the business cost function that is the
project's true north (we optimize cost + false-reject rate, NOT detection AUROC).

Phase 1 will implement:
  * calibration of raw anomaly scores -> probabilities,
  * a PASS / FAIL / ESCALATE policy over a cost matrix,
  * metrics below, appended to the same run directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ESCALATE = "escalate"  # route to a human


@dataclass
class CostMatrix:
    """Business cost of each (decision, true_state) outcome.

    Defaults are placeholders. Real values come from the operation: a missed
    defect (false accept) usually dwarfs a false reject (overkill), and an
    escalation costs a fixed human-review amount regardless of true state.
    """

    false_accept: float = 100.0   # passed an actually-defective part
    false_reject: float = 10.0    # failed an actually-good part (overkill)
    escalation: float = 1.0       # human review, either way
    true_pass: float = 0.0
    true_fail: float = 0.0


@dataclass
class DecisionMetrics:
    n: int
    false_reject_rate: float       # good parts wrongly failed / all good parts
    false_accept_rate: float       # bad parts wrongly passed / all bad parts
    escalation_rate: float         # escalated / all
    total_cost: float
    cost_per_item: float


def evaluate_decisions(*args, **kwargs):  # pragma: no cover - Phase 1
    raise NotImplementedError(
        "Decision-level evaluation lands in Phase 1. It will consume the "
        "image_scores.csv persisted by aiqs.eval.results in Phase 0."
    )
