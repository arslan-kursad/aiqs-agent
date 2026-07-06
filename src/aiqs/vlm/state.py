"""The node-shaped seam for the VLM second-look.

``VLMState`` is a plain dataclass that flows through ``adjudicate(state) -> state``.
In Phase 2B this is wrapped as a LangGraph node with zero re-scaffolding: the state
is already node-shaped and the entry is already a single ``state -> state`` function.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiqs.eval.decision import Decision


@dataclass
class VLMState:
    """One ESCALATE-bucket item as it passes through the VLM adjudication seam.

    Fields are filled in stages: the detector half is known up front; the VLM half
    (verdict/confidence/reasoning), the provisional probability ``p_vlm``, and the
    ``final_decision`` are filled by ``adjudicate``. ``label`` is ground truth, present
    only for EVALUATION — in production it is ``None`` and never consulted.
    """

    image_path: str
    detector_score: float          # raw anomaly score (Phase-0)
    detector_p: float              # Phase-1 calibrated P(defective) (cross Venn-Abers)
    label: int | None = None       # ground truth (eval only; 0=good, 1=defective)
    anomaly_map_path: str | None = None   # 2B crop input; None => full-image-only

    # Filled by the VLM second-look.
    vlm_verdict: str | None = None        # "defect" | "clean" | "unsure"
    vlm_conf: float | None = None         # self-reported confidence in [0, 1]
    vlm_reasoning: str | None = None
    p_vlm: float | None = None            # PROVISIONAL/UNCALIBRATED P(defective)

    # Filled by the abstain rule.
    final_decision: Decision | None = None  # PASS/FAIL = VLM auto-decide; ESCALATE = human
    abstained: bool | None = None            # True => routed to a human reviewer
    trace_id: str | None = None              # Langfuse trace id, when instrumented
    tokens_in: int | None = None             # API usage (None for mock) — Stage-3 cost line
    tokens_out: int | None = None
