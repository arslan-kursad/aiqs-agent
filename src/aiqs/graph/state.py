"""Typed LangGraph state for one item's adjudication.

Verified empirically against the installed langgraph (1.2.8) before this was written:
a Pydantic ``BaseModel`` works directly as ``StateGraph``'s state schema, nodes return
plain ``dict`` partial updates that get merged onto the persisted full state, and
``graph.invoke()``/``get_state()`` always hand back plain dicts — so we type the
schema with Pydantic for validation but treat state as a dict everywhere else (never
construct an ``AdjudicationState`` instance and pass it to ``invoke``; that round-trips
through the checkpointer's pickle fallback instead of the plain-dict fast path).
"""

from __future__ import annotations

from pydantic import BaseModel


class AdjudicationState(BaseModel):
    model_config = {"extra": "forbid"}

    # ---- set at ingest (the request, with artifact defaults already resolved) ----
    item_id: str
    detector_score: float
    target_prevalence: float
    cost_false_accept: float
    cost_false_reject: float
    cost_escalation: float
    image_path: str | None = None
    anomaly_map_path: str | None = None
    lam: float = 0.0  # VLM provisional-p shrinkage (see aiqs.vlm.abstain.confidence_to_p)

    # ---- calibrate ----
    detector_p: float | None = None
    pi_source: float | None = None
    indifference_points: dict[str, float] | None = None

    # ---- cost_policy (tier-1: PASS/FAIL/ESCALATE on the detector alone) ----
    tier1_decision: str | None = None
    expected_costs: dict[str, float] | None = None

    # ---- vlm_second_look (backend call only) ----
    vlm_verdict: str | None = None
    vlm_confidence: float | None = None
    vlm_reasoning: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None

    # ---- vlm_abstain_rule (confidence_to_p + the cost-matrix argmin on p_vlm) ----
    p_vlm: float | None = None
    vlm_decision: str | None = None  # pass|fail|escalate ("escalate" == abstain -> human)

    # ---- human_interrupt ----
    human_decision: str | None = None  # pass|fail
    human_reviewer: str | None = None
    human_note: str | None = None

    # ---- finalize ----
    final_decision: str | None = None  # pass|fail
    resolved_by: str | None = None     # policy|vlm|human
