"""Graph nodes. Each wraps EXISTING pure functions from the Phase-1/2A spine — the
graph orchestrates, it does not reimplement policy math or the VLM call.

Node functions receive a validated ``AdjudicationState`` INSTANCE (not a plain dict) —
verified empirically: with a Pydantic state schema, langgraph validates/hydrates the
merged state into a model instance before calling each node, and only the external
``invoke()``/``get_state()`` boundary hands back plain dicts. Nodes return partial
``dict`` updates, merged onto the persisted state (also verified empirically).

``vlm_second_look`` and ``vlm_abstain_rule`` are deliberately kept as two separate,
minimal nodes rather than one call into ``aiqs.vlm.adjudicate.adjudicate`` (which
composes the same two pieces): this gives per-node audit-trace granularity (each is
its own checkpointed state transition) and, combined with
``tests/test_graph_parity.py``, guarantees the composition never silently drifts from
the single-pass 2A/2B pipeline.

``human_interrupt`` calls ``interrupt()`` with NO side effects before it: verified
empirically that a node re-executes everything before its ``interrupt()`` call on
resume, so any side effect there would double-fire. The VLM call already happened
(once) in the prior ``vlm_second_look`` node, which is not on the replay path.
"""

from __future__ import annotations

from langgraph.types import interrupt

from aiqs.eval.decision import CostMatrix, Decision, decide_one
from aiqs.vlm.abstain import adjudicate_probability, confidence_to_p
from aiqs.vlm.backend import VLMBackend
from aiqs.vlm.state import VLMState

from aiqs.api.artifact import DecisionArtifact
from aiqs.graph.state import AdjudicationState


def _cost(state: AdjudicationState) -> CostMatrix:
    return CostMatrix(false_accept=state.cost_false_accept,
                      false_reject=state.cost_false_reject,
                      escalation=state.cost_escalation)


def ingest(state: AdjudicationState) -> dict:
    """Input hygiene only: no default-resolution here (the API layer already resolved
    cost/prevalence defaults from the artifact before invoking the graph)."""
    score = state.detector_score
    if score != score or score in (float("inf"), float("-inf")):  # NaN/inf guard
        raise ValueError(f"detector_score must be finite, got {score!r}")
    return {}


def make_calibrate_node(artifact: DecisionArtifact):
    def calibrate(state: AdjudicationState) -> dict:
        p = artifact.calibrate(state.detector_score, state.target_prevalence)
        cost = _cost(state)
        return {"detector_p": p, "pi_source": artifact.pi_source,
                "indifference_points": cost.indifference_points()}
    return calibrate


def cost_policy(state: AdjudicationState) -> dict:
    cost = _cost(state)
    p = state.detector_p
    decision = decide_one(p, cost)
    return {"tier1_decision": decision.value,
            "expected_costs": {k.value: v for k, v in cost.expected_costs(p).items()}}


def route_after_cost_policy(state: AdjudicationState) -> str:
    if state.tier1_decision != Decision.ESCALATE.value:
        return "finalize"
    return "vlm_second_look" if state.image_path else "human_interrupt"


def make_vlm_second_look_node(backend: VLMBackend):
    """Calls the backend ONLY (mirrors aiqs.vlm.backend's role) — no abstain logic here."""
    def vlm_second_look(state: AdjudicationState) -> dict:
        vs = VLMState(image_path=state.image_path,
                      detector_score=state.detector_score,
                      detector_p=state.detector_p,
                      anomaly_map_path=state.anomaly_map_path)
        verdict = backend(vs)
        return {"vlm_verdict": verdict.verdict, "vlm_confidence": verdict.confidence,
                "vlm_reasoning": verdict.reasoning,
                "tokens_in": vs.tokens_in, "tokens_out": vs.tokens_out}
    return vlm_second_look


def vlm_abstain_rule(state: AdjudicationState) -> dict:
    """Mirrors aiqs.vlm.abstain's role: confidence_to_p + the cost-matrix argmin."""
    cost = _cost(state)
    p_vlm = confidence_to_p(state.vlm_verdict, state.vlm_confidence, state.lam)
    decision = adjudicate_probability(p_vlm, cost)
    return {"p_vlm": p_vlm, "vlm_decision": decision.value}


def route_after_vlm_abstain(state: AdjudicationState) -> str:
    return ("human_interrupt" if state.vlm_decision == Decision.ESCALATE.value
            else "finalize")


def human_interrupt(state: AdjudicationState) -> dict:
    payload = {
        "item_id": state.item_id,
        "detector_score": state.detector_score,
        "detector_p": state.detector_p,
        "vlm_verdict": state.vlm_verdict,
        "vlm_confidence": state.vlm_confidence,
        "vlm_reasoning": state.vlm_reasoning,
        "question": "Human verdict required: pass or fail?",
    }
    verdict = interrupt(payload)  # dict: {"decision": "pass"|"fail", "reviewer"?, "note"?}
    return {"human_decision": verdict["decision"],
            "human_reviewer": verdict.get("reviewer"),
            "human_note": verdict.get("note")}


def finalize(state: AdjudicationState) -> dict:
    if state.human_decision is not None:
        return {"final_decision": state.human_decision, "resolved_by": "human"}
    if state.vlm_decision is not None and state.vlm_decision != Decision.ESCALATE.value:
        return {"final_decision": state.vlm_decision, "resolved_by": "vlm"}
    return {"final_decision": state.tier1_decision, "resolved_by": "policy"}
