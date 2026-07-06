"""The single seam: ``adjudicate(state) -> state``.

ENTERED ONLY BY ESCALATE-bucket items (the Tier-1 PASS/FAIL path is untouched). One
pass: call the VLM backend, map confidence to a provisional p_vlm, apply the calibrated
abstain rule. No recursion / re-perception (that is 2B). In 2B this exact function is
wrapped as a LangGraph node with no re-scaffolding.
"""

from __future__ import annotations

from aiqs.eval.decision import CostMatrix, Decision
from aiqs.vlm.abstain import adjudicate_probability, confidence_to_p
from aiqs.vlm.backend import VLMBackend
from aiqs.vlm.state import VLMState


def adjudicate(state: VLMState, backend: VLMBackend, cost: CostMatrix,
               lam: float = 0.0) -> VLMState:
    """Run the second-look on one escalated item and fill the decision fields.

    ``lam`` is the provisional-probability shrinkage (0 = trust the VLM confidence;
    higher = more conservative -> more abstentions). The eval sweeps it for the band.
    """
    verdict = backend(state)
    state.vlm_verdict = verdict.verdict
    state.vlm_conf = verdict.confidence
    state.vlm_reasoning = verdict.reasoning
    state.p_vlm = confidence_to_p(verdict.verdict, verdict.confidence, lam)
    state.final_decision = adjudicate_probability(state.p_vlm, cost)
    state.abstained = state.final_decision is Decision.ESCALATE
    return state
