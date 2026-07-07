"""Graph-path tests (mocked VLM, MemorySaver checkpointer — no disk/API).

Covers the four paths from the topology: direct PASS, direct FAIL, ESCALATE->VLM
resolves, and ESCALATE->VLM abstains->interrupt->human resume. Anchor scores come from
``conftest.make_synthetic_artifact`` (verified against ``ivap`` directly below).
"""

from __future__ import annotations

import numpy as np
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from aiqs.eval.decision import Decision, ivap, venn_abers_merge
from aiqs.graph.build import build_graph
from aiqs.vlm.backend import MockVLMBackend

from conftest import ESCALATE_SCORE, FAIL_SCORE, PASS_SCORE


def test_anchor_scores_land_in_the_expected_bands(synthetic_artifact):
    """Sanity-check the fixture's own claim before trusting the graph tests built on it."""
    for score, expected in ((PASS_SCORE, Decision.PASS), (ESCALATE_SCORE, Decision.ESCALATE),
                            (FAIL_SCORE, Decision.FAIL)):
        p0, p1 = ivap(synthetic_artifact.cal_scores, synthetic_artifact.cal_labels,
                      np.array([score]))
        p = venn_abers_merge(p0, p1)[0]
        from aiqs.eval.decision import decide_one
        assert decide_one(p, synthetic_artifact.locked_cost) is expected


def _initial(item_id: str, score: float, *, artifact, image_path=None) -> dict:
    return {
        "item_id": item_id, "detector_score": score,
        "target_prevalence": artifact.pi_source,  # == pi_source -> no prior-shift (native)
        "cost_false_accept": artifact.locked_cost.false_accept,
        "cost_false_reject": artifact.locked_cost.false_reject,
        "cost_escalation": artifact.locked_cost.escalation,
        "image_path": image_path,
    }


def _graph(artifact, backend):
    return build_graph(artifact, backend, MemorySaver())


def test_direct_pass(synthetic_artifact):
    backend = MockVLMBackend()  # never called on this path
    graph = _graph(synthetic_artifact, backend)
    cfg = {"configurable": {"thread_id": "t-pass"}}
    out = graph.invoke(_initial("t-pass", PASS_SCORE, artifact=synthetic_artifact), config=cfg)
    assert out["final_decision"] == Decision.PASS.value
    assert out["resolved_by"] == "policy"
    assert out["tier1_decision"] == Decision.PASS.value
    assert "vlm_verdict" not in out or out["vlm_verdict"] is None
    assert "__interrupt__" not in out


def test_direct_fail(synthetic_artifact):
    backend = MockVLMBackend()  # never called on this path
    graph = _graph(synthetic_artifact, backend)
    cfg = {"configurable": {"thread_id": "t-fail"}}
    out = graph.invoke(_initial("t-fail", FAIL_SCORE, artifact=synthetic_artifact), config=cfg)
    assert out["final_decision"] == Decision.FAIL.value
    assert out["resolved_by"] == "policy"
    assert out["tier1_decision"] == Decision.FAIL.value
    assert "__interrupt__" not in out


def test_escalate_vlm_resolves(synthetic_artifact):
    backend = MockVLMBackend(verdict_fn=lambda state, rng: ("clean", 0.95, "confident clean"))
    graph = _graph(synthetic_artifact, backend)
    cfg = {"configurable": {"thread_id": "t-vlm-resolves"}}
    out = graph.invoke(_initial("t-vlm-resolves", ESCALATE_SCORE, artifact=synthetic_artifact,
                                image_path="fake.png"), config=cfg)
    assert out["tier1_decision"] == Decision.ESCALATE.value
    assert out["vlm_verdict"] == "clean"
    assert out["final_decision"] == Decision.PASS.value
    assert out["resolved_by"] == "vlm"
    assert "__interrupt__" not in out


def test_escalate_no_image_skips_vlm_straight_to_human(synthetic_artifact):
    def _fail_if_called(state, rng):
        raise AssertionError("VLM must not be called when no image is provided")

    backend = MockVLMBackend(verdict_fn=_fail_if_called)
    graph = _graph(synthetic_artifact, backend)
    cfg = {"configurable": {"thread_id": "t-no-image"}}
    out = graph.invoke(_initial("t-no-image", ESCALATE_SCORE, artifact=synthetic_artifact),
                       config=cfg)
    assert out["tier1_decision"] == Decision.ESCALATE.value
    assert out.get("vlm_verdict") is None
    assert "__interrupt__" in out
    assert graph.get_state(cfg).next == ("human_interrupt",)


def test_escalate_vlm_abstains_then_human_resume(synthetic_artifact):
    backend = MockVLMBackend(verdict_fn=lambda state, rng: ("unsure", 0.5, "not sure"))
    graph = _graph(synthetic_artifact, backend)
    cfg = {"configurable": {"thread_id": "t-vlm-abstains"}}
    out = graph.invoke(_initial("t-vlm-abstains", ESCALATE_SCORE, artifact=synthetic_artifact,
                                image_path="fake.png"), config=cfg)
    assert out["vlm_verdict"] == "unsure"
    assert out["vlm_decision"] == Decision.ESCALATE.value
    assert "__interrupt__" in out
    assert graph.get_state(cfg).next == ("human_interrupt",)

    resumed = graph.invoke(
        Command(resume={"decision": "fail", "reviewer": "kursad", "note": "confirmed defect"}),
        config=cfg)
    assert resumed["final_decision"] == Decision.FAIL.value
    assert resumed["resolved_by"] == "human"
    assert resumed["human_reviewer"] == "kursad"
    # The VLM must not have been re-invoked by the resume (it is not on the replay path).
    assert resumed["vlm_verdict"] == "unsure"


def test_ingest_rejects_nan_score(synthetic_artifact):
    graph = _graph(synthetic_artifact, MockVLMBackend())
    cfg = {"configurable": {"thread_id": "t-nan"}}
    with pytest.raises(Exception):
        graph.invoke(_initial("t-nan", float("nan"), artifact=synthetic_artifact), config=cfg)
