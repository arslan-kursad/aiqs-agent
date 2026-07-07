"""Integration test: a REAL committed run artifact (the Phase-2B capsules headline run),
adjudicated end-to-end through the graph AND the FastAPI layer (VLM mocked — no API
calls), asserting the audit trace (``get_state_history``) is complete for every item.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from aiqs.api.artifact import load_artifact
from aiqs.api.main import ServeContext, create_app
from aiqs.graph.build import build_graph
from aiqs.vlm.backend import MockVLMBackend

REAL_RUN = "patchcore-wide_resnet50_2_visa-capsules_20260706T051813Z"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@pytest.fixture(scope="module")
def real_artifact():
    if not (RESULTS_DIR / "runs" / REAL_RUN / "image_scores.csv").exists():
        pytest.skip(f"committed run {REAL_RUN} not present in this checkout")
    return load_artifact(RESULTS_DIR, REAL_RUN)


def test_artifact_reproduces_the_committed_decide_numbers(real_artifact):
    """Cross-check against results/decisions.csv's own row for this run (the same
    numbers aiqs-decide persisted) — the artifact loader must not silently drift."""
    import pandas as pd

    decisions = pd.read_csv(RESULTS_DIR / "decisions.csv")
    row = decisions[decisions["run_id"] == REAL_RUN].iloc[0]
    assert real_artifact.n == int(row["n"])
    assert real_artifact.n_good == int(row["n_normal"])
    assert real_artifact.n_defective == int(row["n_defective"])
    assert real_artifact.auroc == pytest.approx(float(row["image_auroc"]), abs=1e-3)
    assert real_artifact.pi_source == pytest.approx(float(row["pi_source"]), abs=1e-3)
    assert real_artifact.category == "capsules"


def test_graph_end_to_end_on_real_scores_with_complete_trace(real_artifact):
    backend = MockVLMBackend(verdict_fn=lambda state, rng: ("clean", 0.95, "mock: real-run smoke"))
    graph = build_graph(real_artifact, backend, MemorySaver())

    lo = float(real_artifact.cal_scores.min())
    hi = float(real_artifact.cal_scores.max())
    mid = float((lo + hi) / 2)

    for i, score in enumerate([lo, mid, hi]):
        item_id = f"integration-{i}"
        config = {"configurable": {"thread_id": item_id}}
        out = graph.invoke({
            "item_id": item_id, "detector_score": score,
            "target_prevalence": real_artifact.default_target_prevalence,
            "cost_false_accept": real_artifact.locked_cost.false_accept,
            "cost_false_reject": real_artifact.locked_cost.false_reject,
            "cost_escalation": real_artifact.locked_cost.escalation,
            "image_path": "fake.png",
        }, config=config)

        history = list(graph.get_state_history(config))
        assert history, f"no trace recorded for {item_id}"
        # get_state_history returns newest-first; reverse for a chronological audit trail.
        chrono = list(reversed(history))
        assert chrono[0].metadata["step"] == -1  # __start__: no state yet
        assert chrono[0].values in ({}, None)
        # Every node the item passed through left its own checkpointed snapshot, in order.
        # chrono[0] is the pre-start snapshot (next=('__start__',), pregel bookkeeping, not
        # one of our nodes) — skip it.
        node_sequence = [snap.next[0] for snap in chrono[1:] if snap.next]
        expected_prefix = ["ingest", "calibrate", "cost_policy"]
        if out.get("tier1_decision") == "escalate":
            expected_prefix += ["vlm_second_look", "vlm_abstain_rule"]
        expected_prefix += ["finalize"]
        assert node_sequence == expected_prefix, (
            f"unexpected node sequence for {item_id} (score={score}): {node_sequence}")

        final_snap = graph.get_state(config)
        assert final_snap.values["detector_p"] == pytest.approx(
            real_artifact.calibrate(score, real_artifact.default_target_prevalence))
        assert "final_decision" in out or "__interrupt__" in out


def test_api_end_to_end_on_real_run(real_artifact):
    backend = MockVLMBackend(verdict_fn=lambda state, rng: ("clean", 0.8, "mock: real-run API smoke"))
    ctx = ServeContext(real_artifact, backend, MemorySaver(), image_root=None,
                      auth_env="AIQS_TEST_UNSET_KEY_2", provider="mock", model="mock")
    client = TestClient(create_app(ctx))

    cfg_resp = client.get("/config")
    assert cfg_resp.status_code == 200
    assert cfg_resp.json()["category"] == "capsules"

    hi = float(real_artifact.cal_scores.max())
    r = client.post("/adjudicate", json={
        "anomaly_score": hi, "item_id": "real-run-item", "image_b64": "aGVsbG8="})
    assert r.status_code == 200
    body = r.json()
    assert body["vlm"]["fired"] is True
    assert body["decision"] in ("pass", "fail")

    r2 = client.get("/decisions/real-run-item")
    assert r2.status_code == 200
    assert r2.json() == body
