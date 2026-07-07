"""FastAPI contract tests (httpx TestClient, mocked VLM, MemorySaver — no disk/API).

Covers each endpoint, schema validation, the override-regime behavior, /config
redaction, the item_id collision policy (409), and the image_path root restriction.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from aiqs.api.main import ServeContext, create_app
from aiqs.vlm.backend import MockVLMBackend

from conftest import ESCALATE_SCORE, FAIL_SCORE, PASS_SCORE


def _client(synthetic_artifact, *, image_root=None, auth_env="AIQS_TEST_UNSET_KEY",
           verdict_fn=None) -> TestClient:
    backend = MockVLMBackend(
        verdict_fn=verdict_fn or (lambda state, rng: ("clean", 0.9, "mock clean")))
    ctx = ServeContext(synthetic_artifact, backend, MemorySaver(), image_root=image_root,
                       auth_env=auth_env, provider="mock", model="mock")
    return TestClient(create_app(ctx))


# --------------------------------------------------------------------------- #
# /health, /config
# --------------------------------------------------------------------------- #

def test_health(synthetic_artifact):
    r = _client(synthetic_artifact).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["run_id"] == "synthetic_run"
    assert body["auth_enabled"] is False


def test_config_redacts_key_value_never_shows_it(synthetic_artifact, monkeypatch):
    monkeypatch.setenv("AIQS_TEST_UNSET_KEY", "super-secret-value")
    r = _client(synthetic_artifact).get("/config", headers={"x-api-key": "super-secret-value"})
    assert r.status_code == 200
    body = r.json()
    assert body["auth_env_name"] == "AIQS_TEST_UNSET_KEY"
    assert body["auth_enabled"] is True
    assert "super-secret-value" not in r.text
    assert body["locked_cost"] == {"false_accept": 10.0, "false_reject": 3.0, "escalation": 1.0}
    assert body["category"] == "synthetic"


def test_auth_required_when_key_configured(synthetic_artifact, monkeypatch):
    monkeypatch.setenv("AIQS_TEST_UNSET_KEY", "super-secret-value")
    client = _client(synthetic_artifact)
    r = client.get("/config")
    assert r.status_code == 401
    r = client.get("/config", headers={"x-api-key": "wrong"})
    assert r.status_code == 401
    r = client.get("/config", headers={"x-api-key": "super-secret-value"})
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# /adjudicate — the three topology paths
# --------------------------------------------------------------------------- #

def test_adjudicate_direct_pass(synthetic_artifact):
    client = _client(synthetic_artifact)
    r = client.post("/adjudicate", json={"anomaly_score": PASS_SCORE,
                                         "target_prevalence": "native"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "pass"
    assert body["resolved_by"] == "policy"
    assert body["pending_human"] is False
    assert body["item_id"]  # server-generated uuid


def test_adjudicate_direct_fail(synthetic_artifact):
    client = _client(synthetic_artifact)
    r = client.post("/adjudicate", json={"anomaly_score": FAIL_SCORE,
                                         "target_prevalence": "native"})
    assert r.status_code == 200
    assert r.json()["decision"] == "fail"
    assert r.json()["resolved_by"] == "policy"


def test_adjudicate_escalate_vlm_resolves_with_image_b64(synthetic_artifact):
    client = _client(synthetic_artifact)
    b64 = base64.b64encode(b"not really an image, mock never reads bytes").decode()
    r = client.post("/adjudicate", json={"anomaly_score": ESCALATE_SCORE,
                                         "target_prevalence": "native",
                                         "image_b64": b64})
    assert r.status_code == 200
    body = r.json()
    assert body["tier1_decision"] == "escalate"
    assert body["vlm"]["fired"] is True
    assert body["vlm"]["verdict"] == "clean"
    assert body["decision"] == "pass"
    assert body["resolved_by"] == "vlm"


def test_adjudicate_escalate_pending_human_round_trip(synthetic_artifact):
    client = _client(synthetic_artifact)
    r = client.post("/adjudicate", json={"anomaly_score": ESCALATE_SCORE,
                                         "target_prevalence": "native",
                                         "item_id": "rt-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "pending_human"
    assert body["pending_human"] is True

    r = client.get("/decisions/rt-1")
    assert r.status_code == 200
    assert r.json()["pending_human"] is True

    r = client.post("/human-verdict/rt-1", json={"decision": "fail", "reviewer": "kursad"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "fail"
    assert body["resolved_by"] == "human"

    r = client.get("/decisions/rt-1")
    assert r.status_code == 200
    assert r.json()["decision"] == "fail"
    assert r.json()["pending_human"] is False


# --------------------------------------------------------------------------- #
# Overrides: cost matrix + target_prevalence
# --------------------------------------------------------------------------- #

def test_cost_matrix_override_changes_the_decision(synthetic_artifact):
    """A score that FAILs under the locked matrix can PASS under a much more
    escape-tolerant override — proves overrides actually reach the policy math."""
    client = _client(synthetic_artifact)
    r = client.post("/adjudicate", json={"anomaly_score": FAIL_SCORE,
                                         "target_prevalence": "native"})
    assert r.json()["decision"] == "fail"

    r = client.post("/adjudicate", json={
        "anomaly_score": FAIL_SCORE, "target_prevalence": "native",
        "cost_matrix": {"false_accept": 0.01}})
    assert r.json()["decision"] == "pass"


def test_target_prevalence_default_is_the_artifact_default(synthetic_artifact):
    client = _client(synthetic_artifact)
    r = client.post("/adjudicate", json={"anomaly_score": FAIL_SCORE})
    assert r.json()["applied_target_prevalence"] == pytest.approx(
        synthetic_artifact.default_target_prevalence)


# --------------------------------------------------------------------------- #
# item_id collision policy (explicit 409, never silent overwrite)
# --------------------------------------------------------------------------- #

def test_duplicate_item_id_pending_returns_409(synthetic_artifact):
    client = _client(synthetic_artifact)
    client.post("/adjudicate", json={"anomaly_score": ESCALATE_SCORE,
                                     "target_prevalence": "native", "item_id": "dup-1"})
    r = client.post("/adjudicate", json={"anomaly_score": ESCALATE_SCORE,
                                         "target_prevalence": "native", "item_id": "dup-1"})
    assert r.status_code == 409
    assert r.json()["detail"]["see"] == "POST /human-verdict/dup-1"


def test_duplicate_item_id_finalized_returns_409(synthetic_artifact):
    client = _client(synthetic_artifact)
    client.post("/adjudicate", json={"anomaly_score": PASS_SCORE,
                                     "target_prevalence": "native", "item_id": "dup-2"})
    r = client.post("/adjudicate", json={"anomaly_score": PASS_SCORE,
                                         "target_prevalence": "native", "item_id": "dup-2"})
    assert r.status_code == 409
    assert r.json()["detail"]["see"] == "GET /decisions/dup-2"


def test_human_verdict_on_finalized_item_returns_409(synthetic_artifact):
    client = _client(synthetic_artifact)
    client.post("/adjudicate", json={"anomaly_score": PASS_SCORE,
                                     "target_prevalence": "native", "item_id": "dup-3"})
    r = client.post("/human-verdict/dup-3", json={"decision": "fail"})
    assert r.status_code == 409


def test_unknown_item_id_returns_404(synthetic_artifact):
    client = _client(synthetic_artifact)
    assert client.get("/decisions/never-posted").status_code == 404
    assert client.post("/human-verdict/never-posted",
                       json={"decision": "pass"}).status_code == 404


# --------------------------------------------------------------------------- #
# image_path root restriction (path-traversal guard)
# --------------------------------------------------------------------------- #

def test_image_path_disabled_without_image_root(synthetic_artifact):
    client = _client(synthetic_artifact, image_root=None)
    r = client.post("/adjudicate", json={"anomaly_score": PASS_SCORE,
                                         "image_path": "some.png"})
    assert r.status_code == 400
    assert "image_b64" in r.json()["detail"]


def test_image_path_traversal_rejected(tmp_path, synthetic_artifact):
    root = tmp_path / "images"
    root.mkdir()
    (root / "ok.png").write_bytes(b"fake png")
    client = _client(synthetic_artifact, image_root=root)

    r = client.post("/adjudicate", json={"anomaly_score": ESCALATE_SCORE,
                                         "target_prevalence": "native",
                                         "image_path": "../../../../etc/passwd"})
    assert r.status_code == 400
    assert "escapes" in r.json()["detail"]


def test_image_path_within_root_is_accepted(tmp_path, synthetic_artifact):
    root = tmp_path / "images"
    root.mkdir()
    (root / "ok.png").write_bytes(b"fake png")
    client = _client(synthetic_artifact, image_root=root)

    r = client.post("/adjudicate", json={"anomaly_score": ESCALATE_SCORE,
                                         "target_prevalence": "native",
                                         "image_path": "ok.png"})
    assert r.status_code == 200
    assert r.json()["vlm"]["fired"] is True


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #

def test_adjudicate_rejects_unknown_fields(synthetic_artifact):
    client = _client(synthetic_artifact)
    r = client.post("/adjudicate", json={"anomaly_score": PASS_SCORE, "bogus_field": 1})
    assert r.status_code == 422


def test_adjudicate_requires_anomaly_score(synthetic_artifact):
    client = _client(synthetic_artifact)
    r = client.post("/adjudicate", json={})
    assert r.status_code == 422


def test_human_verdict_rejects_invalid_decision(synthetic_artifact):
    client = _client(synthetic_artifact)
    client.post("/adjudicate", json={"anomaly_score": ESCALATE_SCORE,
                                     "target_prevalence": "native", "item_id": "bad-verdict"})
    r = client.post("/human-verdict/bad-verdict", json={"decision": "maybe"})
    assert r.status_code == 422
