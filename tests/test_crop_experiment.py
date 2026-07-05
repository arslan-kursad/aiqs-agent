"""Unit tests for the Stage-3 two-arm experiment (mocked — NO API calls).

Covers: the PRE-REGISTERED escape-classification rules (every branch + the word-boundary
guard + the adequacy ceiling), the paired escape comparison, stable-vs-flip, the 2A
replication descriptives, arm independence (separate states, A carries no map), the
served-model STOP guard, and a full mock end-to-end on a synthetic run dir.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aiqs.config import CropConfig
from aiqs.eval import crop_eval as ce
from aiqs.eval.decision import Decision
from aiqs.vlm import reasoning_rules as rr
from aiqs.vlm.backend import AnthropicVLMBackend
from aiqs.vlm.state import VLMState


# --------------------------------------------------------------------------- #
# Pre-registered classification rules
# --------------------------------------------------------------------------- #

def test_rule_priority_flip_wins():
    # ARM-B says defect -> PERCEPTION, even if the reasoning also contains normalizing words.
    assert rr.classify_escape("defect", "a scratch, not just a reflection",
                              diffuse=False) == rr.PERCEPTION


def test_rule_semantic_on_normalizing_clean():
    assert rr.classify_escape("clean", "the bright spot is a specular reflection",
                              diffuse=False) == rr.SEMANTIC
    assert rr.classify_escape("clean", "this is normal variation of the surface",
                              diffuse=False) == rr.SEMANTIC


def test_rule_word_boundary_abnormal_does_not_normalize():
    # "abnormal(ity)" must NOT match the \bnormal\b-family patterns.
    assert rr.classify_escape("clean", "a clear abnormality on the rim",
                              diffuse=False) == rr.UNCLASSIFIED


def test_rule_unclassified_branches():
    assert rr.classify_escape("unsure", "hard to tell", diffuse=False) == rr.UNCLASSIFIED
    assert rr.classify_escape("clean", "", diffuse=False) == rr.UNCLASSIFIED
    assert rr.classify_escape("clean", None, diffuse=False) == rr.UNCLASSIFIED
    assert rr.classify_escape("clean", "looks fine to me", diffuse=False) == rr.UNCLASSIFIED


def test_rule_diffuse_excluded():
    assert rr.classify_escape("defect", "obvious crack", diffuse=True) == rr.DIFFUSE_EXCLUDED


def test_distribution_and_adequacy_ceiling():
    labels = [rr.PERCEPTION] * 5 + [rr.SEMANTIC] * 3 + [rr.UNCLASSIFIED] * 2 \
        + [rr.DIFFUSE_EXCLUDED] * 4
    d = rr.distribution(labels)
    assert (d["perception"], d["semantic"], d["unclassified"], d["diffuse_excluded"]) \
        == (5, 3, 2, 4)
    assert d["unclassified_rate"] == pytest.approx(0.2)
    assert d["labeling_adequate"] is True
    d2 = rr.distribution([rr.UNCLASSIFIED] * 4 + [rr.PERCEPTION] * 6)
    assert d2["labeling_adequate"] is False       # 0.4 > the pre-registered 0.30 ceiling


# --------------------------------------------------------------------------- #
# Paired comparison + stability + replication
# --------------------------------------------------------------------------- #

def _state(label, decision, verdict="clean", conf=0.9, reasoning="r"):
    s = VLMState(image_path="x.png", detector_score=0.5, detector_p=0.5, label=label)
    s.vlm_verdict, s.vlm_conf, s.vlm_reasoning = verdict, conf, reasoning
    s.final_decision, s.abstained = decision, decision is Decision.ESCALATE
    return s


def test_paired_escape_comparison_counts():
    # 2 runs x 3 defective items; A escapes item0 both runs + item1 run0; B escapes item2 run1.
    esc_a = np.array([[True, True, False], [True, False, False]])
    esc_b = np.array([[False, False, False], [False, False, True]])
    p = ce.paired_escape_comparison(esc_a, esc_b, n_defective=3)
    assert p["escape_rate_a_mean"] == pytest.approx(0.5)      # (2/3 + 1/3)/2
    assert p["escape_rate_b_mean"] == pytest.approx(1 / 6)
    assert p["delta_mean"] > 0
    assert p["fixed_by_crop"] == 3 and p["broken_by_crop"] == 1


def test_stable_vs_flip_definition():
    esc_a = np.array([[True, True, False], [True, False, False]])  # item0 all-K, item1 flips
    st = ce.stable_vs_flip(esc_a)
    assert st == {"escaped_ever": 2, "stable_wrong": 1, "flipping": 1,
                  "stable_fraction": 0.5}


def test_classify_a_escapes_pairs_by_index_and_run():
    a = [[_state(1, Decision.PASS), _state(1, Decision.FAIL)]]        # item0 escapes
    b = [[_state(1, Decision.FAIL, verdict="defect"), _state(1, Decision.FAIL)]]
    d = ce.classify_a_escapes(a, b, diffuse_by_item=[False, False])
    assert d["perception"] == 1 and d["semantic"] == 0


def test_replication_confidence_auc_separates_and_not():
    # correct calls conf 0.9, wrong calls conf 0.3 -> AUC ~ 1.0 (separates)
    states = [ _state(0, Decision.PASS, "clean", 0.9),      # correct
               _state(1, Decision.PASS, "clean", 0.3) ]     # wrong (escape)
    rep = ce.replication_2a([states])
    assert rep["confidence_separation_auc"] == pytest.approx(1.0)
    assert rep["defect_escape_rate_mean"] == pytest.approx(1.0)
    # equal confidences -> AUC 0.5 (does not separate — the 2A observation)
    states2 = [ _state(0, Decision.PASS, "clean", 0.8),
                _state(1, Decision.PASS, "clean", 0.8) ]
    assert ce.replication_2a([states2])["confidence_separation_auc"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Served-model STOP guard (the silent-downgrade lesson, now in the backend)
# --------------------------------------------------------------------------- #

class _FakeMsg:
    def __init__(self, model):
        self.model, self.content, self.usage = model, [], None


def test_served_model_mismatch_stops(monkeypatch):
    backend = AnthropicVLMBackend()

    class _FakeClient:
        class messages:  # noqa: N801 - mimic SDK shape
            @staticmethod
            def create(**kwargs):
                return _FakeMsg("claude-3-5-sonnet-legacy")

    monkeypatch.setattr(backend, "_client_lazy", lambda: _FakeClient())
    with pytest.raises(RuntimeError, match="SERVED MODEL MISMATCH"):
        backend._invoke("sys", [{"type": "text", "text": "q"}],
                        VLMState(image_path="x", detector_score=0.5, detector_p=0.5))


# --------------------------------------------------------------------------- #
# End-to-end mock on a synthetic run dir (maps + manifest + images + scores)
# --------------------------------------------------------------------------- #

@pytest.fixture()
def synthetic_run(tmp_path):
    """A run dir with overlapping scores (so the policy escalates plenty of goods),
    per-item images and peaked anomaly maps + manifest."""
    rng = np.random.default_rng(7)
    n = 80
    labels = np.array([0, 1] * (n // 2))
    scores = np.where(labels == 0, rng.normal(0.42, 0.10, n), rng.normal(0.58, 0.10, n))
    run_dir = tmp_path / "runs" / "synth_run"
    (run_dir / "anomaly_maps").mkdir(parents=True)
    img_dir = tmp_path / "datasets" / "synth" / "test" / "mix"
    img_dir.mkdir(parents=True)

    rows, manifest = [], []
    for i in range(n):
        img = img_dir / f"{i:03d}.png"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img)
        m = np.zeros((16, 16)); m[4:6, 10:12] = 5.0          # clear peak
        np.save(run_dir / "anomaly_maps" / f"mix__{i:03d}.npy", m)
        rows.append((str(img), int(labels[i]), float(scores[i])))
        manifest.append((str(img), f"anomaly_maps/mix__{i:03d}.npy"))

    import pandas as pd
    pd.DataFrame(rows, columns=["image_path", "label", "score"]).to_csv(
        run_dir / "image_scores.csv", index=False)
    pd.DataFrame(manifest, columns=["image_path", "map_path"]).to_csv(
        run_dir / "anomaly_maps" / "manifest.csv", index=False)
    return tmp_path, run_dir


def _run(synthetic_run, **kw):
    from aiqs.vlm_crop import run_two_arm

    repo_root, run_dir = synthetic_run
    defaults = dict(k=2, mock=True, seed=42, folds=5, token_cost=0.0,
                    lambda_grid=[0.0, 1.0], repo_root=repo_root, max_items=None,
                    crop_cfg=CropConfig())
    defaults.update(kw)
    return run_two_arm(run_dir, **defaults)


def test_two_arm_mock_end_to_end(synthetic_run):
    res, comp, states_a, states_b = _run(synthetic_run)
    # arm independence: separate objects, A has no map, B has maps; same item order.
    assert all(s.anomaly_map_path is None for s in states_a[0])
    assert all(s.anomaly_map_path is not None for s in states_b[0])
    assert states_a[0][0] is not states_b[0][0]
    assert [s.image_path for s in states_a[0]] == [s.image_path for s in states_b[0]]
    # peaked maps -> nothing diffuse; evaluation object is fully populated.
    assert res.n_diffuse == 0
    assert res.arm_a.k_runs == 2 and res.arm_b.k_runs == 2
    assert 0 <= res.paired["escape_rate_a_mean"] <= 1
    assert comp["escalate_good"] >= 15


# --------------------------------------------------------------------------- #
# Checkpoint / resume + loud parse fallback (the money-protection layer)
# --------------------------------------------------------------------------- #

def test_checkpoint_resume_never_rebills(synthetic_run, monkeypatch):
    import aiqs.vlm_crop as vc
    from aiqs.vlm.backend import MockVLMBackend

    calls = {"n": 0}

    class CountingMock(MockVLMBackend):
        def __call__(self, state):
            calls["n"] += 1
            return super().__call__(state)

    monkeypatch.setattr(vc, "MockVLMBackend", CountingMock)
    res1, comp, a1, b1 = _run(synthetic_run)
    first = calls["n"]
    assert first > 0

    # Second identical invocation must be FULLY resumed: zero new backend calls,
    # identical paired numbers (reconstructed from the checkpoint, not re-billed).
    calls["n"] = 0
    res2, _, a2, b2 = _run(synthetic_run)
    assert calls["n"] == 0
    assert res2.paired == res1.paired
    assert [s.vlm_verdict for s in a2[0]] == [s.vlm_verdict for s in a1[0]]


def test_parse_failure_falls_back_loud_not_blocking(synthetic_run, monkeypatch):
    import aiqs.vlm_crop as vc
    from aiqs.vlm.backend import VLMParseError

    class RaisingMock:
        def __init__(self, *a, **kw): ...
        def __call__(self, state):
            raise VLMParseError("malformed response")

    monkeypatch.setattr(vc, "MockVLMBackend", RaisingMock)
    res, comp, states_a, states_b = _run(synthetic_run, k=1)
    s = states_a[0][0]
    assert s.vlm_verdict == "unsure" and s.vlm_reasoning.startswith("PARSE_FAILURE")
    assert s.abstained is True                     # conservative outcome: human review


def test_checkpoint_mismatch_refuses(synthetic_run):
    import json as js
    from aiqs.vlm_crop import _ckpt_path

    _run(synthetic_run)                            # writes a real checkpoint
    _, run_dir = synthetic_run
    ckpt = _ckpt_path(run_dir, mock=True)
    lines = ckpt.read_text().strip().split("\n")
    rec = js.loads(lines[0]); rec["image_path"] = "/somewhere/else.png"
    ckpt.write_text(js.dumps(rec) + "\n" + "\n".join(lines[1:]) + "\n")
    with pytest.raises(RuntimeError, match="CHECKPOINT MISMATCH"):
        _run(synthetic_run)
