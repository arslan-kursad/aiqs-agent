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


def test_served_model_dated_full_id_accepted(monkeypatch):
    # An alias may resolve to its dated full id (claude-haiku-4-5 ->
    # claude-haiku-4-5-20251001): that is the SAME model, not a downgrade — no stop.
    backend = AnthropicVLMBackend(model="claude-haiku-4-5")

    class _FakeClient:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kwargs):
                m = _FakeMsg("claude-haiku-4-5-20251001")
                return m

    monkeypatch.setattr(backend, "_client_lazy", lambda: _FakeClient())
    out = backend._invoke("sys", [{"type": "text", "text": "q"}],
                          VLMState(image_path="x", detector_score=0.5, detector_p=0.5))
    assert out == ""                               # empty content -> empty text, no raise


def test_rehearsal_model_gets_separate_checkpoint_namespace(synthetic_run):
    # A haiku rehearsal must NOT resume from (or write into) the canonical checkpoint.
    from aiqs.vlm_crop import _ckpt_path

    _, run_dir = synthetic_run
    canon = _ckpt_path(run_dir, mock=False)
    haiku = _ckpt_path(run_dir, mock=False, model="claude-haiku-4-5")
    assert canon != haiku and "claude-haiku-4-5" in haiku.name


# --------------------------------------------------------------------------- #
# ARM-C provider plumbing: dispatch, namespacing, checkpoint provider-awareness
# --------------------------------------------------------------------------- #

def test_model_suffix_backward_compatible_for_anthropic_default():
    from aiqs.vlm_crop import _model_suffix
    from aiqs.vlm.backend import MODEL

    assert _model_suffix(MODEL) == ""                       # unchanged: canonical
    assert _model_suffix(MODEL, "anthropic") == ""          # explicit provider, same
    assert _model_suffix("claude-haiku-4-5") == "__claude-haiku-4-5"  # unchanged rehearsal


def test_model_suffix_namespaces_non_anthropic_by_provider_and_model():
    from aiqs.vlm_crop import _model_suffix

    s = _model_suffix("gemini-3.1-flash-lite-preview", "openai_compatible")
    assert s == "__openai_compatible__gemini-3.1-flash-lite-preview"


def test_ckpt_path_and_write_results_are_provider_aware(synthetic_run):
    from aiqs.vlm_crop import _ckpt_path, write_results

    _, run_dir = synthetic_run
    anthropic_path = _ckpt_path(run_dir, mock=False, model="claude-haiku-4-5",
                                provider="anthropic")
    armc_path = _ckpt_path(run_dir, mock=False, model="claude-haiku-4-5",
                           provider="openai_compatible")
    assert anthropic_path != armc_path    # same model string, different provider -> no clash

    csv_path = write_results(run_dir, [], [], mock=True, model="gemini-x",
                             provider="openai_compatible")
    assert "openai_compatible__gemini-x" in csv_path.name


def test_build_backend_dispatch_and_validation():
    from aiqs.vlm.backend import AnthropicVLMBackend
    from aiqs.vlm.backend_openai_compatible import OpenAICompatibleVLMBackend
    from aiqs.vlm_crop import build_backend

    assert isinstance(build_backend("anthropic", "claude-sonnet-4-6"), AnthropicVLMBackend)
    oc = build_backend("openai_compatible", "gemini-x", base_url="https://x",
                       api_key_env="K")
    assert isinstance(oc, OpenAICompatibleVLMBackend)
    with pytest.raises(ValueError, match="requires --base-url"):
        build_backend("openai_compatible", "gemini-x")       # missing base_url/api_key_env
    with pytest.raises(ValueError, match="Unsupported provider"):
        build_backend("not-a-provider", "x")


def test_crop_fn_for_wraps_openai_shape_vs_anthropic_shape(synthetic_run):
    from aiqs.vlm_crop import crop_fn_for
    from aiqs.config import CropConfig
    from aiqs.vlm.state import VLMState

    _, run_dir = synthetic_run
    map_path = next((run_dir / "anomaly_maps").glob("*.npy"))
    img_path = next((run_dir.parent.parent / "datasets").rglob("*.png"))
    state = VLMState(image_path=str(img_path), detector_score=0.5, detector_p=0.5,
                     anomaly_map_path=str(map_path))

    anthropic_block, _ = crop_fn_for("anthropic", CropConfig())(state, 512)
    openai_block, _ = crop_fn_for("openai_compatible", CropConfig())(state, 512)
    assert anthropic_block["type"] == "image"                  # Anthropic shape
    assert openai_block["type"] == "image_url"                 # OpenAI-compatible shape


def test_checkpoint_record_has_timestamp_and_restore_tolerates_missing_one(synthetic_run):
    import json

    from aiqs.vlm_crop import _record, _restore
    from aiqs.vlm.state import VLMState

    s = VLMState(image_path="p.png", detector_score=0.5, detector_p=0.5, label=0)
    s.vlm_verdict, s.vlm_conf, s.p_vlm = "clean", 0.9, 0.1
    from aiqs.eval.decision import Decision
    s.final_decision, s.abstained = Decision.PASS, False

    rec = _record("A", 0, 0, s, "claude-sonnet-4-6", "anthropic")
    assert "timestamp" in rec and isinstance(rec["timestamp"], float)

    # Old-format record (no timestamp key) must still restore without crashing.
    old_rec = json.loads(json.dumps(rec))
    del old_rec["timestamp"]
    template = VLMState(image_path="p.png", detector_score=0.5, detector_p=0.5, label=0)
    restored = _restore(old_rec, template, with_map=False, model="claude-sonnet-4-6",
                        provider="anthropic")
    assert restored.vlm_verdict == "clean"


def test_restore_refuses_cross_provider_mismatch(synthetic_run):
    from aiqs.vlm_crop import _record, _restore
    from aiqs.vlm.state import VLMState
    from aiqs.eval.decision import Decision

    s = VLMState(image_path="p.png", detector_score=0.5, detector_p=0.5, label=0)
    s.vlm_verdict, s.vlm_conf, s.p_vlm = "clean", 0.9, 0.1
    s.final_decision, s.abstained = Decision.PASS, False
    rec = _record("A", 0, 0, s, "gemini-x", "openai_compatible")

    template = VLMState(image_path="p.png", detector_score=0.5, detector_p=0.5, label=0)
    with pytest.raises(RuntimeError, match="CHECKPOINT MISMATCH"):
        _restore(rec, template, with_map=False, model="gemini-x", provider="anthropic")


def test_run_smoke_makes_no_checkpoint_file(synthetic_run, monkeypatch):
    """--smoke uses a fresh backend per arm and writes NOTHING to the real checkpoint —
    it must be impossible for a smoke call to be mistaken for (or resumed as) a paid one."""
    from aiqs.config import CropConfig
    from aiqs.vlm.backend import VLMVerdict
    from aiqs.vlm_crop import _ckpt_path, run_smoke

    repo_root, run_dir = synthetic_run

    class _FakeBackend:
        def __call__(self, state):
            state.tokens_in, state.tokens_out = 100, 20
            return VLMVerdict(verdict="clean", confidence=0.9, reasoning="r")

    import aiqs.vlm_crop as vc

    monkeypatch.setattr(vc, "build_backend", lambda *a, **kw: _FakeBackend())
    run_smoke(run_dir, model="gemini-x", provider="openai_compatible",
             base_url="https://x", api_key_env="K", rpm_limit=None,
             repo_root=repo_root, crop_cfg=CropConfig())

    ckpt = _ckpt_path(run_dir, mock=False, model="gemini-x", provider="openai_compatible")
    assert not ckpt.exists()


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
