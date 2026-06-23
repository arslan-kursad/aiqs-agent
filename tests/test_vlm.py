"""Unit tests for the Phase-2A VLM second-look (mocked — NO API calls).

Covers: the abstain-rule math (+ shrinkage + 100/3/1 self-floor), the substrate guard,
structured-output parsing (incl. malformed -> loud), the PINNED "detector wrong"
definition, the error-independence + pre-registered-rule computation on synthetic
buckets, and K-run rule-stability. Pure numpy/pydantic; no torch/anomalib/anthropic.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiqs.eval.decision import CostMatrix, Decision, decide_one
from aiqs.eval import vlm_eval as ve
from aiqs.vlm.abstain import adjudicate_probability, confidence_to_p
from aiqs.vlm.adjudicate import adjudicate
from aiqs.vlm.backend import MockVLMBackend, VLMParseError, VLMVerdict, parse_verdict
from aiqs.vlm.state import VLMState
from aiqs.vlm.substrate import SubstrateError, bucket_composition, substrate_guard

LOCKED = CostMatrix(false_accept=10.0, false_reject=3.0, escalation=1.0)
REALISTIC = CostMatrix(false_accept=100.0, false_reject=3.0, escalation=1.0)


# --------------------------------------------------------------------------- #
# Abstain rule
# --------------------------------------------------------------------------- #

def test_confidence_to_p_map():
    assert confidence_to_p("defect", 1.0) == 1.0
    assert confidence_to_p("clean", 1.0) == 0.0
    assert confidence_to_p("unsure", 0.9) == 0.5
    assert confidence_to_p("defect", 0.0) == 0.5      # zero confidence -> coin flip
    assert confidence_to_p("defect", 0.6) == pytest.approx(0.8)


def test_shrinkage_pulls_toward_half():
    full = confidence_to_p("defect", 0.8, lam=0.0)
    shrunk = confidence_to_p("defect", 0.8, lam=0.5)
    assert abs(shrunk - 0.5) < abs(full - 0.5)
    assert confidence_to_p("defect", 1.0, lam=1.0) == 0.5   # full shrink -> always abstain


def test_confidence_clamped_and_bad_verdict():
    assert confidence_to_p("defect", 2.0) == 1.0            # clamp >1
    assert confidence_to_p("clean", -1.0) == 0.5            # clamp <0
    with pytest.raises(ValueError):
        confidence_to_p("maybe", 0.5)


def test_abstain_rule_is_phase1_policy():
    for p in np.linspace(0, 1, 21):
        assert adjudicate_probability(p, LOCKED) is decide_one(p, LOCKED)


def test_unsure_always_abstains_under_locked():
    p = confidence_to_p("unsure", 1.0)
    assert adjudicate_probability(p, LOCKED) is Decision.ESCALATE


def test_realistic_matrix_self_floors_auto_pass():
    """Under 100/3/1, auto-PASS requires p_vlm < review/escape = 1/100. A confident
    'clean' (p=0.05) must NOT auto-PASS — the escape cost dominates -> abstain/fail."""
    p = confidence_to_p("clean", 0.9)        # 0.05
    assert p > REALISTIC.escalation / REALISTIC.false_accept   # 0.05 > 0.01
    assert adjudicate_probability(p, REALISTIC) is not Decision.PASS


# --------------------------------------------------------------------------- #
# Substrate guard
# --------------------------------------------------------------------------- #

def test_substrate_hard_stop():
    with pytest.raises(SubstrateError):
        substrate_guard(14)


def test_substrate_warn_band():
    warns = substrate_guard(20)
    assert warns and "UNDERPOWERED" in warns[0]
    assert substrate_guard(40) == []          # >= WARN_MIN: no warning


def test_bucket_composition_identity_safe():
    labels = np.array([0, 0, 1, 1])
    decisions = np.array([Decision.ESCALATE, Decision.PASS,
                          Decision.ESCALATE, Decision.FAIL], dtype=object)
    comp = bucket_composition(labels, decisions)
    assert comp["escalate_total"] == 2
    assert comp["escalate_good"] == 1 and comp["escalate_defective"] == 1


# --------------------------------------------------------------------------- #
# Structured-output parsing
# --------------------------------------------------------------------------- #

def test_parse_plain_json():
    v = parse_verdict('{"verdict": "defect", "confidence": 0.8, "reasoning": "crack"}')
    assert v.verdict == "defect" and v.confidence == 0.8


def test_parse_fenced_json():
    raw = 'Here:\n```json\n{"verdict":"clean","confidence":0.6,"reasoning":"reflection"}\n```'
    assert parse_verdict(raw).verdict == "clean"


@pytest.mark.parametrize("bad", [
    "not json at all",
    '{"verdict": "maybe", "confidence": 0.5, "reasoning": "x"}',   # bad verdict
    '{"verdict": "defect", "confidence": 1.5, "reasoning": "x"}',  # out of range
    '{"verdict": "defect", "confidence": 0.5}',                    # missing field
    '{"verdict": "defect", "confidence": 0.5, "reasoning": "x", "extra": 1}',  # extra
    None,
])
def test_parse_malformed_is_loud(bad):
    with pytest.raises(VLMParseError):
        parse_verdict(bad)


# --------------------------------------------------------------------------- #
# Pinned "detector wrong" + VLM call mapping
# --------------------------------------------------------------------------- #

def test_detector_hard_decision_matches_naive():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 30), rng.normal(3, 1, 30)])
    labels = np.array([0] * 30 + [1] * 30)
    from aiqs.eval.decision import cost_optimal_threshold, naive_decide
    t = cost_optimal_threshold(scores, labels, LOCKED)
    expected = np.array([1 if d is Decision.FAIL else 0
                         for d in naive_decide(scores, t, 0.0)], dtype=int)
    np.testing.assert_array_equal(ve.detector_hard_decision(scores, labels, LOCKED),
                                  expected)


def test_vlm_call_label_unsure_uses_lean():
    assert ve.vlm_call_label("defect", 0.1) == 1
    assert ve.vlm_call_label("clean", 0.9) == 0
    assert ve.vlm_call_label("unsure", 0.7) == 1
    assert ve.vlm_call_label("unsure", 0.3) == 0


# --------------------------------------------------------------------------- #
# Statistics + pre-registered rule
# --------------------------------------------------------------------------- #

def test_wilson_interval_bounds():
    lo, hi = ve.wilson_interval(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0
    assert np.isnan(ve.wilson_interval(0, 0)[0])


def test_kappa_extremes():
    a = np.array([0, 0, 1, 1])
    assert ve.cohens_kappa(a, a) == pytest.approx(1.0)
    assert ve.cohens_kappa(a, 1 - a) == pytest.approx(-1.0)


def test_rule_label_three_branches():
    assert ve._rule_label(p_ind_lo=0.6, kappa=0.1) == "independent"
    assert ve._rule_label(p_ind_lo=0.6, kappa=0.9) == "redundant"
    assert ve._rule_label(p_ind_lo=0.3, kappa=0.1) == "theatre"
    assert ve._rule_label(p_ind_lo=float("nan"), kappa=0.0) == "theatre"


def test_error_independence_independent_case():
    # VLM correct everywhere; detector wrong on several -> high P(VLM|det wrong), kappa~0.
    labels = np.array([0] * 8 + [1] * 8)
    det_call = labels.copy()
    det_call[[0, 1, 2, 8, 9, 10]] ^= 1          # 6 detector errors
    vlm_call = labels.copy()                     # VLM perfect
    out = ve.error_independence(vlm_call, det_call, labels, seed=1)
    assert out.label == "independent"
    assert out.p_ind_lo > ve.P_IND_MIN


def test_error_independence_theatre_case():
    # VLM mirrors the detector -> wrong exactly where detector is wrong.
    labels = np.array([0] * 8 + [1] * 8)
    det_call = labels.copy()
    det_call[[0, 1, 2, 8, 9, 10]] ^= 1
    vlm_call = det_call.copy()
    out = ve.error_independence(vlm_call, det_call, labels, seed=1)
    assert out.label == "theatre"


# --------------------------------------------------------------------------- #
# adjudicate seam + K-run stability
# --------------------------------------------------------------------------- #

def _bucket(n_good=10, n_def=10):
    states = [VLMState(image_path=f"g{i}", detector_score=0.4, detector_p=0.3, label=0)
              for i in range(n_good)]
    states += [VLMState(image_path=f"d{i}", detector_score=0.6, detector_p=0.5, label=1)
               for i in range(n_def)]
    return states


def test_adjudicate_oracle_auto_decides():
    backend = MockVLMBackend(seed=0)          # default oracle from label
    s = adjudicate(VLMState("g0", 0.4, 0.3, label=0), backend, LOCKED)
    assert s.final_decision is Decision.PASS and not s.abstained
    assert s.vlm_verdict == "clean"


def test_krun_stability_deterministic_oracle():
    """Deterministic oracle -> identical runs -> 'independent in K/K'."""
    template = _bucket()
    scores = np.array([s.detector_score for s in template])
    labels = np.array([s.label for s in template])
    # Detector errs on a few items (the ESCALATE band is where it is uncertain); the
    # oracle VLM is right there -> independent signal. (Pass det_call directly so the
    # test exercises rule AGGREGATION, not the threshold search.)
    det_call = labels.copy()
    det_call[[0, 1, 2, 10, 11]] ^= 1

    states_per_run = []
    for r in range(ve.RUN_K):
        backend = MockVLMBackend(seed=r)       # oracle ignores seed -> identical
        states_per_run.append([adjudicate(VLMState(s.image_path, s.detector_score,
                                                    s.detector_p, label=s.label),
                                           backend, LOCKED) for s in template])
    res = ve.evaluate(states_per_run, scores, labels, det_call, LOCKED,
                      cost_label="10/3/1", token_cost=0.0,
                      lambda_grid=[0.0, 0.5, 1.0], seed=0)
    assert res.rule_distribution["independent"] == ve.RUN_K
    assert res.rule_modal == "independent"
    assert "5/5" in res.rule_stability
    assert res.accuracy_mean == pytest.approx(1.0)
    # oracle rescues every good and fails every defect; no errors -> shifts right.
    assert res.bidirectional["rescued_to_pass"] == 10
    assert res.bidirectional["correct_fail"] == 10
    assert res.breakeven_shifts_right is True


def test_krun_distribution_with_noise():
    """A noisy mock yields a rule distribution that still sums to K."""
    def noisy(state, rng):
        if rng.random() < 0.5:
            return "unsure", 0.0, "coin"
        v = "defect" if rng.random() < 0.5 else "clean"
        return v, float(rng.uniform(0.5, 1.0)), "noisy"

    template = _bucket()
    scores = np.array([s.detector_score for s in template])
    labels = np.array([s.label for s in template])
    det_call = ve.detector_hard_decision(scores, labels, LOCKED)
    states_per_run = []
    for r in range(ve.RUN_K):
        backend = MockVLMBackend(verdict_fn=noisy, seed=r)
        states_per_run.append([adjudicate(VLMState(s.image_path, s.detector_score,
                                                    s.detector_p, label=s.label),
                                           backend, LOCKED) for s in template])
    res = ve.evaluate(states_per_run, scores, labels, det_call, LOCKED,
                      cost_label="10/3/1", token_cost=0.0,
                      lambda_grid=[0.0, 1.0], seed=0)
    assert sum(res.rule_distribution.values()) == ve.RUN_K
    assert len(res.eff_review_cost_band) == 2
