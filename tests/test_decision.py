"""Unit tests for the Phase-1 decision spine (aiqs.eval.decision).

Focus: the policy math (expected-cost argmin + tie-break), Venn-Abers calibration
sanity, the strong naive baseline, decision metrics, and the degenerate/weak guard.
No torch/anomalib; pure numpy + sklearn.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiqs.eval.decision import (
    CostMatrix,
    Decision,
    DegenerateScoresError,
    check_not_degenerate,
    cost_optimal_threshold,
    cross_venn_abers,
    decide,
    decide_one,
    decision_metrics,
    empirical_auroc,
    escalation_cost_ceiling,
    ivap,
    naive_decide,
    prevalence_weights,
    prior_shift,
    simulate_scores,
    venn_abers_merge,
)

LOCKED = CostMatrix(false_accept=10.0, false_reject=3.0, escalation=1.0)


# --------------------------------------------------------------------------- #
# Policy math
# --------------------------------------------------------------------------- #

def test_locked_decision_regions():
    # PASS if p<=0.10, ESCALATE if 0.10<p<0.667, FAIL if p>=0.667.
    assert decide_one(0.05, LOCKED) is Decision.PASS
    assert decide_one(0.30, LOCKED) is Decision.ESCALATE
    assert decide_one(0.70, LOCKED) is Decision.FAIL


def test_indifference_points_match_matrix():
    pts = LOCKED.indifference_points()
    assert pts["pass_escalate"] == pytest.approx(0.10)
    assert pts["fail_escalate"] == pytest.approx(2.0 / 3.0)
    assert pts["pass_fail"] == pytest.approx(3.0 / 13.0)


def test_tie_break_prefers_safer_action():
    # At p=0.10, E[PASS]=E[ESCALATE]=1.0 -> never silently PASS: ESCALATE wins.
    assert decide_one(0.10, LOCKED) is Decision.ESCALATE
    # At p=2/3, E[FAIL]=E[ESCALATE]=1.0 -> ESCALATE wins over FAIL too.
    assert decide_one(2.0 / 3.0, LOCKED) is Decision.ESCALATE


def test_extreme_escape_cost_never_passes_uncertain():
    # Very high escape cost: an uncertain part must NOT be PASS'd...
    high = CostMatrix(false_accept=1000.0, false_reject=3.0, escalation=1.0)
    for p in (0.01, 0.05, 0.2, 0.5):
        assert decide_one(p, high) is not Decision.PASS
    # ...but a near-certain-good part still may be PASS'd (it's not "never PASS").
    assert decide_one(0.0005, high) is Decision.PASS


def test_escalation_ceiling_collapses_band():
    # At/above the ceiling, ESCALATE is never optimal (coverage -> 1).
    ceil = escalation_cost_ceiling(LOCKED)
    assert ceil == pytest.approx(30.0 / 13.0)
    no_abstain = CostMatrix(false_accept=10.0, false_reject=3.0, escalation=ceil + 0.1)
    decs = {decide_one(p, no_abstain) for p in np.linspace(0, 1, 51)}
    assert Decision.ESCALATE not in decs


# --------------------------------------------------------------------------- #
# Venn-Abers calibration
# --------------------------------------------------------------------------- #

def test_merge_bounds_and_monotonic():
    assert venn_abers_merge(0.0, 0.0) == pytest.approx(0.0)
    assert venn_abers_merge(1.0, 1.0) == pytest.approx(1.0)
    p = venn_abers_merge(np.array([0.2, 0.4]), np.array([0.3, 0.9]))
    assert np.all((p >= 0) & (p <= 1))
    # p0<=p result<=p1
    assert 0.2 <= p[0] <= 0.3


def test_ivap_is_calibrated_monotone():
    rng = np.random.default_rng(0)
    cal_scores = np.concatenate([rng.normal(0.3, 0.05, 60),
                                 rng.normal(0.7, 0.05, 60)])
    cal_labels = np.array([0] * 60 + [1] * 60)
    p0, p1 = ivap(cal_scores, cal_labels, np.array([0.2, 0.5, 0.8]))
    assert np.all(p0 <= p1)
    p = venn_abers_merge(p0, p1)
    assert p[0] < p[1] < p[2]          # higher score -> higher P(defective)
    assert p[0] < 0.2 and p[2] > 0.8   # confident at the extremes


def test_cross_venn_abers_shape_and_range():
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(0.4, 0.1, 50), rng.normal(0.6, 0.1, 50)])
    labels = np.array([0] * 50 + [1] * 50)
    p, p0, p1 = cross_venn_abers(scores, labels, k=5, seed=42)
    assert p.shape == (100,)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    assert np.all(p0 <= p1)


# --------------------------------------------------------------------------- #
# Naive baseline + metrics
# --------------------------------------------------------------------------- #

def test_cost_optimal_threshold_separates_clean_data():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    thr = cost_optimal_threshold(scores, labels, LOCKED)
    assert 0.3 < thr < 0.7
    decs = naive_decide(scores, thr, 0.0)
    m = decision_metrics(labels, decs, LOCKED)
    assert m.total_cost == 0.0
    assert m.false_reject_rate == 0.0 and m.false_accept_rate == 0.0


def test_decision_metrics_hand_example():
    # 2 good, 2 bad. Decisions: good->FAIL (overkill), good->PASS (ok),
    # bad->PASS (escape), bad->ESCALATE.
    labels = np.array([0, 0, 1, 1])
    decs = np.array([Decision.FAIL, Decision.PASS, Decision.PASS, Decision.ESCALATE],
                    dtype=object)
    m = decision_metrics(labels, decs, LOCKED)
    assert m.n == 4 and m.n_auto == 3 and m.n_escalate == 1
    assert m.coverage == pytest.approx(0.75)
    assert m.escalation_rate == pytest.approx(0.25)
    # auto-decided goods = 2 (one FAIL'd) -> FRR 0.5; auto-decided bads = 1 (PASS'd) -> FAR 1.0
    assert m.false_reject_rate == pytest.approx(0.5)
    assert m.false_accept_rate == pytest.approx(1.0)
    # cost: overkill(3) + escape(10) + escalate(1) = 14 over 4 items
    assert m.total_cost == pytest.approx(14.0)
    assert m.cost_per_item == pytest.approx(3.5)


def test_metrics_nan_when_no_auto_decided_class():
    labels = np.array([0, 1])
    decs = np.array([Decision.ESCALATE, Decision.PASS], dtype=object)
    m = decision_metrics(labels, decs, LOCKED)
    assert np.isnan(m.false_reject_rate)   # no auto-decided good part
    assert m.false_accept_rate == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Guard
# --------------------------------------------------------------------------- #

def test_guard_constant_scores_raises_loudly():
    scores = np.full(40, 0.5)
    labels = np.array([0] * 20 + [1] * 20)
    with pytest.raises(DegenerateScoresError) as exc:
        check_not_degenerate(scores, labels, k=5)
    assert "non-degenerate" in str(exc.value).lower()


def test_guard_chance_auroc_raises():
    # AUROC exactly 0.5: identical score distributions per class.
    scores = np.array([0.1, 0.9, 0.1, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert empirical_auroc(scores, labels) == pytest.approx(0.5)
    with pytest.raises(DegenerateScoresError):
        check_not_degenerate(scores, labels, k=2)


def test_guard_weak_detector_warns_not_raises():
    # AUROC ~0.58 (weak but real): all neg at 0.5; 58% of pos strictly above.
    neg = np.full(50, 0.5)
    pos = np.array([0.6] * 29 + [0.4] * 21)
    scores = np.concatenate([neg, pos])
    labels = np.array([0] * 50 + [1] * 50)
    auc = empirical_auroc(scores, labels)
    assert 0.53 < auc < 0.60
    warnings = check_not_degenerate(scores, labels, k=10)
    assert warnings and any("WEAK" in w for w in warnings)


def test_guard_too_few_for_kfold_raises():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(DegenerateScoresError):
        check_not_degenerate(scores, labels, k=10)  # only 2 per class


# --------------------------------------------------------------------------- #
# Prevalence correction (prior shift + importance weighting)
# --------------------------------------------------------------------------- #

def test_prior_shift_identity_when_target_equals_source():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(prior_shift(p, 0.74, 0.74), p)


def test_prior_shift_odds_ratio_is_exact():
    # Under label shift the odds scale by a constant (pi_t/pi_s)*((1-pi_s)/(1-pi_t)).
    p = np.array([0.2, 0.5, 0.74, 0.9])
    pi_s, pi_t = 0.74, 0.02
    pt = prior_shift(p, pi_s, pi_t)
    expected = (pi_t / pi_s) * ((1 - pi_s) / (1 - pi_t))
    ratio = (pt / (1 - pt)) / (p / (1 - p))
    assert np.allclose(ratio, expected)


def test_prior_shift_lowers_and_preserves_order():
    p = np.array([0.3, 0.6, 0.8])
    pt = prior_shift(p, 0.74, 0.02)
    assert np.all(pt < p)                      # lower target prior -> lower probs
    assert np.all((pt >= 0) & (pt <= 1))
    assert np.all(np.diff(pt) > 0)             # monotonic order preserved


def test_prevalence_weights_values():
    labels = np.array([0, 0, 0, 1])            # source prevalence 0.25
    w = prevalence_weights(labels, pi_target=0.02)
    assert w[3] == pytest.approx(0.02 / 0.25)          # defective
    assert w[0] == pytest.approx((1 - 0.02) / (1 - 0.25))  # good


def test_weighted_metrics_match_prevalence_theory():
    # FAIL-all and PASS-all under reweighting must hit the closed forms
    # (1-pi_t)*false_reject and pi_t*false_accept respectively.
    labels = np.array([0] * 30 + [1] * 70)     # source 0.7 defective
    pi_t = 0.02
    w = prevalence_weights(labels, pi_t)
    fail_all = np.array([Decision.FAIL] * 100, dtype=object)
    pass_all = np.array([Decision.PASS] * 100, dtype=object)
    mf = decision_metrics(labels, fail_all, LOCKED, weights=w)
    mp = decision_metrics(labels, pass_all, LOCKED, weights=w)
    assert mf.cost_per_item == pytest.approx((1 - pi_t) * LOCKED.false_reject)
    assert mp.cost_per_item == pytest.approx(pi_t * LOCKED.false_accept)
    assert mf.weighted and mp.weighted


# --------------------------------------------------------------------------- #
# Synthetic generator + end-to-end machinery validation
# --------------------------------------------------------------------------- #

def test_simulate_scores_hits_target_auroc():
    scores, labels = simulate_scores(n=4000, auroc=0.85, prevalence=0.5, seed=1)
    assert abs(empirical_auroc(scores, labels) - 0.85) < 0.03


def test_machinery_beats_naive_when_signal_exists():
    # The decision layer (calibrate -> prior-shift -> cost policy) must beat the
    # strong naive baseline once the detector actually separates classes.
    from aiqs.decide import analyze

    scores, labels = simulate_scores(n=1800, auroc=0.92, prevalence=0.25, seed=0)
    a = analyze(scores, labels, run_name="t", seed=0, folds=10,
                target_prevalence=0.02, grid=9)
    # production regime: lower cost
    assert a.target.m_ours.cost_per_item < a.target.m_naive.cost_per_item
    # native regime: lower cost AND less overkill than the cost-optimal threshold
    assert a.native.m_ours.cost_per_item < a.native.m_naive.cost_per_item
    assert a.native.m_ours.false_reject_rate < a.native.m_naive.false_reject_rate


def test_machinery_no_value_when_signal_absent():
    # Mirror of the real-data finding: at chance-ish separation the layer cannot
    # beat naive (it should not fabricate a win). AUROC ~0.56, above the stop band.
    from aiqs.decide import analyze

    scores, labels = simulate_scores(n=1800, auroc=0.56, prevalence=0.25, seed=2)
    a = analyze(scores, labels, run_name="t", seed=0, folds=10,
                target_prevalence=0.02, grid=9)
    # ours must not be cheaper by more than noise — no manufactured separation
    assert a.target.m_ours.cost_per_item >= a.target.m_naive.cost_per_item - 1e-3


# --------------------------------------------------------------------------- #
# Break-even review cost + operating-envelope regimes
# --------------------------------------------------------------------------- #

def test_breakeven_review_cost_sanity_and_bounds():
    from aiqs.eval.decision import breakeven_review_cost

    scores, labels = simulate_scores(n=2000, auroc=0.92, prevalence=0.3, seed=0)
    p, _, _ = cross_venn_abers(scores, labels, k=10, seed=0)
    thr = cost_optimal_threshold(scores, labels, LOCKED)
    naive_cost = decision_metrics(labels, naive_decide(scores, thr, 0.0), LOCKED).cost_per_item
    rows, c_star = breakeven_review_cost(p, labels, 10.0, 3.0, naive_cost, grid=121)
    # As review -> 0 escalation is free, so OURS MUST beat naive (the required sanity).
    assert rows[0]["beats_naive"]
    assert rows[0]["ours_cost"] <= naive_cost + 1e-9
    # break-even sits within [0, ceiling].
    ceil = 10.0 * 3.0 / (10.0 + 3.0)
    assert 0.0 <= c_star <= ceil * 1.05 + 1e-9


def test_breakeven_is_contiguous_from_zero_not_high_c_tie():
    # Regression: weak detector + high prevalence -> naive = FAIL-all, and OURS also
    # collapses to ~FAIL-all at high review cost, TYING naive there. The break-even must
    # be the low crossing (end of the contiguous winning region from c=0), NOT the
    # spurious high-c tie that a naive max-over-all-wins would return.
    from aiqs.eval.decision import breakeven_review_cost

    scores, labels = simulate_scores(n=800, auroc=0.56, prevalence=0.74, seed=5)
    p, _, _ = cross_venn_abers(scores, labels, k=10, seed=0)
    thr = cost_optimal_threshold(scores, labels, LOCKED)
    naive_cost = decision_metrics(labels, naive_decide(scores, thr, 0.0), LOCKED).cost_per_item
    rows, c_star = breakeven_review_cost(p, labels, 10.0, 3.0, naive_cost, grid=121)
    losses = [r["review_cost"] for r in rows if r["ours_cost"] > naive_cost + 1e-9]
    assert losses, "weak detector should make OURS lose at some review cost"
    assert c_star <= min(losses) + 1e-9     # contiguous region ends before the first loss


def test_verdict_expensive_review_branch_is_diagnosed():
    # Direct, deterministic test of the verdict logic: ours costs MORE than naive but
    # REDUCES overkill+escape -> must be diagnosed "expensive review" (the strong-detector
    # / costly-review regime), NOT the old "no separation to exploit" bug.
    from aiqs.decide import Regime, _verdict
    from aiqs.eval.decision import DecisionMetrics

    ours = DecisionMetrics(n=100, n_auto=80, coverage=0.80, escalation_rate=0.20,
                           false_reject_rate=0.10, false_accept_rate=0.00,
                           total_cost=30.0, cost_per_item=0.30)
    naive = DecisionMetrics(n=100, n_auto=100, coverage=1.0, escalation_rate=0.0,
                            false_reject_rate=0.25, false_accept_rate=0.02,
                            total_cost=22.0, cost_per_item=0.22)
    reg = Regime("native", 0.74, LOCKED, "10/3/1", ours, naive, 0.5, [], [], None, None,
                 breakeven_rows=[{"review_cost": 0.0, "ours_cost": 0.0,
                                  "escalation_rate": 1.0, "beats_naive": True}],
                 breakeven_cost=0.6)
    v = _verdict(reg)
    assert "no separation to exploit" not in v          # regression guard
    assert "expensive review" in v and "Break-even" in v


def test_native_verdict_consistent_with_metrics():
    # On real-shaped simulated data the verdict must agree with the actual cost ranking
    # and never emit the removed buggy phrase.
    from aiqs.decide import analyze, _verdict

    scores, labels = simulate_scores(n=2000, auroc=0.95, prevalence=0.74, seed=3)
    a = analyze(scores, labels, run_name="t", seed=0, folds=10,
                target_prevalence=0.02, grid=11)
    v = _verdict(a.native)
    assert "no separation to exploit" not in v
    if a.native.m_naive.cost_per_item - a.native.m_ours.cost_per_item > 1e-4:
        assert "beats naive" in v
    assert a.native.breakeven_cost is not None        # break-even always computed for native


def test_realistic_escape_dominant_matrix_helps_at_low_prevalence():
    # Under illustrative 10/3/1 at 2% prevalence, PASS-all is optimal (abstention moot).
    # Under realistic escape-dominant 100/3/1, the trade-off returns and OURS should at
    # least match — typically beat — the tuned threshold.
    from aiqs.decide import analyze

    scores, labels = simulate_scores(n=2500, auroc=0.95, prevalence=0.3, seed=1)
    a = analyze(scores, labels, run_name="t", seed=0, folds=10,
                target_prevalence=0.02, grid=11)
    assert a.target.cost_label == "10/3/1"
    assert a.target_realistic.cost_label == "100/3/1"
    assert (a.target_realistic.m_ours.cost_per_item
            <= a.target_realistic.m_naive.cost_per_item + 1e-9)
