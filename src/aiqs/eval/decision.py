"""Phase-1 decision layer: calibrated, cost-aware, *abstaining* decisions.

This is the deterministic decision spine the project's thesis rests on: the value
is not in the detector (commodity) but in turning its raw anomaly scores into
production-trustworthy actions. We optimise a **business cost function** and the
**false-reject rate** (overkill), NOT detection AUROC.

The module is intentionally decoupled from torch/anomalib (plain numpy + scikit's
IsotonicRegression/StratifiedKFold), consistent with the Phase-0 eval backbone: it
consumes the per-image scores persisted to ``results/runs/<id>/image_scores.csv``
and never re-runs the detector.

Pipeline (pure functions, all here; orchestration/IO lives in ``aiqs.decide``):

    raw anomaly score
      -> CONFORMAL calibration -> P(defective)          (Venn-Abers, below)
      -> cost-matrix expected-cost argmin               (decide_one)
      -> PASS / FAIL / ESCALATE(->human)
      -> DecisionMetrics (overkill, escape, escalation, cost)

Calibration = **Venn-Abers** (a member of the conformal-prediction family), built
directly on isotonic regression. We did NOT use MAPIE: its classification API emits
prediction *sets*, whereas the expected-cost argmin needs a scalar P(defective);
and our "model" is a single 1-D score, not a fitted sklearn estimator with
``predict_proba``. Venn-Abers yields exactly the calibrated probability we need in
~30 lines on a dependency we already ship. See CLAUDE.md for the full rationale.

Two calibration modes:
  * ``cross_venn_abers`` (PRIMARY) — out-of-fold over all labelled items. With only
    41 normal parts in the screw test set, a single split leaves ~20 per half (too
    fragile for a headline); cross-conformal uses every item while keeping each
    item's probability leakage-free (predicted by a calibrator that never saw it).
  * ``ivap`` (SECONDARY) — a single calibration->evaluation split (the spec-literal
    inductive Venn-Abers reference). Exactly Venn-valid, but small.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


class Decision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ESCALATE = "escalate"  # route to a human


@dataclass(frozen=True)
class CostMatrix:
    """Business cost of each (decision, true_state) outcome (relative units).

    Defaults are this project's LOCKED matrix: an escaped defect (false accept)
    dwarfs an overkill (false reject), and a human review is cheap relative to
    either error. These three numbers drive the entire policy.
    """

    false_accept: float = 10.0   # passed an actually-defective part (escape)
    false_reject: float = 3.0    # failed an actually-good part (overkill)
    escalation: float = 1.0      # human review, regardless of true state
    true_pass: float = 0.0
    true_fail: float = 0.0

    def expected_costs(self, p: float) -> dict[Decision, float]:
        """Expected cost of each action given p = P(defective)."""
        return {
            Decision.PASS: self.false_accept * p + self.true_pass * (1.0 - p),
            Decision.FAIL: self.true_fail * p + self.false_reject * (1.0 - p),
            Decision.ESCALATE: self.escalation,
        }

    def indifference_points(self) -> dict[str, float]:
        """p-values where the argmin switches (for the default 0/0 correct costs).

        PASS<->ESCALATE at p = esc/fa; FAIL<->ESCALATE at p = 1 - esc/fr;
        PASS<->FAIL at p = fr/(fa+fr). With the locked 10/3/1 matrix:
        0.100, 0.667, 0.231 -> PASS if p<=0.10, ESCALATE if 0.10<p<0.667, FAIL else.
        """
        fa, fr, esc = self.false_accept, self.false_reject, self.escalation
        return {
            "pass_escalate": esc / fa,
            "fail_escalate": 1.0 - esc / fr,
            "pass_fail": fr / (fa + fr),
        }


@dataclass
class DecisionMetrics:
    """Decision-level metrics over an evaluated set.

    Risk metrics (``false_reject_rate`` / ``false_accept_rate``) are computed on the
    AUTO-DECIDED subset (items the policy did not escalate) — this is the selective
    classification convention and what the risk-coverage curve plots. This extends
    the Phase-0 stub, which described them over "all good/bad parts"; abstention
    makes the covered-region denominator the meaningful one. Raw counts are kept so
    either denominator can be re-derived.

    ``total_cost`` is the REALIZED business cost under the (locked) cost matrix:
    escalated items are charged the review cost; auto-decided items are charged the
    relevant error/correct cost. When the policy's escalation cost is swept to trace
    a risk-coverage curve, accounting still uses the locked matrix (the sweep changes
    *behaviour*, not the true business costs).

    Under importance weighting (``weighted=True``, a non-None ``target_prevalence``):
    coverage / escalation_rate / rates / costs become weighted ratios that ESTIMATE
    the quantities at the target production prevalence (the benchmark's 74%-defective
    split is inverted to a realistic low defect rate). ``total_cost`` is then a
    weighted sum and ``cost_per_item`` the expected per-part cost in that population;
    the integer ``n_*`` counts stay RAW (facts about the labelled sample, not
    population estimates). Class-conditional rates (FRR/FAR) are prevalence-invariant
    by construction — the regime shows its effect through which decisions are taken
    (via prior-shifted probabilities) and through cost_per_item's class mix.
    """

    n: int                       # all items
    n_auto: int                  # auto-decided (PASS|FAIL)
    coverage: float              # n_auto / n  (= 1 - escalation_rate)
    escalation_rate: float       # escalated / all
    false_reject_rate: float     # overkill: good&FAIL / auto-decided good   (nan if none)
    false_accept_rate: float     # escape:   bad&PASS  / auto-decided bad    (nan if none)
    total_cost: float            # realized, locked matrix; escalations at review cost
    cost_per_item: float         # total_cost / n
    # raw counts (transparency / re-derivation) — always UNWEIGHTED
    n_pass: int = 0
    n_fail: int = 0
    n_escalate: int = 0
    n_good_auto: int = 0
    n_bad_auto: int = 0
    n_overkill: int = 0          # good & FAIL
    n_escape: int = 0            # bad  & PASS
    # regime
    target_prevalence: float | None = None   # None => native sample prevalence
    weighted: bool = False                   # True => importance-weighted ratios


# --------------------------------------------------------------------------- #
# Calibration: Venn-Abers (conformal family), built on isotonic regression.
# --------------------------------------------------------------------------- #

def venn_abers_merge(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Collapse the Venn-Abers multiprobability [p0, p1] to a scalar probability.

    Standard merge p = p1 / (1 - p0 + p1). The denominator is 1 + (p1 - p0) >= 1
    (since p1 >= p0 by construction), so it is always safe.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    return p1 / (1.0 - p0 + p1)


def ivap(cal_scores: np.ndarray, cal_labels: np.ndarray,
         test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inductive (split) Venn-Abers: calibrate on (cal_scores, cal_labels), predict
    [p0, p1] for each test score.

    For each test score s, isotonic regression is refit on the calibration set
    augmented with (s, 0) -> p0(s), and with (s, 1) -> p1(s); both read off at s.
    This is the transparent O(n_test * n_cal log n_cal) form — fine at our scale
    (hundreds of points). Higher score => higher P(defective), so increasing=True.
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_labels = np.asarray(cal_labels, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)

    p0 = np.empty(test_scores.shape[0], dtype=float)
    p1 = np.empty(test_scores.shape[0], dtype=float)
    for i, s in enumerate(test_scores):
        xs = np.append(cal_scores, s)
        for label, out in ((0.0, p0), (1.0, p1)):
            ys = np.append(cal_labels, label)
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(xs, ys)
            out[i] = float(iso.predict([s])[0])
    return p0, p1


def cross_venn_abers(scores: np.ndarray, labels: np.ndarray, *,
                     k: int = 10, seed: int = 42
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold Venn-Abers over ALL labelled items (the PRIMARY estimate).

    Stratified K-fold: each item is predicted by an inductive Venn-Abers calibrated
    on the OTHER folds, so its probability never saw its own label — every item gets
    a leakage-free calibrated P(defective). This trades exact single-split Venn
    validity for full data usage (the standard, slightly-conservative cross-conformal
    trade-off) — the right call when normal parts are scarce.

    Returns (p, p0, p1) aligned to the input order.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = scores.shape[0]
    p0 = np.full(n, np.nan)
    p1 = np.full(n, np.nan)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for cal_idx, test_idx in skf.split(scores, labels):
        f0, f1 = ivap(scores[cal_idx], labels[cal_idx], scores[test_idx])
        p0[test_idx] = f0
        p1[test_idx] = f1

    return venn_abers_merge(p0, p1), p0, p1


# --------------------------------------------------------------------------- #
# Decision policy: expected-cost argmin with a deterministic, safety-first tie-break.
# --------------------------------------------------------------------------- #

# Tie-break priority (lower index wins on an exact cost tie): never silently PASS an
# uncertain part. Coarse conformal probabilities (few normal calibration points) DO
# produce ties, so this is load-bearing, not cosmetic.
_TIE_PRIORITY = {Decision.ESCALATE: 0, Decision.FAIL: 1, Decision.PASS: 2}


def decide_one(p: float, cost: CostMatrix) -> Decision:
    """Pick the minimum-expected-cost action for a single calibrated probability."""
    costs = cost.expected_costs(p)
    return min(costs, key=lambda a: (costs[a], _TIE_PRIORITY[a]))


def decide(probs: np.ndarray, cost: CostMatrix) -> np.ndarray:
    """Vectorised policy: array of P(defective) -> array of Decision."""
    return np.array([decide_one(float(p), cost) for p in np.asarray(probs)],
                    dtype=object)


# --------------------------------------------------------------------------- #
# Naive (calibration-free) baselines — deliberately STRONG so we don't beat a
# strawman: the fixed threshold is the COST-OPTIMAL single threshold, not 0.5.
# --------------------------------------------------------------------------- #

def cost_optimal_threshold(scores: np.ndarray, labels: np.ndarray,
                           cost: CostMatrix,
                           weights: np.ndarray | None = None) -> float:
    """Single raw-score threshold minimising 2-way (PASS/FAIL, no abstention)
    empirical cost: FAIL above the threshold, PASS below.

    Candidates are midpoints between sorted unique scores (plus open ends). Fit
    in-sample on the data passed (a best case for the naive baseline) — see the
    module/decide notes on why this conservative-toward-us choice is deliberate.
    ``weights`` (optional) make the threshold cost-optimal at the target prevalence,
    so the naive comparison is fair when the metrics are reweighted.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    w = np.ones(scores.shape[0]) if weights is None else np.asarray(weights, float)
    uniq = np.unique(scores)
    # Threshold candidates: below all, between each pair, above all.
    mids = (uniq[:-1] + uniq[1:]) / 2.0 if uniq.size > 1 else uniq
    cands = np.concatenate(([uniq[0] - 1.0], mids, [uniq[-1] + 1.0]))

    best_t, best_c = cands[0], np.inf
    for t in cands:
        fail = scores >= t          # FAIL above/at threshold
        # FAIL on a good part = overkill (false_reject); PASS on a bad part = escape.
        overkill = np.sum(w[fail & (labels == 0)]) * cost.false_reject
        escape = np.sum(w[~fail & (labels == 1)]) * cost.false_accept
        c = overkill + escape
        if c < best_c:
            best_c, best_t = c, t
    return float(best_t)


def naive_decide(scores: np.ndarray, threshold: float,
                 margin: float = 0.0) -> np.ndarray:
    """Threshold policy with an optional calibration-free abstention band: ESCALATE
    when |score - threshold| < margin, else FAIL if score >= threshold, else PASS.

    margin=0 recovers the pure no-abstention fixed-threshold baseline.
    """
    scores = np.asarray(scores, dtype=float)
    out = np.empty(scores.shape[0], dtype=object)
    for i, s in enumerate(scores):
        if margin > 0.0 and abs(s - threshold) < margin:
            out[i] = Decision.ESCALATE
        else:
            out[i] = Decision.FAIL if s >= threshold else Decision.PASS
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def decision_metrics(labels: np.ndarray, decisions: np.ndarray, cost: CostMatrix,
                     weights: np.ndarray | None = None,
                     target_prevalence: float | None = None) -> DecisionMetrics:
    """Score a set of decisions against ground truth under the (locked) cost matrix.

    Risk rates are on the auto-decided subset; ``total_cost`` is realized (escalations
    charged the review cost). ``labels``: 0=good, 1=defective.

    ``weights`` (optional) are per-item importance weights; when given, coverage /
    escalation_rate / rates / costs become weighted ratios (see DecisionMetrics for
    the target-prevalence interpretation). Integer counts stay unweighted.
    """
    labels = np.asarray(labels, dtype=int)
    # Normalise to plain string values: Decision is a str-subclass Enum, so numpy
    # object-array `== Decision.PASS` can collapse to a scalar rather than an
    # elementwise mask. Comparing string values is unambiguous.
    vals = np.array([d.value if isinstance(d, Decision) else str(d)
                     for d in decisions])
    n = labels.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    W = float(w.sum())

    is_pass = vals == Decision.PASS.value
    is_fail = vals == Decision.FAIL.value
    is_esc = vals == Decision.ESCALATE.value
    good = labels == 0
    bad = labels == 1
    auto = is_pass | is_fail

    def wsum(mask) -> float:
        return float(w[mask].sum())

    # Per-item realized cost vector under the (locked) matrix.
    cost_vec = np.select(
        [is_esc, is_pass & good, is_pass & bad, is_fail & good, is_fail & bad],
        [cost.escalation, cost.true_pass, cost.false_accept,
         cost.false_reject, cost.true_fail],
        default=0.0,
    )
    total = float((w * cost_vec).sum())

    w_good_auto = wsum(good & auto)
    w_bad_auto = wsum(bad & auto)
    frr = wsum(good & is_fail) / w_good_auto if w_good_auto > 0 else float("nan")
    far = wsum(bad & is_pass) / w_bad_auto if w_bad_auto > 0 else float("nan")

    return DecisionMetrics(
        n=n, n_auto=int(auto.sum()),
        coverage=wsum(auto) / W if W else float("nan"),
        escalation_rate=wsum(is_esc) / W if W else float("nan"),
        false_reject_rate=frr, false_accept_rate=far,
        total_cost=total, cost_per_item=total / W if W else float("nan"),
        n_pass=int(is_pass.sum()), n_fail=int(is_fail.sum()),
        n_escalate=int(is_esc.sum()),
        n_good_auto=int((good & auto).sum()), n_bad_auto=int((bad & auto).sum()),
        n_overkill=int((good & is_fail).sum()), n_escape=int((bad & is_pass).sum()),
        target_prevalence=target_prevalence, weighted=weights is not None,
    )


# --------------------------------------------------------------------------- #
# Degenerate / weak-detector guard
# --------------------------------------------------------------------------- #

# Guard band: STOP only for an essentially signal-free detector. The project's
# intentional baseline is AUROC ~ 0.559 (weak but real), so the stop band's upper
# edge sits below it (stop iff AUROC in [0.47, 0.53], i.e. within 0.03 of chance);
# 0.53-0.60 earns a non-fatal warning instead.
AUROC_STOP_HALFWIDTH = 0.03   # stop band = 0.5 +/- this
AUROC_WARN = 0.60             # below this (but past the stop band) => warn
MIN_STD = 1e-9


def empirical_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC of scores vs binary labels (1=positive=defective)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=float)
    ranks[order] = np.arange(1, scores.shape[0] + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[labels == 1].sum()
    auc = (r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


class DegenerateScoresError(ValueError):
    """Raised when input scores carry no usable signal for calibration."""


def check_not_degenerate(scores: np.ndarray, labels: np.ndarray, *,
                         k: int = 10) -> list[str]:
    """Stop loudly on signal-free input; return non-fatal warnings otherwise.

    Stops if: near-zero score variance; AUROC within [1-AUROC_STOP, AUROC_STOP]
    of a coin flip; a class missing or too small to K-fold. Warns (returns a
    message, does not raise) on a weak-but-real detector.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = scores.shape[0]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    if n_pos == 0 or n_neg == 0:
        raise DegenerateScoresError(
            f"Need both classes present (good={n_neg}, defective={n_pos}). "
            "Provide a labelled run with normal AND defective parts."
        )
    if float(np.std(scores)) < MIN_STD:
        raise DegenerateScoresError(
            f"Score variance is ~0 (std={np.std(scores):.2e}). The detector emitted "
            "constant scores — calibration would be meaningless. Re-run a real "
            "(non-degenerate) baseline before adjudication."
        )
    auc = empirical_auroc(scores, labels)
    if abs(auc - 0.5) <= AUROC_STOP_HALFWIDTH:  # auc in [0.47, 0.53]
        lo, hi = 0.5 - AUROC_STOP_HALFWIDTH, 0.5 + AUROC_STOP_HALFWIDTH
        raise DegenerateScoresError(
            f"Detector image-AUROC={auc:.3f} is indistinguishable from chance "
            f"(stop band [{lo:.2f}, {hi:.2f}]). Calibration/"
            "decision numbers would be noise. Train a non-degenerate baseline first."
        )
    smaller = min(n_pos, n_neg)
    if smaller < k:
        raise DegenerateScoresError(
            f"Smallest class has {smaller} items but k={k} folds requested; "
            f"cannot stratify. Lower k (<= {smaller}) or supply more data."
        )

    warnings: list[str] = []
    if auc < AUROC_WARN:
        warnings.append(
            f"WEAK detector: image-AUROC={auc:.3f} (< {AUROC_WARN:.2f}). This is the "
            "intended Phase-1 'untrustworthy detector' regime; numbers are real but "
            "the calibration carries wide error bars — interpret accordingly."
        )
    if smaller < 2 * k:
        warnings.append(
            f"SMALL calibration sample: smallest class n={smaller}. Conformal "
            "probabilities are coarse; treat headline rates as indicative."
        )
    return warnings


# --------------------------------------------------------------------------- #
# Risk-coverage sweeps
# --------------------------------------------------------------------------- #

# Above this escalation cost the abstention band vanishes (the policy reduces to the
# 2-way calibrated cost-optimal threshold); for the locked 10/3/1 matrix it is 30/13.
def escalation_cost_ceiling(cost: CostMatrix) -> float:
    """Escalation cost at/above which ESCALATE is never the argmin (coverage = 1)."""
    fa, fr = cost.false_accept, cost.false_reject
    return fa * fr / (fa + fr)


def risk_coverage_ours(probs: np.ndarray, labels: np.ndarray, cost: CostMatrix,
                       c_esc_grid: np.ndarray,
                       weights: np.ndarray | None = None) -> list[dict]:
    """Sweep the policy's escalation cost (abstention aggressiveness) over calibrated
    probabilities; ACCOUNT realized cost with the locked matrix at every point.

    ``probs`` should already be at the analysis prevalence (prior-shifted if a target
    prevalence is in effect); ``weights`` reweight the metrics to that prevalence."""
    rows = []
    for c in c_esc_grid:
        policy = CostMatrix(false_accept=cost.false_accept,
                            false_reject=cost.false_reject,
                            escalation=float(c),
                            true_pass=cost.true_pass, true_fail=cost.true_fail)
        m = decision_metrics(labels, decide(probs, policy), cost, weights=weights)
        rows.append({"policy": "ours", "knob": float(c), "coverage": m.coverage,
                     "escalation_rate": m.escalation_rate,
                     "false_reject_rate": m.false_reject_rate,
                     "false_accept_rate": m.false_accept_rate,
                     "cost_per_item": m.cost_per_item})
    return rows


def risk_coverage_naive(scores: np.ndarray, labels: np.ndarray, threshold: float,
                        cost: CostMatrix, margin_grid: np.ndarray,
                        weights: np.ndarray | None = None) -> list[dict]:
    """Calibration-free margin curve: widen an abstention band around the fixed
    cost-optimal threshold; margin=0 is the no-abstention point."""
    rows = []
    for w in margin_grid:
        m = decision_metrics(labels, naive_decide(scores, threshold, float(w)), cost,
                             weights=weights)
        rows.append({"policy": "naive", "knob": float(w), "coverage": m.coverage,
                     "escalation_rate": m.escalation_rate,
                     "false_reject_rate": m.false_reject_rate,
                     "false_accept_rate": m.false_accept_rate,
                     "cost_per_item": m.cost_per_item})
    return rows


def breakeven_review_cost(probs: np.ndarray, labels: np.ndarray, cost_fa: float,
                          cost_fr: float, naive_cost_per_item: float, *,
                          weights: np.ndarray | None = None, grid: int = 121
                          ) -> tuple[list[dict], float]:
    """Find the review cost below which cost-aware abstention beats a tuned threshold.

    Distinct from ``risk_coverage_ours`` (which fixes the true review cost and sweeps
    abstention aggressiveness): here the swept value IS the true review cost ``c`` — it
    drives BOTH the policy (escalate when cheapest) AND the realized accounting. The
    escalation-free naive baseline's cost does not depend on ``c`` (it never escalates),
    so we sweep ``c`` and find the largest one at which OURS' total cost/item still
    beats ``naive_cost_per_item``.

    As ``c`` -> 0 escalation is free, so OURS MUST win (it escalates every item where
    deciding risks more than 0); if it does not, that is a bug. As ``c`` rises past the
    ceiling ``fa*fr/(fa+fr)`` the abstention band vanishes and OURS -> its 2-way
    threshold. Returns (rows, break_even_cost). rows: review_cost, ours_cost,
    escalation_rate, beats_naive.
    """
    ceil = cost_fa * cost_fr / (cost_fa + cost_fr)
    rows = []
    for c in np.linspace(0.0, ceil * 1.05, grid):
        m = CostMatrix(false_accept=cost_fa, false_reject=cost_fr, escalation=float(c))
        dm = decision_metrics(labels, decide(probs, m), m, weights=weights)
        rows.append({"review_cost": float(c), "ours_cost": dm.cost_per_item,
                     "escalation_rate": dm.escalation_rate,
                     "beats_naive": bool(dm.cost_per_item <= naive_cost_per_item + 1e-12)})
    # Break-even = the end of the CONTIGUOUS winning region from c=0. Scanning from 0,
    # OURS wins cheaply (escalation ~free) and loses as review rises; we take the last c
    # before the first STRICT loss. (A naive max-over-all-wins is wrong: at very high c
    # OURS collapses to its 2-way threshold, which can TIE naive — e.g. both FAIL-all on
    # a weak detector — re-entering the "wins" set far from the meaningful crossing.)
    c_star = rows[-1]["review_cost"]
    for i, r in enumerate(rows):
        if r["ours_cost"] > naive_cost_per_item + 1e-12:   # first strict loss
            c_star = rows[i - 1]["review_cost"] if i > 0 else 0.0
            break
    return rows, c_star


# --------------------------------------------------------------------------- #
# Prevalence correction (label shift): the benchmark test split is ~74% defective;
# production lines run a LOW defect rate. Under the label-shift assumption (the
# class-conditional score densities p(score|y) are the same in the benchmark and in
# production; only the class priors differ), we (a) shift the calibrated
# probabilities to the target prior, and (b) importance-weight the metrics. Both use
# the SAME (target/source) prior ratios, so calibration and evaluation stay consistent.
# --------------------------------------------------------------------------- #

def prior_shift(p_source: np.ndarray, pi_source: float,
                pi_target: float) -> np.ndarray:
    """Re-calibrate P(defective|score) from a source prior to a target prior.

    Standard prior-correction (Saerens et al. / Elkan): under label shift the
    likelihood ratio is invariant, so the odds scale by (pi_t/pi_s)*((1-pi_s)/(1-pi_t)):

        p_t = a*p_s / (a*p_s + b*(1-p_s)),  a = pi_t/pi_s,  b = (1-pi_t)/(1-pi_s).

    Probabilities calibrated at pi_source stay calibrated at pi_target.
    """
    p = np.asarray(p_source, dtype=float)
    a = pi_target / pi_source
    b = (1.0 - pi_target) / (1.0 - pi_source)
    num = a * p
    return num / (num + b * (1.0 - p))


def prevalence_weights(labels: np.ndarray, pi_target: float,
                       pi_source: float | None = None) -> np.ndarray:
    """Per-item importance weights P_target(y)/P_source(y) (label shift): defective
    items get pi_t/pi_s, good items get (1-pi_t)/(1-pi_s). A weighted average over the
    sample then estimates the corresponding quantity at the target prevalence."""
    labels = np.asarray(labels, dtype=int)
    if pi_source is None:
        pi_source = float((labels == 1).mean())
    return np.where(labels == 1, pi_target / pi_source,
                    (1.0 - pi_target) / (1.0 - pi_source)).astype(float)


# --------------------------------------------------------------------------- #
# Synthetic score generator — for the LABELED machinery-validation only (proves the
# code is correct on a detector that actually separates; NOT real-data evidence).
# --------------------------------------------------------------------------- #

def simulate_scores(n: int, auroc: float, prevalence: float,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two unit-variance Gaussians whose separation d = sqrt(2)*Phi^-1(auroc) yields
    the requested image-AUROC, at the requested defect prevalence. Returns (scores,
    labels). Used only by the synthetic validation harness."""
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < prevalence).astype(int)
    d = np.sqrt(2.0) * norm.ppf(auroc)
    scores = rng.normal(0.0, 1.0, n) + d * labels
    return scores, labels
