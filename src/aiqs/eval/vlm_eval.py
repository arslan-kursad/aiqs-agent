"""Phase-2A eval — the HEART of the VLM second-look. Pure numpy; no torch/API.

Order is deliberate: MECHANISM before economics.
  (a) VLM raw accuracy + confusion matrix on the bucket vs ground truth.
  (b) ERROR-INDEPENDENCE vs the detector — the REAL test. A second look only adds value
      if it brings INDEPENDENT signal: does the VLM get it right WHERE THE DETECTOR IS
      WRONG, with errors decorrelated from the detector's? "detector wrong" is PINNED to
      the Phase-1 naive cost-optimal fixed-threshold decision (the honest "no-adjudication
      -layer" comparator), NOT an ESCALATE-free argmin on calibrated p.
      A PRE-REGISTERED rule (constants below, committed before the run) gates the claim:
        load-bearing (CI-guarded): Wilson-lower-CI[P(VLM correct|detector wrong)] > 0.50
        corroborating (secondary): kappa(error vectors) < 0.20, reported WITH a bootstrap
          CI; kappa NEVER carries the claim alone (its CI is wide at small n).
      independent <=> load-bearing AND kappa_point<KAPPA_MAX; load-bearing only =>
      "better but redundant"; load-bearing fails => "theatre" (cost without signal).
  (c) Bidirectional value, separately: goods RESCUED to PASS (overkill down) and
      defectives correctly FAIL'd (escape down).
  (d) New effective review cost (token + VLM-error + human-on-abstention) and the
      break-even SHIFT vs the Phase-1 baseline, as a BAND over the shrinkage knob,
      LED from the conservative end. Reported under both 10/3/1 and 100/3/1.
  (e) Nondeterminism: the rule is recomputed on EACH of K runs; the headline is the MODAL
      outcome with its K-stability ("YES in 5/5"), never a single-run yes/no.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aiqs.eval.decision import (
    CostMatrix,
    Decision,
    cost_optimal_threshold,
    naive_decide,
)

# --- PRE-REGISTERED decision-rule constants (committed BEFORE looking at any run) ---- #
P_IND_MIN = 0.50    # load-bearing: Wilson-lower-CI[P(VLM correct|detector wrong)] must exceed this
KAPPA_MAX = 0.20    # corroborating only: error-vector kappa below this
RUN_K = 5           # default repeats for nondeterminism / rule-stability
# Phase-1 native-74 break-even review cost on the PatchCore run (see CLAUDE.md).
PHASE1_BREAKEVEN_NATIVE = 0.868
# Degeneracy guard (added 2026-07-06, BEFORE the sonnet-4-6 headline run — see CLAUDE.md):
# a rubber-stamp model (one verdict on >= this fraction of a run's calls) can satisfy the
# Wilson-lo>P_IND_MIN test by sheer luck of being "right" on whichever side the detector
# over-rejects, producing a SPURIOUS "independent" claim. Any run at/above this fraction is
# forced to "invalid-degenerate" regardless of p_ind/kappa — the independence claim is not
# meaningful there. Threshold frozen pre-registration-style before the headline run exists.
DEGENERATE_VERDICT_FRAC = 0.95


# --------------------------------------------------------------------------- #
# Small statistics (kept local; no scipy dependency)
# --------------------------------------------------------------------------- #

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n. (0.5, 0.5) if n == 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two binary vectors. nan if undefined (degenerate margins)."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = a.shape[0]
    if n == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pa1, pb1 = float(np.mean(a)), float(np.mean(b))
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def bootstrap_ci(fn, *arrays, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic ``fn(*resampled_arrays)``."""
    arrays = [np.asarray(x) for x in arrays]
    n = arrays[0].shape[0]
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        val = fn(*[x[idx] for x in arrays])
        if not np.isnan(val):
            stats.append(val)
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


# --------------------------------------------------------------------------- #
# Pinned "detector wrong" + VLM call mapping
# --------------------------------------------------------------------------- #

def detector_hard_decision(scores: np.ndarray, labels: np.ndarray,
                           cost: CostMatrix) -> np.ndarray:
    """The detector's counterfactual hard call WITHOUT an adjudication layer: the
    Phase-1 naive cost-optimal fixed threshold (FAIL>=t* else PASS). Returns 1=FAIL/
    defect-call, 0=PASS/clean-call. This is the PINNED basis of "detector wrong"."""
    t = cost_optimal_threshold(scores, labels, cost)
    dec = naive_decide(scores, t, 0.0)
    return np.array([1 if d is Decision.FAIL else 0 for d in dec], dtype=int)


def vlm_call_label(verdict: str, p_vlm: float) -> int:
    """Map a VLM verdict to a binary defect-call. unsure -> its implied lean (p_vlm>=0.5)."""
    if verdict == "defect":
        return 1
    if verdict == "clean":
        return 0
    return int(p_vlm >= 0.5)


# --------------------------------------------------------------------------- #
# Per-run + per-(run, lambda) evaluation
# --------------------------------------------------------------------------- #

@dataclass
class RuleOutcome:
    p_ind_point: float
    p_ind_lo: float
    p_ind_hi: float
    n_detector_wrong: int
    kappa: float
    kappa_lo: float
    kappa_hi: float
    label: str               # "independent" | "redundant" | "theatre" | "invalid-degenerate"

    @property
    def is_independent(self) -> bool:
        return self.label == "independent"


def _rule_label(p_ind_lo: float, kappa: float) -> str:
    load_bearing = (not np.isnan(p_ind_lo)) and p_ind_lo > P_IND_MIN
    corroborating = (not np.isnan(kappa)) and kappa < KAPPA_MAX
    if load_bearing and corroborating:
        return "independent"
    if load_bearing:
        return "redundant"   # better-than-detector but errors correlated
    return "theatre"         # no independent productive signal


def is_degenerate(raw_verdicts, threshold: float = DEGENERATE_VERDICT_FRAC) -> bool:
    """True if one verdict value covers >= ``threshold`` of the run (a rubber stamp).

    Operates on the RAW 3-way verdict ("defect"/"clean"/"unsure"), not the binary
    detector-comparison call — an all-"unsure" run is degenerate too, even though
    ``vlm_call_label`` would arbitrarily lean each item and hide that in the binary vector.
    """
    if not raw_verdicts:
        return False
    n = len(raw_verdicts)
    top = max(raw_verdicts.count(v) for v in set(raw_verdicts))
    return (top / n) >= threshold


def error_independence(vlm_call: np.ndarray, det_call: np.ndarray, labels: np.ndarray,
                       *, seed: int = 0, raw_verdicts=None) -> RuleOutcome:
    """Compute the pre-registered rule inputs on ONE run's bucket.

    vlm_err/det_err are 1 where each is wrong vs ground truth. The load-bearing quantity
    is P(VLM correct | detector wrong) with a Wilson lower bound; kappa(err vectors) is
    secondary, with a bootstrap CI.

    ``raw_verdicts`` (optional, the raw 3-way verdict per item) feeds the DEGENERACY
    guard: a rubber-stamp run (>= DEGENERATE_VERDICT_FRAC one verdict) is forced to
    "invalid-degenerate" regardless of the p_ind/kappa numbers — see the constant's
    docstring. Passing ``None`` (the pre-guard call shape) skips the check, so this stays
    backward compatible with any caller that has not been updated to pass verdicts.
    """
    labels = np.asarray(labels, dtype=int)
    vlm_err = (vlm_call != labels).astype(int)
    det_err = (det_call != labels).astype(int)

    det_wrong = det_err == 1
    n_dw = int(det_wrong.sum())
    k_correct = int(((vlm_err == 0) & det_wrong).sum())
    point = (k_correct / n_dw) if n_dw else float("nan")
    lo, hi = wilson_interval(k_correct, n_dw)

    kappa = cohens_kappa(vlm_err, det_err)
    k_lo, k_hi = bootstrap_ci(cohens_kappa, vlm_err, det_err, seed=seed)

    label = "invalid-degenerate" if is_degenerate(raw_verdicts) else _rule_label(lo, kappa)
    return RuleOutcome(point, lo, hi, n_dw, kappa, k_lo, k_hi, label)


def bidirectional_value(states) -> dict:
    """Goods RESCUED to PASS (overkill down) and defectives correctly FAIL'd (escape down),
    counted on AUTO-decided items only (abstentions go to a human and are neutral here)."""
    rescued = wrong_pass = correct_fail = wrong_fail = 0
    for s in states:
        if s.abstained or s.final_decision is Decision.ESCALATE:
            continue
        if s.final_decision is Decision.PASS:
            if s.label == 0:
                rescued += 1          # good correctly passed (overkill avoided)
            else:
                wrong_pass += 1       # defect shipped (escape) — VLM error
        elif s.final_decision is Decision.FAIL:
            if s.label == 1:
                correct_fail += 1     # defect correctly failed
            else:
                wrong_fail += 1       # good failed (overkill) — VLM error
    return {"rescued_to_pass": rescued, "wrong_pass_escape": wrong_pass,
            "correct_fail": correct_fail, "wrong_fail_overkill": wrong_fail}


def effective_review_cost(states, cost: CostMatrix, token_cost: float) -> float:
    """Per-bucket-item effective review cost with the VLM layer:
    (token cost on every item) + (error cost of wrong auto-decides) + (human review on
    each abstention). Compare to the bare human review cost (cost.escalation) and to the
    Phase-1 break-even: below the bare review cost => the layer pushes break-even RIGHT.
    """
    n = len(states)
    if n == 0:
        return float("nan")
    bd = bidirectional_value(states)
    error_cost = bd["wrong_pass_escape"] * cost.false_accept \
        + bd["wrong_fail_overkill"] * cost.false_reject
    n_abstain = sum(1 for s in states if s.abstained)
    return (token_cost * n + error_cost + cost.escalation * n_abstain) / n


def confusion_matrix(states) -> dict:
    """3x2 verdict (defect/clean/unsure) x ground-truth (good/defective) counts + accuracy.
    Accuracy treats unsure as a non-answer (incorrect)."""
    cells = {v: {0: 0, 1: 0} for v in ("defect", "clean", "unsure")}
    correct = 0
    for s in states:
        cells[s.vlm_verdict][s.label] += 1
        if (s.vlm_verdict == "defect" and s.label == 1) or \
           (s.vlm_verdict == "clean" and s.label == 0):
            correct += 1
    n = len(states)
    return {"cells": cells, "accuracy": (correct / n if n else float("nan")), "n": n}


# --------------------------------------------------------------------------- #
# Top-level: K runs x lambda grid
# --------------------------------------------------------------------------- #

@dataclass
class VLMEval:
    n_bucket: int
    n_good: int
    n_defective: int
    k_runs: int
    token_cost: float
    cost_label: str
    # mechanism (per run, then summarised)
    accuracy_mean: float
    accuracy_runs: list[float]
    confusion_modal: dict
    # error-independence rule across K runs
    rule_outcomes: list[RuleOutcome]
    rule_distribution: dict          # {"independent": k, "redundant": k, "theatre": k,
                                      #  "invalid-degenerate": k}
    rule_modal: str
    rule_stability: str              # e.g. "YES in 5/5 runs"
    # bidirectional value (mean over runs)
    bidirectional: dict
    # break-even shift band over lambda (conservative-led)
    lambda_grid: list[float]
    eff_review_cost_band: list[float]   # mean over runs, per lambda
    breakeven_shifts_right: bool
    warnings: list[str] = field(default_factory=list)


def evaluate(states_per_run: list[list], scores: np.ndarray, labels_bucket: np.ndarray,
             det_call_bucket: np.ndarray, cost: CostMatrix, *,
             cost_label: str, token_cost: float, lambda_grid: list[float],
             warnings: list[str] | None = None, seed: int = 0) -> VLMEval:
    """Aggregate K runs (each a list of adjudicated VLMState for the bucket).

    ``states_per_run`` are evaluated at lam=0 for the mechanism/rule headline; the
    break-even band re-derives decisions across ``lambda_grid`` from the per-run verdicts.
    ``det_call_bucket`` is the pinned detector hard-call (1/0) for the bucket items.
    """
    k = len(states_per_run)
    labels_bucket = np.asarray(labels_bucket, dtype=int)

    # (a) accuracy + confusion, per run
    acc_runs, confusions = [], []
    for states in states_per_run:
        cm = confusion_matrix(states)
        acc_runs.append(cm["accuracy"])
        confusions.append(cm)

    # (b) error-independence rule, recomputed PER RUN
    rule_outcomes = []
    for r, states in enumerate(states_per_run):
        vlm_call = np.array([vlm_call_label(s.vlm_verdict, s.p_vlm) for s in states],
                            dtype=int)
        raw_verdicts = [s.vlm_verdict for s in states]
        rule_outcomes.append(
            error_independence(vlm_call, det_call_bucket, labels_bucket, seed=seed + r,
                               raw_verdicts=raw_verdicts))
    dist = {lab: sum(o.label == lab for o in rule_outcomes)
            for lab in ("independent", "redundant", "theatre", "invalid-degenerate")}
    modal = max(dist, key=dist.get)
    n_indep = dist["independent"]
    # A tie at 0 (e.g. all runs "invalid-degenerate") must NOT read as "independent" just
    # because max() breaks ties by first-seen key — gate YES on independent being a UNIQUE,
    # NONZERO max, not merely equal to whatever max() picked.
    is_yes = n_indep > 0 and n_indep == max(dist.values())
    rule_stability = f"{'YES' if is_yes else 'NO'}: independent in {n_indep}/{k} runs"
    if dist["invalid-degenerate"]:
        rule_stability += (f" (WARNING: {dist['invalid-degenerate']}/{k} runs "
                           "INVALID-DEGENERATE — single verdict >=95%, independence claim "
                           "not meaningful there)")

    # (c) bidirectional value, mean over runs (computed at the verdicts' lam=0 decisions)
    bd_keys = ("rescued_to_pass", "wrong_pass_escape", "correct_fail", "wrong_fail_overkill")
    bd_sum = {key: 0.0 for key in bd_keys}
    for states in states_per_run:
        bd = bidirectional_value(states)
        for key in bd_keys:
            bd_sum[key] += bd[key]
    bidirectional = {key: bd_sum[key] / k for key in bd_keys}

    # (d) break-even band over lambda (re-decide from stored verdicts), mean over runs
    from aiqs.vlm.abstain import adjudicate_probability, confidence_to_p

    band = []
    for lam in lambda_grid:
        per_run_cost = []
        for states in states_per_run:
            relam = []
            for s in states:
                p = confidence_to_p(s.vlm_verdict, s.vlm_conf, lam)
                dec = adjudicate_probability(p, cost)
                relam.append(_Readout(label=s.label, final_decision=dec,
                                      abstained=dec is Decision.ESCALATE))
            per_run_cost.append(effective_review_cost(relam, cost, token_cost))
        band.append(float(np.mean(per_run_cost)))
    # conservative-led: band is reported high-lambda first by the caller; "shifts right"
    # if ANY swept point beats the bare human review cost (escalation).
    shifts_right = any(c < cost.escalation for c in band if not np.isnan(c))

    return VLMEval(
        n_bucket=len(labels_bucket), n_good=int((labels_bucket == 0).sum()),
        n_defective=int((labels_bucket == 1).sum()), k_runs=k, token_cost=token_cost,
        cost_label=cost_label, accuracy_mean=float(np.mean(acc_runs)),
        accuracy_runs=acc_runs, confusion_modal=confusions[0],
        rule_outcomes=rule_outcomes, rule_distribution=dist, rule_modal=modal,
        rule_stability=rule_stability, bidirectional=bidirectional,
        lambda_grid=list(lambda_grid), eff_review_cost_band=band,
        breakeven_shifts_right=shifts_right, warnings=warnings or [])


@dataclass
class _Readout:
    """Minimal stand-in carrying just what cost/value functions read (re-decided lam)."""
    label: int
    final_decision: Decision
    abstained: bool
