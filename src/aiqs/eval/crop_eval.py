"""Phase-2B Stage-3 eval — the two-arm (full vs full+crop) comparison layer.

EXTENDS ``aiqs.eval.vlm_eval`` (each arm is scored by the same 2A ``evaluate``; nothing is
forked). What this module adds is the PAIRED comparison the Stage-3 hypothesis needs:

  (1) escape rate per arm (headline: does the crop reduce it?), per run + mean,
      with the paired McNemar-style discordant counts (escaped-in-A-only = crop FIXED it;
      escaped-in-B-only = crop BROKE it) — same items, same runs, so the difference is
      the crop, not sampling.
  (2) PRE-REGISTERED escape-reasoning classification (``vlm.reasoning_rules``, committed
      before the run): PERCEPTION / SEMANTIC / UNCLASSIFIED (+ diffuse-excluded).
  (3) stable-vs-flip: is an ARM-A escape wrong in ALL K runs (stable-wrong: K-agreement
      would NOT catch it) or does it flip across runs (K-agreement is a usable abstain
      signal)? Pre-registered definition: stable := escaped in every one of the K runs.
  (4) 2A-mechanism replication line (DESCRIPTIVE, no thresholds): does ARM-A on the new
      ground reproduce the 2A mechanism observations — goods get rescued, defects escape
      (lenient bias), self-reported confidence fails to separate correct from wrong
      (AUC ~ 0.5)? Framed qualitatively; the discarded-fork numbers are never the baseline.
  (5) token cost per call, per arm (the crop's 2nd-image increment, measured).

Pure numpy; no torch/API. Everything here is testable with scripted mocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aiqs.eval.decision import (
    CostMatrix,
    Decision,
    cost_optimal_threshold,
    decide_one,
    decision_metrics,
    naive_decide,
    prevalence_weights,
)
from aiqs.eval import vlm_eval as ve
from aiqs.vlm import reasoning_rules as rr

# The COST VERDICT matrices (the project's north star: optimise a business cost function,
# not a detection metric). LOCKED = the benchmark's illustrative 10/3/1; REALISTIC = the
# escape-dominant 100/3/1 (shipping a defect >> a re-inspection). See CLAUDE.md.
_LOCKED = CostMatrix(false_accept=10.0, false_reject=3.0, escalation=1.0)
_REALISTIC = CostMatrix(false_accept=100.0, false_reject=3.0, escalation=1.0)
_TARGET_PREVALENCE = 0.02


def _is_escape(state) -> bool:
    """ARM escape := bucket DEFECTIVE auto-PASSed by the VLM layer (label=1, final PASS)."""
    return state.label == 1 and state.final_decision is Decision.PASS


def escape_matrix(states_per_run: list[list]) -> np.ndarray:
    """(K, n_items) boolean matrix of escapes. Item order must match across runs/arms."""
    return np.array([[_is_escape(s) for s in states] for states in states_per_run],
                    dtype=bool)


def paired_escape_comparison(esc_a: np.ndarray, esc_b: np.ndarray,
                             n_defective: int) -> dict:
    """Per-run escape rates + paired discordant counts (summed over runs)."""
    rate_a = esc_a.sum(axis=1) / n_defective if n_defective else np.full(esc_a.shape[0], np.nan)
    rate_b = esc_b.sum(axis=1) / n_defective if n_defective else np.full(esc_b.shape[0], np.nan)
    return {
        "escape_rate_a_runs": [float(x) for x in rate_a],
        "escape_rate_b_runs": [float(x) for x in rate_b],
        "escape_rate_a_mean": float(np.mean(rate_a)),
        "escape_rate_b_mean": float(np.mean(rate_b)),
        "delta_mean": float(np.mean(rate_a) - np.mean(rate_b)),   # >0 => crop reduces escapes
        "fixed_by_crop": int((esc_a & ~esc_b).sum()),              # (run,item) escaped A only
        "broken_by_crop": int((~esc_a & esc_b).sum()),             # (run,item) escaped B only
    }


def classify_a_escapes(states_a_per_run: list[list], states_b_per_run: list[list],
                       diffuse_by_item: list[bool]) -> dict:
    """Pre-registered classification of every (run, item) ARM-A escape via its ARM-B pair."""
    labels = []
    for states_a, states_b in zip(states_a_per_run, states_b_per_run):
        for i, s_a in enumerate(states_a):
            if _is_escape(s_a):
                s_b = states_b[i]
                labels.append(rr.classify_escape(
                    s_b.vlm_verdict, s_b.vlm_reasoning, diffuse=diffuse_by_item[i]))
    return rr.distribution(labels)


def stable_vs_flip(esc_a: np.ndarray) -> dict:
    """Among items that EVER escaped in ARM-A: stable-wrong (escaped in ALL K runs,
    pre-registered) vs flipping. Stable-wrong escapes are invisible to K-run agreement."""
    ever = esc_a.any(axis=0)
    stable = esc_a.all(axis=0)
    n_ever, n_stable = int(ever.sum()), int(stable.sum())
    return {"escaped_ever": n_ever, "stable_wrong": n_stable,
            "flipping": n_ever - n_stable,
            "stable_fraction": (n_stable / n_ever) if n_ever else float("nan")}


def replication_2a(states_per_run: list[list]) -> dict:
    """DESCRIPTIVE 2A-mechanism check on ARM-A (the full-image arm IS a 2A replication on
    new ground): good-rescue rate, defect-escape rate, and confidence-separation AUC
    (P(conf_correct > conf_wrong); ~0.5 = confidence does not separate — the 2A observation)."""
    rescue_rates, escape_rates = [], []
    conf_correct, conf_wrong = [], []
    for states in states_per_run:
        n_good = sum(1 for s in states if s.label == 0)
        n_def = sum(1 for s in states if s.label == 1)
        bd = ve.bidirectional_value(states)
        rescue_rates.append(bd["rescued_to_pass"] / n_good if n_good else float("nan"))
        escape_rates.append(bd["wrong_pass_escape"] / n_def if n_def else float("nan"))
        for s in states:
            if s.vlm_verdict == "unsure" or s.vlm_conf is None:
                continue
            correct = (s.vlm_verdict == "defect") == (s.label == 1)
            (conf_correct if correct else conf_wrong).append(s.vlm_conf)
    if conf_correct and conf_wrong:
        a, b = np.asarray(conf_correct), np.asarray(conf_wrong)
        auc = float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
    else:
        auc = float("nan")
    return {"good_rescue_rate_mean": float(np.nanmean(rescue_rates)),
            "defect_escape_rate_mean": float(np.nanmean(escape_rates)),
            "confidence_separation_auc": auc}


def _arm_cost_per_item(states_per_run, matrix: CostMatrix, target) -> float:
    """Mean (over K runs) realized cost/item of ONE arm's bucket decisions, re-decided
    from the stored provisional p_vlm under ``matrix`` and importance-weighted to ``target``
    prevalence (None => the bucket's native prevalence). Reuses eval.decision entirely —
    the SAME machinery that scores the Phase-1 policy, now inherited by the VLM layer."""
    costs = []
    for states in states_per_run:
        labels = np.array([s.label for s in states], dtype=int)
        decisions = [decide_one(s.p_vlm, matrix) for s in states]     # plain list: keep enums
        w = None if target is None else prevalence_weights(labels, target, labels.mean())
        m = decision_metrics(labels, decisions, matrix, weights=w, target_prevalence=target)
        costs.append(m.cost_per_item)
    return float(np.mean(costs))


def _naive_cost_per_item(states_run0, matrix: CostMatrix, target) -> float:
    """The 'no VLM layer' comparator: the detector's cost-optimal fixed-threshold call on
    the SAME bucket, under ``matrix`` at ``target`` prevalence."""
    labels = np.array([s.label for s in states_run0], dtype=int)
    scores = np.array([s.detector_score for s in states_run0], dtype=float)
    w = None if target is None else prevalence_weights(labels, target, labels.mean())
    thr = cost_optimal_threshold(scores, labels, matrix, weights=w)
    dec = naive_decide(scores, thr, 0.0)
    return decision_metrics(labels, dec, matrix, weights=w,
                            target_prevalence=target).cost_per_item


def cost_regimes(states_a_per_run, states_b_per_run) -> list[dict]:
    """THE COST VERDICT — the closing move, zero API. Per (matrix x prevalence) cell, the
    realized cost/item of NAIVE (no VLM) / ARM-A (full VLM) / ARM-B (full+crop) plus the
    human-review floor (escalation cost = review everything). Answers the north-star
    question the escape-rate headline cannot: in WHICH cost regime is the crop worth it?

    The crop is a recall lever bought with overkill, so its verdict is prevalence-conditional
    — the VLM layer inherits Phase-1's operating-envelope logic (no universal win; it says
    which regime you are in)."""
    rows = []
    for mname, matrix in (("10/3/1", _LOCKED), ("100/3/1", _REALISTIC)):
        for pname, target in (("native", None), ("2%", _TARGET_PREVALENCE)):
            naive = _naive_cost_per_item(states_a_per_run[0], matrix, target)
            a = _arm_cost_per_item(states_a_per_run, matrix, target)
            b = _arm_cost_per_item(states_b_per_run, matrix, target)
            floor = matrix.escalation
            options = {"naive": naive, "ARM-A": a, "ARM-B": b, "human-review": floor}
            best = min(options, key=options.get)
            rows.append({"matrix": mname, "prevalence": pname, "naive": naive,
                        "arm_a": a, "arm_b": b, "human_floor": floor, "best": best,
                        "crop_vs_full": ("crop-wins" if b < a - 1e-9 else
                                         "crop-loses" if b > a + 1e-9 else "tie")})
    return rows


def token_cost_line(states_per_run: list[list]) -> dict:
    """Mean input/output tokens per call for one arm (None-safe; nan when mock)."""
    ti = [s.tokens_in for states in states_per_run for s in states if s.tokens_in is not None]
    to = [s.tokens_out for states in states_per_run for s in states if s.tokens_out is not None]
    return {"tokens_in_mean": float(np.mean(ti)) if ti else float("nan"),
            "tokens_out_mean": float(np.mean(to)) if to else float("nan"),
            "n_calls_with_usage": len(ti)}


@dataclass
class CropExperiment:
    arm_a: ve.VLMEval                 # full 2A eval of the full-image arm
    arm_b: ve.VLMEval                 # ... of the full+crop arm
    paired: dict                       # (1) escape rates + discordant counts
    classification: dict               # (2) pre-registered PERCEPTION/SEMANTIC/UNCLASSIFIED
    stability: dict                    # (3) stable-vs-flip on ARM-A escapes
    replication: dict                  # (4) descriptive 2A-mechanism line (ARM-A)
    tokens_a: dict                     # (5) per-arm token cost
    tokens_b: dict
    n_diffuse: int                     # items with no crop (ARM-B byte-identical to ARM-A)
    cost_regimes: list[dict] = field(default_factory=list)  # (6) THE COST VERDICT
    warnings: list[str] = field(default_factory=list)


def evaluate_two_arm(states_a_per_run: list[list], states_b_per_run: list[list],
                     bucket_scores: np.ndarray, bucket_labels: np.ndarray,
                     det_call_bucket: np.ndarray, cost, *, token_cost: float,
                     lambda_grid: list[float], diffuse_by_item: list[bool],
                     warnings: list[str] | None = None, seed: int = 0) -> CropExperiment:
    """Score each arm with the UNCHANGED 2A evaluate, then layer the paired comparison."""
    bucket_labels = np.asarray(bucket_labels, dtype=int)
    n_def = int((bucket_labels == 1).sum())

    arm_a = ve.evaluate(states_a_per_run, bucket_scores, bucket_labels, det_call_bucket,
                        cost, cost_label="10/3/1", token_cost=token_cost,
                        lambda_grid=lambda_grid, warnings=warnings, seed=seed)
    arm_b = ve.evaluate(states_b_per_run, bucket_scores, bucket_labels, det_call_bucket,
                        cost, cost_label="10/3/1", token_cost=token_cost,
                        lambda_grid=lambda_grid, warnings=warnings, seed=seed)

    esc_a = escape_matrix(states_a_per_run)
    esc_b = escape_matrix(states_b_per_run)
    return CropExperiment(
        arm_a=arm_a, arm_b=arm_b,
        paired=paired_escape_comparison(esc_a, esc_b, n_def),
        classification=classify_a_escapes(states_a_per_run, states_b_per_run, diffuse_by_item),
        stability=stable_vs_flip(esc_a),
        replication=replication_2a(states_a_per_run),
        tokens_a=token_cost_line(states_a_per_run),
        tokens_b=token_cost_line(states_b_per_run),
        n_diffuse=int(sum(diffuse_by_item)),
        cost_regimes=cost_regimes(states_a_per_run, states_b_per_run),
        warnings=warnings or [])
