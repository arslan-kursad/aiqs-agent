"""Parity guards (mandated before the graph could be trusted): the graph's node
decomposition must never silently drift from the existing Phase-1/2A pure functions
it wraps. Two independent checks:

1. ``vlm_second_look`` + ``vlm_abstain_rule`` composed directly (no graph/checkpointer)
   must reproduce ``aiqs.vlm.adjudicate.adjudicate`` byte-for-byte on the identical
   VLMState/backend/cost/lam — the two-node split is legitimate ONLY if nothing but
   composition happens in the wrapper it replaces.
2. The graph's ``cost_policy`` node must reproduce ``decide_one`` exactly (same p, same
   cost -> same decision), and its ``calibrate`` node must prior-shift via the SAME
   ``aiqs.eval.decision.prior_shift`` path ``aiqs-decide`` uses, not a re-derivation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiqs.eval.decision import CostMatrix, Decision, decide_one, ivap, prior_shift, venn_abers_merge
from aiqs.vlm.adjudicate import adjudicate
from aiqs.vlm.backend import MockVLMBackend
from aiqs.vlm.state import VLMState

from aiqs.api.artifact import DecisionArtifact
from aiqs.graph.nodes import cost_policy, make_calibrate_node, make_vlm_second_look_node, vlm_abstain_rule
from aiqs.graph.state import AdjudicationState

LOCKED = CostMatrix(false_accept=10.0, false_reject=3.0, escalation=1.0)


def _mk_graph_state(**overrides) -> AdjudicationState:
    base = dict(
        item_id="parity-1", detector_score=0.7, target_prevalence=0.02,
        cost_false_accept=LOCKED.false_accept, cost_false_reject=LOCKED.false_reject,
        cost_escalation=LOCKED.escalation, image_path="fake.png",
    )
    base.update(overrides)
    return AdjudicationState(**base)


# --------------------------------------------------------------------------- #
# 1. vlm_second_look + vlm_abstain_rule vs adjudicate()
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("verdict,conf,lam", [
    ("defect", 0.83, 0.0),
    ("clean", 0.6, 0.0),
    ("unsure", 0.5, 0.0),
    ("defect", 0.95, 0.4),
    ("clean", 0.2, 0.0),
    ("defect", 1.0, 1.0),  # full shrinkage -> always abstain
])
def test_vlm_node_composition_matches_adjudicate(verdict, conf, lam):
    backend = MockVLMBackend(verdict_fn=lambda state, rng: (verdict, conf, "why"))

    direct_state = VLMState(image_path="fake.png", detector_score=0.7, detector_p=0.3)
    direct_state = adjudicate(direct_state, backend, LOCKED, lam=lam)

    gstate = _mk_graph_state(detector_score=0.7, lam=lam)
    gstate = gstate.model_copy(update={"detector_p": 0.3})
    update1 = make_vlm_second_look_node(backend)(gstate)
    gstate = gstate.model_copy(update=update1)
    update2 = vlm_abstain_rule(gstate)
    gstate = gstate.model_copy(update=update2)

    assert gstate.vlm_verdict == direct_state.vlm_verdict
    assert gstate.vlm_confidence == direct_state.vlm_conf
    assert gstate.vlm_reasoning == direct_state.vlm_reasoning
    assert gstate.p_vlm == direct_state.p_vlm
    assert gstate.vlm_decision == direct_state.final_decision.value
    assert (gstate.vlm_decision == Decision.ESCALATE.value) == direct_state.abstained


# --------------------------------------------------------------------------- #
# 2. cost_policy node vs decide_one(); calibrate node vs prior_shift()
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("p", list(np.linspace(0.0, 1.0, 11)))
def test_cost_policy_node_matches_decide_one_exactly(p):
    gstate = _mk_graph_state(detector_score=0.0)
    gstate = gstate.model_copy(update={"detector_p": float(p)})
    out = cost_policy(gstate)
    assert out["tier1_decision"] == decide_one(float(p), LOCKED).value


def test_calibrate_node_uses_the_same_prior_shift_as_decide_py():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95, 1.0])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    artifact = DecisionArtifact(
        run_id="synthetic", run_dir=Path("."), category="synthetic",
        cal_scores=scores, cal_labels=labels, pi_source=0.5, auroc=0.9,
        n=8, n_good=4, n_defective=4, guard_warnings=[],
    )
    calibrate = make_calibrate_node(artifact)
    gstate = _mk_graph_state(detector_score=0.95, target_prevalence=0.02)
    out = calibrate(gstate)

    p0, p1 = ivap(scores, labels, np.array([0.95]))
    p_native = venn_abers_merge(p0, p1)[0]
    expected = prior_shift(np.array([p_native]), 0.5, 0.02)[0]
    assert out["detector_p"] == pytest.approx(expected)


def test_serve_time_p_differs_from_oof_p_by_design():
    """Document (as an executable check, not just a comment) that serve-time inductive
    Venn-Abers is a DIFFERENT valid estimator from the run's committed cross/OOF
    Venn-Abers in decision_scores.csv — feeding one of the run's own scores back through
    ``calibrate`` is expected to diverge somewhat, and must never be "fixed" to match."""
    from aiqs.eval.decision import cross_venn_abers

    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 40), rng.normal(2, 1, 40)])
    labels = np.array([0] * 40 + [1] * 40)
    p_oof, _, _ = cross_venn_abers(scores, labels, k=10, seed=42)

    artifact = DecisionArtifact(
        run_id="synthetic", run_dir=Path("."), category="synthetic",
        cal_scores=scores, cal_labels=labels, pi_source=0.5, auroc=0.9,
        n=80, n_good=40, n_defective=40, guard_warnings=[],
    )
    p_serve = np.array([artifact.calibrate(float(s)) for s in scores])

    # Both are valid Venn-Abers probabilities (bounded, monotone-ish in score) but are
    # NOT expected to match: full-set IVAP (serve) vs out-of-fold IVAP (OOF/decide.py).
    assert not np.allclose(p_oof, p_serve, atol=1e-9)
    # Policy-relevant: exact equality would be a red flag (leakage), not a bug to fix.
