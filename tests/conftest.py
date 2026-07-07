"""Shared fixtures for the Phase-3 graph/API test suites.

The synthetic artifact's calibration set is fixed (seed=0) and its three anchor scores
are chosen so each policy band is reachable under NATIVE prevalence (pi_source=0.5, no
prior-shift) with the LOCKED 10/3/1 cost matrix:

    score=-2.0 -> p~0.024  -> PASS      (< 0.10)
    score= 1.5 -> p =0.5   -> ESCALATE  (0.10 < p < 0.667)
    score= 6.0 -> p~0.976  -> FAIL      (>= 0.667)

(Verified directly against ``aiqs.eval.decision.ivap`` — see test_graph.py.)
"""

from __future__ import annotations

import numpy as np
import pytest

from aiqs.api.artifact import DecisionArtifact

PASS_SCORE = -2.0
ESCALATE_SCORE = 1.5
FAIL_SCORE = 6.0


def make_synthetic_artifact(run_dir) -> DecisionArtifact:
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 40), rng.normal(4, 1, 40)])
    labels = np.array([0] * 40 + [1] * 40)
    return DecisionArtifact(
        run_id="synthetic_run", run_dir=run_dir, category="synthetic",
        cal_scores=scores, cal_labels=labels, pi_source=0.5, auroc=0.99,
        n=80, n_good=40, n_defective=40, guard_warnings=[],
    )


@pytest.fixture
def synthetic_artifact(tmp_path) -> DecisionArtifact:
    run_dir = tmp_path / "synthetic_run"
    run_dir.mkdir()
    return make_synthetic_artifact(run_dir)
