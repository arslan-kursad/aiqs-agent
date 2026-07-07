"""The serve-time "decision artifact": everything ``aiqs-serve`` reads from a
completed run directory. Torch-free by construction — it reuses ``aiqs.decide``'s own
run-resolution and score-loading (``_find_run_dir`` / ``_load_scores``) instead of
re-implementing them, so serving can never drift from the CLI's provenance rules.

Calibration = LIVE inductive Venn-Abers (``aiqs.eval.decision.ivap``), not a persisted
"fitted calibrator" object. This is not a workaround: a Venn-Abers prediction for a test
point is DEFINED as re-fitting isotonic regression with that point appended under each
hypothetical label (see ``ivap``'s docstring) — there is no fitted object to cache short
of literally storing the whole calibration set, which is exactly what we do (the run's
own image_scores.csv). A persisted "calibrator" would be an approximation of this, not
the real thing.

IMPORTANT — this serve-time p will NOT bit-match the run's own ``decision_scores.csv``:
that file holds CROSS/out-of-fold Venn-Abers probabilities (``aiqs.decide.analyze``),
where every item's p comes from a calibrator that excluded that item's own fold. Serving
a brand-new, never-labelled score instead calibrates against the FULL labelled set (plain
inductive Venn-Abers) — the correct thing to do for a genuinely new item, since there is
no fold to exclude it from. Both are valid Venn-Abers estimators; they are simply not the
same estimator, and feeding one of the run's own scores back into ``calibrate`` will give
a slightly different (and, for that item, leaked/optimistic) p than its OOF row. Do not
"fix" this discrepancy — it is expected and documented here and in CLAUDE.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aiqs.decide import LOCKED_COST, REALISTIC_COST, _find_run_dir, _load_scores
from aiqs.eval.decision import (
    CostMatrix,
    check_not_degenerate,
    empirical_auroc,
    ivap,
    prior_shift,
    venn_abers_merge,
)

DEFAULT_TARGET_PREVALENCE = 0.02

# Run directories are named f"{run_id}_{stamp}" (see eval/results.py), where run_id ends
# "..._{dataset}-{category}" (see config.Config.run_id) and stamp is a UTC
# "YYYYmmddTHHMMSSZ" timestamp. config.yaml is NOT a reliable category source: evaluate.py
# persists the raw --config FILE TEXT verbatim, which can predate a CLI --category
# override (observed on the capsules headline run: config.yaml says "candle"). The
# directory name is the one field guaranteed to reflect what was actually run.
_STAMP_RE = re.compile(r"_\d{8}T\d{6}Z$")


def _category_from_run_dir(run_dir: Path) -> str:
    stem = _STAMP_RE.sub("", run_dir.name)
    return stem.rsplit("-", 1)[-1]


@dataclass
class DecisionArtifact:
    """Everything ``aiqs-serve`` needs from one completed run directory."""

    run_id: str
    run_dir: Path
    category: str
    cal_scores: np.ndarray
    cal_labels: np.ndarray
    pi_source: float          # native (sample) defect prevalence
    auroc: float
    n: int
    n_good: int
    n_defective: int
    guard_warnings: list[str]
    default_target_prevalence: float = DEFAULT_TARGET_PREVALENCE
    locked_cost: CostMatrix = field(default_factory=lambda: LOCKED_COST)
    realistic_cost: CostMatrix = field(default_factory=lambda: REALISTIC_COST)

    def calibrate(self, raw_score: float, target_prevalence: float | None = None) -> float:
        """Live inductive Venn-Abers P(defective) for one new raw anomaly score.

        ``target_prevalence`` (optional) prior-shifts the result via the SAME
        ``aiqs.eval.decision.prior_shift`` (Saerens/Elkan) path ``aiqs-decide`` uses —
        one policy-math implementation, imported here, never re-derived.
        """
        p0, p1 = ivap(self.cal_scores, self.cal_labels,
                      np.array([raw_score], dtype=float))
        p = float(venn_abers_merge(p0, p1)[0])
        if (target_prevalence is not None
                and abs(target_prevalence - self.pi_source) > 1e-12):
            p = float(prior_shift(np.array([p]), self.pi_source, target_prevalence)[0])
        return p


def load_artifact(results_dir: Path, run: str | None = None) -> DecisionArtifact:
    """Load and validate one run directory as a serving artifact.

    Re-runs ``check_not_degenerate`` at load time (same guard ``aiqs-decide`` applies):
    a broken/signal-free run must not silently start serving nonsense decisions.
    """
    run_dir = _find_run_dir(results_dir, run)
    scores, labels, _paths = _load_scores(run_dir)
    guard_warnings = check_not_degenerate(scores, labels)
    n = int(labels.shape[0])
    n_pos = int((labels == 1).sum())
    n_neg = n - n_pos
    return DecisionArtifact(
        run_id=run_dir.name,
        run_dir=run_dir,
        category=_category_from_run_dir(run_dir),
        cal_scores=scores,
        cal_labels=labels,
        pi_source=n_pos / n,
        auroc=empirical_auroc(scores, labels),
        n=n, n_good=n_neg, n_defective=n_pos,
        guard_warnings=guard_warnings,
    )
