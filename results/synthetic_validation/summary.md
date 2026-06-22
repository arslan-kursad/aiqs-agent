# ⚠️ SYNTHETIC machinery validation — NOT real-data evidence

These numbers come from **simulated** anomaly scores with a *known*, controllable separation. Their ONLY purpose is to prove the decision code is correct on a detector that actually separates classes. They say **nothing** about real performance and must never be quoted as a result. Real-data results live in `results/runs/<id>/summary.md`.

## Decision metrics (Phase 1)

_Calibrated, cost-aware, abstaining policy over per-image scores. Cost matrix (locked, relative): false-accept/escape = 10, false-reject/overkill = 3, escalate/review = 1. Calibration = cross (out-of-fold) Venn-Abers — every item's P(defective) is predicted by a calibrator that never saw it._

Detector image-AUROC = **0.926**; sample = 3000 items (2246 good / 754 defective, native prevalence 25%).

### Native prevalence (25% defective) — honest substrate

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.670 | 0.330 | 0.042 | 0.107 | 0.5460 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.177 | 0.118 | 0.6947 |
| naive @ OUR escape rate (apples-to-apples) | 0.964 | 0.036 | 0.171 | 0.110 | 0.674 |
| ours @ ~10% escalation (marker, not the policy) | 0.902 | 0.098 | 0.112 | 0.127 | 0.613 |

_Verdict:_ OURS beats naive by 0.1487 cost/item (21% lower).

### Target prevalence (2% defective) — production economics (label-shift corrected)

_Probabilities prior-shifted and metrics importance-weighted to the target defect rate (the benchmark's defect-heavy split is not a production line). Same correction applied to calibration and evaluation._

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.956 | 0.044 | 0.000 | 0.860 | 0.1319 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.011 | 0.581 | 0.1489 |
| naive @ OUR escape rate (apples-to-apples) | 0.421 | 0.579 | 0.000 | 0.769 | 0.581 |
| ours @ ~10% escalation (marker, not the policy) | 0.897 | 0.103 | 0.000 | 0.990 | 0.154 |

_Verdict:_ OURS beats naive by 0.0170 cost/item (11% lower).

Secondary reference (single 50/50 split inductive Venn-Abers, held-out half, n=1500, native prevalence): coverage 0.674, overkill 0.013, cost/item 0.5280.

Plots: `risk_coverage.png` (native) + `risk_coverage_target.png` (target); full sweep `risk_coverage.csv`; per-item probabilities + decisions `decision_scores.csv`.

