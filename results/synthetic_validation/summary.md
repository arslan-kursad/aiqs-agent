# ⚠️ SYNTHETIC machinery validation — NOT real-data evidence

These numbers come from **simulated** anomaly scores with a *known*, controllable separation. Their ONLY purpose is to prove the decision code is correct on a detector that actually separates classes. They say **nothing** about real performance and must never be quoted as a result. Real-data results live in `results/runs/<id>/summary.md`.

## Decision metrics (Phase 1)

_Calibrated, cost-aware, abstaining policy over per-image scores. Calibration = cross (out-of-fold) Venn-Abers — every item's P(defective) is predicted by a calibrator that never saw it. Cost matrices (relative, escape/overkill/review): illustrative **10/3/1**; realistic escape-dominant **100/3/1** (shipping a defect ≫ a re-inspection)._

Detector image-AUROC = **0.926**; sample = 3000 items (2246 good / 754 defective, native prevalence 25%).

**Operating-envelope result (NOT a pass/fail).** Whether cost-aware abstention beats a well-tuned threshold depends on the regime, and this run maps where the boundary sits:

- **Detector strength.** image-AUROC = 0.926 (STRONG — the threshold already captures most of the value).
- **Review cost.** Abstention helps while review is cheap: here it beats the tuned threshold when review cost < 1.858 (native, escape/overkill = 10/3); at the locked review = 1 it does not.
- **Cost asymmetry.** At low prevalence the illustrative 10/3/1 matrix makes escapes too cheap → PASS-all is optimal and abstention is moot. Under a realistic escape-dominant matrix (100/3/1: shipping a defect ≫ a re-inspection) the trade-off returns: there OURS BEATS naive at review = 1.

**Takeaway:** cost-aware abstention is the right tool for **cheap review**, **weak/uncertain detectors**, or **escape-dominant low-prevalence** economics; with a strong detector + expensive review + mild cost asymmetry, a tuned threshold already suffices. The layer reports which regime you are in rather than asserting a universal win.

### Native prevalence (25% defective) — matrix 10/3/1

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.670 | 0.330 | 0.042 | 0.107 | 0.5460 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.177 | 0.118 | 0.6947 |
| naive @ OUR escape rate (apples-to-apples) | 0.964 | 0.036 | 0.171 | 0.110 | 0.674 |
| ours @ ~10% escalation (marker, not the policy) | 0.902 | 0.098 | 0.112 | 0.127 | 0.613 |

_Verdict:_ OURS beats naive by 0.1487 cost/item (21% lower) — abstention pays off.

**Break-even review cost (anti-cherry-pick, full sweep in `breakeven.csv` / `risk_coverage_breakeven.png`):** cost-aware abstention beats the tuned threshold on TOTAL cost when **review cost < 1.858** (units where escape=10, overkill=3). As review→0 escalation is free and OURS wins outright; as review rises the overhead overtakes the error savings.

### Target prevalence (2% defective), ILLUSTRATIVE matrix 10/3/1 — label-shift corrected

_Probabilities prior-shifted and metrics importance-weighted to the target defect rate. Note: under 10/3/1 at this prevalence escapes are so cheap that PASS-all is cost-optimal (escape rate → 1.0). That is a **cost-matrix property, not a bug** — it motivates the realistic matrix below._

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.956 | 0.044 | 0.000 | 0.860 | 0.1319 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.011 | 0.581 | 0.1489 |
| naive @ OUR escape rate (apples-to-apples) | 0.421 | 0.579 | 0.000 | 0.769 | 0.581 |
| ours @ ~10% escalation (marker, not the policy) | 0.897 | 0.103 | 0.000 | 0.990 | 0.154 |

_Verdict:_ OURS beats naive by 0.0170 cost/item (11% lower) — abstention pays off.

### Target prevalence (2% defective), REALISTIC matrix 100/3/1 — escape-dominant

_Same prior-shift/weights; escape now costs 100× an overkill, reflecting a real line where shipping a defect dwarfs a re-inspection. This restores the asymmetry that makes catching defects worthwhile at low prevalence._

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.753 | 0.247 | 0.000 | 0.568 | 0.4354 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.124 | 0.171 | 0.7061 |
| naive @ OUR escape rate (apples-to-apples) | 1.000 | 0.000 | 0.124 | 0.171 | 0.706 |
| ours @ ~10% escalation (marker, not the policy) | 0.899 | 0.101 | 0.034 | 0.251 | 0.533 |

_Verdict:_ OURS beats naive by 0.2707 cost/item (38% lower) — abstention pays off.

**Break-even review cost (anti-cherry-pick, full sweep in `breakeven.csv` / `risk_coverage_breakeven.png`):** cost-aware abstention beats the tuned threshold on TOTAL cost when **review cost < 2.727** (units where escape=100, overkill=3). As review→0 escalation is free and OURS wins outright; as review rises the overhead overtakes the error savings.

Secondary reference (single 50/50 split inductive Venn-Abers, held-out half, n=1500, native prevalence): coverage 0.674, overkill 0.013, cost/item 0.5280.

Plots: `risk_coverage.png` (native), `risk_coverage_target.png` (target 10/3/1), `risk_coverage_target_realistic.png` (target 100/3/1), `risk_coverage_breakeven.png` (review-cost sweep); full sweeps `risk_coverage.csv` + `breakeven.csv`; per-item `decision_scores.csv`.

