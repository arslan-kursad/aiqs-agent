# Baseline run — `patchcore-wide_resnet50_2_visa-capsules`

_Generated: 20260705T181237Z_

## Run metadata

| key | value |
| --- | --- |
| run_id | patchcore-wide_resnet50_2_visa-capsules |
| seed | 42 |
| category | capsules |
| dataset | visa |
| model | patchcore |
| model_size | wide_resnet50_2 |
| image_size | 256x256 |
| max_steps | 70000 |
| accelerator | auto |

## Detection metrics

| metric | value |
| --- | --- |
| pixel_auroc | None |
| pixel_aupro | None |
| pixel_aupimo | None |
| image_auroc | 0.7387 |
| image_f1score | 0.7222 |

## Decision metrics (Phase 1)

_Calibrated, cost-aware, abstaining policy over per-image scores. Calibration = cross (out-of-fold) Venn-Abers — every item's P(defective) is predicted by a calibrator that never saw it. Cost matrices (relative, escape/overkill/review): illustrative **10/3/1**; realistic escape-dominant **100/3/1** (shipping a defect ≫ a re-inspection)._

Detector image-AUROC = **0.739**; sample = 160 items (60 good / 100 defective, native prevalence 62%).

**Operating-envelope result (NOT a pass/fail).** Whether cost-aware abstention beats a well-tuned threshold depends on the regime, and this run maps where the boundary sits:

- **Detector strength.** image-AUROC = 0.739 (WEAK — lots of borderline items for abstention to catch).
- **Review cost.** Abstention helps while review is cheap: here it beats the tuned threshold when review cost < 1.514 (native, escape/overkill = 10/3); at the locked review = 1 it does not.
- **Cost asymmetry.** At low prevalence the illustrative 10/3/1 matrix makes escapes too cheap → PASS-all is optimal and abstention is moot. Under a realistic escape-dominant matrix (100/3/1: shipping a defect ≫ a re-inspection) the trade-off returns: there OURS still does not beat naive at review = 1.

**Takeaway:** cost-aware abstention is the right tool for **cheap review**, **weak/uncertain detectors**, or **escape-dominant low-prevalence** economics; with a strong detector + expensive review + mild cost asymmetry, a tuned threshold already suffices. The layer reports which regime you are in rather than asserting a universal win.

### Native prevalence (62% defective) — matrix 10/3/1

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.319 | 0.681 | 1.000 | 0.000 | 0.7375 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.950 | 0.000 | 1.0688 |
| naive @ OUR escape rate (apples-to-apples) | 1.000 | 0.000 | 0.950 | 0.000 | 1.069 |
| ours @ ~10% escalation (marker, not the policy) | 0.875 | 0.125 | 1.000 | 0.000 | 1.137 |

_Verdict:_ OURS beats naive by 0.3313 cost/item (31% lower) — abstention pays off.

**Break-even review cost (anti-cherry-pick, full sweep in `breakeven.csv` / `risk_coverage_breakeven.png`):** cost-aware abstention beats the tuned threshold on TOTAL cost when **review cost < 1.514** (units where escape=10, overkill=3). As review→0 escalation is free and OURS wins outright; as review rises the overhead overtakes the error savings.

### Target prevalence (2% defective), ILLUSTRATIVE matrix 10/3/1 — label-shift corrected

_Probabilities prior-shifted and metrics importance-weighted to the target defect rate. Note: under 10/3/1 at this prevalence escapes are so cheap that PASS-all is cost-optimal (escape rate → 1.0). That is a **cost-matrix property, not a bug** — it motivates the realistic matrix below._

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.978 | 0.022 | 0.000 | 1.000 | 0.1695 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.000 | 0.650 | 0.1300 |
| naive @ OUR escape rate (apples-to-apples) | 0.941 | 0.059 | 0.000 | 1.000 | 0.161 |
| ours @ ~10% escalation (marker, not the policy) | 0.924 | 0.076 | 0.000 | 1.000 | 0.172 |

_Verdict:_ Abstention does not reduce errors here, so it only adds review cost — a tuned threshold (or FAIL/PASS-all) is already cost-optimal in this regime.

### Target prevalence (2% defective), REALISTIC matrix 100/3/1 — escape-dominant

_Same prior-shift/weights; escape now costs 100× an overkill, reflecting a real line where shipping a defect dwarfs a re-inspection. This restores the asymmetry that makes catching defects worthwhile at low prevalence._

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.266 | 0.734 | 0.000 | 1.000 | 1.1743 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.067 | 0.480 | 1.1560 |
| naive @ OUR escape rate (apples-to-apples) | 0.818 | 0.182 | 0.041 | 0.506 | 1.180 |
| ours @ ~10% escalation (marker, not the policy) | 0.908 | 0.092 | 0.000 | 1.000 | 1.052 |

_Verdict:_ STRONG detector / expensive review: abstention cuts overkill 0.067→0.000, but at review cost 1 the escalation overhead exceeds the error savings, so a tuned threshold wins on TOTAL cost (by 0.0183/item). Break-even: OURS wins once review cost < 0.968.

**Break-even review cost (anti-cherry-pick, full sweep in `breakeven.csv` / `risk_coverage_breakeven.png`):** cost-aware abstention beats the tuned threshold on TOTAL cost when **review cost < 0.968** (units where escape=100, overkill=3). As review→0 escalation is free and OURS wins outright; as review rises the overhead overtakes the error savings.

Secondary reference (single 50/50 split inductive Venn-Abers, held-out half, n=80, native prevalence): coverage 0.375, overkill 1.000, cost/item 0.7375.

Plots: `risk_coverage.png` (native), `risk_coverage_target.png` (target 10/3/1), `risk_coverage_target_realistic.png` (target 100/3/1), `risk_coverage_breakeven.png` (review-cost sweep); full sweeps `risk_coverage.csv` + `breakeven.csv`; per-item `decision_scores.csv`.

