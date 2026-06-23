# Baseline run — `patchcore-wide_resnet50_2_mvtec-capsule`

_Generated: 20260623T140858Z_

## Run metadata

| key | value |
| --- | --- |
| run_id | patchcore-wide_resnet50_2_mvtec-capsule |
| seed | 42 |
| category | capsule |
| dataset | mvtec |
| model | patchcore |
| model_size | wide_resnet50_2 |
| image_size | 256x256 |
| max_steps | -1 |
| accelerator | auto |

## Detection metrics

| metric | value |
| --- | --- |
| pixel_auroc | 0.9872 |
| pixel_aupro | 0.9193 |
| pixel_aupimo | 0.5102 |
| image_auroc | 0.9765 |
| image_f1score | 0.9767 |

## Decision metrics (Phase 1)

_Calibrated, cost-aware, abstaining policy over per-image scores. Calibration = cross (out-of-fold) Venn-Abers — every item's P(defective) is predicted by a calibrator that never saw it. Cost matrices (relative, escape/overkill/review): illustrative **10/3/1**; realistic escape-dominant **100/3/1** (shipping a defect ≫ a re-inspection)._

Detector image-AUROC = **0.976**; sample = 132 items (23 good / 109 defective, native prevalence 83%).

**Operating-envelope result (NOT a pass/fail).** Whether cost-aware abstention beats a well-tuned threshold depends on the regime, and this run maps where the boundary sits:

- **Detector strength.** image-AUROC = 0.976 (STRONG — the threshold already captures most of the value).
- **Review cost.** Abstention helps while review is cheap: here it beats the tuned threshold when review cost < 0.808 (native, escape/overkill = 10/3); at the locked review = 1 it does not.
- **Cost asymmetry.** At low prevalence the illustrative 10/3/1 matrix makes escapes too cheap → PASS-all is optimal and abstention is moot. Under a realistic escape-dominant matrix (100/3/1: shipping a defect ≫ a re-inspection) the trade-off returns: there OURS BEATS naive at review = 1.

**Takeaway:** cost-aware abstention is the right tool for **cheap review**, **weak/uncertain detectors**, or **escape-dominant low-prevalence** economics; with a strong detector + expensive review + mild cost asymmetry, a tuned threshold already suffices. The layer reports which regime you are in rather than asserting a universal win.

### Native prevalence (83% defective) — matrix 10/3/1

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.818 | 0.182 | 0.250 | 0.000 | 0.2045 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.217 | 0.009 | 0.1894 |
| naive @ OUR escape rate (apples-to-apples) | 0.902 | 0.098 | 0.154 | 0.000 | 0.144 |
| ours @ ~10% escalation (marker, not the policy) | 0.902 | 0.098 | 0.167 | 0.009 | 0.220 |

_Verdict:_ Abstention does not reduce errors here, so it only adds review cost — a tuned threshold (or FAIL/PASS-all) is already cost-optimal in this regime.

**Break-even review cost (anti-cherry-pick, full sweep in `breakeven.csv` / `risk_coverage_breakeven.png`):** cost-aware abstention beats the tuned threshold on TOTAL cost when **review cost < 0.808** (units where escape=10, overkill=3). As review→0 escalation is free and OURS wins outright; as review rises the overhead overtakes the error savings.

### Target prevalence (2% defective), ILLUSTRATIVE matrix 10/3/1 — label-shift corrected

_Probabilities prior-shifted and metrics importance-weighted to the target defect rate. Note: under 10/3/1 at this prevalence escapes are so cheap that PASS-all is cost-optimal (escape rate → 1.0). That is a **cost-matrix property, not a bug** — it motivates the realistic matrix below._

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.948 | 0.052 | 0.000 | 1.000 | 0.1551 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.000 | 0.385 | 0.0771 |
| naive @ OUR escape rate (apples-to-apples) | 0.957 | 0.043 | 0.000 | 0.390 | 0.119 |
| ours @ ~10% escalation (marker, not the policy) | 0.895 | 0.105 | 0.000 | 1.000 | 0.110 |

_Verdict:_ Abstention does not reduce errors here, so it only adds review cost — a tuned threshold (or FAIL/PASS-all) is already cost-optimal in this regime.

### Target prevalence (2% defective), REALISTIC matrix 100/3/1 — escape-dominant

_Same prior-shift/weights; escape now costs 100× an overkill, reflecting a real line where shipping a defect dwarfs a re-inspection. This restores the asymmetry that makes catching defects worthwhile at low prevalence._

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.938 | 0.062 | 0.000 | 1.000 | 0.1534 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.043 | 0.028 | 0.1829 |
| naive @ OUR escape rate (apples-to-apples) | 0.913 | 0.087 | 0.048 | 0.030 | 0.270 |
| ours @ ~10% escalation (marker, not the policy) | 0.896 | 0.104 | 0.000 | 1.000 | 0.178 |

_Verdict:_ OURS beats naive by 0.0294 cost/item (16% lower) — abstention pays off.

**Break-even review cost (anti-cherry-pick, full sweep in `breakeven.csv` / `risk_coverage_breakeven.png`):** cost-aware abstention beats the tuned threshold on TOTAL cost when **review cost < 1.223** (units where escape=100, overkill=3). As review→0 escalation is free and OURS wins outright; as review rises the overhead overtakes the error savings.

Secondary reference (single 50/50 split inductive Venn-Abers, held-out half, n=66, native prevalence): coverage 0.848, overkill 1.000, cost/item 0.2879.

Plots: `risk_coverage.png` (native), `risk_coverage_target.png` (target 10/3/1), `risk_coverage_target_realistic.png` (target 100/3/1), `risk_coverage_breakeven.png` (review-cost sweep); full sweeps `risk_coverage.csv` + `breakeven.csv`; per-item `decision_scores.csv`.

**Small-n caveat.** Only n_normal = 23 good parts → wide error bars; treat rates as indicative. CPU/Colab portfolio demo, not a production validation.

