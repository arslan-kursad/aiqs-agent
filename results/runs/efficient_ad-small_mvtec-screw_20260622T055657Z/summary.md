# Baseline run — `efficient_ad-small_mvtec-screw`

_Generated: 20260622T055657Z_

## Run metadata

| key | value |
| --- | --- |
| run_id | efficient_ad-small_mvtec-screw |
| seed | 42 |
| category | screw |
| dataset | mvtec |
| model | efficient_ad |
| model_size | small |
| image_size | 256x256 |
| max_steps | 600 |
| accelerator | auto |

## Detection metrics

| metric | value |
| --- | --- |
| pixel_auroc | 0.9396 |
| pixel_aupro | 0.8211 |
| pixel_aupimo | 0.0023 |
| image_auroc | 0.5586 |
| image_f1score | 0.6854 |

## Decision metrics (Phase 1)

_Calibrated, cost-aware, abstaining policy over per-image scores. Calibration = cross (out-of-fold) Venn-Abers — every item's P(defective) is predicted by a calibrator that never saw it. Cost matrices (relative, escape/overkill/review): illustrative **10/3/1**; realistic escape-dominant **100/3/1** (shipping a defect ≫ a re-inspection)._

Detector image-AUROC = **0.559**; sample = 160 items (41 good / 119 defective, native prevalence 74%).

**Operating-envelope result (NOT a pass/fail).** Whether cost-aware abstention beats a well-tuned threshold depends on the regime, and this run maps where the boundary sits:

- **Detector strength.** image-AUROC = 0.559 (WEAK — lots of borderline items for abstention to catch).
- **Review cost.** Abstention helps while review is cheap: here it beats the tuned threshold when review cost < 0.888 (native, escape/overkill = 10/3); at the locked review = 1 it does not.
- **Cost asymmetry.** At low prevalence the illustrative 10/3/1 matrix makes escapes too cheap → PASS-all is optimal and abstention is moot. Under a realistic escape-dominant matrix (100/3/1: shipping a defect ≫ a re-inspection) the trade-off returns: there OURS BEATS naive at review = 1.

**Takeaway:** cost-aware abstention is the right tool for **cheap review**, **weak/uncertain detectors**, or **escape-dominant low-prevalence** economics; with a strong detector + expensive review + mild cost asymmetry, a tuned threshold already suffices. The layer reports which regime you are in rather than asserting a universal win.

### Native prevalence (74% defective) — matrix 10/3/1

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.744 | 0.256 | 1.000 | 0.000 | 0.8938 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 1.000 | 0.000 | 0.7688 |
| naive @ OUR escape rate (apples-to-apples) | 1.000 | 0.000 | 1.000 | 0.000 | 0.769 |
| ours @ ~10% escalation (marker, not the policy) | 0.887 | 0.113 | 1.000 | 0.000 | 0.825 |

_Verdict:_ Abstention does not reduce errors here, so it only adds review cost — a tuned threshold (or FAIL/PASS-all) is already cost-optimal in this regime.

**Break-even review cost (anti-cherry-pick, full sweep in `breakeven.csv` / `risk_coverage_breakeven.png`):** cost-aware abstention beats the tuned threshold on TOTAL cost when **review cost < 0.888** (units where escape=10, overkill=3). As review→0 escalation is free and OURS wins outright; as review rises the overhead overtakes the error savings.

### Target prevalence (2% defective), ILLUSTRATIVE matrix 10/3/1 — label-shift corrected

_Probabilities prior-shifted and metrics importance-weighted to the target defect rate. Note: under 10/3/1 at this prevalence escapes are so cheap that PASS-all is cost-optimal (escape rate → 1.0). That is a **cost-matrix property, not a bug** — it motivates the realistic matrix below._

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 1.000 | 0.000 | 0.000 | 1.000 | 0.2000 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.000 | 1.000 | 0.2000 |
| naive @ OUR escape rate (apples-to-apples) | 1.000 | 0.000 | 0.000 | 1.000 | 0.200 |
| ours @ ~10% escalation (marker, not the policy) | 0.879 | 0.121 | 0.000 | 1.000 | 0.309 |

_Verdict:_ Abstention does not reduce errors here, so it only adds review cost — a tuned threshold (or FAIL/PASS-all) is already cost-optimal in this regime.

### Target prevalence (2% defective), REALISTIC matrix 100/3/1 — escape-dominant

_Same prior-shift/weights; escape now costs 100× an overkill, reflecting a real line where shipping a defect dwarfs a re-inspection. This restores the asymmetry that makes catching defects worthwhile at low prevalence._

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.049 | 0.951 | 0.000 | 1.000 | 1.0520 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.268 | 0.563 | 1.9148 |
| naive @ OUR escape rate (apples-to-apples) | 0.656 | 0.344 | 0.333 | 0.613 | 1.628 |
| ours @ ~10% escalation (marker, not the policy) | 1.000 | 0.000 | 0.268 | 0.807 | 2.402 |

_Verdict:_ OURS beats naive by 0.8628 cost/item (45% lower) — abstention pays off.

**Break-even review cost (anti-cherry-pick, full sweep in `breakeven.csv` / `risk_coverage_breakeven.png`):** cost-aware abstention beats the tuned threshold on TOTAL cost when **review cost < 2.243** (units where escape=100, overkill=3). As review→0 escalation is free and OURS wins outright; as review rises the overhead overtakes the error savings.

Secondary reference (single 50/50 split inductive Venn-Abers, held-out half, n=80, native prevalence): coverage 0.900, overkill 1.000, cost/item 0.8500.

Plots: `risk_coverage.png` (native), `risk_coverage_target.png` (target 10/3/1), `risk_coverage_target_realistic.png` (target 100/3/1), `risk_coverage_breakeven.png` (review-cost sweep); full sweeps `risk_coverage.csv` + `breakeven.csv`; per-item `decision_scores.csv`.

**Guard warnings:**
- WEAK detector: image-AUROC=0.559 (< 0.60). This is the intended Phase-1 'untrustworthy detector' regime; numbers are real but the calibration carries wide error bars — interpret accordingly.

**Small-n caveat.** Only n_normal = 41 good parts → wide error bars; treat rates as indicative. CPU/Colab portfolio demo, not a production validation.

