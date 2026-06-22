# Baseline run — `patchcore-wide_resnet50_2_mvtec-screw`

_Generated: 20260622T130823Z_

## Run metadata

| key | value |
| --- | --- |
| run_id | patchcore-wide_resnet50_2_mvtec-screw |
| seed | 42 |
| category | screw |
| dataset | mvtec |
| model | patchcore |
| model_size | wide_resnet50_2 |
| image_size | 256x256 |
| max_steps | -1 |
| accelerator | auto |

## Detection metrics

| metric | value |
| --- | --- |
| pixel_auroc | 0.9888 |
| pixel_aupro | 0.9402 |
| pixel_aupimo | 0.4553 |
| image_auroc | 0.9758 |
| image_f1score | 0.9547 |

## Decision metrics (Phase 1)

_Calibrated, cost-aware, abstaining policy over per-image scores. Cost matrix (locked, relative): false-accept/escape = 10, false-reject/overkill = 3, escalate/review = 1. Calibration = cross (out-of-fold) Venn-Abers — every item's P(defective) is predicted by a calibrator that never saw it._

Detector image-AUROC = **0.976**; sample = 160 items (41 good / 119 defective, native prevalence 74%).

### Native prevalence (74% defective) — honest substrate

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.838 | 0.163 | 0.160 | 0.000 | 0.2375 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.293 | 0.000 | 0.2250 |
| naive @ OUR escape rate (apples-to-apples) | 1.000 | 0.000 | 0.293 | 0.000 | 0.225 |
| ours @ ~10% escalation (marker, not the policy) | 0.894 | 0.106 | 0.200 | 0.000 | 0.219 |

_Verdict:_ OURS costs 0.0125 MORE/item than naive — abstention adds cost here (no separation to exploit / FAIL-or-PASS-all already optimal).

### Target prevalence (2% defective) — production economics (label-shift corrected)

_Probabilities prior-shifted and metrics importance-weighted to the target defect rate (the benchmark's defect-heavy split is not a production line). Same correction applied to calibration and evaluation._

| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |
| --- | --- | --- | --- | --- | --- |
| **OURS — LOCKED (review=1)** | 0.965 | 0.035 | 0.000 | 1.000 | 0.1226 |
| naive fixed threshold (no abstention) | 1.000 | 0.000 | 0.000 | 0.454 | 0.0908 |
| naive @ OUR escape rate (apples-to-apples) | 0.975 | 0.025 | 0.000 | 0.473 | 0.114 |
| ours @ ~10% escalation (marker, not the policy) | 0.910 | 0.090 | 0.000 | 1.000 | 0.110 |

_Verdict:_ OURS costs 0.0318 MORE/item than naive — abstention adds cost here (no separation to exploit / FAIL-or-PASS-all already optimal).

Secondary reference (single 50/50 split inductive Venn-Abers, held-out half, n=80, native prevalence): coverage 0.850, overkill 0.154, cost/item 0.2250.

Plots: `risk_coverage.png` (native) + `risk_coverage_target.png` (target); full sweep `risk_coverage.csv`; per-item probabilities + decisions `decision_scores.csv`.

**Root cause of the null result (read before quoting any number).** Two compounding facts: (1) the detector is intentionally weak (image-AUROC 0.976) — at the reduced 600-step CPU budget the per-image signal is essentially noise, so the calibrated P(defective) collapses to ~the base rate for BOTH classes and no policy (calibrated or not) can separate parts; (2) the benchmark split is 74%-defective — inverted from a production line — so even a perfect-economics policy reduces to FAIL-all (native) or PASS-all (target). The decision layer correctly reports it can extract no value here; the honesty guard flagged exactly this. **This is a CPU-constrained portfolio demo with only n_normal = 41 good parts (wide error bars), NOT a production validation.** A positive risk-coverage headline needs a real-separation detector — see the GPU upgrade path in README/CLAUDE.md. The decision machinery itself is validated on synthetic separating scores (`aiqs-sim-decision`).

