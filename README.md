# AIQS-Agent

**Agentic Adjudication Layer for Industrial Visual Quality Inspection.**

An agentic decision + reasoning layer that sits **on top of** an off-the-shelf
anomaly detector and makes its outputs production-trustworthy. The value is in the
**decision layer** — cost-aware, calibrated, *abstaining* decisions with auditable
reasoning traces — not in the detector (a solved commodity). We optimize a business
cost function and the false-reject rate, **not** detection AUROC.

> Detailed north star, constraints, and architecture live in [CLAUDE.md](CLAUDE.md).

## Status

- **Phase 0 — detection baseline + eval backbone.** Off-the-shelf
  [Anomalib](https://github.com/openvinotoolkit/anomalib) EfficientAD on one MVTec
  AD category, with a reusable evaluation backbone. Write-up:
  [docs/PHASE0_REPORT.md](docs/PHASE0_REPORT.md).
- **Phase 1 — calibrated, cost-aware, abstaining decision layer.** Cross
  (out-of-fold) Venn-Abers calibration → cost-matrix `PASS/FAIL/ESCALATE` →
  risk-coverage, with prevalence (label-shift) correction and an honesty guard. **The
  layer + guard are complete and validated (23/23 tests).** On the current weak
  600-step detector (image-AUROC 0.559) the guard correctly **refuses a false-positive
  headline**: an honest null result (no separable signal to exploit). The positive
  risk-coverage headline is **pending a real-separation detector** (GPU baseline,
  below); the decision machinery is validated end-to-end on synthetic separating
  scores (`make sim`). No LLM/agent yet.

## Quickstart

```bash
make install                 # uv sync (creates .venv, installs pinned stack)
make smoke                   # fast end-to-end sanity run (~10 steps)
make baseline CATEGORY=screw # full train + eval, writes results/
make decide                  # Phase-1 adjudication on the latest run
make sim                     # SYNTHETIC machinery validation (NOT real-data evidence)
make test                    # unit tests
```

Results: `results/metrics.csv`, per-run `summary.md`, and Phase-1
`risk_coverage{,_target}.png` + `decision_scores.csv` + `results/decisions.csv`.

## Stack & host notes

Pinned for an **Intel (x86_64) macOS** host (CPU-only): `anomalib 1.2.0`,
`torch 2.2.2`. See [CLAUDE.md](CLAUDE.md) for why.

### GPU / arm64 upgrade path (turnkey real baseline)

The Phase-1 decision layer is solid; what it needs is a detector with real
per-image separation. On a CUDA or Apple-Silicon host:

1. Drop the version caps in `pyproject.toml` and move to **anomalib 2.x + current
   torch** (the caps exist only because this Intel-mac has no newer-torch wheel).
2. Optionally switch the dataset to **MVTec AD 2** (supported in anomalib 2.x).
3. Train at the **full step budget** (`configs/default.yaml`, 70k steps) — the 600-step
   CPU budget is the root cause of the weak image-AUROC, not the category.
4. `make baseline` → `make decide`. With a separating detector the risk-coverage
   headline (overkill ↓, cost ↓ vs naive at matched escape) materialises — exactly as
   `make sim` shows on synthetic scores.

If EfficientAD's **image-level** AUROC lags, swap in **PatchCore** (also off-the-shelf
in Anomalib) — it tends to be stronger at image-level separation, which is what this
decision-layer demo needs. The decision layer is detector-agnostic: it only consumes
`image_scores.csv`.

## Layout

```
configs/        YAML configs (category is configurable)
src/aiqs/
  config.py     typed config + CLI overrides
  data.py       MVTec AD datamodule
  train.py      train EfficientAD on one category
  evaluate.py   evaluate a checkpoint, persist results
  decide.py     Phase-1 cost-aware adjudication over persisted scores
  simulate_decision.py  synthetic machinery validation (walled off)
  eval/         evaluation backbone (metrics, persistence, decision policy/calibration)
results/        metrics.csv + per-run summaries + Phase-1 decisions (committed)
tests/          unit tests for the decision policy / calibration / guard
```
