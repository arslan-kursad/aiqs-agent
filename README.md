# AIQS-Agent

**Agentic Adjudication Layer for Industrial Visual Quality Inspection.**

An agentic decision + reasoning layer that sits **on top of** an off-the-shelf
anomaly detector and makes its outputs production-trustworthy. The value is in the
**decision layer** — cost-aware, calibrated, *abstaining* decisions with auditable
reasoning traces — not in the detector (a solved commodity). We optimize a business
cost function and the false-reject rate, **not** detection AUROC.

> Detailed north star, constraints, and architecture live in [CLAUDE.md](CLAUDE.md).

## Status

**Phase 0 — foundation + detection baseline.** Off-the-shelf
[Anomalib](https://github.com/openvinotoolkit/anomalib) EfficientAD on one MVTec AD
category, with a reusable evaluation backbone. No LLM/agent yet.

## Quickstart

```bash
make install                 # uv sync (creates .venv, installs pinned stack)
make smoke                   # fast end-to-end sanity run (~10 steps)
make baseline CATEGORY=screw # full train + eval, writes results/
```

Results are written to `results/` (`metrics.csv` + a per-run `summary.md`).

## Stack & host notes

Pinned for an **Intel (x86_64) macOS** host (CPU-only): `anomalib 1.2.0`,
`torch 2.2.2`. See [CLAUDE.md](CLAUDE.md) for why, and the upgrade path to
anomalib 2.x on an arm64/CUDA host.

## Layout

```
configs/        YAML configs (category is configurable)
src/aiqs/
  config.py     typed config + CLI overrides
  data.py       MVTec AD datamodule
  train.py      train EfficientAD on one category
  evaluate.py   evaluate a checkpoint, persist results
  eval/         evaluation backbone (metrics, persistence, Phase-1 decision stub)
results/        metrics.csv + per-run summaries (committed)
```
