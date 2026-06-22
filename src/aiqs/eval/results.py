"""Persistence backbone for evaluation runs.

This is the spine everything later is measured against. It is intentionally
decoupled from anomalib / torch (plain pandas + stdlib) so that:

  * Phase 0 writes detection metrics (AUROC / AUPRO / AUPIMO) here, and
  * Phase 1 extends the SAME run directory with decision-level metrics
    (false-reject rate, escalation rate, decision cost) computed from the
    per-image scores we persist now — without re-running the detector.

Layout written under results/:
    results/metrics.csv                      # one appended row per eval run
    results/runs/<run_dir>/summary.md        # human-readable per-run summary
    results/runs/<run_dir>/image_scores.csv  # per-image scores (Phase-1 input)
    results/runs/<run_dir>/config.yaml       # exact config used (provenance)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

METRICS_CSV = "metrics.csv"


@dataclass
class EvalResult:
    """Everything one evaluation run produces."""

    run_id: str
    meta: dict = field(default_factory=dict)        # run-metadata columns
    metrics: dict = field(default_factory=dict)     # metric_name -> float value
    # Per-image, aligned arrays/lists. Optional but strongly recommended: this is
    # what the Phase-1 adjudication layer consumes.
    image_paths: list[str] | None = None
    image_labels: list[int] | None = None           # 0 = normal, 1 = anomalous
    image_scores: list[float] | None = None         # raw anomaly score
    extras: dict = field(default_factory=dict)       # notes, fallbacks, env, ...

    def timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_dir(results_dir: Path, result: EvalResult, stamp: str) -> Path:
    d = results_dir / "runs" / f"{result.run_id}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_metrics_csv(results_dir: Path, row: dict) -> Path:
    csv_path = results_dir / METRICS_CSV
    df_row = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, df_row], ignore_index=True)
    else:
        combined = df_row
    combined.to_csv(csv_path, index=False)
    return csv_path


def _write_image_scores(run_dir: Path, result: EvalResult) -> Path | None:
    if result.image_scores is None:
        return None
    data = {"score": result.image_scores}
    if result.image_labels is not None:
        data = {"label": result.image_labels, **data}
    if result.image_paths is not None:
        data = {"image_path": result.image_paths, **data}
    path = run_dir / "image_scores.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _write_summary_md(run_dir: Path, result: EvalResult, stamp: str) -> Path:
    lines = [f"# Baseline run — `{result.run_id}`", "", f"_Generated: {stamp}_", ""]

    lines.append("## Run metadata")
    lines.append("")
    lines.append("| key | value |")
    lines.append("| --- | --- |")
    for k, v in result.meta.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Detection metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k, v in result.metrics.items():
        vs = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        lines.append(f"| {k} | {vs} |")
    lines.append("")

    if result.extras:
        lines.append("## Notes")
        lines.append("")
        for k, v in result.extras.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("> Phase-1 placeholder: decision-level metrics (false-reject "
                 "rate, escalation rate, decision cost) will be appended here, "
                 "computed from `image_scores.csv`.")
    lines.append("")

    path = run_dir / "summary.md"
    path.write_text("\n".join(lines))
    return path


def persist(result: EvalResult, results_dir: str | Path,
            config_yaml: str | None = None) -> Path:
    """Write metrics.csv row + per-run summary/scores. Returns the run dir."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.timestamp()
    run_dir = _run_dir(results_dir, result, stamp)

    # Flat CSV row: timestamp + metadata + metrics (metrics prefixed to avoid
    # collisions with metadata keys).
    row = {"timestamp": stamp, **result.meta}
    for k, v in result.metrics.items():
        row[f"metric.{k}"] = v
    _append_metrics_csv(results_dir, row)

    _write_image_scores(run_dir, result)
    _write_summary_md(run_dir, result, stamp)
    if config_yaml is not None:
        (run_dir / "config.yaml").write_text(config_yaml)

    return run_dir


def print_baseline_summary(result: EvalResult, run_dir: Path | None = None) -> None:
    """Clear end-of-run console summary."""
    bar = "=" * 64
    print(f"\n{bar}\n  AIQS-Agent — Phase 0 baseline: {result.run_id}\n{bar}")
    for k, v in result.meta.items():
        print(f"  {k:<14} {v}")
    print("  " + "-" * 60)
    for k, v in result.metrics.items():
        vs = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        print(f"  {k:<24} {vs}")
    if result.extras:
        print("  " + "-" * 60)
        for k, v in result.extras.items():
            print(f"  note: {k} = {v}")
    if run_dir is not None:
        print("  " + "-" * 60)
        print(f"  results -> {run_dir}")
    print(bar + "\n")
