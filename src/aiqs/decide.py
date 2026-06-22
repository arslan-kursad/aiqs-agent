"""Phase-1 adjudication: turn persisted anomaly scores into cost-optimal actions.

    uv run aiqs-decide                          # latest run, target prevalence 2%
    uv run aiqs-decide --run <run_id>
    uv run aiqs-decide --target-prevalence 0.05

Consumes ``results/runs/<id>/image_scores.csv`` (Phase-0 output) and, WITHOUT
re-running the detector, calibrates + decides. Two regimes are reported on every run:

  * NATIVE — the benchmark test split as-is (~74% defective). Honest, but NOT the
    economics the thesis targets: with a weak detector and overkill cheap relative to
    the dominant escape risk, "FAIL everything" is near-optimal and abstention only
    spends review budget. We keep this (it is the honest substrate result).
  * TARGET — importance-weighted + prior-shifted to a realistic LOW production defect
    rate (default 2%). This is the regime the "reduce overkill" thesis is about. The
    benchmark's 74%-defective split is an artefact of detection benchmarking, not a
    production line; the label-shift correction (see eval.decision.prior_shift /
    prevalence_weights) maps it to production economics, consistently for calibration
    AND evaluation.

The headline is reported honestly: on a weak (~0.56 AUROC) detector the decision layer
cannot manufacture separation that is not there, so OURS ~= NAIVE in BOTH regimes. A
positive risk-coverage headline awaits a real-separation detector (see the GPU upgrade
path in README/CLAUDE.md); the machinery itself is validated on synthetic separating
scores by ``aiqs-sim-decision`` (walled off under results/synthetic_validation/).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aiqs.eval.decision import (  # noqa: E402
    CostMatrix,
    DecisionMetrics,
    check_not_degenerate,
    cost_optimal_threshold,
    cross_venn_abers,
    decide,
    decision_metrics,
    empirical_auroc,
    escalation_cost_ceiling,
    ivap,
    naive_decide,
    prevalence_weights,
    prior_shift,
    risk_coverage_naive,
    risk_coverage_ours,
    venn_abers_merge,
)

LOCKED_COST = CostMatrix(false_accept=10.0, false_reject=3.0, escalation=1.0)
DECISIONS_CSV = "decisions.csv"


# --------------------------------------------------------------------------- #
# Analysis (pure compute — shared by the real run and the synthetic validation)
# --------------------------------------------------------------------------- #

@dataclass
class Regime:
    """One prevalence regime's policy comparison + risk-coverage sweeps."""
    name: str
    prevalence: float
    m_ours: DecisionMetrics
    m_naive: DecisionMetrics
    thr: float
    ours_rows: list[dict]
    naive_rows: list[dict]
    marker10: dict | None
    matched: dict | None     # naive margin point at OUR escape rate


@dataclass
class Analysis:
    run_name: str
    n: int
    n_neg: int
    n_pos: int
    pi_source: float
    auroc: float
    warnings: list[str]
    target_prevalence: float
    p_cross: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    p_target: np.ndarray
    native: Regime
    target: Regime
    m_ours_split: DecisionMetrics


def _nearest_row(rows: list[dict], key: str, target: float) -> dict | None:
    valid = [r for r in rows if not np.isnan(r[key])]
    if not valid:
        return None
    return min(valid, key=lambda r: abs(r[key] - target))


def _row_from_metrics(m: DecisionMetrics) -> dict:
    return {"coverage": m.coverage, "escalation_rate": m.escalation_rate,
            "false_reject_rate": m.false_reject_rate,
            "false_accept_rate": m.false_accept_rate,
            "cost_per_item": m.cost_per_item}


def _build_regime(name, prevalence, scores, labels, probs, weights, grid) -> Regime:
    m_ours = decision_metrics(labels, decide(probs, LOCKED_COST), LOCKED_COST,
                              weights=weights, target_prevalence=prevalence)
    thr = cost_optimal_threshold(scores, labels, LOCKED_COST, weights=weights)
    m_naive = decision_metrics(labels, naive_decide(scores, thr, 0.0), LOCKED_COST,
                               weights=weights, target_prevalence=prevalence)

    ceil = escalation_cost_ceiling(LOCKED_COST)
    c_grid = np.linspace(0.0, ceil * 1.06, grid)
    max_dev = float(np.max(np.abs(scores - thr)))
    m_grid = np.linspace(0.0, max_dev * 1.01, grid)
    ours_rows = risk_coverage_ours(probs, labels, LOCKED_COST, c_grid, weights=weights)
    naive_rows = risk_coverage_naive(scores, labels, thr, LOCKED_COST, m_grid,
                                     weights=weights)

    marker10 = _nearest_row(ours_rows, "escalation_rate", 0.10)
    matched = _nearest_row(naive_rows, "false_accept_rate", m_ours.false_accept_rate)
    return Regime(name, prevalence, m_ours, m_naive, thr, ours_rows, naive_rows,
                  marker10, matched)


def analyze(scores: np.ndarray, labels: np.ndarray, *, run_name: str, seed: int,
            folds: int, target_prevalence: float, grid: int) -> Analysis:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = labels.shape[0]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    pi_source = n_pos / n
    auroc = empirical_auroc(scores, labels)
    warnings = check_not_degenerate(scores, labels, k=folds)

    # PRIMARY calibration: cross / out-of-fold Venn-Abers over all items.
    p_cross, p0, p1 = cross_venn_abers(scores, labels, k=folds, seed=seed)
    # Prior-shift to the target prevalence + matching importance weights.
    p_target = prior_shift(p_cross, pi_source, target_prevalence)
    weights = prevalence_weights(labels, target_prevalence, pi_source)

    native = _build_regime("native", pi_source, scores, labels, p_cross, None, grid)
    target = _build_regime(f"target_{target_prevalence:g}", target_prevalence,
                           scores, labels, p_target, weights, grid)

    # SECONDARY reference: single 50/50 split inductive Venn-Abers (native regime).
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    cal_idx, eval_idx = next(sss.split(scores, labels))
    p_eval = venn_abers_merge(*ivap(scores[cal_idx], labels[cal_idx], scores[eval_idx]))
    m_ours_split = decision_metrics(labels[eval_idx], decide(p_eval, LOCKED_COST),
                                    LOCKED_COST)

    return Analysis(run_name, n, n_neg, n_pos, pi_source, auroc, warnings,
                    target_prevalence, p_cross, p0, p1, p_target, native, target,
                    m_ours_split)


# --------------------------------------------------------------------------- #
# Rendering / persistence
# --------------------------------------------------------------------------- #

def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def _make_plot(regime: Regime, out_path: Path, title: str) -> None:
    def curve(rows, key):
        rows = sorted(rows, key=lambda r: r["coverage"])
        return [r["coverage"] for r in rows], [r[key] for r in rows]

    naive_point = _row_from_metrics(regime.m_naive)
    ours_locked = _row_from_metrics(regime.m_ours)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    for ax, key, ylabel, sub in (
        (ax1, "false_reject_rate", "overkill rate (auto-decided goods)", "overkill"),
        (ax2, "cost_per_item", "decision cost / item", "decision cost"),
    ):
        xo, yo = curve(regime.ours_rows, key)
        xn, yn = curve(regime.naive_rows, key)
        ax.plot(xo, yo, "-", color="#1f77b4", lw=2,
                label="ours: calibrated cost-aware abstention")
        ax.plot(xn, yn, "--", color="#888888", lw=1.6,
                label="naive: cost-optimal threshold + margin")
        ax.scatter([naive_point["coverage"]], [naive_point[key]], marker="s", s=70,
                   color="#d62728", zorder=5,
                   label="naive fixed threshold (no abstention)")
        ax.scatter([ours_locked["coverage"]], [ours_locked[key]], marker="*", s=240,
                   color="#1f77b4", edgecolor="black", zorder=6,
                   label="LOCKED policy (review cost = 1)")
        if regime.marker10 is not None:
            ax.scatter([regime.marker10["coverage"]], [regime.marker10[key]],
                       marker="o", s=60, facecolor="none", edgecolor="#1f77b4",
                       zorder=6, label="ours @ ~10% escalation (marker)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Risk-coverage — {sub}")
        ax.grid(True, alpha=0.3)
    ax2.set_xlabel("coverage  (= 1 - escalation rate)")
    ax1.legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _regime_table(reg: Regime) -> list[str]:
    L = ["| policy | coverage | escalation | overkill (FRR) | escape (FAR) | cost/item |",
         "| --- | --- | --- | --- | --- | --- |",
         f"| **OURS — LOCKED (review=1)** | {_fmt(reg.m_ours.coverage)} | "
         f"{_fmt(reg.m_ours.escalation_rate)} | {_fmt(reg.m_ours.false_reject_rate)} | "
         f"{_fmt(reg.m_ours.false_accept_rate)} | {reg.m_ours.cost_per_item:.4f} |",
         f"| naive fixed threshold (no abstention) | 1.000 | 0.000 | "
         f"{_fmt(reg.m_naive.false_reject_rate)} | {_fmt(reg.m_naive.false_accept_rate)} "
         f"| {reg.m_naive.cost_per_item:.4f} |"]
    if reg.matched:
        L.append(f"| naive @ OUR escape rate (apples-to-apples) | "
                 f"{_fmt(reg.matched['coverage'])} | "
                 f"{_fmt(1 - reg.matched['coverage'])} | "
                 f"{_fmt(reg.matched['false_reject_rate'])} | "
                 f"{_fmt(reg.matched['false_accept_rate'])} | "
                 f"{_fmt(reg.matched['cost_per_item'])} |")
    if reg.marker10:
        L.append(f"| ours @ ~10% escalation (marker, not the policy) | "
                 f"{_fmt(reg.marker10['coverage'])} | "
                 f"{_fmt(reg.marker10['escalation_rate'])} | "
                 f"{_fmt(reg.marker10['false_reject_rate'])} | "
                 f"{_fmt(reg.marker10['false_accept_rate'])} | "
                 f"{_fmt(reg.marker10['cost_per_item'])} |")
    return L


def _append_decisions_csv(results_dir: Path, row: dict) -> Path:
    csv_path = results_dir / DECISIONS_CSV
    df_row = pd.DataFrame([row])
    if csv_path.exists():
        df_row = pd.concat([pd.read_csv(csv_path), df_row], ignore_index=True)
    df_row.to_csv(csv_path, index=False)
    return csv_path


def _verdict(reg: Regime) -> str:
    d = reg.m_naive.cost_per_item - reg.m_ours.cost_per_item
    if d > 1e-4:
        return (f"OURS beats naive by {d:.4f} cost/item "
                f"({d / reg.m_naive.cost_per_item:.0%} lower).")
    if d < -1e-4:
        return (f"OURS costs {-d:.4f} MORE/item than naive — abstention adds cost here "
                "(no separation to exploit / FAIL-or-PASS-all already optimal).")
    return "OURS ~= naive (no separable signal for the decision layer to exploit)."


def write_outputs(analysis: Analysis, out_dir: Path, *, synthetic: bool,
                  results_dir: Path | None = None,
                  scores: np.ndarray | None = None,
                  labels: np.ndarray | None = None,
                  image_paths: list[str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    a = analysis

    # risk-coverage CSV (both regimes, tagged)
    rows = []
    for reg in (a.native, a.target):
        for r in reg.ours_rows + reg.naive_rows:
            rows.append({"regime": reg.name, "prevalence": reg.prevalence, **r})
    pd.DataFrame(rows).to_csv(out_dir / "risk_coverage.csv", index=False)

    # plots: native (the honest null) + target (production economics)
    tag = "SYNTHETIC machinery validation" if synthetic else a.run_name
    _make_plot(a.native, out_dir / "risk_coverage.png",
               f"{tag} — NATIVE prevalence {a.pi_source:.0%} defective")
    _make_plot(a.target, out_dir / "risk_coverage_target.png",
               f"{tag} — TARGET prevalence {a.target_prevalence:.0%} defective")

    # per-item probabilities + decisions
    ds = pd.DataFrame({"label": labels if labels is not None else [],
                       "score": scores if scores is not None else []})
    if image_paths is not None:
        ds.insert(0, "image_path", image_paths)
    ds["p_cross"] = a.p_cross
    ds["p0_cross"] = a.p0
    ds["p1_cross"] = a.p1
    ds["p_target"] = a.p_target
    ds["decision_native"] = [d.value for d in decide(a.p_cross, LOCKED_COST)]
    ds["decision_target"] = [d.value for d in decide(a.p_target, LOCKED_COST)]
    ds.to_csv(out_dir / "decision_scores.csv", index=False)

    _write_summary(analysis, out_dir, synthetic=synthetic)

    if not synthetic and results_dir is not None:
        row = {
            "timestamp": pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ"),
            "run_id": a.run_name, "calibration": "cross_venn_abers",
            "n": a.n, "n_normal": a.n_neg, "n_defective": a.n_pos,
            "image_auroc": round(a.auroc, 4), "pi_source": round(a.pi_source, 4),
            "target_prevalence": a.target_prevalence,
            # native regime
            "native_escalation_rate": round(a.native.m_ours.escalation_rate, 4),
            "native_overkill": round(a.native.m_ours.false_reject_rate, 4),
            "native_cost_per_item": round(a.native.m_ours.cost_per_item, 4),
            "native_naive_cost_per_item": round(a.native.m_naive.cost_per_item, 4),
            # target regime
            "target_escalation_rate": round(a.target.m_ours.escalation_rate, 4),
            "target_overkill": round(a.target.m_ours.false_reject_rate, 4),
            "target_cost_per_item": round(a.target.m_ours.cost_per_item, 4),
            "target_naive_cost_per_item": round(a.target.m_naive.cost_per_item, 4),
        }
        _append_decisions_csv(results_dir, row)


def _write_summary(analysis: Analysis, out_dir: Path, *, synthetic: bool) -> None:
    a = analysis
    L: list[str] = []
    if synthetic:
        L += ["# ⚠️ SYNTHETIC machinery validation — NOT real-data evidence", "",
              "These numbers come from **simulated** anomaly scores with a *known*, "
              "controllable separation. Their ONLY purpose is to prove the decision "
              "code is correct on a detector that actually separates classes. They say "
              "**nothing** about real performance and must never be quoted as a result. "
              "Real-data results live in `results/runs/<id>/summary.md`.", ""]
    L += ["## Decision metrics (Phase 1)", "",
          "_Calibrated, cost-aware, abstaining policy over per-image scores. Cost "
          "matrix (locked, relative): false-accept/escape = 10, false-reject/overkill "
          "= 3, escalate/review = 1. Calibration = cross (out-of-fold) Venn-Abers — "
          "every item's P(defective) is predicted by a calibrator that never saw it._",
          "",
          f"Detector image-AUROC = **{a.auroc:.3f}**; sample = {a.n} items "
          f"({a.n_neg} good / {a.n_pos} defective, native prevalence "
          f"{a.pi_source:.0%}).", ""]

    L += [f"### Native prevalence ({a.pi_source:.0%} defective) — honest substrate", ""]
    L += _regime_table(a.native)
    L += ["", f"_Verdict:_ {_verdict(a.native)}", ""]

    L += [f"### Target prevalence ({a.target_prevalence:.0%} defective) — production "
          "economics (label-shift corrected)", "",
          "_Probabilities prior-shifted and metrics importance-weighted to the target "
          "defect rate (the benchmark's defect-heavy split is not a production line). "
          "Same correction applied to calibration and evaluation._", ""]
    L += _regime_table(a.target)
    L += ["", f"_Verdict:_ {_verdict(a.target)}", ""]

    L += [f"Secondary reference (single 50/50 split inductive Venn-Abers, held-out "
          f"half, n={a.m_ours_split.n}, native prevalence): coverage "
          f"{_fmt(a.m_ours_split.coverage)}, overkill "
          f"{_fmt(a.m_ours_split.false_reject_rate)}, cost/item "
          f"{a.m_ours_split.cost_per_item:.4f}.", "",
          "Plots: `risk_coverage.png` (native) + `risk_coverage_target.png` (target); "
          "full sweep `risk_coverage.csv`; per-item probabilities + decisions "
          "`decision_scores.csv`.", ""]
    if a.warnings:
        L.append("**Guard warnings:**")
        L += [f"- {w}" for w in a.warnings]
        L.append("")
    if not synthetic:
        L += [
            "**Root cause of the null result (read before quoting any number).** Two "
            f"compounding facts: (1) the detector is intentionally weak (image-AUROC "
            f"{a.auroc:.3f}) — at the reduced 600-step CPU budget the per-image signal "
            "is essentially noise, so the calibrated P(defective) collapses to ~the "
            "base rate for BOTH classes and no policy (calibrated or not) can separate "
            "parts; (2) the benchmark split is "
            f"{a.pi_source:.0%}-defective — inverted from a production line — so even a "
            "perfect-economics policy reduces to FAIL-all (native) or PASS-all "
            "(target). The decision layer correctly reports it can extract no value "
            "here; the honesty guard flagged exactly this. **This is a CPU-constrained "
            f"portfolio demo with only n_normal = {a.n_neg} good parts (wide error "
            "bars), NOT a production validation.** A positive risk-coverage headline "
            "needs a real-separation detector — see the GPU upgrade path in "
            "README/CLAUDE.md. The decision machinery itself is validated on synthetic "
            "separating scores (`aiqs-sim-decision`).", ""]
    _append_or_replace_summary(out_dir, L, replace=synthetic)


def _append_or_replace_summary(out_dir: Path, lines: list[str], *, replace: bool):
    path = out_dir / "summary.md"
    body = "\n".join(lines) + "\n"
    if replace or not path.exists():
        path.write_text(body)
        return
    existing = "\n".join(
        ln for ln in path.read_text().splitlines() if "Phase-1 placeholder" not in ln
    ).rstrip()
    # Drop any previously-appended decision section so re-runs stay idempotent.
    marker = "## Decision metrics (Phase 1)"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip()
    path.write_text(existing + "\n\n" + body)


# --------------------------------------------------------------------------- #
# Real-run entry point
# --------------------------------------------------------------------------- #

def _find_run_dir(results_dir: Path, run: str | None) -> Path:
    runs_root = results_dir / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(f"No runs under {runs_root}. Run an eval first.")
    candidates = sorted(p for p in runs_root.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories under {runs_root}.")
    if run is None:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    if (runs_root / run).is_dir():
        return runs_root / run
    matches = [p for p in candidates if run in p.name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No run matching '{run}'. Available:\n  "
            + "\n  ".join(p.name for p in candidates))
    raise ValueError(f"'{run}' matches multiple runs:\n  "
                     + "\n  ".join(p.name for p in matches))


def _load_scores(run_dir: Path):
    path = run_dir / "image_scores.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Phase-0 eval must persist per-image scores first.")
    df = pd.read_csv(path)
    score_col = next(c for c in ("score", "anomaly_score") if c in df.columns)
    label_col = next(c for c in ("label", "gt_label") if c in df.columns)
    paths = df["image_path"].tolist() if "image_path" in df.columns else None
    return (df[score_col].to_numpy(dtype=float),
            df[label_col].to_numpy(dtype=int), paths)


def _print_console(a: Analysis) -> None:
    print("=" * 66)
    print(f"  AIQS-Agent — Phase 1 adjudication: {a.run_name}")
    print("=" * 66)
    print(f"  items={a.n}  good={a.n_neg}  defective={a.n_pos}  "
          f"base-rate(def)={a.pi_source:.2%}  image-AUROC={a.auroc:.3f}")
    for w in a.warnings:
        print(f"  [warn] {w}")
    for reg, head in ((a.native, f"NATIVE ({a.pi_source:.0%} defective)"),
                      (a.target, f"TARGET ({a.target_prevalence:.0%} defective)")):
        print("  " + "-" * 62)
        print(f"  {head}:")
        print(f"    OURS  esc={_fmt(reg.m_ours.escalation_rate)} "
              f"cov={_fmt(reg.m_ours.coverage)} "
              f"overkill={_fmt(reg.m_ours.false_reject_rate)} "
              f"escape={_fmt(reg.m_ours.false_accept_rate)} "
              f"cost/item={reg.m_ours.cost_per_item:.4f}")
        print(f"    NAIVE cost/item={reg.m_naive.cost_per_item:.4f} "
              f"overkill={_fmt(reg.m_naive.false_reject_rate)} (thr={reg.thr:.3f})")
        print(f"    => {_verdict(reg)}")
    print("=" * 66 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 cost-aware adjudication.")
    parser.add_argument("--run", help="results/runs/<run_id> (default: latest).")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--target-prevalence", type=float, default=0.02,
                        help="production defect rate to reweight to (default 0.02).")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid", type=int, default=61)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_dir = _find_run_dir(results_dir, args.run)
    scores, labels, paths = _load_scores(run_dir)

    analysis = analyze(scores, labels, run_name=run_dir.name, seed=args.seed,
                       folds=args.folds, target_prevalence=args.target_prevalence,
                       grid=args.grid)
    write_outputs(analysis, run_dir, synthetic=False, results_dir=results_dir,
                  scores=scores, labels=labels, image_paths=paths)
    _print_console(analysis)
    print(f"  artifacts -> {run_dir}/risk_coverage.png(+_target), risk_coverage.csv, "
          "decision_scores.csv; appended results/decisions.csv\n")


if __name__ == "__main__":
    main()
