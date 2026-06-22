"""LABELED machinery validation for the Phase-1 decision layer — SYNTHETIC scores.

    uv run aiqs-sim-decision

⚠️  This is NOT real-data evidence and must NEVER be quoted as a result. Its only
purpose is to prove the decision CODE is correct on a detector that actually
separates classes — something the current weak (0.559-AUROC) real detector does not.
It generates synthetic anomaly scores with a *known* separation (default image-AUROC
~0.92) at a moderate prevalence, runs the FULL real pipeline (cross Venn-Abers
calibration -> prior-shift to a low production prevalence -> cost-matrix policy ->
risk-coverage), and shows the expected outcome: with real separation, OUR calibrated
cost-aware abstention BEATS the strong naive baseline.

Outputs are walled off under results/synthetic_validation/ with a loud SYNTHETIC
banner in summary.md and are never written to results/decisions.csv (the real-run
ledger). Same code path as `aiqs-decide`; only the input scores differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aiqs.decide import analyze, write_outputs, _print_console, _verdict
from aiqs.eval.decision import simulate_scores


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic machinery validation (NOT real).")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--auroc", type=float, default=0.92,
                   help="separation of the synthetic detector (image-AUROC).")
    p.add_argument("--prevalence", type=float, default=0.25,
                   help="defect rate used to GENERATE the synthetic sample.")
    p.add_argument("--target-prevalence", type=float, default=0.02,
                   help="production defect rate to reweight to.")
    p.add_argument("--out", default="results/synthetic_validation")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", type=int, default=61)
    args = p.parse_args()

    scores, labels = simulate_scores(args.n, args.auroc, args.prevalence, args.seed)

    print("\n" + "!" * 66)
    print("  SYNTHETIC MACHINERY VALIDATION — NOT REAL-DATA EVIDENCE")
    print("  (proves the decision code is correct on a separating detector)")
    print("!" * 66)

    analysis = analyze(scores, labels, run_name="synthetic_validation",
                       seed=args.seed, folds=args.folds,
                       target_prevalence=args.target_prevalence, grid=args.grid)
    out_dir = Path(args.out)
    write_outputs(analysis, out_dir, synthetic=True, scores=scores, labels=labels)
    _print_console(analysis)

    # Machinery acceptance: with real separation, ours should beat naive at the
    # production target prevalence.
    tgt = analysis.target
    gain = tgt.m_naive.cost_per_item - tgt.m_ours.cost_per_item
    print(f"  MACHINERY CHECK (target {args.target_prevalence:.0%}): {_verdict(tgt)}")
    print(f"  outputs -> {out_dir}/ (summary.md is loudly marked SYNTHETIC)\n")
    if gain <= 0:
        raise SystemExit(
            f"Machinery check FAILED: ours did not beat naive (gain={gain:.4f}). "
            "With a 0.92-AUROC detector the layer should extract value — investigate.")


if __name__ == "__main__":
    main()
